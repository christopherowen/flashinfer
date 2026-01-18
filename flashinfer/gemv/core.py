"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

FP4 Block-Scaled GEMV for MoE Decode Optimization
=================================================

This module provides GEMV kernels optimized for small batch sizes (M<=16) where
the grouped GEMM kernel's 128x128 tiles result in poor compute efficiency.

For M=1 decode:
- Grouped GEMM (128x128 tiles): ~0.78% compute efficiency
- GEMV: ~100% compute efficiency (processes exactly what's needed)

The crossover point depends on the model dimensions, but typically:
- M <= 8: GEMV is more efficient
- M >= 32: Grouped GEMM amortizes tile overhead better
- M = 9-31: Need benchmarking to determine

Usage
-----
This module is used internally by the MoE dispatch logic to select between
GEMV and grouped GEMM based on the batch size.

For direct use:

    from flashinfer.gemv import gemv_fp4_blockscaled, should_use_gemv_for_moe
    
    if should_use_gemv_for_moe(num_tokens, hidden_dim, num_experts):
        # Use GEMV path
        output = gemv_fp4_blockscaled(...)
    else:
        # Use grouped GEMM path
        output = cutlass_fused_moe(...)

MXFP4 GEMV Kernel Variants
--------------------------

For dense layers (QKV projection, O projection, LM head), we provide several
kernel variants optimized for different use cases:

1. ``gemv_mxfp4_dp4a(input, weight, scale)``
   - All-in-one: quantizes BF16 activations inside the kernel
   - Best for: Single GEMV call, simplest API
   - Overhead: Quantization repeated for each call

2. ``quantize_activations_q8(input)`` + ``gemv_mxfp4_dp4a_prequant(q8, weight, scale, K)``
   - Two-step: separate quantization and GEMV
   - Best for: Multiple GEMV with same activations (e.g., Q, K, V separately)
   - Benefit: Quantization done once, reused across calls

3. ``gemv_mxfp4_dp4a_fused_qkv(q8, w_q, s_q, w_k, s_k, w_v, s_v, K)``
   - Fused: Computes Q, K, V in single kernel launch
   - Best for: Attention QKV projection
   - Benefit: 1.5-1.6x faster than 3 separate calls
   - Why: Eliminates 2 kernel launches, better L2 cache reuse

Kernel Selection Guide (gpt-oss-120b dimensions: K=2880)
--------------------------------------------------------

+------------------+--------+--------+------------------+--------------------+
| Layer            | K      | N      | Recommended      | Notes              |
+==================+========+========+==================+====================+
| QKV projection   | 2880   | varies | fused_qkv        | 1.64x speedup      |
+------------------+--------+--------+------------------+--------------------+
| O projection     | 2880   | 2880   | prequant         | Reuse Q8 from attn |
+------------------+--------+--------+------------------+--------------------+
| LM Head          | 2880   | 201088 | prequant         | Memory-bound       |
+------------------+--------+--------+------------------+--------------------+

Performance Notes
-----------------

- For K=2880 (gpt-oss-120b), kernels are launch-overhead bound, not memory-bound
- Fused QKV is critical: individual K/V projections take ~6μs (mostly launch overhead)
- LM Head (N=201088) is memory-bound and benefits from vectorized loads

Dynamic Kernel Configuration
----------------------------

Kernel launch parameters (NWARPS, ROWS_PER_BLOCK) are automatically selected
based on problem dimensions to maximize throughput:

**NWARPS Selection** (based on K):
- K <= 1536: NWARPS=1 (1.5 iters/thread)
- K <= 3072: NWARPS=2 (1.5 iters/thread) ← gpt-oss-120b
- K <= 6144: NWARPS=2 (3 iters/thread)
- K > 6144:  NWARPS=4 (larger models)

**ROWS_PER_BLOCK Selection** (based on N):
- N <= 256:   ROWS=2  (maximize parallelism)
- N <= 512:   ROWS=4  (K/V projection)
- N <= 4096:  ROWS=8  (Q/O projection)
- N <= 65536: ROWS=8  (good balance)
- N > 65536:  ROWS=16 (LM Head)

Example for gpt-oss-120b (K=2880):
- K/V projection (N=360):   NWARPS=2, ROWS=4 → ~8μs
- Q/O projection (N=2880):  NWARPS=2, ROWS=8 → ~17μs
- LM Head (N=201088):       NWARPS=2, ROWS=16 → ~1.4ms
"""

import torch
from typing import Optional, List, Tuple

# Threshold for switching between GEMV and grouped GEMM
# Determined empirically based on SM121 gpt-oss-120b benchmarks
GEMV_M_THRESHOLD = 8

# Minimum hidden dimension for GEMV efficiency
# Below this, the kernel launch overhead dominates
GEMV_MIN_K = 1024


def should_use_gemv_for_moe(
    num_tokens: int,
    hidden_dim: int,
    num_active_experts: int = 1,
    force_gemv: bool = False,
) -> bool:
    """
    Determine whether to use GEMV instead of grouped GEMM for MoE.
    
    Parameters
    ----------
    num_tokens : int
        Number of tokens being processed (M dimension).
        For decode with batch_size=1 and top_k=8, this is typically 8.
    hidden_dim : int
        Hidden dimension of the model (K dimension).
    num_active_experts : int
        Number of experts that have tokens assigned.
    force_gemv : bool
        If True, always use GEMV regardless of heuristics.
    
    Returns
    -------
    bool
        True if GEMV should be used, False for grouped GEMM.
    
    Notes
    -----
    The heuristics are based on the following observations from SM121:
    
    1. Grouped GEMM uses 128x128 tiles, so M=1 wastes 127/128 = 99.2% of compute
    2. GEMV has no such waste but has higher per-call overhead
    3. For gpt-oss-120b with 60 MoE layers, kernel launch overhead is significant
    
    Current thresholds:
    - M <= 8: GEMV is more efficient (includes top_k=8 decode)
    - K >= 1024: GEMV amortizes launch overhead
    """
    if force_gemv:
        return True
    
    # Don't use GEMV for small K - launch overhead dominates
    if hidden_dim < GEMV_MIN_K:
        return False
    
    # Use GEMV for small M (decode-like workloads)
    if num_tokens <= GEMV_M_THRESHOLD:
        return True
    
    return False


def gemv_fp4_blockscaled(
    activations: torch.Tensor,
    weights: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute FP4 block-scaled GEMV: D = alpha * A * B + beta * C
    
    This is designed for single-row or few-row matrix-vector multiplication
    where the grouped GEMM 128x128 tiles are inefficient.
    
    Parameters
    ----------
    activations : torch.Tensor
        Activation tensor of shape [M, K] in FP4 packed format (uint8).
        M should be small (typically 1-8 for decode).
    weights : torch.Tensor
        Weight tensor of shape [K, N] in FP4 packed format (uint8).
        For GEMV, N is typically 1, but this handles N > 1 as batched GEMV.
    activation_scales : torch.Tensor
        Scale factors for activations, shape depends on block size.
    weight_scales : torch.Tensor
        Scale factors for weights, shape depends on block size.
    output : Optional[torch.Tensor]
        Output tensor of shape [M, N]. If None, a new tensor is allocated.
    alpha : float
        Scaling factor for the product.
    beta : float
        Scaling factor for the bias (if provided).
    bias : Optional[torch.Tensor]
        Bias tensor of shape [M, N]. Only used if beta != 0.
    
    Returns
    -------
    torch.Tensor
        Output tensor of shape [M, N] in bfloat16.
    
    Raises
    ------
    RuntimeError
        If the kernel fails or inputs are invalid.
    
    Notes
    -----
    The FP4 format is MXFP4 with:
    - 4-bit weights packed as uint8 (2 weights per byte)
    - Scale factors in FP8 (float8_e4m3fn)
    - Block size of 32 elements per scale factor
    
    For MoE, this is called once per expert with the tokens routed to that expert.
    """
    # Input validation
    assert activations.dtype == torch.uint8, "Activations must be packed FP4 (uint8)"
    assert weights.dtype == torch.uint8, "Weights must be packed FP4 (uint8)"
    
    m = activations.shape[0]
    k_packed = activations.shape[1]  # Packed dimension (K/2)
    k = k_packed * 2  # Actual K dimension
    
    n = weights.shape[1] if weights.dim() > 1 else 1
    
    # Allocate output if not provided
    if output is None:
        output = torch.empty(m, n, dtype=torch.bfloat16, device=activations.device)
    
    # Get the JIT-compiled module
    from ..jit import get_gemv_module
    module = get_gemv_module()
    
    # Call the kernel
    if bias is not None:
        module.gemv_fp4_blockscaled(
            m, k, alpha, beta,
            activations, weights, bias, output,
            activation_scales, weight_scales
        )
    else:
        module.gemv_fp4_blockscaled(
            m, k, alpha, 0.0,
            activations, weights, output, output,  # C = D when no bias
            activation_scales, weight_scales
        )
    
    return output


def batched_gemv_fp4(
    num_experts: int,
    activations: torch.Tensor,
    weights: torch.Tensor,
    activation_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Compute batched FP4 GEMV for multiple experts.
    
    This handles the MoE case where we need to compute GEMV for each active
    expert. The batching is done in a single kernel launch for efficiency.
    
    Parameters
    ----------
    num_experts : int
        Number of experts to process.
    activations : torch.Tensor
        Activation tensor of shape [num_experts, M, K] in FP4 packed format.
    weights : torch.Tensor
        Weight tensor of shape [num_experts, K, N] in FP4 packed format.
    activation_scales : torch.Tensor
        Scale factors for activations.
    weight_scales : torch.Tensor
        Scale factors for weights.
    output : Optional[torch.Tensor]
        Output tensor of shape [num_experts, M, N]. If None, allocated.
    alpha : float
        Scaling factor.
    
    Returns
    -------
    torch.Tensor
        Output tensor of shape [num_experts, M, N] in bfloat16.
    """
    m = activations.shape[1]
    k_packed = activations.shape[2]
    k = k_packed * 2
    n = weights.shape[2] if weights.dim() > 2 else 1
    
    if output is None:
        output = torch.empty(num_experts, m, n, dtype=torch.bfloat16, device=activations.device)
    
    from ..jit import get_gemv_module
    module = get_gemv_module()
    
    module.batched_gemv_fp4(
        num_experts, m, k, alpha,
        activations, weights, output,
        activation_scales, weight_scales
    )
    
    return output


def gemv_moe_fc1(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_scales: torch.Tensor,
    token_expert_indices: torch.Tensor,
    token_expert_weights: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MoE FC1 (gate + up projection) using GEMV.
    
    This is the first GEMM in the MoE block: hidden_states -> gate_up_proj.
    For gpt-oss-120b: [M, 4096] x [4096, 11776] -> [M, 11776]
    
    The function handles:
    1. Quantizing activations to FP4 with block scaling
    2. Routing tokens to experts
    3. Computing GEMV for each expert
    4. Combining results with expert weights
    
    Parameters
    ----------
    hidden_states : torch.Tensor
        Input hidden states [M, hidden_dim] in bfloat16.
    expert_weights : torch.Tensor
        Expert FC1 weights [num_experts, inter_dim, hidden_dim] in FP4 packed.
    expert_scales : torch.Tensor
        Expert weight scales.
    token_expert_indices : torch.Tensor
        Which experts each token is routed to [M, top_k].
    token_expert_weights : torch.Tensor
        Routing weights for each token-expert pair [M, top_k].
    output : Optional[torch.Tensor]
        Output tensor [M, inter_dim]. If None, allocated.
    
    Returns
    -------
    torch.Tensor
        FC1 output [M, inter_dim] in bfloat16.
    
    Notes
    -----
    This function is designed for small M (decode). For large M (prefill),
    use cutlass_fused_moe which has better throughput.
    """
    # TODO: Implement full MoE FC1 with routing
    # For now, this is a placeholder that shows the intended interface
    raise NotImplementedError(
        "gemv_moe_fc1 is not yet implemented. "
        "Use cutlass_fused_moe for now and set GEMV_M_THRESHOLD=0."
    )


def gemv_moe_fc2(
    intermediate: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_scales: torch.Tensor,
    token_expert_indices: torch.Tensor,
    token_expert_weights: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MoE FC2 (down projection) using GEMV.
    
    This is the second GEMM in the MoE block: intermediate -> down_proj.
    For gpt-oss-120b: [M, 5888] x [5888, 4096] -> [M, 4096]
    
    Parameters
    ----------
    intermediate : torch.Tensor
        Intermediate tensor [M, inter_dim] in bfloat16 (after SwiGLU).
    expert_weights : torch.Tensor
        Expert FC2 weights [num_experts, hidden_dim, inter_dim] in FP4 packed.
    expert_scales : torch.Tensor
        Expert weight scales.
    token_expert_indices : torch.Tensor
        Which experts each token is routed to [M, top_k].
    token_expert_weights : torch.Tensor
        Routing weights for each token-expert pair [M, top_k].
    output : Optional[torch.Tensor]
        Output tensor [M, hidden_dim]. If None, allocated.
    
    Returns
    -------
    torch.Tensor
        FC2 output [M, hidden_dim] in bfloat16.
    """
    # TODO: Implement full MoE FC2 with routing
    raise NotImplementedError(
        "gemv_moe_fc2 is not yet implemented. "
        "Use cutlass_fused_moe for now and set GEMV_M_THRESHOLD=0."
    )


# Cached module for DP4A GEMV
_gemv_dp4a_module = None


def _get_gemv_dp4a_module():
    """Get the compiled DP4A GEMV module, compiling it if necessary."""
    global _gemv_dp4a_module
    
    if _gemv_dp4a_module is None:
        from ..jit.gemv import gen_gemv_fp4_sm120_module
        spec = gen_gemv_fp4_sm120_module()
        _gemv_dp4a_module = spec.build_and_load()
    
    return _gemv_dp4a_module


def gemv_mxfp4_dp4a(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MXFP4 GEMV using DP4A instructions with fused activation quantization.
    
    This is optimized for decode (small M) where tensor core tiles are wasteful.
    The kernel fuses BF16->INT8 activation quantization, avoiding a separate
    quantization kernel call.
    
    Parameters
    ----------
    input : torch.Tensor
        Input activations [M, K] in bfloat16.
        M should be small (typically 1-8 for decode).
    weight : torch.Tensor
        Weight tensor [N, K/2] in uint8 (packed MXFP4, 2 values per byte).
    weight_scale : torch.Tensor
        Scale factors [N, K/32] in uint8 (E8M0 format).
    output : Optional[torch.Tensor]
        Output tensor [M, N] in bfloat16. If None, allocated.
    
    Returns
    -------
    torch.Tensor
        Output tensor [M, N] in bfloat16.
    
    Notes
    -----
    This kernel:
    1. Takes BF16 activations (no pre-quantization needed)
    2. Quantizes activations to INT8 inside the kernel (fused)
    3. Uses DP4A instructions for efficient INT8 dot products
    4. Applies MXFP4 weight scales
    5. Outputs BF16
    
    Compared to Marlin:
    - Marlin: FP4 weights -> dequant to BF16 -> BF16 GEMM
    - This: BF16 acts -> INT8 (fused) -> DP4A with FP4 weights -> BF16
    
    For M=1-8, this is faster because:
    - No tensor core tile waste (128x128 tiles)
    - DP4A processes exactly what's needed
    - Fused quantization avoids kernel launch overhead
    
    Example
    -------
    >>> import torch
    >>> from flashinfer.gemv import gemv_mxfp4_dp4a
    >>> 
    >>> # Simulate decode with batch size 1
    >>> M, K, N = 1, 4096, 4096
    >>> input = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    >>> weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    >>> weight_scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device="cuda")
    >>> 
    >>> output = gemv_mxfp4_dp4a(input, weight, weight_scale)
    >>> output.shape
    torch.Size([1, 4096])
    """
    # Input validation
    assert input.dtype == torch.bfloat16, f"Input must be bfloat16, got {input.dtype}"
    assert weight.dtype == torch.uint8, f"Weight must be uint8 (packed FP4), got {weight.dtype}"
    assert weight_scale.dtype == torch.uint8, f"Weight scale must be uint8 (E8M0), got {weight_scale.dtype}"
    
    M = input.shape[0]
    K = input.shape[1]
    N = weight.shape[0]
    
    # Validate dimensions
    assert weight.shape[1] == K // 2, f"Weight shape mismatch: expected [N, {K//2}], got {weight.shape}"
    assert weight_scale.shape == (N, K // 32), f"Scale shape mismatch: expected [{N}, {K//32}], got {weight_scale.shape}"
    
    # Allocate output if not provided
    if output is None:
        output = torch.empty(M, N, dtype=torch.bfloat16, device=input.device)
    
    # Get the JIT-compiled module
    module = _get_gemv_dp4a_module()
    
    # Call the kernel
    module.gemv_fp4_dp4a(M, N, K, weight, weight_scale, input, output)
    
    return output


def quantize_activations_q8(
    input: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Quantize BF16 activations to Q8_1 format (interleaved for DP4A).
    
    This function quantizes activations once, and the output can be reused
    across multiple GEMV calls (e.g., for different weight matrices).
    
    Args:
        input: [M, K] BF16 activations
        output: Optional [M, K//32, 36] uint8 buffer for Q8_1 blocks
                Each block_q8_1 is 36 bytes: 32 int8 values + half2 scale
    
    Returns:
        [M, K//32, 36] uint8 tensor containing quantized activations
    """
    M, K = input.shape
    assert K % 32 == 0, f"K must be divisible by 32, got {K}"
    
    n_blocks = K // 32
    
    if output is None:
        # block_q8_1 is 36 bytes: 32 x int8 + half2 (4 bytes)
        output = torch.empty(M, n_blocks, 36, dtype=torch.uint8, device=input.device)
    
    module = _get_gemv_dp4a_module()
    module.quantize_activations_q8(M, K, input, output)
    
    return output


def gemv_mxfp4_dp4a_prequant(
    q8_activations: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    K: int,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    GEMV with pre-quantized activations.
    
    This is faster than gemv_mxfp4_dp4a when activations are reused across
    multiple weight matrices (e.g., qkv_proj, o_proj in same layer).
    
    Args:
        q8_activations: [M, K//32, 36] uint8 from quantize_activations_q8
        weight: [N, K//2] uint8 packed FP4 weights
        weight_scale: [N, K//32] uint8 E8M0 scales
        K: Original activation dimension (needed since q8 is packed)
        output: Optional [M, N] BF16 output buffer
    
    Returns:
        [M, N] BF16 tensor: result of input @ weight.T
    """
    M = q8_activations.shape[0]
    N = weight.shape[0]
    
    if output is None:
        output = torch.empty(M, N, dtype=torch.bfloat16, device=weight.device)
    
    module = _get_gemv_dp4a_module()
    module.gemv_fp4_dp4a_prequant(M, N, K, weight, weight_scale, q8_activations, output)
    
    return output


def gemv_mxfp4_dp4a_prefetch(
    q8_activations: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    K: int,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    GEMV with pre-quantized activations and software prefetching.
    
    Experimental version that uses explicit L2 prefetch instructions
    to hide memory latency. May be faster on some workloads.
    
    Args:
        q8_activations: [M, K//32, 36] uint8 from quantize_activations_q8
        weight: [N, K//2] uint8 packed FP4 weights
        weight_scale: [N, K//32] uint8 E8M0 scales
        K: Original activation dimension
        output: Optional [M, N] BF16 output buffer
    
    Returns:
        [M, N] BF16 tensor
    """
    M = q8_activations.shape[0]
    N = weight.shape[0]
    
    if output is None:
        output = torch.empty(M, N, dtype=torch.bfloat16, device=weight.device)
    
    module = _get_gemv_dp4a_module()
    module.gemv_fp4_dp4a_prefetch(M, N, K, weight, weight_scale, q8_activations, output)
    
    return output


def gemv_mxfp4_dp4a_fused_qkv(
    q8_activations: torch.Tensor,
    weight_q: torch.Tensor, scale_q: torch.Tensor,
    weight_k: torch.Tensor, scale_k: torch.Tensor,
    weight_v: torch.Tensor, scale_v: torch.Tensor,
    K: int,
    output_q: Optional[torch.Tensor] = None,
    output_k: Optional[torch.Tensor] = None,
    output_v: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused QKV GEMV: compute Q, K, V projections in single kernel launch.
    
    This is faster than calling gemv_mxfp4_dp4a_prequant 3 times because:
    - Single kernel launch (saves ~5μs overhead per call)
    - Activations loaded once from L2, reused for all 3 matrices
    
    Args:
        q8_activations: [M, K//32, 36] uint8 from quantize_activations_q8
        weight_q: [N_q, K//2] uint8 packed FP4 for Q projection
        scale_q: [N_q, K//32] uint8 E8M0 scales for Q
        weight_k: [N_k, K//2] uint8 packed FP4 for K projection
        scale_k: [N_k, K//32] uint8 E8M0 scales for K
        weight_v: [N_v, K//2] uint8 packed FP4 for V projection
        scale_v: [N_v, K//32] uint8 E8M0 scales for V
        K: Original activation dimension
        output_q/k/v: Optional output buffers
    
    Returns:
        Tuple of (Q, K, V) tensors, each [M, N_x] BF16
    """
    M = q8_activations.shape[0]
    N_q = weight_q.shape[0]
    N_k = weight_k.shape[0]
    N_v = weight_v.shape[0]
    
    if output_q is None:
        output_q = torch.empty(M, N_q, dtype=torch.bfloat16, device=weight_q.device)
    if output_k is None:
        output_k = torch.empty(M, N_k, dtype=torch.bfloat16, device=weight_k.device)
    if output_v is None:
        output_v = torch.empty(M, N_v, dtype=torch.bfloat16, device=weight_v.device)
    
    module = _get_gemv_dp4a_module()
    module.gemv_fp4_dp4a_fused_qkv(
        M, N_q, N_k, N_v, K,
        weight_q, scale_q,
        weight_k, scale_k,
        weight_v, scale_v,
        q8_activations,
        output_q, output_k, output_v
    )
    
    return output_q, output_k, output_v


