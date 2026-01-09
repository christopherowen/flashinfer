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


