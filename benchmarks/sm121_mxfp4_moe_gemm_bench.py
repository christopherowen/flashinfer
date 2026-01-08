#!/usr/bin/env python3
"""
SM121 MXFP4 MoE GEMM Microbenchmark

This benchmark measures performance of the SM121 native FP4 MoE GEMM path
for both prefill-like (large M) and decode-like (small M) workloads.

Usage:
    python benchmarks/sm121_mxfp4_moe_gemm_bench.py [--regime prefill|decode|both]

Requirements:
    - NVIDIA GPU with SM121 compute capability (e.g., GB10)
    - FlashInfer installed with SM121 support
    - CUDA 12.8+ with FP4 support
"""

import argparse
import time
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np

# FlashInfer imports
try:
    import flashinfer
    from flashinfer.utils import get_compute_capability
    from flashinfer.fused_moe import (
        trtllm_fp4_block_scale_moe,
        trtllm_bf16_moe,
        RoutingMethodType,
        GatedActType,
    )
    _HAS_FUSED_MOE = True
except ImportError as e:
    print(f"WARNING: FlashInfer fused_moe not available: {e}")
    _HAS_FUSED_MOE = False

try:
    import flashinfer
    from flashinfer.utils import get_compute_capability
except ImportError:
    print("ERROR: FlashInfer not installed. Please install first.")
    exit(1)


def check_sm121():
    """Check if we're running on SM121."""
    cc = get_compute_capability(torch.device("cuda"))
    major, minor = cc
    if major != 12 or minor != 1:
        print(f"WARNING: This benchmark is designed for SM121, but running on SM{major}{minor}")
        return False
    return True


def create_moe_test_data(
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
    use_fp4_weights: bool = True,
) -> Dict:
    """Create test data for MoE GEMM benchmark.
    
    Args:
        num_tokens: Number of input tokens
        hidden_dim: Hidden dimension (must be divisible by 128 for block scaling)
        num_experts: Number of experts
        topk: Number of experts per token
        dtype: Data type for activations (bfloat16 or float16)
        use_fp4_weights: If True, create FP4 quantized weights for real benchmark
    
    Returns:
        Dict with activations, weights, scales, and routing info
    """
    device = torch.device("cuda")
    
    # Intermediate dim = 4 * hidden_dim (typical MLP expansion)
    intermediate_dim = 4 * hidden_dim
    
    # Routing logits [num_tokens, num_experts]
    routing_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)
    
    # Hidden states [num_tokens, hidden_dim]
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    
    if use_fp4_weights:
        # FP4 weights are packed into uint8 (2 FP4 values per byte)
        # Weight shapes for gated MLP (gate + up projection combined):
        #   gemm1: [num_experts, 2*intermediate_dim, hidden_dim // 2] (packed)
        #   gemm2: [num_experts, hidden_dim, intermediate_dim // 2] (packed)
        
        gemm1_weights = torch.randint(
            0, 256, 
            (num_experts, 2 * intermediate_dim, hidden_dim // 2),
            dtype=torch.uint8, device=device
        )
        gemm2_weights = torch.randint(
            0, 256,
            (num_experts, hidden_dim, intermediate_dim // 2),
            dtype=torch.uint8, device=device
        )
        
        # Scale factors for FP4 block scaling (float8_e4m3fn)
        # For MXFP4: scales are [num_experts, dim, hidden // 32]
        gemm1_scales = torch.ones(
            num_experts, 2 * intermediate_dim, hidden_dim // 32,
            dtype=torch.float8_e4m3fn, device=device
        )
        gemm2_scales = torch.ones(
            num_experts, hidden_dim, intermediate_dim // 32,
            dtype=torch.float8_e4m3fn, device=device
        )
        
        # Hidden states scale for identity mode (all ones = 0x7F in UE8M0)
        # Shape: [num_tokens, hidden_dim // 32] for MXFP8
        hidden_states_scale = torch.ones(
            num_tokens, hidden_dim // 32,
            dtype=torch.float8_e4m3fn, device=device
        )
    else:
        # BF16 weights for placeholder mode
        gemm1_weights = torch.randn(
            num_experts, 2 * intermediate_dim, hidden_dim,
            dtype=dtype, device=device
        )
        gemm2_weights = torch.randn(
            num_experts, hidden_dim, intermediate_dim,
            dtype=dtype, device=device
        )
        gemm1_scales = None
        gemm2_scales = None
        hidden_states_scale = None
    
    return {
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "routing_logits": routing_logits,
        "gemm1_weights": gemm1_weights,
        "gemm1_scales": gemm1_scales,
        "gemm2_weights": gemm2_weights,
        "gemm2_scales": gemm2_scales,
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_experts": num_experts,
        "topk": topk,
        "use_fp4_weights": use_fp4_weights,
    }


# =============================================================================
# BENCHMARK MODES
# =============================================================================
# Mode 1: Real FlashInfer FP4 MoE path (requires SM100+ and FP4 support)
# Mode 2: BF16 baseline (for comparison)
# Mode 3: Placeholder (fallback when APIs unavailable)
# =============================================================================

def run_real_fp4_moe(data: Dict) -> torch.Tensor:
    """Run real FlashInfer FP4 block-scaled MoE."""
    if not _HAS_FUSED_MOE:
        raise RuntimeError("FlashInfer fused_moe not available")
    
    return trtllm_fp4_block_scale_moe(
        routing_logits=data["routing_logits"],
        routing_bias=None,
        hidden_states=data["hidden_states"],
        hidden_states_scale=data["hidden_states_scale"],
        gemm1_weights=data["gemm1_weights"],
        gemm1_weights_scale=data["gemm1_scales"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=data["gemm2_weights"],
        gemm2_weights_scale=data["gemm2_scales"],
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=data["num_experts"],
        top_k=data["topk"],
        n_group=None,
        topk_group=None,
        intermediate_size=data["intermediate_dim"],
        local_expert_offset=0,
        local_num_experts=data["num_experts"],
        routed_scaling_factor=None,
        routing_method_type=int(RoutingMethodType.Default),
        do_finalize=True,
        gated_act_type=int(GatedActType.Silu) if hasattr(GatedActType, 'Silu') else 0,
    )[0]


def run_bf16_moe(data: Dict) -> torch.Tensor:
    """Run BF16 MoE baseline."""
    if not _HAS_FUSED_MOE:
        raise RuntimeError("FlashInfer fused_moe not available")
    
    return trtllm_bf16_moe(
        routing_logits=data["routing_logits"],
        routing_bias=None,
        hidden_states=data["hidden_states"],
        gemm1_weights=data["gemm1_weights"],
        gemm2_weights=data["gemm2_weights"],
        num_experts=data["num_experts"],
        top_k=data["topk"],
        n_group=None,
        topk_group=None,
        intermediate_size=data["intermediate_dim"],
        local_expert_offset=0,
        local_num_experts=data["num_experts"],
        routed_scaling_factor=None,
        routing_method_type=int(RoutingMethodType.Default),
    )


def run_placeholder(data: Dict) -> torch.Tensor:
    """Run placeholder (simple matmul)."""
    return torch.matmul(data["hidden_states"], data["gemm1_weights"][0, :data["hidden_dim"], :].T)


def benchmark_prefill(hidden_dim: int = 4096, num_experts: int = 8, num_warmup: int = 5, num_iters: int = 20):
    """Benchmark prefill-like workload (large M per group)."""
    
    print("\n" + "="*60)
    print("PREFILL REGIME BENCHMARK")
    print("="*60)
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num experts: {num_experts}")
    print(f"  FlashInfer fused_moe available: {_HAS_FUSED_MOE}")
    
    # Prefill workload: larger batch sizes
    batch_sizes = [64, 128, 256, 512, 1024]
    topk = 2
    
    results = []
    
    for num_tokens in batch_sizes:
        # Try FP4 path first, fall back to BF16, then placeholder
        use_fp4 = _HAS_FUSED_MOE
        use_bf16 = False
        mode = "FP4"
        
        if use_fp4:
            try:
                data = create_moe_test_data(num_tokens, hidden_dim, num_experts, topk, use_fp4_weights=True)
                run_fn = lambda: run_real_fp4_moe(data)
                # Test run
                _ = run_fn()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  FP4 path failed: {e}, falling back to BF16")
                use_fp4 = False
                use_bf16 = _HAS_FUSED_MOE
        
        if not use_fp4 and use_bf16:
            try:
                data = create_moe_test_data(num_tokens, hidden_dim, num_experts, topk, use_fp4_weights=False)
                run_fn = lambda: run_bf16_moe(data)
                mode = "BF16"
                _ = run_fn()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  BF16 path failed: {e}, falling back to placeholder")
                use_bf16 = False
        
        if not use_fp4 and not use_bf16:
            data = create_moe_test_data(num_tokens, hidden_dim, num_experts, topk, use_fp4_weights=False)
            run_fn = lambda: run_placeholder(data)
            mode = "PLACEHOLDER"
        
        # Calculate theoretical FLOPS
        tokens_per_expert = num_tokens * topk / num_experts
        intermediate_dim = data["intermediate_dim"]
        flops_per_expert = 2 * tokens_per_expert * hidden_dim * intermediate_dim
        total_flops = flops_per_expert * num_experts
        
        # Warmup
        for _ in range(num_warmup):
            _ = run_fn()
            torch.cuda.synchronize()
        
        # Timed iterations
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(num_iters):
            _ = run_fn()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time_ms = (end - start) / num_iters * 1000
        tflops = total_flops / (avg_time_ms / 1000) / 1e12
        
        results.append({
            "num_tokens": num_tokens,
            "avg_time_ms": avg_time_ms,
            "tflops": tflops,
            "tokens_per_expert": tokens_per_expert,
            "mode": mode,
        })
        
        print(f"\n  Tokens: {num_tokens:5d}  |  Time: {avg_time_ms:8.3f} ms  |  {tflops:.2f} TFLOPS [{mode}]")
    
    return results


def benchmark_decode(hidden_dim: int = 4096, num_experts: int = 8, num_warmup: int = 10, num_iters: int = 50):
    """Benchmark decode-like workload (small M per group).
    
    For decode, tile selection is CRITICAL - wrong tile = massive latency.
    This tests the SM121 FP4 path with small batch sizes (1-32 tokens).
    """
    
    print("\n" + "="*60)
    print("DECODE REGIME BENCHMARK")
    print("="*60)
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num experts: {num_experts}")
    print(f"  FlashInfer fused_moe available: {_HAS_FUSED_MOE}")
    
    # Decode workload: small batch sizes (1-32 tokens)
    batch_sizes = [1, 2, 4, 8, 16, 32]
    topk = 2
    
    results = []
    
    for num_tokens in batch_sizes:
        # Try FP4 path first, fall back to BF16, then placeholder
        use_fp4 = _HAS_FUSED_MOE
        use_bf16 = False
        mode = "FP4"
        
        if use_fp4:
            try:
                data = create_moe_test_data(num_tokens, hidden_dim, num_experts, topk, use_fp4_weights=True)
                run_fn = lambda: run_real_fp4_moe(data)
                # Test run
                _ = run_fn()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  FP4 path failed for {num_tokens} tokens: {e}, falling back")
                use_fp4 = False
                use_bf16 = _HAS_FUSED_MOE
        
        if not use_fp4 and use_bf16:
            try:
                data = create_moe_test_data(num_tokens, hidden_dim, num_experts, topk, use_fp4_weights=False)
                run_fn = lambda: run_bf16_moe(data)
                mode = "BF16"
                _ = run_fn()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  BF16 path failed: {e}, falling back to placeholder")
                use_bf16 = False
        
        if not use_fp4 and not use_bf16:
            data = create_moe_test_data(num_tokens, hidden_dim, num_experts, topk, use_fp4_weights=False)
            run_fn = lambda: run_placeholder(data)
            mode = "PLACEHOLDER"
        
        # Calculate theoretical FLOPS
        tokens_per_expert = num_tokens * topk / num_experts
        intermediate_dim = data["intermediate_dim"]
        flops_per_expert = 2 * tokens_per_expert * hidden_dim * intermediate_dim
        total_flops = flops_per_expert * num_experts
        
        # Warmup
        for _ in range(num_warmup):
            _ = run_fn()
            torch.cuda.synchronize()
        
        # Timed iterations
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(num_iters):
            _ = run_fn()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time_ms = (end - start) / num_iters * 1000
        
        # For small batches, measure throughput in tokens/sec
        tokens_per_sec = num_tokens / (avg_time_ms / 1000)
        
        results.append({
            "num_tokens": num_tokens,
            "avg_time_ms": avg_time_ms,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_expert": tokens_per_expert,
            "mode": mode,
        })
        
        print(f"\n  Tokens: {num_tokens:3d}  |  Time: {avg_time_ms:8.3f} ms  |  {tokens_per_sec:.0f} tok/s [{mode}]")
    
    return results


def benchmark_tile_comparison(hidden_dim: int = 4096):
    """Compare different tile configurations."""
    
    print("\n" + "="*60)
    print("TILE CONFIGURATION COMPARISON")
    print("="*60)
    
    # Common SM120 tile configurations from generate_kernels.py
    tile_configs = [
        {"m": 64, "n": 128, "k": 128, "desc": "64x128x128 (decode)"},
        {"m": 128, "n": 128, "k": 128, "desc": "128x128x128 (balanced)"},
        {"m": 128, "n": 256, "k": 128, "desc": "128x256x128 (wide N)"},
        {"m": 64, "n": 192, "k": 128, "desc": "64x192x128 (for dim 2880)"},
    ]
    
    print("\nNote: Actual kernel selection depends on FlashInfer's heuristics.")
    print("This shows the tile configurations available for SM120/SM121:\n")
    
    for config in tile_configs:
        print(f"  {config['desc']}")
        print(f"    M-tile: {config['m']:3d}  N-tile: {config['n']:3d}  K-tile: {config['k']:3d}")
    
    print("\n  SM120 N-tiles include 192 for models with hidden_dim = 2880")
    print("  (gpt-oss-120b uses 2880 and 2880 % 192 = 0)")


def main():
    parser = argparse.ArgumentParser(description="SM121 MXFP4 MoE GEMM Benchmark")
    parser.add_argument(
        "--regime",
        choices=["prefill", "decode", "both", "tiles"],
        default="both",
        help="Benchmark regime: prefill (large M), decode (small M), both, or tiles (config comparison)"
    )
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SM121 MXFP4 MoE GEMM Microbenchmark")
    print("="*60)
    
    # Check device
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    device_name = torch.cuda.get_device_name()
    print(f"\nDevice: {device_name}")
    
    is_sm121 = check_sm121()
    if not is_sm121:
        print("\nProceeding anyway for comparison purposes...")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num experts: {args.num_experts}")
    print(f"  Warmup iters: {args.warmup}")
    print(f"  Timed iters: {args.iters}")
    
    # Run benchmarks
    if args.regime == "prefill" or args.regime == "both":
        benchmark_prefill(args.hidden_dim, args.num_experts, args.warmup, args.iters)
    
    if args.regime == "decode" or args.regime == "both":
        benchmark_decode(args.hidden_dim, args.num_experts, args.warmup * 2, args.iters * 2)
    
    if args.regime == "tiles":
        benchmark_tile_comparison(args.hidden_dim)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print("\nNOTE: This is a placeholder benchmark using torch.matmul.")
    print("Replace with actual FlashInfer MoE GEMM calls when the SM121")
    print("native FP4 path is fully integrated.")
    print("\nTo verify native FP4 usage, run:")
    print("  python scripts/verify_sm121_fp4_mma.py")


if __name__ == "__main__":
    main()

