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
) -> Dict:
    """Create test data for MoE GEMM benchmark.
    
    Returns:
        Dict with activations, weights, and routing info
    """
    device = torch.device("cuda")
    
    # Activations [num_tokens, hidden_dim]
    activations = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    
    # FP4 weights per expert [num_experts, intermediate_dim, hidden_dim]
    # For simplicity, use intermediate_dim = 4 * hidden_dim (typical MLP expansion)
    intermediate_dim = 4 * hidden_dim
    
    # Create random weights (in real case, these would be quantized FP4)
    # For benchmark, we simulate with random data
    weights = torch.randn(num_experts, intermediate_dim, hidden_dim, dtype=dtype, device=device)
    
    # Routing: assign topk experts per token
    expert_indices = torch.randint(0, num_experts, (num_tokens, topk), device=device)
    expert_weights = torch.softmax(torch.randn(num_tokens, topk, device=device), dim=-1)
    
    return {
        "activations": activations,
        "weights": weights,
        "expert_indices": expert_indices,
        "expert_weights": expert_weights,
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_experts": num_experts,
        "topk": topk,
    }


# =============================================================================
# BENCHMARK STATUS: PLACEHOLDER
# =============================================================================
# This benchmark currently uses torch.matmul as a PLACEHOLDER for the actual
# FlashInfer SM121 MoE GEMM path. It does NOT validate:
#   - The CUTLASS grouped GEMM path
#   - The activation quantizer cost
#   - Kernel selection heuristics
#   - End-to-end MoE performance
#
# To be meaningful, this benchmark must:
# 1. Prepare inputs in FlashInfer's grouped GEMM format (grouped pointers, shapes)
# 2. Call the actual FlashInfer fused MoE entrypoint
# 3. Time separately:
#    a. Quantizer alone (BF16->FP8 + SFA generation)
#    b. GEMM alone (FP8xFP4 grouped GEMM)
#    c. Quantizer + GEMM together (end-to-end)
#
# TODO: Once FlashInfer SM121 MoE API is wired up:
# - Replace torch.matmul with flashinfer.moe.mxfp4_grouped_gemm() or similar
# - Add quantizer timing with CUDA events
# - Report kernel selected and tile config
# =============================================================================

_USE_REAL_FLASHINFER_PATH = False  # Set to True once API is available


def benchmark_prefill(hidden_dim: int = 4096, num_experts: int = 8, num_warmup: int = 5, num_iters: int = 20):
    """Benchmark prefill-like workload (large M per group).
    
    WARNING: Currently using torch.matmul placeholder, NOT the actual SM121 path.
    """
    
    print("\n" + "="*60)
    print("PREFILL REGIME BENCHMARK")
    if not _USE_REAL_FLASHINFER_PATH:
        print("*** PLACEHOLDER MODE (torch.matmul) - not real SM121 path ***")
    print("="*60)
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num experts: {num_experts}")
    
    # Prefill workload: larger batch sizes
    batch_sizes = [64, 128, 256, 512, 1024]
    topk = 2
    
    results = []
    
    for num_tokens in batch_sizes:
        data = create_moe_test_data(num_tokens, hidden_dim, num_experts, topk)
        
        # Calculate theoretical FLOPS
        tokens_per_expert = num_tokens * topk / num_experts
        intermediate_dim = data["intermediate_dim"]
        flops_per_expert = 2 * tokens_per_expert * hidden_dim * intermediate_dim
        total_flops = flops_per_expert * num_experts
        
        if _USE_REAL_FLASHINFER_PATH:
            # TODO: Replace with actual FlashInfer MoE call
            # quantizer_time, gemm_time, total_time = flashinfer.moe.benchmark_mxfp4_grouped_gemm(...)
            pass
        else:
            # PLACEHOLDER: Simple matmul (not actual grouped GEMM)
            def run_placeholder():
                return torch.matmul(data["activations"], data["weights"][0].T)
        
        # Warmup
        for _ in range(num_warmup):
            _ = run_placeholder() if not _USE_REAL_FLASHINFER_PATH else None
            torch.cuda.synchronize()
        
        # Timed iterations
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(num_iters):
            _ = run_placeholder() if not _USE_REAL_FLASHINFER_PATH else None
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time_ms = (end - start) / num_iters * 1000
        tflops = total_flops / (avg_time_ms / 1000) / 1e12
        
        results.append({
            "num_tokens": num_tokens,
            "avg_time_ms": avg_time_ms,
            "tflops": tflops,
            "tokens_per_expert": tokens_per_expert,
            "is_placeholder": not _USE_REAL_FLASHINFER_PATH,
        })
        
        marker = "[PLACEHOLDER]" if not _USE_REAL_FLASHINFER_PATH else ""
        print(f"\n  Tokens: {num_tokens:5d}  |  Time: {avg_time_ms:8.3f} ms  |  {tflops:.2f} TFLOPS {marker}")
    
    return results


def benchmark_decode(hidden_dim: int = 4096, num_experts: int = 8, num_warmup: int = 10, num_iters: int = 50):
    """Benchmark decode-like workload (small M per group).
    
    WARNING: Currently using torch.matmul placeholder, NOT the actual SM121 path.
    For decode, tile selection is CRITICAL - wrong tile = massive latency.
    """
    
    print("\n" + "="*60)
    print("DECODE REGIME BENCHMARK")
    if not _USE_REAL_FLASHINFER_PATH:
        print("*** PLACEHOLDER MODE (torch.matmul) - not real SM121 path ***")
    print("="*60)
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num experts: {num_experts}")
    
    # Decode workload: small batch sizes (1-32 tokens)
    batch_sizes = [1, 2, 4, 8, 16, 32]
    topk = 2
    
    results = []
    
    for num_tokens in batch_sizes:
        data = create_moe_test_data(num_tokens, hidden_dim, num_experts, topk)
        
        # Calculate theoretical FLOPS
        tokens_per_expert = num_tokens * topk / num_experts
        intermediate_dim = data["intermediate_dim"]
        flops_per_expert = 2 * tokens_per_expert * hidden_dim * intermediate_dim
        total_flops = flops_per_expert * num_experts
        
        if _USE_REAL_FLASHINFER_PATH:
            # TODO: Replace with actual FlashInfer MoE call
            pass
        else:
            # PLACEHOLDER
            def run_placeholder():
                return torch.matmul(data["activations"], data["weights"][0].T)
        
        # Warmup
        for _ in range(num_warmup):
            _ = run_placeholder() if not _USE_REAL_FLASHINFER_PATH else None
            torch.cuda.synchronize()
        
        # Timed iterations
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(num_iters):
            _ = run_placeholder() if not _USE_REAL_FLASHINFER_PATH else None
        
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
            "is_placeholder": not _USE_REAL_FLASHINFER_PATH,
        })
        
        marker = "[PLACEHOLDER]" if not _USE_REAL_FLASHINFER_PATH else ""
        print(f"\n  Tokens: {num_tokens:3d}  |  Time: {avg_time_ms:8.3f} ms  |  {tokens_per_sec:.0f} tok/s {marker}")
    
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

