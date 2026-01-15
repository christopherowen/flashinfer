
import argparse
import functools
import numpy as np
import torch
import torch.nn.functional as F
from flashinfer import mxfp8_quantize, mxfp4_quantize
from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
from flashinfer.testing.utils import bench_gpu_time

def compute_routing(router_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts

def benchmark_sm120_tiles(
    num_tokens=1,
    hidden_size=5120,
    intermediate_size=14336,
    num_experts=8,
    top_k=8,
    backend="120"
):
    print(f"Benchmarking SM120 tiles for tokens={num_tokens}, top_k={top_k}, backend={backend}")
    
    # Check if SM120 is supported/available
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Create dummy inputs
    e = num_experts
    n = intermediate_size
    k = hidden_size
    
    # Inputs
    x = torch.randn(num_tokens, k, device=device, dtype=dtype)
    router_logits = torch.randn(num_tokens, e, device=device, dtype=dtype)
    
    # Routing
    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    selected_experts = selected_experts.int()
    
    # Weights (Unquantized for now, will quantize)
    w1 = torch.randn(e, 2 * n, k, device=device, dtype=dtype) # FC1: K -> 2*N
    w2 = torch.randn(e, k, n, device=device, dtype=dtype)     # FC2: N -> K
    
    # Quantize weights (MXFP4)
    # w1 shape: (E, 2*N, K). mxfp4_quantize returns scales for blocked K?
    w1_q, w1_scale = mxfp4_quantize(w1)
    w2_q, w2_scale = mxfp4_quantize(w2)
    
    # Quantize input (MXFP8)
    x_q, x_scale = mxfp8_quantize(x)
    
    # Output buffer
    output = torch.empty(num_tokens, k, device=device, dtype=dtype)
    
    # Scales preparation for runner
    # Reshape scales to 3D: (num_experts, rows, cols_blocks)
    # w1_scale from mxfp4 is flattened 2D: (E * 2*N, K/32). Reshape to (E, 2*N, K/32)
    w1_s = w1_scale.view(num_experts, 2 * intermediate_size, -1)
    # w2_scale from mxfp4 (on (E, K, N)) is flattened 2D: (E * K, N/32). Reshape to (E, K, N/32)
    w2_s = w2_scale.view(num_experts, hidden_size, -1)
    
    tiles_to_test = [(128, 128)]  # known-good baseline
    results = {}
    
    for tile_mn in tiles_to_test:
        print(f"\nTesting tile_mn={tile_mn}...")
        try:
            # We explicitly request the module with specific logical tile (M,N)
            runner = get_cutlass_fused_moe_module(backend="120", tile_mn=tile_mn)
            
            # Helper to run
            def run_fn():
                # Prepare quant scales list (4 tensors for MXFP4: FC1 Block, FC1 Global, FC2 Block, FC2 Global)
                # Note: block scales must be viewed as int32 (packed 4x uint8)
                qs = [
                    w1_s.view(torch.int32), # FC1 block scales
                    torch.ones(num_experts, device="cuda", dtype=torch.float32), # FC1 global scale
                    w2_s.view(torch.int32), # FC2 block scales
                    torch.ones(num_experts, device="cuda", dtype=torch.float32), # FC2 global scale
                ]
                
                runner.cutlass_fused_moe(
                    output,                 # output
                    x_q,                    # input
                    selected_experts,       # token_selected_experts
                    routing_weights,        # token_final_scales
                    w1_q.view(torch.int64), # fc1_expert_weights - VIEW AS INT64
                    None,                   # fc1_expert_biases
                    w2_q.view(torch.int64), # fc2_expert_weights - VIEW AS INT64
                    None,                   # fc2_expert_biases
                    dtype,                  # output_dtype
                    qs,                     # quant_scales - POSITIONAL
                    input_sf=x_scale, 
                    use_mxfp8_act_scaling=True, # Enable MXFP8 activation scaling
                )
            
            # Warmup
            print("Warmup...")
            for _ in range(5):
                run_fn()
            torch.cuda.synchronize()
            
            # Bench
            print("Benchmarking...")
            times = bench_gpu_time(run_fn)
            avg_time = np.mean(times)
            results[tile_mn] = avg_time
            print(f"tile_mn={tile_mn}: {avg_time*1000:.3f} us")
            
        except Exception as e:
            print(f"Failed tile_mn={tile_mn}: {e}")
            import traceback
            traceback.print_exc()

    print("\nSummary:")
    for tile_mn, t in results.items():
        print(f"tile_mn={tile_mn}: {t*1000:.3f} us")

if __name__ == "__main__":
    benchmark_sm120_tiles()
