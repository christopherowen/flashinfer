#!/usr/bin/env python3
"""Test gated FC1 kernel with 64x64 tile.

This test also compares baseline vs Layer 1A+ (two-GEMM bringup + BF16->FP8 quantization),
controlled via env vars that affect the JIT build.
"""

import os
import torch
torch.manual_seed(42)

from flashinfer.fused_moe import core as moe_core
from flashinfer import mxfp4_quantize, mxfp8_quantize

# Test gated FC1 kernel with 64x64 tile
batch_size = 16
hidden_size = 5120
inter_size = 6528  # intermediate size after gate split
num_experts = 8
top_k = 2

device = torch.device("cuda:0")
dtype = torch.bfloat16

print("Testing gated FC1 kernel with 64x64 tile...")
print(f"  batch_size={batch_size}, hidden_size={hidden_size}, inter_size={inter_size}")
print(f"  num_experts={num_experts}, top_k={top_k}")

# Create input (BF16)
hidden_states = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

# Create gated MLP weights - gate and linear combined
# Shape: [num_experts, 2*inter_size, hidden_size]
gate_up_weights = torch.randn(num_experts, 2 * inter_size, hidden_size, dtype=dtype, device=device)

# Create FC2 weights - Shape: [num_experts, hidden_size, inter_size]
fc2_weights = torch.randn(num_experts, hidden_size, inter_size, dtype=dtype, device=device)

# Create routing (token_final_scales must be float32)
topk_weights = torch.randn(batch_size, top_k, dtype=torch.float32, device=device).abs()
topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
topk_indices = torch.randint(0, num_experts, (batch_size, top_k), device=device, dtype=torch.int32)

# Quantize weights using mxfp4
gate_up_flat = gate_up_weights.view(-1, hidden_size)
gate_up_fp4, gate_up_scales = mxfp4_quantize(gate_up_flat.contiguous())
gate_up_fp4 = gate_up_fp4.view(num_experts, 2 * inter_size, -1)
gate_up_scales = gate_up_scales.view(num_experts, 2 * inter_size, -1)

fc2_flat = fc2_weights.view(-1, inter_size)
fc2_fp4, fc2_scales = mxfp4_quantize(fc2_flat.contiguous())
fc2_fp4 = fc2_fp4.view(num_experts, hidden_size, -1)
fc2_scales = fc2_scales.view(num_experts, hidden_size, -1)

print("  Weights quantized to MXFP4")

# Quantize input to FP8 (required for MXFP4 path)
input_fp8, input_sf = mxfp8_quantize(hidden_states.contiguous())

print("  Input quantized to MXFP8")

major, minor = torch.cuda.get_device_capability()
backend = f"{major * 10 + minor}"

# Reinterpret weights as int64 (required by CUTLASS kernel, packs 16 FP4 values per int64)
# See vLLM's mxfp4.py: layer.w13_weight.contiguous().view(torch.long)
fc1_weights_long = gate_up_fp4.contiguous().view(torch.long)
fc2_weights_long = fc2_fp4.contiguous().view(torch.long)

print(f"  FC1 weights shape (long): {fc1_weights_long.shape}, dtype: {fc1_weights_long.dtype}")
print(f"  FC2 weights shape (long): {fc2_weights_long.shape}, dtype: {fc2_weights_long.dtype}")

# Prepare quant_scales for MXFP4+MXFP8 quantization:
# Expects 4 scales: [fc1_weight_block, fc1_global, fc2_weight_block, fc2_global]
# fc1/fc2_weight_block: int32 (packed 4 FP8 scales per int32)
# fc1/fc2_global: float32 per-expert scale

# Reshape and reinterpret block scales as int32 (4 FP8 per int32)
fc1_block_scales = gate_up_scales.view(num_experts, 2 * inter_size, -1).contiguous().view(torch.int32)
fc2_block_scales = fc2_scales.view(num_experts, hidden_size, -1).contiguous().view(torch.int32)

# Create global scales (1 per expert, float32)
fc1_global = torch.ones(num_experts, dtype=torch.float32, device=device)
fc2_global = torch.ones(num_experts, dtype=torch.float32, device=device)

quant_scales = [fc1_block_scales, fc1_global, fc2_block_scales, fc2_global]

print(f"  FC1 block scales shape: {fc1_block_scales.shape}, dtype: {fc1_block_scales.dtype}")
print(f"  FC2 block scales shape: {fc2_block_scales.shape}, dtype: {fc2_block_scales.dtype}")

def run_once(label: str, env_overrides: dict[str, str | None]):
    env_keys = [
        "FLASHINFER_GATED_FC1_LAUNCH",
        "FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP",
        "FLASHINFER_GATED_FC1_TWO_GEMM_NO_GATE_OFFSET",
        "FLASHINFER_JIT_DEBUG",
        "FLASHINFER_JIT_VERBOSE",
        "FLASHINFER_FUSED_MOE_BUILD_PROFILE",
    ]
    saved = {k: os.environ.get(k) for k in env_keys}
    try:
        for k, v in env_overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        # IMPORTANT: this function is cached; clear it when env vars change.
        moe_core.get_cutlass_fused_moe_module.cache_clear()

        # Force build/load with current env flags
        _ = moe_core.get_cutlass_fused_moe_module(backend, use_fast_build=True, tile_mn=(128, 128))

        print(f"  [{label}] calling cutlass_fused_moe...")
        out = moe_core.cutlass_fused_moe(
            input=input_fp8,
            token_selected_experts=topk_indices,
            token_final_scales=topk_weights,
            fc1_expert_weights=fc1_weights_long,
            fc2_expert_weights=fc2_weights_long,
            quant_scales=quant_scales,  # [fc1_block, fc1_global, fc2_block, fc2_global]
            output_dtype=dtype,
            input_sf=input_sf,
            use_mxfp8_act_scaling=True,  # Enable MXFP8 activation scaling
            auto_tile_select=False,      # Force 128x128 default tile
        )
        if isinstance(out, list):
            out = out[0]
        torch.cuda.synchronize()

        finite = torch.isfinite(out).all().item()
        out_f = out.float()
        print(f"  [{label}] finite={finite}, mean={out_f.mean().item():.4f}, std={out_f.std().item():.4f}")
        return out
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


print("  Running baseline vs Layer 1A+ comparison...")

try:
    # Baseline: no gated FC1 bringup
    out_base = run_once(
        "baseline",
        {
            "FLASHINFER_GATED_FC1_LAUNCH": None,
            "FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP": None,
            "FLASHINFER_GATED_FC1_TWO_GEMM_NO_GATE_OFFSET": None,
            "FLASHINFER_JIT_DEBUG": "0",
            "FLASHINFER_JIT_VERBOSE": "0",
            "FLASHINFER_FUSED_MOE_BUILD_PROFILE": "mxfp4_minimal",
        },
    )

    # Layer 1A+ (current implementation): two-GEMM + BF16 SwiGLU then BF16->FP8 quantize for FC2
    out_1a_plus = run_once(
        "layer1a_plus",
        {
            "FLASHINFER_GATED_FC1_LAUNCH": "1",
            "FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP": "1",
            "FLASHINFER_GATED_FC1_TWO_GEMM_NO_GATE_OFFSET": None,
            "FLASHINFER_JIT_DEBUG": "0",
            "FLASHINFER_JIT_VERBOSE": "0",
            "FLASHINFER_FUSED_MOE_BUILD_PROFILE": "mxfp4_minimal",
        },
    )

    # Compare
    diff = (out_1a_plus - out_base).float()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    denom = out_base.float().abs().mean().item() + 1e-6
    rel_mean = mean_abs / denom
    print(f"  Compare: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, rel_mean={rel_mean:.6f}")

except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
