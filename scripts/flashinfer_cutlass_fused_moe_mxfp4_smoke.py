import torch


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def main() -> None:
    # We intentionally keep this tiny; we only need to exercise CUTLASS runner init.
    device = "cuda"
    num_tokens = 2
    num_experts = 1
    topk = 1

    # Must satisfy MXFP4 path checks (divisible by 128).
    hidden_size = 256
    inter_size = 256

    # FP8 activations (required for MXFP4 path)
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=torch.float16).to(
        torch.float8_e4m3fn
    )

    # Route every token to expert 0
    token_selected_experts = torch.zeros(
        (num_tokens, topk), device=device, dtype=torch.int32
    )
    token_final_scales = torch.ones((num_tokens, topk), device=device, dtype=torch.float32)

    # Packed FP4 weights surfaced as int64 (FlashInfer / TRTLLM mapping)
    # fc2: [E, hidden, inter/16]
    # fc1 (Swiglu): [E, 2*inter, hidden/16]
    fc2_expert_weights = torch.zeros(
        (num_experts, hidden_size, inter_size // 16), device=device, dtype=torch.int64
    )
    fc1_expert_weights = torch.zeros(
        (num_experts, 2 * inter_size, hidden_size // 16), device=device, dtype=torch.int64
    )

    # Quant scales for isWMxfp4AFp8Quant(): 5 tensors
    # - int32 packed block scales + float32 globals.
    FP8_PER_INT32 = 4
    SFVEC = 32

    # Conservative: for SM12x MXFPX SF vector size 32, the alignments are at least 128.
    # We choose hidden/inter divisible by 128 so align_to() is a no-op.
    MinNDimAlignmentMXFPX = 128
    MinKDimAlignmentMXFPX = 128

    hs_aligned_k = align_to(hidden_size, MinKDimAlignmentMXFPX)
    hs_aligned_n = align_to(hidden_size, MinNDimAlignmentMXFPX)
    inter_aligned_n = align_to(inter_size, MinNDimAlignmentMXFPX)
    inter_aligned_k = align_to(inter_size, MinKDimAlignmentMXFPX)

    fc1_weight_block = torch.zeros(
        (num_experts, inter_aligned_n * 2, hs_aligned_k // (FP8_PER_INT32 * SFVEC)),
        device=device,
        dtype=torch.int32,
    )
    fc2_weight_block = torch.zeros(
        (num_experts, hs_aligned_n, inter_aligned_k // (FP8_PER_INT32 * SFVEC)),
        device=device,
        dtype=torch.int32,
    )
    fc1_global = torch.ones((num_experts,), device=device, dtype=torch.float32)
    fc2_act_global = torch.ones((), device=device, dtype=torch.float32)  # scalar is OK
    fc2_global = torch.ones((num_experts,), device=device, dtype=torch.float32)

    quant_scales = [fc1_weight_block, fc1_global, fc2_act_global, fc2_weight_block, fc2_global]

    # Build the SM120/SM121 module directly (fast build) to avoid compiling unrelated backends.
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    major, minor = torch.cuda.get_device_capability()
    backend = f"{major * 10 + minor}"
    mod = get_cutlass_fused_moe_module(backend, use_fast_build=True)

    out = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    print(
        f"[smoke] backend={backend} x={tuple(x.shape)} w1={tuple(fc1_expert_weights.shape)} w2={tuple(fc2_expert_weights.shape)}",
        flush=True,
    )

    y = mod.cutlass_fused_moe(
        out,
        x,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        None,
        fc2_expert_weights,
        None,
        torch.bfloat16,
        quant_scales,
        None,  # input_sf
        None,  # swiglu_alpha
        None,  # swiglu_beta
        None,  # swiglu_limit
        1,  # tp_size
        0,  # tp_rank
        1,  # ep_size
        0,  # ep_rank
        1,  # cluster_size
        0,  # cluster_rank
        use_packed_weights=False,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        tune_max_num_tokens=256,
        enable_pdl=False,
        activation_type=3,  # ActivationType.Swiglu (value per core.py)
    )

    torch.cuda.synchronize()
    # The function returns (output, [intermediate tensors]) or just output
    if isinstance(y, (list, tuple)):
        y = y[0] if len(y) > 0 else out
    print("[smoke] ok y.shape=", tuple(y.shape), "dtype=", y.dtype, flush=True)


if __name__ == "__main__":
    main()

