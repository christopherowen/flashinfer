import time
import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from flashinfer import mxfp4_quantize, mxfp8_quantize

def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

log("Reproducing SM121 MXFP4 MoE Crash...")

device = "cuda"
num_tokens = 128  # Keep small; JIT compile time is mostly independent of this
hidden_size = 2944
intermediate_size = 5888 // 2  # 2944
num_experts = 128
top_k = 4  # From log: token_selected_experts shape (4096, 4)

# Create input (BF16 -> FP8)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
log("Allocating input + quantizing to FP8 (mxfp8_quantize)...")
input_bf16 = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
input_fp8, input_scale = mxfp8_quantize(input_bf16, True, 32)

# Routing
log("Allocating routing + topk...")
routing_logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=device)
topk_weights, topk_indices = torch.topk(routing_logits, top_k, dim=-1)
topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.float32)
topk_indices = topk_indices.to(torch.int32)

# Weights (MXFP4)
# NOTE: These are dummy buffers to reach initialize(); they are not valid packed FP4 weights.
log("Allocating dummy weights/scales/biases...")
# FC1: [experts, intermediate*2, hidden/2] -> [128, 5888, 2944/16=184]
fc1_weights = torch.randint(0, 100, (num_experts, intermediate_size * 2, hidden_size // 16), dtype=torch.int64, device=device)
# FC2: [experts, hidden, intermediate/2] -> [128, 2944, 2944/16=184]
fc2_weights = torch.randint(0, 100, (num_experts, hidden_size, intermediate_size // 16), dtype=torch.int64, device=device)

# Scales
# fc1_weight_block_scale: [128, 5888, 23] (23 * 128 = 2944)
fc1_scales = torch.randint(0, 100, (num_experts, intermediate_size * 2, 23), dtype=torch.int32, device=device)
fc2_scales = torch.randint(0, 100, (num_experts, hidden_size, 23), dtype=torch.int32, device=device)

fc1_global = torch.randn(num_experts, dtype=torch.float32, device=device)
fc2_global = torch.randn(num_experts, dtype=torch.float32, device=device)

quant_scales = [fc1_scales, fc1_global, fc2_scales, fc2_global]

# Biases
fc1_bias = torch.randn(num_experts, intermediate_size * 2, dtype=torch.bfloat16, device=device)
fc2_bias = torch.randn(num_experts, hidden_size, dtype=torch.bfloat16, device=device)

# Swiglu params
swiglu_alpha = torch.randn(num_experts, dtype=torch.float32, device=device)
swiglu_beta = torch.randn(num_experts, dtype=torch.float32, device=device)
swiglu_limit = torch.randn(num_experts, dtype=torch.float32, device=device)

print('Input shapes:')
print(f'  input: {input_fp8.shape}, {input_fp8.dtype}')
print(f'  input_scale: {input_scale.shape}, {input_scale.dtype}')
print(f'  fc1_weights: {fc1_weights.shape}')
print(f'  fc1_scales: {fc1_scales.shape}')

try:
    log("Calling cutlass_fused_moe (first call may JIT compile and take minutes)...")
    output = cutlass_fused_moe(
        input=input_fp8,
        input_sf=input_scale,
        token_selected_experts=topk_indices,
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_weights,
        fc2_expert_weights=fc2_weights,
        output_dtype=torch.bfloat16,
        quant_scales=quant_scales,
        fc1_expert_biases=fc1_bias,
        fc2_expert_biases=fc2_bias,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        activation_type=ActivationType.Swiglu,
        use_mxfp8_act_scaling=True,
    )
    torch.cuda.synchronize()
    log("cutlass_fused_moe returned")
    print('SUCCESS!')
    if isinstance(output, list):
        print(f'Output shape: {output[0].shape}')
    else:
        print(f'Output shape: {output.shape}')

except Exception as e:
    print('CRASH REPRODUCED:')
    print(e)
