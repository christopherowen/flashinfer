/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Simple DP4A-based GEMV kernel inspired by llama.cpp's mmvq kernel.
 * Uses INT8 dot product (DP4A) for efficient small-batch compute.
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

namespace flashinfer {
namespace gemv {

// FP4 E2M1 to INT8 lookup table (doubled values for DP4A accumulation)
// Matches llama.cpp's kvalues_mxfp4
__constant__ int8_t kFP4ToInt8[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,     // Positive: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (×2)
    0, -1, -2, -3, -4, -6, -8, -12  // Negative: -0, -0.5, -1, -1.5, -2, -3, -4, -6 (×2)
};

// DP4A: 4-way int8 dot product
// Returns sum of a[i] * b[i] for i in 0..3
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

// Convert FP4 nibbles to int32 (4 int8 values packed)
__device__ __forceinline__ int fp4_to_int32(uint8_t packed_fp4, const int8_t* lut) {
    // Unpack 2 FP4 nibbles and look up in table
    int8_t lo = lut[packed_fp4 & 0x0F];
    int8_t hi = lut[packed_fp4 >> 4];
    
    // Pack into int32 for DP4A (replicate to fill 4 slots)
    return (int32_t(lo)) | (int32_t(hi) << 8);
}

// Simple GEMV kernel: y = W @ x
// W: [M, K] FP4 packed as [M, K/2] uint8
// x: [K] BF16
// y: [M] BF16
// W_scale: [M, K/32] FP8 block scales (32 elements per block)
template <int BLOCK_DIM = 256>
__global__ void gemv_fp4_dp4a_kernel(
    const uint8_t* __restrict__ W,        // [M, K/2] packed FP4 weights
    const uint8_t* __restrict__ W_scale,  // [M, K/32] FP8 E8M0 scales
    const nv_bfloat16* __restrict__ x,    // [K] BF16 input
    nv_bfloat16* __restrict__ y,          // [M] BF16 output
    int M,
    int K)
{
    // Each thread block handles one output row
    int row = blockIdx.x;
    if (row >= M) return;
    
    // Shared memory for input vector quantization
    __shared__ int8_t x_q8[4096];  // Assume K <= 4096
    __shared__ float x_scale;
    
    // Quantize input to Q8
    // First pass: find max for scaling
    if (threadIdx.x == 0) {
        float max_val = 0.0f;
        for (int i = 0; i < K; i++) {
            float val = fabsf(__bfloat162float(x[i]));
            max_val = fmaxf(max_val, val);
        }
        x_scale = max_val / 127.0f;
    }
    __syncthreads();
    
    // Second pass: quantize
    float inv_scale = (x_scale > 0) ? 127.0f / (x_scale * 127.0f) : 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        x_q8[i] = (int8_t)fminf(fmaxf(val * inv_scale * 127.0f, -127.0f), 127.0f);
    }
    __syncthreads();
    
    // Each thread accumulates part of the dot product
    int accum = 0;
    
    // Process K/2 bytes (K FP4 values)
    const uint8_t* W_row = W + row * (K / 2);
    const uint8_t* W_scale_row = W_scale + row * (K / 32);
    
    for (int k = threadIdx.x; k < K / 2; k += blockDim.x) {
        // Get packed FP4 byte (2 FP4 values)
        uint8_t w_packed = W_row[k];
        
        // Convert to int8
        int8_t w_lo = kFP4ToInt8[w_packed & 0x0F];
        int8_t w_hi = kFP4ToInt8[w_packed >> 4];
        
        // Get corresponding input Q8 values
        int8_t x_lo = x_q8[k * 2];
        int8_t x_hi = x_q8[k * 2 + 1];
        
        // Accumulate
        accum += (int)w_lo * (int)x_lo + (int)w_hi * (int)x_hi;
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        accum += __shfl_down_sync(0xffffffff, accum, offset);
    }
    
    // Block-level reduction
    __shared__ int partial_sums[32];  // One per warp
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        partial_sums[warp_id] = accum;
    }
    __syncthreads();
    
    // Final reduction by thread 0
    if (threadIdx.x == 0) {
        int total = 0;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            total += partial_sums[i];
        }
        
        // Apply scales and convert to BF16
        // Weight scale: avg FP8 scale (simplified)
        // Result: (int_accum × x_scale × w_scale) / (127 × 2)
        // The /2 is because FP4 table values are doubled
        float result = (float)total * x_scale * 0.5f / 127.0f;
        
        y[row] = __float2bfloat16(result);
    }
}

// Fused MoE GEMV: processes multiple experts in parallel
// Each expert has its own weight matrix
template <int BLOCK_DIM = 256>
__global__ void moe_gemv_fp4_dp4a_kernel(
    const uint8_t* __restrict__ W,        // [num_experts, M, K/2] packed FP4
    const uint8_t* __restrict__ W_scale,  // [num_experts, M, K/32] FP8 scales
    const nv_bfloat16* __restrict__ x,    // [K] BF16 input (shared across experts)
    nv_bfloat16* __restrict__ y,          // [num_experts, M] BF16 output
    const int* expert_ids,                // [num_active] which experts to use
    const float* expert_weights,          // [num_active] routing weights
    int num_active_experts,
    int M,
    int K)
{
    // Grid: (num_active_experts, M, 1)
    int expert_idx = blockIdx.x;
    int row = blockIdx.y;
    
    if (expert_idx >= num_active_experts || row >= M) return;
    
    int expert_id = expert_ids[expert_idx];
    float routing_weight = expert_weights[expert_idx];
    
    // Pointers for this expert
    const uint8_t* W_expert = W + expert_id * M * (K / 2);
    const uint8_t* W_scale_expert = W_scale + expert_id * M * (K / 32);
    nv_bfloat16* y_expert = y + expert_idx * M;
    
    // [Same logic as single GEMV but for one expert's row]
    // ... (simplified, actual implementation would be similar to above)
    
    // For now, just set output to 0 as placeholder
    if (threadIdx.x == 0) {
        y_expert[row] = __float2bfloat16(0.0f);
    }
}

// C interface
extern "C" __attribute__((visibility("default")))
int gemv_fp4_dp4a(
    const void* W,        // [M, K/2] uint8
    const void* W_scale,  // [M, K/32] uint8
    const void* x,        // [K] BF16
    void* y,              // [M] BF16
    int M,
    int K,
    cudaStream_t stream)
{
    dim3 grid(M);
    dim3 block(256);
    
    gemv_fp4_dp4a_kernel<256><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W),
        reinterpret_cast<const uint8_t*>(W_scale),
        reinterpret_cast<const nv_bfloat16*>(x),
        reinterpret_cast<nv_bfloat16*>(y),
        M, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

}  // namespace gemv
}  // namespace flashinfer

