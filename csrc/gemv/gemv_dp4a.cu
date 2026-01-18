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

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Simple GEMV kernel: y = W @ x
// W: [N, K] FP4 packed as [N, K/2] uint8 (N = output dim, K = input dim)
// x: [K] BF16
// y: [N] BF16
// W_scale: [N, K/32] FP8 block scales (32 elements per block)
template <int BLOCK_DIM = 256>
__global__ void gemv_fp4_dp4a_kernel(
    const uint8_t* __restrict__ W,        // [N, K/2] packed FP4 weights
    const uint8_t* __restrict__ W_scale,  // [N, K/32] FP8 E8M0 scales
    const nv_bfloat16* __restrict__ x,    // [K] BF16 input
    nv_bfloat16* __restrict__ y,          // [N] BF16 output
    int N,
    int K)
{
    // Each thread block handles one output row
    int row = blockIdx.x;
    if (row >= N) return;
    
    constexpr int WARP_SIZE = 32;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = BLOCK_DIM / WARP_SIZE;
    
    // Shared memory for input vector quantization and reductions
    __shared__ int8_t x_q8[4096];  // Assume K <= 4096
    __shared__ float warp_max[8];  // Max per warp (up to 8 warps)
    __shared__ float x_scale;
    
    // =========== PARALLEL QUANTIZATION ===========
    // Step 1: Each thread finds local max for its elements
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = fabsf(__bfloat162float(x[i]));
        local_max = fmaxf(local_max, val);
    }
    
    // Step 2: Warp-level reduction
    float warp_max_val = warp_reduce_max(local_max);
    
    // Step 3: Store warp max to shared memory
    if (lane_id == 0) {
        warp_max[warp_id] = warp_max_val;
    }
    __syncthreads();
    
    // Step 4: Final reduction by first warp
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_max[lane_id] : 0.0f;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            x_scale = val / 127.0f;
        }
    }
    __syncthreads();
    
    // Step 5: Parallel quantization
    float inv_scale = (x_scale > 0.0f) ? 1.0f / x_scale : 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        int q = __float2int_rn(val * inv_scale);
        x_q8[i] = (int8_t)max(-127, min(127, q));
    }
    __syncthreads();
    
    // =========== MAIN DOT PRODUCT ===========
    // Each thread accumulates part of the dot product
    // Process in blocks of 32 elements (one scale per block)
    
    const uint8_t* W_row = W + row * (K / 2);
    const uint8_t* W_scale_row = W_scale + row * (K / 32);
    
    float accum = 0.0f;
    const int n_blocks = K / 32;
    
    // Each thread handles multiple K-blocks
    for (int block_idx = threadIdx.x; block_idx < n_blocks; block_idx += blockDim.x) {
        // Get weight scale for this block (E8M0 format)
        uint8_t w_scale_e8m0 = W_scale_row[block_idx];
        // E8M0 to float: 2^(e - 127) where e is the 8-bit value
        float w_scale = (w_scale_e8m0 == 0) ? 0.0f : exp2f((float)w_scale_e8m0 - 127.0f);
        
        // Process 32 elements in this block (16 packed bytes)
        int block_accum = 0;
        const uint8_t* W_block = W_row + block_idx * 16;  // 16 bytes = 32 FP4 values
        const int8_t* x_block = x_q8 + block_idx * 32;
        
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            // Load 4 packed bytes (8 FP4 values)
            uint32_t w_packed4 = *reinterpret_cast<const uint32_t*>(W_block + i);
            
            // Unpack and convert to int8 using lookup table
            int8_t w0 = kFP4ToInt8[(w_packed4 >>  0) & 0x0F];
            int8_t w1 = kFP4ToInt8[(w_packed4 >>  4) & 0x0F];
            int8_t w2 = kFP4ToInt8[(w_packed4 >>  8) & 0x0F];
            int8_t w3 = kFP4ToInt8[(w_packed4 >> 12) & 0x0F];
            int8_t w4 = kFP4ToInt8[(w_packed4 >> 16) & 0x0F];
            int8_t w5 = kFP4ToInt8[(w_packed4 >> 20) & 0x0F];
            int8_t w6 = kFP4ToInt8[(w_packed4 >> 24) & 0x0F];
            int8_t w7 = kFP4ToInt8[(w_packed4 >> 28) & 0x0F];
            
            // Pack weights for DP4A: 4 int8 values in one int32
            int w_dp4a_0 = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) | 
                           ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
            int w_dp4a_1 = (int(w4) & 0xFF) | ((int(w5) & 0xFF) << 8) | 
                           ((int(w6) & 0xFF) << 16) | ((int(w7) & 0xFF) << 24);
            
            // Load 8 int8 activation values (note: FP4 is interleaved as lo/hi nibbles)
            // Byte i contains values for positions i*2 and i*2+1
            int x_dp4a_0 = *reinterpret_cast<const int*>(x_block + i * 2);
            int x_dp4a_1 = *reinterpret_cast<const int*>(x_block + i * 2 + 4);
            
            // DP4A: accumulate 4 int8 products at a time
            block_accum = __dp4a(w_dp4a_0, x_dp4a_0, block_accum);
            block_accum = __dp4a(w_dp4a_1, x_dp4a_1, block_accum);
        }
        
        // Apply weight scale for this block
        // The FP4 table values are doubled, so divide by 2
        accum += (float)block_accum * w_scale * 0.5f;
    }
    
    // =========== REDUCTION ===========
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        accum += __shfl_down_sync(0xffffffff, accum, offset);
    }
    
    // Block-level reduction
    __shared__ float partial_sums[8];  // One per warp (max 8 warps)
    
    if (lane_id == 0) {
        partial_sums[warp_id] = accum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? partial_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (lane_id == 0) {
            // Apply activation scale
            float result = val * x_scale;
            y[row] = __float2bfloat16(result);
        }
    }
}

// Batched GEMV: y[m,:] = W @ x[m,:] for m in 0..M-1
// Handles M > 1 by processing one (m, n) pair per block
template <int BLOCK_DIM = 256>
__global__ void gemv_fp4_dp4a_batched_kernel(
    const uint8_t* __restrict__ W,        // [N, K/2] packed FP4 weights
    const uint8_t* __restrict__ W_scale,  // [N, K/32] FP8 E8M0 scales
    const nv_bfloat16* __restrict__ x,    // [M, K] BF16 input (M rows)
    nv_bfloat16* __restrict__ y,          // [M, N] BF16 output
    int M,  // Batch size
    int N,  // Output dimension  
    int K)  // Input dimension
{
    // Grid: (N, M, 1) - one block per (output_row, batch_idx)
    const int n = blockIdx.x;  // Output row
    const int m = blockIdx.y;  // Batch index
    
    if (n >= N || m >= M) return;
    
    constexpr int WARP_SIZE = 32;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = BLOCK_DIM / WARP_SIZE;
    
    // Input for this batch element
    const nv_bfloat16* x_row = x + m * K;
    
    // Shared memory for quantization
    __shared__ int8_t x_q8[4096];
    __shared__ float warp_max[8];
    __shared__ float x_scale;
    
    // Parallel quantization (same as single GEMV)
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = fabsf(__bfloat162float(x_row[i]));
        local_max = fmaxf(local_max, val);
    }
    
    float warp_max_val = warp_reduce_max(local_max);
    if (lane_id == 0) warp_max[warp_id] = warp_max_val;
    __syncthreads();
    
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_max[lane_id] : 0.0f;
        val = warp_reduce_max(val);
        if (lane_id == 0) x_scale = val / 127.0f;
    }
    __syncthreads();
    
    float inv_scale = (x_scale > 0.0f) ? 1.0f / x_scale : 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = __bfloat162float(x_row[i]);
        int q = __float2int_rn(val * inv_scale);
        x_q8[i] = (int8_t)max(-127, min(127, q));
    }
    __syncthreads();
    
    // Dot product (same as single GEMV)
    const uint8_t* W_row = W + n * (K / 2);
    const uint8_t* W_scale_row = W_scale + n * (K / 32);
    
    float accum = 0.0f;
    const int n_blocks = K / 32;
    
    for (int block_idx = threadIdx.x; block_idx < n_blocks; block_idx += blockDim.x) {
        uint8_t w_scale_e8m0 = W_scale_row[block_idx];
        float w_scale = (w_scale_e8m0 == 0) ? 0.0f : exp2f((float)w_scale_e8m0 - 127.0f);
        
        int block_accum = 0;
        const uint8_t* W_block = W_row + block_idx * 16;
        const int8_t* x_block = x_q8 + block_idx * 32;
        
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            uint32_t w_packed4 = *reinterpret_cast<const uint32_t*>(W_block + i);
            
            int8_t w0 = kFP4ToInt8[(w_packed4 >>  0) & 0x0F];
            int8_t w1 = kFP4ToInt8[(w_packed4 >>  4) & 0x0F];
            int8_t w2 = kFP4ToInt8[(w_packed4 >>  8) & 0x0F];
            int8_t w3 = kFP4ToInt8[(w_packed4 >> 12) & 0x0F];
            int8_t w4 = kFP4ToInt8[(w_packed4 >> 16) & 0x0F];
            int8_t w5 = kFP4ToInt8[(w_packed4 >> 20) & 0x0F];
            int8_t w6 = kFP4ToInt8[(w_packed4 >> 24) & 0x0F];
            int8_t w7 = kFP4ToInt8[(w_packed4 >> 28) & 0x0F];
            
            int w_dp4a_0 = (int(w0) & 0xFF) | ((int(w1) & 0xFF) << 8) | 
                           ((int(w2) & 0xFF) << 16) | ((int(w3) & 0xFF) << 24);
            int w_dp4a_1 = (int(w4) & 0xFF) | ((int(w5) & 0xFF) << 8) | 
                           ((int(w6) & 0xFF) << 16) | ((int(w7) & 0xFF) << 24);
            
            int x_dp4a_0 = *reinterpret_cast<const int*>(x_block + i * 2);
            int x_dp4a_1 = *reinterpret_cast<const int*>(x_block + i * 2 + 4);
            
            block_accum = __dp4a(w_dp4a_0, x_dp4a_0, block_accum);
            block_accum = __dp4a(w_dp4a_1, x_dp4a_1, block_accum);
        }
        
        accum += (float)block_accum * w_scale * 0.5f;
    }
    
    // Reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        accum += __shfl_down_sync(0xffffffff, accum, offset);
    }
    
    __shared__ float partial_sums[8];
    if (lane_id == 0) partial_sums[warp_id] = accum;
    __syncthreads();
    
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? partial_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (lane_id == 0) {
            float result = val * x_scale;
            y[m * N + n] = __float2bfloat16(result);
        }
    }
}

// C interface
extern "C" __attribute__((visibility("default")))
int gemv_fp4_dp4a(
    const void* W,        // [N, K/2] uint8 packed FP4 weights
    const void* W_scale,  // [N, K/32] uint8 E8M0 scales  
    const void* x,        // [K] BF16 input
    void* y,              // [N] BF16 output
    int N,                // Output dimension
    int K,                // Input dimension (must be divisible by 32)
    cudaStream_t stream)
{
    // Validate inputs
    if (K % 32 != 0) {
        printf("GEMV error: K=%d must be divisible by 32\n", K);
        return -1;
    }
    if (K > 4096) {
        printf("GEMV error: K=%d exceeds max supported (4096)\n", K);
        return -1;
    }
    
    dim3 grid(N);
    dim3 block(256);
    
    gemv_fp4_dp4a_kernel<256><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W),
        reinterpret_cast<const uint8_t*>(W_scale),
        reinterpret_cast<const nv_bfloat16*>(x),
        reinterpret_cast<nv_bfloat16*>(y),
        N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

// Batched C interface: y[m,:] = W @ x[m,:] for m in 0..M-1
extern "C" __attribute__((visibility("default")))
int gemv_fp4_dp4a_batched(
    const void* W,        // [N, K/2] uint8 packed FP4 weights
    const void* W_scale,  // [N, K/32] uint8 E8M0 scales  
    const void* x,        // [M, K] BF16 input (batched)
    void* y,              // [M, N] BF16 output
    int M,                // Batch size
    int N,                // Output dimension
    int K,                // Input dimension (must be divisible by 32)
    cudaStream_t stream)
{
    if (K % 32 != 0) {
        printf("GEMV error: K=%d must be divisible by 32\n", K);
        return -1;
    }
    if (K > 4096) {
        printf("GEMV error: K=%d exceeds max supported (4096)\n", K);
        return -1;
    }
    
    dim3 grid(N, M);
    dim3 block(256);
    
    gemv_fp4_dp4a_batched_kernel<256><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W),
        reinterpret_cast<const uint8_t*>(W_scale),
        reinterpret_cast<const nv_bfloat16*>(x),
        reinterpret_cast<nv_bfloat16*>(y),
        M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

}  // namespace gemv
}  // namespace flashinfer


