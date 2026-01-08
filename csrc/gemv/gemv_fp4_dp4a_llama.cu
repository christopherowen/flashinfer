/*
 * FP4 GEMV Kernel using DP4A (following llama.cpp approach)
 * 
 * This kernel implements MXFP4 matrix-vector multiplication using INT8 DP4A
 * instructions instead of Tensor Cores. This is more efficient for small
 * batch sizes (M=1) where Tensor Core tile overhead dominates.
 *
 * Key insight from llama.cpp:
 * - Convert FP4 weights to INT8 via lookup table (not floating point conversion)
 * - Quantize BF16 activations to INT8 with per-block scale
 * - Use DP4A for 4x int8 dot products per instruction
 * - Apply scale factors at the end
 *
 * Reference: llama.cpp ggml/src/ggml-cuda/vecdotq.cuh
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace flashinfer {
namespace gemv {

//=============================================================================
// Constants and Lookup Tables
//=============================================================================

// E2M1 values doubled (for int8 range), matching llama.cpp's kvalues_mxfp4
// These are the 16 possible FP4 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
// Doubled: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
__constant__ int8_t kvalues_fp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,      // Positive values (index 0-7)
    0, -1, -2, -3, -4, -6, -8, -12  // Negative values (index 8-15)
};

// Block size for MXFP4 quantization
constexpr int QK_MXFP4 = 32;

// Block size for Q8_1 activation quantization
constexpr int QK8_1 = 32;

//=============================================================================
// Utility Functions (from llama.cpp)
//=============================================================================

// Load 4 bytes from unaligned address
__device__ __forceinline__ int get_int_b1(const void* x, const int& i32) {
    const uint8_t* x8 = (const uint8_t*)x;
    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;
    return x32;
}

// Convert 8 FP4 indices (packed in 32 bits) to 8 int8 values using lookup table
// Uses __byte_perm for efficient byte selection on CUDA
// q4: 8 x 4-bit indices packed in 32 bits
// Returns int2 with even indices in .x and odd indices in .y
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*)table;
    
    // __byte_perm selects bytes based on the lower 16 bits in its third argument
    // Do 2 iterations over the 32 bits in q4 with 0 and 16 shift
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
    
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    
    // Reorder: even indices in .x, odd indices in .y
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), 
                     __byte_perm(tmp[0], tmp[1], 0x7531));
}

// DP4A: 4x int8 dot product with int32 accumulation
// Computes: d = sum(a[i] * b[i] for i in 0..3) + c
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    int result;
#if __CUDA_ARCH__ >= 610
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" 
                 : "=r"(result) 
                 : "r"(a), "r"(b), "r"(c));
#else
    // Fallback for older architectures
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    result = c;
    for (int k = 0; k < 4; ++k) {
        result += int(a8[k]) * int(b8[k]);
    }
#endif
    return result;
}

// Convert E8M0 scale factor to float
// E8M0: 8-bit exponent only, no mantissa, bias = 127
__device__ __forceinline__ float e8m0_to_fp32(uint8_t e) {
    // E8M0: value = 2^(e - 127)
    uint32_t bits = (uint32_t(e) << 23);  // Put exponent in float32 format
    return __uint_as_float(bits);
}

//=============================================================================
// MXFP4 Block Structure (matching llama.cpp's block_mxfp4)
//=============================================================================

// MXFP4 block: 1 E8M0 scale + 16 packed FP4 nibbles = 32 values
struct block_mxfp4 {
    uint8_t e;                    // E8M0 scale factor
    uint8_t qs[QK_MXFP4 / 2];     // 16 bytes = 32 packed 4-bit values
};

// Q8_1 block for quantized activations: 32 int8 values + scale + sum
struct block_q8_1 {
    half2 ds;   // .x = scale (d), .y = sum (s) for bias correction
    int8_t qs[QK8_1];  // 32 quantized values
};

//=============================================================================
// Core Vector Dot Product (matching llama.cpp's vec_dot_mxfp4_q8_1)
//=============================================================================

// Vector dot product: FP4 weights × Q8_1 activations
// This processes one block (32 elements) at a time
constexpr int VDR_MXFP4_Q8_1 = 2;  // Vector depth ratio

__device__ __forceinline__ float vec_dot_mxfp4_q8_1(
    const void* __restrict__ vbq,      // MXFP4 weights
    const block_q8_1* __restrict__ bq8_1,  // Q8_1 activations  
    const int& kbx,                     // Block index for weights
    const int& iqs                      // Sub-block index
) {
    const block_mxfp4* bq4 = (const block_mxfp4*)vbq + kbx;
    const int* q8 = (const int*)bq8_1->qs + iqs;
    
    int sumi = 0;
    #pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1; ++l) {
        // Load 4 bytes = 8 FP4 values
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        // Convert to int8 via lookup table
        const int2 v = get_int_from_table_16(aux_q4, kvalues_fp4);
        
        // DP4A: 4x int8 dot products
        sumi = dp4a(v.x, q8[l + 0], sumi);  // Even nibbles
        sumi = dp4a(v.y, q8[l + 4], sumi);  // Odd nibbles
    }
    
    // Apply scale factors:
    // - Weight scale: E8M0 value * 0.5 (because kvalues_fp4 is doubled)
    // - Activation scale: d from Q8_1
    const float d = e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * float(sumi);
}

//=============================================================================
// Quantize BF16 Activation to Q8_1 Format
//=============================================================================

// Quantize a block of 32 BF16 values to Q8_1 format
__device__ void quantize_bf16_to_q8_1(
    const nv_bfloat16* __restrict__ input,  // [32] BF16 values
    block_q8_1* __restrict__ output          // Output Q8_1 block
) {
    // Find max absolute value for scaling
    float amax = 0.0f;
    float vals[QK8_1];
    
    #pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        vals[i] = __bfloat162float(input[i]);
        amax = fmaxf(amax, fabsf(vals[i]));
    }
    
    // Compute scale: map max value to 127
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? 127.0f / amax : 0.0f;  // Inverse scale
    
    // Quantize and compute sum for bias correction
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        const float v = vals[i] * id;
        const int8_t q = (int8_t)roundf(v);
        output->qs[i] = q;
        sum += float(q);
    }
    
    // Store scale and sum
    output->ds = __halves2half2(__float2half(d), __float2half(sum * d));
}

//=============================================================================
// Main GEMV Kernel: Y = X @ W^T (M=1 optimized)
//=============================================================================

// Grid: (N / BLOCK_N, batch)
// Block: (BLOCK_N) threads
template <int BLOCK_N = 128>
__global__ void gemv_mxfp4_dp4a_kernel(
    const uint8_t* __restrict__ weights,     // [N, K/2] packed FP4 (interpreted as block_mxfp4)
    const uint8_t* __restrict__ weight_scales, // [N, K/32] E8M0 scales (embedded in weights)
    const nv_bfloat16* __restrict__ input,   // [K] BF16 activation
    nv_bfloat16* __restrict__ output,        // [N] BF16 output
    int N,                                   // Output dimension
    int K                                    // Reduction dimension
) {
    // Dynamic shared memory for quantized activations
    extern __shared__ block_q8_1 q8_blocks[];
    
    const int n_block = blockIdx.x;
    const int n_start = n_block * BLOCK_N;
    const int tid = threadIdx.x;
    
    // Number of blocks in K dimension
    const int n_k_blocks = K / QK_MXFP4;
    const int n_q8_blocks = K / QK8_1;
    
    // Cooperative quantization of activations (once per block)
    // All threads participate
    for (int i = tid; i < n_q8_blocks; i += BLOCK_N) {
        quantize_bf16_to_q8_1(input + i * QK8_1, &q8_blocks[i]);
    }
    __syncthreads();
    
    // Each thread handles one output element
    const int n = n_start + tid;
    if (n >= N) return;
    
    // Pointers for this output row's weights
    const block_mxfp4* w_blocks = (const block_mxfp4*)(weights) + n * n_k_blocks;
    
    // Accumulate dot product
    float sum = 0.0f;
    
    // Process blocks of 32 elements
    for (int kb = 0; kb < n_k_blocks; ++kb) {
        // Each block_mxfp4 corresponds to one block_q8_1
        const block_q8_1* q8 = &q8_blocks[kb];
        
        // Sum over sub-blocks within the 32-element block
        for (int iqs = 0; iqs < QK_MXFP4 / 8; iqs += VDR_MXFP4_Q8_1) {
            sum += vec_dot_mxfp4_q8_1(w_blocks, q8, kb, iqs);
        }
    }
    
    // Write output
    output[n] = __float2bfloat16(sum);
}

//=============================================================================
// MoE-Optimized GEMV: Process Multiple Experts
//=============================================================================

// For MoE, we need to process 8 experts per token
// This kernel batches all experts together for better efficiency
template <int BLOCK_N = 128>
__global__ void gemv_mxfp4_moe_dp4a_kernel(
    const uint8_t* __restrict__* expert_weights,  // [num_experts] pointers to weight matrices
    const nv_bfloat16* __restrict__ input,        // [K] BF16 activation (shared across experts)
    const int* __restrict__ expert_ids,            // [topk] expert indices to use
    const float* __restrict__ expert_weights_scale, // [topk] routing weights
    nv_bfloat16* __restrict__ output,              // [N] BF16 output (accumulated)
    int num_experts,                               // Total number of experts
    int topk,                                      // Number of experts to use
    int N,                                         // Output dimension per expert
    int K                                          // Hidden dimension
) {
    const int n_block = blockIdx.x;
    const int expert_idx = blockIdx.y;  // Which of the topk experts
    
    const int n_start = n_block * BLOCK_N;
    const int tid = threadIdx.x;
    const int n = n_start + tid;
    
    if (n >= N || expert_idx >= topk) return;
    
    const int expert_id = expert_ids[expert_idx];
    const float routing_weight = expert_weights_scale[expert_idx];
    
    // Get this expert's weights
    const int n_k_blocks = K / QK_MXFP4;
    const block_mxfp4* w_blocks = (const block_mxfp4*)(expert_weights[expert_id]) + n * n_k_blocks;
    
    // Shared memory for quantized activations (shared across experts)
    extern __shared__ block_q8_1 q8_blocks[];
    
    // Quantize activations cooperatively
    const int n_q8_blocks = K / QK8_1;
    if (expert_idx == 0) {  // Only first expert block quantizes
        for (int i = tid; i < n_q8_blocks; i += BLOCK_N) {
            quantize_bf16_to_q8_1(input + i * QK8_1, &q8_blocks[i]);
        }
    }
    __syncthreads();
    
    // Compute dot product
    float sum = 0.0f;
    for (int kb = 0; kb < n_k_blocks; ++kb) {
        const block_q8_1* q8 = &q8_blocks[kb];
        for (int iqs = 0; iqs < QK_MXFP4 / 8; iqs += VDR_MXFP4_Q8_1) {
            sum += vec_dot_mxfp4_q8_1(w_blocks, q8, kb, iqs);
        }
    }
    
    // Apply routing weight and accumulate to output
    atomicAdd(reinterpret_cast<float*>(output + n), 
              __bfloat162float(output[n]) + sum * routing_weight);
}

//=============================================================================
// Python/C++ Interface
//=============================================================================

extern "C" {

// Simple GEMV: Y = X @ W^T
// weights: [N, K/2] packed FP4 with embedded scales
// input: [K] BF16
// output: [N] BF16
void gemv_fp4_dp4a(
    const void* weights,
    const void* weight_scales,  // Not used, scales embedded in weights
    const void* input,
    void* output,
    int N,
    int K,
    cudaStream_t stream
) {
    constexpr int BLOCK_N = 128;
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_N);
    
    // Dynamic shared memory: K/32 blocks of block_q8_1 (36 bytes each)
    const int n_q8_blocks = K / QK8_1;
    const size_t smem_size = n_q8_blocks * sizeof(block_q8_1);
    
    gemv_mxfp4_dp4a_kernel<BLOCK_N><<<grid, block, smem_size, stream>>>(
        (const uint8_t*)weights,
        (const uint8_t*)weight_scales,
        (const nv_bfloat16*)input,
        (nv_bfloat16*)output,
        N,
        K
    );
}

}  // extern "C"

}  // namespace gemv
}  // namespace flashinfer

