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
 * This version is adapted for vLLM's MXFP4 format:
 * - weights: [N, K/2] uint8 (packed FP4, 2 values per byte)
 * - weight_scales: [N, K/32] uint8 (E8M0 block scales)
 * - activations: [M, K] bfloat16 (quantized to INT8 inside kernel)
 *
 * Reference: llama.cpp ggml/src/ggml-cuda/vecdotq.cuh
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// TVM FFI bindings
#include "../tvm_ffi_utils.h"

using tvm::ffi::TensorView;

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
// Quantize BF16 Activation to Q8_1 Format (Interleaved for DP4A)
//=============================================================================

// Quantize a block of 32 BF16 values to Q8_1 format with INTERLEAVED layout
// to match the output of get_int_from_table_16.
//
// The interleaving pattern:
// - qs[0..15]  = even indices: act[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
// - qs[16..31] = odd indices:  act[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
//
// This matches get_int_from_table_16 which returns:
// - v.x = table values for even nibbles (low nibbles of each byte)
// - v.y = table values for odd nibbles (high nibbles of each byte)
//
// With this layout, dp4a(v.x, q8[l], sumi) and dp4a(v.y, q8[l+4], sumi)
// correctly pairs FP4 weights with their corresponding activations.
__device__ void quantize_bf16_to_q8_1_interleaved(
    const nv_bfloat16* __restrict__ input,  // [32] BF16 values in sequential order
    block_q8_1* __restrict__ output          // Output Q8_1 block in interleaved order
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
    
    // Quantize with interleaved storage
    float sum = 0.0f;
    
    // First half: even indices (0, 2, 4, ..., 30)
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int src_idx = i * 2;  // 0, 2, 4, ..., 30
        const float v = vals[src_idx] * id;
        const int8_t q = (int8_t)roundf(v);
        output->qs[i] = q;
        sum += float(q);
    }
    
    // Second half: odd indices (1, 3, 5, ..., 31)
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int src_idx = i * 2 + 1;  // 1, 3, 5, ..., 31
        const float v = vals[src_idx] * id;
        const int8_t q = (int8_t)roundf(v);
        output->qs[16 + i] = q;
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
        quantize_bf16_to_q8_1_interleaved(input + i * QK8_1, &q8_blocks[i]);
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
            quantize_bf16_to_q8_1_interleaved(input + i * QK8_1, &q8_blocks[i]);
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
// VLLM-Compatible GEMV Kernel: Separate weight and scale tensors
// Optimized version matching llama.cpp's approach with DP4A vectorization
//=============================================================================

// Warp size for CUDA
constexpr int WARP_SIZE = 32;

// Warp reduction sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Vector dot product for vLLM format using vectorized DP4A
// Matches llama.cpp's approach exactly for maximum performance
//
// With interleaved Q8 layout:
// - qs[0..15] = even activations (indices 0,2,4,...,30)
// - qs[16..31] = odd activations (indices 1,3,5,...,31)
//
// And get_int_from_table_16 output:
// - v.x = int8 values for even nibbles (positions 0,2,4,6 in 4 bytes)
// - v.y = int8 values for odd nibbles (positions 1,3,5,7 in 4 bytes)
//
// The dp4a pairing works correctly:
// dp4a(v.x, q8[l]) pairs even weights with even activations
// dp4a(v.y, q8[l+4]) pairs odd weights with odd activations
//
// Parameters:
// - iqs: sub-block index (0 or 2), determines which 16 of 32 elements to process
// - Each call processes 16 elements via 2 iterations of 8 elements each
constexpr int VDR_MXFP4_Q8_1_MMVQ = 2;  // Vector depth ratio: 2 iterations per call

__device__ __forceinline__ float vec_dot_vllm_mxfp4_q8_1_dp4a(
    const uint8_t* __restrict__ weights,    // [K/2] packed FP4 for this row
    const uint8_t* __restrict__ scales,     // [K/32] E8M0 scales for this row
    const block_q8_1* __restrict__ q8,      // Q8_1 activation block (INTERLEAVED)
    const int kb,                            // Block index (0 to K/32-1)
    const int iqs                            // Sub-block index (0 or 2)
) {
    // Get scale for this block
    const float weight_scale = e8m0_to_fp32(scales[kb]) * 0.5f;
    const float act_scale = __low2float(q8->ds);
    
    // Get weights for this block: 16 bytes = 32 x 4-bit values
    const uint8_t* w_ptr = weights + kb * 16;
    
    // Get activation ints - with interleaved layout:
    // q8_qs[0..3] = even activations 0-15 as 4 int32
    // q8_qs[4..7] = odd activations 0-15 as 4 int32
    const int* q8_qs = (const int*)q8->qs + iqs;
    
    int sumi = 0;
    
    #pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        // Load 4 bytes = 8 FP4 values
        const int aux_q4 = get_int_b1(w_ptr, iqs + l);
        
        // Convert to int8 via lookup table (vectorized, 8 values at once)
        // v.x = int8 for even nibbles, v.y = int8 for odd nibbles
        const int2 v = get_int_from_table_16(aux_q4, kvalues_fp4);
        
        // DP4A: 4x int8 dot products each
        // Even nibbles × even activations
        sumi = dp4a(v.x, q8_qs[l + 0], sumi);
        // Odd nibbles × odd activations  
        sumi = dp4a(v.y, q8_qs[l + 4], sumi);
    }
    
    return weight_scale * act_scale * float(sumi);
}

// Helper to load int4 from potentially unaligned address
__device__ __forceinline__ int4 load_int4_unaligned(const uint8_t* ptr) {
    int4 result;
    result.x = get_int_b1(ptr, 0);
    result.y = get_int_b1(ptr, 1);
    result.z = get_int_b1(ptr, 2);
    result.w = get_int_b1(ptr, 3);
    return result;
}

// Vectorized version processing entire 32-element block in one call
// Unrolled for better instruction-level parallelism
__device__ __forceinline__ float vec_dot_vllm_mxfp4_q8_1_dp4a_vec(
    const uint8_t* __restrict__ weights,    // [K/2] packed FP4 for this row
    const uint8_t* __restrict__ scales,     // [K/32] E8M0 scales for this row
    const block_q8_1* __restrict__ q8,      // Q8_1 activation block (INTERLEAVED)
    const int kb                             // Block index (0 to K/32-1)
) {
    // Get scale for this block
    const float weight_scale = e8m0_to_fp32(scales[kb]) * 0.5f;
    const float act_scale = __low2float(q8->ds);
    
    // Load 16 bytes of weights (may be unaligned)
    const uint8_t* w_ptr = weights + kb * 16;
    const int4 w_vec = load_int4_unaligned(w_ptr);
    
    // Load activations (block_q8_1.qs may not be aligned)
    const int* q8_qs = reinterpret_cast<const int*>(q8->qs);
    
    int sumi = 0;
    
    // Process all 4 int32s (16 bytes = 32 FP4 values)
    // Fully unrolled for better ILP
    
    // Bytes 0-3: FP4 indices 0-7
    int2 v0 = get_int_from_table_16(w_vec.x, kvalues_fp4);
    sumi = dp4a(v0.x, q8_qs[0], sumi);  // even: indices 0,2,4,6
    sumi = dp4a(v0.y, q8_qs[4], sumi);  // odd: indices 1,3,5,7
    
    // Bytes 4-7: FP4 indices 8-15
    int2 v1 = get_int_from_table_16(w_vec.y, kvalues_fp4);
    sumi = dp4a(v1.x, q8_qs[1], sumi);  // even: indices 8,10,12,14
    sumi = dp4a(v1.y, q8_qs[5], sumi);  // odd: indices 9,11,13,15
    
    // Bytes 8-11: FP4 indices 16-23
    int2 v2 = get_int_from_table_16(w_vec.z, kvalues_fp4);
    sumi = dp4a(v2.x, q8_qs[2], sumi);  // even: indices 16,18,20,22
    sumi = dp4a(v2.y, q8_qs[6], sumi);  // odd: indices 17,19,21,23
    
    // Bytes 12-15: FP4 indices 24-31
    int2 v3 = get_int_from_table_16(w_vec.w, kvalues_fp4);
    sumi = dp4a(v3.x, q8_qs[3], sumi);  // even: indices 24,26,28,30
    sumi = dp4a(v3.y, q8_qs[7], sumi);  // odd: indices 25,27,29,31
    
    return weight_scale * act_scale * float(sumi);
}

// Optimized GEMV kernel with warp-level parallelism for K dimension
// Processes multiple output rows per block to amortize activation quantization cost
// 
// Key optimization: ROWS_PER_BLOCK=8 means we compute 8 output elements
// while only quantizing activations once per block.
//
// Grid: (ceil(N/ROWS_PER_BLOCK), M)
// Block: (WARP_SIZE, NWARPS) = (32, NWARPS) threads
template <int NWARPS = 4, int ROWS_PER_BLOCK = 8>
__global__ void gemv_vllm_mxfp4_dp4a_kernel(
    const uint8_t* __restrict__ weights,      // [N, K/2] packed FP4
    const uint8_t* __restrict__ weight_scales, // [N, K/32] E8M0 scales
    const nv_bfloat16* __restrict__ input,    // [M, K] BF16 activation
    nv_bfloat16* __restrict__ output,         // [M, N] BF16 output
    int M,                                     // Batch size (number of rows)
    int N,                                     // Output dimension
    int K                                      // Hidden dimension
) {
    // Shared memory for quantized activations
    extern __shared__ block_q8_1 q8_blocks[];
    
    const int m = blockIdx.y;  // Which input row (batch dimension)
    const int n_start = blockIdx.x * ROWS_PER_BLOCK;  // Starting output row
    
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;  // Linear thread ID
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    
    // Number of blocks in K dimension
    const int n_k_blocks = K / QK_MXFP4;
    const int n_q8_blocks = K / QK8_1;
    
    // Pointer to this input row
    const nv_bfloat16* row_input = input + m * K;
    
    // Threads per block
    constexpr int THREADS_PER_BLOCK = NWARPS * WARP_SIZE;
    
    // Cooperative quantization of activations (all threads participate)
    // Using INTERLEAVED layout to match get_int_from_table_16 output
    for (int i = tid; i < n_q8_blocks; i += THREADS_PER_BLOCK) {
        quantize_bf16_to_q8_1_interleaved(row_input + i * QK8_1, &q8_blocks[i]);
    }
    __syncthreads();
    
    // Work distribution matching llama.cpp:
    // - VDR = 2: each vec_dot call processes 16 elements (half a block)
    // - qi = 4: number of int32 groups in Q8 (32 int8 / 8 per group = 4)
    // - Each thread handles a different (kb, iqs) pair
    constexpr int qi = 4;
    constexpr int vdr = VDR_MXFP4_Q8_1_MMVQ;  // = 2
    constexpr int blocks_per_iter = vdr * THREADS_PER_BLOCK / qi;
    
    // Partial sums for each output row this block handles
    // With ROWS_PER_BLOCK=8, each thread tracks 8 partial sums
    float tmp[ROWS_PER_BLOCK] = {0.0f};
    
    // Determine which sub-block this thread handles
    const int iqs = vdr * (tid % (qi / vdr));  // 0 or 2
    
    // Iterate over K dimension in strides
    for (int kb = tid / (qi / vdr); kb < n_k_blocks; kb += blocks_per_iter) {
        // Process all ROWS_PER_BLOCK output rows with the same activation block
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            const int n = n_start + row;
            if (n < N) {
                const uint8_t* row_weights = weights + n * (K / 2);
                const uint8_t* row_scales = weight_scales + n * n_k_blocks;
                
                tmp[row] += vec_dot_vllm_mxfp4_q8_1_dp4a(
                    row_weights, row_scales, &q8_blocks[kb], kb, iqs);
            }
        }
    }
    
    // Cross-warp reduction via shared memory
    // Each thread has partial sums for ROWS_PER_BLOCK outputs
    __shared__ float tmp_shared[NWARPS > 1 ? NWARPS - 1 : 1][ROWS_PER_BLOCK][WARP_SIZE];
    
    // Non-zero warps write their partial sums to shared memory
    if (warp_id > 0) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            tmp_shared[warp_id - 1][row][lane_id] = tmp[row];
        }
    }
    __syncthreads();
    
    // Only warp 0 continues with final reduction
    if (warp_id > 0) return;
    
    // Warp 0: sum results from other warps (per-lane)
    #pragma unroll
    for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; ++w) {
            tmp[row] += tmp_shared[w][row][lane_id];
        }
    }
    
    // Now do warp reduction to sum across all 32 lanes
    #pragma unroll
    for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
        tmp[row] = warp_reduce_sum(tmp[row]);
    }
    
    // Lane 0 writes final results
    if (lane_id == 0) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            const int n = n_start + row;
            if (n < N) {
                output[m * N + n] = __float2bfloat16(tmp[row]);
            }
        }
    }
}

//=============================================================================
// Pre-Quantization Kernel (runs once, output reused by all GEMV blocks)
//=============================================================================

// Quantize BF16 activations to interleaved Q8_1 format in global memory
// Grid: (ceil(K/32 / THREADS_PER_BLOCK), M)
// Block: THREADS_PER_BLOCK threads
template <int THREADS_PER_BLOCK = 256>
__global__ void quantize_activations_kernel(
    const nv_bfloat16* __restrict__ input,    // [M, K] BF16 activations
    block_q8_1* __restrict__ output,           // [M, K/32] Q8_1 blocks
    int M,
    int K
) {
    const int m = blockIdx.y;
    const int n_q8_blocks = K / QK8_1;
    
    // Each thread quantizes one or more 32-element blocks
    for (int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x; 
         i < n_q8_blocks; 
         i += gridDim.x * THREADS_PER_BLOCK) {
        quantize_bf16_to_q8_1_interleaved(
            input + m * K + i * QK8_1,
            output + m * n_q8_blocks + i
        );
    }
}

//=============================================================================
// GEMV Kernel with Pre-Quantized Activations (no shared memory for activations)
//=============================================================================

// Prefetch helper using inline PTX for L2 prefetch
__device__ __forceinline__ void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

// This kernel reads pre-quantized activations from global memory (L2 cached)
// Uses VECTORIZED int4 loads (16 bytes) for better memory throughput.
// Each thread processes entire K-blocks (no iqs splitting).
//
// Grid: (ceil(N/ROWS_PER_BLOCK), M)
// Block: (WARP_SIZE, NWARPS) = (32, NWARPS) threads
template <int NWARPS = 4, int ROWS_PER_BLOCK = 8>
__global__ void gemv_vllm_mxfp4_dp4a_prequant_kernel(
    const uint8_t* __restrict__ weights,           // [N, K/2] packed FP4
    const uint8_t* __restrict__ weight_scales,     // [N, K/32] E8M0 scales
    const block_q8_1* __restrict__ q8_activations, // [M, K/32] pre-quantized
    nv_bfloat16* __restrict__ output,              // [M, N] BF16 output
    int M,
    int N,
    int K
) {
    const int m = blockIdx.y;
    const int n_start = blockIdx.x * ROWS_PER_BLOCK;
    
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    
    const int n_k_blocks = K / QK_MXFP4;
    
    // Pointer to pre-quantized activations for this input row
    const block_q8_1* q8_row = q8_activations + m * n_k_blocks;
    
    constexpr int THREADS_PER_BLOCK = NWARPS * WARP_SIZE;
    
    float tmp[ROWS_PER_BLOCK] = {0.0f};
    
    // Vectorized: each thread handles complete K-blocks
    // No iqs splitting - vec_dot_vec processes entire 32-element block
    for (int kb = tid; kb < n_k_blocks; kb += THREADS_PER_BLOCK) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            const int n = n_start + row;
            if (n < N) {
                const uint8_t* row_weights = weights + n * (K / 2);
                const uint8_t* row_scales = weight_scales + n * n_k_blocks;
                
                // Vectorized: 16-byte int4 load, processes entire block
                tmp[row] += vec_dot_vllm_mxfp4_q8_1_dp4a_vec(
                    row_weights, row_scales, &q8_row[kb], kb);
            }
        }
    }
    
    // Cross-warp reduction via shared memory
    __shared__ float tmp_shared[NWARPS > 1 ? NWARPS - 1 : 1][ROWS_PER_BLOCK][WARP_SIZE];
    
    if (warp_id > 0) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            tmp_shared[warp_id - 1][row][lane_id] = tmp[row];
        }
    }
    __syncthreads();
    
    if (warp_id > 0) return;
    
    #pragma unroll
    for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; ++w) {
            tmp[row] += tmp_shared[w][row][lane_id];
        }
    }
    
    #pragma unroll
    for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
        tmp[row] = warp_reduce_sum(tmp[row]);
    }
    
    if (lane_id == 0) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            const int n = n_start + row;
            if (n < N) {
                output[m * N + n] = __float2bfloat16(tmp[row]);
            }
        }
    }
}

//=============================================================================
// GEMV Kernel with Software Prefetching
//=============================================================================

// Version with warp-cooperative L2 prefetching
// Only lane 0 of each warp prefetches, reducing overhead by 32x
// Also prefetches activation blocks which are shared across all output rows
template <int NWARPS = 4, int ROWS_PER_BLOCK = 8>
__global__ void gemv_vllm_mxfp4_dp4a_prefetch_kernel(
    const uint8_t* __restrict__ weights,           // [N, K/2] packed FP4
    const uint8_t* __restrict__ weight_scales,     // [N, K/32] E8M0 scales
    const block_q8_1* __restrict__ q8_activations, // [M, K/32] pre-quantized
    nv_bfloat16* __restrict__ output,              // [M, N] BF16 output
    int M,
    int N,
    int K
) {
    const int m = blockIdx.y;
    const int n_start = blockIdx.x * ROWS_PER_BLOCK;
    
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    
    const int n_k_blocks = K / QK_MXFP4;
    const int stride_k = K / 2;
    
    const block_q8_1* q8_row = q8_activations + m * n_k_blocks;
    
    constexpr int THREADS_PER_BLOCK = NWARPS * WARP_SIZE;
    
    float tmp[ROWS_PER_BLOCK] = {0.0f};
    
    // Main loop with warp-cooperative prefetching
    for (int kb = tid; kb < n_k_blocks; kb += THREADS_PER_BLOCK) {
        const int kb_next = kb + THREADS_PER_BLOCK;
        
        // Warp-cooperative prefetch: only lane 0 prefetches for next iteration
        // This reduces prefetch overhead by 32x while still providing hints
        if (lane_id == 0 && kb_next < n_k_blocks) {
            // Prefetch activation block (shared across all rows, high value)
            prefetch_l2(&q8_row[kb_next]);
            
            // Prefetch first weight row only (let hardware handle spatial locality)
            if (n_start < N) {
                prefetch_l2(weights + n_start * stride_k + kb_next * 16);
            }
        }
        
        // Compute current K-block
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            const int n = n_start + row;
            if (n < N) {
                const uint8_t* row_weights = weights + n * stride_k;
                const uint8_t* row_scales = weight_scales + n * n_k_blocks;
                
                tmp[row] += vec_dot_vllm_mxfp4_q8_1_dp4a_vec(
                    row_weights, row_scales, &q8_row[kb], kb);
            }
        }
    }
    
    // Cross-warp reduction
    __shared__ float tmp_shared[NWARPS > 1 ? NWARPS - 1 : 1][ROWS_PER_BLOCK][WARP_SIZE];
    
    if (warp_id > 0) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            tmp_shared[warp_id - 1][row][lane_id] = tmp[row];
        }
    }
    __syncthreads();
    
    if (warp_id > 0) return;
    
    #pragma unroll
    for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; ++w) {
            tmp[row] += tmp_shared[w][row][lane_id];
        }
    }
    
    #pragma unroll
    for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
        tmp[row] = warp_reduce_sum(tmp[row]);
    }
    
    if (lane_id == 0) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            const int n = n_start + row;
            if (n < N) {
                output[m * N + n] = __float2bfloat16(tmp[row]);
            }
        }
    }
}

//=============================================================================
// Internal launcher for prefetch version
//=============================================================================

cudaError_t run_gemv_fp4_dp4a_prefetch(
    int M, int N, int K,
    const void* weights,
    const void* weight_scales,
    const void* q8_activations,
    void* output,
    cudaStream_t stream
) {
    constexpr int NWARPS = 4;
    constexpr int ROWS_PER_BLOCK = 8;
    
    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, M);
    dim3 block(WARP_SIZE, NWARPS);
    
    const size_t smem_reduce = (NWARPS - 1) * ROWS_PER_BLOCK * WARP_SIZE * sizeof(float);
    
    gemv_vllm_mxfp4_dp4a_prefetch_kernel<NWARPS, ROWS_PER_BLOCK><<<grid, block, smem_reduce, stream>>>(
        (const uint8_t*)weights,
        (const uint8_t*)weight_scales,
        (const block_q8_1*)q8_activations,
        (nv_bfloat16*)output,
        M, N, K
    );
    
    return cudaGetLastError();
}

//=============================================================================
// Fused QKV GEMV Kernel - Process 3 weight matrices in one launch
//=============================================================================

// Fused kernel for QKV projection: computes Q, K, V outputs simultaneously
// Benefits:
// - Single kernel launch (saves ~5μs per call)
// - Activations loaded once from L2, reused for all 3 matrices
// - Better GPU utilization
//
// Grid: (ceil(N/ROWS_PER_BLOCK), M, 3)  -- z-dim selects Q/K/V
// Block: (WARP_SIZE, NWARPS) threads
template <int NWARPS = 4, int ROWS_PER_BLOCK = 8>
__global__ void gemv_vllm_mxfp4_dp4a_fused_qkv_kernel(
    const uint8_t* __restrict__ weights_q,         // [N_q, K/2] packed FP4
    const uint8_t* __restrict__ weights_k,         // [N_k, K/2] packed FP4
    const uint8_t* __restrict__ weights_v,         // [N_v, K/2] packed FP4
    const uint8_t* __restrict__ scales_q,          // [N_q, K/32] E8M0
    const uint8_t* __restrict__ scales_k,          // [N_k, K/32] E8M0
    const uint8_t* __restrict__ scales_v,          // [N_v, K/32] E8M0
    const block_q8_1* __restrict__ q8_activations, // [M, K/32] pre-quantized
    nv_bfloat16* __restrict__ output_q,            // [M, N_q] 
    nv_bfloat16* __restrict__ output_k,            // [M, N_k]
    nv_bfloat16* __restrict__ output_v,            // [M, N_v]
    int M, int N_q, int N_k, int N_v, int K
) {
    const int m = blockIdx.y;
    const int n_start = blockIdx.x * ROWS_PER_BLOCK;
    const int qkv_idx = blockIdx.z;  // 0=Q, 1=K, 2=V
    
    // Select which weight/output to use based on z-index
    const uint8_t* weights;
    const uint8_t* scales;
    nv_bfloat16* output;
    int N;
    
    if (qkv_idx == 0) {
        weights = weights_q; scales = scales_q; output = output_q; N = N_q;
    } else if (qkv_idx == 1) {
        weights = weights_k; scales = scales_k; output = output_k; N = N_k;
    } else {
        weights = weights_v; scales = scales_v; output = output_v; N = N_v;
    }
    
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    
    const int n_k_blocks = K / QK_MXFP4;
    
    // Activations are shared across Q, K, V (L2 cached)
    const block_q8_1* q8_row = q8_activations + m * n_k_blocks;
    
    constexpr int THREADS_PER_BLOCK = NWARPS * WARP_SIZE;
    
    float tmp[ROWS_PER_BLOCK] = {0.0f};
    
    for (int kb = tid; kb < n_k_blocks; kb += THREADS_PER_BLOCK) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            const int n = n_start + row;
            if (n < N) {
                const uint8_t* row_weights = weights + n * (K / 2);
                const uint8_t* row_scales = scales + n * n_k_blocks;
                
                tmp[row] += vec_dot_vllm_mxfp4_q8_1_dp4a_vec(
                    row_weights, row_scales, &q8_row[kb], kb);
            }
        }
    }
    
    // Cross-warp reduction
    __shared__ float tmp_shared[NWARPS > 1 ? NWARPS - 1 : 1][ROWS_PER_BLOCK][WARP_SIZE];
    
    if (warp_id > 0) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            tmp_shared[warp_id - 1][row][lane_id] = tmp[row];
        }
    }
    __syncthreads();
    
    if (warp_id > 0) return;
    
    #pragma unroll
    for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
        #pragma unroll
        for (int w = 0; w < NWARPS - 1; ++w) {
            tmp[row] += tmp_shared[w][row][lane_id];
        }
    }
    
    #pragma unroll
    for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
        tmp[row] = warp_reduce_sum(tmp[row]);
    }
    
    if (lane_id == 0) {
        #pragma unroll
        for (int row = 0; row < ROWS_PER_BLOCK; ++row) {
            const int n = n_start + row;
            if (n < N) {
                output[m * N + n] = __float2bfloat16(tmp[row]);
            }
        }
    }
}

//=============================================================================
// Dynamic Configuration Selection
//=============================================================================

// Select optimal NWARPS based on K dimension and N dimension
// Goal: ~2-4 iterations per thread for good occupancy without wasted work
// For very large N (like LM Head), more warps help hide memory latency
__host__ inline int select_nwarps(int K, int N = 0) {
    const int n_k_blocks = K / QK_MXFP4;  // Number of K-blocks to process
    
    // For very large N (LM Head), use more warps to hide memory latency
    // despite fewer iterations per thread
    if (N > 100000) {
        return 4;  // More warps for memory-bound workloads
    }
    
    // Target: 2-4 K-blocks per thread for good work balance
    // NWARPS * 32 threads, each processes n_k_blocks / (NWARPS * 32) blocks
    
    if (n_k_blocks <= 48) {        // K <= 1536: 1 warp gives 1.5 iters/thread
        return 1;
    } else if (n_k_blocks <= 96) { // K <= 3072 (gpt-oss-120b): 2 warps gives 1.5 iters/thread
        return 2;
    } else if (n_k_blocks <= 192) { // K <= 6144: 2 warps gives 3 iters/thread
        return 2;
    } else {                        // K > 6144: 4 warps for larger models
        return 4;
    }
}

// Select optimal ROWS_PER_BLOCK based on N dimension
// Goal: Enough blocks for good SM occupancy, but not too small (launch overhead)
__host__ inline int select_rows_per_block(int N, int sm_count = 16) {
    // GB10 has 16 SMs, want at least 2 waves of blocks
    // int min_blocks = sm_count * 2;  // (unused, for future tuning)
    
    if (N <= 256) {           // Very small (e.g., tiny K/V heads)
        return 2;              // More blocks for parallelism
    } else if (N <= 512) {    // Small N (K/V projection: N=360)
        return 4;              // 360/4 = 90 blocks
    } else if (N <= 4096) {   // Medium N (Q/O projection: N=2880)
        return 8;              // 2880/8 = 360 blocks
    } else if (N <= 65536) {  // Large N
        return 8;              // Good balance
    } else {                   // Very large N (LM Head: N=201088)
        return 16;             // Fewer blocks, more work per block
    }
}

//=============================================================================
// Template instantiation helpers
//=============================================================================

// Prequant kernel launcher with specific NWARPS and ROWS
template <int NWARPS, int ROWS_PER_BLOCK>
cudaError_t launch_gemv_prequant(
    int M, int N, int K,
    const void* weights,
    const void* weight_scales,
    const void* q8_activations,
    void* output,
    cudaStream_t stream
) {
    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, M);
    dim3 block(WARP_SIZE, NWARPS);
    
    const size_t smem_reduce = (NWARPS > 1 ? NWARPS - 1 : 1) * ROWS_PER_BLOCK * WARP_SIZE * sizeof(float);
    
    gemv_vllm_mxfp4_dp4a_prequant_kernel<NWARPS, ROWS_PER_BLOCK><<<grid, block, smem_reduce, stream>>>(
        (const uint8_t*)weights,
        (const uint8_t*)weight_scales,
        (const block_q8_1*)q8_activations,
        (nv_bfloat16*)output,
        M, N, K
    );
    
    return cudaGetLastError();
}

// Fused QKV kernel launcher with specific NWARPS and ROWS
template <int NWARPS, int ROWS_PER_BLOCK>
cudaError_t launch_gemv_fused_qkv(
    int M, int N_q, int N_k, int N_v, int K,
    const void* weights_q, const void* scales_q,
    const void* weights_k, const void* scales_k,
    const void* weights_v, const void* scales_v,
    const void* q8_activations,
    void* output_q, void* output_k, void* output_v,
    cudaStream_t stream
) {
    int max_N = max(N_q, max(N_k, N_v));
    
    dim3 grid((max_N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, M, 3);
    dim3 block(WARP_SIZE, NWARPS);
    
    const size_t smem_reduce = (NWARPS > 1 ? NWARPS - 1 : 1) * ROWS_PER_BLOCK * WARP_SIZE * sizeof(float);
    
    gemv_vllm_mxfp4_dp4a_fused_qkv_kernel<NWARPS, ROWS_PER_BLOCK><<<grid, block, smem_reduce, stream>>>(
        (const uint8_t*)weights_q, (const uint8_t*)weights_k, (const uint8_t*)weights_v,
        (const uint8_t*)scales_q, (const uint8_t*)scales_k, (const uint8_t*)scales_v,
        (const block_q8_1*)q8_activations,
        (nv_bfloat16*)output_q, (nv_bfloat16*)output_k, (nv_bfloat16*)output_v,
        M, N_q, N_k, N_v, K
    );
    
    return cudaGetLastError();
}

//=============================================================================
// Dispatch macros for common configurations
//=============================================================================

#define DISPATCH_PREQUANT(nwarps, rows) \
    if (selected_nwarps == nwarps && selected_rows == rows) { \
        return launch_gemv_prequant<nwarps, rows>(M, N, K, weights, weight_scales, q8_activations, output, stream); \
    }

#define DISPATCH_FUSED_QKV(nwarps, rows) \
    if (selected_nwarps == nwarps && selected_rows == rows) { \
        return launch_gemv_fused_qkv<nwarps, rows>(M, N_q, N_k, N_v, K, \
            weights_q, scales_q, weights_k, scales_k, weights_v, scales_v, \
            q8_activations, output_q, output_k, output_v, stream); \
    }

//=============================================================================
// Internal launcher for fused QKV (with dynamic config)
//=============================================================================

cudaError_t run_gemv_fp4_dp4a_fused_qkv(
    int M, int N_q, int N_k, int N_v, int K,
    const void* weights_q, const void* scales_q,
    const void* weights_k, const void* scales_k,
    const void* weights_v, const void* scales_v,
    const void* q8_activations,
    void* output_q, void* output_k, void* output_v,
    cudaStream_t stream
) {
    // Dynamic configuration based on largest N and K
    // For fused QKV, use the max N to determine ROWS and NWARPS
    const int max_N = max(N_q, max(N_k, N_v));
    const int selected_nwarps = select_nwarps(K, max_N);
    const int selected_rows = select_rows_per_block(max_N);
    
    // Dispatch to specialized kernel
    // For gpt-oss-120b: K=2880, N_q=2880, N_k=N_v=360
    DISPATCH_FUSED_QKV(1, 2);
    DISPATCH_FUSED_QKV(1, 4);
    DISPATCH_FUSED_QKV(2, 2);
    DISPATCH_FUSED_QKV(2, 4);   // Would be selected for N_k=N_v=360
    DISPATCH_FUSED_QKV(2, 8);   // Selected for max(2880, 360) = 2880
    DISPATCH_FUSED_QKV(2, 16);  // Large QKV with K<=3072
    DISPATCH_FUSED_QKV(4, 4);
    DISPATCH_FUSED_QKV(4, 8);
    DISPATCH_FUSED_QKV(4, 16);
    
    // Fallback
    return launch_gemv_fused_qkv<4, 8>(M, N_q, N_k, N_v, K,
        weights_q, scales_q, weights_k, scales_k, weights_v, scales_v,
        q8_activations, output_q, output_k, output_v, stream);
}

//=============================================================================
// Internal launcher for pre-quantized version (with dynamic config)
//=============================================================================

cudaError_t run_gemv_fp4_dp4a_prequant(
    int M, int N, int K,
    const void* weights,
    const void* weight_scales,
    const void* q8_activations,  // Pre-quantized [M, K/32] block_q8_1
    void* output,
    cudaStream_t stream
) {
    // Dynamic configuration selection
    // Pass N to select_nwarps for large N (LM Head) optimization
    const int selected_nwarps = select_nwarps(K, N);
    const int selected_rows = select_rows_per_block(N);
    
    // Dispatch to specialized kernel
    // Common configurations for gpt-oss-120b (K=2880):
    DISPATCH_PREQUANT(1, 2);   // Very small N
    DISPATCH_PREQUANT(1, 4);   // Small N, small K
    DISPATCH_PREQUANT(2, 2);   // Very small N, medium K
    DISPATCH_PREQUANT(2, 4);   // K/V projection (N=360, K=2880)
    DISPATCH_PREQUANT(2, 8);   // Q/O projection (N=2880, K=2880)
    DISPATCH_PREQUANT(2, 16);  // Medium N with K<=3072
    DISPATCH_PREQUANT(4, 4);   // Small N, large K
    DISPATCH_PREQUANT(4, 8);   // Large N, large K (default)
    DISPATCH_PREQUANT(4, 16);  // LM Head with K>3072
    
    // Fallback to default if no match
    return launch_gemv_prequant<2, 8>(M, N, K, weights, weight_scales, q8_activations, output, stream);
}

cudaError_t run_quantize_activations(
    int M, int K,
    const void* input,
    void* output,
    cudaStream_t stream
) {
    constexpr int THREADS = 256;
    const int n_q8_blocks = K / QK8_1;
    
    // Launch enough blocks to cover all K/32 blocks
    dim3 grid((n_q8_blocks + THREADS - 1) / THREADS, M);
    dim3 block(THREADS);
    
    quantize_activations_kernel<THREADS><<<grid, block, 0, stream>>>(
        (const nv_bfloat16*)input,
        (block_q8_1*)output,
        M, K
    );
    
    return cudaGetLastError();
}

//=============================================================================
// Template launcher for original kernel (in-kernel quantization)
//=============================================================================

template <int NWARPS, int ROWS_PER_BLOCK>
cudaError_t launch_gemv_dp4a(
    int M, int N, int K,
    const void* weights,
    const void* weight_scales,
    const void* input,
    void* output,
    cudaStream_t stream
) {
    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, M);
    dim3 block(WARP_SIZE, NWARPS);
    
    const int n_q8_blocks = K / QK8_1;
    const size_t smem_q8 = n_q8_blocks * sizeof(block_q8_1);
    const size_t smem_reduce = (NWARPS > 1 ? NWARPS - 1 : 1) * ROWS_PER_BLOCK * WARP_SIZE * sizeof(float);
    const size_t smem_size = smem_q8 + smem_reduce;
    
    gemv_vllm_mxfp4_dp4a_kernel<NWARPS, ROWS_PER_BLOCK><<<grid, block, smem_size, stream>>>(
        (const uint8_t*)weights,
        (const uint8_t*)weight_scales,
        (const nv_bfloat16*)input,
        (nv_bfloat16*)output,
        M, N, K
    );
    
    return cudaGetLastError();
}

#define DISPATCH_DP4A(nwarps, rows) \
    if (selected_nwarps == nwarps && selected_rows == rows) { \
        return launch_gemv_dp4a<nwarps, rows>(M, N, K, weights, weight_scales, input, output, stream); \
    }

//=============================================================================
// Internal launcher (original with in-kernel quantization, dynamic config)
//=============================================================================

cudaError_t run_gemv_fp4_dp4a(
    int M, int N, int K,
    const void* weights,
    const void* weight_scales,
    const void* input,
    void* output,
    cudaStream_t stream
) {
    // Dynamic configuration selection
    // Pass N to select_nwarps for large N (LM Head) optimization
    const int selected_nwarps = select_nwarps(K, N);
    const int selected_rows = select_rows_per_block(N);
    
    // Dispatch to specialized kernel
    DISPATCH_DP4A(1, 2);
    DISPATCH_DP4A(1, 4);
    DISPATCH_DP4A(2, 2);
    DISPATCH_DP4A(2, 4);
    DISPATCH_DP4A(2, 8);
    DISPATCH_DP4A(2, 16);   // Medium N with K<=3072
    DISPATCH_DP4A(4, 4);
    DISPATCH_DP4A(4, 8);
    DISPATCH_DP4A(4, 16);
    
    // Fallback
    return launch_gemv_dp4a<2, 8>(M, N, K, weights, weight_scales, input, output, stream);
}

}  // namespace gemv
}  // namespace flashinfer

//=============================================================================
// TVM FFI Bindings
//=============================================================================

// GEMV for MXFP4: output[M, N] = input[M, K] @ weights[N, K].T
// Fuses BF16->INT8 activation quantization inside the kernel
void gemv_fp4_dp4a(
    int64_t M,
    int64_t N, 
    int64_t K,
    TensorView weights,       // [N, K/2] uint8 packed FP4
    TensorView weight_scales, // [N, K/32] uint8 E8M0
    TensorView input,         // [M, K] bfloat16
    TensorView output         // [M, N] bfloat16
) {
    cudaStream_t stream = get_current_stream();
    
    cudaError_t err = flashinfer::gemv::run_gemv_fp4_dp4a(
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        weights.data_ptr(),
        weight_scales.data_ptr(),
        input.data_ptr(),
        output.data_ptr(),
        stream
    );
    
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "GEMV FP4 DP4A kernel failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemv_fp4_dp4a, gemv_fp4_dp4a);

// Quantize BF16 activations to Q8_1 format (interleaved for DP4A)
// This is called once, and the output is reused across multiple GEMV calls
void quantize_activations_q8(
    int64_t M,
    int64_t K,
    TensorView input,   // [M, K] bfloat16
    TensorView output   // [M, K/32, 36] uint8 (block_q8_1 = 36 bytes)
) {
    cudaStream_t stream = get_current_stream();
    
    cudaError_t err = flashinfer::gemv::run_quantize_activations(
        static_cast<int>(M),
        static_cast<int>(K),
        input.data_ptr(),
        output.data_ptr(),
        stream
    );
    
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "Quantize activations kernel failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_activations_q8, quantize_activations_q8);

// GEMV with pre-quantized activations
// Faster than gemv_fp4_dp4a when activations are reused across layers
void gemv_fp4_dp4a_prequant(
    int64_t M,
    int64_t N,
    int64_t K,
    TensorView weights,         // [N, K/2] uint8 packed FP4
    TensorView weight_scales,   // [N, K/32] uint8 E8M0
    TensorView q8_activations,  // [M, K/32, 36] uint8 (block_q8_1)
    TensorView output           // [M, N] bfloat16
) {
    cudaStream_t stream = get_current_stream();
    
    cudaError_t err = flashinfer::gemv::run_gemv_fp4_dp4a_prequant(
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        weights.data_ptr(),
        weight_scales.data_ptr(),
        q8_activations.data_ptr(),
        output.data_ptr(),
        stream
    );
    
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "GEMV FP4 DP4A prequant kernel failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemv_fp4_dp4a_prequant, gemv_fp4_dp4a_prequant);

// GEMV with pre-quantized activations and software prefetching
// Experimental: may be faster on some workloads by hiding memory latency
void gemv_fp4_dp4a_prefetch(
    int64_t M,
    int64_t N,
    int64_t K,
    TensorView weights,
    TensorView weight_scales,
    TensorView q8_activations,
    TensorView output
) {
    cudaStream_t stream = get_current_stream();
    
    cudaError_t err = flashinfer::gemv::run_gemv_fp4_dp4a_prefetch(
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        weights.data_ptr(),
        weight_scales.data_ptr(),
        q8_activations.data_ptr(),
        output.data_ptr(),
        stream
    );
    
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "GEMV FP4 DP4A prefetch kernel failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemv_fp4_dp4a_prefetch, gemv_fp4_dp4a_prefetch);

// Fused QKV GEMV: compute Q, K, V projections in single kernel launch
// Saves kernel launch overhead and reuses activations across all 3 matrices
void gemv_fp4_dp4a_fused_qkv(
    int64_t M, int64_t N_q, int64_t N_k, int64_t N_v, int64_t K,
    TensorView weights_q, TensorView scales_q,
    TensorView weights_k, TensorView scales_k,
    TensorView weights_v, TensorView scales_v,
    TensorView q8_activations,
    TensorView output_q, TensorView output_k, TensorView output_v
) {
    cudaStream_t stream = get_current_stream();
    
    cudaError_t err = flashinfer::gemv::run_gemv_fp4_dp4a_fused_qkv(
        static_cast<int>(M),
        static_cast<int>(N_q), static_cast<int>(N_k), static_cast<int>(N_v),
        static_cast<int>(K),
        weights_q.data_ptr(), scales_q.data_ptr(),
        weights_k.data_ptr(), scales_k.data_ptr(),
        weights_v.data_ptr(), scales_v.data_ptr(),
        q8_activations.data_ptr(),
        output_q.data_ptr(), output_k.data_ptr(), output_v.data_ptr(),
        stream
    );
    
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "GEMV FP4 DP4A fused QKV kernel failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemv_fp4_dp4a_fused_qkv, gemv_fp4_dp4a_fused_qkv);

//=============================================================================
// Transposed Weight Layout for Better Memory Coalescing
//=============================================================================
//
// Original layout:   weights[N, K/2]     - each row is one output's weights
// Transposed layout: weights_t[K/32, N, 16] - each "row" is one K-block for all outputs
//
// With transposed layout, threads reading different output rows (n) will read
// consecutive memory addresses, enabling perfect coalescing.
//
// Memory access pattern comparison:
//   Original:   Thread i reads weights[n_i][kb*16:kb*16+16] - stride of K/2 bytes
//   Transposed: Thread i reads weights_t[kb][n_i][:16]      - stride of 16 bytes (coalesced!)

namespace flashinfer {
namespace gemv {

//=============================================================================
// Transpose kernel: [N, K/2] -> [K/32, N, 16]
//=============================================================================

__global__ void transpose_weights_kernel(
    const uint8_t* __restrict__ weights,      // [N, K/2] input
    uint8_t* __restrict__ weights_t,          // [K/32, N, 16] output
    int N,
    int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int kb = blockIdx.y;
    
    if (n >= N) return;
    
    const int n_k_blocks = K / 32;
    
    // Read 16 bytes from original layout
    const uint8_t* src = weights + n * (K / 2) + kb * 16;
    
    // Write 16 bytes to transposed layout
    uint8_t* dst = weights_t + kb * N * 16 + n * 16;
    
    // Copy 16 bytes (could use int4 for efficiency)
    int4 data = load_int4_unaligned(src);
    *reinterpret_cast<int4*>(dst) = data;
}

//=============================================================================
// Transpose kernel for scales: [N, K/32] -> [K/32, N]
//=============================================================================

__global__ void transpose_scales_kernel(
    const uint8_t* __restrict__ scales,       // [N, K/32] input
    uint8_t* __restrict__ scales_t,           // [K/32, N] output
    int N,
    int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int kb = blockIdx.y;
    
    if (n >= N) return;
    
    const int n_k_blocks = K / 32;
    
    // Read 1 byte from original layout
    uint8_t scale = scales[n * n_k_blocks + kb];
    
    // Write 1 byte to transposed layout
    scales_t[kb * N + n] = scale;
}

//=============================================================================
// GEMV with transposed weights - fully coalesced memory access
//=============================================================================

// Each thread handles one output row, iterates over K-blocks
// Memory access is coalesced: consecutive threads read consecutive memory
__global__ void gemv_transposed_kernel(
    const uint8_t* __restrict__ weights_t,    // [K/32, N, 16] transposed weights
    const uint8_t* __restrict__ scales_t,     // [K/32, N] transposed scales
    const block_q8_1* __restrict__ q8_activations, // [M, K/32] pre-quantized
    nv_bfloat16* __restrict__ output,         // [M, N]
    int M,
    int N,
    int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;  // Output row
    const int m = blockIdx.y;                              // Batch index
    
    if (n >= N) return;
    
    const int n_k_blocks = K / 32;
    const block_q8_1* q8_row = q8_activations + m * n_k_blocks;
    
    float sum = 0.0f;
    
    // Loop over K-blocks
    for (int kb = 0; kb < n_k_blocks; kb++) {
        // Coalesced read: consecutive threads read consecutive 16-byte chunks
        const uint8_t* w_ptr = weights_t + kb * N * 16 + n * 16;
        const int4 w_vec = *reinterpret_cast<const int4*>(w_ptr);
        
        // Coalesced read: consecutive threads read consecutive scale bytes
        const float weight_scale = e8m0_to_fp32(scales_t[kb * N + n]) * 0.5f;
        
        // Activations (shared across all threads, L2 cached)
        const block_q8_1* q8 = &q8_row[kb];
        const float act_scale = __low2float(q8->ds);
        const int* q8_qs = reinterpret_cast<const int*>(q8->qs);
        
        int sumi = 0;
        
        // Process 32 FP4 values (16 bytes)
        int2 v0 = get_int_from_table_16(w_vec.x, kvalues_fp4);
        sumi = dp4a(v0.x, q8_qs[0], sumi);
        sumi = dp4a(v0.y, q8_qs[4], sumi);
        
        int2 v1 = get_int_from_table_16(w_vec.y, kvalues_fp4);
        sumi = dp4a(v1.x, q8_qs[1], sumi);
        sumi = dp4a(v1.y, q8_qs[5], sumi);
        
        int2 v2 = get_int_from_table_16(w_vec.z, kvalues_fp4);
        sumi = dp4a(v2.x, q8_qs[2], sumi);
        sumi = dp4a(v2.y, q8_qs[6], sumi);
        
        int2 v3 = get_int_from_table_16(w_vec.w, kvalues_fp4);
        sumi = dp4a(v3.x, q8_qs[3], sumi);
        sumi = dp4a(v3.y, q8_qs[7], sumi);
        
        sum += weight_scale * act_scale * float(sumi);
    }
    
    output[m * N + n] = __float2bfloat16(sum);
}

//=============================================================================
// Launchers
//=============================================================================

cudaError_t run_transpose_weights(
    int N, int K,
    const void* weights,
    void* weights_t,
    cudaStream_t stream
) {
    const int n_k_blocks = K / 32;
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x, n_k_blocks);
    
    transpose_weights_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)weights,
        (uint8_t*)weights_t,
        N, K
    );
    
    return cudaGetLastError();
}

cudaError_t run_transpose_scales(
    int N, int K,
    const void* scales,
    void* scales_t,
    cudaStream_t stream
) {
    const int n_k_blocks = K / 32;
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x, n_k_blocks);
    
    transpose_scales_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)scales,
        (uint8_t*)scales_t,
        N, K
    );
    
    return cudaGetLastError();
}

cudaError_t run_gemv_transposed(
    int M, int N, int K,
    const void* weights_t,
    const void* scales_t,
    const void* q8_activations,
    void* output,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x, M);
    
    gemv_transposed_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)weights_t,
        (const uint8_t*)scales_t,
        (const block_q8_1*)q8_activations,
        (nv_bfloat16*)output,
        M, N, K
    );
    
    return cudaGetLastError();
}

}  // namespace gemv
}  // namespace flashinfer

//=============================================================================
// TVM FFI Bindings for transposed kernels
//=============================================================================

// Transpose weights from [N, K/2] to [K/32, N, 16]
void transpose_weights_fp4(
    int64_t N,
    int64_t K,
    TensorView weights,      // [N, K/2] uint8
    TensorView weights_t     // [K/32, N, 16] uint8
) {
    cudaStream_t stream = get_current_stream();
    
    cudaError_t err = flashinfer::gemv::run_transpose_weights(
        static_cast<int>(N),
        static_cast<int>(K),
        weights.data_ptr(),
        weights_t.data_ptr(),
        stream
    );
    
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "Transpose weights kernel failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(transpose_weights_fp4, transpose_weights_fp4);

// Transpose scales from [N, K/32] to [K/32, N]
void transpose_scales_fp4(
    int64_t N,
    int64_t K,
    TensorView scales,       // [N, K/32] uint8
    TensorView scales_t      // [K/32, N] uint8
) {
    cudaStream_t stream = get_current_stream();
    
    cudaError_t err = flashinfer::gemv::run_transpose_scales(
        static_cast<int>(N),
        static_cast<int>(K),
        scales.data_ptr(),
        scales_t.data_ptr(),
        stream
    );
    
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "Transpose scales kernel failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(transpose_scales_fp4, transpose_scales_fp4);

// GEMV with transposed weights - fully coalesced memory access
void gemv_fp4_transposed(
    int64_t M,
    int64_t N,
    int64_t K,
    TensorView weights_t,     // [K/32, N, 16] uint8 transposed
    TensorView scales_t,      // [K/32, N] uint8 transposed
    TensorView q8_activations, // [M, K/32, 36] uint8
    TensorView output          // [M, N] bfloat16
) {
    cudaStream_t stream = get_current_stream();
    
    cudaError_t err = flashinfer::gemv::run_gemv_transposed(
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        weights_t.data_ptr(),
        scales_t.data_ptr(),
        q8_activations.data_ptr(),
        output.data_ptr(),
        stream
    );
    
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "GEMV transposed kernel failed: " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemv_fp4_transposed, gemv_fp4_transposed);

