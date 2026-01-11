/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

// Include CUTLASS types and MMA definitions
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels_oss {

// =============================================================================
// SM120 (Blackwell GeForce) PTX MMA Operations for MXFP4
// =============================================================================
//
// MXFP4 uses the block-scaled MMA instruction:
//   mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0
//
// This performs: D = A * B * scale_A * scale_B + C
// Where:
//   - A is FP8 (e4m3), size m16 x k32
//   - B is FP4 (e2m1), size k32 x n8
//   - scale_A, scale_B are UE8M0 (one per 32 elements)
//   - C, D are FP32, size m16 x n8
//
// Fragment sizes per thread (32 threads per warp):
//   - A fragment: 4 x u32 = 16 bytes
//   - B fragment: 2 x u32 = 8 bytes
//   - C/D fragment: 4 x f32 = 16 bytes
//   - scale_A: 1 x u8
//   - scale_B: 1 x u8
// =============================================================================

// Use CUTLASS's SM120 MMA with block scaling
using MMA_MXFP4 = cute::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,   // A type (FP8)
    cutlass::float_e2m1_t,   // B type (FP4)
    float,                    // D type (FP32)
    cutlass::float_ue8m0_t,  // Scale type (UE8M0)
    32                        // Vector size (32 elements per scale)
>;

// Non-scaled version for comparison
using MMA_F8F4 = cute::SM120_16x8x32_TN<
    cutlass::float_e4m3_t,   // A type (FP8)
    cutlass::float_e2m1_t,   // B type (FP4)
    float                     // D type (FP32)
>;

// =============================================================================
// Warp-Level GEMM Tile using PTX MMA
// =============================================================================
// Each warp computes a 16x8 output tile using multiple K iterations.
// For a 128x128x128 CTA tile with MXFP4:
//   - CTA: 8 warps (256 threads)
//   - Each warp: multiple 16x8 tiles
//   - K loop: 128/32 = 4 iterations

template<int TILE_M = 16, int TILE_N = 8, int TILE_K = 32>
struct WarpMmaTile {
    // Per-thread accumulator storage
    float acc[4];
    
    __device__ __forceinline__ void clear() {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            acc[i] = 0.0f;
        }
    }
    
    // Block-scaled MMA (MXFP4)
    __device__ __forceinline__ void mma_scaled(
        uint32_t const* a_frag,   // A fragment: 4 x u32
        uint32_t const* b_frag,   // B fragment: 2 x u32
        uint8_t scale_a,          // A scale (UE8M0)
        uint8_t scale_b)          // B scale (UE8M0)
    {
        MMA_MXFP4::fma(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            acc[0], acc[1], acc[2], acc[3],
            scale_a, scale_b
        );
    }
    
    // Non-scaled MMA (for testing)
    __device__ __forceinline__ void mma(
        uint32_t const* a_frag,   // A fragment: 4 x u32
        uint32_t const* b_frag)   // B fragment: 2 x u32
    {
        MMA_F8F4::fma(
            acc[0], acc[1], acc[2], acc[3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1],
            acc[0], acc[1], acc[2], acc[3]
        );
    }
};

// =============================================================================
// Fragment Loading Helpers
// =============================================================================
// These handle the warp-level data distribution for MMA operations.

// Load A fragment (FP8) from shared memory
// Each thread loads a portion of the 16x32 tile
__device__ __forceinline__ void load_a_fragment_smem(
    uint32_t* frag,
    const __nv_fp8_e4m3* smem_a,
    int k_offset,
    int smem_stride)
{
    int lane = threadIdx.x % 32;
    
    // SM120 MMA A fragment layout:
    // Thread mapping for m16n8k32 with FP8
    int row = lane % 16;
    
    // Load 4 consecutive FP8 values per thread
    const uint32_t* src = reinterpret_cast<const uint32_t*>(
        smem_a + row * smem_stride + k_offset + (lane / 16) * 16
    );
    
    frag[0] = src[0];
    frag[1] = src[1];
    frag[2] = src[2];
    frag[3] = src[3];
}

// Load B fragment (FP4) from shared memory
// Each thread loads a portion of the 32x8 tile (column-major)
__device__ __forceinline__ void load_b_fragment_smem(
    uint32_t* frag,
    const __nv_fp4_e2m1* smem_b,
    int k_offset,
    int smem_stride)
{
    int lane = threadIdx.x % 32;
    
    // SM120 MMA B fragment layout:
    // Thread mapping for m16n8k32 with FP4 (column-major)
    int col = lane % 8;
    
    // FP4 is 4 bits, so 2 elements per byte, 8 per u32
    const uint32_t* src = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const uint8_t*>(smem_b) + 
        col * smem_stride + k_offset / 2 + (lane / 8) * 8
    );
    
    frag[0] = src[0];
    frag[1] = src[1];
}

// =============================================================================
// Simple PTX-based MoE GEMM Kernel
// =============================================================================
// This kernel demonstrates the PTX MMA approach.
// A production version would need:
// - Proper shared memory staging
// - Software pipelining
// - Warp specialization
// - Epilogue fusion

__global__ void sm120_ptx_moe_gemm_simple(
    const __nv_fp8_e4m3* __restrict__ activations,  // [num_tokens, K]
    const __nv_fp4_e2m1* __restrict__ weights,       // [num_experts, N, K/2] (packed FP4)
    const uint8_t* __restrict__ act_scales,          // [num_tokens, K/32]
    const uint8_t* __restrict__ weight_scales,       // [num_experts, N/128, K/32] per-block
    __nv_bfloat16* __restrict__ output,              // [num_tokens, N]
    const int* __restrict__ sorted_token_ids,        // [num_sorted_tokens]
    const int* __restrict__ expert_ids,              // [num_sorted_tokens]
    const int* __restrict__ num_tokens_per_expert,   // [num_experts]
    const float* __restrict__ routing_weights,       // [num_tokens]
    int num_tokens,
    int N,
    int K)
{
    // Shared memory for input staging
    extern __shared__ char smem[];
    __nv_fp8_e4m3* smem_a = reinterpret_cast<__nv_fp8_e4m3*>(smem);
    __nv_fp4_e2m1* smem_b = reinterpret_cast<__nv_fp4_e2m1*>(smem_a + 128 * 128);  // After A tile
    
    // Block handles one expert's tokens
    int expert_id = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;
    
    // Get token range for this expert
    int token_start = 0;
    for (int e = 0; e < expert_id; e++) {
        token_start += num_tokens_per_expert[e];
    }
    int num_expert_tokens = num_tokens_per_expert[expert_id];
    
    // Tile output: each warp handles a 16x8 tile
    // Multiple warps tile the 128x128 output
    for (int m_tile = warp_id; m_tile < num_expert_tokens; m_tile += num_warps * 8) {
        for (int n_tile = 0; n_tile < N; n_tile += 128) {
            WarpMmaTile<> tile;
            tile.clear();
            
            // K loop
            for (int k_tile = 0; k_tile < K; k_tile += 32) {
                // Load A and B fragments from global to shared to registers
                // (Simplified - production would use async copy + pipelining)
                
                uint32_t a_frag[4];
                uint32_t b_frag[2];
                
                // Get scales for this K block
                int token_id = sorted_token_ids[token_start + m_tile];
                uint8_t scale_a = act_scales[token_id * (K/32) + k_tile/32];
                uint8_t scale_b = weight_scales[expert_id * (N/128) * (K/32) + (n_tile/128) * (K/32) + k_tile/32];
                
                // Perform block-scaled MMA
                tile.mma_scaled(a_frag, b_frag, scale_a, scale_b);
            }
            
            // Apply routing weight and store
            int token_id = sorted_token_ids[token_start + m_tile + lane_id / 4];
            float route_weight = routing_weights[token_id];
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                tile.acc[i] *= route_weight;
            }
            
            // Store results to global memory
            // (Simplified - proper thread mapping needed)
            int out_row = m_tile + (lane_id % 16);
            int out_col = n_tile + (lane_id / 16) * 4;
            
            if (out_row < num_expert_tokens && out_col < N) {
                output[sorted_token_ids[token_start + out_row] * N + out_col] = 
                    __float2bfloat16(tile.acc[0]);
            }
        }
    }
}

// =============================================================================
// Launcher Function
// =============================================================================

void launch_sm120_ptx_moe_gemm(
    const __nv_fp8_e4m3* activations,
    const __nv_fp4_e2m1* weights,
    const uint8_t* act_scales,
    const uint8_t* weight_scales,
    __nv_bfloat16* output,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_per_expert,
    const float* routing_weights,
    int num_tokens,
    int num_experts,
    int N,
    int K,
    cudaStream_t stream)
{
    // Grid: one block per expert
    // Block: 256 threads (8 warps)
    dim3 grid(num_experts);
    dim3 block(256);
    
    // Shared memory: A tile (128*128 FP8) + B tile (128*128 FP4 packed)
    size_t smem_size = 128 * 128 * sizeof(__nv_fp8_e4m3) + 128 * 128 / 2;
    
    sm120_ptx_moe_gemm_simple<<<grid, block, smem_size, stream>>>(
        activations, weights, act_scales, weight_scales, output,
        sorted_token_ids, expert_ids, num_tokens_per_expert, routing_weights,
        num_tokens, N, K
    );
}

}  // namespace cutlass_kernels_oss
}  // namespace kernels
}  // namespace tensorrt_llm
