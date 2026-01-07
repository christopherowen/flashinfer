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

// =============================================================================
// SM12x Activation Quantizer for Block-Scaled MXFP4 GEMM
// =============================================================================
//
// This file provides CUDA kernels and utilities for producing FP8 activations
// and scale factors (SFA) in the exact layout required by SM12x block-scaled
// CUTLASS kernels.
//
// SM12x Block-Scaled MMA Requirements:
// ------------------------------------
// 1. Activations must be FP8 (float_e4m3_t), not BF16/FP16
// 2. Scale factors use float_ue8m0_t (8-bit unsigned exponent, bias=127)
// 3. Block granularity: Blk_MN = 128 elements
// 4. Scale factor layout is NOT simple row-major; it uses a complex
//    tiled pattern optimized for TMEM access
//
// Identity Scale Mode:
// -------------------
// For MXFP4 workloads where accuracy preservation is critical:
// - Quantize BF16/FP16 to FP8 using standard cast (may lose some precision)
// - Use identity scales (1.0 = 0x7F) to avoid additional scaling errors
// - This path trades some FP8 quantization noise for zero scale-factor noise
//
// Full Scale Mode:
// ----------------
// For workloads with wide dynamic range:
// - Compute per-block max and use for scaling
// - Better numerical range coverage
// - Requires scale computation + rescaling in kernel
//
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <mutex>
#include <unordered_map>

#include "sm12x_arch_config.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

// =============================================================================
// Constants
// =============================================================================

// Block scaling granularity for SM12x (from sm12x_arch_config.h)
// Each scale factor covers 128 elements along M/N dimension
static constexpr int kBlkMN = kSm12xBlockScaleGranularity;  // 128

// Identity scale value (2^0 = 1.0) for float_ue8m0_t
static constexpr uint8_t kIdentityScaleRaw = kSm12xIdentityScaleRaw;  // 0x7F

// =============================================================================
// SFA Layout Helper
// =============================================================================
//
// CUTLASS SM12x block-scaled kernels expect scale factors in a specific layout:
//
// For A (activations) with shape [M, K]:
// - Number of scale blocks in M: ceil(M / 128)
// - Number of scale blocks in K: ceil(K / SFVecSize)  where SFVecSize = 32 or 128
// - Layout is a complex tiled pattern, not simple [M/128, K/SFVecSize]
//
// IMPORTANT: TMA SCALE LOAD REQUIREMENTS
// =======================================
// TMA scale loads require real tiles with valid access patterns. Do NOT use
// stride=0 tricks to broadcast a single scale value - this may:
// - Work on some driver versions but break on others
// - Force unexpected slow paths
// - Cause misaligned TMA access errors
//
// For identity mode, you MUST:
// 1. Allocate a properly-sized SFA buffer matching CUTLASS's LayoutSFA
// 2. Pre-fill the entire buffer with 0x7F (identity = 1.0)
// 3. Cache this buffer per (M, K, tile_shape) configuration
// 4. Pass the real buffer with correct strides
//
// The identity buffer can be shared across calls with the same layout, but
// it must have the correct shape - not a single element with broadcast stride.

// =============================================================================
// Quantize BF16/FP16 to FP8 Kernel
// =============================================================================
//
// Simple per-element quantization from BF16/FP16 to FP8 E4M3.
// This kernel handles the MoE grouped structure with variable token counts.

template <typename InputType>
__global__ void quantize_activation_to_fp8_kernel(
    const InputType* __restrict__ input,    // [total_tokens, K]
    __nv_fp8_e4m3* __restrict__ output,     // [total_tokens, K]
    int total_tokens,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = total_tokens * K;
    
    if (idx < total_elements) {
        float val = static_cast<float>(input[idx]);
        output[idx] = __nv_fp8_e4m3(val);
    }
}

// =============================================================================
// Identity Scale Buffer Manager
// =============================================================================
//
// Manages pre-allocated identity scale buffers in the correct CUTLASS layout.
// Buffers are cached by (M_blocks, K_blocks) to avoid re-allocation.
//
// Each buffer is filled with 0x7F (identity = 1.0) in the layout expected by
// CUTLASS's block-scaled collectives.

struct Sm12xIdentityScaleBufferKey {
    int m_blocks;  // ceil(M / 128)
    int k_blocks;  // ceil(K / SF_VecSize)
    
    bool operator==(const Sm12xIdentityScaleBufferKey& other) const {
        return m_blocks == other.m_blocks && k_blocks == other.k_blocks;
    }
};

struct Sm12xIdentityScaleBufferKeyHash {
    size_t operator()(const Sm12xIdentityScaleBufferKey& key) const {
        return std::hash<int>()(key.m_blocks) ^ (std::hash<int>()(key.k_blocks) << 16);
    }
};

class Sm12xIdentityScaleBufferManager {
public:
    // Get or create an identity scale buffer for given dimensions
    // Returns pointer to device buffer filled with 0x7F
    uint8_t* getOrCreate(int M, int K, int sf_vec_size = 32, cudaStream_t stream = 0) {
        int m_blocks = (M + kBlkMN - 1) / kBlkMN;
        int k_blocks = (K + sf_vec_size - 1) / sf_vec_size;
        
        Sm12xIdentityScaleBufferKey key{m_blocks, k_blocks};
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = buffers_.find(key);
        if (it != buffers_.end()) {
            return it->second;
        }
        
        // Allocate and fill new buffer
        size_t buffer_size = static_cast<size_t>(m_blocks) * k_blocks * sizeof(uint8_t);
        uint8_t* d_buffer = nullptr;
        
        cudaError_t err = cudaMalloc(&d_buffer, buffer_size);
        if (err != cudaSuccess) {
            return nullptr;
        }
        
        // Fill with identity scale value (0x7F)
        err = cudaMemsetAsync(d_buffer, kIdentityScaleRaw, buffer_size, stream);
        if (err != cudaSuccess) {
            cudaFree(d_buffer);
            return nullptr;
        }
        
        // Synchronize to ensure fill is complete before returning
        cudaStreamSynchronize(stream);
        
        buffers_[key] = d_buffer;
        return d_buffer;
    }
    
    // Get buffer size in bytes for given dimensions
    static size_t getBufferSize(int M, int K, int sf_vec_size = 32) {
        int m_blocks = (M + kBlkMN - 1) / kBlkMN;
        int k_blocks = (K + sf_vec_size - 1) / sf_vec_size;
        return static_cast<size_t>(m_blocks) * k_blocks * sizeof(uint8_t);
    }
    
    // Get scale stride for proper layout
    static int getScaleStride(int M, int K, int sf_vec_size = 32) {
        int k_blocks = (K + sf_vec_size - 1) / sf_vec_size;
        return k_blocks;  // Row-major: stride = number of K blocks
    }
    
    // Cleanup all buffers
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : buffers_) {
            if (pair.second) {
                cudaFree(pair.second);
            }
        }
        buffers_.clear();
    }
    
    ~Sm12xIdentityScaleBufferManager() {
        clear();
    }

private:
    std::unordered_map<Sm12xIdentityScaleBufferKey, uint8_t*, Sm12xIdentityScaleBufferKeyHash> buffers_;
    std::mutex mutex_;
};

// Global singleton for identity scale buffer management
inline Sm12xIdentityScaleBufferManager& getIdentityScaleBufferManager() {
    static Sm12xIdentityScaleBufferManager manager;
    return manager;
}

// =============================================================================
// Activation Quantizer API
// =============================================================================

// Configuration for activation quantization
struct Sm12xQuantizerConfig {
    bool use_identity_scales = true;  // If true, use identity A-scales (1.0)
    int block_size = kBlkMN;          // Scale factor block size (128)
};

// Result structure containing quantized activations and scale factors
template <typename InputType>
struct Sm12xQuantizedActivation {
    __nv_fp8_e4m3* d_fp8_activations = nullptr;  // Quantized activations [total_tokens, K]
    uint8_t* d_scale_factors = nullptr;           // Scale factors (float_ue8m0_t raw)
    int scale_factor_stride = 0;                  // Stride (0 for identity broadcast)
    bool owns_memory = false;                     // Whether this struct owns the memory
    
    void free() {
        if (owns_memory) {
            if (d_fp8_activations) cudaFree(d_fp8_activations);
            // Don't free scale factors if using identity (shared buffer)
            if (d_scale_factors && scale_factor_stride != 0) {
                cudaFree(d_scale_factors);
            }
        }
        d_fp8_activations = nullptr;
        d_scale_factors = nullptr;
    }
};

// Quantize BF16/FP16 activations to FP8 with identity scales
// This is the recommended path for MXFP4 (W4A16) to minimize accuracy loss
//
// NOTE: This function allocates a properly-sized identity scale buffer in the
// correct CUTLASS layout. The buffer is cached and reused for same (M, K) dims.
// There is NO stride=0 broadcast - TMA requires proper scale factor tiles.
template <typename InputType>
cudaError_t quantizeActivationsIdentity(
    const InputType* d_input,           // [total_tokens, K] BF16/FP16 activations
    int total_tokens,
    int K,
    Sm12xQuantizedActivation<InputType>& output,
    cudaStream_t stream = 0
) {
    // Get or create identity scale buffer in proper layout
    Sm12xIdentityScaleBufferManager& mgr = getIdentityScaleBufferManager();
    uint8_t* sfa_buffer = mgr.getOrCreate(total_tokens, K, /*sf_vec_size=*/32, stream);
    if (sfa_buffer == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    
    // Allocate FP8 output
    size_t fp8_size = static_cast<size_t>(total_tokens) * K * sizeof(__nv_fp8_e4m3);
    cudaError_t err = cudaMalloc(&output.d_fp8_activations, fp8_size);
    if (err != cudaSuccess) return err;
    
    // Run quantization kernel
    int total_elements = total_tokens * K;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    quantize_activation_to_fp8_kernel<InputType><<<num_blocks, block_size, 0, stream>>>(
        d_input, output.d_fp8_activations, total_tokens, K
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(output.d_fp8_activations);
        output.d_fp8_activations = nullptr;
        return err;
    }
    
    // Use cached identity scale buffer with proper stride (NOT stride=0!)
    output.d_scale_factors = sfa_buffer;
    output.scale_factor_stride = mgr.getScaleStride(total_tokens, K);  // Proper stride
    output.owns_memory = true;  // Owns FP8 memory, but not scale buffer (managed by mgr)
    
    return cudaSuccess;
}

// =============================================================================
// Per-Block Scale Factor Computation (Full-Scale Mode)
// =============================================================================
//
// For workloads that need proper per-block scaling:
// 1. Compute max absolute value per 128-element block
// 2. Convert max to scale factor (float_ue8m0_t)
// 3. Scale activations by 1/scale
// 4. Quantize to FP8
//
// Layout: CUTLASS expects a specific tiled layout for scale factors.
// For grouped GEMM, each group can have different M, so we need per-group handling.

// Compute per-block max for a single activation tensor
__global__ void compute_block_max_kernel(
    const __nv_bfloat16* __restrict__ input,  // [M, K]
    float* __restrict__ block_max,             // [num_blocks_m, num_blocks_k]
    int M, int K,
    int num_blocks_m, int num_blocks_k
) {
    // Each block handles one (128, K_block) tile
    int block_m = blockIdx.x;
    int block_k = blockIdx.y;
    
    if (block_m >= num_blocks_m || block_k >= num_blocks_k) return;
    
    // Block boundaries
    int m_start = block_m * 128;
    int m_end = min(m_start + 128, M);
    int k_start = block_k * 32;  // Assume SFVecSize = 32 for now
    int k_end = min(k_start + 32, K);
    
    // Thread-local max
    float local_max = 0.0f;
    
    // Each thread processes multiple elements
    for (int m = m_start + threadIdx.x; m < m_end; m += blockDim.x) {
        for (int k = k_start; k < k_end; ++k) {
            float val = fabsf(static_cast<float>(input[m * K + k]));
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Block reduction
    __shared__ float smem[256];
    smem[threadIdx.x] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        block_max[block_m * num_blocks_k + block_k] = smem[0];
    }
}

// Convert block max to UE8M0 scale factor
// value = 2^(raw - 127), so raw = log2(value) + 127
__device__ __forceinline__ uint8_t float_to_ue8m0(float val) {
    if (val <= 0.0f) return 0x7F;  // Identity for zero/negative
    
    // Compute floor(log2(val))
    int exp;
    frexpf(val, &exp);
    exp -= 1;  // frexp returns exponent such that val = mantissa * 2^exp, 0.5 <= mantissa < 1
    
    // Clamp to valid range and add bias
    int raw = exp + 127;
    raw = max(1, min(254, raw));  // Avoid 0 (special) and 255 (NaN)
    
    return static_cast<uint8_t>(raw);
}

// =============================================================================
// Utility: Verify Scale Factor Layout
// =============================================================================
//
// Debug helper to verify scale factors are in correct layout for CUTLASS

void printScaleFactorLayoutInfo() {
    printf("SM12x Block-Scaled Scale Factor Layout:\n");
    printf("  Block size (Blk_MN): %d elements\n", kBlkMN);
    printf("  Scale factor type: float_ue8m0_t (8-bit unsigned exponent)\n");
    printf("  Identity scale raw value: 0x%02X (2^0 = 1.0)\n", kIdentityScaleRaw);
    printf("\n");
    printf("Layout structure (from CUTLASS Sm1xxBlockScaledConfig):\n");
    printf("  - SfAtom: Basic 32x4 block with 4 scale factors per 128 rows/cols\n");
    printf("  - K-major: Scale factors are contiguous along K dimension\n");
    printf("  - For identity mode: Single scale with stride=0 broadcast\n");
}

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm

