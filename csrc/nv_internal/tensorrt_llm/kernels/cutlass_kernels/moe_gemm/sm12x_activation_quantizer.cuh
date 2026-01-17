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
// Identity Scale Mode (current implementation):
// ----------------------------------------------
// - Quantize BF16/FP16 to FP8 using standard cast
// - Use identity scales (1.0 = 0x7F) to avoid scaling errors
// - This path trades some FP8 quantization noise for zero scale-factor noise
//
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <type_traits>

#include "sm12x_arch_config.h"

// SM12x LayoutSFA utilities (CUTLASS-derived sizing)
#include "sm12x_layout_sfa_utils.h"

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
// SFA Buffer Size Estimation (NOT LayoutSFA-derived!)
// =============================================================================
//
// WARNING: This is a SIZE ESTIMATION, not a proper LayoutSFA derivation!
// 
// CUTLASS's block-scaled layout is NOT simple row-major. The real layout is
// defined by Sm1xxBlockScaledConfig in sm100_blockscaled_layout.hpp:
//
//   SfAtom = ((32,4), (SFVecSize, 4)) with stride ((16,4), (0, 1))
//   Blk_MN = 128, Blk_SF = 4, SFVecSize = 32
//
// The CORRECT way to compute SFA size is in the LAUNCHER where the kernel's
// LayoutSFA type is known:
//   using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
//   size_t capacity = cute::cosize(LayoutSFA{}, problem_shape);
//
// WHY THIS WORKS FOR IDENTITY MODE ONLY:
// - Identity scales are all 0x7F (same value everywhere)
// - Layout pattern doesn't matter if every byte is identical
// - We just need to allocate ENOUGH bytes (buffer >= kernel's requirement)
// - memset fills the entire buffer with 0x7F
//
// WHY THIS DOES NOT WORK FOR COMPUTED SCALES:
// - Computed scales must be written in the correct tiled layout
// - Row-major write to a swizzled buffer = garbage for TMA
// - Full-scale mode must use proper layout transformation
//
// This struct provides a CONSERVATIVE size estimate that should be >= CUTLASS's
// actual requirement. If you see TMA errors or scale misreads, verify the
// computed size matches the kernel's actual LayoutSFA capacity.

// =============================================================================
// Sm12xLayoutSFASizes - Wrapper around CUTLASS-derived LayoutSFA utilities
// =============================================================================
//
// This struct provides backward-compatible sizing APIs that delegate to
// Sm12xLayoutSFAUtils for CUTLASS-derived buffer sizes.
//
// IMPORTANT: For the most accurate sizing, use the launcher's kernel-derived
// size (computeKernelSFABufferSize<CollectiveMainloop>) when available.
// This fallback is conservative and should always be >= the kernel's requirement.

struct Sm12xLayoutSFASizes {
    // Constants from the utility header
    static constexpr int kSFVecSize = kSFVecSize_SM12x;  // 32
    static constexpr int kBlkMN = kBlkMN_SM12x;          // 128
    static constexpr int kBlkSF = kBlkSF_SM12x;          // 4
    static constexpr int kAtomSize = 128;                // Size of one SfAtom in bytes (32 * 4)
    
    // Compute buffer size for SFA (in bytes) using CUTLASS LayoutSFA APIs
    // This delegates to Sm12xLayoutSFAUtils for CUTLASS-derived sizing
    // Explicitly use 4-arg version to avoid ambiguity (N=0 for SFA since it doesn't depend on N)
    static size_t computeSFABufferSize(int M, int K, int L = 1) {
        return Sm12xLayoutSFAUtils::computeBufferSize(M, 0, K, L);
    }
    
    // Stride for the layout (for legacy compatibility)
    // NOTE: LayoutSFA is complex/swizzled. For identity scales (all same value),
    // the actual stride doesn't matter for correctness. This returns a compatible
    // value for APIs that require a stride parameter.
    static int computeSFAStride(int M, int K) {
        // Each M-block of 128 elements has (K/128) atoms of 128 bytes each
        int num_k_atoms = (K + (kSFVecSize * kBlkSF) - 1) / (kSFVecSize * kBlkSF);
        return num_k_atoms * kAtomSize;  // Stride in bytes between M-blocks
    }
    
    // Verify buffer size matches expected for known shapes
    static bool verifySizeForShape(int M, int K, size_t expected_bytes) {
        size_t computed = computeSFABufferSize(M, K);
        return computed >= expected_bytes;
    }
    
    // Get a cache key for the given dimensions
    static uint64_t getCacheKey(int M, int K, int L = 1) {
        return Sm12xLayoutSFAUtils::getCacheKey(M, K, L);
    }
};

// =============================================================================
// Quantize BF16/FP16 to FP8 Kernels
// =============================================================================

// Scalar version (baseline, for reference)
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
// Vectorized Quantization Kernel (8 elements per thread)
// =============================================================================
//
// This kernel processes 8 elements per thread for better memory throughput:
// - Loads 8 BF16/FP16 values (128 bits) directly from global memory
// - Converts to 8 FP8 values
// - Stores 8 FP8 values (64 bits) per thread
//
// IMPORTANT: This kernel ONLY handles the aligned portion (num_vecs * 8).
// Tail elements MUST be handled by a separate scalar kernel launch.
// DO NOT try to handle tail in this kernel - the math is error-prone.

// Vectorized load + convert for BF16 (loads directly from global memory)
__device__ __forceinline__ void convert_bf16x8_to_fp8x8_global(
    const __nv_bfloat16* __restrict__ global_input,
    __nv_fp8_e4m3* output
) {
    // Load 4 bfloat162 directly from global memory (128 bits = 16 bytes)
    const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(global_input);
    __nv_bfloat162 v0 = src2[0];
    __nv_bfloat162 v1 = src2[1];
    __nv_bfloat162 v2 = src2[2];
    __nv_bfloat162 v3 = src2[3];
    
    // Convert to FP8
    output[0] = __nv_fp8_e4m3(__bfloat162float(__low2bfloat16(v0)));
    output[1] = __nv_fp8_e4m3(__bfloat162float(__high2bfloat16(v0)));
    output[2] = __nv_fp8_e4m3(__bfloat162float(__low2bfloat16(v1)));
    output[3] = __nv_fp8_e4m3(__bfloat162float(__high2bfloat16(v1)));
    output[4] = __nv_fp8_e4m3(__bfloat162float(__low2bfloat16(v2)));
    output[5] = __nv_fp8_e4m3(__bfloat162float(__high2bfloat16(v2)));
    output[6] = __nv_fp8_e4m3(__bfloat162float(__low2bfloat16(v3)));
    output[7] = __nv_fp8_e4m3(__bfloat162float(__high2bfloat16(v3)));
}

// Vectorized load + convert for FP16 (loads directly from global memory)
__device__ __forceinline__ void convert_fp16x8_to_fp8x8_global(
    const __half* __restrict__ global_input,
    __nv_fp8_e4m3* output
) {
    // Load 4 half2 directly from global memory (128 bits = 16 bytes)
    const __half2* src2 = reinterpret_cast<const __half2*>(global_input);
    __half2 v0 = src2[0];
    __half2 v1 = src2[1];
    __half2 v2 = src2[2];
    __half2 v3 = src2[3];
    
    // Convert to FP8
    output[0] = __nv_fp8_e4m3(__half2float(__low2half(v0)));
    output[1] = __nv_fp8_e4m3(__half2float(__high2half(v0)));
    output[2] = __nv_fp8_e4m3(__half2float(__low2half(v1)));
    output[3] = __nv_fp8_e4m3(__half2float(__high2half(v1)));
    output[4] = __nv_fp8_e4m3(__half2float(__low2half(v2)));
    output[5] = __nv_fp8_e4m3(__half2float(__high2half(v2)));
    output[6] = __nv_fp8_e4m3(__half2float(__low2half(v3)));
    output[7] = __nv_fp8_e4m3(__half2float(__high2half(v3)));
}

// Main vectorized kernel - ONLY handles aligned portion
// Tail elements are handled by separate scalar kernel
template <typename InputType>
__global__ void quantize_activation_to_fp8_vectorized_kernel(
    const InputType* __restrict__ input,    // [total_tokens, K]
    __nv_fp8_e4m3* __restrict__ output,     // [total_tokens, K]
    int num_vecs                             // Number of 8-element vectors to process
) {
    constexpr int kVecSize = 8;
    
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx >= num_vecs) return;
    
    int base_idx = vec_idx * kVecSize;
    
    // Aligned buffer for 8 FP8 values (64 bits total)
    // alignas(8) ensures proper alignment for 64-bit operations
    alignas(8) __nv_fp8_e4m3 output_buf[kVecSize];
    
    // Load directly from global memory and convert
    // Use std::is_same_v for proper type dispatch (not sizeof comparison)
    if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
        convert_bf16x8_to_fp8x8_global(input + base_idx, output_buf);
    } else if constexpr (std::is_same_v<InputType, __half>) {
        convert_fp16x8_to_fp8x8_global(input + base_idx, output_buf);
    } else {
        // Generic fallback for other types
        #pragma unroll
        for (int i = 0; i < kVecSize; i++) {
            output_buf[i] = __nv_fp8_e4m3(static_cast<float>(input[base_idx + i]));
        }
    }
    
    // Strict-aliasing-safe 64-bit store using __builtin_memcpy
    // __builtin_memcpy is optimized away by nvcc to a single 64-bit store
    // This avoids UB from reinterpret_cast type-punning
    uint2 packed;
    __builtin_memcpy(&packed, output_buf, sizeof(packed));
    
    // Store to output (output + base_idx is 8-byte aligned because
    // base_idx = vec_idx * 8 and FP8 is 1 byte)
    uint2* out_ptr = reinterpret_cast<uint2*>(output + base_idx);
    *out_ptr = packed;
}

// Tail kernel for remaining elements (called after vectorized kernel)
template <typename InputType>
__global__ void quantize_activation_to_fp8_tail_kernel(
    const InputType* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    int start_idx,          // First element to process
    int total_elements      // Total number of elements
) {
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        output[idx] = __nv_fp8_e4m3(static_cast<float>(input[idx]));
    }
}

// =============================================================================
// Identity Scale Buffer Manager (HOST-ONLY)
// =============================================================================
//
// Manages pre-allocated identity scale buffers in the correct CUTLASS layout.
// Buffers are cached by (device_id, M_blocks, K_blocks, sf_vec_size).
//
// Each buffer is filled with 0x7F (identity = 1.0) using blocking init on the
// default stream (one-time cost, globally visible to all streams).
//
// IMPORTANT: This manager does NOT use cudaStreamSynchronize inside getOrCreate
// to avoid stalling user streams or breaking CUDA graph capture.
//
// NOTE: This class uses std::mutex and std::unordered_map which are HOST-ONLY.
// It is guarded with !defined(__CUDA_ARCH__) to prevent device compilation issues.

#if !defined(__CUDA_ARCH__)  // HOST-ONLY: uses std::mutex, std::unordered_map

struct Sm12xIdentityScaleBufferKey {
    int device_id;    // CUDA device ID (critical for multi-GPU)
    int m_blocks;     // ceil(M / 128)
    int k_blocks;     // ceil(K / (SFVecSize * BlkSF))
    int sf_vec_size;  // Scale factor vector size (typically 32)
    
    bool operator==(const Sm12xIdentityScaleBufferKey& other) const {
        return device_id == other.device_id &&
               m_blocks == other.m_blocks && 
               k_blocks == other.k_blocks &&
               sf_vec_size == other.sf_vec_size;
    }
};

struct Sm12xIdentityScaleBufferKeyHash {
    size_t operator()(const Sm12xIdentityScaleBufferKey& key) const {
        // Combine all fields into hash
        size_t h = std::hash<int>()(key.device_id);
        h ^= std::hash<int>()(key.m_blocks) << 8;
        h ^= std::hash<int>()(key.k_blocks) << 16;
        h ^= std::hash<int>()(key.sf_vec_size) << 24;
        return h;
    }
};

class Sm12xIdentityScaleBufferManager {
public:
    // ==========================================================================
    // PRIMARY API: getOrCreateWithSize (kernel-derived size)
    // ==========================================================================
    //
    // Use this API when you have the kernel-derived buffer size from the launcher.
    // This is the CORRECT way to allocate SFA buffers that match CUTLASS's LayoutSFA.
    //
    // The launcher computes the exact required size using:
    //   size_t required_bytes = computeKernelSFABufferSize<CollectiveMainloop>(M, K, L);
    //
    // Then calls:
    //   uint8_t* sfa = manager.getOrCreateWithSize(required_bytes);
    //
    // The buffer is cached by (device_id, required_bytes) to avoid re-allocation.
    
    uint8_t* getOrCreateWithSize(size_t required_bytes) {
        // Get current device
        int device_id = 0;
        cudaGetDevice(&device_id);
        
        // Use required_bytes directly as the cache key (no ambiguity)
        uint64_t cache_key = (static_cast<uint64_t>(device_id) << 48) | required_bytes;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = size_based_buffers_.find(cache_key);
        if (it != size_based_buffers_.end()) {
            return it->second;
        }
        
        // Ensure we're on the correct device
        cudaSetDevice(device_id);
        
        uint8_t* d_buffer = nullptr;
        cudaError_t err = cudaMalloc(&d_buffer, required_bytes);
        if (err != cudaSuccess) {
            return nullptr;
        }
        
        // Fill with identity scale value (0x7F) using BLOCKING memset
        // This is a one-time cost at buffer creation and ensures global visibility
        // without requiring stream synchronization in the hot path.
        err = cudaMemset(d_buffer, kIdentityScaleRaw, required_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_buffer);
            return nullptr;
        }
        
        size_based_buffers_[cache_key] = d_buffer;
        return d_buffer;
    }
    
    // ==========================================================================
    // FALLBACK API: getOrCreate (dimension-based, uses conservative sizing)
    // ==========================================================================
    //
    // Use this when you don't have the kernel-derived size (e.g., pre-warming).
    // This uses Sm12xLayoutSFASizes::computeSFABufferSize which is conservative.
    //
    // NOTE: The buffer may be larger than strictly necessary, but that's safe.
    // For identity scales (all 0x7F), oversizing has no correctness impact.
    
    uint8_t* getOrCreate(int M, int K, int sf_vec_size = 32) {
        // Get current device
        int device_id = 0;
        cudaGetDevice(&device_id);
        
        // Compute key components using LayoutSFA-compatible sizing
        int m_blocks = (M + kBlkMN - 1) / kBlkMN;
        int k_atoms = (K + (sf_vec_size * Sm12xLayoutSFASizes::kBlkSF) - 1) / 
                      (sf_vec_size * Sm12xLayoutSFASizes::kBlkSF);
        
        Sm12xIdentityScaleBufferKey key{device_id, m_blocks, k_atoms, sf_vec_size};
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = buffers_.find(key);
        if (it != buffers_.end()) {
            return it->second;
        }
        
        // Ensure we're on the correct device
        cudaSetDevice(device_id);
        
        // Use CUTLASS LayoutSFA-compatible sizing (conservative)
        size_t buffer_size = Sm12xLayoutSFASizes::computeSFABufferSize(M, K);
        uint8_t* d_buffer = nullptr;
        
        cudaError_t err = cudaMalloc(&d_buffer, buffer_size);
        if (err != cudaSuccess) {
            return nullptr;
        }
        
        // Fill with identity scale value (0x7F) using BLOCKING memset
        err = cudaMemset(d_buffer, kIdentityScaleRaw, buffer_size);
        if (err != cudaSuccess) {
            cudaFree(d_buffer);
            return nullptr;
        }
        
        buffers_[key] = d_buffer;
        buffer_sizes_[key] = buffer_size;
        return d_buffer;
    }
    
    // Pre-warm buffers for known MoE shapes to avoid alloc/sync during inference
    // Call this at engine initialization with your model's dimensions
    void prewarm(const std::vector<std::pair<int, int>>& shapes, int sf_vec_size = 32) {
        for (const auto& [M, K] : shapes) {
            getOrCreate(M, K, sf_vec_size);
        }
    }
    
    // Pre-warm with kernel-derived sizes (preferred when you know the exact sizes)
    void prewarmWithSizes(const std::vector<size_t>& required_sizes) {
        for (size_t size : required_sizes) {
            getOrCreateWithSize(size);
        }
    }
    
    // Pre-warm common MoE shapes (call at module init)
    void prewarmCommonShapes() {
        // Common hidden dimensions for MoE models
        std::vector<std::pair<int, int>> shapes = {
            // (max_tokens, hidden_dim) pairs for common models
            {64, 2880},   {128, 2880},   {256, 2880},   {512, 2880},   // gpt-oss-120b
            {64, 4096},   {128, 4096},   {256, 4096},   {512, 4096},   // common
            {64, 11520},  {128, 11520},  {256, 11520},  {512, 11520},  // gpt-oss-120b intermediate
            {64, 14336},  {128, 14336},  {256, 14336},  {512, 14336},  // Mixtral-like
        };
        prewarm(shapes);
    }
    
    // Get buffer size in bytes for given dimensions (CUTLASS LayoutSFA compatible)
    static size_t getBufferSize(int M, int K, int sf_vec_size = 32) {
        return Sm12xLayoutSFASizes::computeSFABufferSize(M, K);
    }
    
    // Get scale stride for layout
    static int getScaleStride(int M, int K, int sf_vec_size = 32) {
        return Sm12xLayoutSFASizes::computeSFAStride(M, K);
    }
    
    // ==========================================================================
    // PREWARM API: Pre-allocate buffers during model initialization
    // ==========================================================================
    //
    // Call this during model load (not during inference) to pre-allocate identity
    // scale buffers for expected problem sizes. This moves the blocking cudaMemset
    // out of the inference hot path.
    //
    // Parameters:
    //   buffer_sizes: Vector of required buffer sizes (in bytes)
    //                 Get these from computeKernelSFABufferSize<CollectiveMainloop>()
    //
    // Example (during vLLM model initialization):
    //   std::vector<size_t> sizes = computeExpectedSFABufferSizes(model_config);
    //   getIdentityScaleBufferManager().prewarm(sizes);
    //
    void prewarm(const std::vector<size_t>& buffer_sizes) {
        for (size_t size : buffer_sizes) {
            getOrCreateWithSize(size);
        }
    }
    
    // Prewarm with dimension-based API (fallback, conservative sizing)
    void prewarmWithDimensions(const std::vector<std::tuple<int, int>>& mk_pairs, 
                                int sf_vec_size = 32) {
        for (const auto& [M, K] : mk_pairs) {
            getOrCreate(M, K, sf_vec_size);
        }
    }
    
    // Cleanup all buffers
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : buffers_) {
            if (pair.second) {
                cudaSetDevice(pair.first.device_id);
                cudaFree(pair.second);
            }
        }
        buffers_.clear();
        buffer_sizes_.clear();
        
        // Also clear size-based buffers
        for (auto& pair : size_based_buffers_) {
            if (pair.second) {
                // Extract device_id from cache key
                int dev = static_cast<int>(pair.first >> 48);
                cudaSetDevice(dev);
                cudaFree(pair.second);
            }
        }
        size_based_buffers_.clear();
    }
    
    ~Sm12xIdentityScaleBufferManager() {
        clear();
    }

private:
    // Dimension-based cache (fallback API)
    std::unordered_map<Sm12xIdentityScaleBufferKey, uint8_t*, Sm12xIdentityScaleBufferKeyHash> buffers_;
    std::unordered_map<Sm12xIdentityScaleBufferKey, size_t, Sm12xIdentityScaleBufferKeyHash> buffer_sizes_;
    
    // Size-based cache (primary API with kernel-derived sizes)
    // Key: (device_id << 48) | required_bytes
    std::unordered_map<uint64_t, uint8_t*> size_based_buffers_;
    
    std::mutex mutex_;
};

// Global singleton for identity scale buffer management
inline Sm12xIdentityScaleBufferManager& getIdentityScaleBufferManager() {
    static Sm12xIdentityScaleBufferManager manager;
    return manager;
}

// =============================================================================
// SFA Pointer Array Manager
// =============================================================================
//
// For grouped GEMM, CUTLASS expects ElementSF const** (device pointer to device
// pointers). This manager caches device-side pointer arrays to avoid hot-path
// allocations.
//
// Key insight: For identity scales, all groups use the SAME identity buffer,
// so the pointer array just contains N copies of the same pointer.
//
// USAGE:
//   auto& ptr_mgr = getSFAPointerArrayManager();
//   uint8_t const** d_sfa_ptrs = ptr_mgr.getOrCreate(num_groups, identity_sfa_ptr);
//   hopper_inputs.fpX_block_scaling_factors_act = d_sfa_ptrs;
//
// The manager caches by (device_id, num_groups, identity_ptr) so repeated calls
// with the same parameters return the cached array without allocation.
//

class Sm12xSFAPointerArrayManager {
public:
    // Get or create a device-side pointer array where all entries point to identity_sfa
    // Returns device pointer to array of num_groups pointers
    uint8_t const** getOrCreate(int num_groups, uint8_t const* identity_sfa_ptr) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        
        // Cache key: (device_id, num_groups, identity_sfa_ptr)
        CacheKey key{device_id, num_groups, identity_sfa_ptr};
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        
        // Allocate and fill on this device
        cudaSetDevice(device_id);
        
        // Allocate device-side pointer array
        uint8_t const** d_ptr_array = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr_array, num_groups * sizeof(uint8_t*));
        if (err != cudaSuccess) {
            return nullptr;
        }
        
        // Fill host array with identity pointers
        std::vector<uint8_t const*> h_ptrs(num_groups, identity_sfa_ptr);
        
        // Copy to device (blocking - one-time init cost)
        err = cudaMemcpy(d_ptr_array, h_ptrs.data(), 
                         num_groups * sizeof(uint8_t*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_ptr_array);
            return nullptr;
        }
        
        cache_[key] = d_ptr_array;
        return d_ptr_array;
    }
    
    // Prewarm for common group counts
    void prewarm(const std::vector<int>& group_counts, uint8_t const* identity_sfa_ptr) {
        for (int num_groups : group_counts) {
            getOrCreate(num_groups, identity_sfa_ptr);
        }
    }
    
    // Clear all cached arrays (for cleanup)
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [key, ptr] : cache_) {
            if (ptr) {
                cudaSetDevice(key.device_id);
                cudaFree(const_cast<uint8_t**>(ptr));
            }
        }
        cache_.clear();
    }
    
    ~Sm12xSFAPointerArrayManager() {
        clear();
    }

private:
    struct CacheKey {
        int device_id;
        int num_groups;
        uint8_t const* identity_ptr;
        
        bool operator==(const CacheKey& other) const {
            return device_id == other.device_id &&
                   num_groups == other.num_groups &&
                   identity_ptr == other.identity_ptr;
        }
    };
    
    struct CacheKeyHash {
        size_t operator()(const CacheKey& key) const {
            size_t h = std::hash<int>()(key.device_id);
            h ^= std::hash<int>()(key.num_groups) << 8;
            h ^= std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(key.identity_ptr)) << 16;
            return h;
        }
    };
    
    std::unordered_map<CacheKey, uint8_t const**, CacheKeyHash> cache_;
    std::mutex mutex_;
};

// Global singleton for SFA pointer array management
inline Sm12xSFAPointerArrayManager& getSFAPointerArrayManager() {
    static Sm12xSFAPointerArrayManager manager;
    return manager;
}

// =============================================================================
// Unified SM12x Prewarm Function
// =============================================================================
//
// Call this during model initialization to pre-allocate all SM12x buffers
// that would otherwise be allocated lazily on the hot path.
//
// This function:
// 1. Pre-allocates identity scale buffers for expected problem sizes
// 2. Pre-allocates SFA pointer arrays for expected expert counts
//
// Parameters:
//   num_experts: Number of experts in the MoE model (e.g., 8, 16, 64)
//   max_tokens: Maximum expected batch size (total tokens across all sequences)
//   hidden_size: Hidden dimension size (K dimension)
//   intermediate_size: Intermediate dimension (for FFN layers)
//   sf_vec_size: Scale factor vector size (default 32 for MXFP4)
//
// Example (during vLLM model load):
//   prewarmSm12xMoEBuffers(64, 8192, 4096, 16384);
//
// NOTE: This performs blocking cudaMalloc/cudaMemset/cudaMemcpy operations.
//       Call during initialization, NOT during inference.
//
inline void prewarmSm12xMoEBuffers(int num_experts, 
                                    int max_tokens,
                                    int hidden_size, 
                                    int intermediate_size,
                                    int sf_vec_size = 32) {
    // Get managers
    auto& identity_mgr = getIdentityScaleBufferManager();
    auto& ptr_mgr = getSFAPointerArrayManager();
    
    // Prewarm identity scale buffers for common (M, K) combinations
    // M dimension variations (batch sizes)
    std::vector<int> m_sizes = {1, 32, 128, 512, 1024, 2048, 4096};
    if (max_tokens > 4096) {
        m_sizes.push_back(max_tokens);
    }
    
    // K dimension variations (hidden/intermediate sizes)
    std::vector<int> k_sizes = {hidden_size, intermediate_size};
    
    std::vector<std::tuple<int, int>> mk_pairs;
    for (int m : m_sizes) {
        for (int k : k_sizes) {
            mk_pairs.emplace_back(m, k);
        }
    }
    identity_mgr.prewarmWithDimensions(mk_pairs, sf_vec_size);
    
    // Prewarm pointer arrays for this expert count
    // First, get an identity buffer for the largest expected size
    int max_m = m_sizes.back();
    int max_k = (hidden_size > intermediate_size) ? hidden_size : intermediate_size;
    uint8_t* identity_ptr = identity_mgr.getOrCreate(max_m, max_k, sf_vec_size);
    
    if (identity_ptr) {
        ptr_mgr.prewarm({num_experts}, identity_ptr);
    }
}

#endif  // !defined(__CUDA_ARCH__)  // End HOST-ONLY section

// =============================================================================
// Activation Quantizer API
// =============================================================================
//
// CUTLASS LayoutSFA Integration:
// ------------------------------
// This implementation uses Sm12xLayoutSFASizes to compute buffer sizes that
// are compatible with CUTLASS's block-scaled collectives.
//
// The scale factor layout is based on CUTLASS's Sm1xxBlockScaledConfig:
//   - SfAtom: ((32,4), (SFVecSize, 4)) with stride ((16,4), (0, 1))
//   - Blk_MN = 128 elements per scale block
//   - Blk_SF = 4 scale factors per K block
//   - Buffer size includes alignment for TMA (256-byte boundaries)
//
// For IDENTITY scales (all 0x7F), memset works regardless of tiled layout.
// For COMPUTED scales, proper tiled fill would be needed.

// Workspace sizes for pre-allocation (uses CUTLASS LayoutSFA-compatible sizing)
struct Sm12xQuantizerWorkspaceSizes {
    size_t fp8_activation_bytes;   // Size for FP8 activations
    size_t sfa_bytes;              // Size for A-scale factors (CUTLASS LayoutSFA compatible)
    size_t total_bytes;            // Total workspace size
    int sfa_stride;                // Stride for scale factors
    
    // Compute sizes for given dimensions
    static Sm12xQuantizerWorkspaceSizes compute(int total_tokens, int K, int sf_vec_size = 32) {
        Sm12xQuantizerWorkspaceSizes sizes;
        
        // FP8 activation size (simple: total_tokens * K bytes, aligned to 256)
        sizes.fp8_activation_bytes = static_cast<size_t>(total_tokens) * K * sizeof(__nv_fp8_e4m3);
        sizes.fp8_activation_bytes = (sizes.fp8_activation_bytes + 255) & ~size_t(255);
        
        // SFA size using CUTLASS LayoutSFA-compatible computation
        sizes.sfa_bytes = Sm12xLayoutSFASizes::computeSFABufferSize(total_tokens, K);
        sizes.sfa_stride = Sm12xLayoutSFASizes::computeSFAStride(total_tokens, K);
        
        sizes.total_bytes = sizes.fp8_activation_bytes + sizes.sfa_bytes;
        
        return sizes;
    }
};

// Result structure containing quantized activations and scale factors
// Does NOT own memory - caller provides workspace
struct Sm12xQuantizedActivationView {
    __nv_fp8_e4m3* d_fp8_activations = nullptr;  // Points into workspace
    uint8_t* d_scale_factors = nullptr;           // Points into workspace
    int scale_factor_stride = 0;                  // Row-major stride (k_blocks)
};

// WORKSPACE-BASED quantizer (FP8 conversion only, SFA ownership at caller)
// Caller pre-allocates workspace once and reuses across calls.
//
// IMPORTANT: This quantizer ONLY handles BF16/FP16 -> FP8 conversion.
// Identity SFA buffers are NOT managed here - they are owned by the launcher/dispatch site.
// This separation ensures kernel-derived sizing is used for SFA buffers.
//
// Features:
// - Zero allocations in hot path
// - Quantizer does NOT touch SFA when identity_sfa_ptr is provided
// - Vectorized quantization kernel (8 elements per thread)
//
// Usage:
//   1. Launcher acquires identity SFA via computeSm120IdentitySFABufferSize() + acquireSm120IdentitySFABuffer()
//   2. Call Sm12xQuantizerWorkspaceSizes::compute() to get FP8 workspace size
//   3. Pre-allocate FP8 workspace (once at engine init)
//   4. Call quantizeActivationsOnly() per inference step
//   5. Pass both FP8 output and identity_sfa_ptr to the kernel
//
// For backward compatibility, quantizeActivationsWithWorkspace() is still available
// but prefers caller-provided identity_sfa_ptr over internal acquisition.

// PRIMARY API: Quantize activations ONLY (SFA provided by caller)
// This is the correct API for identity scale mode.
// The caller provides the identity SFA buffer acquired via acquireSm120IdentitySFABuffer()
template <typename InputType>
cudaError_t quantizeActivationsOnly(
    const InputType* d_input,           // [total_tokens, K] BF16/FP16 activations
    int total_tokens,
    int K,
    void* d_fp8_workspace,              // Pre-allocated workspace for FP8 output
    size_t workspace_bytes,             // Size of workspace (must be >= total_tokens * K)
    __nv_fp8_e4m3*& d_fp8_output,       // Output: pointer to FP8 activations
    cudaStream_t stream = 0
) {
    size_t required_bytes = static_cast<size_t>(total_tokens) * K * sizeof(__nv_fp8_e4m3);
    required_bytes = (required_bytes + 255) & ~size_t(255);  // Align to 256
    
    if (workspace_bytes < required_bytes) {
        return cudaErrorInvalidValue;  // Workspace too small
    }
    
    d_fp8_output = reinterpret_cast<__nv_fp8_e4m3*>(d_fp8_workspace);
    
    // Run VECTORIZED quantization kernel (8 elements per thread) + TAIL kernel
    int total_elements = total_tokens * K;
    constexpr int kVecSize = 8;
    int num_vecs = total_elements / kVecSize;
    int tail_start = num_vecs * kVecSize;
    int tail_count = total_elements - tail_start;
    int block_size = 256;
    
    // Step 1: Vectorized kernel for aligned portion
    if (num_vecs > 0) {
        int num_blocks = (num_vecs + block_size - 1) / block_size;
        quantize_activation_to_fp8_vectorized_kernel<InputType><<<num_blocks, block_size, 0, stream>>>(
            d_input, d_fp8_output, num_vecs
        );
    }
    
    // Step 2: Tail kernel for remaining elements
    if (tail_count > 0) {
        int tail_blocks = (tail_count + block_size - 1) / block_size;
        quantize_activation_to_fp8_tail_kernel<InputType><<<tail_blocks, block_size, 0, stream>>>(
            d_input, d_fp8_output, tail_start, total_elements
        );
    }
    
    return cudaGetLastError();
}

// BACKWARD COMPATIBLE API: Quantize activations with identity SFA
// Caller MUST provide identity_sfa_ptr (use acquireSm120IdentitySFABuffer())
// use_identity_scales MUST be true (non-identity mode not implemented)
template <typename InputType>
cudaError_t quantizeActivationsWithWorkspace(
    const InputType* d_input,           // [total_tokens, K] BF16/FP16 activations
    int total_tokens,
    int K,
    void* d_workspace,                  // Pre-allocated workspace for FP8 output
    size_t workspace_bytes,             // Size of workspace
    bool use_identity_scales,           // MUST be true (non-identity not implemented)
    Sm12xQuantizedActivationView& output,
    cudaStream_t stream = 0,
    uint8_t* identity_sfa_ptr = nullptr, // Caller-provided identity SFA buffer (required)
    int identity_sfa_stride = 0          // Stride for identity SFA
) {
    // Only identity scale mode is supported
    if (!use_identity_scales) {
        return cudaErrorNotSupported;  // Non-identity mode not implemented
    }
    
    // Caller MUST provide identity SFA buffer
    if (identity_sfa_ptr == nullptr) {
        return cudaErrorInvalidValue;  // Use acquireSm120IdentitySFABuffer()
    }
    
    // Compute required sizes
    auto sizes = Sm12xQuantizerWorkspaceSizes::compute(total_tokens, K);
    
    // Check workspace size
    if (workspace_bytes < sizes.fp8_activation_bytes) {
        return cudaErrorInvalidValue;
    }
    
    // Set up output
    uint8_t* ws = static_cast<uint8_t*>(d_workspace);
    output.d_scale_factors = identity_sfa_ptr;
    output.scale_factor_stride = identity_sfa_stride > 0 ? identity_sfa_stride : sizes.sfa_stride;
    
    // FP8 output always from workspace start
    output.d_fp8_activations = reinterpret_cast<__nv_fp8_e4m3*>(ws);
    
    // Run VECTORIZED quantization kernel (8 elements per thread) + TAIL kernel
    int total_elements = total_tokens * K;
    constexpr int kVecSize = 8;
    int num_vecs = total_elements / kVecSize;
    int tail_start = num_vecs * kVecSize;
    int tail_count = total_elements - tail_start;
    int block_size = 256;
    
    // Step 1: Vectorized kernel for aligned portion
    if (num_vecs > 0) {
        int num_blocks = (num_vecs + block_size - 1) / block_size;
        quantize_activation_to_fp8_vectorized_kernel<InputType><<<num_blocks, block_size, 0, stream>>>(
            d_input, output.d_fp8_activations, num_vecs
        );
    }
    
    // Step 2: Tail kernel for remaining elements (separate launch, no complex in-kernel math)
    if (tail_count > 0) {
        int tail_blocks = (tail_count + block_size - 1) / block_size;
        quantize_activation_to_fp8_tail_kernel<InputType><<<tail_blocks, block_size, 0, stream>>>(
            d_input, output.d_fp8_activations, tail_start, total_elements
        );
    }
    
    // Fallback: if no vectors (very small input), just use scalar
    if (num_vecs == 0 && tail_count == 0) {
        // This shouldn't happen (would mean total_elements == 0), but be safe
        return cudaSuccess;
    }
    
    return cudaGetLastError();
}

// Legacy API with per-call allocation (for testing only, NOT for production)
// Use quantizeActivationsWithWorkspace() in production!
template <typename InputType>
[[deprecated("Use quantizeActivationsWithWorkspace() for production")]]
cudaError_t quantizeActivationsIdentity(
    const InputType* d_input,
    int total_tokens,
    int K,
    Sm12xQuantizedActivationView& output,
    cudaStream_t stream = 0
) {
    // Get or create identity scale buffer (cached, blocking init on first creation)
    Sm12xIdentityScaleBufferManager& mgr = getIdentityScaleBufferManager();
    uint8_t* sfa_buffer = mgr.getOrCreate(total_tokens, K, /*sf_vec_size=*/32);
    if (sfa_buffer == nullptr) {
        return cudaErrorMemoryAllocation;
    }
    
    // Allocate FP8 output (WARNING: malloc in hot path!)
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
    
    output.d_scale_factors = sfa_buffer;
    output.scale_factor_stride = mgr.getScaleStride(total_tokens, K);
    
    return cudaSuccess;
}

// =============================================================================
// Utility: Verify Scale Factor Layout
// =============================================================================
//
// Debug helper to verify scale factors are in correct layout for CUTLASS

inline void printScaleFactorLayoutInfo() {
    printf("SM12x Block-Scaled Scale Factor Layout:\n");
    printf("  Block size (Blk_MN): %d elements\n", kBlkMN);
    printf("  Scale factor type: float_ue8m0_t (8-bit unsigned exponent)\n");
    printf("  Identity scale raw value: 0x%02X (2^0 = 1.0)\n", kIdentityScaleRaw);
    printf("\n");
    printf("Layout structure (from CUTLASS Sm1xxBlockScaledConfig):\n");
    printf("  - SfAtom: Basic 32x4 block with 4 scale factors per 128 rows/cols\n");
    printf("  - K-major: Scale factors are contiguous along K dimension\n");
    printf("  - Identity mode: All scales = 0x7F, memset works\n");
}

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm

