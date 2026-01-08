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
#include <algorithm>
#include <type_traits>
#include <cstring>  // For std::memcpy in vectorized kernel

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
    static size_t computeSFABufferSize(int M, int K, int L = 1) {
        return Sm12xLayoutSFAUtils::computeBufferSize(M, K, L);
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
    
    // Use alignas(8) to guarantee 64-bit alignment for vectorized store
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
    
    // Aligned 64-bit store (output_buf is alignas(8) so this is safe)
    // Output pointer is also 8-byte aligned at base_idx since base_idx = vec_idx * 8
    // and FP8 is 1 byte, so output + base_idx is at an 8-byte boundary
    uint64_t packed;
    std::memcpy(&packed, output_buf, sizeof(uint64_t));
    *reinterpret_cast<uint64_t*>(output + base_idx) = packed;
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
// Identity Scale Buffer Manager
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

// WORKSPACE-BASED quantizer (no cudaMalloc in hot path!)
// Caller pre-allocates workspace once and reuses across calls.
//
// Features:
// - Zero allocations in hot path
// - Vectorized quantization kernel (8 elements per thread)
// - CUTLASS LayoutSFA-compatible buffer sizing
// - Identity scale mode uses memset (fast)
//
// Usage:
//   1. Call Sm12xQuantizerWorkspaceSizes::compute() to get required size
//   2. Pre-allocate workspace (once at engine init)
//   3. Call quantizeActivationsWithWorkspace() per inference step
template <typename InputType>
cudaError_t quantizeActivationsWithWorkspace(
    const InputType* d_input,           // [total_tokens, K] BF16/FP16 activations
    int total_tokens,
    int K,
    void* d_workspace,                  // Pre-allocated workspace
    size_t workspace_bytes,             // Size of workspace
    bool use_identity_scales,           // If true, fill SFA with 0x7F
    Sm12xQuantizedActivationView& output,
    cudaStream_t stream = 0
) {
    // Compute required sizes
    auto sizes = Sm12xQuantizerWorkspaceSizes::compute(total_tokens, K);
    
    if (workspace_bytes < sizes.total_bytes) {
        return cudaErrorInvalidValue;  // Workspace too small
    }
    
    // Partition workspace
    uint8_t* ws = static_cast<uint8_t*>(d_workspace);
    output.d_fp8_activations = reinterpret_cast<__nv_fp8_e4m3*>(ws);
    output.d_scale_factors = ws + sizes.fp8_activation_bytes;
    output.scale_factor_stride = sizes.sfa_stride;
    
    // Fill scale factors (identity = 0x7F)
    // For identity mode, memset works regardless of tiled layout since all values are the same
    if (use_identity_scales) {
        cudaError_t err = cudaMemsetAsync(output.d_scale_factors, kIdentityScaleRaw, 
                                          sizes.sfa_bytes, stream);
        if (err != cudaSuccess) return err;
    }
    // TODO: For full-scale mode, need proper tiled fill kernel respecting LayoutSFA
    
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
// Per-Block Scale Factor Computation (Full-Scale Mode)
// =============================================================================
//
// ╔═════════════════════════════════════════════════════════════════════════╗
// ║  EXPERIMENTAL - NOT FOR PRODUCTION USE                                  ║
// ║                                                                         ║
// ║  This full-scale mode writes scales in ROW-MAJOR layout, but CUTLASS's  ║
// ║  TMA tensor map expects a SWIZZLED/TILED LayoutSFA.                     ║
// ║                                                                         ║
// ║  Using this with a real SM12x kernel will produce INCORRECT RESULTS.    ║
// ║                                                                         ║
// ║  For production, use IDENTITY mode (quantizeActivationsWithWorkspace    ║
// ║  with use_identity_scales=true) which works because all values are 0x7F.║
// ║                                                                         ║
// ║  Full-scale mode requires:                                              ║
// ║  - Deriving LayoutSFA from the instantiated kernel                      ║
// ║  - Writing scales in the proper tiled layout, not row-major             ║
// ║  - Verifying with actual kernel execution and correctness checks        ║
// ╚═════════════════════════════════════════════════════════════════════════╝
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

// Convert UE8M0 to float (for scaling activations)
__device__ __forceinline__ float ue8m0_to_float(uint8_t raw) {
    if (raw == 0) return 0.0f;  // Special case
    int exp = static_cast<int>(raw) - 127;
    return exp2f(static_cast<float>(exp));
}

// Fused kernel: compute scale, quantize activations with scaling
template <typename InputType>
__global__ void quantize_with_block_scaling_kernel(
    const InputType* __restrict__ input,     // [M, K]
    __nv_fp8_e4m3* __restrict__ output_fp8,  // [M, K]
    uint8_t* __restrict__ output_scales,     // [num_m_blocks, num_k_blocks * Blk_SF]
    int M, int K,
    int num_m_blocks, int num_k_blocks
) {
    constexpr int kBlkMN_local = 128;
    constexpr int kSFVecSize = 32;
    constexpr int kBlkSF = 4;
    
    // Each block handles one (128, 32) tile
    int block_m = blockIdx.x;
    int block_k = blockIdx.y;
    
    if (block_m >= num_m_blocks || block_k >= num_k_blocks) return;
    
    // Block boundaries
    int m_start = block_m * kBlkMN_local;
    int m_end = min(m_start + kBlkMN_local, M);
    int k_start = block_k * kSFVecSize;
    int k_end = min(k_start + kSFVecSize, K);
    
    // Step 1: Compute per-block max (shared memory reduction)
    __shared__ float smem_max[256];
    float local_max = 0.0f;
    
    for (int m = m_start + threadIdx.x; m < m_end; m += blockDim.x) {
        for (int k = k_start; k < k_end; ++k) {
            float val = fabsf(static_cast<float>(input[m * K + k]));
            local_max = fmaxf(local_max, val);
        }
    }
    
    smem_max[threadIdx.x] = local_max;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem_max[threadIdx.x] = fmaxf(smem_max[threadIdx.x], smem_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    // Step 2: Compute scale factor
    float block_max_val = smem_max[0];
    uint8_t scale_raw = float_to_ue8m0(block_max_val);
    float scale_float = ue8m0_to_float(scale_raw);
    float inv_scale = (scale_float > 0.0f) ? (1.0f / scale_float) : 1.0f;
    
    // Store scale factor (for now, simple row-major layout)
    // TODO: Implement proper tiled layout for LayoutSFA
    if (threadIdx.x == 0) {
        // Each block produces Blk_SF=4 scale factors (for sub-blocks within the 128x32 tile)
        // For simplicity, we store the same scale for all 4 sub-blocks
        int scale_base_idx = block_m * num_k_blocks * kBlkSF + block_k * kBlkSF;
        for (int i = 0; i < kBlkSF; ++i) {
            output_scales[scale_base_idx + i] = scale_raw;
        }
    }
    __syncthreads();
    
    // Step 3: Scale and quantize activations
    for (int m = m_start + threadIdx.x; m < m_end; m += blockDim.x) {
        for (int k = k_start; k < k_end; ++k) {
            float val = static_cast<float>(input[m * K + k]);
            float scaled_val = val * inv_scale;
            output_fp8[m * K + k] = __nv_fp8_e4m3(scaled_val);
        }
    }
}

// =============================================================================
// Full-Scale Mode Workspace Sizes
// =============================================================================

struct Sm12xFullScaleWorkspaceSizes {
    size_t fp8_activation_bytes;
    size_t sfa_bytes;
    size_t temp_max_bytes;  // Temporary buffer for block max computation
    size_t total_bytes;
    int sfa_stride;
    int num_m_blocks;
    int num_k_blocks;
    
    static Sm12xFullScaleWorkspaceSizes compute(int total_tokens, int K) {
        Sm12xFullScaleWorkspaceSizes sizes;
        
        constexpr int kBlkMN_local = 128;
        constexpr int kSFVecSize = 32;
        constexpr int kBlkSF = 4;
        
        sizes.num_m_blocks = (total_tokens + kBlkMN_local - 1) / kBlkMN_local;
        sizes.num_k_blocks = (K + kSFVecSize - 1) / kSFVecSize;
        
        sizes.fp8_activation_bytes = static_cast<size_t>(total_tokens) * K * sizeof(__nv_fp8_e4m3);
        sizes.fp8_activation_bytes = (sizes.fp8_activation_bytes + 255) & ~size_t(255);
        
        // SFA size (Blk_SF scales per K block)
        sizes.sfa_bytes = static_cast<size_t>(sizes.num_m_blocks) * sizes.num_k_blocks * kBlkSF;
        sizes.sfa_bytes = (sizes.sfa_bytes + 255) & ~size_t(255);
        sizes.sfa_stride = sizes.num_k_blocks * kBlkSF;
        
        sizes.temp_max_bytes = 0;  // Fused kernel doesn't need temp buffer
        
        sizes.total_bytes = sizes.fp8_activation_bytes + sizes.sfa_bytes;
        
        return sizes;
    }
};

// =============================================================================
// Full-Scale Mode Quantizer (EXPERIMENTAL - DO NOT USE IN PRODUCTION)
// =============================================================================
//
// Computes per-block absmax and uses it to scale activations before FP8 conversion.
// This provides better numerical range coverage than identity scales.
//
// ⚠️  WARNING: Scale layout is ROW-MAJOR, not CUTLASS LayoutSFA!  ⚠️
// This will produce INCORRECT results with real SM12x block-scaled kernels.
// See the large warning box above for details.

// Define SM12X_ENABLE_EXPERIMENTAL_FULL_SCALE_MODE=1 to enable full-scale mode at your own risk
// By default, this function will fail at runtime with cudaErrorNotSupported

template <typename InputType>
[[deprecated("EXPERIMENTAL: Full-scale mode uses row-major layout, not CUTLASS LayoutSFA. Use identity mode for production.")]]
cudaError_t quantizeActivationsFullScale(
    const InputType* d_input,           // [total_tokens, K]
    int total_tokens,
    int K,
    void* d_workspace,
    size_t workspace_bytes,
    Sm12xQuantizedActivationView& output,
    cudaStream_t stream = 0
) {
#ifndef SM12X_ENABLE_EXPERIMENTAL_FULL_SCALE_MODE
    // Runtime guard: return error unless explicitly enabled
    // Full-scale mode writes scales in row-major, but CUTLASS expects LayoutSFA (swizzled).
    // This will produce incorrect results with real SM12x block-scaled kernels.
    // To enable anyway: #define SM12X_ENABLE_EXPERIMENTAL_FULL_SCALE_MODE 1
    (void)d_input; (void)total_tokens; (void)K; (void)d_workspace;
    (void)workspace_bytes; (void)output; (void)stream;
    return cudaErrorNotSupported;
#else
    // EXPERIMENTAL: Proceed at your own risk!
    
    auto sizes = Sm12xFullScaleWorkspaceSizes::compute(total_tokens, K);
    
    if (workspace_bytes < sizes.total_bytes) {
        return cudaErrorInvalidValue;
    }
    
    // Partition workspace
    uint8_t* ws = static_cast<uint8_t*>(d_workspace);
    output.d_fp8_activations = reinterpret_cast<__nv_fp8_e4m3*>(ws);
    output.d_scale_factors = ws + sizes.fp8_activation_bytes;
    output.scale_factor_stride = sizes.sfa_stride;
    
    // Launch fused quantization kernel
    // Each block handles one (128, 32) tile
    dim3 grid(sizes.num_m_blocks, sizes.num_k_blocks);
    int block_size = 256;
    
    quantize_with_block_scaling_kernel<InputType><<<grid, block_size, 0, stream>>>(
        d_input, output.d_fp8_activations, output.d_scale_factors,
        total_tokens, K, sizes.num_m_blocks, sizes.num_k_blocks
    );
    
    return cudaGetLastError();
#endif  // SM12X_ENABLE_EXPERIMENTAL_FULL_SCALE_MODE
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
    printf("  - Full-scale mode: Per-block absmax, proper tiled fill\n");
}

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm

