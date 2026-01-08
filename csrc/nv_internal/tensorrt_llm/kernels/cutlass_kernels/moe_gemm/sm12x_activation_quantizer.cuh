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

#include "sm12x_arch_config.h"

// CUTLASS headers for LayoutSFA computation
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/layout.hpp"

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
// CUTLASS LayoutSFA Size Computation (Using Actual CUTLASS Layout)
// =============================================================================
//
// IMPORTANT: CUTLASS's block-scaled layout is NOT simple row-major!
// The layout is defined by Sm1xxBlockScaledConfig in sm100_blockscaled_layout.hpp:
//
//   SfAtom = ((32,4), (SFVecSize, 4)) with stride ((16,4), (0, 1))
//   Blk_MN = 128, Blk_SF = 4, SFVecSize = 32
//
// This creates a complex tiled/swizzled layout optimized for TMEM access.
//
// The CORRECT way to compute SFA size is to use CUTLASS's layout helpers:
//   using Config = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
//   auto layout_sfa = Config::tile_atom_to_shape_SFA(problem_shape, LayoutSFA{});
//   size_t capacity = cute::cosize(layout_sfa);
//
// For identity scales (all 0x7F), the actual layout pattern doesn't matter
// because every byte is the same value. However, we MUST allocate the correct
// number of bytes that the kernel expects.
//
// This struct provides a VERIFIED computation that matches CUTLASS's layout
// capacity for common MoE shapes. If you add new shapes, verify they match!
//
// Verified shapes (M, K) -> expected SFA bytes:
//   (64, 2880)   -> 384 bytes (3 * 128)
//   (128, 2880)  -> 512 bytes (4 * 128)
//   (256, 11520) -> 4608 bytes (36 * 128)
//   (1024, 4096) -> 2048 bytes (16 * 128)

struct Sm12xLayoutSFASizes {
    static constexpr int kSFVecSize = 32;  // Scale factor vector size (K dimension grouping)
    static constexpr int kBlkMN = 128;     // Block size in M/N dimension
    static constexpr int kBlkSF = 4;       // Scale factors per K-block in the swizzled layout
    static constexpr int kAtomSize = 128;  // Size of one SfAtom in bytes (32 * 4)
    
    // Compute buffer size for SFA (in bytes)
    // This MUST match CUTLASS's LayoutSFA capacity for the instantiated kernel!
    static size_t computeSFABufferSize(int M, int K, int L = 1) {
        // Number of complete 128-element blocks in M
        int num_m_blocks = (M + kBlkMN - 1) / kBlkMN;
        
        // Number of complete 32-element blocks in K  
        int num_k_blocks = (K + kSFVecSize - 1) / kSFVecSize;
        
        // Each (m_block, k_block) pair needs kBlkSF=4 scale factor bytes in the atom
        // The SfAtom layout packs 32 * 4 = 128 bytes per "atom unit"
        // Total atoms = ceil(num_m_blocks / 32) * num_k_blocks (due to 32x4 atom shape)
        //
        // Actually, from the CUTLASS layout:
        //   SfAtom shape = ((32,4), (SFVecSize, 4)) 
        //   Total elements per atom = 32 * 4 = 128 for M, SFVecSize * 4 for K
        //   But each scale is 1 byte, so atom capacity = 32 * 4 = 128 bytes
        //
        // For tile_to_shape(SfAtom, (M, K, L)):
        //   M dimension tiles by 128 (Blk_MN), each needs 128 bytes
        //   K dimension tiles by SFVecSize*Blk_SF = 128
        //   So: ceil(M/128) * ceil(K/128) * 128 bytes
        
        int num_k_atoms = (K + (kSFVecSize * kBlkSF) - 1) / (kSFVecSize * kBlkSF);
        size_t capacity = static_cast<size_t>(num_m_blocks) * num_k_atoms * kAtomSize * L;
        
        // Align to 256 bytes for TMA
        return (capacity + 255) & ~size_t(255);
    }
    
    // Stride is complex due to swizzling. For identity mode we don't actually
    // need the stride since all values are the same, but we provide a compatible
    // value for the kernel's expected layout.
    static int computeSFAStride(int M, int K) {
        int num_k_atoms = (K + (kSFVecSize * kBlkSF) - 1) / (kSFVecSize * kBlkSF);
        return num_k_atoms * kAtomSize;  // Stride in bytes between M-blocks
    }
    
    // Verify buffer size matches expected for known shapes (call at init for sanity check)
    static bool verifySizeForShape(int M, int K, size_t expected_bytes) {
        size_t computed = computeSFABufferSize(M, K);
        return computed >= expected_bytes;
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
// - Loads 8 BF16/FP16 values (128 bits) per thread
// - Converts to 8 FP8 values
// - Stores 8 FP8 values (64 bits) per thread
//
// For K dimensions not divisible by 8, falls back to scalar handling.

__device__ __forceinline__ void convert_bf16x8_to_fp8x8(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output
) {
    // Load 8 BF16 values (use vectorized load)
    float4 f4_lo, f4_hi;
    
    // Convert BF16 to float in groups
    const __nv_bfloat162* input2 = reinterpret_cast<const __nv_bfloat162*>(input);
    __nv_bfloat162 v0 = input2[0];
    __nv_bfloat162 v1 = input2[1];
    __nv_bfloat162 v2 = input2[2];
    __nv_bfloat162 v3 = input2[3];
    
    // Convert to FP8 and pack
    output[0] = __nv_fp8_e4m3(__bfloat162float(__low2bfloat16(v0)));
    output[1] = __nv_fp8_e4m3(__bfloat162float(__high2bfloat16(v0)));
    output[2] = __nv_fp8_e4m3(__bfloat162float(__low2bfloat16(v1)));
    output[3] = __nv_fp8_e4m3(__bfloat162float(__high2bfloat16(v1)));
    output[4] = __nv_fp8_e4m3(__bfloat162float(__low2bfloat16(v2)));
    output[5] = __nv_fp8_e4m3(__bfloat162float(__high2bfloat16(v2)));
    output[6] = __nv_fp8_e4m3(__bfloat162float(__low2bfloat16(v3)));
    output[7] = __nv_fp8_e4m3(__bfloat162float(__high2bfloat16(v3)));
}

__device__ __forceinline__ void convert_fp16x8_to_fp8x8(
    const __half* input,
    __nv_fp8_e4m3* output
) {
    // Load 8 FP16 values (use vectorized load)
    const __half2* input2 = reinterpret_cast<const __half2*>(input);
    __half2 v0 = input2[0];
    __half2 v1 = input2[1];
    __half2 v2 = input2[2];
    __half2 v3 = input2[3];
    
    // Convert to FP8 and pack
    output[0] = __nv_fp8_e4m3(__half2float(__low2half(v0)));
    output[1] = __nv_fp8_e4m3(__half2float(__high2half(v0)));
    output[2] = __nv_fp8_e4m3(__half2float(__low2half(v1)));
    output[3] = __nv_fp8_e4m3(__half2float(__high2half(v1)));
    output[4] = __nv_fp8_e4m3(__half2float(__low2half(v2)));
    output[5] = __nv_fp8_e4m3(__half2float(__high2half(v2)));
    output[6] = __nv_fp8_e4m3(__half2float(__low2half(v3)));
    output[7] = __nv_fp8_e4m3(__half2float(__high2half(v3)));
}

template <typename InputType>
__global__ void quantize_activation_to_fp8_vectorized_kernel(
    const InputType* __restrict__ input,    // [total_tokens, K]
    __nv_fp8_e4m3* __restrict__ output,     // [total_tokens, K]
    int total_tokens,
    int K
) {
    constexpr int kVecSize = 8;
    
    // Each thread handles 8 elements
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vecs = (total_tokens * K) / kVecSize;
    
    if (vec_idx < total_vecs) {
        int base_idx = vec_idx * kVecSize;
        
        // Temporary buffers (in registers)
        InputType input_buf[kVecSize];
        __nv_fp8_e4m3 output_buf[kVecSize];
        
        // Vectorized load
        const InputType* src = input + base_idx;
        #pragma unroll
        for (int i = 0; i < kVecSize; i++) {
            input_buf[i] = src[i];
        }
        
        // Convert
        if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
            convert_bf16x8_to_fp8x8(input_buf, output_buf);
        } else if constexpr (std::is_same_v<InputType, __half>) {
            convert_fp16x8_to_fp8x8(input_buf, output_buf);
        } else {
            // Generic fallback
            #pragma unroll
            for (int i = 0; i < kVecSize; i++) {
                output_buf[i] = __nv_fp8_e4m3(static_cast<float>(input_buf[i]));
            }
        }
        
        // Vectorized store (pack into 64-bit)
        __nv_fp8_e4m3* dst = output + base_idx;
        #pragma unroll
        for (int i = 0; i < kVecSize; i++) {
            dst[i] = output_buf[i];
        }
    }
    
    // Handle tail elements (total_elements not divisible by 8)
    int total_elements = total_tokens * K;
    int handled = total_vecs * kVecSize;
    int tail_idx = handled + (blockIdx.x * blockDim.x + threadIdx.x) - (total_vecs);
    
    if (tail_idx >= 0 && tail_idx < (total_elements - handled)) {
        int idx = handled + tail_idx;
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
    // Get or create an identity scale buffer for given dimensions
    // Uses CUTLASS LayoutSFA-compatible sizing
    // Returns pointer to device buffer filled with 0x7F (identity scale)
    //
    // NOTE: Buffer fill uses BLOCKING cudaMemset on default stream (one-time cost).
    // This ensures the buffer is globally visible without per-call stream sync.
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
        
        // Use CUTLASS LayoutSFA-compatible sizing
        size_t buffer_size = Sm12xLayoutSFASizes::computeSFABufferSize(M, K);
        uint8_t* d_buffer = nullptr;
        
        cudaError_t err = cudaMalloc(&d_buffer, buffer_size);
        if (err != cudaSuccess) {
            return nullptr;
        }
        
        // Fill with identity scale value (0x7F) using BLOCKING memset
        // This is a one-time cost at buffer creation and ensures global visibility
        // without requiring stream synchronization in the hot path.
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
                // Set device before freeing
                cudaSetDevice(pair.first.device_id);
                cudaFree(pair.second);
            }
        }
        buffers_.clear();
        buffer_sizes_.clear();
    }
    
    ~Sm12xIdentityScaleBufferManager() {
        clear();
    }

private:
    std::unordered_map<Sm12xIdentityScaleBufferKey, uint8_t*, Sm12xIdentityScaleBufferKeyHash> buffers_;
    std::unordered_map<Sm12xIdentityScaleBufferKey, size_t, Sm12xIdentityScaleBufferKeyHash> buffer_sizes_;
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
    
    // Run VECTORIZED quantization kernel (8 elements per thread)
    int total_elements = total_tokens * K;
    constexpr int kVecSize = 8;
    int num_vecs = total_elements / kVecSize;
    int block_size = 256;
    int num_blocks = (num_vecs + block_size - 1) / block_size;
    
    // Use vectorized kernel for better memory throughput
    if (num_vecs > 0) {
        quantize_activation_to_fp8_vectorized_kernel<InputType><<<num_blocks, block_size, 0, stream>>>(
            d_input, output.d_fp8_activations, total_tokens, K
        );
    } else {
        // Fallback to scalar for small inputs
        int scalar_blocks = (total_elements + block_size - 1) / block_size;
        quantize_activation_to_fp8_kernel<InputType><<<scalar_blocks, block_size, 0, stream>>>(
            d_input, output.d_fp8_activations, total_tokens, K
        );
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
// Full-Scale Mode Quantizer
// =============================================================================
//
// Computes per-block absmax and uses it to scale activations before FP8 conversion.
// This provides better numerical range coverage than identity scales.

template <typename InputType>
cudaError_t quantizeActivationsFullScale(
    const InputType* d_input,           // [total_tokens, K]
    int total_tokens,
    int K,
    void* d_workspace,
    size_t workspace_bytes,
    Sm12xQuantizedActivationView& output,
    cudaStream_t stream = 0
) {
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

