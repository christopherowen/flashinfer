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
// SM12x LayoutSFA Utilities
// =============================================================================
//
// This header provides utilities for computing the correct SFA (Scale Factor A)
// buffer size and layout using CUTLASS's Sm1xxBlockScaledConfig.
//
// CUTLASS's block-scaled layouts are complex tiled structures optimized for
// TMEM access. The layout is NOT simple row-major - it uses a 512B basic block
// structure with 128 M elements × 4 SF elements per block.
//
// This utility derives buffer sizes directly from CUTLASS's layout APIs to
// ensure correctness for TMA loads.
//
// Usage:
//   #include "sm12x_layout_sfa_utils.h"
//   size_t sfa_bytes = Sm12xLayoutSFAUtils::computeBufferSize(M, K, L);
//   // Or with kernel-derived layout:
//   size_t sfa_elements = cute::cosize(SfConfig::tile_atom_to_shape_SFA(problem_shape));
//
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <cassert>

#ifdef ENABLE_FP4
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/layout.hpp"
#endif

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

// =============================================================================
// SM12x Block-Scaled Configuration Constants
// =============================================================================

// Block scaling granularity for SM12x
// Each scale factor covers 128 elements along M/N dimension
static constexpr int kBlkMN_SM12x = 128;

// Scale factor vector size (K dimension grouping)
static constexpr int kSFVecSize_SM12x = 32;

// Number of scale factors per K-block in the swizzled layout
static constexpr int kBlkSF_SM12x = 4;

// =============================================================================
// LayoutSFA Size Computation Using CUTLASS APIs
// =============================================================================
//
// HOW LAYOUTSFA CAPACITY IS DERIVED:
// ----------------------------------
// CUTLASS's Sm1xxBlockScaledConfig::tile_atom_to_shape_SFA(problem_shape) creates
// a layout with the following structure:
//
//   SfAtom = Layout<Shape<(32,4), (SFVecSize,4)>, Stride<(16,4), (0,1)>>
//
// The tile_to_shape function tiles this atom across the (M,K,L) dimensions:
//   layout_sfa = tile_to_shape(SfAtom{}, make_shape(M,K,L), Step<_2,_1,_3>{})
//
// The resulting layout's cosize (codomain size) tells us the maximum linear
// index + 1, which equals the number of UE8M0 bytes needed.
//
// For identity scales (all 0x7F), the actual pattern doesn't matter because
// every byte has the same value. However, we MUST allocate exactly the number
// of bytes the kernel's TMA descriptor expects.
//
// WHERE LAYOUTSFA IS VERIFIED:
// ----------------------------
// In the launcher (moe_gemm_sm120_mixed_input_launcher.inl), after the kernel
// types are fully instantiated, we can extract:
//   using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
//
// This is the actual layout type the kernel uses. We verify that our computed
// buffer size is >= the cosize of the instantiated layout for the given problem
// shape in debug builds.
//

struct Sm12xLayoutSFAUtils {
    // Compute the number of UE8M0 bytes required for SFA buffer
    // This uses CUTLASS's layout APIs directly for correctness
    // Accepts full (M, N, K, L) problem shape as CUTLASS expects
    static size_t computeBufferSize(int M, int N, int K, int L = 1) {
#ifdef ENABLE_FP4
        // Use CUTLASS's Sm1xxBlockScaledConfig to compute the exact layout
        using SfConfig = cutlass::detail::Sm1xxBlockScaledConfig<kSFVecSize_SM12x>;
        
        // Create full problem shape (M, N, K, L)
        // CUTLASS's tile_atom_to_shape_SFA extracts (M, K, L) internally for SFA
        auto problem_shape = cute::make_shape(M, N, K, L);
        
        // Get the SFA layout for this problem shape
        auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(problem_shape);
        
        // cosize gives the maximum linear index + 1 (the required buffer size)
        size_t sfa_elements = cute::cosize(layout_sfa);
        
        // Each element is 1 byte (float_ue8m0_t)
        // Align to 256 bytes for TMA
        return (sfa_elements + 255) & ~size_t(255);
#else
        // Fallback calculation when FP4 is not enabled
        return computeBufferSizeFallback(M, K, L);
#endif
    }
    
    // Convenience overload for when N is not relevant (e.g., prewarm)
    // Uses N=0 which CUTLASS will ignore for SFA layout computation
    static size_t computeBufferSize(int M, int K, int L = 1) {
        return computeBufferSize(M, 0, K, L);
    }
    
    // Fallback computation without CUTLASS headers
    // This matches the CUTLASS layout structure based on documentation
    static size_t computeBufferSizeFallback(int M, int K, int L = 1) {
        // Number of 128-element blocks in M dimension
        int num_m_blocks = (M + kBlkMN_SM12x - 1) / kBlkMN_SM12x;
        
        // Number of 32-element blocks in K dimension
        int num_k_blocks = (K + kSFVecSize_SM12x - 1) / kSFVecSize_SM12x;
        
        // Each (m_block, k_block) pair needs 4 scale factors (kBlkSF_SM12x)
        // The atom structure is (32,4) x (SFVecSize,4) with specific strides
        // Total elements per atom = 32 * 4 = 128 bytes
        size_t sfa_elements = static_cast<size_t>(num_m_blocks) * num_k_blocks * 
                              kBlkSF_SM12x * L;
        
        // Multiply by atom factor (32 elements per block in M)
        sfa_elements *= 32;
        
        // Align to 256 bytes for TMA
        return (sfa_elements + 255) & ~size_t(255);
    }
    
    // Verify computed size against CUTLASS layout (debug only)
    // Returns true if sizes match (or within tolerance)
    template <typename LayoutSFA>
    static bool verifySize(size_t allocated_bytes, int M, int K, int L = 1) {
#ifdef ENABLE_FP4
        using SfConfig = cutlass::detail::Sm1xxBlockScaledConfig<kSFVecSize_SM12x>;
        auto problem_shape = cute::make_shape(M, 0, K, L);
        auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(problem_shape);
        size_t required = cute::cosize(layout_sfa);
        return allocated_bytes >= required;
#else
        return true;  // Cannot verify without CUTLASS
#endif
    }
    
    // Get a signature/key for caching purposes
    // Returns a unique value for (num_m_blocks, num_k_blocks, L)
    static uint64_t getCacheKey(int M, int K, int L = 1) {
        int num_m_blocks = (M + kBlkMN_SM12x - 1) / kBlkMN_SM12x;
        int num_k_blocks = (K + kSFVecSize_SM12x - 1) / kSFVecSize_SM12x;
        // Pack into uint64_t: high 16 bits = L, next 24 = m_blocks, low 24 = k_blocks
        return (static_cast<uint64_t>(L) << 48) |
               (static_cast<uint64_t>(num_m_blocks) << 24) |
               static_cast<uint64_t>(num_k_blocks);
    }
};

// =============================================================================
// Kernel-Derived LayoutSFA Size Computation
// =============================================================================
//
// This template function is meant to be called from the launcher where the
// GemmKernel type is fully instantiated. It extracts LayoutSFA from the kernel
// and computes the exact buffer size using the kernel's actual layout type.
//
// IMPORTANT: The CollectiveMainloop type exposes LayoutSFA which defines the
// exact memory layout expected by TMA. We use tile_atom_to_shape_SFA with the
// FULL problem shape (M, N, K, L) to get the runtime layout instance, then
// cute::cosize() gives the required buffer capacity.
//
// Usage in launcher:
//   size_t sfa_bytes = computeKernelSFABufferSize<CollectiveMainloop>(M, N, K, L);
//

#ifdef ENABLE_FP4

// Primary API: Compute SFA buffer size from kernel's CollectiveMainloop type
// Uses the full (M, N, K, L) problem shape as CUTLASS expects
template <typename CollectiveMainloop>
size_t computeKernelSFABufferSize(int M, int N, int K, int L = 1) {
    // Extract the SfConfig that CollectiveMainloop uses for block-scaled layouts
    // The CollectiveMainloop is built with ElementABlockScaled (e.g., mx_float8_t<float_e4m3_t>)
    // which dictates the scale factor layout through Sm1xxBlockScaledConfig
    //
    // The SFVecSize for MXFP types is 32 (MXFPXBlockScaleVectorSize in moe_gemm_kernels.h)
    using SfConfig = cutlass::detail::Sm1xxBlockScaledConfig<kSFVecSize_SM12x>;
    
    // Create the full problem shape (M, N, K, L)
    // CUTLASS's tile_atom_to_shape_SFA uses (M, N, K, L) and extracts (M, K, L) for SFA
    auto problem_shape = cute::make_shape(M, N, K, L);
    
    // Get the SFA layout for this problem shape
    // This is the SAME layout the kernel's TMA descriptor will expect
    auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(problem_shape);
    
    // cosize gives the maximum linear index + 1 (the required buffer capacity in bytes)
    size_t sfa_elements = cute::cosize(layout_sfa);
    
    // Align to 256 bytes for TMA requirements
    return (sfa_elements + 255) & ~size_t(255);
}

// Convenience overload that takes N from the problem (for cases where N is known)
template <typename CollectiveMainloop>
size_t computeKernelSFABufferSizeFromShape(int M_max, int N, int K, int L = 1) {
    return computeKernelSFABufferSize<CollectiveMainloop>(M_max, N, K, L);
}

// Debug assertion for SFA buffer size verification
// Call this from the launcher or higher-level code in debug builds
template <typename CollectiveMainloop>
void assertSFABufferSizeCorrect(size_t allocated_bytes, int M, int N, int K, int L,
                                 const char* kernel_name) {
#ifndef NDEBUG
    size_t required = computeKernelSFABufferSize<CollectiveMainloop>(M, N, K, L);
    if (allocated_bytes < required) {
        fprintf(stderr, 
                "ERROR: SFA buffer size mismatch for kernel '%s'\n"
                "  Problem shape: M=%d, N=%d, K=%d, L=%d\n"
                "  Required bytes (from CUTLASS cosize): %zu\n"
                "  Allocated bytes: %zu\n"
                "  Shortfall: %zu bytes\n",
                kernel_name, M, N, K, L, required, allocated_bytes, required - allocated_bytes);
        assert(allocated_bytes >= required && "SFA buffer too small for CUTLASS layout!");
    }
#endif
}

// Non-templated verification using fallback computation (for prewarm/init)
inline void assertSFABufferSizeCorrectFallback(size_t allocated_bytes, int M, int N, int K, int L,
                                                const char* context) {
#ifndef NDEBUG
    size_t required = Sm12xLayoutSFAUtils::computeBufferSize(M, N, K, L);
    if (allocated_bytes < required) {
        fprintf(stderr, 
                "ERROR: SFA buffer size mismatch in '%s'\n"
                "  Problem shape: M=%d, N=%d, K=%d, L=%d\n"
                "  Required bytes (from LayoutSFAUtils): %zu\n"
                "  Allocated bytes: %zu\n"
                "  Shortfall: %zu bytes\n",
                context, M, N, K, L, required, allocated_bytes, required - allocated_bytes);
        assert(allocated_bytes >= required && "SFA buffer too small!");
    }
#endif
}

// Overload without N for convenience
inline void assertSFABufferSizeCorrectFallback(size_t allocated_bytes, int M, int K, int L,
                                                const char* context) {
    assertSFABufferSizeCorrectFallback(allocated_bytes, M, 0, K, L, context);
}
#endif  // ENABLE_FP4

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm

