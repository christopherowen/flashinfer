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

// Include FP4/FP8 type headers (guarded by feature flags)
#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

// =============================================================================
// SM12x (Blackwell Thorough) Architecture Configuration
// =============================================================================
//
// This header provides unified architecture detection for SM120 and SM121.
//
// CUTLASS defines separate macros for each SM version:
//   - CUTLASS_ARCH_MMA_SM120_SUPPORTED: When __CUDA_ARCH__ == 1200
//   - CUTLASS_ARCH_MMA_SM121_SUPPORTED: When __CUDA_ARCH__ == 1210
//
// Both SM120 and SM121 share the same CUTLASS arch tag (cutlass::arch::Sm120),
// but when compiling for sm_121a, only CUTLASS_ARCH_MMA_SM121_SUPPORTED is set.
//
// This header defines CUTLASS_ARCH_MMA_SM12x_SUPPORTED to detect either.
//
// =============================================================================

// Unified SM12x support detection (SM120 or SM121)
#ifndef CUTLASS_ARCH_MMA_SM12x_SUPPORTED
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#define CUTLASS_ARCH_MMA_SM12x_SUPPORTED 1
#endif
#endif

namespace tensorrt_llm::kernels::cutlass_kernels {

// =============================================================================
// Compile-time feature detection
// =============================================================================

constexpr bool kIsSm12xSupported =
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED)
    true;
#else
    false;
#endif

constexpr bool kIsFp4Enabled =
#if defined(ENABLE_FP4)
    true;
#else
    false;
#endif

constexpr bool kIsSm12xFp4Supported = kIsSm12xSupported && kIsFp4Enabled;

// =============================================================================
// SM12x Block-Scaled Constants
// =============================================================================
//
// SM12x block-scaled MMA uses float_ue8m0_t for scale factors:
//   value = 2^(storage - 127)  [FP32 exponent bias]
//
// Identity scale (1.0): storage = 127 = 0x7F → 2^0 = 1.0
//
constexpr uint8_t kSm12xIdentityScaleRaw = 0x7F;

// Block scaling granularity: 128 elements per scale factor
constexpr int kSm12xBlockScaleGranularity = 128;

// =============================================================================
// SM12x K-bytes → K-elements Conversion (Single Source of Truth)
// =============================================================================
//
// SM12x tile configs specify K in bytes (e.g., CtaShape128x128x128B = 128 bytes).
// CUTLASS TileShape expects K in elements.
//
// Conversion uses ACTIVATION TYPE T (matching existing SHAPE_CASE convention):
//   K_elements = K_bytes * 8 / sizeof_bits<T>
//
// Examples:
//   NVFP4 (T=FP4):  128B → (128*8)/4 = 256 elements
//   FP8×FP4 (T=FP8): 128B → (128*8)/8 = 128 elements
//
// This helper is used by BOTH dispatch and launcher to ensure they match.
//
// Usage:
//   dispatch: constexpr int K = Sm12xKBytesToElements<T, 128>::value;
//   launcher: constexpr int K = Sm12xKBytesToElements<T, CTA_K_>::value;
//

// Primary template: use cutlass::sizeof_bits for standard types
template <typename T, int KBytes>
struct Sm12xKBytesToElements {
  // K_elements = K_bytes * 8 / bits_per_element
  static constexpr int value = KBytes;  // Fallback for unknown types
};

#if defined(ENABLE_FP4)
// Specialization for FP4 activation: 4 bits per element
template <int KBytes>
struct Sm12xKBytesToElements<__nv_fp4_e2m1, KBytes> {
  static constexpr int value = KBytes * 2;  // 8 bits/byte ÷ 4 bits/elem
};
#endif

#if defined(ENABLE_FP8)
// Specialization for FP8 activation: 8 bits per element
template <int KBytes>
struct Sm12xKBytesToElements<__nv_fp8_e4m3, KBytes> {
  static constexpr int value = KBytes;  // 8 bits/byte ÷ 8 bits/elem = 1
};
#endif

// Convenience constexpr function wrapper
template <typename T, int KBytes>
constexpr int sm12xKBytesToElements() {
  return Sm12xKBytesToElements<T, KBytes>::value;
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels
