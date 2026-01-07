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
// SM12x (Blackwell Thorough) Architecture Configuration
// =============================================================================
//
// This header provides unified architecture detection for SM120 and SM121.
//
// CUTLASS Architecture Macros:
//   - CUTLASS_ARCH_MMA_SM120_SUPPORTED: Defined when __CUDA_ARCH__ == 1200
//   - CUTLASS_ARCH_MMA_SM121_SUPPORTED: Defined when __CUDA_ARCH__ == 1210
//
// Both SM120 and SM121 share the same CUTLASS arch tag (cutlass::arch::Sm120),
// but compilation and feature macros differ. When compiling for sm_121a,
// CUTLASS_ARCH_MMA_SM121_SUPPORTED is set but NOT CUTLASS_ARCH_MMA_SM120_SUPPORTED.
//
// This header defines a unified macro CUTLASS_ARCH_MMA_SM12x_SUPPORTED to detect
// either architecture, avoiding the need to check both macros everywhere.
//
// Usage:
//   #include "sm12x_arch_config.h"
//   #if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED)
//     // SM120 or SM121 code path
//   #endif
//
// =============================================================================

// Unified SM12x support detection
// Covers both SM120 (__CUDA_ARCH__==1200) and SM121 (__CUDA_ARCH__==1210)
#ifndef CUTLASS_ARCH_MMA_SM12x_SUPPORTED
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#define CUTLASS_ARCH_MMA_SM12x_SUPPORTED 1
#endif
#endif

// Convenience constexpr for template metaprogramming
namespace tensorrt_llm::kernels::cutlass_kernels {

// Check if SM12x (SM120 or SM121) is supported at compile time
constexpr bool kIsSm12xSupported =
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED)
    true;
#else
    false;
#endif

// Check if FP4 operations are enabled
constexpr bool kIsFp4Enabled =
#if defined(ENABLE_FP4)
    true;
#else
    false;
#endif

// Combined check for SM12x FP4 support
constexpr bool kIsSm12xFp4Supported = kIsSm12xSupported && kIsFp4Enabled;

}  // namespace tensorrt_llm::kernels::cutlass_kernels


