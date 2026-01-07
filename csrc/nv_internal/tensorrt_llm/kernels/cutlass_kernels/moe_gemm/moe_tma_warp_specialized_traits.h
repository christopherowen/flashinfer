/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "../include/moe_gemm_kernels.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass_extensions/epilogue_helpers.h"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

namespace tensorrt_llm::kernels::cutlass_kernels {

// SM120/SM121 (Blackwell Thorough) arch
// Supports:
// - NVFP4: FP4 x FP4 (same type)
// - FP8xFP4: FP8 activations x FP4 weights
// - MXFP4: BF16/FP16 activations x FP4 weights (via FP8xFP4 path with activation quantization)
//
// Note: MXFP4 (BF16/FP16 x FP4) is supported on SM120/SM121 through a workaround where
// activations are represented as a tuple with identity scales. This satisfies the SM120
// block-scaled builder requirement that both operands have the same scale factor type.
template <typename T, typename WeightType,
          typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion =
              TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidSM120MOESpecialisation() {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#if defined(ENABLE_FP4)
  // NVFP4: FP4 x FP4 (same type)
  constexpr bool IsNVFP4 = cutlass::platform::is_same<T, __nv_fp4_e2m1>::value &&
                           cutlass::platform::is_same<T, WeightType>::value;
  // FP8xFP4: FP8 activations x FP4 weights  
  constexpr bool IsFP8xFP4 = cutlass::platform::is_same<T, __nv_fp8_e4m3>::value &&
                             cutlass::platform::is_same<WeightType, __nv_fp4_e2m1>::value;
  // MXFP4 (W4A16): BF16/FP16 activations x FP4 weights
  // Uses FP8xFP4 infrastructure with activation quantization
  constexpr bool IsMXFP4_BF16 = cutlass::platform::is_same<T, __nv_bfloat16>::value &&
                                cutlass::platform::is_same<WeightType, __nv_fp4_e2m1>::value;
  constexpr bool IsMXFP4_FP16 = cutlass::platform::is_same<T, half>::value &&
                                cutlass::platform::is_same<WeightType, __nv_fp4_e2m1>::value;
  constexpr bool IsMXFP4 = IsMXFP4_BF16 || IsMXFP4_FP16;
  
  return (IsNVFP4 || IsFP8xFP4 || IsMXFP4) &&
         cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value;
#else
  return false;
#endif
#else
  return false;  // CUTLASS_ARCH_MMA_SM120_SUPPORTED is set when SM120 kernels are enabled
#endif
}

// Helper to check if a configuration is MXFP4 (W4A16) on SM120
template <typename T, typename WeightType>
constexpr bool isSM120MXFP4() {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) && defined(ENABLE_FP4)
  return (cutlass::platform::is_same<T, __nv_bfloat16>::value ||
          cutlass::platform::is_same<T, half>::value) &&
         cutlass::platform::is_same<WeightType, __nv_fp4_e2m1>::value;
#else
  return false;
#endif
}

template <typename T, typename WeightType,
          typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion =
              TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidBlackwellMOESpecialisation() {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)  // TODO Is there a better choice
#if defined(ENABLE_FP4)
  return (cutlass::platform::is_same<T, WeightType>::value ||
#if defined(ENABLE_FP4)
          (cutlass::platform::is_same<T, __nv_fp8_e4m3>::value &&
           cutlass::platform::is_same<WeightType, __nv_fp4_e2m1>::value)
#else
          false
#endif
              ) &&
         cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value;
#else
  return cutlass::platform::is_same<T, WeightType>::value &&
         cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value;
#endif
#else
  return false;  // CUTLASS_ARCH_MMA_SM100_SUPPORTED is set when Blackwell kernels are enabled
#endif
}

// Hopper arch
template <typename T, typename WeightType,
          typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion =
              TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidHopperMOESpecialisation() {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  return (cutlass::platform::is_same<T, WeightType>::value ||
          (cutlass::platform::is_same<cutlass::uint4b_t, WeightType>::value &&
           cutlass::platform::is_same<T, __nv_fp8_e4m3>::value)
#ifdef ENABLE_FP4
          || (cutlass::platform::is_same<__nv_fp4_e2m1, WeightType>::value &&
              !cutlass::platform::is_same<T, __nv_fp8_e4m3>::value)
#endif
              )
#ifdef ENABLE_FP4
         && !cutlass::platform::is_same<T, __nv_fp4_e2m1>::value
#endif
         && cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value;
#else
  return false;  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED is set when Hopper kernels are
                 // enabled
#endif
}

template <typename T, typename WeightType,
          typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion =
              TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidTmaWarpSpecializedMOESpecialisation() {
  // Check at least one of the implementations are valid
  return isValidSM120MOESpecialisation<T, WeightType>() ||
         isValidBlackwellMOESpecialisation<T, WeightType, EpilogueTag, Fusion>() ||
         isValidHopperMOESpecialisation<T, WeightType, EpilogueTag, Fusion>();
}

// Hopper arch
template <typename T, typename WeightType,
          typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
          TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion =
              TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidAmpereMOESpecialisation() {
#ifdef ENABLE_FP4
  return !std::is_same_v<T, __nv_fp4_e2m1> && !std::is_same_v<WeightType, __nv_fp4_e2m1>;
#else
  return true;  // Default to true
#endif
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels
