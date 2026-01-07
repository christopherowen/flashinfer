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

#include <cuda_runtime_api.h>

#include "../../include/moe_gemm_kernels.h"
#include "cutlass_extensions/gemm_configs.h"
#include "cutlass_extensions/weight_only_quant_op.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels_oss {

using tensorrt_llm::kernels::cutlass_kernels::GroupedGemmInput;
using tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;

// SM120 Mixed-Input Grouped GEMM Launcher
// This launcher supports mixed-precision operations on SM120/SM121 (Blackwell) architecture:
// - FP8 activations x FP4 weights (FP8xFP4)
// - BF16/FP16 activations x FP4 weights (MXFP4/W4A16) - uses FP8xFP4 path with quantized activations
//
// Key features:
// - Uses block-scaled collective builder for SM120/SM121
// - Supports grouped GEMM for MoE workloads
// - Integrates with CUTLASS SM120 block-scaled infrastructure
//
// IMPORTANT: A-side scaling factors are mandatory for SM120 block-scaled kernels.
// The SM120 mma.mx_f4f6f8 instructions only support FP8/FP6/FP4 inputs, so BF16/FP16
// activations must be pre-quantized to FP8 before calling this launcher.
//
// For MXFP4 (W4A16) workloads where you want to preserve activation accuracy:
//   - Pre-quantize BF16/FP16 -> FP8 (float_e4m3_t)
//   - Use IDENTITY A-SCALES to avoid accuracy loss from block scaling
//
// Identity Scale Implementation:
//   - Scale factor type: float_ue8m0_t (unsigned 8-bit exponent, uses FP32 bias=127)
//   - Identity scale value: raw byte = 0x7F (127) → 2^(127-127) = 2^0 = 1.0
//   - Broadcast trick: allocate single identity scale, use stride=0 to broadcast
//
// Example identity scale setup:
//   #include "sm12x_arch_config.h"
//   constexpr uint8_t IDENTITY_SCALE_RAW = kSm12xIdentityScaleRaw;  // 0x7F = 1.0
//   float_ue8m0_t identity = float_ue8m0_t::bitcast(IDENTITY_SCALE_RAW);
//   // Allocate one element, pass with stride=0 for broadcast
//
// WARNING: CUTLASS header says "exp_bias: 8" but convert_to_float uses FP32's
// bias of 127. Always use 0x7F for identity, NOT 8!
//
template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          typename CTAShape, typename ClusterShape, bool IsMXFP4 = false>
void sm120_mixed_input_moe_gemm_kernelLauncher(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);

}  // namespace cutlass_kernels_oss
}  // namespace kernels
}  // namespace tensorrt_llm

