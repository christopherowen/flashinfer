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

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "moe_gemm_sm120_mixed_input_launcher.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels_oss {

using namespace tensorrt_llm::kernels::cutlass_kernels;
namespace tk = tensorrt_llm::common;

using namespace cute;

// SM120/SM121 support detection
// Both SM120 and SM121 use cutlass::arch::Sm120, but have separate macros
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#define CUTLASS_ARCH_MMA_SM12x_SUPPORTED_LAUNCHER 1
#endif

// SM120/SM121 Mixed-Input MOE GEMM Kernel Launcher Implementation
//
// This launcher implements mixed-precision grouped GEMM for SM120/SM121 (Blackwell Thorough)
// architecture. It leverages CUTLASS's block-scaled collective builder which provides
// native support for:
// - NVFP4 x NVFP4 with block scaling (nv_float4_t types)
// - MXF8 x MXF4 with block scaling (mx_float8_t and mx_float4_t types)
//
// For MXFP4 workloads (BF16/FP16 activations with FP4 weights), this launcher
// uses the FP8xFP4 path where activations are pre-quantized to FP8.
//
// The kernel supports grouped GEMM through PtrArray interfaces, making it suitable
// for Mixture-of-Experts (MoE) inference workloads.
//
// NOTE: Both SM120 (__CUDA_ARCH__==1200) and SM121 (__CUDA_ARCH__==1210) can use
// this launcher. When compiling for SM121, CUTLASS_ARCH_MMA_SM121_SUPPORTED is set.
template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          typename CTAShape, typename ClusterShape, bool IsMXFP4>
void sm120_mixed_input_moe_gemm_kernelLauncher(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED_LAUNCHER) && defined(ENABLE_FP4)
  /////////////////////////////////////////////////////////////////////////////
  // Type definitions
  /////////////////////////////////////////////////////////////////////////////

  // Input element types (from user-facing API)
  using ElementAInput = typename TllmToCutlassTypeAdapter<T>::type;
  using ElementB = typename TllmToCutlassTypeAdapter<WeightType>::type;
  
  // Scale factor type for block-scaled operations (UE8M0 = unsigned 8-bit exponent)
  // Identity scale: raw value 8 represents 2^(8-8) = 1.0
  using ElementSF = cutlass::float_ue8m0_t;
  
  // For SM120 block-scaled MMA, the hardware only supports FP8/FP6/FP4 inputs.
  // When IsMXFP4=true (BF16/FP16 activations with FP4 weights), activations must
  // be pre-quantized to FP8 before the GEMM. The caller is responsible for:
  //   1. Quantizing BF16/FP16 -> FP8 (float_e4m3_t)
  //   2. Providing A-scale pointers (can be identity scales = 1.0 to avoid accuracy loss)
  //
  // For identity A-scales with broadcastable stride tricks:
  //   - Allocate single float_ue8m0_t with raw value 8 (2^0 = 1.0)
  //   - Set stride to 0 to broadcast across all blocks
  //
  // ElementA for the kernel is always FP8 when using block-scaled path.
  // The IsMXFP4 flag indicates the original input was BF16/FP16 (for dispatch).
  using ElementA = cutlass::float_e4m3_t;
  
  // For SM120, we use the mx_float types which wrap the data + scale factor
  using ElementABlockScaled = cutlass::mx_float8_t<ElementA>;
  using ElementBBlockScaled = cutlass::mx_float4_t<ElementB>;

  // Layout configuration (TN layout - A is row-major, B is column-major)
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  
  // Transposed layouts for internal operations
  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  // Strides for grouped GEMM (pointer-based)
  using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

  // Alignment requirements (128-bit aligned)
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int AlignmentB = 128;  // For FP4 weights, use larger alignment

  // Output element types
  using ElementC = typename TllmToCutlassTypeAdapter<GemmOutputType>::type;
  using ElementD = ElementC;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Accumulator type
  using ElementAccumulator = float;

  // Architecture and operation class
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Tile and cluster shapes
  using TileShape = CTAShape;

  /////////////////////////////////////////////////////////////////////////////
  // Kernel schedule and dispatch policy
  /////////////////////////////////////////////////////////////////////////////
  
  // Use cooperative schedule for better performance on grouped GEMM
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

  // Stage count - auto-computed based on available shared memory
  using StageCountType = cutlass::gemm::collective::StageCountAuto;

  /////////////////////////////////////////////////////////////////////////////
  // Collective epilogue
  /////////////////////////////////////////////////////////////////////////////
  
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC,
      ElementD, LayoutD*, AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  /////////////////////////////////////////////////////////////////////////////
  // Collective mainloop
  /////////////////////////////////////////////////////////////////////////////

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementABlockScaled, LayoutA*, AlignmentA,
      ElementBBlockScaled, LayoutB*, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  /////////////////////////////////////////////////////////////////////////////
  // GEMM kernel and adapter
  /////////////////////////////////////////////////////////////////////////////

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      TmaWarpSpecializedGroupedGemmInput::ProblemShape,
      CollectiveMainloop, CollectiveEpilogue, void, void>;

  using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
  using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename CollectiveMainloop::LayoutSFB;

  GemmGrouped gemm;
  using Args = typename GemmGrouped::Arguments;
  Args arguments;

  /////////////////////////////////////////////////////////////////////////////
  // Set up epilogue arguments
  /////////////////////////////////////////////////////////////////////////////
  
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;

  /////////////////////////////////////////////////////////////////////////////
  // Set up hardware info
  /////////////////////////////////////////////////////////////////////////////
  
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count_;

  /////////////////////////////////////////////////////////////////////////////
  // Set up mainloop arguments
  /////////////////////////////////////////////////////////////////////////////
  //
  // A-SCALE HANDLING:
  // The SM120 block-scaled kernel requires scale factors for both A and B matrices.
  // For MXFP4 (W4A16) workloads, you have two options for A-scales:
  //
  // 1. IDENTITY SCALES (recommended for accuracy):
  //    - Pass ptr to single float_ue8m0_t with value 8 (2^0 = 1.0)
  //    - Use stride layout with zeros to broadcast
  //    - This preserves quantized FP8 values without scaling
  //
  // 2. PROPER A-SCALES (for dynamic range handling):
  //    - Compute per-block max and use for scaling
  //    - Better for activations with wide dynamic range
  //    - Requires scale computation kernel before GEMM
  //
  // The caller provides A-scales via hopper_inputs.fpX_block_scaling_factors_act
  // and the layout via hopper_inputs.fpX_block_scaling_factors_stride_act.
  //
  typename CollectiveMainloop::Arguments mainloop_args{
      reinterpret_cast<ElementA const**>(hopper_inputs.ptr_act),       // Pre-quantized FP8 activations
      reinterpret_cast<StrideA*>(hopper_inputs.stride_act),
      reinterpret_cast<ElementB const**>(hopper_inputs.ptr_weight),    // FP4 weights
      reinterpret_cast<StrideB*>(hopper_inputs.stride_weight),
      reinterpret_cast<ElementSF const**>(hopper_inputs.fpX_block_scaling_factors_act),   // A-scales (can be identity)
      reinterpret_cast<LayoutSFA*>(hopper_inputs.fpX_block_scaling_factors_stride_act),
      reinterpret_cast<ElementSF const**>(hopper_inputs.fpX_block_scaling_factors_weight), // B-scales (weight scales)
      reinterpret_cast<LayoutSFB*>(hopper_inputs.fpX_block_scaling_factors_stride_weight)};

  /////////////////////////////////////////////////////////////////////////////
  // Set up epilogue arguments
  /////////////////////////////////////////////////////////////////////////////
  
  typename CollectiveEpilogue::Arguments epilogue_args{
      fusion_args,
      nullptr,  // C ptr
      nullptr,  // C stride
      reinterpret_cast<ElementD**>(hopper_inputs.ptr_d),
      reinterpret_cast<StrideD*>(hopper_inputs.stride_d)};

  /////////////////////////////////////////////////////////////////////////////
  // Construct full arguments
  /////////////////////////////////////////////////////////////////////////////
  
  arguments = Args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      hopper_inputs.shape_info,
      mainloop_args,
      epilogue_args,
      hw_info};

  /////////////////////////////////////////////////////////////////////////////
  // Handle workspace size query
  /////////////////////////////////////////////////////////////////////////////
  
  if (workspace_size != nullptr) {
    *workspace_size = gemm.get_workspace_size(arguments);
    return;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Validate and execute
  /////////////////////////////////////////////////////////////////////////////
  
  if (gemm.get_workspace_size(arguments) > hopper_inputs.gemm_workspace_size) {
    TLLM_LOG_ERROR(
        "[SM120 Mixed-Input Grouped GEMM] Insufficient workspace: required %zu, allocated %zu",
        gemm.get_workspace_size(arguments), hopper_inputs.gemm_workspace_size);
    return;
  }

  auto can_implement = gemm.can_implement(arguments);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg =
        "SM120 mixed-input grouped GEMM cannot be implemented. Error: " +
        std::string(cutlassGetStatusString(can_implement));
    TLLM_LOG_ERROR("[SM120 Mixed-Input Grouped GEMM] %s", err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  auto init_status = gemm.initialize(arguments, hopper_inputs.gemm_workspace, inputs.stream);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "Failed to initialize SM120 mixed-input grouped GEMM. Error: " +
        std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error(err_msg);
  }

  auto run_status = gemm.run(inputs.stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run SM120 mixed-input grouped GEMM. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error(err_msg);
  }

#else
  TLLM_THROW("SM120/SM121 mixed-input GEMM requires CUTLASS_ARCH_MMA_SM12x_SUPPORTED (SM120 or SM121) and ENABLE_FP4");
#endif  // CUTLASS_ARCH_MMA_SM12x_SUPPORTED_LAUNCHER && ENABLE_FP4
}

}  // namespace cutlass_kernels_oss
}  // namespace kernels
}  // namespace tensorrt_llm

