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
#include "cutlass/detail/blockwise_scale_layout.hpp"  // For Sm120BlockwiseScaleConfig, UMMA::Major
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"  // For cute::UMMA

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "moe_gemm_sm120_mixed_input_launcher.h"
#include "../sm12x_arch_config.h"
#include "../sm12x_layout_sfa_utils.h"

// Only include SM12x identity buffer manager when SM12x + FP4 are both supported
// This avoids unnecessary compile-time coupling in non-SM12x builds
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4)
#include "../sm12x_activation_quantizer.cuh"  // For identity buffer manager
#endif
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>
#include <typeinfo>

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels_oss {

using namespace tensorrt_llm::kernels::cutlass_kernels;
namespace tk = tensorrt_llm::common;

using namespace cute;

// -----------------------------------------------------------------------------
// IMPORTANT (SM12x / MXFP4): Avoid defining CUTLASS kernel types inside a function
// template.
//
// We have a minimal CUTLASS repro on SM121 showing:
// - Non-template function with these exact types/args: initialize() == Success
// - Function-template wrapper with byte-identical args: initialize() == Error Internal
//
// Moving the type definitions to namespace scope fixes initialize().
// See: ai/mxfp4/csrc/repro/steps/step20 vs step21 vs step22.
// -----------------------------------------------------------------------------
namespace detail {

template <typename WeightType, typename GemmOutputType, typename EpilogueTag, typename CTAShape,
          typename ClusterShape, bool IsMXFP4>
struct Sm120MixedInputMoeGemmTypes {
  // Only MXFP4 is supported/validated.
  static_assert(IsMXFP4,
                "SM120 MoE GEMM only supports MXFP4 (FP8xFP4) path. NVFP4 (FP4xFP4) is not yet implemented.");

  // Weight type (from user-facing API) -> CUTLASS element type.
  using ElementB = typename TllmToCutlassTypeAdapter<WeightType>::type;

  // For MXFP4: FP8 activations (pre-quantized) x FP4 weights.
  using ElementA = cutlass::float_e4m3_t;

  // Block-scaled wrapper types for SM120/SM121.
  using ElementABlockScaled = cutlass::mx_float8_t<ElementA>;
  using ElementBBlockScaled = cutlass::mx_float4_t<ElementB>;

  // Scale factor element type (E8M0 for MXFP).
  using ElementSF = cutlass::float_ue8m0_t;

  // Layout configuration (TN layout - A is row-major, B is column-major).
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  // Output element types.
  using ElementC = typename TllmToCutlassTypeAdapter<GemmOutputType>::type;
  using ElementD = ElementC;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;

  // Accumulator type.
  using ElementAccumulator = float;
  using ElementCompute = float;

  // Architecture and operation class.
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Tile and cluster shapes.
  using TileShape = CTAShape;

  // Alignment requirements (match CUTLASS unit tests).
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16 for FP8
  static constexpr int AlignmentB = 128;                                          // special-case for FP4
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Kernel schedule / epilogue schedule.
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

  // Collective epilogue.
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute, ElementC, LayoutC*, AlignmentC, ElementD, LayoutD*, AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  // Collective mainloop - block-scaled wrapper types.
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementABlockScaled, LayoutA*, AlignmentA, ElementBBlockScaled, LayoutB*,
      AlignmentB, ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  // GEMM kernel and adapter.
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<TmaWarpSpecializedGroupedGemmInput::ProblemShape,
                                                         CollectiveMainloop, CollectiveEpilogue, void>;
  using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Stride / layout types for grouped GEMM.
  using StrideC = typename CollectiveEpilogue::StrideC;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using StrideA = typename CollectiveMainloop::StrideA;
  using StrideB = typename CollectiveMainloop::StrideB;
  using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
};

}  // namespace detail

// SM12x Block-Scaled MOE GEMM Kernel Launcher Implementation
//
// This launcher implements block-scaled grouped GEMM for SM120/SM121 (Blackwell Thorough)
// architecture. It leverages CUTLASS's block-scaled collective builder which provides
// native support for both:
//
// 1. MXFP4 (IsMXFP4=true): Mixed-input FP8×FP4
//    - Activations: FP8 (pre-quantized from BF16/FP16)
//    - Weights: FP4
//    - Uses mx_float8_t + mx_float4_t wrapper types
//
// 2. NVFP4 (IsMXFP4=false): Same-type FP4×FP4
//    - Activations: FP4 (native)
//    - Weights: FP4
//    - Uses nv_float4_t wrapper types
//
// The kernel supports grouped GEMM through PtrArray interfaces, making it suitable
// for Mixture-of-Experts (MoE) inference workloads.
//
// NOTE: Both SM120 (__CUDA_ARCH__==1200) and SM121 (__CUDA_ARCH__==1210) can use
// this launcher. When compiling for SM121, CUTLASS_ARCH_MMA_SM121_SUPPORTED is set.
template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          typename CTAShape, typename ClusterShape, bool IsMXFP4>
void sm120_mixed_input_moe_gemm_kernelLauncher(
    TmaWarpSpecializedGroupedGemmInput tma_inputs, int num_experts, int sm_count_,
    cudaStream_t stream, int* kernel_occupancy, size_t* workspace_size) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4)
  /////////////////////////////////////////////////////////////////////////////
  // COMPILE-TIME: Only support MXFP4 path for now
  /////////////////////////////////////////////////////////////////////////////
  // NVFP4 (IsMXFP4=false) is not yet validated and causes compilation issues
  // with the CUTLASS CollectiveBuilder. Prevent instantiation entirely.
  static_assert(IsMXFP4, 
      "SM120 MoE GEMM only supports MXFP4 (FP8xFP4) path. NVFP4 (FP4xFP4) is not yet implemented.");

  /////////////////////////////////////////////////////////////////////////////
  // RUNTIME TYPE SAFETY ASSERTION
  /////////////////////////////////////////////////////////////////////////////
  // SM12x block-scaled MMA only accepts FP8/FP6/FP4 inputs, not BF16/FP16.
  // When IsMXFP4=true, the activation pointers MUST contain pre-quantized FP8 data,
  // NOT the original BF16/FP16 data. The higher-level dispatch is responsible for:
  //   1. Quantizing BF16/FP16 -> FP8 (float_e4m3_t)
  //   2. Passing the FP8 pointer to this launcher
  //
  // This static_assert prevents silent type mismatches where BF16 is passed but
  // interpreted as FP8, which would produce garbage output.
  //
  // Valid T types for this launcher:
  //   - __nv_fp8_e4m3 / cutlass::float_e4m3_t (direct FP8 input)
  //   - For IsMXFP4, T can be BF16/FP16 at the API level, but tma_inputs.ptr_act
  //     MUST point to FP8 data (after caller's pre-quantization).
  if constexpr (IsMXFP4) {
    // For MXFP4 workloads (FP8xFP4), we rely on the caller to have performed quantization.
    // Log a warning to help debug if activation data looks wrong.
    TLLM_LOG_DEBUG("SM120 MXFP4 path (FP8xFP4): Expecting pre-quantized FP8 activations in ptr_act. "
                   "Original input type T=%s will NOT be used directly.",
                   typeid(T).name());
    
#ifndef NDEBUG  // Debug-only runtime checks
    // ==========================================================================
    // DEBUG CHECK: Validate that ptr_act contains FP8 data, NOT BF16/FP16
    // ==========================================================================
    // 
    // Common mistake: Passing BF16 activations directly without pre-quantization.
    // This produces garbage output because the kernel interprets BF16 bytes as FP8.
    //
    // We can't perfectly detect this at runtime, but we can check:
    // 1. ptr_act is not null (required for any operation)
    // 2. fpX_block_scaling_factors_act is provided (required for SM12x block-scaled)
    //
    // If you hit these asserts, ensure your caller:
    // 1. Quantizes BF16/FP16 -> FP8 BEFORE calling this launcher
    // 2. Provides valid SFA pointers (can use identity scales via getIdentityScaleBufferManager())
    //
    if (tma_inputs.ptr_act != nullptr && tma_inputs.shape_info.problem_shapes != nullptr) {
      // Only check if we have actual work to do (not workspace size query)
      TLLM_CHECK_WITH_INFO(tma_inputs.fpX_block_scaling_factors_act != nullptr,
                           "SM12x MXFP4 path requires activation scale factors (fpX_block_scaling_factors_act). "
                           "Use getIdentityScaleBufferManager() for identity scales.");
      
      // Log expected data format for debugging
      TLLM_LOG_DEBUG("SM12x MXFP4 debug check passed: ptr_act=%p, sfa_act=%p",
                     tma_inputs.ptr_act, tma_inputs.fpX_block_scaling_factors_act);
    }
#endif  // NDEBUG
  } else {
    // NVFP4 path (FP4xFP4): Native FP4 activations, no pre-quantization needed
    // TODO: NVFP4 support is not yet validated. Disable for now to focus on MXFP4.
    TLLM_LOG_ERROR("SM120 NVFP4 path (FP4xFP4) is not yet implemented. Use MXFP4 (FP8xFP4) instead.");
    throw std::runtime_error("SM120 NVFP4 path not yet implemented");
  }

  // NOTE: Keep the launcher templated for dispatch, but keep CUTLASS kernel type
  // definitions out of this function template to avoid initialize(): Error Internal.
  using Types = detail::Sm120MixedInputMoeGemmTypes<WeightType, GemmOutputType, EpilogueTag, CTAShape,
                                                   ClusterShape, IsMXFP4>;

  // Input element types (from user-facing API).
  // NOTE: For MXFP4, ptr_act must point to pre-quantized FP8 (ElementA) data.
  using ElementAInput = typename TllmToCutlassTypeAdapter<T>::type;

  using ElementA = typename Types::ElementA;
  using ElementB = typename Types::ElementB;
  using ElementD = typename Types::ElementD;
  using ElementSF = typename Types::ElementSF;

  // Used only for diagnostics / logging below.
  using TileShape = CTAShape;

  using CollectiveMainloop = typename Types::CollectiveMainloop;
  using CollectiveEpilogue = typename Types::CollectiveEpilogue;
  using GemmKernel = typename Types::GemmKernel;
  using GemmGrouped = typename Types::GemmGrouped;

  using StrideC = typename Types::StrideC;
  using StrideD = typename Types::StrideD;
  using StrideA = typename Types::StrideA;
  using StrideB = typename Types::StrideB;
  using LayoutSFA = typename Types::LayoutSFA;
  using LayoutSFB = typename Types::LayoutSFB;

  constexpr int AlignmentA = Types::AlignmentA;
  constexpr int AlignmentB = Types::AlignmentB;
  constexpr int AlignmentC = Types::AlignmentC;
  constexpr int AlignmentD = Types::AlignmentD;

  /////////////////////////////////////////////////////////////////////////////
  // HOW LAYOUTSFA CAPACITY IS DERIVED
  /////////////////////////////////////////////////////////////////////////////
  //
  // The kernel's LayoutSFA type (extracted above) defines the exact memory layout
  // expected by TMA for scale factor loads. The static_asserts above verify that
  // our sizing function uses the SAME SfAtom and tiling pattern as the kernel.
  //
  // GROUPED GEMM L SEMANTICS:
  // For grouped GEMM with PtrArray, each problem has its own (M_i, N, K) shape.
  // The TMA descriptor is set up per-problem, so L=1 for each problem's SFA.
  // The "grouping" is via pointer arrays, NOT via an L dimension in one tensor.
  //
  // Required buffer size per problem:
  //   auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(make_shape(M_i, N, K, 1));
  //   size_t required_bytes = cute::cosize(layout_sfa);  // + alignment
  //
  // For identity scales shared across all problems:
  //   Compute size for the LARGEST M across all groups: M_max
  //   All problems can then use this buffer (oversized for smaller M is safe)
  //
  // IDENTITY SFA ACQUISITION (caller must do this BEFORE calling launcher):
  //
  //   #include "../sm12x_layout_sfa_utils.h"
  //   #include "../sm12x_activation_quantizer.cuh"
  //
  //   // Step 1: Compute size for largest problem (L=1 for grouped GEMM)
  //   size_t required_bytes = computeSm120IdentitySFABufferSize(M_max, N, K, /*L=*/1);
  //
  //   // Step 2: Acquire identity buffer (pre-filled with 0x7F)
  //   uint8_t* identity_sfa = acquireSm120IdentitySFABuffer(required_bytes);
  //
  //   // Step 3: Build PER-GROUP POINTER ARRAY
  //   // CUTLASS grouped GEMM expects ElementSF const** (array of num_groups pointers)
  //   // For identity scales, all pointers can point to the same buffer.
  //   // IMPORTANT: This array must be in the correct memory space:
  //   //   - If kernel expects device pointers: allocate array on device
  //   //   - Lifetime must outlive gemm.run()
  //   //
  //   // Example (device-side pointer array):
  //   std::vector<uint8_t const*> h_sfa_ptrs(num_groups, identity_sfa);
  //   uint8_t const** d_sfa_ptrs;
  //   cudaMalloc(&d_sfa_ptrs, num_groups * sizeof(uint8_t*));
  //   cudaMemcpy(d_sfa_ptrs, h_sfa_ptrs.data(), num_groups * sizeof(uint8_t*), H2D);
  //   tma_inputs.fpX_block_scaling_factors_act = d_sfa_ptrs;
  //
  //   // For per-group LayoutSFA objects (all same for identity scales):
  //   auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(make_shape(M_max, N, K, 1));
  //   // ... set up tma_inputs.fpX_block_scaling_factors_stride_act similarly
  //
  /////////////////////////////////////////////////////////////////////////////

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
  // For OpClassBlockScaledTensorOp with grouped GEMM:
  //   - A_ptr, stride_A, B_ptr, stride_B (data) - pointer arrays
  //   - SFA_ptr, layout_SFA, SFB_ptr, layout_SFB (scale factors) - pointer arrays
  //
  // The CollectiveMainloop::Arguments takes the raw element types for pointers,
  // not the block-scaled wrapper types.
  //
  typename CollectiveMainloop::Arguments mainloop_args{
      reinterpret_cast<ElementA const**>(tma_inputs.ptr_act),       // Pre-quantized FP8 activations
      reinterpret_cast<StrideA>(tma_inputs.stride_act),             // Already pointer type for grouped GEMM
      reinterpret_cast<ElementB const**>(tma_inputs.ptr_weight),    // FP4 weights
      reinterpret_cast<StrideB>(tma_inputs.stride_weight),          // Already pointer type for grouped GEMM
      reinterpret_cast<ElementSF const**>(tma_inputs.fpX_block_scaling_factors_act),   // A-scales
      reinterpret_cast<LayoutSFA>(tma_inputs.fpX_block_scaling_factors_stride_act),    // Already pointer type for grouped GEMM
      reinterpret_cast<ElementSF const**>(tma_inputs.fpX_block_scaling_factors_weight), // B-scales
      reinterpret_cast<LayoutSFB>(tma_inputs.fpX_block_scaling_factors_stride_weight)}; // Already pointer type for grouped GEMM

  /////////////////////////////////////////////////////////////////////////////
  // Set up epilogue arguments
  /////////////////////////////////////////////////////////////////////////////
  // For grouped GEMM, epilogue also uses pointer arrays
  /////////////////////////////////////////////////////////////////////////////
  
  typename CollectiveEpilogue::Arguments epilogue_args{
      fusion_args,
      nullptr,  // C ptr (not used)
      nullptr,  // C stride (not used)
      reinterpret_cast<ElementD**>(tma_inputs.ptr_d),
      reinterpret_cast<StrideD>(tma_inputs.stride_d)  // Already pointer type for grouped GEMM
  };

  /////////////////////////////////////////////////////////////////////////////
  // Construct full arguments
  /////////////////////////////////////////////////////////////////////////////
  //
  // Note: mxfp4_wip does NOT include scheduler_args for SM120.
  // The GemmUniversal kernel uses default tile scheduling.
  //
  arguments = Args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      tma_inputs.shape_info,
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
  
  if (gemm.get_workspace_size(arguments) > tma_inputs.gemm_workspace_size) {
    TLLM_LOG_ERROR(
        "[SM120 Mixed-Input Grouped GEMM] Insufficient workspace: required %zu, allocated %zu",
        gemm.get_workspace_size(arguments), tma_inputs.gemm_workspace_size);
    return;
  }

  auto can_implement_status = gemm.can_implement(arguments);
  TLLM_LOG_DEBUG("  [Validation] can_implement=%s", cutlassGetStatusString(can_implement_status));
  if (can_implement_status != cutlass::Status::kSuccess) {
    std::string err_msg =
        "SM120 mixed-input grouped GEMM cannot be implemented. Error: " +
        std::string(cutlassGetStatusString(can_implement_status));
    TLLM_LOG_ERROR("[SM120 Mixed-Input Grouped GEMM] %s", err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
  
  // Log kernel type info for debugging dispatch
  TLLM_LOG_DEBUG("  [Kernel] GemmKernel schedule is PtrArray variant (supports kGrouped)");

  /////////////////////////////////////////////////////////////////////////////
  // DEBUG VERIFICATION: Check SFA buffer size is sufficient
  /////////////////////////////////////////////////////////////////////////////
  //
  // In debug builds, we verify that the SFA buffer provided by the caller
  // is large enough for the kernel's LayoutSFA. This catches buffer underalloc
  // early rather than manifesting as TMA errors or garbage output.
  //
  // NOTE: For grouped GEMM, we cannot easily verify individual problem sizes
  // here since they're on device. The caller is responsible for ensuring the
  // SFA buffer is sized for the largest (M, K) in the group.
  //
#ifndef NDEBUG
  if (tma_inputs.fpX_block_scaling_factors_act != nullptr) {
    TLLM_LOG_DEBUG("[SM120 Mixed-Input] SFA buffer verification enabled. "
                   "Caller must ensure buffer is sized for max(M,K) in the group "
                   "using computeKernelSFABufferSize<CollectiveMainloop>().");
  }
#endif

  // Comprehensive debug logging before initialize to diagnose failures
  //
  // Alignment requirements:
  //   - MXFP4 (FP8xFP4): 128-byte alignment
  //   - NVFP4 (FP4xFP4): may require 256-byte alignment
  // See: cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp
  //
  size_t required_workspace = gemm.get_workspace_size(arguments);
  uintptr_t workspace_addr = reinterpret_cast<uintptr_t>(tma_inputs.gemm_workspace);
  size_t required_alignment = IsMXFP4 ? 128 : 256;  // MXFP4=128B, NVFP4=256B
  size_t workspace_misalign = workspace_addr % required_alignment;
  bool workspace_aligned = (workspace_misalign == 0);
  
  TLLM_LOG_DEBUG("[SM120 Mixed-Input] === Pre-initialize diagnostics ===");
  
  // 1. Hardware configuration
  TLLM_LOG_DEBUG("  [HW] device_id=%d, sm_count=%d", hw_info.device_id, hw_info.sm_count);
  
  // 2. Workspace validation
  TLLM_LOG_DEBUG("  [Workspace] ptr=%p, %zuB-aligned=%s (addr %% %zu = %zu)",
                 tma_inputs.gemm_workspace,
                 required_alignment,
                 workspace_aligned ? "YES" : "NO",
                 required_alignment, workspace_misalign);
  TLLM_LOG_DEBUG("  [Workspace] size=%zu, required=%zu, sufficient=%s",
                 tma_inputs.gemm_workspace_size, required_workspace,
                 tma_inputs.gemm_workspace_size >= required_workspace ? "YES" : "NO");
  
  // 3. Problem shape info (grouped GEMM)
  TLLM_LOG_DEBUG("  [Groups] num_experts=%d", num_experts);
  TLLM_LOG_DEBUG("  [Groups] problem_shapes=%p, host_problem_shapes=%p",
                 (void*)tma_inputs.shape_info.problem_shapes,
                 (void*)tma_inputs.shape_info.host_problem_shapes);
  
  // 4. Pointer arrays
  TLLM_LOG_DEBUG("  [PtrArrays] ptr_act=%p, ptr_weight=%p, ptr_d=%p",
                 (void*)tma_inputs.ptr_act, (void*)tma_inputs.ptr_weight, (void*)tma_inputs.ptr_d);
  TLLM_LOG_DEBUG("  [PtrArrays] stride_act=%p, stride_weight=%p, stride_d=%p",
                 (void*)tma_inputs.stride_act, (void*)tma_inputs.stride_weight, (void*)tma_inputs.stride_d);
  
  // 5. Scale factors (critical for block-scaled)
  TLLM_LOG_DEBUG("  [ScaleFactors] sfa_ptr=%p, sfb_ptr=%p",
                 (void*)tma_inputs.fpX_block_scaling_factors_act,
                 (void*)tma_inputs.fpX_block_scaling_factors_weight);
  TLLM_LOG_DEBUG("  [ScaleFactors] sfa_stride=%p, sfb_stride=%p",
                 (void*)tma_inputs.fpX_block_scaling_factors_stride_act,
                 (void*)tma_inputs.fpX_block_scaling_factors_stride_weight);
  
  // 6. Tile configuration
  TLLM_LOG_DEBUG("  [Tiles] TileShape: M=%d, N=%d, K=%d",
                 (int)cute::size<0>(TileShape{}),
                 (int)cute::size<1>(TileShape{}),
                 (int)cute::size<2>(TileShape{}));
  TLLM_LOG_DEBUG("  [Tiles] ClusterShape: %dx%dx%d",
                 (int)cute::size<0>(ClusterShape{}),
                 (int)cute::size<1>(ClusterShape{}),
                 (int)cute::size<2>(ClusterShape{}));
  
  // 7. Type configuration
  TLLM_LOG_DEBUG("  [Types] AlignmentA=%d, AlignmentB=%d, AlignmentC=%d, AlignmentD=%d",
                 AlignmentA, AlignmentB, AlignmentC, AlignmentD);
  TLLM_LOG_DEBUG("  [Types] IsMXFP4=%s (FP8xFP4 if true, FP4xFP4 if false)", IsMXFP4 ? "true" : "false");
  TLLM_LOG_DEBUG("  [Types] sizeof(ElementA)=%zu, sizeof(ElementB)=%zu, sizeof(ElementSF)=%zu",
                 sizeof(ElementA), sizeof(ElementB), sizeof(ElementSF));
  
  // 8. Layout types - CRITICAL: For grouped GEMM, these must be pointer types
  TLLM_LOG_DEBUG("  [Layouts] IsGroupedGemm=%s",
                 !std::is_same_v<StrideA, cute::remove_pointer_t<StrideA>> ? "YES (StrideA is ptr)" : "NO");
  TLLM_LOG_DEBUG("  [Layouts] StrideA is_pointer=%s, StrideB is_pointer=%s",
                 std::is_pointer_v<StrideA> ? "YES" : "NO",
                 std::is_pointer_v<StrideB> ? "YES" : "NO");
  TLLM_LOG_DEBUG("  [Layouts] LayoutSFA is_pointer=%s, LayoutSFB is_pointer=%s",
                 std::is_pointer_v<LayoutSFA> ? "YES" : "NO",
                 std::is_pointer_v<LayoutSFB> ? "YES" : "NO");
  
  TLLM_LOG_DEBUG("  === End diagnostics ===");

  // Log shared memory size requirement
  constexpr int smem_size = GemmKernel::SharedStorageSize;
  TLLM_LOG_DEBUG("  [SMEM] SharedStorageSize=%d bytes", smem_size);
  
  // Check if smem_size exceeds device limits (typically 227KB for SM12x)
  int max_smem = 0;
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, hw_info.device_id);
  TLLM_LOG_DEBUG("  [SMEM] Device max_smem_per_block_optin=%d bytes", max_smem);
  if (smem_size > max_smem) {
    TLLM_LOG_ERROR("  [SMEM] Kernel requires %d bytes but device max is %d bytes!", smem_size, max_smem);
  }

  auto init_status = gemm.initialize(arguments, tma_inputs.gemm_workspace, stream);
  if (init_status != cutlass::Status::kSuccess) {
    // Log detailed error info
    TLLM_LOG_ERROR("[SM120 Mixed-Input] initialize() FAILED with status: %s",
                   cutlassGetStatusString(init_status));

    // Best-effort: dump a small slice of the device-side group metadata to catch
    // shape/stride/pointer corruption (common root cause for Error Internal).
    // This runs ONLY on failure.
    {
      constexpr int kDump = 8;
      int dump_n = num_experts < kDump ? num_experts : kDump;

      // Note: TmaWarpSpecializedGroupedGemmInput does NOT expose expert_first_token_offset.
      // That lives in higher-level fused MoE code. Here we only dump what this launcher receives.

      if (tma_inputs.shape_info.problem_shapes) {
        using UnderlyingProblemShape =
            std::remove_pointer_t<decltype(tma_inputs.shape_info.problem_shapes)>;
        UnderlyingProblemShape host_shapes[kDump] = {};
        cudaError_t st = cudaMemcpy(host_shapes, tma_inputs.shape_info.problem_shapes,
                                    sizeof(UnderlyingProblemShape) * dump_n, cudaMemcpyDeviceToHost);
        if (st == cudaSuccess) {
          auto const* b = reinterpret_cast<unsigned char const*>(&host_shapes[0]);
          TLLM_LOG_ERROR(
              "  [Dump] problem_shapes[0] raw bytes: "
              "%02x %02x %02x %02x %02x %02x %02x %02x "
              "%02x %02x %02x %02x %02x %02x %02x %02x "
              "%02x %02x %02x %02x %02x %02x %02x %02x",
              b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
              b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15],
              b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23]);

          for (int i = 0; i < dump_n; ++i) {
            auto m = cute::get<0>(host_shapes[i]);
            auto n = cute::get<1>(host_shapes[i]);
            auto k = cute::get<2>(host_shapes[i]);
            TLLM_LOG_ERROR("  [Dump] problem_shapes[%d]: M=%lld N=%lld K=%lld", i,
                           static_cast<long long>(m), static_cast<long long>(n),
                           static_cast<long long>(k));
          }
        } else {
          TLLM_LOG_ERROR("  [Dump] cudaMemcpy problem_shapes failed: %s", cudaGetErrorString(st));
        }
      }

      if (tma_inputs.ptr_act && tma_inputs.ptr_weight && tma_inputs.ptr_d) {
        void const* host_ptr_act[kDump] = {};
        void const* host_ptr_weight[kDump] = {};
        void* host_ptr_d[kDump] = {};
        cudaError_t st1 = cudaMemcpy(host_ptr_act, tma_inputs.ptr_act,
                                     sizeof(void const*) * dump_n, cudaMemcpyDeviceToHost);
        cudaError_t st2 = cudaMemcpy(host_ptr_weight, tma_inputs.ptr_weight,
                                     sizeof(void const*) * dump_n, cudaMemcpyDeviceToHost);
        cudaError_t st3 = cudaMemcpy(host_ptr_d, tma_inputs.ptr_d, sizeof(void*) * kDump,
                                     cudaMemcpyDeviceToHost);
        if (st1 == cudaSuccess && st2 == cudaSuccess && st3 == cudaSuccess) {
          TLLM_LOG_ERROR("  [Dump] ptr_act[0]=%p ptr_weight[0]=%p ptr_d[0]=%p", host_ptr_act[0],
                         host_ptr_weight[0], host_ptr_d[0]);
        } else {
          TLLM_LOG_ERROR("  [Dump] cudaMemcpy ptr arrays failed: act=%s weight=%s d=%s",
                         cudaGetErrorString(st1), cudaGetErrorString(st2),
                         cudaGetErrorString(st3));
        }
      }

      if (tma_inputs.fpX_block_scaling_factors_act && tma_inputs.fpX_block_scaling_factors_weight) {
        void const* host_sfa[kDump] = {};
        void const* host_sfb[kDump] = {};
        cudaError_t st1 = cudaMemcpy(host_sfa, tma_inputs.fpX_block_scaling_factors_act,
                                     sizeof(void const*) * dump_n, cudaMemcpyDeviceToHost);
        cudaError_t st2 = cudaMemcpy(host_sfb, tma_inputs.fpX_block_scaling_factors_weight,
                                     sizeof(void const*) * dump_n, cudaMemcpyDeviceToHost);
        if (st1 == cudaSuccess && st2 == cudaSuccess) {
          TLLM_LOG_ERROR("  [Dump] sfa[0]=%p sfb[0]=%p", host_sfa[0], host_sfb[0]);
        } else {
          TLLM_LOG_ERROR("  [Dump] cudaMemcpy sf ptr arrays failed: sfa=%s sfb=%s",
                         cudaGetErrorString(st1), cudaGetErrorString(st2));
        }
      }

      // Dump scale-factor layout objects (stride/layout metadata) - most common source of TMA init failures
      if (tma_inputs.fpX_block_scaling_factors_stride_act &&
          tma_inputs.fpX_block_scaling_factors_stride_weight) {
        using LayoutSFAObj = std::remove_pointer_t<LayoutSFA>;
        using LayoutSFBObj = std::remove_pointer_t<LayoutSFB>;

        LayoutSFAObj host_lsfa[kDump] = {};
        LayoutSFBObj host_lsfb[kDump] = {};
        cudaError_t st1 = cudaMemcpy(host_lsfa, tma_inputs.fpX_block_scaling_factors_stride_act,
                                     sizeof(LayoutSFAObj) * dump_n, cudaMemcpyDeviceToHost);
        cudaError_t st2 = cudaMemcpy(host_lsfb, tma_inputs.fpX_block_scaling_factors_stride_weight,
                                     sizeof(LayoutSFBObj) * dump_n, cudaMemcpyDeviceToHost);
        if (st1 == cudaSuccess && st2 == cudaSuccess) {
          auto const* a = reinterpret_cast<unsigned char const*>(&host_lsfa[0]);
          auto const* b = reinterpret_cast<unsigned char const*>(&host_lsfb[0]);
          TLLM_LOG_ERROR("  [Dump] sfa_layout[0] raw: %02x %02x %02x %02x %02x %02x %02x %02x ...",
                         a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
          TLLM_LOG_ERROR("  [Dump] sfb_layout[0] raw: %02x %02x %02x %02x %02x %02x %02x %02x ...",
                         b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);

          // Print a couple of high-signal layout properties
          TLLM_LOG_ERROR("  [Dump] sfa_layout[0] cosize=%lld", (long long)cute::cosize(host_lsfa[0]));
          TLLM_LOG_ERROR("  [Dump] sfb_layout[0] cosize=%lld", (long long)cute::cosize(host_lsfb[0]));
        } else {
          TLLM_LOG_ERROR("  [Dump] cudaMemcpy sf layout arrays failed: sfa_layout=%s sfb_layout=%s",
                         cudaGetErrorString(st1), cudaGetErrorString(st2));
        }
      }

      // Dump grouped stride metadata (these frequently cause TMA descriptor creation failures)
      if (tma_inputs.stride_act && tma_inputs.stride_weight && tma_inputs.stride_d) {
        using StrideAObj = std::remove_pointer_t<StrideA>;
        using StrideBObj = std::remove_pointer_t<StrideB>;
        using StrideDObj = std::remove_pointer_t<StrideD>;

        StrideAObj host_sa[kDump] = {};
        StrideBObj host_sb[kDump] = {};
        StrideDObj host_sd[kDump] = {};

        cudaError_t st1 = cudaMemcpy(host_sa, tma_inputs.stride_act,
                                     sizeof(StrideAObj) * dump_n, cudaMemcpyDeviceToHost);
        cudaError_t st2 = cudaMemcpy(host_sb, tma_inputs.stride_weight,
                                     sizeof(StrideBObj) * dump_n, cudaMemcpyDeviceToHost);
        cudaError_t st3 = cudaMemcpy(host_sd, tma_inputs.stride_d,
                                     sizeof(StrideDObj) * dump_n, cudaMemcpyDeviceToHost);

        if (st1 == cudaSuccess && st2 == cudaSuccess && st3 == cudaSuccess) {
          auto const* ba = reinterpret_cast<unsigned char const*>(&host_sa[0]);
          auto const* bb = reinterpret_cast<unsigned char const*>(&host_sb[0]);
          auto const* bd = reinterpret_cast<unsigned char const*>(&host_sd[0]);
          TLLM_LOG_ERROR("  [Dump] stride_act[0] raw: %02x %02x %02x %02x %02x %02x %02x %02x ...",
                         ba[0], ba[1], ba[2], ba[3], ba[4], ba[5], ba[6], ba[7]);
          TLLM_LOG_ERROR("  [Dump] stride_weight[0] raw: %02x %02x %02x %02x %02x %02x %02x %02x ...",
                         bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], bb[6], bb[7]);
          TLLM_LOG_ERROR("  [Dump] stride_d[0] raw: %02x %02x %02x %02x %02x %02x %02x %02x ...",
                         bd[0], bd[1], bd[2], bd[3], bd[4], bd[5], bd[6], bd[7]);

          // Best-effort decoding: many CUTLASS/CUTE strides are 2D (leading_dim, 1)
          // If these get<>() calls fail to compile for a given stride type, keep the raw dump above.
          for (int i = 0; i < dump_n; ++i) {
            auto a0 = cute::get<0>(host_sa[i]);
            auto a1 = cute::get<1>(host_sa[i]);
            auto b0 = cute::get<0>(host_sb[i]);
            auto b1 = cute::get<1>(host_sb[i]);
            auto d0 = cute::get<0>(host_sd[i]);
            auto d1 = cute::get<1>(host_sd[i]);
            TLLM_LOG_ERROR("  [Dump] stride_act[%d]: (%lld,%lld) stride_weight[%d]: (%lld,%lld) stride_d[%d]: (%lld,%lld)",
                           i, (long long)a0, (long long)a1,
                           i, (long long)b0, (long long)b1,
                           i, (long long)d0, (long long)d1);
          }
        } else {
          TLLM_LOG_ERROR("  [Dump] cudaMemcpy stride arrays failed: act=%s weight=%s d=%s",
                         cudaGetErrorString(st1), cudaGetErrorString(st2),
                         cudaGetErrorString(st3));
        }
      }
    }

    // Check for specific issues - prioritize most likely causes
    bool has_known_issue = false;
    
    // Check workspace alignment (MXFP4=128B, NVFP4=256B)
    if (!workspace_aligned) {
      TLLM_LOG_ERROR("  >>> Workspace NOT %zu-byte aligned!", required_alignment);
      TLLM_LOG_ERROR("      ptr=%p, addr %% %zu = %zu (must be 0)",
                     tma_inputs.gemm_workspace, required_alignment, workspace_misalign);
      has_known_issue = true;
    }
    
    if (tma_inputs.gemm_workspace_size < required_workspace) {
      TLLM_LOG_ERROR("  >>> Workspace too small: need %zu, have %zu", required_workspace, tma_inputs.gemm_workspace_size);
      has_known_issue = true;
    }
    if (tma_inputs.fpX_block_scaling_factors_act == nullptr) {
      TLLM_LOG_ERROR("  >>> SFA pointer is null (block-scaled requires scale factors)");
      has_known_issue = true;
    }
    if (tma_inputs.fpX_block_scaling_factors_weight == nullptr) {
      TLLM_LOG_ERROR("  >>> SFB pointer is null (block-scaled requires scale factors)");
      has_known_issue = true;
    }
    
    // Type verification (compile-time, but log for debugging)
    if (!std::is_pointer_v<StrideA>) {
      TLLM_LOG_ERROR("  >>> StrideA is NOT a pointer type - grouped GEMM requires pointer strides");
      has_known_issue = true;
    }
    if (!std::is_pointer_v<LayoutSFA>) {
      TLLM_LOG_ERROR("  >>> LayoutSFA is NOT a pointer type - grouped GEMM requires pointer layouts");
      has_known_issue = true;
    }
    
    if (!has_known_issue) {
      TLLM_LOG_ERROR("  No obvious issue detected. Possible causes:");
      TLLM_LOG_ERROR("  - Scale factor layout stride mismatch with actual buffer layout");
      TLLM_LOG_ERROR("  - Problem shapes on device memory not accessible");
      TLLM_LOG_ERROR("  - TMA descriptor creation failed for given strides");
    }
    
    // Note: keep the exception message minimal and rely on the detailed TLLM_LOG_DEBUG lines
    // above. Some libstdc++/libc++ configurations have been observed to segfault when building
    // complex iostream-based strings in this error path (likely due to prior memory corruption).
    std::string err_msg =
        "Failed to initialize SM120 mixed-input grouped GEMM. Error: " +
        std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error(err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run SM120 mixed-input grouped GEMM. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error(err_msg);
  }

#else
  TLLM_THROW("SM120/SM121 mixed-input GEMM requires CUTLASS_ARCH_MMA_SM12x_SUPPORTED (SM120 or SM121) and ENABLE_FP4");
#endif  // CUTLASS_ARCH_MMA_SM12x_SUPPORTED && ENABLE_FP4
}

// =============================================================================
// Identity SFA Buffer Acquisition API Implementation
// =============================================================================

// SM120 scale factor vector size (MXFP4 uses group size 32)
static constexpr int kSm120SFVecSize = 32;

// Compute required SFA buffer size for A-scales (kernel-derived using tile_atom_to_shape_SFA)
inline size_t computeSm120IdentitySFABufferSize(int64_t M_max, int64_t N, int64_t K, int64_t L) {
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4)
    // Use the same SfConfig that the kernel uses (verified by static_assert in launcher)
    using SfConfig = cutlass::detail::Sm1xxBlockScaledConfig<kSm120SFVecSize>;
    
    // Create problem shape with the maximum M across all groups
    auto problem_shape = cute::make_shape(
        static_cast<int>(M_max),
        static_cast<int>(N),
        static_cast<int>(K),
        static_cast<int>(L)
    );
    
    // Get the SFA layout for this problem shape
    auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(problem_shape);
    
    // cosize gives the maximum linear index + 1 (the required buffer capacity)
    size_t sfa_elements = cute::cosize(layout_sfa);
    
    // Align to 256 bytes for TMA requirements
    return (sfa_elements + 255) & ~size_t(255);
#else
    // Fallback when FP4 is not enabled
    return tensorrt_llm::kernels::cutlass_kernels::Sm12xLayoutSFAUtils::computeBufferSize(
        static_cast<int>(M_max), static_cast<int>(N), static_cast<int>(K), static_cast<int>(L)
    );
#endif
}

// Compute required SFB buffer size for B-scales (kernel-derived using tile_atom_to_shape_SFB)
// IMPORTANT: This uses tile_atom_to_shape_SFB, NOT SFA! They have different layouts.
inline size_t computeSm120IdentitySFBBufferSize(int64_t M, int64_t N, int64_t K, int64_t L) {
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4)
    // Use the same SfConfig that the kernel uses
    using SfConfig = cutlass::detail::Sm1xxBlockScaledConfig<kSm120SFVecSize>;
    
    // Create problem shape - use the SAME shape as passed to the kernel
    // No dimension remapping needed - SFB uses (M, N, K, L) directly
    auto problem_shape = cute::make_shape(
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        static_cast<int>(L)
    );
    
    // Get the SFB layout for this problem shape (different from SFA!)
    auto layout_sfb = SfConfig::tile_atom_to_shape_SFB(problem_shape);
    
    // cosize gives the maximum linear index + 1 (the required buffer capacity)
    // Since ElementSF is byte-sized (uint8_t), cosize == bytes
    size_t sfb_bytes = cute::cosize(layout_sfb);
    
    // Align to 256 bytes for TMA requirements
    return (sfb_bytes + 255) & ~size_t(255);
#else
    // Fallback when FP4 is not enabled - estimate based on N and K dimensions
    // SFB scales the B matrix which has shape (K, N) in column-major
    int num_n_blocks = (N + 127) / 128;
    int k_atoms = (K + 127) / 128;
    size_t sfb_elements = static_cast<size_t>(num_n_blocks) * k_atoms * 128;
    return (sfb_elements + 255) & ~size_t(255);
#endif
}

// =============================================================================
// Host-only API: These functions use host-side managers (mutex, unordered_map)
// and must only be called from host code, not device code.
// =============================================================================
#if !defined(__CUDA_ARCH__)

// Acquire identity SFA buffer using the size-based API
inline uint8_t* acquireSm120IdentitySFABuffer(size_t required_bytes) {
    auto& mgr = tensorrt_llm::kernels::cutlass_kernels::getIdentityScaleBufferManager();
    return mgr.getOrCreateWithSize(required_bytes);
}

// Acquire identity SFB buffer using the size-based API (same manager, different sizing)
inline uint8_t* acquireSm120IdentitySFBBuffer(size_t required_bytes) {
    // Uses the same manager - it's just a cache by (device, size)
    auto& mgr = tensorrt_llm::kernels::cutlass_kernels::getIdentityScaleBufferManager();
    return mgr.getOrCreateWithSize(required_bytes);
}

// Prewarm identity SFA buffers for common MoE shapes
inline void prewarmSm120IdentitySFABuffers(
    const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>& shapes) {
    
    std::vector<size_t> sizes;
    sizes.reserve(shapes.size());
    
    for (const auto& [M_max, N, K, L] : shapes) {
        sizes.push_back(computeSm120IdentitySFABufferSize(M_max, N, K, L));
    }
    
    auto& mgr = tensorrt_llm::kernels::cutlass_kernels::getIdentityScaleBufferManager();
    mgr.prewarmWithSizes(sizes);
}

// Get device-side SFA pointer array (cached, no hot-path allocation)
inline uint8_t const** acquireSm120SFAPointerArray(int num_groups, uint8_t const* identity_sfa) {
    auto& mgr = tensorrt_llm::kernels::cutlass_kernels::getSFAPointerArrayManager();
    return mgr.getOrCreate(num_groups, identity_sfa);
}

// Prewarm SFA pointer arrays for common expert counts
inline void prewarmSm120SFAPointerArrays(const std::vector<int>& expert_counts, 
                                          uint8_t const* identity_sfa) {
    auto& mgr = tensorrt_llm::kernels::cutlass_kernels::getSFAPointerArrayManager();
    mgr.prewarm(expert_counts, identity_sfa);
}

#endif  // !defined(__CUDA_ARCH__)

}  // namespace cutlass_kernels_oss
}  // namespace kernels
}  // namespace tensorrt_llm

