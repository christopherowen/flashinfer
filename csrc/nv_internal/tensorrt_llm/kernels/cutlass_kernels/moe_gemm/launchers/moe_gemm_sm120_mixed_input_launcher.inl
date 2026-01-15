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

// =============================================================================
// SM120 Block-Scaled Mixed-Input MoE GEMM Launcher
// =============================================================================
//
// CRITICAL (still required): CUTLASS grouped GEMM `initialize()` can return
// `Error Internal` on SM12x if the CUTLASS kernel type graph is instantiated
// inside a function template, even when `can_implement()` returns Success.
//
// This is NOT hypothetical/stale. It is reproducible in our current dev
// environment via the standalone repro bisection:
// - step20 (types inside function template): initialize=Error Internal
// - step21 (non-template): initialize=Success
// - step22 (template, but kernel types moved to namespace scope): initialize=Success
// See: `docs/cutlass_error_internal_repro/README.md` (step20/21/22).
//
// Therefore: keep CUTLASS kernel types at namespace scope in a non-templated
// context, and only template the outer launcher for dispatch if needed.
//
// TILE AND SWAP SELECTION:
// Python JIT sets three compile-time flags:
//   -DLOGICAL_TILE_M=<M>   Logical M tile (token dimension)
//   -DLOGICAL_TILE_N=<N>   Logical N tile (output dimension)
//   -DSWAP_AB=0|1          Whether to use transposed mode (swap A/B operands)
//
// SWAP_AB is the primary selection mechanism:
//   - SWAP_AB=0: Standard mode (logical M >= 128)
//   - SWAP_AB=1: Transposed mode (logical M < 128, uses swap-hack)
//
// The Python code determines SWAP_AB based on tile shape, so C++ code does NOT
// need to hardcode magic tile values like "LOGICAL_TILE_M == 32".
//
// Each mode is defined in a separate namespace to avoid CUTLASS Error Internal.
//
// =============================================================================

#pragma once

// Default logical tile configuration
#ifndef LOGICAL_TILE_M
#define LOGICAL_TILE_M 128
#endif
#ifndef LOGICAL_TILE_N
#define LOGICAL_TILE_N 128
#endif

// SWAP_AB: Compile-time flag for transposed mode (swap A/B operands).
// This is set by Python JIT based on tile shape: swap_ab = (logical_m < 128).
// The C++ code uses this flag, NOT hardcoded tile size checks.
#ifndef SWAP_AB
#define SWAP_AB 0
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "moe_gemm_sm120_mixed_input_launcher.h"
// #include "../sm12x_arch_config.h"

#if defined(COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS)
#define CUTLASS_ARCH_MMA_SM12x_SUPPORTED
#endif

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

namespace tensorrt_llm::kernels::cutlass_kernels_oss {

using namespace tensorrt_llm::kernels::cutlass_kernels;

// =============================================================================
// FIXED (NON-TEMPLATED) CUTLASS TYPES FOR SM120 MXFP4
// =============================================================================
//
// These types MUST be at namespace scope in a NON-TEMPLATED context.
// Using a struct/class template is still "templated instantiation" and can
// reintroduce the `initialize=Error Internal` failure on SM12x.
//
// Following exactly the pattern from step22 in the repro.
// Macros reduce duplication while maintaining separate namespaces.
//
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4)

// =============================================================================
// MACRO: Standard Mode (TILE_M >= 128)
// =============================================================================
// Standard GEMM: D = A @ W
// - A = FP8 activations (ElementInputA = float_e4m3_t)
// - B = FP4 weights (ElementInputB = float_e2m1_t)
// - Output layout: RowMajor
//
#define DEFINE_SM120_MXFP4_STANDARD_NAMESPACE(NAMESPACE_NAME, TILE_M_VAL, TILE_N_VAL, TILE_K_VAL) \
namespace NAMESPACE_NAME {                                                                        \
                                                                                                  \
using namespace cute;                                                                             \
                                                                                                  \
/* Element types: FP8 activations × FP4 weights → BF16 output */                                  \
using ElementInputA = cutlass::float_e4m3_t;                                                      \
using ElementInputB = cutlass::float_e2m1_t;                                                      \
using ElementA = cutlass::mx_float8_t<ElementInputA>;                                             \
using ElementB = cutlass::mx_float4_t<ElementInputB>;                                             \
                                                                                                  \
using ElementC = cutlass::bfloat16_t;                                                             \
using ElementD = cutlass::bfloat16_t;                                                             \
                                                                                                  \
using ElementAccumulator = float;                                                                 \
using ElementCompute = float;                                                                     \
using ElementSF = cutlass::float_ue8m0_t;                                                         \
                                                                                                  \
/* Architecture and operator class */                                                             \
using ArchTag = cutlass::arch::Sm120;                                                             \
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;                                  \
                                                                                                  \
/* Layouts (TN: A row-major, B column-major) */                                                   \
using LayoutA = cutlass::layout::RowMajor;                                                        \
using LayoutB = cutlass::layout::ColumnMajor;                                                     \
using LayoutC = cutlass::layout::RowMajor;                                                        \
                                                                                                  \
/* Alignment requirements */                                                                      \
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;  /* 16 for FP8 */    \
constexpr int AlignmentB = 128;  /* Special-case for FP4 */                                       \
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;                           \
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;                           \
                                                                                                  \
/* Tile shape */                                                                                  \
using TileShape_MNK = Shape<cute::Int<TILE_M_VAL>, cute::Int<TILE_N_VAL>, cute::Int<TILE_K_VAL>>; \
using ClusterShape_MNK = Shape<_1, _1, _1>;                                                       \
                                                                                                  \
/* Epilogue collective */                                                                         \
using CollectiveEpilogue =                                                                        \
    typename cutlass::epilogue::collective::CollectiveBuilder<                                    \
        ArchTag, OperatorClass, TileShape_MNK, ClusterShape_MNK,                                  \
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,      \
        ElementC, LayoutC*, AlignmentC, ElementD, LayoutC*, AlignmentD,                           \
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;                       \
                                                                                                  \
/* Stage count (carve out space for epilogue shared memory) */                                    \
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<                             \
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;                        \
                                                                                                  \
/* Mainloop collective                                                                            \
 * NOTE: The SM100 generic launcher uses KernelScheduleAuto for SM120.                            \
 * KernelPtrArrayTmaWarpSpecializedPingpong is hardcoded here; consider testing                   \
 * with KernelScheduleAuto if compilation fails or performance is suboptimal.                     \
 */                                                                                               \
using CollectiveMainloop =                                                                        \
    typename cutlass::gemm::collective::CollectiveBuilder<                                        \
        ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,   \
        ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCount,                          \
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;                   \
                                                                                                  \
/* Problem shape for grouped GEMM */                                                              \
using ProblemShape =                                                                              \
    cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;                           \
                                                                                                  \
/* GEMM kernel and adapter */                                                                     \
using GemmKernel =                                                                                \
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>; \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                             \
                                                                                                  \
/* Stride and layout types for grouped GEMM */                                                    \
using StrideA = typename CollectiveMainloop::StrideA;                                             \
using StrideB = typename CollectiveMainloop::StrideB;                                             \
using StrideD = typename CollectiveEpilogue::StrideD;                                             \
using LayoutSFA = typename CollectiveMainloop::LayoutSFA;                                         \
using LayoutSFB = typename CollectiveMainloop::LayoutSFB;                                         \
                                                                                                  \
}  /* namespace NAMESPACE_NAME */

// =============================================================================
// MACRO: Transpose Mode (TILE_M < 128, decode-optimized)
// =============================================================================
// Transpose GEMM: D^T = W^T @ A^T (then output written as D)
// - A = FP4 weights (ElementInputA = float_e2m1_t) - SWAPPED
// - B = FP8 activations (ElementInputB = float_e4m3_t) - SWAPPED
// - Output layout: ColumnMajor (writes D^T which appears as D in row-major)
// - Virtual M = 128 (fixed, large dimension from original N)
// - Virtual N = TILE_N_VAL (small, token dimension, was original M)
//
// This works around tcgen05 hardware minimum M=64 by putting the small
// dimension in N (which has minimum 8).
//
// CONSTRAINT: Original N dimension (intermediate_size or hidden_size) must be
// divisible by 128. This is satisfied by most models (e.g., 14336/128=112).
//
#define DEFINE_SM120_MXFP4_TRANSPOSED_NAMESPACE(NAMESPACE_NAME, TILE_N_VAL)                       \
namespace NAMESPACE_NAME {                                                                        \
                                                                                                  \
using namespace cute;                                                                             \
                                                                                                  \
/* Element types SWAPPED: A is FP4 (Weight), B is FP8 (Act) */                                    \
using ElementInputA = cutlass::float_e2m1_t;                                                      \
using ElementInputB = cutlass::float_e4m3_t;                                                      \
using ElementA = cutlass::mx_float4_t<ElementInputA>;                                             \
using ElementB = cutlass::mx_float8_t<ElementInputB>;                                             \
                                                                                                  \
using ElementC = cutlass::bfloat16_t;                                                             \
using ElementD = cutlass::bfloat16_t;                                                             \
                                                                                                  \
using ElementAccumulator = float;                                                                 \
using ElementCompute = float;                                                                     \
using ElementSF = cutlass::float_ue8m0_t;                                                         \
                                                                                                  \
/* Architecture and operator class */                                                             \
using ArchTag = cutlass::arch::Sm120;                                                             \
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;                                  \
                                                                                                  \
/* Layouts: A (Weight) row-major, B (Act) column-major, C/D column-major for transpose */         \
using LayoutA = cutlass::layout::RowMajor;                                                        \
using LayoutB = cutlass::layout::ColumnMajor;                                                     \
using LayoutC = cutlass::layout::ColumnMajor;                                                     \
                                                                                                  \
/* Alignment requirements */                                                                      \
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;  /* 32 for FP4 */    \
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;  /* 16 for FP8 */    \
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;                           \
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;                           \
                                                                                                  \
/* Tile shape: Virtual M=128 (fixed), N=TILE_N_VAL (small token dim), K=128 */                    \
using TileShape_MNK = Shape<_128, cute::Int<TILE_N_VAL>, _128>;                                   \
using ClusterShape_MNK = Shape<_1, _1, _1>;                                                       \
                                                                                                  \
/* Epilogue collective */                                                                         \
using CollectiveEpilogue =                                                                        \
    typename cutlass::epilogue::collective::CollectiveBuilder<                                    \
        ArchTag, OperatorClass, TileShape_MNK, ClusterShape_MNK,                                  \
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,      \
        ElementC, LayoutC*, AlignmentC, ElementD, LayoutC*, AlignmentD,                           \
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;                       \
                                                                                                  \
/* Stage count (carve out space for epilogue shared memory) */                                    \
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<                             \
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;                        \
                                                                                                  \
/* Mainloop collective                                                                            \
 * NOTE: The SM100 generic launcher uses KernelScheduleAuto for SM120.                            \
 * KernelPtrArrayTmaWarpSpecializedPingpong is hardcoded here; consider testing                   \
 * with KernelScheduleAuto if compilation fails or performance is suboptimal.                     \
 */                                                                                               \
using CollectiveMainloop =                                                                        \
    typename cutlass::gemm::collective::CollectiveBuilder<                                        \
        ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,   \
        ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCount,                          \
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;                   \
                                                                                                  \
/* Problem shape for grouped GEMM */                                                              \
using ProblemShape =                                                                              \
    cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;                           \
                                                                                                  \
/* GEMM kernel and adapter */                                                                     \
using GemmKernel =                                                                                \
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>; \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                             \
                                                                                                  \
/* Stride and layout types for grouped GEMM */                                                    \
using StrideA = typename CollectiveMainloop::StrideA;                                             \
using StrideB = typename CollectiveMainloop::StrideB;                                             \
using StrideD = typename CollectiveEpilogue::StrideD;                                             \
using LayoutSFA = typename CollectiveMainloop::LayoutSFA;                                         \
using LayoutSFB = typename CollectiveMainloop::LayoutSFB;                                         \
                                                                                                  \
}  /* namespace NAMESPACE_NAME */

// =============================================================================
// NAMESPACE INSTANTIATIONS
// =============================================================================
//
// Namespace selection is controlled by SWAP_AB (set by Python JIT):
//   - SWAP_AB=0: Standard mode (logical M >= 128)
//   - SWAP_AB=1: Transposed mode (logical M < 128, uses swap-hack)
//
// The Python JIT determines SWAP_AB based on tile shape and passes it here.
// This avoids hardcoding magic tile values in C++.
//

// Namespace instantiation based on SWAP_AB (set by Python JIT).
// Each JIT compilation creates exactly one namespace with the specified tiles.
//
// Naming convention:
//   sm120_mxfp4_bf16 - base name (arch, format, activation type)
//   No tile dimensions in name - they come from JIT flags
//
#if SWAP_AB
// Swapped mode (logical M < 128): uses transposed layout to work around tcgen05 M>=64 constraint
DEFINE_SM120_MXFP4_TRANSPOSED_NAMESPACE(sm120_mxfp4_bf16, LOGICAL_TILE_M)
#else
// Standard mode (logical M >= 128): normal layout
DEFINE_SM120_MXFP4_STANDARD_NAMESPACE(sm120_mxfp4_bf16, LOGICAL_TILE_M, LOGICAL_TILE_N, 128)
#endif

#endif  // CUTLASS_ARCH_MMA_SM12x_SUPPORTED && ENABLE_FP4

// =============================================================================
// SM120 Launcher Implementation
// =============================================================================

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag,
          typename TileShape, typename ClusterShape, bool IsMXFP4>
void sm120_mixed_input_moe_gemm_kernelLauncher(
    TmaWarpSpecializedGroupedGemmInput tma_inputs, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* occupancy,
    size_t* workspace_size) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4)

  // Only MXFP4 is supported
  static_assert(IsMXFP4,
      "SM120 MoE GEMM only supports MXFP4 (FP8xFP4). NVFP4 (FP4xFP4) not yet implemented.");

  // swap_ab is determined by compile-time SWAP_AB flag (set by Python JIT).
  // Override runtime swap_ab to match compile-time flag, similar to SM100 approach.
  // This ensures consistency regardless of what the caller set.
  constexpr bool kSwapAB = SWAP_AB;
  tma_inputs.swap_ab = kSwapAB;
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] swap_ab=%s (from SWAP_AB compile flag)",
      kSwapAB ? "true" : "false");

  // Reject finalize fusion for now (SM120 launcher uses simple epilogue without finalize support)
  TLLM_CHECK_WITH_INFO(
      tma_inputs.fusion != TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE,
      "SM120 MoE GEMM does not support finalize fusion. Use NONE or ACTIVATION epilogue.");

  // Use the JIT-compiled namespace (single namespace per compilation).
  // Tile dimensions and swap mode are baked in via LOGICAL_TILE_M, LOGICAL_TILE_N, SWAP_AB.
  using namespace sm120_mxfp4_bf16;
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] tile_mn=(%d,%d) swap_ab=%s",
      LOGICAL_TILE_M, LOGICAL_TILE_N, kSwapAB ? "true" : "false");

  Gemm gemm;

  // Hardware info
  cutlass::KernelHardwareInfo hw_info{};
  hw_info.device_id = 0;
  hw_info.sm_count = multi_processor_count;

  // Operand wiring based on swap_ab (set above from compile-time SWAP_AB):
  // - swap_ab=false => A=act, B=weight, SFA=act_sf, SFB=weight_sf
  // - swap_ab=true  => A=weight, B=act, SFA=weight_sf, SFB=act_sf
  void const** ptr_A = tma_inputs.swap_ab ? tma_inputs.ptr_weight : tma_inputs.ptr_act;
  void const** ptr_B = tma_inputs.swap_ab ? tma_inputs.ptr_act : tma_inputs.ptr_weight;
  void* stride_A = tma_inputs.swap_ab ? tma_inputs.stride_weight : tma_inputs.stride_act;
  void* stride_B = tma_inputs.swap_ab ? tma_inputs.stride_act : tma_inputs.stride_weight;
  // NOTE: TmaWarpSpecializedGroupedGemmInput stores SF pointers as raw bytes
  // (ElementSF == uint8_t). The CUTLASS mainloop expects ElementSF as
  // cutlass::float_ue8m0_t. We keep the stored type here and reinterpret_cast
  // at the call site, matching the prior code path.
  TmaWarpSpecializedGroupedGemmInput::ElementSF const** sf_A =
      tma_inputs.swap_ab ? tma_inputs.fpX_block_scaling_factors_weight
                         : tma_inputs.fpX_block_scaling_factors_act;
  TmaWarpSpecializedGroupedGemmInput::ElementSF const** sf_B =
      tma_inputs.swap_ab ? tma_inputs.fpX_block_scaling_factors_act
                         : tma_inputs.fpX_block_scaling_factors_weight;
  void* sf_stride_A =
      tma_inputs.swap_ab ? tma_inputs.fpX_block_scaling_factors_stride_weight
                         : tma_inputs.fpX_block_scaling_factors_stride_act;
  void* sf_stride_B =
      tma_inputs.swap_ab ? tma_inputs.fpX_block_scaling_factors_stride_act
                         : tma_inputs.fpX_block_scaling_factors_stride_weight;

  // Full arguments - following step22 pattern: inline brace initialization
  // Standard Mode: D = A @ W
  // Operand A = activations (FP8), Operand B = weights (FP4)
  typename Gemm::Arguments arguments = {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      tma_inputs.shape_info,
      // Mainloop arguments (inline, not as CollectiveMainloop::Arguments)
      {reinterpret_cast<ElementInputA const**>(ptr_A),
       reinterpret_cast<StrideA>(stride_A),
       reinterpret_cast<ElementInputB const**>(ptr_B),
       reinterpret_cast<StrideB>(stride_B),
       reinterpret_cast<ElementSF const**>(sf_A),
       reinterpret_cast<LayoutSFA>(sf_stride_A),
       reinterpret_cast<ElementSF const**>(sf_B),
       reinterpret_cast<LayoutSFB>(sf_stride_B)},
      // Epilogue arguments (inline)
      {{},  // thread args (will set alpha/beta below)
       nullptr,  // C ptr (not used)
       nullptr,  // C stride (not used)
       reinterpret_cast<ElementD**>(tma_inputs.ptr_d),
       reinterpret_cast<StrideD>(tma_inputs.stride_d)},
      hw_info};
  
  // Set epilogue thread args
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  // If only workspace size is requested, return early
  if (workspace_size != nullptr) {
    *workspace_size = gemm.get_workspace_size(arguments);
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] workspace_size=%zu", *workspace_size);
    return;
  }

  // Validate inputs
  TLLM_CHECK_WITH_INFO(tma_inputs.isValid(), "SM120 MXFP4 MoE: Invalid TMA inputs");

  // Check can_implement
  auto can_impl = gemm.can_implement(arguments);
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] can_implement=%s num_groups=%d", 
      cutlassGetStatusString(can_impl), tma_inputs.shape_info.groups());
  if (can_impl != cutlass::Status::kSuccess) {
    TLLM_THROW("SM120 MXFP4 MoE: can_implement failed: %s", cutlassGetStatusString(can_impl));
  }

  // Validate workspace size
  size_t required_workspace = gemm.get_workspace_size(arguments);
  if (tma_inputs.gemm_workspace_size < required_workspace) {
    TLLM_THROW("SM120 MXFP4 MoE: workspace too small (%zu < %zu)",
               tma_inputs.gemm_workspace_size, required_workspace);
  }

  // Initialize GEMM
  auto init_status = gemm.initialize(arguments, tma_inputs.gemm_workspace, stream);
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] initialize=%s workspace=%p size=%zu",
      cutlassGetStatusString(init_status), tma_inputs.gemm_workspace, 
      tma_inputs.gemm_workspace_size);
  if (init_status != cutlass::Status::kSuccess) {
    TLLM_THROW("SM120 MXFP4 MoE: initialize failed: %s", cutlassGetStatusString(init_status));
  }

  // Run GEMM
  auto run_status = gemm.run(stream);
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] run=%s", cutlassGetStatusString(run_status));
  if (run_status != cutlass::Status::kSuccess) {
    TLLM_THROW("SM120 MXFP4 MoE: run failed: %s", cutlassGetStatusString(run_status));
  }

  if (occupancy != nullptr) {
    *occupancy = 1;  // Placeholder
  }

#else
  TLLM_THROW("SM120 MoE GEMM requires CUTLASS_ARCH_MMA_SM12x_SUPPORTED and ENABLE_FP4");
#endif
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels_oss
