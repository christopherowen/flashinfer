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

// Optional debug dumping (host-side) for SM120 TMA/stride/layout objects.
#include <cstdlib>
#include <sstream>
#include <type_traits>

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
/* Epilogue tile: ensure EPI_TILE_N divides CTA_N (= TILE_N_VAL).                                 \
 *                                                                                                 \
 * SM90/SM120 TMA warp-specialized epilogues require exact partitioning (no remainder predication):\
 *   EPI_TILE_N | CTA_N                                                                           \
 * For small CTA_N like 16, EpilogueTileAuto may pick N=32 and fail to compile.                    \
 */                                                                                                \
constexpr int kEpiTileN = (TILE_N_VAL < 32 ? TILE_N_VAL : 32);                                    \
using EpilogueTile_MN = cute::tuple<cute::C<64>, cute::C<kEpiTileN>>;                              \
                                                                                                  \
/* Epilogue collective */                                                                         \
using CollectiveEpilogue =                                                                        \
    typename cutlass::epilogue::collective::CollectiveBuilder<                                    \
        ArchTag, OperatorClass, TileShape_MNK, ClusterShape_MNK,                                  \
        EpilogueTile_MN, ElementAccumulator, ElementCompute,                                      \
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
/* Alignment requirements - FP4 operand needs special 128 alignment (same as standard mode) */  \
constexpr int AlignmentA = 128;  /* Special-case for FP4 (ElementA is weight in transposed) */   \
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;  /* 16 for FP8 */   \
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;                          \
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;                          \
                                                                                                  \
/* Tile shape: Virtual M=128 (fixed), N=TILE_N_VAL (small token dim), K=128 */                    \
using TileShape_MNK = Shape<_128, cute::Int<TILE_N_VAL>, _128>;                                   \
using ClusterShape_MNK = Shape<_1, _1, _1>;                                                       \
                                                                                                  \
/* Epilogue tile: ensure EPI_TILE_N divides CTA_N (= TILE_N_VAL). */                              \
constexpr int kEpiTileN = (TILE_N_VAL < 32 ? TILE_N_VAL : 32);                                    \
using EpilogueTile_MN = cute::tuple<cute::C<64>, cute::C<kEpiTileN>>;                              \
                                                                                                  \
/* Epilogue collective */                                                                         \
using CollectiveEpilogue =                                                                        \
    typename cutlass::epilogue::collective::CollectiveBuilder<                                    \
        ArchTag, OperatorClass, TileShape_MNK, ClusterShape_MNK,                                  \
        EpilogueTile_MN, ElementAccumulator, ElementCompute,                                      \
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

  // swap_ab for the kernel is baked in by the compile-time SWAP_AB flag (set by Python JIT).
  // At runtime, `tma_inputs.swap_ab` must match because it controls how ptr/stride arrays are
  // interpreted and how M/N are written into the grouped problem shapes.
  //
  // However, during workspace-size queries the runtime `tma_inputs.swap_ab` may reflect the
  // default heuristic profile (often swap_ab=false) even when this module was compiled with
  // SWAP_AB=1. In that case, we force it to the compiled mode for the workspace query path.
  constexpr bool kSwapAB = SWAP_AB;
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] swap_ab (compiled)=%s swap_ab (runtime)=%s",
      kSwapAB ? "true" : "false",
      tma_inputs.swap_ab ? "true" : "false");

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

  // ==========================================================================
  // TYPE SAFETY CHECKS
  // ==========================================================================
  // The SM120 block-scaled collective expects SINGLE stride values (uniform for all groups),
  // while TmaWarpSpecializedGroupedGemmInput stores stride ARRAYS (one per group).
  // We verify the stored types are compatible with what we expect to dereference.
  //
  // Key insight: For MoE, all experts have the same weight dimensions, so strides are uniform.
  // We dereference stride_A[0] to get the shared stride value.
  static_assert(sizeof(StrideA) == sizeof(TmaWarpSpecializedGroupedGemmInput::StrideA),
      "SM120 StrideA size must match TmaWsInput::StrideA");
  static_assert(sizeof(StrideB) == sizeof(TmaWarpSpecializedGroupedGemmInput::StrideB),
      "SM120 StrideB size must match TmaWsInput::StrideB");
  static_assert(sizeof(StrideD) == sizeof(TmaWarpSpecializedGroupedGemmInput::StrideD),
      "SM120 StrideD size must match TmaWsInput::StrideD");
  
  // If only workspace size is requested, use minimal arguments (strides may be NULL).
  // Force runtime swap_ab to match compiled mode in this path.
  if (workspace_size != nullptr) {
    tma_inputs.swap_ab = kSwapAB;
    typename Gemm::Arguments ws_arguments = {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        tma_inputs.shape_info,
        // Mainloop args with default strides (workspace query doesn't need actual strides)
        {nullptr, StrideA{}, nullptr, StrideB{}, nullptr, LayoutSFA{}, nullptr, LayoutSFB{}},
        // Epilogue args with default strides
        {{}, nullptr, nullptr, nullptr, StrideD{}},
        hw_info};
    ws_arguments.epilogue.thread.alpha = 1.0f;
    ws_arguments.epilogue.thread.beta = 0.0f;
    *workspace_size = gemm.get_workspace_size(ws_arguments);
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] workspace_size=%zu", *workspace_size);
    return;
  }

  // For actual execution paths, require runtime swap_ab to match compiled mode.
  TLLM_CHECK_WITH_INFO(
      tma_inputs.swap_ab == kSwapAB,
      "SM120 MXFP4 MoE: swap_ab mismatch between compiled kernel (SWAP_AB) and runtime inputs");

  // Validate stride pointers before dereference
  TLLM_CHECK_WITH_INFO(stride_A != nullptr, "SM120 MXFP4 MoE: stride_A is null");
  TLLM_CHECK_WITH_INFO(stride_B != nullptr, "SM120 MXFP4 MoE: stride_B is null");
  TLLM_CHECK_WITH_INFO(tma_inputs.stride_d != nullptr, "SM120 MXFP4 MoE: stride_d is null");
  TLLM_CHECK_WITH_INFO(sf_stride_A != nullptr, "SM120 MXFP4 MoE: sf_stride_A is null");
  TLLM_CHECK_WITH_INFO(sf_stride_B != nullptr, "SM120 MXFP4 MoE: sf_stride_B is null");

  // ==========================================================================
  // DEBUG: Dump first-group (group 0) packed stride + SF layout objects
  // ==========================================================================
  // Enable with:
  //   export TLLM_SM120_MOE_DUMP_TMA=1
  //
  // This copies the packed-stride/layout objects from device workspace to host
  // and prints them. This is intended to help diagnose swap_ab crashes where
  // the transposed SM120 kernel expects a different stride/layout "shape order"
  // than what the stride-fill kernel wrote.
  if (auto const* dump_env = std::getenv("TLLM_SM120_MOE_DUMP_TMA");
      dump_env != nullptr && dump_env[0] == '1') {
    // Print compile-time type/size relationships (helps confirm mismatches).
    using TmaInput = TmaWarpSpecializedGroupedGemmInput;
    using Mxfp4LayoutSF = typename TmaInput::MXFPXBlockScaledConfig::LayoutSF;
    using KernelStrideA = StrideA;
    using KernelStrideB = StrideB;
    using KernelStrideD = StrideD;
    using KernelStrideAVal = std::remove_pointer_t<KernelStrideA>;
    using KernelStrideBVal = std::remove_pointer_t<KernelStrideB>;
    using KernelStrideDVal = std::remove_pointer_t<KernelStrideD>;
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] sizeof(StrideA)=%zu sizeof(TmaInput::StrideA)=%zu",
        sizeof(StrideA), sizeof(typename TmaInput::StrideA));
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] sizeof(StrideB)=%zu sizeof(TmaInput::StrideB)=%zu",
        sizeof(StrideB), sizeof(typename TmaInput::StrideB));
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] sizeof(StrideD)=%zu sizeof(TmaInput::StrideD_T)=%zu sizeof(TmaInput::StrideD)=%zu",
        sizeof(StrideD), sizeof(typename TmaInput::StrideD_T),
        sizeof(typename TmaInput::StrideD));
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] is_same(remove_ptr(KernelStrideA), TmaInput::StrideA)=%d",
        int(std::is_same_v<KernelStrideAVal, typename TmaInput::StrideA>));
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] is_same(remove_ptr(KernelStrideB), TmaInput::StrideB)=%d",
        int(std::is_same_v<KernelStrideBVal, typename TmaInput::StrideB>));
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] is_same(remove_ptr(KernelStrideD), TmaInput::StrideD)=%d is_same(remove_ptr(KernelStrideD), TmaInput::StrideD_T)=%d",
        int(std::is_same_v<KernelStrideDVal, typename TmaInput::StrideD>),
        int(std::is_same_v<KernelStrideDVal, typename TmaInput::StrideD_T>));
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] sizeof(LayoutSFA)=%zu sizeof(MXFPX LayoutSF)=%zu",
        sizeof(LayoutSFA), sizeof(Mxfp4LayoutSF));
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] sizeof(LayoutSFB)=%zu sizeof(MXFPX LayoutSF)=%zu",
        sizeof(LayoutSFB), sizeof(Mxfp4LayoutSF));
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] is_same(LayoutSFA, MXFPX LayoutSF)=%d is_same(LayoutSFB, MXFPX LayoutSF)=%d",
        int(std::is_same_v<LayoutSFA, Mxfp4LayoutSF>), int(std::is_same_v<LayoutSFB, Mxfp4LayoutSF>));
    TLLM_LOG_DEBUG(
        "[SM120 MXFP4 MoE][dump] is_same(remove_ptr(LayoutSFA), MXFPX LayoutSF)=%d is_same(remove_ptr(LayoutSFB), MXFPX LayoutSF)=%d",
        int(std::is_same_v<std::remove_pointer_t<LayoutSFA>, Mxfp4LayoutSF>),
        int(std::is_same_v<std::remove_pointer_t<LayoutSFB>, Mxfp4LayoutSF>));

    // Copy group-0 problem shape and stride/layout objects to host.
    // Note: problem_shapes is device memory.
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
    UnderlyingProblemShape host_problem_shape{};
    // The stride-fill kernel writes TmaInput::{StrideA,StrideB,StrideD/StrideD_T} into these arrays.
    // Do NOT use the SM120 kernel's Stride* aliases here; they may differ (and that's what we're diagnosing).
    using HostStrideA = typename TmaInput::StrideA;
    using HostStrideB = typename TmaInput::StrideB;
    using HostStrideD = std::conditional_t<kSwapAB, typename TmaInput::StrideD_T, typename TmaInput::StrideD>;
    HostStrideA host_strideA{};
    HostStrideB host_strideB{};
    HostStrideD host_strideD{};
    Mxfp4LayoutSF host_layout_sfa{};
    Mxfp4LayoutSF host_layout_sfb{};

    auto memcpy_and_sync = [&](void* dst, void const* src, size_t bytes, char const* what) {
      cudaError_t st = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
      if (st != cudaSuccess) {
        TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] cudaMemcpyAsync(%s) failed: %s", what,
                       cudaGetErrorString(st));
        return false;
      }
      st = cudaStreamSynchronize(stream);
      if (st != cudaSuccess) {
        TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] cudaStreamSynchronize(%s) failed: %s", what,
                       cudaGetErrorString(st));
        return false;
      }
      return true;
    };

    // Problem shape 0
    (void)memcpy_and_sync(&host_problem_shape, tma_inputs.shape_info.problem_shapes,
                          sizeof(host_problem_shape), "problem_shape[0]");
    // Strides/layouts 0
    (void)memcpy_and_sync(&host_strideA, stride_A, sizeof(host_strideA), "strideA[0]");
    (void)memcpy_and_sync(&host_strideB, stride_B, sizeof(host_strideB), "strideB[0]");
    (void)memcpy_and_sync(&host_strideD, tma_inputs.stride_d, sizeof(host_strideD), "strideD[0]");
    (void)memcpy_and_sync(&host_layout_sfa, sf_stride_A, sizeof(host_layout_sfa), "layoutSFA[0]");
    (void)memcpy_and_sync(&host_layout_sfb, sf_stride_B, sizeof(host_layout_sfb), "layoutSFB[0]");

    std::ostringstream oss_ps;
    oss_ps << host_problem_shape;
    std::ostringstream oss_sa;
    oss_sa << host_strideA;
    std::ostringstream oss_sb;
    oss_sb << host_strideB;
    std::ostringstream oss_sd;
    oss_sd << host_strideD;
    std::ostringstream oss_lsfa;
    oss_lsfa << host_layout_sfa;
    std::ostringstream oss_lsfb;
    oss_lsfb << host_layout_sfb;

    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] group0 problem_shape=%s", oss_ps.str().c_str());
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] group0 strideA=%s", oss_sa.str().c_str());
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] group0 strideB=%s", oss_sb.str().c_str());
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] group0 strideD=%s", oss_sd.str().c_str());
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] group0 layoutSFA=%s", oss_lsfa.str().c_str());
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] group0 layoutSFB=%s", oss_lsfb.str().c_str());

    // ------------------------------------------------------------------------
    // Stride sanity checks: print expected vs actual (swap mode)
    // ------------------------------------------------------------------------
    // NOTE: This checks only the 2D matrix stride components and is intentionally
    // conservative. It is meant to catch the most common “shape order” mismatch
    // that can yield invalid TMA descriptors and illegal-instruction traps.
    //
    // For swap_ab transposed mode, the compiled SM120 kernel namespace defines:
    // - LayoutA = RowMajor  (A = weights in swap mode)
    // - LayoutB = ColumnMajor (B = activations in swap mode)
    // - StrideD uses TmaInput::StrideD_T (transposed output)
    //
    // The group problem shape is (M,N,K) for the GEMM kernel.
    auto to_i64 = [](auto x) -> int64_t {
      using X = std::remove_cv_t<std::remove_reference_t<decltype(x)>>;
      if constexpr (std::is_integral_v<X>) {
        return static_cast<int64_t>(x);
      } else {
        // cute::C<N> has a ::value
        return static_cast<int64_t>(X::value);
      }
    };

    // UnderlyingProblemShape prints as "(M,N,K)".
    int64_t const ps_m = to_i64(cute::get<0>(host_problem_shape));
    int64_t const ps_n = to_i64(cute::get<1>(host_problem_shape));
    int64_t const ps_k = to_i64(cute::get<2>(host_problem_shape));

    auto stride2_str = [&](int64_t s0, int64_t s1) {
      std::ostringstream oss;
      oss << "(" << s0 << "," << s1 << ")";
      return oss.str();
    };

    // Extract the first two stride components.
    int64_t const a_s0 = to_i64(cute::get<0>(host_strideA));
    int64_t const a_s1 = to_i64(cute::get<1>(host_strideA));
    int64_t const b_s0 = to_i64(cute::get<0>(host_strideB));
    int64_t const b_s1 = to_i64(cute::get<1>(host_strideB));
    int64_t const d_s0 = to_i64(cute::get<0>(host_strideD));
    int64_t const d_s1 = to_i64(cute::get<1>(host_strideD));

    // Expected (swap_ab compiled kernel):
    // NOTE: CUTLASS "B" stride types are in (N,K) mode order (see `cutlass/detail/layout.hpp`
    // TagToStrideB* specializations), so packed ColumnMajor B is (K,1), not (1,K).
    //
    // - A (RowMajor)      modes [M,K] => strides (K,1)
    // - B (ColumnMajor)   modes [N,K] => strides (K,1)
    // - D_T (ColumnMajor) modes [M,N] => strides (1,M)
    int64_t const exp_a_s0 = ps_k;
    int64_t const exp_a_s1 = 1;
    int64_t const exp_b_s0 = ps_k;
    int64_t const exp_b_s1 = 1;
    int64_t const exp_d_s0 = 1;
    int64_t const exp_d_s1 = ps_m;

    auto log_match = [&](char const* name, int64_t act0, int64_t act1, int64_t exp0, int64_t exp1) {
      bool ok = (act0 == exp0) && (act1 == exp1);
      TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] %s stride expected=%s actual=%s %s (ps_m=%ld ps_n=%ld ps_k=%ld)",
                     name,
                     stride2_str(exp0, exp1).c_str(),
                     stride2_str(act0, act1).c_str(),
                     ok ? "OK" : "MISMATCH",
                     (long)ps_m, (long)ps_n, (long)ps_k);
    };

    if (kSwapAB) {
      log_match("A(RowMajor MxK)", a_s0, a_s1, exp_a_s0, exp_a_s1);
      log_match("B(ColumnMajor NxK)", b_s0, b_s1, exp_b_s0, exp_b_s1);
      log_match("D_T(ColumnMajor MxN)", d_s0, d_s1, exp_d_s0, exp_d_s1);
    } else {
      // Standard (non-swap) path is already known-good. Still print the raw
      // numbers to help compare across modes.
      TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] (non-swap) ps_m=%ld ps_n=%ld ps_k=%ld A=(%ld,%ld) B=(%ld,%ld) D=(%ld,%ld)",
                     (long)ps_m, (long)ps_n, (long)ps_k,
                     (long)a_s0, (long)a_s1,
                     (long)b_s0, (long)b_s1,
                     (long)d_s0, (long)d_s1);
    }
  }
  
  // Full arguments for actual kernel run
  // IMPORTANT: CUTLASS grouped GEMM expects stride/layout *arrays* for grouped mode.
  // Depending on the CollectiveMainloop/Epilogue type aliases, these may be represented as
  // pointer types (e.g., StrideA == TmaInputStrideA*) or value types. Handle both.
  auto strideA_arg = [&]() -> StrideA {
    if constexpr (std::is_pointer_v<StrideA>) {
      return reinterpret_cast<StrideA>(stride_A);
    } else {
      return *reinterpret_cast<StrideA const*>(stride_A);
    }
  }();
  auto strideB_arg = [&]() -> StrideB {
    if constexpr (std::is_pointer_v<StrideB>) {
      return reinterpret_cast<StrideB>(stride_B);
    } else {
      return *reinterpret_cast<StrideB const*>(stride_B);
    }
  }();
  auto strideD_arg = [&]() -> StrideD {
    if constexpr (std::is_pointer_v<StrideD>) {
      return reinterpret_cast<StrideD>(tma_inputs.stride_d);
    } else {
      return *reinterpret_cast<StrideD const*>(tma_inputs.stride_d);
    }
  }();
  auto layoutSFA_arg = [&]() -> LayoutSFA {
    if constexpr (std::is_pointer_v<LayoutSFA>) {
      return reinterpret_cast<LayoutSFA>(sf_stride_A);
    } else {
      return *reinterpret_cast<LayoutSFA const*>(sf_stride_A);
    }
  }();
  auto layoutSFB_arg = [&]() -> LayoutSFB {
    if constexpr (std::is_pointer_v<LayoutSFB>) {
      return reinterpret_cast<LayoutSFB>(sf_stride_B);
    } else {
      return *reinterpret_cast<LayoutSFB const*>(sf_stride_B);
    }
  }();

  typename Gemm::Arguments arguments = {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      tma_inputs.shape_info,
      // Mainloop arguments
      {reinterpret_cast<ElementInputA const**>(ptr_A),
       strideA_arg,
       reinterpret_cast<ElementInputB const**>(ptr_B),
       strideB_arg,
       reinterpret_cast<ElementSF const**>(sf_A),
       layoutSFA_arg,
       reinterpret_cast<ElementSF const**>(sf_B),
       layoutSFB_arg},
      // Epilogue arguments
      {{},  // thread args (will set alpha/beta below)
       nullptr,  // C ptr (not used)
       nullptr,  // C stride (not used)
       reinterpret_cast<ElementD**>(tma_inputs.ptr_d),
       strideD_arg},
      hw_info};
  
  // Set epilogue thread args
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

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

  // Extra debug: record launch grid shape (helps correlate SASS plane-strides to C++ indexing)
  if (auto const* dump_env = std::getenv("TLLM_SM120_MOE_DUMP_TMA");
      dump_env != nullptr && dump_env[0] == '1') {
    dim3 grid = Gemm::get_grid_shape(arguments, tma_inputs.gemm_workspace);
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] grid=(%u,%u,%u) hw_info.sm_count=%d",
                   grid.x, grid.y, grid.z, hw_info.sm_count);
  }

  // Extra debug: dump tensormap workspace bytes (A/B/SFA/SFB planes) for one SM slot.
  // Enable with:
  //   export TLLM_SM120_MOE_DUMP_TENSORMAPS=1
  // Optional:
  //   export TLLM_SM120_MOE_DUMP_TENSORMAPS_POSTRUN=1
  //
  // Rationale:
  //  - We observed a consistent trap at a `UTMALDG.4D ... [UR44] ...` instruction where
  //    `UR44/UR46/UR48/UR50` look like four 128B-strided "planes" indexed by `gridDim.x` (== sm_count).
  //  - CUTLASS uses a 4-plane per-SM tensormap workspace (A, B, SFA, SFB) in the SM120 block-scaled mainloop.
  //  - Dumping the raw descriptor bytes lets us quickly see if plane 4 (SFB) is malformed/uninitialized.
  if (auto const* dump_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS");
      dump_env != nullptr && dump_env[0] == '1') {
    auto dump_tensormaps = [&](const char* tag) {
      constexpr size_t kDescBytes = 128;  // sizeof(cute::TmaDescriptor) on these kernels
      const int sm_count = hw_info.sm_count;
      const size_t expected_bytes = size_t(4) * size_t(sm_count) * kDescBytes;
      const size_t copy_bytes = std::min(expected_bytes, tma_inputs.gemm_workspace_size);

      std::vector<uint8_t> host(copy_bytes, 0);
      cudaError_t copy_err = cudaMemcpyAsync(
          host.data(), tma_inputs.gemm_workspace, copy_bytes, cudaMemcpyDeviceToHost, stream);
      cudaError_t sync_err = cudaStreamSynchronize(stream);

      TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] tensormaps(%s): workspace=%p copy_bytes=%zu expected=%zu memcpy=%s sync=%s",
                     tag, tma_inputs.gemm_workspace, copy_bytes, expected_bytes,
                     cudaGetErrorString(copy_err), cudaGetErrorString(sync_err));

      auto format_hex16 = [](const uint8_t* p) {
        std::array<char, 16 * 3 + 1> out{};
        size_t off = 0;
        for (int i = 0; i < 16; ++i) {
          off += std::snprintf(out.data() + off, out.size() - off, "%02x%s",
                               unsigned(p[i]), (i == 15) ? "" : " ");
        }
        return out;
      };

      // By default dump the slot corresponding to sm_idx=0 (matches the plane indexing formula).
      int sm_idx = 0;
      if (auto const* idx_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS_SMIDX");
          idx_env != nullptr && idx_env[0] != '\0') {
        sm_idx = std::atoi(idx_env);
      }
      sm_idx = std::max(0, std::min(sm_idx, sm_count - 1));

      auto dump_plane = [&](int plane, const char* name) {
        const size_t idx = size_t(sm_idx) + size_t(plane) * size_t(sm_count);
        const size_t byte_off = idx * kDescBytes;
        if (byte_off + kDescBytes > host.size()) {
          TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] tensormaps(%s)[%s] plane=%d sm_idx=%d: OOB (byte_off=%zu host=%zu)",
                         tag, name, plane, sm_idx, byte_off, host.size());
          return;
        }
        const uint8_t* p = host.data() + byte_off;
        for (size_t line = 0; line < kDescBytes; line += 16) {
          auto hex = format_hex16(p + line);
          TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] tensormaps(%s)[%s] plane=%d sm_idx=%d off=%zu bytes[%zu..%zu]=%s",
                         tag, name, plane, sm_idx, byte_off, line, line + 15, hex.data());
        }
      };

      dump_plane(0, "A");
      dump_plane(1, "B");
      dump_plane(2, "SFA");
      dump_plane(3, "SFB");
    };

    dump_tensormaps("pre_run");

    // If requested, dump again after a successful run (useful for comparing to failing tiles).
    // Note: the second dump happens below once `run_status` is known.
    (void)dump_tensormaps;
  }

  // Run GEMM
  auto run_status = gemm.run(stream);
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] run=%s", cutlassGetStatusString(run_status));
  if (auto const* dump_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS_POSTRUN");
      dump_env != nullptr && dump_env[0] == '1' &&
      std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS") != nullptr &&
      run_status == cutlass::Status::kSuccess) {
    // Re-run the same dump logic by toggling the main flag (the lambda is in the other block).
    // We simply re-enter the block by duplicating minimal logic here (avoids refactoring further).
    constexpr size_t kDescBytes = 128;
    const int sm_count = hw_info.sm_count;
    const size_t expected_bytes = size_t(4) * size_t(sm_count) * kDescBytes;
    const size_t copy_bytes = std::min(expected_bytes, tma_inputs.gemm_workspace_size);

    std::vector<uint8_t> host(copy_bytes, 0);
    cudaError_t copy_err = cudaMemcpyAsync(
        host.data(), tma_inputs.gemm_workspace, copy_bytes, cudaMemcpyDeviceToHost, stream);
    cudaError_t sync_err = cudaStreamSynchronize(stream);

    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] tensormaps(post_run): workspace=%p copy_bytes=%zu expected=%zu memcpy=%s sync=%s",
                   tma_inputs.gemm_workspace, copy_bytes, expected_bytes,
                   cudaGetErrorString(copy_err), cudaGetErrorString(sync_err));
  }
  if (run_status != cutlass::Status::kSuccess) {
    // Get the actual CUDA error for better diagnostics
    cudaError_t cuda_err = cudaGetLastError();
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] CUDA error after run: %s (%d)", 
                   cudaGetErrorString(cuda_err), static_cast<int>(cuda_err));
    TLLM_THROW("SM120 MXFP4 MoE: run failed: %s (CUDA: %s)", 
               cutlassGetStatusString(run_status), cudaGetErrorString(cuda_err));
  }

  if (occupancy != nullptr) {
    *occupancy = 1;  // Placeholder
  }

#else
  TLLM_THROW("SM120 MoE GEMM requires CUTLASS_ARCH_MMA_SM12x_SUPPORTED and ENABLE_FP4");
#endif
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels_oss
