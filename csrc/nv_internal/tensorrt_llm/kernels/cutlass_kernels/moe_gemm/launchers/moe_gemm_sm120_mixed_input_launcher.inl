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

// Gated FC1 kernel types
#include "cutlass_extensions/gemm/collective/sm120_blockscaled_mma_gated_array_tma.hpp"
#include "cutlass_extensions/gemm/kernel/sm120_gemm_gated_array_tma_warpspecialized.hpp"
#include "cutlass_extensions/epilogue/sm120_gated_swiglu_epilogue.hpp"

// =============================================================================
// Path A (two-GEMM bringup): custom epilogue fusion op for SwiGLU multiply
// =============================================================================
// Gate GEMM epilogue computes:
//   D = C * SiLU(alpha * acc)
// where:
//   - acc is the gate GEMM accumulator (A @ W_gate)
//   - C is the linear GEMM output (A @ W_linear), passed as epilogue source
//
// This avoids the standalone doGatedActivationKernel and avoids the 6-plane gated mainloop.
//
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

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

#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4) && defined(FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP)
namespace cutlass {
namespace epilogue {
namespace fusion {
using namespace cute;

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  cutlass::FloatRoundStyle RoundStyle_ = cutlass::FloatRoundStyle::round_to_nearest
>
struct SwigluMul : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  static constexpr bool IsSourceSupported = true;

  using ElementScalar = ElementScalar_;
  static constexpr int AlignmentScalar = 1;
  static constexpr auto RoundStyle = RoundStyle_;
};

// D = C * SiLU(alpha * acc)
template<
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest
>
using Sm90SwigluMulPtrArray =
  Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
    Sm90SrcFetch<ElementSource>, // C (linear)
    Sm90EVT<Sm90Compute<cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute, RoundStyle>,
      Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
        Sm90ScalarBroadcastPtrArray<ElementScalar, Stride<_0,_0,int64_t>>,
        Sm90AccFetch
      >
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  class ElementScalar,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC,
                                             StagesD,
                                             FragmentSize,
                                             ReuseSmemC,
                                             DelayTmaStore,
                                             NumEpilogueWarpGroups>,
    fusion::SwigluMul<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Sm90SwigluMulPtrArray<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
                         ElementCompute, ElementSource, ElementScalar, RoundStyle> {
  using Impl = Sm90SwigluMulPtrArray<typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
                                    ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::SwigluMul<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};

    using ActivationArguments =
        typename Sm90Compute<cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return {
        {}, // leaf args: C
        {   // unary: SiLU(alpha * acc)
          {  // binary: alpha * acc
            {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}},
            {},
            {}
          },
          activation
        },
        {} // binary: multiply
      };
    }
  };

  using Impl::Impl;
};

} // namespace fusion
} // namespace epilogue
} // namespace cutlass
#endif

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
/* Epilogue tile selection:                                                                       \
 * SM90/SM120 TMA warp-specialized epilogues require exact partitioning (no remainder predication):\
 *   EPI_TILE_N | CTA_N                                                                           \
 *                                                                                                 \
 * For large CTA_N (>= 64): use explicit (64, 64) for 2 iterations on N=128                       \
 *   - EpilogueTileAuto selects (64, 32) which gives 4 iterations - suboptimal                    \
 *   - (64, 64) gives 2 iterations, ~1.5% perf improvement                                        \
 *   - (64, 128) fails at runtime due to smem constraints                                         \
 * For small CTA_N (< 64): use explicit tile to ensure divisibility                               \
 */                                                                                                \
constexpr int kEpiTileN_Small = (TILE_N_VAL < 32 ? TILE_N_VAL : 32);                              \
constexpr int kEpiTileN_Large = 64;  /* Best trade-off: 2 iterations, fits in smem */             \
constexpr int kEpiTileN = (TILE_N_VAL >= 64) ? kEpiTileN_Large : kEpiTileN_Small;                 \
using EpilogueTile_MN = cute::tuple<cute::C<64>, cute::C<kEpiTileN>>; \
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

#if defined(FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP)
// Variant of the standard namespace where the epilogue computes:
//   D = C * SiLU(alpha * acc)
// with C provided as the epilogue source (linear GEMM output) and acc from gate GEMM.
#define DEFINE_SM120_MXFP4_STANDARD_SWIGLU_MUL_NAMESPACE(NAMESPACE_NAME, TILE_M_VAL, TILE_N_VAL, TILE_K_VAL) \
namespace NAMESPACE_NAME {                                                                        \
                                                                                                  \
using namespace cute;                                                                             \
                                                                                                  \
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
using ArchTag = cutlass::arch::Sm120;                                                             \
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;                                  \
                                                                                                  \
using LayoutA = cutlass::layout::RowMajor;                                                        \
using LayoutB = cutlass::layout::ColumnMajor;                                                     \
using LayoutC = cutlass::layout::RowMajor;                                                        \
                                                                                                  \
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;                      \
constexpr int AlignmentB = 128;                                                                   \
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;                           \
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;                           \
                                                                                                  \
using TileShape_MNK = Shape<cute::Int<TILE_M_VAL>, cute::Int<TILE_N_VAL>, cute::Int<TILE_K_VAL>>; \
using ClusterShape_MNK = Shape<_1, _1, _1>;                                                       \
                                                                                                  \
constexpr int kEpiTileN_Small = (TILE_N_VAL < 32 ? TILE_N_VAL : 32);                              \
constexpr int kEpiTileN_Large = 64;                                                               \
constexpr int kEpiTileN = (TILE_N_VAL >= 64) ? kEpiTileN_Large : kEpiTileN_Small;                 \
using EpilogueTile_MN = cute::tuple<cute::C<64>, cute::C<kEpiTileN>>;                              \
                                                                                                  \
using FusionOp = cutlass::epilogue::fusion::SwigluMul<ElementD, ElementCompute, ElementC, ElementCompute>; \
                                                                                                  \
using CollectiveEpilogue =                                                                        \
    typename cutlass::epilogue::collective::CollectiveBuilder<                                    \
        ArchTag, OperatorClass, TileShape_MNK, ClusterShape_MNK,                                  \
        EpilogueTile_MN, ElementAccumulator, ElementCompute,                                      \
        ElementC, LayoutC*, AlignmentC, ElementD, LayoutC*, AlignmentD,                           \
        cutlass::epilogue::collective::EpilogueScheduleAuto, FusionOp>::CollectiveOp;             \
                                                                                                  \
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<                             \
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;                        \
                                                                                                  \
using CollectiveMainloop =                                                                        \
    typename cutlass::gemm::collective::CollectiveBuilder<                                        \
        ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,   \
        ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCount,                          \
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;                   \
                                                                                                  \
using ProblemShape =                                                                              \
    cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;                           \
                                                                                                  \
using GemmKernel =                                                                                \
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>; \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                             \
                                                                                                  \
using StrideA = typename CollectiveMainloop::StrideA;                                             \
using StrideB = typename CollectiveMainloop::StrideB;                                             \
using StrideC = typename CollectiveEpilogue::StrideC;                                             \
using StrideD = typename CollectiveEpilogue::StrideD;                                             \
using LayoutSFA = typename CollectiveMainloop::LayoutSFA;                                         \
using LayoutSFB = typename CollectiveMainloop::LayoutSFB;                                         \
                                                                                                  \
}  /* namespace NAMESPACE_NAME */
#endif  // FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP

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
/* Epilogue tile: same conditional logic as standard namespace.                                   \
 * In transposed mode, TILE_N_VAL is the small token dimension (16, 32), so we use explicit tile. \
 */                                                                                                \
constexpr int kEpiTileN_Explicit = (TILE_N_VAL < 32 ? TILE_N_VAL : 32);                           \
using EpilogueTile_Explicit = cute::tuple<cute::C<64>, cute::C<kEpiTileN_Explicit>>;               \
using EpilogueTile_Auto = cutlass::epilogue::collective::EpilogueTileAuto;                        \
using EpilogueTile_MN = cute::conditional_t<(TILE_N_VAL >= 64), EpilogueTile_Auto, EpilogueTile_Explicit>; \
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
// MACRO: Gated Mode (FC1 with fused SwiGLU) - Minimal Bring-Up
// =============================================================================
// For minimal bring-up, we use the SAME kernel as standard mode but run it
// TWICE (once for linear, once for gate), then apply SwiGLU on CPU/separate kernel.
//
// This validates:
// 1. Weight pointer splitting (linear vs gate halves)
// 2. Scale factor pointer splitting
// 3. Correct N dimension (inter_size, not 2*inter_size)
// 4. doGatedActivation bypass logic
//
// Phase 2 will replace this with true dual-accumulator mainloop.
//
// CONSTRAINT: inter_size % 32 == 0 (SF_VEC_SIZE alignment)
//
// =============================================================================
// MACRO: Gated FC1 Mode
// =============================================================================
// Gated FC1 GEMM with fused SwiGLU activation:
// - Uses dual-accumulator pattern: A @ W_linear and A @ W_gate simultaneously
// - Epilogue applies SwiGLU: output = SiLU(gate) * linear
// - Output is BF16 [M, inter_size] (half the width of standard FC1)
//
// This kernel eliminates the standalone doGatedActivation kernel, saving one
// HBM round-trip and kernel launch overhead.
//
// CONSTRAINT: inter_size % 32 == 0 (SF_VEC_SIZE alignment)
//
#define DEFINE_SM120_MXFP4_GATED_NAMESPACE(NAMESPACE_NAME, TILE_M_VAL, TILE_N_VAL, TILE_K_VAL) \
namespace NAMESPACE_NAME {                                                                     \
                                                                                               \
using namespace cute;                                                                          \
                                                                                               \
/* Element types: FP8 activations × FP4 weights → BF16 output */                               \
using ElementInputA = cutlass::float_e4m3_t;                                                   \
using ElementInputB = cutlass::float_e2m1_t;                                                   \
using ElementA = cutlass::mx_float8_t<ElementInputA>;                                          \
using ElementB = cutlass::mx_float4_t<ElementInputB>;                                          \
using ElementAux = ElementB;  /* Gate weights same type as linear weights */                  \
                                                                                               \
using ElementC = cutlass::bfloat16_t;                                                          \
using ElementD = cutlass::bfloat16_t;                                                          \
using ElementOutput = cutlass::bfloat16_t;  /* SwiGLU output */                               \
                                                                                               \
using ElementAccumulator = float;                                                              \
using ElementCompute = float;                                                                  \
using ElementSF = cutlass::float_ue8m0_t;                                                      \
                                                                                               \
/* Architecture and operator class */                                                          \
using ArchTag = cutlass::arch::Sm120;                                                          \
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;                               \
                                                                                               \
/* Layouts (TN: A row-major, B column-major) */                                                \
using LayoutA = cutlass::layout::RowMajor;                                                     \
using LayoutB = cutlass::layout::ColumnMajor;                                                  \
using LayoutAux = LayoutB;  /* Gate weights same layout as linear */                          \
using LayoutC = cutlass::layout::RowMajor;                                                     \
using LayoutOutput = cutlass::layout::RowMajor;                                                \
                                                                                               \
/* Alignment requirements */                                                                   \
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;  /* 16 for FP8 */ \
constexpr int AlignmentB = 128;  /* Special-case for FP4 */                                   \
constexpr int AlignmentAux = AlignmentB;                                                      \
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;                        \
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;                        \
constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;              \
                                                                                               \
/* Tile shape (same as standard mode for occupancy parity) */                                  \
using TileShape_MNK = Shape<cute::Int<TILE_M_VAL>, cute::Int<TILE_N_VAL>, cute::Int<TILE_K_VAL>>; \
using ClusterShape_MNK = Shape<_1, _1, _1>;                                                    \
                                                                                               \
/* Problem shape for grouped GEMM (N is inter_size, not 2*inter_size) */                       \
using ProblemShape =                                                                           \
    cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;                        \
                                                                                               \
/* Output stride type for gated epilogue */                                                    \
using StrideOutput = cutlass::detail::TagToStrideC_t<LayoutOutput*>;                           \
                                                                                               \
/* Epilogue for gated path                                                                     \
 * Since SwiGLU is now applied INSIDE the mainloop (single-accumulator mma()),                 \
 * we can use the STANDARD epilogue - it just stores the already-fused result.                 \
 * This allows using the standard GemmUniversal kernel instead of GemmUniversalGated!          \
 *                                                                                             \
 * Epilogue tile: use same auto-selection as standard mode.                                    \
 * SM120 EpilogueTileAuto selects (64,32) which is the validated choice.                       \
 * Hardcoding (64,64) causes "illegal instruction" at runtime because the                      \
 * epilogue TMA store descriptor doesn't support 64-wide N on SM120.                           \
 */                                                                                            \
constexpr int kGatedEpiTileN = (TILE_N_VAL < 32 ? TILE_N_VAL : 32);                            \
using EpilogueTile_Explicit = cute::tuple<cute::C<64>, cute::C<kGatedEpiTileN>>;                \
using EpilogueTile_Auto = cutlass::epilogue::collective::EpilogueTileAuto;                      \
using EpilogueTile_MN = cute::conditional_t<(TILE_N_VAL >= 64),                                 \
                                             EpilogueTile_Auto, EpilogueTile_Explicit>;         \
using CollectiveEpilogue =                                                                     \
    typename cutlass::epilogue::collective::CollectiveBuilder<                                 \
        ArchTag, OperatorClass, TileShape_MNK, ClusterShape_MNK,                               \
        EpilogueTile_MN, ElementAccumulator, ElementCompute,                                   \
        ElementOutput, LayoutOutput*, AlignmentOutput, ElementOutput, LayoutOutput*, AlignmentOutput, \
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;                           \
                                                                                               \
/* Stage count (gated mainloop needs more SMEM for Aux operand). */                            \
/* CUTLASS SM120 block-scaled mainloop requires Stages >= 2. */                                \
constexpr int kGatedStages = 2;                                                                \
constexpr int kSchedulerPipelineStages = 1;                                                    \
                                                                                               \
/* Gated dispatch policy */                                                                    \
using GatedDispatchPolicy = cutlass::gemm::collective::MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated< \
    kGatedStages, kSchedulerPipelineStages, ClusterShape_MNK,                                  \
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>;                                  \
                                                                                               \
/* Build mainloop types using standard CollectiveBuilder, then adapt for gated */              \
using BaseMainloop =                                                                           \
    typename cutlass::gemm::collective::CollectiveBuilder<                                     \
        ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,\
        ElementAccumulator, TileShape_MNK, ClusterShape_MNK,                                   \
        cutlass::gemm::collective::StageCount<kGatedStages>,                                   \
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;                \
                                                                                               \
/* Gated mainloop - inherits from base and adds Aux operand + dual accumulator */             \
/* NOTE: Must pass the *Pair* and *Atoms* types (tuples), not individual extracted types */   \
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<                           \
    GatedDispatchPolicy,                                                                       \
    TileShape_MNK,                                                                             \
    typename BaseMainloop::ElementPairA,                                                       \
    typename BaseMainloop::StridePairA,                                                        \
    typename BaseMainloop::ElementPairB,                                                       \
    typename BaseMainloop::StridePairB,                                                        \
    typename BaseMainloop::TiledMma,                                                           \
    typename BaseMainloop::GmemTiledCopyPairA,    /* PAIR type, not singular */                \
    typename BaseMainloop::SmemLayoutAtomsA,      /* ATOMS type (plural), not singular */      \
    typename BaseMainloop::SmemCopyAtomsA,        /* ATOMS type (plural), not singular */      \
    cute::identity,                                                                            \
    typename BaseMainloop::GmemTiledCopyPairB,    /* PAIR type, not singular */                \
    typename BaseMainloop::SmemLayoutAtomsB,      /* ATOMS type (plural), not singular */      \
    typename BaseMainloop::SmemCopyAtomsB,        /* ATOMS type (plural), not singular */      \
    cute::identity>;                                                                           \
                                                                                               \
/* Gated GEMM kernel                                                                           \
 * Since SwiGLU is now applied inside the mainloop (via single-accumulator mma() overload),    \
 * we can use the STANDARD GemmUniversal kernel - no custom kernel needed!                     \
 * The mainloop's mma() computes both GEMMs, applies SiLU, and returns a single accumulator.   \
 */                                                                                            \
/* NOTE: Using standard GemmUniversal because:                                                 \
 *   - Gated mainloop's single-accumulator mma() applies SiLU INLINE                           \
 *   - load_init() returns tuple (compatible with standard kernel)                             \
 *   - Standard epilogue stores the already-fused result                                       \
 */                                                                                            \
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<                                       \
    ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;                               \
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                          \
                                                                                               \
/* Stride and layout types for grouped GEMM */                                                 \
using StrideA = typename CollectiveMainloop::StrideA;                                          \
using StrideB = typename CollectiveMainloop::StrideB;                                          \
using StrideAux = typename CollectiveMainloop::StrideAux;                                      \
using StrideD = typename CollectiveEpilogue::StrideD;  /* Match epilogue's expected type */           \
using LayoutSFA = typename CollectiveMainloop::LayoutSFA;                                      \
using LayoutSFB = typename CollectiveMainloop::LayoutSFB;                                      \
using LayoutSFAux = typename CollectiveMainloop::LayoutSFAux;                                  \
                                                                                               \
/* Gated mode marker */                                                                        \
constexpr bool kIsGatedMode = true;                                                            \
                                                                                               \
/* Compile-time validation (from user guidance point 6) */                                     \
static_assert(CollectiveMainloop::IsGated, "CollectiveMainloop must be gated");                \
static_assert(std::is_trivially_copyable_v<typename CollectiveMainloop::Params>,               \
              "Mainloop Params must be trivially copyable");                                   \
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
// GATED_FC1 flag (future): enables gated FC1 namespace for SwiGLU fusion
//

// Namespace instantiation based on SWAP_AB (set by Python JIT).
// Each JIT compilation creates exactly one namespace with the specified tiles.
//
// Naming convention:
//   sm120_mxfp4_bf16 - base name (arch, format, activation type)
//   sm120_mxfp4_bf16_gated - gated variant for FC1+SwiGLU fusion
//   No tile dimensions in name - they come from JIT flags
//
#if SWAP_AB
// Swapped mode (logical M < 128): uses transposed layout to work around tcgen05 M>=64 constraint
DEFINE_SM120_MXFP4_TRANSPOSED_NAMESPACE(sm120_mxfp4_bf16, LOGICAL_TILE_M)
#else
// Standard mode (logical M >= 128): normal layout
DEFINE_SM120_MXFP4_STANDARD_NAMESPACE(sm120_mxfp4_bf16, LOGICAL_TILE_M, LOGICAL_TILE_N, 128)
#endif

// Gated mode (FC1 with fused SwiGLU) - instantiated when FLASHINFER_GATED_FC1 is defined
// The gated mainloop uses 6 SMEM arrays (instead of 4), requiring ~50% more shared memory.
// GB10 (SM121) has 101KB shared memory.
//
// SMEM estimates (mainloop + epilogue) depend strongly on stage count:
//   - 1 stage enables CTA_N=128 tiles within SM121's 101KB limit (goal: avoid small-N TMA hazards)
//
// K=128 is required for TMA scale factor layout compatibility.
#ifdef FLASHINFER_GATED_FC1
#if !SWAP_AB
// Gated mode only supports standard (non-swapped) layout for now
// Default tile is 64x64x128 (fits SM121 SMEM).
// Optionally enable CTA_N=128 experimentation via FLASHINFER_GATED_FC1_CTA_N128.
#ifdef FLASHINFER_GATED_FC1_CTA_N128
DEFINE_SM120_MXFP4_GATED_NAMESPACE(sm120_mxfp4_bf16_gated, 64, 128, 128)
#else
DEFINE_SM120_MXFP4_GATED_NAMESPACE(sm120_mxfp4_bf16_gated, 64, 64, 128)
#endif
#endif
#endif  // FLASHINFER_GATED_FC1

#ifdef FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP
#if !SWAP_AB
// Gate GEMM variant epilogue: D = C * SiLU(alpha * acc)
DEFINE_SM120_MXFP4_STANDARD_SWIGLU_MUL_NAMESPACE(sm120_mxfp4_bf16_swiglu_mul, LOGICAL_TILE_M, LOGICAL_TILE_N, 128)
#endif
#endif  // FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP

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
    //
    // Important gotcha: CUTLASS packed-stride tuple ordering is often in "mode order"
    // (e.g. StrideB can be stored as [N,K] rather than [K,N]). That means the same
    // memory layout can appear as either:
    //  - ColumnMajor KxN: (sK=1, sN=K)  -> tuple (1, K) if the tuple is [K,N]
    //  - ColumnMajor KxN: (sN=K, sK=1)  -> tuple (K, 1) if the tuple is [N,K]
    //
    // To avoid false positives (and to answer "why is expected (1,K) vs actual (K,1)?"),
    // we accept either ordering but print which interpretation matched.
    auto log_match_2d = [&](char const* name,
                            int64_t act0, int64_t act1,
                            int64_t exp_dim0, int64_t exp_dim1,
                            char const* dims01, char const* dims10) {
      bool ok_01 = (act0 == exp_dim0) && (act1 == exp_dim1);
      bool ok_10 = (act0 == exp_dim1) && (act1 == exp_dim0);
      char const* verdict =
          ok_01 ? dims01 : (ok_10 ? dims10 : "MISMATCH");
      TLLM_LOG_DEBUG(
          "[SM120 MXFP4 MoE][dump] %s stride expected=%s (or swapped %s) actual=%s => %s (ps_m=%ld ps_n=%ld ps_k=%ld)",
          name,
          stride2_str(exp_dim0, exp_dim1).c_str(),
          stride2_str(exp_dim1, exp_dim0).c_str(),
          stride2_str(act0, act1).c_str(),
          verdict,
          (long)ps_m, (long)ps_n, (long)ps_k);
    };

    if (kSwapAB) {
      // A is RowMajor MxK: (sM=K, sK=1)
      log_match_2d("A(RowMajor MxK)", a_s0, a_s1, /*exp=*/ps_k, /*exp=*/1,
                   /*dims01=*/"OK as [M,K]",
                   /*dims10=*/"OK as [K,M] (tuple order swapped)");

      // B is ColumnMajor KxN: (sK=1, sN=K)
      log_match_2d("B(ColumnMajor KxN)", b_s0, b_s1, /*exp=*/1, /*exp=*/ps_k,
                   /*dims01=*/"OK as [K,N]",
                   /*dims10=*/"OK as [N,K] (tuple order swapped)");

      // D_T is ColumnMajor MxN: (sM=1, sN=M)
      log_match_2d("D_T(ColumnMajor MxN)", d_s0, d_s1, /*exp=*/1, /*exp=*/ps_m,
                   /*dims01=*/"OK as [M,N]",
                   /*dims10=*/"OK as [N,M] (tuple order swapped)");
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

  // Extra debug: split `initialize()` into its two conceptual phases.
  // This helps diagnose "Error Internal" during initialize for specific tiles.
  auto ws_init_status =
      Gemm::GemmKernel::initialize_workspace(arguments, tma_inputs.gemm_workspace, stream, nullptr);
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] initialize_workspace=%s",
                 cutlassGetStatusString(ws_init_status));
  if (ws_init_status != cutlass::Status::kSuccess) {
    cudaError_t cuda_err = cudaGetLastError();
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    TLLM_THROW("SM120 MXFP4 MoE: initialize_workspace failed: %s (cudaGetLastError=%s sync=%s)",
               cutlassGetStatusString(ws_init_status),
               cudaGetErrorString(cuda_err),
               cudaGetErrorString(sync_err));
  }

  // For CUTLASS 3.x kernels, `initialize()` will attempt to opt-in to large dynamic shared memory
  // via cudaFuncSetAttribute. If that call fails, CUTLASS returns kErrorInternal after clearing
  // the CUDA error state, which makes it hard to diagnose from the caller.
  //
  // Probe it explicitly here so we can see the exact smem size and the CUDA error.
  {
    int smem_size = int(Gemm::GemmKernel::SharedStorageSize);
    int optin_limit = -1;
    (void)cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] kernel SharedStorageSize=%dB cudaDevAttrMaxSharedMemoryPerBlockOptin=%dB",
                   smem_size, optin_limit);
    if (smem_size >= (48 << 10)) {
      cudaError_t attr_err = cudaFuncSetAttribute(
          cutlass::device_kernel<typename Gemm::GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (attr_err != cudaSuccess) {
        // Clear sticky error and throw with details.
        (void)cudaGetLastError();
        TLLM_THROW("SM120 MXFP4 MoE: cudaFuncSetAttribute(MaxDynamicSharedMemorySize=%d) failed: %s",
                   smem_size,
                   cudaGetErrorString(attr_err));
      }
    }
  }

  // Initialize GEMM
  auto init_status = gemm.initialize(arguments, tma_inputs.gemm_workspace, stream);
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] initialize=%s workspace=%p size=%zu",
      cutlassGetStatusString(init_status), tma_inputs.gemm_workspace, 
      tma_inputs.gemm_workspace_size);
  if (init_status != cutlass::Status::kSuccess) {
    // `initialize()` may fail after issuing internal CUDA work (e.g., device memcpys),
    // so capture CUDA error state to disambiguate CUTLASS vs CUDA-runtime failures.
    cudaError_t cuda_err = cudaGetLastError();
    cudaError_t sync_err = cudaStreamSynchronize(stream);

    // If requested, dump the tensormap region even on init failure. This helps
    // distinguish "descriptor not written at all" vs "descriptor written but invalid".
    if (auto const* dump_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS_ON_INIT_FAIL");
        dump_env != nullptr && dump_env[0] == '1') {
      constexpr size_t kDescBytes = 128;
      const int sm_count = hw_info.sm_count;
      const size_t expected_bytes = size_t(4) * size_t(sm_count) * kDescBytes;
      const size_t copy_bytes = std::min(expected_bytes, tma_inputs.gemm_workspace_size);

      std::vector<uint8_t> host(copy_bytes, 0);
      cudaError_t copy_err = cudaMemcpyAsync(
          host.data(), tma_inputs.gemm_workspace, copy_bytes, cudaMemcpyDeviceToHost, stream);
      cudaError_t dump_sync_err = cudaStreamSynchronize(stream);

      size_t base_off = 0;
      if (auto const* off_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS_OFFSET");
          off_env != nullptr && off_env[0] != '\0') {
        base_off = static_cast<size_t>(std::strtoull(off_env, nullptr, 0));
      }

      auto format_hex16 = [](const uint8_t* p) {
        std::array<char, 16 * 3 + 1> out{};
        size_t off = 0;
        for (int i = 0; i < 16; ++i) {
          off += std::snprintf(out.data() + off, out.size() - off, "%02x%s",
                               unsigned(p[i]), (i == 15) ? "" : " ");
        }
        return out;
      };

      TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] init_failed tensormaps: workspace=%p copy_bytes=%zu expected=%zu base_off=%zu memcpy=%s sync=%s",
                     tma_inputs.gemm_workspace, copy_bytes, expected_bytes, base_off,
                     cudaGetErrorString(copy_err), cudaGetErrorString(dump_sync_err));

      // Dump a few candidate planes for sm_idx=0.
      for (int plane = 0; plane <= 6; ++plane) {
        size_t idx = size_t(0) + size_t(plane) * size_t(sm_count);
        size_t byte_off = base_off + idx * kDescBytes;
        if (byte_off + 16 <= host.size()) {
          auto hex = format_hex16(host.data() + byte_off);
          TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] init_failed tensormaps[P%d] sm_idx=0 off=%zu bytes[0..15]=%s",
                         plane, byte_off, hex.data());
        }
      }
    }

    TLLM_THROW("SM120 MXFP4 MoE: initialize failed: %s (cudaGetLastError=%s sync=%s)",
               cutlassGetStatusString(init_status),
               cudaGetErrorString(cuda_err),
               cudaGetErrorString(sync_err));
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
  //   export TLLM_SM120_MOE_DUMP_TENSORMAPS_FULL=1   # copy full workspace (recommended)
  //   export TLLM_SM120_MOE_DUMP_TENSORMAPS_SLOT=300 # dump a specific 128B slot index
  //
  // Rationale:
  //  - We observed a consistent trap at a `UTMALDG.4D ... [UR44] ...` instruction where
  //    `UR44/UR46/UR48/UR50` look like four 128B-strided "planes" indexed by `gridDim.x` (== sm_count).
  //  - CUTLASS uses a 4-plane per-SM tensormap workspace (A, B, SFA, SFB) in the SM120 block-scaled mainloop.
  //  - Dumping the raw descriptor bytes lets us quickly see if plane 4 (SFB) is malformed/uninitialized.
  auto dump_tensormaps = [&](const char* tag) {
    constexpr size_t kDescBytes = 128;  // sizeof(cute::TmaDescriptor) on these kernels
    const int sm_count = hw_info.sm_count;
    const size_t expected_bytes = size_t(4) * size_t(sm_count) * kDescBytes;
    bool const dump_full = [&] {
      if (auto const* full_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS_FULL");
          full_env != nullptr && full_env[0] == '1') {
        return true;
      }
      return false;
    }();
    const size_t copy_bytes =
        dump_full ? tma_inputs.gemm_workspace_size
                  : std::min(expected_bytes, tma_inputs.gemm_workspace_size);

    std::vector<uint8_t> host(copy_bytes, 0);
    cudaError_t copy_err = cudaMemcpyAsync(
        host.data(), tma_inputs.gemm_workspace, copy_bytes, cudaMemcpyDeviceToHost, stream);
    cudaError_t sync_err = cudaStreamSynchronize(stream);

    // CUTLASS partitions the overall GEMM workspace; the SM120 mainloop's `Params::tensormaps`
    // pointer may point to an offset inside `gemm_workspace` (observed in cuda-gdb).
    // Allow overriding this offset so we can dump the correct region.
    size_t base_off = 0;
    if (auto const* off_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS_OFFSET");
        off_env != nullptr && off_env[0] != '\0') {
      base_off = static_cast<size_t>(std::strtoull(off_env, nullptr, 0));
    }

    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE][dump] tensormaps(%s): workspace=%p copy_bytes=%zu expected=%zu base_off=%zu memcpy=%s sync=%s",
                   tag, tma_inputs.gemm_workspace, copy_bytes, expected_bytes, base_off,
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

    // Optionally dump an explicit 128B slot (useful when you know URxx offset).
    if (auto const* slot_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS_SLOT");
        slot_env != nullptr && slot_env[0] != '\0') {
      long long slot_ll = std::atoll(slot_env);
      if (slot_ll >= 0) {
        size_t const slot = static_cast<size_t>(slot_ll);
        size_t const byte_off = base_off + slot * kDescBytes;
        if (byte_off + kDescBytes <= host.size()) {
          const uint8_t* p = host.data() + byte_off;
          for (size_t line = 0; line < kDescBytes; line += 16) {
            auto hex = format_hex16(p + line);
            TLLM_LOG_DEBUG(
                "[SM120 MXFP4 MoE][dump] tensormaps(%s)[SLOT] slot=%zu byte_off=%zu bytes[%zu..%zu]=%s",
                tag, slot, byte_off, line, line + 15, hex.data());
          }
        } else {
          TLLM_LOG_DEBUG(
              "[SM120 MXFP4 MoE][dump] tensormaps(%s)[SLOT] slot=%zu byte_off=%zu: OOB (host=%zu)",
              tag, slot, byte_off, host.size());
        }
      }
    }

    auto dump_plane = [&](int plane, const char* name) {
      const size_t idx = size_t(sm_idx) + size_t(plane) * size_t(sm_count);
      const size_t byte_off = base_off + idx * kDescBytes;
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

    // Historically we assumed 4 planes (A,B,SFA,SFB). However, on SM120/121 the workspace
    // may include additional planes; and cuda-gdb showed the faulting descriptor can live at
    // plane indices >= 3 (e.g., slot = plane*sm_count + sm_idx).
    //
    // Dump a wider range so we can correlate UR44/46/48/50 -> (plane, sm_idx) precisely.
    dump_plane(0, "P0");
    dump_plane(1, "P1");
    dump_plane(2, "P2");
    dump_plane(3, "P3");
    dump_plane(4, "P4");
    dump_plane(5, "P5");
    dump_plane(6, "P6");
  };

  bool const dump_pre = [&] {
    if (auto const* dump_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS");
        dump_env != nullptr && dump_env[0] == '1') {
      return true;
    }
    return false;
  }();
  bool const dump_post = [&] {
    if (auto const* dump_env = std::getenv("TLLM_SM120_MOE_DUMP_TENSORMAPS_POSTRUN");
        dump_env != nullptr && dump_env[0] == '1') {
      return true;
    }
    return false;
  }();

  if (dump_pre) {
    dump_tensormaps("pre_run");
  }

  // Run GEMM
  auto run_status = gemm.run(stream);
  TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] run=%s", cutlassGetStatusString(run_status));

  // IMPORTANT: Some failure modes (e.g., illegal-instruction traps in the kernel) can leave the CUDA
  // stream in an error state even if CUTLASS returns `Success`. Under debug, surface this immediately
  // so subsequent calls (memcpys / next GEMM) don't mask the root cause.
  if (dump_pre || dump_post) {
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    cudaError_t cuda_err = cudaGetLastError();
    if (sync_err != cudaSuccess || cuda_err != cudaSuccess) {
      TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] CUDA error after run (even though status=%s): %s (%d)",
                     cutlassGetStatusString(run_status),
                     cudaGetErrorString(cuda_err), static_cast<int>(cuda_err));
      if (dump_post) {
        dump_tensormaps("post_run_cuda_error");
      }
      TLLM_THROW("SM120 MXFP4 MoE: CUDA error after run: sync=%s getLastError=%s",
                 cudaGetErrorString(sync_err),
                 cudaGetErrorString(cuda_err));
    }
  }

  if (run_status != cutlass::Status::kSuccess) {
    // Get the actual CUDA error for better diagnostics
    cudaError_t cuda_err = cudaGetLastError();
    TLLM_LOG_DEBUG("[SM120 MXFP4 MoE] CUDA error after run: %s (%d)", 
                   cudaGetErrorString(cuda_err), static_cast<int>(cuda_err));

    if (dump_post) {
      // Attempt to dump workspace even on failure (may help diagnose partial initialization).
      dump_tensormaps("post_run_failure");
    }

    TLLM_THROW("SM120 MXFP4 MoE: run failed: %s (CUDA: %s)", 
               cutlassGetStatusString(run_status), cudaGetErrorString(cuda_err));
  }

  if (dump_post) {
    dump_tensormaps("post_run_success");
  }

  if (occupancy != nullptr) {
    *occupancy = 1;  // Placeholder
  }

#else
  TLLM_THROW("SM120 MoE GEMM requires CUTLASS_ARCH_MMA_SM12x_SUPPORTED and ENABLE_FP4");
#endif
}

// =============================================================================
// SM120 Gated FC1 Launcher (Layer 1A)
// =============================================================================
//
// This launcher runs the fused gated FC1 kernel with SwiGLU in the epilogue:
//   - Input: FP8 activations [M, K]
//   - Weights: FP4 [2*inter_size, K] (linear + gate halves)
//   - Output: BF16 [M, inter_size] (after SwiGLU)
//
// The kernel computes:
//   linear = A @ W_linear
//   gate = A @ W_gate
//   output = SiLU(gate) * linear
//
// This eliminates the standalone doGatedActivation kernel, saving:
//   - One HBM read of [M, 2*inter_size] BF16
//   - One HBM write of [M, inter_size] BF16
//   - One kernel launch
//
// Weight layout (per expert):
//   - W_linear: first inter_size columns
//   - W_gate: second inter_size columns
//   - Pointer offset: (inter_size * K / 2) bytes (FP4 packed)
//   - Scale offset: (inter_size * ceil(K/32)) elements (MXFP4)
//

#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4) && defined(FLASHINFER_GATED_FC1)

// Note: Gate weight/SF pointers and gated output pointers/strides are now
// pre-computed by the upstream computeStridesTmaWarpSpecializedKernel and
// stored in TmaWarpSpecializedGroupedGemmInput::gated_fc1.
// The separate computeGatedPointersAndStrides kernel and GatedFC1WorkspaceLayout
// have been removed - the gated launcher uses pre-computed arrays directly,
// eliminating the cudaStreamSynchronize that broke CUDA graph capture.

#endif  // CUTLASS_ARCH_MMA_SM12x_SUPPORTED && ENABLE_FP4 && FLASHINFER_GATED_FC1

// =============================================================================
// SM120 Two-GEMM Gated FC1 Launcher (Path A bringup)
// =============================================================================
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4) && defined(FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP)

template <typename ElementOutput, typename StrideD>
__global__ void computeOutputPointersAndStrides(
    ElementOutput* out_base,
    ElementOutput** __restrict__ ptr_out,
    StrideD* __restrict__ stride_out,
    int64_t const* __restrict__ expert_first_token_offset,
    int64_t ld_n,
    int num_experts) {
  int e = threadIdx.x + blockIdx.x * blockDim.x;
  if (e >= num_experts) return;
  int64_t tokens_before = expert_first_token_offset[e];
  ptr_out[e] = out_base + tokens_before * ld_n;
  StrideD s{};
  // Always set leading dimension (stride along M).
  cute::get<0>(s) = ld_n;
  // If the remaining stride components are runtime values, set them explicitly.
  // This avoids accidental (0,0,0) strides for some Stride types.
  if constexpr (std::is_lvalue_reference_v<decltype(cute::get<1>(s))>) {
    using T1 = std::remove_reference_t<decltype(cute::get<1>(s))>;
    if constexpr (std::is_integral_v<T1>) {
      cute::get<1>(s) = 1;
    }
  }
  if constexpr (std::is_lvalue_reference_v<decltype(cute::get<2>(s))>) {
    using T2 = std::remove_reference_t<decltype(cute::get<2>(s))>;
    if constexpr (std::is_integral_v<T2>) {
      cute::get<2>(s) = 0;
    }
  }
  stride_out[e] = s;
}

template <typename ElementOutput, typename StrideD>
__global__ void computeOutputPointersAndStridesConst(
    ElementOutput* out_base,
    ElementOutput const** __restrict__ ptr_out,
    StrideD* __restrict__ stride_out,
    int64_t const* __restrict__ expert_first_token_offset,
    int64_t ld_n,
    int num_experts) {
  int e = threadIdx.x + blockIdx.x * blockDim.x;
  if (e >= num_experts) return;
  int64_t tokens_before = expert_first_token_offset[e];
  ptr_out[e] = out_base + tokens_before * ld_n;
  StrideD s{};
  cute::get<0>(s) = ld_n;
  if constexpr (std::is_lvalue_reference_v<decltype(cute::get<1>(s))>) {
    using T1 = std::remove_reference_t<decltype(cute::get<1>(s))>;
    if constexpr (std::is_integral_v<T1>) {
      cute::get<1>(s) = 1;
    }
  }
  if constexpr (std::is_lvalue_reference_v<decltype(cute::get<2>(s))>) {
    using T2 = std::remove_reference_t<decltype(cute::get<2>(s))>;
    if constexpr (std::is_integral_v<T2>) {
      cute::get<2>(s) = 0;
    }
  }
  stride_out[e] = s;
}

__global__ void setProblemShapesN(
    cutlass::gemm::GroupProblemShape<cute::Shape<int64_t, int64_t, int64_t>>::UnderlyingProblemShape* problem_shapes,
    int num_experts,
    int64_t new_n) {
  int e = threadIdx.x + blockIdx.x * blockDim.x;
  if (e >= num_experts) return;
  auto ps = problem_shapes[e];
  cute::get<1>(ps) = new_n;
  problem_shapes[e] = ps;
}

template <typename ElementSF>
__global__ void computeGateWeightAndSfPointers(
    void const* const* __restrict__ ptr_weight_linear,
    void const** __restrict__ ptr_weight_gate,
    ElementSF const* const* __restrict__ sf_linear,
    ElementSF const** __restrict__ sf_gate,
    int64_t gate_weight_bytes,
    int64_t gate_sf_elems,
    int num_experts) {
  int e = threadIdx.x + blockIdx.x * blockDim.x;
  if (e >= num_experts) return;
#ifdef FLASHINFER_GATED_FC1_TWO_GEMM_NO_GATE_OFFSET
  (void)gate_weight_bytes;
  (void)gate_sf_elems;
  ptr_weight_gate[e] = ptr_weight_linear[e];
  sf_gate[e] = sf_linear[e];
#else
  auto w_linear = reinterpret_cast<char const*>(ptr_weight_linear[e]);
  ptr_weight_gate[e] = w_linear + gate_weight_bytes;
  sf_gate[e] = sf_linear[e] + gate_sf_elems;
#endif
}

struct TwoGemmFC1WorkspaceLayout {
  size_t gemm_workspace_offset;
  size_t gemm_workspace_size;
  size_t ptr_weight_gate_offset;
  size_t sf_gate_offset;
  size_t ptr_linear_offset;
  size_t ptr_linear_c_offset;
  size_t stride_linear_offset;
  size_t ptr_out_offset;
  size_t stride_out_offset;
  size_t total_size;

  template <class StrideT>
  static TwoGemmFC1WorkspaceLayout compute(size_t base_gemm_ws_size, int num_experts,
                                          size_t stride_c_elem_size) {
    TwoGemmFC1WorkspaceLayout layout{};
    constexpr size_t kAlign = 256;
    constexpr size_t kPtrAlign = 16;
    auto align_up = [](size_t x, size_t a) { return (x + a - 1) / a * a; };

    layout.gemm_workspace_offset = 0;
    layout.gemm_workspace_size = align_up(base_gemm_ws_size, kAlign);

    size_t ptr_array_size = align_up(size_t(num_experts) * sizeof(void*), kPtrAlign);
    size_t sf_array_size = align_up(size_t(num_experts) * sizeof(void*), kPtrAlign);
    size_t stride_array_size = align_up(size_t(num_experts) * sizeof(StrideT), kPtrAlign);
    (void)stride_c_elem_size;  // No longer needed; C stride aliases D stride.

    layout.ptr_weight_gate_offset = layout.gemm_workspace_size;
    layout.sf_gate_offset = layout.ptr_weight_gate_offset + ptr_array_size;
    layout.ptr_linear_offset = layout.sf_gate_offset + sf_array_size;
    layout.ptr_linear_c_offset = layout.ptr_linear_offset + ptr_array_size;
    layout.stride_linear_offset = layout.ptr_linear_c_offset + ptr_array_size;
    layout.ptr_out_offset = layout.stride_linear_offset + stride_array_size;
    layout.stride_out_offset = layout.ptr_out_offset + ptr_array_size;
    layout.total_size = layout.stride_out_offset + stride_array_size;
    return layout;
  }
};

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag,
          typename TileShape, typename ClusterShape, bool IsMXFP4>
void sm120_two_gemm_gated_fc1_kernelLauncher(
    TmaWarpSpecializedGroupedGemmInput tma_inputs,
    void* linear_output,
    void* swiglu_output,
    int64_t const* expert_first_token_offset,
    int64_t inter_size,
    int64_t hidden_size,
    int num_experts,
    int multi_processor_count,
    cudaStream_t stream,
    int* occupancy,
    size_t* workspace_size) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

  static_assert(IsMXFP4, "SM120 two-GEMM gated FC1 only supports MXFP4 (FP8xFP4).");
  TLLM_CHECK_WITH_INFO(!tma_inputs.swap_ab, "SM120 two-GEMM gated FC1 does not support swap_ab mode");
  TLLM_CHECK_WITH_INFO(inter_size % 32 == 0, "SM120 two-GEMM gated FC1 requires inter_size divisible by 32");

  namespace LinearNS = sm120_mxfp4_bf16;
  namespace GateNS = sm120_mxfp4_bf16_swiglu_mul;

  // Hardware info
  cutlass::KernelHardwareInfo hw_info{};
  hw_info.device_id = 0;
  hw_info.sm_count = multi_processor_count;

  // Weight packing note (vLLM CUTLASS MoE):
  // w1 is packed as [gate; up] along the output-N dimension (2*inter_size).
  // SwiGLU expects: out = SiLU(gate) * up.
  //
  // Therefore:
  // - First half (base)  : gate
  // - Second half (+off) : up
  //
  // This two-GEMM bringup must compute UP first (to feed as C), then compute
  // GATE in the accumulator and fuse: D = C * SiLU(acc_gate).
  int64_t const k_blocks = (hidden_size + 31) / 32;
  int64_t const gate_weight_bytes = (inter_size * hidden_size) / 2;
  int64_t const gate_sf_elems = inter_size * k_blocks;

  // Workspace query: use the larger of the two GEMMs (they share the same tensormap workspace region)
  LinearNS::Gemm linear_gemm;
  GateNS::Gemm gate_gemm;

  typename LinearNS::Gemm::Arguments ws_args_linear = {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      tma_inputs.shape_info,
      {nullptr, typename LinearNS::StrideA{}, nullptr, typename LinearNS::StrideB{},
       nullptr, typename LinearNS::LayoutSFA{}, nullptr, typename LinearNS::LayoutSFB{}},
      {{}, nullptr, nullptr, nullptr, typename LinearNS::StrideD{}},
      hw_info};
  ws_args_linear.epilogue.thread.alpha = 1.0f;
  ws_args_linear.epilogue.thread.beta = 0.0f;

  typename GateNS::Gemm::Arguments ws_args_gate = {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      tma_inputs.shape_info,
      {nullptr, typename GateNS::StrideA{}, nullptr, typename GateNS::StrideB{},
       nullptr, typename GateNS::LayoutSFA{}, nullptr, typename GateNS::LayoutSFB{}},
      {{}, nullptr, nullptr, nullptr, typename GateNS::StrideD{}},
      hw_info};
  ws_args_gate.epilogue.thread.alpha = 1.0f;

  size_t base_ws = std::max(linear_gemm.get_workspace_size(ws_args_linear),
                            gate_gemm.get_workspace_size(ws_args_gate));

  using StrideDVal = std::remove_pointer_t<typename LinearNS::StrideD>;
  static_assert(sizeof(StrideDVal) == sizeof(std::remove_pointer_t<typename GateNS::StrideD>),
                "StrideD value types must match between linear and gate GEMMs");
  using StrideCVal = std::remove_pointer_t<typename GateNS::StrideC>;
  static_assert(sizeof(StrideCVal) == sizeof(StrideDVal),
                "Two-GEMM bringup assumes StrideCVal matches StrideDVal");

  auto ws_layout =
      TwoGemmFC1WorkspaceLayout::compute<StrideDVal>(base_ws, num_experts, sizeof(StrideCVal));

  if (workspace_size != nullptr) {
    *workspace_size = ws_layout.total_size;
    return;
  }

  TLLM_CHECK_WITH_INFO(tma_inputs.gemm_workspace_size >= ws_layout.total_size,
                       "SM120 two-GEMM gated FC1: gemm_workspace too small (have=%zu need=%zu)",
                       tma_inputs.gemm_workspace_size, ws_layout.total_size);

  auto* ws_base = reinterpret_cast<uint8_t*>(tma_inputs.gemm_workspace);
  void* gemm_workspace = ws_base + ws_layout.gemm_workspace_offset;

  auto** ptr_weight_gate = reinterpret_cast<void const**>(ws_base + ws_layout.ptr_weight_gate_offset);
  auto** sf_gate = reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF const**>(
      ws_base + ws_layout.sf_gate_offset);
  auto** ptr_linear = reinterpret_cast<cutlass::bfloat16_t**>(ws_base + ws_layout.ptr_linear_offset);
  auto** ptr_linear_c =
      reinterpret_cast<cutlass::bfloat16_t const**>(ws_base + ws_layout.ptr_linear_c_offset);
  auto* stride_linear = reinterpret_cast<StrideDVal*>(ws_base + ws_layout.stride_linear_offset);
  auto** ptr_out = reinterpret_cast<cutlass::bfloat16_t**>(ws_base + ws_layout.ptr_out_offset);
  auto* stride_out = reinterpret_cast<StrideDVal*>(ws_base + ws_layout.stride_out_offset);

  // Update grouped problem shapes N = inter_size (in-place) for the duration of this launcher.
  {
    int threads = 128;
    int blocks = (num_experts + threads - 1) / threads;
    setProblemShapesN<<<blocks, threads, 0, stream>>>(
        tma_inputs.shape_info.problem_shapes, num_experts, inter_size);
  }

  // Compute pointer arrays and "up" (second-half) pointers.
  {
    int threads = 128;
    int blocks = (num_experts + threads - 1) / threads;
    computeOutputPointersAndStrides<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<cutlass::bfloat16_t*>(linear_output),
        ptr_linear, stride_linear, expert_first_token_offset, inter_size, num_experts);
    computeOutputPointersAndStridesConst<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<cutlass::bfloat16_t*>(linear_output),
        ptr_linear_c, stride_linear, expert_first_token_offset, inter_size, num_experts);
    computeOutputPointersAndStrides<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<cutlass::bfloat16_t*>(swiglu_output),
        ptr_out, stride_out, expert_first_token_offset, inter_size, num_experts);
    computeGateWeightAndSfPointers<<<blocks, threads, 0, stream>>>(
        tma_inputs.ptr_weight,
        ptr_weight_gate,  // NOTE: this is the UP half (second half)
        tma_inputs.fpX_block_scaling_factors_weight,
        sf_gate,          // NOTE: this is the UP half (second half)
        gate_weight_bytes, gate_sf_elems, num_experts);
  }

  // The bringup path uses device-side pointer/stride kernels and then calls into
  // CUTLASS initialize() on host. Per the Layer 1A plan (and prior observed
  // segfault/garbage), we must ensure these kernels have completed before any
  // host-side reads of device pointer arrays.
  //
  // Keep the stronger sync optional for perf experiments, but default to ON
  // in bringup because correctness > performance here.
  auto env_flag = [](char const* name) -> bool {
    // Treat unset, empty, or "0" as false. Any other value enables the flag.
    char const* v = std::getenv(name);
    return (v != nullptr) && (v[0] != '\0') && (v[0] != '0');
  };
  const bool dbg_no_sync = env_flag("FLASHINFER_GATED_FC1_NO_SYNC");
  auto is_stream_capturing = [&]() -> bool {
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    cudaError_t cap_err = cudaStreamIsCapturing(stream, &cap_status);
    // If query fails, be conservative and assume we're not capturing.
    if (cap_err != cudaSuccess) {
      return false;
    }
    return cap_status != cudaStreamCaptureStatusNone;
  }();
  if (!dbg_no_sync && !is_stream_capturing) {
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
      TLLM_THROW("SM120 two-GEMM gated FC1: sync after pointer kernels failed: %s",
                 cudaGetErrorString(sync_err));
    }
  }

  // ---------------------------------------------------------------------------
  // Instrumentation: validate bringup assumptions against Layer 1A plan.
  //
  // This bringup path relies on a set of invariants that, if violated, can
  // produce *finite but garbage* output:
  //   - Gate pointers must equal linear pointers + (inter_size * hidden_size)/2 bytes
  //   - Gate SF pointers must equal linear SF pointers + inter_size * ceil(hidden_size/32) elems
  //   - Per-expert output pointers must equal base + expert_first_token_offset[e] * inter_size
  //   - CUTLASS initialize() must not observe partially-written pointer arrays
  //
  // The original implementation work tracked a real bug where host-side
  // initialize() observed zero/garbage because device pointer kernels had not
  // completed. Keep this optional and off by default.
  // ---------------------------------------------------------------------------
  const bool dbg_sync = env_flag("FLASHINFER_GATED_FC1_DEBUG_SYNC");
  const bool dbg_validate = env_flag("FLASHINFER_GATED_FC1_DEBUG_VALIDATE_PTRS");
  const bool dbg_validate_effective = (dbg_validate && !is_stream_capturing);
  const bool dbg_sync_effective = (dbg_sync && !is_stream_capturing);
  if ((dbg_sync_effective || dbg_validate_effective)) {
    // Make sure the pointer/shape update kernels have completed before any
    // host-side CUTLASS initialize() that may read device pointer arrays.
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
      TLLM_LOG_WARNING(
          "[SM120 two-GEMM gated FC1] cudaStreamSynchronize failed: %s",
          cudaGetErrorString(sync_err));
    }
  }
  if (dbg_validate && is_stream_capturing) {
    // NOTE: cudaMemcpy(...DeviceToHost) and host synchronizes are not permitted
    // during CUDA graph capture. Skip validation to avoid invalidating capture.
    TLLM_LOG_WARNING(
        "[SM120 two-GEMM gated FC1][DBG] skipping pointer validation during CUDA graph capture");
  }
  if (dbg_validate_effective) {
    // Copy a few device pointer values back to host and validate arithmetic.
    // NOTE: We only validate pointer values / offsets. We never dereference
    // device pointers on host.
    constexpr int kCheck = 4;
    void const* h_gate_w[kCheck] = {nullptr};
    void const* h_gate_sf[kCheck] = {nullptr};
    void const* h_w_lin[kCheck] = {nullptr};
    void const* h_sf_lin[kCheck] = {nullptr};
    void const* h_lin_out[kCheck] = {nullptr};
    void const* h_lin_c[kCheck] = {nullptr};
    void const* h_out[kCheck] = {nullptr};
    int64_t h_offsets[kCheck + 1] = {0};

    cudaMemcpy(h_gate_w, ptr_weight_gate, sizeof(void*) * kCheck, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gate_sf, sf_gate, sizeof(void*) * kCheck, cudaMemcpyDeviceToHost);
    // Base weight pointers (linear half) and scaling-factor pointers.
    // These are the reference for validating ptr_weight_gate / sf_gate arithmetic.
    cudaMemcpy(h_w_lin, tma_inputs.ptr_weight, sizeof(void*) * kCheck, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sf_lin, tma_inputs.fpX_block_scaling_factors_weight, sizeof(void*) * kCheck,
               cudaMemcpyDeviceToHost);
    // Linear output pointers (D pointers) and the alias used as C for the gate GEMM.
    cudaMemcpy(h_lin_out, ptr_linear, sizeof(void*) * kCheck, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lin_c, ptr_linear_c, sizeof(void*) * kCheck, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, ptr_out, sizeof(void*) * kCheck, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offsets, expert_first_token_offset, sizeof(int64_t) * (kCheck + 1),
               cudaMemcpyDeviceToHost);

    TLLM_LOG_WARNING(
        "[SM120 two-GEMM gated FC1][DBG] inter=%ld hidden=%ld k_blocks=%ld gate_weight_bytes=%ld gate_sf_elems=%ld",
        (long)inter_size, (long)hidden_size, (long)k_blocks,
        (long)gate_weight_bytes, (long)gate_sf_elems);
    for (int i = 0; i < kCheck; ++i) {
      auto w_lin = reinterpret_cast<uintptr_t>(h_w_lin[i]);
      auto gate = reinterpret_cast<uintptr_t>(h_gate_w[i]);
      auto gate_sf = reinterpret_cast<uintptr_t>(h_gate_sf[i]);
      auto lin_out = reinterpret_cast<uintptr_t>(h_lin_out[i]);
      auto lin_c = reinterpret_cast<uintptr_t>(h_lin_c[i]);
      auto out = reinterpret_cast<uintptr_t>(h_out[i]);
      int64_t tok0 = h_offsets[i];
      int64_t tok1 = h_offsets[i + 1];

      if (h_w_lin[i] == nullptr || h_out[i] == nullptr) {
        TLLM_LOG_WARNING(
            "[SM120 two-GEMM gated FC1][DBG] expert[%d] NULL ptrs: w_lin=%p out=%p gate=%p gate_sf=%p sf_lin=%p",
            i, h_w_lin[i], h_out[i], h_gate_w[i], h_gate_sf[i], h_sf_lin[i]);
      }
      if (gate != w_lin + (uintptr_t)gate_weight_bytes) {
        TLLM_LOG_WARNING(
            "[SM120 two-GEMM gated FC1][DBG] expert[%d] up ptr mismatch: up=%p gate(base)=%p (delta=%ld expected=%ld)",
            i, (void*)gate, (void*)w_lin,
            (long)(gate - w_lin), (long)gate_weight_bytes);
      }
      // Gate SF pointer arithmetic is in elements. Validate delta in elements matches expected.
      if (h_gate_sf[i] == nullptr) {
        TLLM_LOG_WARNING("[SM120 two-GEMM gated FC1][DBG] expert[%d] gate_sf is NULL", i);
      }
      if (h_sf_lin[i] == nullptr) {
        TLLM_LOG_WARNING("[SM120 two-GEMM gated FC1][DBG] expert[%d] sf_lin is NULL", i);
      }
      if (h_gate_sf[i] != nullptr && h_sf_lin[i] != nullptr) {
        auto sf_lin_addr = reinterpret_cast<uintptr_t>(h_sf_lin[i]);
        auto sf_gate_addr = reinterpret_cast<uintptr_t>(h_gate_sf[i]);
        int64_t sf_delta_elems =
            (int64_t)((sf_gate_addr - sf_lin_addr) / sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF));
        if (sf_delta_elems != gate_sf_elems) {
          TLLM_LOG_WARNING(
              "[SM120 two-GEMM gated FC1][DBG] expert[%d] up_sf delta mismatch: up_sf=%p gate_sf(base)=%p (delta_elems=%ld expected=%ld)",
              i, h_gate_sf[i], h_sf_lin[i], (long)sf_delta_elems, (long)gate_sf_elems);
        }
      }
      // Validate that linear C pointer array matches linear output pointer array.
      // Validate that linear C pointer array matches linear output pointer array.
      if (lin_c != lin_out) {
        TLLM_LOG_WARNING(
            "[SM120 two-GEMM gated FC1][DBG] expert[%d] ptr_linear_c != ptr_linear: c=%p lin=%p",
            i, (void*)lin_c, (void*)lin_out);
      }
      // Validate linear output pointer offset against expert_first_token_offset (same as swiglu out).
      auto expected_lin_out = reinterpret_cast<uintptr_t>(linear_output) +
                              (uintptr_t)tok0 * (uintptr_t)inter_size * sizeof(cutlass::bfloat16_t);
      if (lin_out != expected_lin_out) {
        TLLM_LOG_WARNING(
            "[SM120 two-GEMM gated FC1][DBG] expert[%d] linear out ptr mismatch: lin_out=%p expected=%p (tok0=%ld tok1=%ld)",
            i, (void*)lin_out, (void*)expected_lin_out, (long)tok0, (long)tok1);
      }
      // Validate output pointer offset against expert_first_token_offset.
      auto expected_out = reinterpret_cast<uintptr_t>(swiglu_output) +
                          (uintptr_t)tok0 * (uintptr_t)inter_size * sizeof(cutlass::bfloat16_t);
      if (out != expected_out) {
        TLLM_LOG_WARNING(
            "[SM120 two-GEMM gated FC1][DBG] expert[%d] out ptr mismatch: out=%p expected=%p (tok0=%ld tok1=%ld)",
            i, (void*)out, (void*)expected_out, (long)tok0, (long)tok1);
      }
      // Quick monotonicity check for offsets.
      if (tok1 < tok0) {
        TLLM_LOG_WARNING(
            "[SM120 two-GEMM gated FC1][DBG] expert[%d] offsets not monotonic: %ld -> %ld",
            i, (long)tok0, (long)tok1);
      }
    }
  }

  // 1) UP GEMM (second half): A @ W_up -> linear_output (used as epilogue source C)
  {
    auto tma_linear = tma_inputs;
    // Use UP (second half) weights + SFs.
    tma_linear.ptr_weight = ptr_weight_gate;
    tma_linear.fpX_block_scaling_factors_weight = sf_gate;
    tma_linear.ptr_d = reinterpret_cast<void**>(ptr_linear);
    tma_linear.stride_d = reinterpret_cast<void*>(stride_linear);
    // No C source
    tma_linear.ptr_c = nullptr;
    tma_linear.stride_c = nullptr;

    sm120_mixed_input_moe_gemm_kernelLauncher<T, WeightType, OutputType,
        tensorrt_llm::cutlass_extensions::EpilogueOpDefault,
        TileShape, ClusterShape, IsMXFP4>(
        tma_linear, num_experts, multi_processor_count, stream, nullptr, nullptr);
  }

  // 2) GATE GEMM (first half, base) with fused SwiGLU epilogue:
  //    D = C(up) * SiLU(acc_gate)
  {
    auto tma_gate = tma_inputs;
    // NOTE: For gate we use the BASE (first half) weights + SFs from tma_inputs.
    // Provide C (linear) and D (SwiGLU BF16).
    // IMPORTANT: avoid const-qualification casts by using a dedicated const** pointer array.
    // Also alias C stride to D stride to avoid layout/tuple-order mismatches.
    tma_gate.ptr_c = reinterpret_cast<void const**>(ptr_linear_c);
    tma_gate.stride_c = reinterpret_cast<void*>(stride_linear);
    tma_gate.ptr_d = reinterpret_cast<void**>(ptr_out);
    tma_gate.stride_d = reinterpret_cast<void*>(stride_out);

    // Build and run gate GEMM directly using the SwigluMul epilogue types.
    //
    // IMPORTANT: Match the standard SM120 launcher’s handling of stride/layout
    // objects. Depending on the CUTLASS kernel specialization, these may be:
    // - pointer types (device pointers passed through), or
    // - value types stored in host-accessible memory (dereferenced here).
    // Getting this wrong produces finite-but-garbage outputs.
    auto strideA_arg = [&]() -> typename GateNS::StrideA {
      if constexpr (std::is_pointer_v<typename GateNS::StrideA>) {
        return reinterpret_cast<typename GateNS::StrideA>(tma_gate.stride_act);
      } else {
        return *reinterpret_cast<typename GateNS::StrideA const*>(tma_gate.stride_act);
      }
    }();
    auto strideB_arg = [&]() -> typename GateNS::StrideB {
      if constexpr (std::is_pointer_v<typename GateNS::StrideB>) {
        return reinterpret_cast<typename GateNS::StrideB>(tma_gate.stride_weight);
      } else {
        return *reinterpret_cast<typename GateNS::StrideB const*>(tma_gate.stride_weight);
      }
    }();
    auto strideC_arg = [&]() -> typename GateNS::StrideC {
      if constexpr (std::is_pointer_v<typename GateNS::StrideC>) {
        return reinterpret_cast<typename GateNS::StrideC>(tma_gate.stride_c);
      } else {
        return *reinterpret_cast<typename GateNS::StrideC const*>(tma_gate.stride_c);
      }
    }();
    auto layoutSFA_arg = [&]() -> typename GateNS::LayoutSFA {
      if constexpr (std::is_pointer_v<typename GateNS::LayoutSFA>) {
        return reinterpret_cast<typename GateNS::LayoutSFA>(tma_gate.fpX_block_scaling_factors_stride_act);
      } else {
        return *reinterpret_cast<typename GateNS::LayoutSFA const*>(tma_gate.fpX_block_scaling_factors_stride_act);
      }
    }();
    auto layoutSFB_arg = [&]() -> typename GateNS::LayoutSFB {
      if constexpr (std::is_pointer_v<typename GateNS::LayoutSFB>) {
        return reinterpret_cast<typename GateNS::LayoutSFB>(tma_gate.fpX_block_scaling_factors_stride_weight);
      } else {
        return *reinterpret_cast<typename GateNS::LayoutSFB const*>(tma_gate.fpX_block_scaling_factors_stride_weight);
      }
    }();

    GateNS::Gemm gemm;
    typename GateNS::Gemm::Arguments arguments = {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        tma_gate.shape_info,
        {
          reinterpret_cast<typename GateNS::CollectiveMainloop::ElementA const**>(tma_gate.ptr_act),
          strideA_arg,
          reinterpret_cast<typename GateNS::CollectiveMainloop::ElementB const**>(tma_gate.ptr_weight),
          strideB_arg,
          reinterpret_cast<typename GateNS::CollectiveMainloop::ElementSF const**>(tma_gate.fpX_block_scaling_factors_act),
          layoutSFA_arg,
          reinterpret_cast<typename GateNS::CollectiveMainloop::ElementSF const**>(tma_gate.fpX_block_scaling_factors_weight),
          layoutSFB_arg,
        },
        {
          {}, // fusion args (alpha is set below)
          reinterpret_cast<typename GateNS::ElementC const**>(tma_gate.ptr_c),
          strideC_arg,
          reinterpret_cast<typename GateNS::ElementD**>(tma_gate.ptr_d),
          reinterpret_cast<typename GateNS::StrideD>(tma_gate.stride_d),
        },
        hw_info
    };
    arguments.epilogue.thread.alpha = 1.0f;

    auto init_status = gemm.initialize(arguments, gemm_workspace, stream);
    TLLM_CHECK_WITH_INFO(init_status == cutlass::Status::kSuccess,
                         "SM120 two-GEMM gated FC1: gate GEMM initialize failed: %s",
                         cutlassGetStatusString(init_status));
    auto run_status = gemm.run(stream);
    cudaError_t cuda_err = cudaPeekAtLastError();
    if (run_status != cutlass::Status::kSuccess || cuda_err != cudaSuccess) {
      cudaDeviceSynchronize();
      cudaError_t sync_err = cudaGetLastError();
      TLLM_THROW("SM120 two-GEMM gated FC1: gate GEMM run failed: %s (CUDA: %s / %s)",
                 cutlassGetStatusString(run_status),
                 cudaGetErrorString(cuda_err),
                 cudaGetErrorString(sync_err));
    }
  }

  // Restore the original grouped problem shape N (FC1 output dim) after bringup.
  //
  // Layer 1A bringup temporarily sets N=inter_size so each GEMM computes [M, inter_size].
  // The owning fused-MoE call graph may reuse the same device-side problem_shapes buffer
  // for subsequent GEMMs. Leaving it mutated can poison later GEMMs and produce
  // "finite garbage" outputs.
  {
    int threads = 128;
    int blocks = (num_experts + threads - 1) / threads;
    setProblemShapesN<<<blocks, threads, 0, stream>>>(
        tma_inputs.shape_info.problem_shapes, num_experts, inter_size * 2);
  }

  if (occupancy != nullptr) {
    *occupancy = 1;
  }
}

#endif  // CUTLASS_ARCH_MMA_SM12x_SUPPORTED && ENABLE_FP4 && FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP

// setProblemShapesN for the fused gated FC1 path.
// (The two-GEMM bringup path defines its own copy above under a separate guard.)
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4) && defined(FLASHINFER_GATED_FC1) && !defined(FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP)
__global__ void setProblemShapesN(
    cutlass::gemm::GroupProblemShape<cute::Shape<int64_t, int64_t, int64_t>>::UnderlyingProblemShape* problem_shapes,
    int num_experts,
    int64_t new_n) {
  int e = threadIdx.x + blockIdx.x * blockDim.x;
  if (e >= num_experts) return;
  auto ps = problem_shapes[e];
  cute::get<1>(ps) = new_n;
  problem_shapes[e] = ps;
}
#endif  // FLASHINFER_GATED_FC1 && !FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag,
          typename TileShape, typename ClusterShape, bool IsMXFP4>
void sm120_gated_fc1_moe_gemm_kernelLauncher(
    TmaWarpSpecializedGroupedGemmInput tma_inputs,
    int64_t inter_size,
    int64_t hidden_size,
    int num_experts,
    int multi_processor_count,
    cudaStream_t stream,
    int* occupancy,
    size_t* workspace_size,
    GatedFC1SwigluParams swiglu_params) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4) && defined(FLASHINFER_GATED_FC1)

  // Only MXFP4 supported
  static_assert(IsMXFP4,
      "SM120 Gated FC1 only supports MXFP4 (FP8xFP4).");

  // Guardrails
  TLLM_CHECK_WITH_INFO(inter_size % 32 == 0,
      "SM120 Gated FC1 requires inter_size divisible by 32 (SF_VEC_SIZE)");
  TLLM_CHECK_WITH_INFO(!tma_inputs.swap_ab,
      "SM120 Gated FC1 does not support swap_ab mode");

  // Gated FC1 extension must be pre-populated by the upstream stride kernel
  TLLM_CHECK_WITH_INFO(tma_inputs.gated_fc1.enabled,
      "SM120 Gated FC1: gated_fc1 extension not enabled (was setupTmaWarpSpecializedInputs called?)");
  TLLM_CHECK_WITH_INFO(tma_inputs.gated_fc1.ptr_weight_gate != nullptr,
      "SM120 Gated FC1: ptr_weight_gate is null");
  TLLM_CHECK_WITH_INFO(tma_inputs.gated_fc1.sf_gate != nullptr,
      "SM120 Gated FC1: sf_gate is null");
  TLLM_CHECK_WITH_INFO(tma_inputs.gated_fc1.ptr_output != nullptr,
      "SM120 Gated FC1: ptr_output is null");
  TLLM_CHECK_WITH_INFO(tma_inputs.gated_fc1.stride_output != nullptr,
      "SM120 Gated FC1: stride_output is null");

  // Use gated namespace
  using namespace sm120_mxfp4_bf16_gated;
  TLLM_LOG_DEBUG("[SM120 Gated FC1] tile_mn=(%d,%d) inter_size=%ld hidden_size=%ld num_experts=%d",
      LOGICAL_TILE_M, LOGICAL_TILE_N, (long)inter_size, (long)hidden_size, num_experts);

  Gemm gemm;

  // Hardware info
  cutlass::KernelHardwareInfo hw_info{};
  hw_info.device_id = 0;
  hw_info.sm_count = multi_processor_count;

  // Alignment check (debug)
  int64_t const gate_weight_bytes = (inter_size * hidden_size) / 2;  // FP4: 2 elements/byte
  TLLM_CHECK_WITH_INFO(gate_weight_bytes % 16 == 0,
      "SM120 Gated FC1: gate weight offset must be 16-byte aligned for TMA");

  // Workspace size query
  if (workspace_size != nullptr) {
    typename Gemm::Arguments ws_arguments{};
    ws_arguments.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    ws_arguments.problem_shape = tma_inputs.shape_info;
    // mainloop/epilogue fields left default-initialized (placeholders)
    ws_arguments.epilogue.thread.alpha = 1.0f;
    ws_arguments.epilogue.thread.beta = 0.0f;
    ws_arguments.hw_info = hw_info;
    *workspace_size = gemm.get_workspace_size(ws_arguments);
    TLLM_LOG_DEBUG("[SM120 Gated FC1] workspace_size=%zu", *workspace_size);
    return;
  }

  // Validate inputs
  TLLM_CHECK_WITH_INFO(tma_inputs.stride_act != nullptr, "SM120 Gated FC1: stride_act is null");
  TLLM_CHECK_WITH_INFO(tma_inputs.stride_weight != nullptr, "SM120 Gated FC1: stride_weight is null");
  TLLM_CHECK_WITH_INFO(tma_inputs.gemm_workspace != nullptr, "SM120 Gated FC1: workspace is null");

  using StrideDVal = std::remove_pointer_t<StrideD>;

  // Gate weight/SF pointers and gated output pointers/strides are pre-computed
  // by the upstream computeStridesTmaWarpSpecializedKernel - no separate device
  // kernel or cudaStreamSynchronize needed here. This preserves CUDA graph compatibility.
  void const** ptr_weight_gate = tma_inputs.gated_fc1.ptr_weight_gate;
  using TmaElementSF = TmaWarpSpecializedGroupedGemmInput::ElementSF;
  using MainloopElementSF = typename CollectiveMainloop::ElementSF;
  static_assert(sizeof(TmaElementSF) == sizeof(MainloopElementSF), "ElementSF size mismatch");
  TmaElementSF const** sf_gate = tma_inputs.gated_fc1.sf_gate;
  auto* ptr_gated_output = reinterpret_cast<ElementOutput**>(tma_inputs.gated_fc1.ptr_output);
  auto* stride_gated_output = reinterpret_cast<StrideDVal*>(tma_inputs.gated_fc1.stride_output);

  // Build GEMM arguments using pre-computed pointer arrays.
  // The gated mainloop expects:
  //   - ptr_A: activations (FP8)
  //   - ptr_B: linear weights (FP4)
  //   - ptr_Aux: gate weights (FP4)
  //   - ptr_SFA: activation scale factors
  //   - ptr_SFB: linear weight scale factors
  //   - ptr_SFAux: gate weight scale factors

  // Input strides: handle both pointer and value types (matches non-gated launcher pattern)
  auto stride_a = [&]() -> StrideA {
    if constexpr (std::is_pointer_v<StrideA>) {
      return reinterpret_cast<StrideA>(tma_inputs.stride_act);
    } else {
      StrideA host_stride{};
      cudaMemcpy(&host_stride, tma_inputs.stride_act, sizeof(StrideA), cudaMemcpyDeviceToHost);
      return host_stride;
    }
  }();
  auto stride_b = [&]() -> StrideB {
    if constexpr (std::is_pointer_v<StrideB>) {
      return reinterpret_cast<StrideB>(tma_inputs.stride_weight);
    } else {
      StrideB host_stride{};
      cudaMemcpy(&host_stride, tma_inputs.stride_weight, sizeof(StrideB), cudaMemcpyDeviceToHost);
      return host_stride;
    }
  }();
  auto stride_aux = [&]() -> StrideAux {
    if constexpr (std::is_pointer_v<StrideAux>) {
      return reinterpret_cast<StrideAux>(tma_inputs.stride_weight);  // Gate weights have same stride
    } else {
      StrideAux host_stride{};
      cudaMemcpy(&host_stride, tma_inputs.stride_weight, sizeof(StrideAux), cudaMemcpyDeviceToHost);
      return host_stride;
    }
  }();

  auto layout_sfa = [&]() -> LayoutSFA {
    if constexpr (std::is_pointer_v<LayoutSFA>) {
      return reinterpret_cast<LayoutSFA>(tma_inputs.fpX_block_scaling_factors_stride_act);
    } else {
      LayoutSFA host_layout{};
      cudaMemcpy(&host_layout, tma_inputs.fpX_block_scaling_factors_stride_act, sizeof(LayoutSFA), cudaMemcpyDeviceToHost);
      return host_layout;
    }
  }();
  auto layout_sfb = [&]() -> LayoutSFB {
    if constexpr (std::is_pointer_v<LayoutSFB>) {
      return reinterpret_cast<LayoutSFB>(tma_inputs.fpX_block_scaling_factors_stride_weight);
    } else {
      LayoutSFB host_layout{};
      cudaMemcpy(&host_layout, tma_inputs.fpX_block_scaling_factors_stride_weight, sizeof(LayoutSFB), cudaMemcpyDeviceToHost);
      return host_layout;
    }
  }();
  auto layout_sfaux = layout_sfb;  // Gate SFs have same layout as linear weights

  // Type-safety checks
  using EpilogueElementD = typename CollectiveEpilogue::ElementD;
  static_assert(std::is_same_v<ElementOutput, EpilogueElementD>,
                "ElementOutput must match CollectiveEpilogue::ElementD");
  static_assert(std::is_trivially_copyable_v<StrideDVal>,
                "StrideDVal must be trivially copyable");

  // The upstream computeStridesTmaWarpSpecializedKernel writes per-expert shapes
  // with N = gemm1_n = 2 * inter_size (the full FC1 weight width).  But the gated
  // kernel runs two GEMMs (B and Aux), each with N = inter_size.  Fix in-place,
  // matching the pattern used by the two-GEMM fallback launcher.
  {
    int threads = 128;
    int blocks = (num_experts + threads - 1) / threads;
    setProblemShapesN<<<blocks, threads, 0, stream>>>(
        tma_inputs.shape_info.problem_shapes, num_experts, inter_size);
  }

  // Build Gemm::Arguments using field-by-field assignment.
  // Aggregate brace-init of the outer struct triggers brace-elision issues with
  // CUTLASS's deeply nested template types (StrideA, LayoutSFA, SwigluBiasParams).
  // Field-by-field is the pattern used by CUTLASS examples for complex argument types.
  typename Gemm::Arguments arguments{};
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  arguments.problem_shape = tma_inputs.shape_info;
  // Mainloop args
  arguments.mainloop.ptr_A   = reinterpret_cast<typename CollectiveMainloop::ElementA const**>(tma_inputs.ptr_act);
  arguments.mainloop.dA      = stride_a;
  arguments.mainloop.ptr_B   = reinterpret_cast<typename CollectiveMainloop::ElementB const**>(tma_inputs.ptr_weight);
  arguments.mainloop.dB      = stride_b;
  arguments.mainloop.ptr_SFA = reinterpret_cast<MainloopElementSF const**>(tma_inputs.fpX_block_scaling_factors_act);
  arguments.mainloop.layout_SFA = layout_sfa;
  arguments.mainloop.ptr_SFB = reinterpret_cast<MainloopElementSF const**>(tma_inputs.fpX_block_scaling_factors_weight);
  arguments.mainloop.layout_SFB = layout_sfb;
  // Gated extension: Aux pointers (gate weights) - pre-computed upstream
  arguments.mainloop.ptr_Aux  = reinterpret_cast<typename CollectiveMainloop::ElementAux const**>(ptr_weight_gate);
  arguments.mainloop.dAux     = stride_aux;
  arguments.mainloop.ptr_SFAux = reinterpret_cast<MainloopElementSF const**>(sf_gate);
  arguments.mainloop.layout_SFAux = layout_sfaux;
  // SwigluBias activation parameters (assign fields directly to work around
  // nvcc limitation with copy-assignment of nested structs in deep templates)
  arguments.mainloop.swiglu.d_alpha = swiglu_params.d_alpha;
  arguments.mainloop.swiglu.d_beta  = swiglu_params.d_beta;
  arguments.mainloop.swiglu.d_limit = swiglu_params.d_limit;
  // Epilogue args
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta  = 0.0f;
  arguments.epilogue.ptr_D = ptr_gated_output;
  arguments.epilogue.dD    = stride_gated_output;
  // Hardware info
  arguments.hw_info = hw_info;

  TLLM_LOG_DEBUG("[SM120 Gated FC1] SwigluBias device ptrs: alpha=%p, beta=%p, limit=%p",
      swiglu_params.d_alpha, swiglu_params.d_beta, swiglu_params.d_limit);

  // Debug: log key configuration
  TLLM_LOG_DEBUG("[SM120 Gated FC1] Arguments: num_groups=%d, inter_size=%ld, hidden_size=%ld",
      tma_inputs.shape_info.groups(), inter_size, hidden_size);
  TLLM_LOG_DEBUG("[SM120 Gated FC1] Workspace: gemm=%p, ptr_gate=%p, sf_gate=%p",
      tma_inputs.gemm_workspace, ptr_weight_gate, sf_gate);
  
  // Verify workspace size is sufficient
  size_t required_ws = gemm.get_workspace_size(arguments);
  TLLM_LOG_DEBUG("[SM120 Gated FC1] Required workspace=%zu, have=%zu", required_ws, tma_inputs.gemm_workspace_size);
  TLLM_CHECK_WITH_INFO(tma_inputs.gemm_workspace_size >= required_ws, 
      "SM120 Gated FC1: workspace too small (have=%zu, need=%zu)", tma_inputs.gemm_workspace_size, required_ws);
  
  // Check can_implement before initialize
  auto can_impl = gemm.can_implement(arguments);
  if (can_impl != cutlass::Status::kSuccess) {
    TLLM_THROW("SM120 Gated FC1: can_implement failed: %s (num_groups=%d)",
               cutlassGetStatusString(can_impl), tma_inputs.shape_info.groups());
  }
  TLLM_LOG_DEBUG("[SM120 Gated FC1] can_implement passed");

  // Debug: Check GEMM kernel shared memory size before initialize
  TLLM_LOG_DEBUG("[SM120 Gated FC1] GemmKernel::SharedStorageSize=%d bytes",
      int(Gemm::GemmKernel::SharedStorageSize));
  
  // Detailed SMEM breakdown
  {
    using Mainloop = typename Gemm::GemmKernel::CollectiveMainloop;
    using Epilogue = typename Gemm::GemmKernel::CollectiveEpilogue;
    using MainloopTensorStorage = typename Mainloop::TensorStorage;
    using EpilogueTensorStorage = typename Epilogue::TensorStorage;
    
    // Get individual array sizes from mainloop
    constexpr size_t smem_A = cute::cosize_v<typename Mainloop::SmemLayoutA> * sizeof(typename Mainloop::TiledMma::ValTypeA);
    constexpr size_t smem_B = cute::cosize_v<typename Mainloop::SmemLayoutB> * sizeof(typename Mainloop::TiledMma::ValTypeB);
    constexpr size_t smem_Aux = cute::cosize_v<typename Mainloop::SmemLayoutAux> * sizeof(typename Mainloop::TiledMma::ValTypeB);
    constexpr size_t smem_SFA = cute::cosize_v<typename Mainloop::SmemLayoutSFA> * sizeof(typename Mainloop::ElementSF);
    constexpr size_t smem_SFB = cute::cosize_v<typename Mainloop::SmemLayoutSFB> * sizeof(typename Mainloop::ElementSF);
    constexpr size_t smem_SFAux = cute::cosize_v<typename Mainloop::SmemLayoutSFAux> * sizeof(typename Mainloop::ElementSF);
    
    constexpr size_t mainloop_tensor_total = sizeof(MainloopTensorStorage);
    constexpr size_t epilogue_tensor_total = sizeof(EpilogueTensorStorage);
    constexpr size_t kernel_total = Gemm::GemmKernel::SharedStorageSize;
    
    printf("\n");
    printf("============================================================\n");
    printf("GATED FC1 SMEM BREAKDOWN (bytes)\n");
    printf("============================================================\n");
    printf("Mainloop TensorStorage:\n");
    printf("  smem_A (FP8 activations):     %8zu B\n", smem_A);
    printf("  smem_B (FP4 linear weights):  %8zu B\n", smem_B);
    printf("  smem_Aux (FP4 gate weights):  %8zu B\n", smem_Aux);
    printf("  smem_SFA (E8M0 scale A):      %8zu B\n", smem_SFA);
    printf("  smem_SFB (E8M0 scale B):      %8zu B\n", smem_SFB);
    printf("  smem_SFAux (E8M0 scale Aux):  %8zu B\n", smem_SFAux);
    printf("  -------------------------------------------\n");
    printf("  Raw array total:              %8zu B\n", smem_A + smem_B + smem_Aux + smem_SFA + smem_SFB + smem_SFAux);
    printf("  sizeof(MainloopTensorStorage):%8zu B (includes alignment)\n", mainloop_tensor_total);
    printf("\n");
    printf("Epilogue TensorStorage:         %8zu B\n", epilogue_tensor_total);
    printf("\n");
    printf("Kernel SharedStorageSize:       %8zu B\n", kernel_total);
    printf("Device max SMEM:                %8d B\n", 101376);
    printf("Gap (need to save):             %8zd B\n", (ssize_t)kernel_total - 101376);
    printf("\n");
    printf("TMA Transaction Bytes (BASE CLASS):\n");
    printf("  TmaTransactionBytesMK:        %8u B (A + SFA per stage)\n", Mainloop::Base::TmaTransactionBytesMK);
    printf("  TmaTransactionBytesNK:        %8u B (B + SFB per stage)\n", Mainloop::Base::TmaTransactionBytesNK);
    printf("  TmaTransactionBytes:          %8u B (total per stage)\n", Mainloop::Base::TmaTransactionBytes);
    printf("  sizeof_bits<ElementA>:        %8d\n", (int)cutlass::sizeof_bits<typename Mainloop::ElementA>::value);
    printf("  sizeof_bits<ElementB>:        %8d\n", (int)cutlass::sizeof_bits<typename Mainloop::ElementB>::value);
    printf("  sizeof(SmemAllocTypeA):       %8zu\n", sizeof(typename Mainloop::SmemAllocTypeA));
    printf("  sizeof(SmemAllocTypeB):       %8zu\n", sizeof(typename Mainloop::SmemAllocTypeB));
    printf("  IsF8F6F4:                     %8d\n", (int)Mainloop::Base::IsF8F6F4);
    
    // Check for FP4 transaction byte mismatch
    constexpr size_t smemLayoutB_size = cute::size(cute::take<0,2>(typename Mainloop::SmemLayoutB{}));
    constexpr size_t smemLayoutB_cosize = cute::cosize(cute::take<0,2>(typename Mainloop::SmemLayoutB{}));
    constexpr size_t actual_B_bytes_per_stage = smemLayoutB_cosize * sizeof(typename Mainloop::SmemAllocTypeB);
    constexpr size_t formula_B_bytes = cutlass::bits_to_bytes(smemLayoutB_size * cutlass::sizeof_bits<typename Mainloop::ElementB>::value);
    printf("  SmemLayoutB per stage:\n");
    printf("    size():                     %8zu\n", smemLayoutB_size);
    printf("    cosize():                   %8zu\n", smemLayoutB_cosize);
    printf("    actual_B_bytes (cosize*sizeof(SmemAllocTypeB)): %8zu\n", actual_B_bytes_per_stage);
    printf("    formula_B_bytes (bits_to_bytes(size*sizeof_bits<ElementB>)): %8zu\n", formula_B_bytes);
    if (actual_B_bytes_per_stage != formula_B_bytes) {
      printf("  *** MISMATCH: TMA transaction bytes for B are WRONG! ***\n");
      printf("  *** TMA loads %zu bytes but barrier expects only %zu bytes ***\n",
             actual_B_bytes_per_stage, formula_B_bytes);
    }
    printf("============================================================\n");
    printf("\n");
  }
  
  // Initialize GEMM
  auto init_status = gemm.initialize(arguments, tma_inputs.gemm_workspace, stream);
  if (init_status != cutlass::Status::kSuccess) {
    cudaError_t cuda_err = cudaGetLastError();
    // Also check device properties for max shared memory
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    TLLM_LOG_WARNING("[SM120 Gated FC1] Device max shared memory per block: %d bytes",
        props.sharedMemPerBlockOptin);

    // Optional debug: dump tensormap workspace bytes even on init failure.
    // Useful when `initialize` succeeds for some tiles but fails/traps for others.
    if (auto const* dump_env = std::getenv("TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_ON_INIT_FAIL");
        dump_env != nullptr && dump_env[0] == '1') {
      constexpr size_t kDescBytes = 128;
      const int sm_count = hw_info.sm_count;
      const size_t expected_bytes = size_t(6) * size_t(sm_count) * kDescBytes;  // A,B,SFA,SFB,Aux,SFAux
      const size_t copy_bytes = std::min(expected_bytes, tma_inputs.gemm_workspace_size);

      std::vector<uint8_t> host(copy_bytes, 0);
      cudaError_t copy_err = cudaMemcpyAsync(host.data(), tma_inputs.gemm_workspace, copy_bytes,
                                            cudaMemcpyDeviceToHost, stream);
      cudaError_t dump_sync_err = cudaStreamSynchronize(stream);

      size_t base_off = 0;
      if (auto const* off_env = std::getenv("TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_OFFSET");
          off_env != nullptr && off_env[0] != '\0') {
        base_off = static_cast<size_t>(std::strtoull(off_env, nullptr, 0));
      }

      auto format_hex16 = [](const uint8_t* p) {
        std::array<char, 16 * 3 + 1> out{};
        size_t off = 0;
        for (int i = 0; i < 16; ++i) {
          off += std::snprintf(out.data() + off, out.size() - off, "%02x%s",
                               unsigned(p[i]), (i == 15) ? "" : " ");
        }
        return out;
      };

      TLLM_LOG_DEBUG(
          "[SM120 Gated FC1][dump] init_failed tensormaps: workspace=%p copy_bytes=%zu expected=%zu base_off=%zu memcpy=%s sync=%s",
          tma_inputs.gemm_workspace, copy_bytes, expected_bytes, base_off,
          cudaGetErrorString(copy_err), cudaGetErrorString(dump_sync_err));

      // Dump first 16 bytes of each plane for sm_idx=0 (enough to detect all-zero descriptors).
      for (int plane = 0; plane < 6; ++plane) {
        size_t idx = size_t(0) + size_t(plane) * size_t(sm_count);
        size_t byte_off = base_off + idx * kDescBytes;
        if (byte_off + 16 <= host.size()) {
          auto hex = format_hex16(host.data() + byte_off);
          TLLM_LOG_DEBUG("[SM120 Gated FC1][dump] init_failed tensormaps[P%d] sm_idx=0 off=%zu bytes[0..15]=%s",
                         plane, byte_off, hex.data());
        }
      }
    }

    TLLM_THROW("SM120 Gated FC1: initialize failed: %s (CUDA: %s). "
               "SharedStorageSize=%d, maxSharedMem=%zu",
               cutlassGetStatusString(init_status), cudaGetErrorString(cuda_err),
               int(Gemm::GemmKernel::SharedStorageSize), props.sharedMemPerBlockOptin);
  }
  TLLM_LOG_DEBUG("[SM120 Gated FC1] GEMM initialized successfully");

  // ── Host-side TMA descriptor dump ──────────────────────────────────────
  // Always dump the first 64 bytes of each host TMA descriptor right after
  // initialize().  These blobs were created by cuTensorMapEncodeTiled() and
  // contain the dimension/stride/swizzle fields that the GPU will use.
  // A mismatch or all-zero descriptor means the descriptor is malformed.
  {
    auto const& p = gemm.params();

    auto dump_tma_desc = [](const char* name, void const* desc) {
      if (!desc) {
        printf("[SM120 Gated FC1 TMA] %s: <null>\n", name);
        return;
      }
      auto const* b = reinterpret_cast<uint8_t const*>(desc);
      // Print first 4 × 16-byte lines (64 bytes): enough to see
      // dataType, dims, globalStrides, boxDim, swizzle fields.
      for (int line = 0; line < 4; ++line) {
        printf("[SM120 Gated FC1 TMA] %s [%3d..%3d]:", name, line * 16, line * 16 + 15);
        for (int i = 0; i < 16; ++i) printf(" %02x", unsigned(b[line * 16 + i]));
        printf("\n");
      }
    };

    dump_tma_desc("A   (base.tma_load_a)",   p.mainloop.base.tma_load_a.get_tma_descriptor());
    dump_tma_desc("B   (base.tma_load_b)",   p.mainloop.base.tma_load_b.get_tma_descriptor());
    dump_tma_desc("SFA (base.tma_load_sfa)", p.mainloop.base.tma_load_sfa.get_tma_descriptor());
    dump_tma_desc("SFB (base.tma_load_sfb)", p.mainloop.base.tma_load_sfb.get_tma_descriptor());
    dump_tma_desc("Aux (aux.tma_load_aux)",  p.mainloop.aux.tma_load_aux.get_tma_descriptor());
    dump_tma_desc("SFAux(aux.tma_load_sfaux)", p.mainloop.aux.tma_load_sfaux.get_tma_descriptor());

    // Quick all-zero check for each descriptor
    auto is_all_zero = [](void const* desc) -> bool {
      if (!desc) return true;
      auto const* b = reinterpret_cast<uint8_t const*>(desc);
      for (int i = 0; i < 128; ++i) if (b[i] != 0) return false;
      return true;
    };
    const char* names[] = {"A", "B", "SFA", "SFB", "Aux", "SFAux"};
    void const* descs[] = {
      p.mainloop.base.tma_load_a.get_tma_descriptor(),
      p.mainloop.base.tma_load_b.get_tma_descriptor(),
      p.mainloop.base.tma_load_sfa.get_tma_descriptor(),
      p.mainloop.base.tma_load_sfb.get_tma_descriptor(),
      p.mainloop.aux.tma_load_aux.get_tma_descriptor(),
      p.mainloop.aux.tma_load_sfaux.get_tma_descriptor(),
    };
    for (int i = 0; i < 6; ++i) {
      if (is_all_zero(descs[i])) {
        printf("[SM120 Gated FC1 TMA] *** WARNING: %s descriptor is ALL ZEROS ***\n", names[i]);
      }
    }
  }

  // Extra debug: dump tensormap workspace bytes (A/B/SFA/SFB/Aux/SFAux planes) for one SM slot.
  //
  // Enable with:
  //   export TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS=1
  // Optional:
  //   export TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_POSTRUN=1
  //   export TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_FULL=1   # copy full workspace (recommended)
  //   export TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_SLOT=300 # dump a specific 128B slot index
  //
  // Notes:
  // - CUTLASS partitions the overall GEMM workspace; the SM120 mainloop's `Params::tensormaps`
  //   pointer may point to an offset inside `gemm_workspace` (observed in cuda-gdb).
  // - Use *_OFFSET to adjust the base offset when correlating against `UR50/UR48/UR46/UR44`.
  auto dump_gated_tensormaps = [&](const char* tag) {
    constexpr size_t kDescBytes = 128;  // sizeof(cute::TmaDescriptor) for these kernels
    const int sm_count = hw_info.sm_count;

    // Gated FC1 mainloop uses 6 planes: A, B, SFA, SFB, Aux, SFAux.
    const size_t expected_bytes = size_t(6) * size_t(sm_count) * kDescBytes;
    bool const dump_full = [&] {
      if (auto const* full_env = std::getenv("TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_FULL");
          full_env != nullptr && full_env[0] == '1') {
        return true;
      }
      return false;
    }();

    const size_t copy_bytes =
        dump_full ? tma_inputs.gemm_workspace_size
                  : std::min(expected_bytes, tma_inputs.gemm_workspace_size);

    std::vector<uint8_t> host(copy_bytes, 0);
    cudaError_t copy_err = cudaMemcpyAsync(host.data(), tma_inputs.gemm_workspace, copy_bytes,
                                          cudaMemcpyDeviceToHost, stream);
    cudaError_t sync_err = cudaStreamSynchronize(stream);

    // Determine tensormap base offset inside gemm_workspace.
    // Prefer env override; otherwise read the CUTLASS kernel params, which contain the exact
    // `params.mainloop.base.tensormaps` pointer used by the mainloop.
    size_t base_off = 0;
    bool base_off_from_env = false;
    if (auto const* off_env = std::getenv("TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_OFFSET");
        off_env != nullptr && off_env[0] != '\0') {
      base_off = static_cast<size_t>(std::strtoull(off_env, nullptr, 0));
      base_off_from_env = true;
    } else {
      // GemmUniversalAdapter exposes the kernel Params via gemm.params().
      // For this gated kernel, the mainloop params are a composed type that stores the
      // tensormap pointer at params_.mainloop.base.tensormaps.
      auto const& p = gemm.params();
      auto const* tm_ptr = p.mainloop.base.tensormaps;
      auto const* ws_ptr = reinterpret_cast<uint8_t const*>(tma_inputs.gemm_workspace);
      auto const* tm_u8 = reinterpret_cast<uint8_t const*>(tm_ptr);
      if (tm_u8 >= ws_ptr) {
        base_off = static_cast<size_t>(tm_u8 - ws_ptr);
      }
    }

    // Also print the resolved tensormaps pointer for easy correlation with cuda-gdb URxx regs.
    {
      auto const& p = gemm.params();
      auto const* tm_ptr = p.mainloop.base.tensormaps;
      TLLM_LOG_DEBUG(
          "[SM120 Gated FC1][dump] tensormaps(%s): workspace=%p tensormaps_ptr=%p copy_bytes=%zu expected=%zu base_off=%zu (from_%s) memcpy=%s sync=%s",
          tag, tma_inputs.gemm_workspace, static_cast<void const*>(tm_ptr),
          copy_bytes, expected_bytes, base_off, base_off_from_env ? "env" : "params",
          cudaGetErrorString(copy_err), cudaGetErrorString(sync_err));
    }

    auto format_hex16 = [](const uint8_t* p) {
      std::array<char, 16 * 3 + 1> out{};
      size_t off = 0;
      for (int i = 0; i < 16; ++i) {
        off += std::snprintf(out.data() + off, out.size() - off, "%02x%s",
                             unsigned(p[i]), (i == 15) ? "" : " ");
      }
      return out;
    };

    int sm_idx = 0;
    if (auto const* idx_env = std::getenv("TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_SMIDX");
        idx_env != nullptr && idx_env[0] != '\0') {
      sm_idx = std::atoi(idx_env);
    }
    sm_idx = std::max(0, std::min(sm_idx, sm_count - 1));

    // Optionally dump an explicit 128B slot (useful when you know URxx-derived slot index).
    if (auto const* slot_env = std::getenv("TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_SLOT");
        slot_env != nullptr && slot_env[0] != '\0') {
      long long slot_ll = std::atoll(slot_env);
      if (slot_ll >= 0) {
        size_t const slot = static_cast<size_t>(slot_ll);
        size_t const byte_off = base_off + slot * kDescBytes;
        if (byte_off + kDescBytes <= host.size()) {
          const uint8_t* p = host.data() + byte_off;
          for (size_t line = 0; line < kDescBytes; line += 16) {
            auto hex = format_hex16(p + line);
            TLLM_LOG_DEBUG(
                "[SM120 Gated FC1][dump] tensormaps(%s)[SLOT] slot=%zu byte_off=%zu bytes[%zu..%zu]=%s",
                tag, slot, byte_off, line, line + 15, hex.data());
          }
        } else {
          TLLM_LOG_DEBUG(
              "[SM120 Gated FC1][dump] tensormaps(%s)[SLOT] slot=%zu byte_off=%zu: OOB (host=%zu)",
              tag, slot, byte_off, host.size());
        }
      }
    }

    auto dump_plane = [&](int plane, const char* name) {
      const size_t idx = size_t(sm_idx) + size_t(plane) * size_t(sm_count);
      const size_t byte_off = base_off + idx * kDescBytes;
      if (byte_off + kDescBytes > host.size()) {
        TLLM_LOG_DEBUG(
            "[SM120 Gated FC1][dump] tensormaps(%s)[%s] plane=%d sm_idx=%d: OOB (byte_off=%zu host=%zu)",
            tag, name, plane, sm_idx, byte_off, host.size());
        return;
      }
      const uint8_t* p = host.data() + byte_off;
      for (size_t line = 0; line < kDescBytes; line += 16) {
        auto hex = format_hex16(p + line);
        TLLM_LOG_DEBUG(
            "[SM120 Gated FC1][dump] tensormaps(%s)[%s] plane=%d sm_idx=%d off=%zu bytes[%zu..%zu]=%s",
            tag, name, plane, sm_idx, byte_off, line, line + 15, hex.data());
      }
    };

    dump_plane(0, "A");
    dump_plane(1, "B");
    dump_plane(2, "SFA");
    dump_plane(3, "SFB");
    dump_plane(4, "Aux");
    dump_plane(5, "SFAux");
  };

  bool const dump_pre = [&] {
    if (auto const* dump_env = std::getenv("TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS");
        dump_env != nullptr && dump_env[0] == '1') {
      return true;
    }
    return false;
  }();
  bool const dump_post = [&] {
    if (auto const* dump_env = std::getenv("TLLM_SM120_GATED_FC1_DUMP_TENSORMAPS_POSTRUN");
        dump_env != nullptr && dump_env[0] == '1') {
      return true;
    }
    return false;
  }();
  if (dump_pre) {
    dump_gated_tensormaps("pre_run");
  }

  // Debug: print grid dimensions before run
  {
    printf("[SM120 Gated FC1 DEBUG] num_groups=%d, inter_size=%ld, hidden_size=%ld, SharedStorageSize=%zu\n",
           tma_inputs.shape_info.groups(), inter_size, hidden_size, 
           size_t(Gemm::GemmKernel::SharedStorageSize));
    
    // Check if problem is valid before run
    auto grid = Gemm::get_grid_shape(arguments);
    auto block = Gemm::GemmKernel::get_block_shape();
    printf("[SM120 Gated FC1 DEBUG] Grid: (%d, %d, %d), Block: (%d, %d, %d)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);
    
    // Check for any grid dimension being 0
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
      printf("[SM120 Gated FC1 DEBUG] ERROR: Grid has zero dimension!\n");
    }
  }
  
  // Check for any pre-existing CUDA errors before run
  {
    cudaError_t pre_err = cudaPeekAtLastError();
    if (pre_err != cudaSuccess) {
      printf("[SM120 Gated FC1 DEBUG] Pre-run CUDA error: %d (%s)\n",
             int(pre_err), cudaGetErrorString(pre_err));
    }
    cudaDeviceSynchronize();  // Ensure any async errors are caught
    pre_err = cudaPeekAtLastError();
    if (pre_err != cudaSuccess) {
      printf("[SM120 Gated FC1 DEBUG] Post-sync CUDA error: %d (%s)\n",
             int(pre_err), cudaGetErrorString(pre_err));
      cudaGetLastError();  // Clear the error
    }
    
    // Set the smem size explicitly before run
    int smem_size = Gemm::GemmKernel::SharedStorageSize;
    printf("[SM120 Gated FC1 DEBUG] Requesting smem_size=%d bytes\n", smem_size);
    if (smem_size >= (48 << 10)) {
      cudaError_t smem_result = cudaFuncSetAttribute(
          cutlass::device_kernel<typename Gemm::GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (smem_result != cudaSuccess) {
        printf("[SM120 Gated FC1 DEBUG] cudaFuncSetAttribute FAILED: %d (%s)\n",
               int(smem_result), cudaGetErrorString(smem_result));
      } else {
        printf("[SM120 Gated FC1 DEBUG] cudaFuncSetAttribute succeeded for smem=%d\n", smem_size);
      }
    }
  }
  
  // Run GEMM
  auto run_status = gemm.run(stream);
  cudaError_t cuda_err = cudaPeekAtLastError();  // Get error immediately after run

  // IMPORTANT: An illegal-instruction trap can leave the stream in an error state even if CUTLASS
  // returns `Success`. Under debug, surface this immediately to preserve the first-fault signal.
  if (dump_pre || dump_post) {
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    cudaError_t last_err = cudaGetLastError();
    if (sync_err != cudaSuccess || last_err != cudaSuccess) {
      TLLM_LOG_DEBUG("[SM120 Gated FC1] CUDA error after run (status=%s): sync=%s getLastError=%s",
                     cutlassGetStatusString(run_status),
                     cudaGetErrorString(sync_err),
                     cudaGetErrorString(last_err));
      if (dump_post) {
        dump_gated_tensormaps("post_run_cuda_error");
      }
      TLLM_THROW("SM120 Gated FC1: CUDA error after run: sync=%s getLastError=%s",
                 cudaGetErrorString(sync_err),
                 cudaGetErrorString(last_err));
    }
  }
  
  if (run_status != cutlass::Status::kSuccess) {
    // Sync to ensure any async kernel errors are surfaced
    cudaDeviceSynchronize();
    cudaError_t sync_err = cudaPeekAtLastError();
    printf("[SM120 Gated FC1 DEBUG] run failed: status=%d, CUDA error after run=%d (%s), after sync=%d (%s)\n",
           int(run_status), int(cuda_err), cudaGetErrorString(cuda_err),
           int(sync_err), cudaGetErrorString(sync_err));
    if (dump_post) {
      dump_gated_tensormaps("post_run_failure");
    }
    TLLM_THROW("SM120 Gated FC1: run failed: %s (CUDA: %s)",
               cutlassGetStatusString(run_status), cudaGetErrorString(sync_err));
  }
  TLLM_LOG_DEBUG("[SM120 Gated FC1] GEMM run launched");

  if (dump_post) {
    dump_gated_tensormaps("post_run_success");
  }

  if (occupancy != nullptr) {
    *occupancy = 1;
  }

#else
  (void)tma_inputs;
  (void)inter_size; (void)hidden_size;
  (void)num_experts; (void)multi_processor_count; (void)stream;
  (void)occupancy; (void)workspace_size; (void)swiglu_params;
  TLLM_THROW("SM120 Gated FC1 requires CUTLASS_ARCH_MMA_SM12x_SUPPORTED, ENABLE_FP4, and FLASHINFER_GATED_FC1");
#endif
}

// =============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS for gated FC1 launcher
// =============================================================================
// These are required because the template is defined here but called from
// cutlass_fused_moe_kernels.cuh with specific types.
//
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4) && defined(FLASHINFER_GATED_FC1)

// FP8 activations × FP4 weights → BF16 output (MXFP4)
// Default tile is 64x64x128 (fits SM121 SMEM).
// Optionally enable CTA_N=128 experimentation via FLASHINFER_GATED_FC1_CTA_N128.
template void sm120_gated_fc1_moe_gemm_kernelLauncher<
    __nv_fp8_e4m3,      // T (activation type)
    __nv_fp4_e2m1,      // WeightType
    __nv_bfloat16,      // OutputType
    void,               // EpilogueTag
#ifdef FLASHINFER_GATED_FC1_CTA_N128
    cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>,  // TileShape (64x128x128)
#else
    cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>,   // TileShape (64x64x128)
#endif
    cute::Shape<cute::_1, cute::_1, cute::_1>,                   // ClusterShape
    true                // IsMXFP4
>(
    TmaWarpSpecializedGroupedGemmInput tma_inputs,
    int64_t inter_size,
    int64_t hidden_size,
    int num_experts,
    int multi_processor_count,
    cudaStream_t stream,
    int* occupancy,
    size_t* workspace_size,
    GatedFC1SwigluParams swiglu_params);

#endif  // CUTLASS_ARCH_MMA_SM12x_SUPPORTED && ENABLE_FP4 && FLASHINFER_GATED_FC1

// =============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS for two-GEMM gated FC1 launcher (Path A)
// =============================================================================
#if defined(CUTLASS_ARCH_MMA_SM12x_SUPPORTED) && defined(ENABLE_FP4) && defined(FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP)

template void sm120_two_gemm_gated_fc1_kernelLauncher<
    __nv_fp8_e4m3,      // T (activation type)
    __nv_fp4_e2m1,      // WeightType
    __nv_bfloat16,      // OutputType
    void,               // EpilogueTag
    cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>,  // TileShape (compile-only tag)
    cute::Shape<cute::_1, cute::_1, cute::_1>,                    // ClusterShape
    true                // IsMXFP4
>(
    TmaWarpSpecializedGroupedGemmInput tma_inputs,
    void* linear_output,
    void* swiglu_output,
    int64_t const* expert_first_token_offset,
    int64_t inter_size,
    int64_t hidden_size,
    int num_experts,
    int multi_processor_count,
    cudaStream_t stream,
    int* occupancy,
    size_t* workspace_size);

#endif  // CUTLASS_ARCH_MMA_SM12x_SUPPORTED && ENABLE_FP4 && FLASHINFER_GATED_FC1_TWO_GEMM_BRINGUP

}  // namespace tensorrt_llm::kernels::cutlass_kernels_oss
