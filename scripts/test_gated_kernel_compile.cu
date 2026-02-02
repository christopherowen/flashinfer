/**
 * Standalone compile test for GemmUniversalGated kernel.
 * 
 * This file instantiates just the kernel types to catch template errors
 * without running the full vLLM stack.
 * 
 * Compile with:
 *   cd /workspace/flashinfer && \
 *   nvcc -std=c++17 -arch=sm_121a \
 *     -I 3rdparty/cutlass/include \
 *     -I csrc/nv_internal/tensorrt_llm/cutlass_extensions/include \
 *     -I csrc/nv_internal/tensorrt_llm \
 *     -I /usr/local/cuda/include \
 *     -DENABLE_FP4 -DFLASHINFER_GATED_FC1 \
 *     -c scripts/test_gated_kernel_compile.cu \
 *     -o /tmp/test_gated_kernel_compile.o \
 *     2>&1 | head -100
 */

#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_bf16.h>

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"  // Primary CollectiveMma template
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// Our custom gated kernel
#include "cutlass_extensions/gemm/kernel/sm120_gemm_gated_array_tma_warpspecialized.hpp"
#include "cutlass_extensions/gemm/collective/sm120_blockscaled_mma_gated_array_tma.hpp"
#include "cutlass_extensions/epilogue/sm120_gated_swiglu_epilogue.hpp"

// Type definitions matching the gated namespace
namespace test_gated {

using namespace cute;

// Basic types
using ElementA = __nv_fp8_e4m3;
using ElementB = __nv_fp4_e2m1;
using ElementAccumulator = float;
using ElementOutput = cutlass::bfloat16_t;
using ElementSF = cutlass::float_ue8m0_t;

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;

// Tile shape
using TileShape_MNK = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;

// Architecture
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Problem shape for grouped GEMM
using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

// Gated dispatch policy
using GatedDispatchPolicy = cutlass::gemm::collective::MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated<
    /*Stages=*/2,
    /*SchedulerPipelineStageCount=*/2,
    ClusterShape_MNK,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>;

// ============================================================================
// Step 1: Verify dispatch policy exists
// ============================================================================
static_assert(GatedDispatchPolicy::IsGated, "GatedDispatchPolicy must have IsGated=true");

// ============================================================================
// Step 2: Try to instantiate the gated epilogue
// ============================================================================
using StrideOutput = cute::Stride<int64_t, cute::Int<1>, int64_t>;
using GatedEpilogue = cutlass::epilogue::Sm120GatedSwiGLUEpilogue<
    TileShape_MNK,
    ElementOutput,      // bfloat16 output
    StrideOutput,
    ElementAccumulator  // float accumulator
>;

// Verify epilogue has required aliases
static_assert(sizeof(typename GatedEpilogue::ElementOutput) > 0, "Epilogue must have ElementOutput");

// ============================================================================
// Step 3: Verify types compile
// ============================================================================
static_assert(sizeof(GatedDispatchPolicy) > 0, "GatedDispatchPolicy must exist");
static_assert(sizeof(GatedEpilogue) > 0, "GatedEpilogue must exist");
static_assert(GatedDispatchPolicy::IsGated == true, "Must be gated");

} // namespace test_gated

// Dummy kernel to ensure file compiles
__global__ void dummy_kernel() {}

int main() {
    printf("Gated kernel compile test passed!\n");
    printf("  GatedDispatchPolicy::IsGated = %d\n", test_gated::GatedDispatchPolicy::IsGated);
    printf("  sizeof(GatedEpilogue) = %zu\n", sizeof(test_gated::GatedEpilogue));
    return 0;
}
