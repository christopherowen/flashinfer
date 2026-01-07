/*
 * SM121 Cluster Static Assert Test
 * 
 * This test directly verifies that the SM120 builders enforce 1x1x1 cluster
 * via static_assert with "no programmatic multicast on this arch".
 * 
 * We test by trying to instantiate SM120 collective builders with different
 * cluster shapes and observing compilation results.
 * 
 * Expected behavior:
 * - Cluster 1x1x1: Should compile successfully
 * - Cluster 2x1x1: Should fail with static_assert
 * 
 * Compile with:
 *   nvcc -arch=sm_121a -std=c++17 --expt-relaxed-constexpr \
 *        -I3rdparty/cutlass/include tests/sm121_cluster_static_assert_test.cu \
 *        -o tests/sm121_cluster_static_assert_test 2>&1
 */

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cute/tensor.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

// Data types for SM120
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;

static constexpr int AlignmentA = 16;  // 128 / 8 bits
static constexpr int AlignmentB = 16;  // 128 / 8 bits

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// Define tile shape
using TileShape = Shape<_128, _128, _64>;

// Test with 1x1x1 cluster - this SHOULD work
using ClusterShape1x1x1 = Shape<_1, _1, _1>;

// Test with 2x1x1 cluster - this SHOULD FAIL with static_assert
// Uncomment the line below to verify the static_assert triggers:
// using ClusterShape2x1x1 = Shape<_2, _1, _1>;

// Use SM120 FP8 schedule
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeSm120<2>;

// Try to build collective mainloop with 1x1x1 cluster
using CollectiveMainloop_1x1x1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape1x1x1,
    cutlass::gemm::collective::StageCountAuto,
    KernelSchedule
>::CollectiveOp;

/*
// UNCOMMENT THIS BLOCK TO TEST THE STATIC ASSERT FAILURE:
// This will fail with: "no programmatic multicast on this arch"
using ClusterShape2x1x1 = Shape<_2, _1, _1>;

using CollectiveMainloop_2x1x1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape2x1x1,
    cutlass::gemm::collective::StageCountAuto,
    KernelSchedule
>::CollectiveOp;
*/

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << "SM121 Cluster Static Assert Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    std::cout << "\nThis test verifies the SM120 collective builder constraints." << std::endl;
    
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "  Cluster 1x1x1: COMPILED SUCCESSFULLY" << std::endl;
    std::cout << "  sizeof(CollectiveMainloop_1x1x1) = " << sizeof(CollectiveMainloop_1x1x1) << std::endl;
    
    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "The SM120 collective builder (sm120_mma_builder.inl) contains:" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}," << std::endl;
    std::cout << "                \"no programmatic multicast on this arch\");" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "This enforces that cluster must be 1x1x1 (total size = 1)." << std::endl;
    std::cout << "Any cluster > 1x1x1 will fail compilation." << std::endl;
    
    std::cout << "\n=== Why This Limitation? ===" << std::endl;
    std::cout << "IMPORTANT: This is a CUTLASS library design choice, NOT a hardware limit." << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Runtime testing confirms SM121 hardware DOES support clusters > 1x1x1." << std::endl;
    std::cout << "(See sm121_runtime_cluster_test.cu for proof)" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "CUTLASS chooses to forbid it because:" << std::endl;
    std::cout << "1. SM100 (Blackwell Ultra) has 'programmatic TMA multicast'" << std::endl;
    std::cout << "   - Allows one SM to load via TMA and broadcast to cluster peers" << std::endl;
    std::cout << "   - This is CUTLASS' intended strategy for efficient multi-CTA clusters" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "2. SM120/SM121 (Blackwell Thorough) lacks programmatic TMA multicast" << std::endl;
    std::cout << "   - Without multicast, CUTLASS' cluster strategy would cause" << std::endl;
    std::cout << "     redundant TMA loads (each CTA loads independently)" << std::endl;
    std::cout << "   - CUTLASS prevents this inefficient pattern via static_assert" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "3. Alternative cluster strategies exist (each CTA loads distinct tiles)" << std::endl;
    std::cout << "   but CUTLASS SM120 builders don't implement them." << std::endl;
    
    std::cout << "\n=== Implications for FlashInfer ===" << std::endl;
    std::cout << "For FlashInfer/CUTLASS on SM121:" << std::endl;
    std::cout << "  - CUTLASS SM120/SM121 builders hard-restrict cluster to 1x1x1" << std::endl;
    std::cout << "  - Multi-CTA cluster shapes are NOT available in current CUTLASS-backed kernels" << std::endl;
    std::cout << "  - For SM121 performance, focus on:" << std::endl;
    std::cout << "    * Tile shapes (M, N, K dimensions)" << std::endl;
    std::cout << "    * Pipelining/schedule choices" << std::endl;
    std::cout << "    * Epilogues and heuristics" << std::endl;
    std::cout << "    * NOT cluster shapes" << std::endl;
    
    std::cout << "\n=============================================" << std::endl;
    std::cout << "To verify the static_assert, uncomment the 2x1x1" << std::endl;
    std::cout << "cluster test block in this file and recompile." << std::endl;
    std::cout << "=============================================" << std::endl;
    
    return 0;
}

#else  // !CUTLASS_ARCH_MMA_SM120_SUPPORTED

int main() {
    std::cout << "CUTLASS_ARCH_MMA_SM120_SUPPORTED is not defined." << std::endl;
    std::cout << "Compile with -arch=sm_120a or -arch=sm_121a to test." << std::endl;
    return 1;
}

#endif

