/*
 * SM121 Cluster Shape Support Compilation Test
 * 
 * This minimal test verifies what architecture macros are defined when
 * compiling for SM121, and explains the 1x1x1 cluster limitation.
 * 
 * Compile with:
 *   nvcc -arch=sm_121a -std=c++17 --expt-relaxed-constexpr \
 *        -I3rdparty/cutlass/include tests/sm121_cluster_test.cu \
 *        -o tests/sm121_cluster_test 2>&1
 */

#include <iostream>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/dispatch_policy.hpp"

// Check what macros are defined
void print_arch_support() {
    std::cout << "\n=== CUTLASS Architecture Macro Status ===" << std::endl;
    
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    std::cout << "  CUTLASS_ARCH_MMA_SM120_SUPPORTED: YES" << std::endl;
#else
    std::cout << "  CUTLASS_ARCH_MMA_SM120_SUPPORTED: NO" << std::endl;
#endif

#if defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
    std::cout << "  CUTLASS_ARCH_MMA_SM121_SUPPORTED: YES" << std::endl;
#else
    std::cout << "  CUTLASS_ARCH_MMA_SM121_SUPPORTED: NO" << std::endl;
#endif

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    std::cout << "  CUTLASS_ARCH_MMA_SM100_SUPPORTED: YES" << std::endl;
#else
    std::cout << "  CUTLASS_ARCH_MMA_SM100_SUPPORTED: NO" << std::endl;
#endif
}

// Test: Check if SM120 and SM100 schedules exist
void test_schedules() {
    std::cout << "\n=== Schedule Types Compilation Test ===" << std::endl;
    
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    // SM120 schedules (template parameter is pipeline stage count)
    std::cout << "  SM120 Cooperative<2>: " 
              << sizeof(cutlass::gemm::KernelTmaWarpSpecializedCooperativeSm120<2>) << " bytes" << std::endl;
    std::cout << "  SM120 Pingpong<2>: " 
              << sizeof(cutlass::gemm::KernelTmaWarpSpecializedPingpongSm120<2>) << " bytes" << std::endl;
    std::cout << "  SM120 schedules compile: SUCCESS" << std::endl;
#else
    std::cout << "  SM120 schedules: SKIPPED (macro not defined)" << std::endl;
#endif

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    // SM100 1SM and 2SM schedules
    std::cout << "  SM100 1SM: " 
              << sizeof(cutlass::gemm::KernelTmaWarpSpecialized1SmSm100) << " bytes" << std::endl;
    std::cout << "  SM100 2SM: " 
              << sizeof(cutlass::gemm::KernelTmaWarpSpecialized2SmSm100) << " bytes" << std::endl;
    std::cout << "  SM100 PtrArray 1SM: " 
              << sizeof(cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100) << " bytes" << std::endl;
    std::cout << "  SM100 PtrArray 2SM: " 
              << sizeof(cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100) << " bytes" << std::endl;
    std::cout << "  SM100 schedules compile: SUCCESS" << std::endl;
#else
    std::cout << "  SM100 schedules: SKIPPED (macro not defined)" << std::endl;
#endif
}

__global__ void device_arch_check() {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("  Device __CUDA_ARCH__: %d\n", __CUDA_ARCH__);
    }
#endif
}

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << "SM121 Cluster Shape Compilation Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Check device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    std::cout << "\nDevice: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Number of SMs: " << props.multiProcessorCount << std::endl;
    
    print_arch_support();
    test_schedules();
    
    // Run device kernel to verify CUDA arch
    std::cout << "\n=== Device Execution ===" << std::endl;
    device_arch_check<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        std::cout << "  Kernel execution: SUCCESS" << std::endl;
    } else {
        std::cout << "  Kernel execution: FAILED - " << cudaGetErrorString(err) << std::endl;
    }
    
    std::cout << "\n=============================================" << std::endl;
    std::cout << "ANALYSIS: Why 1x1x1 cluster only for SM120/SM121?" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "The CUTLASS SM120 collective builders enforce 1x1x1 cluster" << std::endl;
    std::cout << "via static_assert in these files:" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  sm120_mma_builder.inl:" << std::endl;
    std::cout << "    static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}," << std::endl;
    std::cout << "                  \"no programmatic multicast on this arch\");" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  Also in:" << std::endl;
    std::cout << "    - sm120_blockscaled_mma_builder.inl" << std::endl;
    std::cout << "    - sm120_sparse_mma_builder.inl" << std::endl;
    std::cout << "    - sm120_blockscaled_sparse_mma_builder.inl" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "The key phrase is 'no programmatic multicast on this arch'." << std::endl;
    std::cout << "" << std::endl;
    std::cout << "EXPLANATION:" << std::endl;
    std::cout << "  - SM100 (Blackwell Ultra) has 'programmatic TMA multicast'" << std::endl;
    std::cout << "    which allows efficient data sharing across multiple SMs" << std::endl;
    std::cout << "    in a cluster. This enables 2SM/4SM cooperative execution." << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  - SM120/SM121 (Blackwell Thorough) lacks this hardware feature." << std::endl;
    std::cout << "    Without TMA multicast, clusters > 1x1x1 would require" << std::endl;
    std::cout << "    separate TMA loads per SM, eliminating the bandwidth benefit." << std::endl;
    std::cout << "" << std::endl;
    std::cout << "This is a HARDWARE limitation, not just a CUTLASS software choice." << std::endl;
    std::cout << "SM100 and SM120/SM121 are different products with different features." << std::endl;
    std::cout << "" << std::endl;
    std::cout << "IMPLICATION:" << std::endl;
    std::cout << "  For FlashInfer on SM121, we can only use 1x1x1 clusters." << std::endl;
    std::cout << "  We should focus on tile shape optimization (M, N, K dimensions)" << std::endl;
    std::cout << "  rather than trying to use multi-SM clustering." << std::endl;
    std::cout << "=============================================" << std::endl;
    
    return 0;
}
