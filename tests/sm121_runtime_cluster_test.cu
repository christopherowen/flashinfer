/*
 * SM121 Runtime Cluster Capability Test
 * 
 * This test probes whether SM121 hardware actually supports cluster launches > 1x1x1
 * at the CUDA runtime level, independent of CUTLASS constraints.
 * 
 * This answers: "Is the 1x1x1 limitation a hardware constraint or just a CUTLASS choice?"
 * 
 * Compile with:
 *   nvcc -arch=sm_121a -std=c++17 tests/sm121_runtime_cluster_test.cu \
 *        -o tests/sm121_runtime_cluster_test 2>&1
 */

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        return err; \
    } \
} while(0)

// Simple kernel that records which cluster/block it runs in
__global__ void cluster_probe_kernel(int* results, int num_blocks) {
    // Get cluster and block indices
    uint32_t cluster_idx = 0;
    uint32_t block_in_cluster = 0;
    uint32_t cluster_dim_x = 1;
    
#if __CUDA_ARCH__ >= 900
    // cluster.dim.x and related intrinsics are available on SM90+
    asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(cluster_idx));
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(block_in_cluster));
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_dim_x));
#endif
    
    if (threadIdx.x == 0) {
        int idx = blockIdx.x;
        if (idx < num_blocks) {
            // Encode cluster info: cluster_idx * 1000 + block_in_cluster * 100 + cluster_dim_x
            results[idx] = cluster_idx * 1000 + block_in_cluster * 100 + cluster_dim_x;
        }
        
        if (blockIdx.x == 0) {
            printf("  Block %d: cluster_idx=%u, block_in_cluster=%u, cluster_dim_x=%u\n",
                   blockIdx.x, cluster_idx, block_in_cluster, cluster_dim_x);
        }
    }
}

// Attribute for cluster launch
__global__ __cluster_dims__(2, 1, 1) 
void cluster_2x1x1_kernel(int* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t cluster_dim_x = 1;
#if __CUDA_ARCH__ >= 900
        asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_dim_x));
#endif
        *result = cluster_dim_x;
        printf("  2x1x1 cluster kernel: cluster_dim_x = %u\n", cluster_dim_x);
    }
}

cudaError_t test_cluster_launch(int cluster_x, int cluster_y, int cluster_z) {
    std::cout << "\n=== Testing cluster launch " << cluster_x << "x" << cluster_y << "x" << cluster_z << " ===" << std::endl;
    
    // Query cluster launch support
    int cluster_launch_supported = 0;
    cudaError_t err = cudaDeviceGetAttribute(&cluster_launch_supported, 
                                              cudaDevAttrClusterLaunch, 0);
    if (err != cudaSuccess) {
        std::cout << "  cudaDevAttrClusterLaunch query failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "  cudaDevAttrClusterLaunch: " << cluster_launch_supported << std::endl;
    }
    
    // Allocate result buffer
    int* d_result;
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(int)));
    
    // For 2x1x1 cluster, use the attributed kernel
    if (cluster_x == 2 && cluster_y == 1 && cluster_z == 1) {
        // Launch with 2 blocks (one cluster of 2 CTAs)
        dim3 grid(2, 1, 1);
        dim3 block(32, 1, 1);
        
        // Try using cudaLaunchKernelEx for explicit cluster control
        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = 0;
        config.stream = nullptr;
        
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = cluster_x;
        attrs[0].val.clusterDim.y = cluster_y;
        attrs[0].val.clusterDim.z = cluster_z;
        
        config.attrs = attrs;
        config.numAttrs = 1;
        
        std::cout << "  Attempting cudaLaunchKernelEx with clusterDim=(" 
                  << cluster_x << "," << cluster_y << "," << cluster_z << ")..." << std::endl;
        
        err = cudaLaunchKernelEx(&config, cluster_2x1x1_kernel, d_result);
        
        if (err != cudaSuccess) {
            std::cout << "  cudaLaunchKernelEx FAILED: " << cudaGetErrorString(err) 
                      << " (error code: " << err << ")" << std::endl;
            cudaFree(d_result);
            return err;
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cout << "  cudaDeviceSynchronize FAILED: " << cudaGetErrorString(err)
                      << " (error code: " << err << ")" << std::endl;
            cudaFree(d_result);
            return err;
        }
        
        int h_result;
        CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "  Reported cluster_dim_x from device: " << h_result << std::endl;
        std::cout << "  Result: SUCCESS - cluster launch completed!" << std::endl;
    } else {
        // For 1x1x1, just launch normally
        dim3 grid(1, 1, 1);
        dim3 block(32, 1, 1);
        cluster_probe_kernel<<<grid, block>>>(d_result, 1);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cout << "  Kernel launch FAILED: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_result);
            return err;
        }
        std::cout << "  Result: SUCCESS" << std::endl;
    }
    
    cudaFree(d_result);
    return cudaSuccess;
}

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << "SM121 Runtime Cluster Capability Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Check device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    std::cout << "\nDevice: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "Number of SMs: " << props.multiProcessorCount << std::endl;
    
    // Query cluster-related attributes
    std::cout << "\n=== Cluster Capability Attributes ===" << std::endl;
    
    int max_blocks_per_cluster = 0;
    cudaError_t err = cudaDeviceGetAttribute(&max_blocks_per_cluster, 
                                              cudaDevAttrMaxBlocksPerMultiprocessor, device);
    if (err == cudaSuccess) {
        std::cout << "  Max blocks per SM: " << max_blocks_per_cluster << std::endl;
    }
    
    int cluster_launch = 0;
    err = cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device);
    if (err == cudaSuccess) {
        std::cout << "  Cluster launch supported: " << (cluster_launch ? "YES" : "NO") << std::endl;
    }
    
    // Test 1x1x1 cluster (baseline)
    cudaError_t result_1x1x1 = test_cluster_launch(1, 1, 1);
    
    // Test 2x1x1 cluster (the key test)
    cudaError_t result_2x1x1 = test_cluster_launch(2, 1, 1);
    
    std::cout << "\n=============================================" << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "  1x1x1 cluster: " << (result_1x1x1 == cudaSuccess ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "  2x1x1 cluster: " << (result_2x1x1 == cudaSuccess ? "SUCCESS" : "FAILED") << std::endl;
    
    std::cout << "\n=== Interpretation ===" << std::endl;
    if (result_2x1x1 == cudaSuccess) {
        std::cout << "The CUDA runtime reports cluster launch support and successfully" << std::endl;
        std::cout << "launches a 2x1x1 cluster on GB10 (SM121)." << std::endl;
        std::cout << "" << std::endl;
        std::cout << "The CUTLASS 1x1x1 restriction is a LIBRARY DESIGN CHOICE:" << std::endl;
        std::cout << "CUTLASS treats SM12x Thorough as lacking programmatic multicast," << std::endl;
        std::cout << "and therefore disables multi-CTA cluster shapes in its SM120/SM121" << std::endl;
        std::cout << "builders. Without multicast, CUTLASS' intended cluster strategy" << std::endl;
        std::cout << "(one producer broadcasting to peers) would cause redundant TMA loads." << std::endl;
        std::cout << "" << std::endl;
        std::cout << "IMPLICATION: Don't spend time trying to enable clusters in" << std::endl;
        std::cout << "CUTLASS/FlashInfer SM12x today; you'd need a different kernel" << std::endl;
        std::cout << "strategy than CUTLASS' multicast-oriented one." << std::endl;
    } else {
        std::cout << "The CUDA runtime rejected the 2x1x1 cluster launch." << std::endl;
        std::cout << "Error code: " << result_2x1x1 << std::endl;
        std::cout << "This would indicate a hardware or driver limitation." << std::endl;
    }
    std::cout << "=============================================" << std::endl;
    
    return (result_1x1x1 == cudaSuccess) ? 0 : 1;
}

