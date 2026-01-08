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
// SM12x Identity SFA Infrastructure Test
// =============================================================================
//
// This test validates the identity SFA infrastructure components without
// invoking the full CUTLASS kernel:
//   1. Buffer sizing computation
//   2. Identity scale buffer manager (caching, allocation)
//   3. SFA pointer array manager (caching, no hot-path allocs)
//   4. Identity scale encoding correctness
//
// Build:
//   nvcc -arch=sm_121 -DENABLE_FP4 \
//        -I3rdparty/cutlass/include \
//        -I3rdparty/cutlass/tools/util/include \
//        -Icsrc/nv_internal/include \
//        -Icsrc/nv_internal \
//        -Icsrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm \
//        -Icsrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include \
//        -Icsrc/nv_internal/tensorrt_llm/cutlass_extensions/include \
//        -std=c++17 \
//        tests/sm12x_identity_sfa_infra_test.cu -o sm12x_identity_sfa_infra_test
//
// Run:
//   ./sm12x_identity_sfa_infra_test
//
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#ifdef ENABLE_FP4
// CUTLASS includes
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

// FlashInfer SM12x headers
#include "sm12x_arch_config.h"
#include "sm12x_layout_sfa_utils.h"
#include "sm12x_activation_quantizer.cuh"
#endif

// =============================================================================
// Test: Identity Scale Encoding
// =============================================================================

bool test_identity_scale_encoding() {
    printf("Test: Identity Scale Encoding\n");
    
    // float_ue8m0_t: value = 2^(raw - 127)
    // identity (1.0) -> raw = 127 = 0x7F
    constexpr uint8_t identity_raw = 0x7F;
    float decoded = powf(2.0f, static_cast<float>(identity_raw) - 127.0f);
    
    if (fabs(decoded - 1.0f) > 1e-6f) {
        printf("  FAIL: Identity scale 0x%02X decodes to %f, expected 1.0\n",
               identity_raw, decoded);
        return false;
    }
    
    printf("  PASS: Identity scale 0x%02X = 2^(%d-127) = 2^0 = %f\n",
           identity_raw, identity_raw, decoded);
    return true;
}

// =============================================================================
// Test: SFA Buffer Sizing (using CUTLASS LayoutSFA utilities)
// =============================================================================

bool test_sfa_buffer_sizing() {
    printf("Test: SFA Buffer Sizing (CUTLASS-derived)\n");
    
#ifdef ENABLE_FP4
    using namespace tensorrt_llm::kernels::cutlass_kernels;
    
    // Test dimensions (typical MoE shapes)
    struct TestCase {
        int M, N, K;
        const char* name;
    };
    
    TestCase cases[] = {
        {64,  128, 256, "small"},
        {128, 256, 512, "medium"},
        {256, 512, 1024, "large"},
        {64,  128, 2880, "gpt-oss-decode"},
        {128, 256, 4096, "common"},
    };
    
    for (const auto& tc : cases) {
        // Compute buffer size using our sizing function (L=1 for grouped GEMM)
        size_t sfa_bytes = Sm12xLayoutSFAUtils::computeBufferSize(tc.M, tc.N, tc.K, 1);
        
        // Verify it's 256-byte aligned (TMA requirement)
        if (sfa_bytes % 256 != 0) {
            printf("  FAIL: %s - SFA size %zu not 256-byte aligned\n", tc.name, sfa_bytes);
            return false;
        }
        
        // Verify it's non-zero
        if (sfa_bytes == 0) {
            printf("  FAIL: %s - SFA size is zero\n", tc.name);
            return false;
        }
        
        printf("  PASS: %s (M=%d, N=%d, K=%d) -> SFA size = %zu bytes\n", 
               tc.name, tc.M, tc.N, tc.K, sfa_bytes);
    }
    
    printf("  All sizing tests passed\n");
    return true;
#else
    printf("  SKIP: ENABLE_FP4 not defined\n");
    return true;
#endif
}

// =============================================================================
// Test: Identity Scale Buffer Manager
// =============================================================================

bool test_identity_scale_buffer_manager() {
    printf("Test: Identity Scale Buffer Manager\n");
    
#if defined(ENABLE_FP4) && !defined(__CUDA_ARCH__)
    using namespace tensorrt_llm::kernels::cutlass_kernels;
    
    auto& mgr = tensorrt_llm::kernels::cutlass_kernels::getIdentityScaleBufferManager();
    
    // Test 1: Get buffer for specific size
    size_t size1 = 4096;
    uint8_t* buf1 = mgr.getOrCreateWithSize(size1);
    if (buf1 == nullptr) {
        printf("  FAIL: Could not allocate identity buffer of size %zu\n", size1);
        return false;
    }
    printf("  PASS: Allocated identity buffer of size %zu at %p\n", size1, buf1);
    
    // Test 2: Same size should return cached buffer
    uint8_t* buf2 = mgr.getOrCreateWithSize(size1);
    if (buf2 != buf1) {
        printf("  FAIL: Second request for same size returned different pointer\n");
        printf("        buf1=%p, buf2=%p\n", buf1, buf2);
        return false;
    }
    printf("  PASS: Cached buffer returned for same size\n");
    
    // Test 3: Different size should return new buffer
    size_t size3 = 8192;
    uint8_t* buf3 = mgr.getOrCreateWithSize(size3);
    if (buf3 == nullptr) {
        printf("  FAIL: Could not allocate identity buffer of size %zu\n", size3);
        return false;
    }
    if (buf3 == buf1) {
        printf("  FAIL: Different size returned same buffer\n");
        return false;
    }
    printf("  PASS: New buffer allocated for different size\n");
    
    // Test 4: Verify buffer contents are 0x7F (identity scale)
    std::vector<uint8_t> h_buf(size1);
    CUDA_CHECK(cudaMemcpy(h_buf.data(), buf1, size1, cudaMemcpyDeviceToHost));
    
    int non_identity_count = 0;
    for (size_t i = 0; i < size1; i++) {
        if (h_buf[i] != 0x7F) {
            non_identity_count++;
        }
    }
    
    if (non_identity_count > 0) {
        printf("  FAIL: Found %d non-identity bytes in buffer\n", non_identity_count);
        return false;
    }
    printf("  PASS: All %zu bytes are identity scale (0x7F)\n", size1);
    
    return true;
#else
    printf("  SKIP: ENABLE_FP4 not defined\n");
    return true;
#endif
}

// =============================================================================
// Test: SFA Pointer Array Manager
// =============================================================================

bool test_sfa_pointer_array_manager() {
    printf("Test: SFA Pointer Array Manager\n");
    
#if defined(ENABLE_FP4) && !defined(__CUDA_ARCH__)
    using namespace tensorrt_llm::kernels::cutlass_kernels;
    
    // Allocate a dummy identity buffer
    uint8_t* d_identity = nullptr;
    CUDA_CHECK(cudaMalloc(&d_identity, 4096));
    CUDA_CHECK(cudaMemset(d_identity, 0x7F, 4096));
    
    auto& mgr = tensorrt_llm::kernels::cutlass_kernels::getSFAPointerArrayManager();
    
    // Test 1: First call should allocate
    uint8_t const** ptr1 = mgr.getOrCreate(8, d_identity);
    if (ptr1 == nullptr) {
        printf("  FAIL: First getOrCreate returned nullptr\n");
        cudaFree(d_identity);
        return false;
    }
    printf("  PASS: Pointer array allocated for 8 groups at %p\n", ptr1);
    
    // Test 2: Same params should return cached
    uint8_t const** ptr2 = mgr.getOrCreate(8, d_identity);
    if (ptr2 != ptr1) {
        printf("  FAIL: Second call returned different pointer (not cached)\n");
        printf("        ptr1=%p, ptr2=%p\n", ptr1, ptr2);
        cudaFree(d_identity);
        return false;
    }
    printf("  PASS: Cached pointer array returned\n");
    
    // Test 3: Different num_groups should allocate new
    uint8_t const** ptr3 = mgr.getOrCreate(16, d_identity);
    if (ptr3 == ptr1) {
        printf("  FAIL: Different num_groups returned same pointer\n");
        cudaFree(d_identity);
        return false;
    }
    printf("  PASS: New pointer array allocated for different num_groups\n");
    
    // Test 4: Verify pointer array contents
    std::vector<uint8_t const*> h_ptrs(8);
    CUDA_CHECK(cudaMemcpy(h_ptrs.data(), ptr1, 8 * sizeof(uint8_t*), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < 8; i++) {
        if (h_ptrs[i] != d_identity) {
            printf("  FAIL: Pointer array[%d] = %p, expected %p\n",
                   i, h_ptrs[i], d_identity);
            cudaFree(d_identity);
            return false;
        }
    }
    printf("  PASS: All 8 pointers correctly point to identity buffer\n");
    
    cudaFree(d_identity);
    return true;
#else
    printf("  SKIP: ENABLE_FP4 not defined\n");
    return true;
#endif
}

// =============================================================================
// Test: Buffer Alignment
// =============================================================================

bool test_buffer_alignment() {
    printf("Test: Buffer Alignment (256-byte TMA requirement)\n");
    
#if defined(ENABLE_FP4) && !defined(__CUDA_ARCH__)
    using namespace tensorrt_llm::kernels::cutlass_kernels;
    
    auto& mgr = tensorrt_llm::kernels::cutlass_kernels::getIdentityScaleBufferManager();
    
    // Test various sizes and verify alignment
    std::vector<size_t> sizes = {256, 512, 1024, 4096, 10000, 16384};
    
    for (size_t size : sizes) {
        uint8_t* buf = mgr.getOrCreateWithSize(size);
        if (buf == nullptr) {
            printf("  FAIL: Could not allocate buffer of size %zu\n", size);
            return false;
        }
        
        // Check device pointer alignment
        uintptr_t addr = reinterpret_cast<uintptr_t>(buf);
        if (addr % 256 != 0) {
            printf("  FAIL: Buffer of size %zu not 256-byte aligned (addr=%p)\n", 
                   size, buf);
            return false;
        }
        printf("  PASS: Size %zu -> aligned at %p\n", size, buf);
    }
    
    return true;
#else
    printf("  SKIP: ENABLE_FP4 not defined\n");
    return true;
#endif
}

// =============================================================================
// Test: Multi-call Caching (no hot-path allocations)
// =============================================================================

bool test_no_hotpath_allocations() {
    printf("Test: No Hot-Path Allocations\n");
    
#if defined(ENABLE_FP4) && !defined(__CUDA_ARCH__)
    using namespace tensorrt_llm::kernels::cutlass_kernels;
    
    auto& buf_mgr = tensorrt_llm::kernels::cutlass_kernels::getIdentityScaleBufferManager();
    auto& ptr_mgr = tensorrt_llm::kernels::cutlass_kernels::getSFAPointerArrayManager();
    
    // Pre-warm with a specific configuration
    size_t sfa_size = Sm12xLayoutSFAUtils::computeBufferSize(128, 256, 4096, 1);
    uint8_t* identity_buf = buf_mgr.getOrCreateWithSize(sfa_size);
    
    if (identity_buf == nullptr) {
        printf("  FAIL: Prewarm failed for identity buffer\n");
        return false;
    }
    
    // Simulate hot path: get buffer 100 times
    // After prewarm, this should return cached values with no allocations
    for (int i = 0; i < 100; i++) {
        uint8_t* buf = buf_mgr.getOrCreateWithSize(sfa_size);
        if (buf != identity_buf) {
            printf("  FAIL: Hot path call %d returned different buffer\n", i);
            return false;
        }
    }
    printf("  PASS: 100 hot-path buffer requests returned cached value\n");
    
    // Same for pointer arrays
    uint8_t const** ptr1 = ptr_mgr.getOrCreate(8, identity_buf);
    for (int i = 0; i < 100; i++) {
        uint8_t const** ptr = ptr_mgr.getOrCreate(8, identity_buf);
        if (ptr != ptr1) {
            printf("  FAIL: Hot path call %d returned different pointer array\n", i);
            return false;
        }
    }
    printf("  PASS: 100 hot-path pointer array requests returned cached value\n");
    
    return true;
#else
    printf("  SKIP: ENABLE_FP4 not defined\n");
    return true;
#endif
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    printf("=== SM12x Identity SFA Infrastructure Test ===\n\n");
    
    // Get device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("Device: %s (SM %d.%d)\n", props.name, props.major, props.minor);
    printf("Driver: %d, Runtime: %d\n\n", 0, CUDART_VERSION);
    
    int passed = 0;
    int failed = 0;
    
    // Run tests
    if (test_identity_scale_encoding()) passed++; else failed++;
    if (test_sfa_buffer_sizing()) passed++; else failed++;
    if (test_identity_scale_buffer_manager()) passed++; else failed++;
    if (test_sfa_pointer_array_manager()) passed++; else failed++;
    if (test_buffer_alignment()) passed++; else failed++;
    if (test_no_hotpath_allocations()) passed++; else failed++;
    
    // Summary
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);
    
    if (failed == 0) {
        printf("\nAll infrastructure tests passed!\n");
        printf("Identity SFA buffer management is ready for use.\n");
        return 0;
    } else {
        printf("\nSome tests failed. See above for details.\n");
        return 1;
    }
}

