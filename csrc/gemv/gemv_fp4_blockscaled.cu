/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file gemv_fp4_blockscaled.cu
 * \brief FP4 block-scaled GEMV kernel for MoE decode optimization.
 *
 * This kernel is designed for small batch sizes (M<=16) where the grouped GEMM
 * kernel's 128x128 tiles result in poor compute efficiency.
 *
 * Implementations:
 * 1. Software dequant fallback (ACTIVE) - ~7 tok/s
 * 2. CUTLASS GemvBlockScaled native (WIP) - target ~58 tok/s
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

// TVM FFI bindings
#include "../tvm_ffi_utils.h"

using tvm::ffi::TensorView;

//==============================================================================
// Feature flags - enable when ready
//==============================================================================
// CUTLASS GemvBlockScaled integration is complex due to:
// 1. Namespace conflicts between cute::Tensor and other Tensor types
// 2. Custom epilogue requirements (FP4 output only in stock epilogue)
// 3. Template parameter matching with kernel internals
//
// For now, use the optimized fallback path.
// #define USE_CUTLASS_GEMV_NATIVE  // Enable CUTLASS native path

#ifdef USE_CUTLASS_GEMV_NATIVE
#include "cutlass/gemm/device/gemv_blockscaled.h"
#include "cutlass/gemm/kernel/gemv_blockscaled.h"
#include "gemv_epilogue_bf16.h"
#endif

namespace flashinfer {
namespace gemv {

// Configuration constants
static constexpr int kBlockSize = 32;  // MXFP4 block size for scale factors

//==============================================================================
// Software Dequantization Fallback
//==============================================================================

/*!
 * \brief Simple FP4 dequantization for GEMV fallback
 */
__global__ void gemv_fp4_dequant_kernel(
    const uint8_t* __restrict__ A_packed,   // [M, K/2] packed FP4 activations
    const uint8_t* __restrict__ B_packed,   // [K/2, N] packed FP4 weights  
    const uint8_t* __restrict__ A_scales,   // Activation scales (FP8 e4m3 as uint8)
    const uint8_t* __restrict__ B_scales,   // Weight scales (FP8 e4m3 as uint8)
    __nv_bfloat16* __restrict__ D,          // [M, N] output
    int M, int K, int N,
    float alpha, float beta
) {
    // Each thread handles one output element
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m >= M || n >= N) return;
    
    // Accumulate in FP32
    float acc = 0.0f;
    
    // K is the unpacked dimension
    int K_packed = K / 2;
    
    for (int k = 0; k < K_packed; k++) {
        // Unpack FP4 values (2 per byte)
        uint8_t a_byte = A_packed[m * K_packed + k];
        uint8_t b_byte = B_packed[k * N + n];
        
        // Low nibble and high nibble
        int a_lo = a_byte & 0x0F;
        int a_hi = (a_byte >> 4) & 0x0F;
        int b_lo = b_byte & 0x0F;
        int b_hi = (b_byte >> 4) & 0x0F;
        
        // Simple FP4 e2m1 dequantization (approximate)
        auto dequant_fp4 = [](int val) -> float {
            static const float lut[16] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
            };
            return lut[val];
        };
        
        // Get scale factors (one per block of 32 elements)
        int k_block = (k * 2) / kBlockSize;
        int m_block = m / kBlockSize;
        int n_block = n / kBlockSize;
        int num_k_blocks = (K + kBlockSize - 1) / kBlockSize;
        
        uint8_t a_scale_bits = A_scales[m_block * num_k_blocks + k_block];
        uint8_t b_scale_bits = B_scales[n_block * num_k_blocks + k_block];
        
        auto fp8_to_float = [](uint8_t bits) -> float {
            __nv_fp8_e4m3 fp8;
            memcpy(&fp8, &bits, 1);
            return float(fp8);
        };
        
        float a_scale = fp8_to_float(a_scale_bits);
        float b_scale = fp8_to_float(b_scale_bits);
        
        float a_val_lo = dequant_fp4(a_lo) * a_scale;
        float a_val_hi = dequant_fp4(a_hi) * a_scale;
        float b_val_lo = dequant_fp4(b_lo) * b_scale;
        float b_val_hi = dequant_fp4(b_hi) * b_scale;
        
        acc += a_val_lo * b_val_lo + a_val_hi * b_val_hi;
    }
    
    // Write output
    float result = alpha * acc;
    if (beta != 0.0f) {
        result += beta * __bfloat162float(D[m * N + n]);
    }
    D[m * N + n] = __float2bfloat16(result);
}

cudaError_t run_gemv_fp4_dequant(
    int m, int k, int n,
    float alpha, float beta,
    const void* ptr_A,
    const void* ptr_B,
    void* ptr_D,
    const void* ptr_SFA,
    const void* ptr_SFB,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    
    gemv_fp4_dequant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ptr_A),
        reinterpret_cast<const uint8_t*>(ptr_B),
        reinterpret_cast<const uint8_t*>(ptr_SFA),
        reinterpret_cast<const uint8_t*>(ptr_SFB),
        reinterpret_cast<__nv_bfloat16*>(ptr_D),
        m, k, n,
        alpha, beta
    );
    
    return cudaGetLastError();
}

//==============================================================================
// CUTLASS GemvBlockScaled Native Implementation
//==============================================================================

#ifdef USE_CUTLASS_GEMV_NATIVE

// Type definitions for CUTLASS GEMV
using ElementA = cutlass::float_e2m1_t;  // FP4 e2m1
using ElementSFA = cutlass::float_e4m3_t;  // FP8 scale
using LayoutA = cutlass::layout::RowMajor;

using ElementB = cutlass::float_e2m1_t;  // FP4 e2m1
using ElementSFB = cutlass::float_e4m3_t;  // FP8 scale

using ElementC = cutlass::bfloat16_t;  // BF16 output
using ElementD = cutlass::bfloat16_t;
using LayoutD = cutlass::layout::ColumnMajor;

using ElementAccumulatorMainloop = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

static constexpr int kVectorSize = 16;
static constexpr int kElementsPerAccess = 32;  // Fixed for FP4

using ThreadShape = cutlass::gemm::GemmShape<16, 8>;

// Custom epilogue for BF16 output
using EpilogueOp = flashinfer::epilogue::GemvEpilogueBF16<
    kVectorSize,
    ThreadShape,
    ElementCompute,
    ElementAccumulator,
    ElementC,
    ElementD,
    LayoutD
>;

// GEMV kernel type
using GemvKernel = cutlass::gemm::kernel::GemvBlockScaled<
    ElementA, LayoutA, ElementB, ElementD,
    ElementAccumulatorMainloop, EpilogueOp, kElementsPerAccess
>;

// Device wrapper
using GemvDevice = cutlass::gemm::device::GemvBlockScaled<GemvKernel>;

cudaError_t run_gemv_fp4_native(
    int m, int k, int n,
    float alpha, float beta,
    const void* ptr_A,
    const void* ptr_B,
    void* ptr_D,
    const void* ptr_SFA,
    const void* ptr_SFB,
    cudaStream_t stream
) {
    // Problem size (M rows, K cols, N=1 for GEMV)
    cutlass::MatrixCoord problem_size(m, k);
    int batch_count = n;  // Handle N>1 as batched GEMV

    // Construct arguments
    typename GemvDevice::Arguments arguments{
        problem_size,
        batch_count,
        typename EpilogueOp::Params{
            cutlass::TensorRef<ElementD, LayoutD>(
                reinterpret_cast<ElementD*>(ptr_D),
                LayoutD(m)
            ),
            ElementCompute(alpha),
            ElementCompute(beta),
            m * sizeof(ElementD),  // batch_stride_d
            m                      // stride_d
        },
        cutlass::TensorRef<const ElementA, LayoutA>(
            reinterpret_cast<const ElementA*>(ptr_A),
            LayoutA(k)
        ),
        reinterpret_cast<const ElementB*>(ptr_B),
        nullptr,  // ptr_C (no bias)
        reinterpret_cast<ElementD*>(ptr_D),
        reinterpret_cast<const ElementSFA*>(ptr_SFA),
        reinterpret_cast<const ElementSFB*>(ptr_SFB),
        k,              // stride_A
        m * k,          // batch_stride_A
        k,              // batch_stride_B
        m,              // batch_stride_C
        m,              // batch_stride_D
        0,              // batch_stride_SFA (computed internally)
        0,              // batch_stride_SFB
        0               // batch_stride_SFD (not used)
    };

    // Instantiate and run
    GemvDevice gemv_op;

    cutlass::Status status = gemv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = GemvDevice::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemv_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidConfiguration;
    }

    status = gemv_op(stream);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorLaunchFailure;
}

#endif  // USE_CUTLASS_GEMV_NATIVE

//==============================================================================
// Main dispatch function
//==============================================================================

cudaError_t run_gemv_fp4(
    int m, int k, int n,
    float alpha, float beta,
    const void* ptr_A,
    const void* ptr_B,
    void* ptr_D,
    const void* ptr_SFA,
    const void* ptr_SFB,
    cudaStream_t stream
) {
#ifdef USE_CUTLASS_GEMV_NATIVE
    return run_gemv_fp4_native(m, k, n, alpha, beta, ptr_A, ptr_B, ptr_D, ptr_SFA, ptr_SFB, stream);
#else
    return run_gemv_fp4_dequant(m, k, n, alpha, beta, ptr_A, ptr_B, ptr_D, ptr_SFA, ptr_SFB, stream);
#endif
}

}  // namespace gemv
}  // namespace flashinfer

//==============================================================================
// TVM FFI Bindings
//==============================================================================

void gemv_fp4_blockscaled(
    int64_t m,
    int64_t k,
    int64_t n,
    double alpha,
    double beta,
    TensorView A,
    TensorView B,
    TensorView D,
    TensorView SFA,
    TensorView SFB
) {
    cudaStream_t stream = nullptr;

    cudaError_t err = flashinfer::gemv::run_gemv_fp4(
        static_cast<int>(m),
        static_cast<int>(k),
        static_cast<int>(n),
        static_cast<float>(alpha),
        static_cast<float>(beta),
        A.data_ptr(),
        B.data_ptr(),
        D.data_ptr(),
        SFA.data_ptr(),
        SFB.data_ptr(),
        stream
    );

    TVM_FFI_ICHECK(err == cudaSuccess)
        << "GEMV FP4 kernel failed: " << cudaGetErrorString(err);
}

void batched_gemv_fp4(
    int64_t num_experts,
    int64_t m,
    int64_t k,
    int64_t n,
    double alpha,
    TensorView A,
    TensorView B,
    TensorView D,
    TensorView SFA,
    TensorView SFB
) {
    cudaStream_t stream = nullptr;

    size_t a_stride = m * (k / 2);
    size_t b_stride = (k / 2) * n;
    size_t d_stride = m * n * sizeof(__nv_bfloat16);
    size_t sfa_stride = ((m + 31) / 32) * ((k + 31) / 32);
    size_t sfb_stride = ((n + 31) / 32) * ((k + 31) / 32);
    
    for (int64_t e = 0; e < num_experts; e++) {
        cudaError_t err = flashinfer::gemv::run_gemv_fp4(
            static_cast<int>(m),
            static_cast<int>(k),
            static_cast<int>(n),
            static_cast<float>(alpha),
            0.0f,
            reinterpret_cast<const uint8_t*>(A.data_ptr()) + e * a_stride,
            reinterpret_cast<const uint8_t*>(B.data_ptr()) + e * b_stride,
            reinterpret_cast<uint8_t*>(D.data_ptr()) + e * d_stride,
            reinterpret_cast<const uint8_t*>(SFA.data_ptr()) + e * sfa_stride,
            reinterpret_cast<const uint8_t*>(SFB.data_ptr()) + e * sfb_stride,
            stream
        );

        TVM_FFI_ICHECK(err == cudaSuccess)
            << "Batched GEMV FP4 kernel failed at expert " << e 
            << ": " << cudaGetErrorString(err);
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemv_fp4_blockscaled, gemv_fp4_blockscaled);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batched_gemv_fp4, batched_gemv_fp4);
