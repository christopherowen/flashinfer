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
 * Uses CUTLASS GemvBlockScaled kernel with:
 * - ElementA: float_e2m1_t (FP4 e2m1, activations, quantized from BF16)
 * - ElementB: float_e2m1_t (FP4 e2m1, weights)
 * - ElementC/D: bfloat16_t (output)
 * - Block scaling with SFVecSize=16
 *
 * CURRENT STATUS:
 * - Phase 1: Software dequant fallback (IMPLEMENTED, ~7 tok/s)
 * - Phase 2: CUTLASS GemvBlockScaled native (TODO, target ~58 tok/s)
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

// For CUTLASS GemvBlockScaled (Phase 2)
// #include "cutlass/gemm/device/gemv_blockscaled.h"
// #include "cutlass/gemm/kernel/gemv_blockscaled.h"
// #include "cutlass/epilogue/threadblock/epilogue_with_scaling_factor.h"
// #include "cutlass/detail/sm100_blockscaled_layout.hpp"

// TVM FFI bindings
#include "../tvm_ffi_utils.h"

using tvm::ffi::TensorView;

namespace flashinfer {
namespace gemv {

// Configuration constants
static constexpr int kBlockSize = 32;  // MXFP4 block size for scale factors

//==============================================================================
// Software Dequantization Fallback (Phase 1)
//==============================================================================

/*!
 * \brief Simple FP4 dequantization for GEMV fallback
 *
 * This is a simple implementation that dequantizes FP4 weights and performs
 * a standard GEMV in BF16. It's not optimal but serves as a baseline and
 * fallback path.
 *
 * For production, we should integrate the actual CUTLASS GemvBlockScaled kernel.
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
        // FP4 e2m1 has values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
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
        
        // Scale factor indexing (simplified - FP8 e4m3 stored as uint8)
        // Convert FP8 to float via __nv_cvt_fp8_to_fp32
        uint8_t a_scale_bits = A_scales[m_block * num_k_blocks + k_block];
        uint8_t b_scale_bits = B_scales[n_block * num_k_blocks + k_block];
        
        // Simple FP8 e4m3 to float conversion
        auto fp8_to_float = [](uint8_t bits) -> float {
            __nv_fp8_e4m3 fp8;
            memcpy(&fp8, &bits, 1);
            return float(fp8);
        };
        
        float a_scale = fp8_to_float(a_scale_bits);
        float b_scale = fp8_to_float(b_scale_bits);
        
        // Dequantize and multiply
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

/*!
 * \brief Run FP4 GEMV with dequantization fallback
 */
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
// CUTLASS GemvBlockScaled Native (Phase 2) - TODO
//==============================================================================

/*
 * Phase 2 implementation will use CUTLASS GemvBlockScaled directly:
 *
 * using ElementA = cutlass::float_e2m1_t;  // FP4
 * using ElementB = cutlass::float_e2m1_t;  // FP4
 * using ElementC = cutlass::bfloat16_t;
 * using ElementD = cutlass::bfloat16_t;
 * using ElementAccumulator = float;
 * using ElementSFA = cutlass::float_e4m3_t;  // FP8 scale
 * using ElementSFB = cutlass::float_e4m3_t;  // FP8 scale
 * 
 * static constexpr int kElementsPerAccess = 32;  // Fixed for FP4
 * static constexpr int kThreadCount = 128;
 * static constexpr int kThreadsPerRow = 16;
 * static constexpr int kSFVecSize = 16;
 *
 * This requires:
 * 1. Custom epilogue for BF16 output with scale factor
 * 2. Proper scale factor layout matching MXFP4 format
 * 3. Integration with FlashInfer's activation quantization
 */

}  // namespace gemv
}  // namespace flashinfer

//==============================================================================
// TVM FFI Bindings
//==============================================================================

/*!
 * \brief TVM FFI function for FP4 GEMV
 *
 * D = alpha * A @ B + beta * D
 * A: [M, K/2] packed FP4 (uint8)
 * B: [K/2, N] packed FP4 (uint8)
 * D: [M, N] bfloat16
 * SFA: activation scale factors (FP8 as uint8)
 * SFB: weight scale factors (FP8 as uint8)
 */
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
    cudaStream_t stream = nullptr;  // Default stream

    cudaError_t err = flashinfer::gemv::run_gemv_fp4_dequant(
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

/*!
 * \brief TVM FFI function for batched FP4 GEMV
 */
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

    // For now, loop over experts (can be optimized with batched kernel later)
    size_t a_stride = m * (k / 2);
    size_t b_stride = (k / 2) * n;
    size_t d_stride = m * n * sizeof(__nv_bfloat16);
    size_t sfa_stride = ((m + 31) / 32) * ((k + 31) / 32);
    size_t sfb_stride = ((n + 31) / 32) * ((k + 31) / 32);
    
    for (int64_t e = 0; e < num_experts; e++) {
        cudaError_t err = flashinfer::gemv::run_gemv_fp4_dequant(
            static_cast<int>(m),
            static_cast<int>(k),
            static_cast<int>(n),
            static_cast<float>(alpha),
            0.0f,  // beta = 0 for batched
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

// Export functions
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemv_fp4_blockscaled, gemv_fp4_blockscaled);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batched_gemv_fp4, batched_gemv_fp4);
