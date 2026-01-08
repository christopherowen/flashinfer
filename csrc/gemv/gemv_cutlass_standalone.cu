/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Standalone CUTLASS GemvBlockScaled test to work out integration issues.
 * This file is compiled separately to isolate namespace issues.
 */

// IMPORTANT: Include cute/tensor.hpp FIRST to avoid namespace conflicts
#include "cute/tensor.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <cstdio>

// CUTLASS headers (after cute)
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemv_blockscaled.h"
#include "cutlass/gemm/kernel/gemv_blockscaled.h"
#include "cutlass/epilogue/threadblock/epilogue_with_scaling_factor.h"
#include "cutlass/util/device_memory.h"

namespace flashinfer {
namespace gemv {

//==============================================================================
// Type definitions matching CUTLASS GemvBlockScaled
//==============================================================================

using ElementA = cutlass::float_e2m1_t;           // FP4 activation
using ElementB = cutlass::float_e2m1_t;           // FP4 weight
using ElementC = cutlass::float_e2m1_t;           // FP4 (unused for alpha=1, beta=0)
using ElementD = cutlass::float_e2m1_t;           // FP4 output (with scale factor)
using ElementSFA = cutlass::float_e4m3_t;         // FP8 activation scale
using ElementSFB = cutlass::float_e4m3_t;         // FP8 weight scale  
using ElementSFD = cutlass::float_e4m3_t;         // FP8 output scale
using ElementAccumulatorMainloop = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using LayoutSFD = cutlass::layout::ColumnMajor;

static constexpr int kVectorSize = 16;
static constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementA>::value;  // 32

using ThreadShape = cutlass::gemm::GemmShape<16, 8>;

// Epilogue type (FP4 output with scale factor)
using EpilogueOp = cutlass::epilogue::threadblock::GemvEpilogueWithScalingFactor<
    kVectorSize,
    ThreadShape,
    ElementCompute,
    ElementAccumulator,
    ElementC,
    ElementD,
    ElementSFD,
    LayoutD,
    LayoutSFD>;

// GEMV kernel type
using GemvKernel = cutlass::gemm::kernel::GemvBlockScaled<
    ElementA, LayoutA, ElementB, ElementD, ElementAccumulatorMainloop, EpilogueOp, kElementsPerAccess>;

// Device-level GEMV
using Gemv = cutlass::gemm::device::GemvBlockScaled<GemvKernel>;

//==============================================================================
// Test function
//==============================================================================

extern "C" __attribute__((visibility("default")))
int test_cutlass_gemv(
    int M,  // Number of rows (output dimension)
    int K,  // Number of columns (reduction dimension)
    int batch_count,
    const void* ptr_A,      // [M, K] packed FP4
    const void* ptr_SFA,    // [M/128, K/32] FP8 scales
    const void* ptr_B,      // [K] packed FP4
    const void* ptr_SFB,    // [K/32] FP8 scales
    void* ptr_D,            // [M] packed FP4 output
    void* ptr_SFD,          // [M/16] FP8 output scales
    float alpha,
    float beta,
    float epilogue_st,
    cudaStream_t stream)
{
    printf("test_cutlass_gemv called: M=%d, K=%d, batch=%d\n", M, K, batch_count);

    // Calculate strides
    int64_t stride_A = K;
    int64_t batch_stride_A = M * K;
    int64_t batch_stride_B = K;
    int64_t batch_stride_C = M;
    int64_t batch_stride_D = M;
    
    // Scale factor strides (simplified)
    int k_blocks = (K + 31) / 32;  // K elements per block = 32
    int m_blocks = (M + 127) / 128;  // M elements per block = 128
    int64_t batch_stride_SFA = m_blocks * 128 * k_blocks * 4;  // Blk_SF = 4
    int64_t batch_stride_SFB = k_blocks * 4;
    int64_t batch_stride_SFD = (M + 15) / 16;  // 16 elements per SFD block

    // Create TensorRef for A (note: CUTLASS expects non-const)
    cutlass::TensorRef<ElementA, LayoutA> ref_A(
        const_cast<ElementA*>(reinterpret_cast<const ElementA*>(ptr_A)),
        LayoutA::packed({M, K})
    );

    // Construct arguments
    typename Gemv::Arguments arguments{
        cutlass::MatrixCoord{M, K},          // problem_size
        batch_count,                          // batch_count
        typename EpilogueOp::Params{
            cutlass::TensorRef<ElementD, LayoutD>(
                reinterpret_cast<ElementD*>(ptr_D),
                LayoutD::packed({M, 1})
            ),                                // tensor_d
            reinterpret_cast<ElementSFD*>(ptr_SFD),  // scale_factor_d_ptr
            alpha,                            // alpha
            beta,                             // beta
            epilogue_st,                      // st
            batch_stride_SFD,                 // batch_stride_sfd
            static_cast<int64_t>(M)           // stride_d
        },
        ref_A,                                // ref_A
        reinterpret_cast<const ElementB*>(ptr_B),   // ptr_B
        reinterpret_cast<const ElementC*>(ptr_D),   // ptr_C (reuse D for beta=0)
        reinterpret_cast<ElementD*>(ptr_D),         // ptr_D
        reinterpret_cast<const ElementSFA*>(ptr_SFA),  // ptr_SFA
        reinterpret_cast<const ElementSFB*>(ptr_SFB),  // ptr_SFB
        stride_A,                             // stride_A
        batch_stride_A,                       // batch_stride_A
        batch_stride_B,                       // batch_stride_B
        batch_stride_C,                       // batch_stride_C
        batch_stride_D,                       // batch_stride_D
        batch_stride_SFA,                     // batch_stride_SFA
        batch_stride_SFB,                     // batch_stride_SFB
        batch_stride_SFD                      // batch_stride_SFD
    };

    // Create operator
    Gemv gemv_op;

    // Check if arguments are valid
    cutlass::Status status = gemv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        printf("can_implement() failed: %d\n", static_cast<int>(status));
        return -1;
    }

    // Get workspace size
    size_t workspace_size = Gemv::get_workspace_size(arguments);
    printf("workspace_size: %zu\n", workspace_size);

    // Allocate workspace
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Initialize
    status = gemv_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        printf("initialize() failed: %d\n", static_cast<int>(status));
        return -2;
    }

    // Run
    status = gemv_op(stream);
    if (status != cutlass::Status::kSuccess) {
        printf("operator() failed: %d\n", static_cast<int>(status));
        return -3;
    }

    printf("GEMV succeeded!\n");
    return 0;
}

//==============================================================================
// BF16 output version - converts FP4*scale to BF16
//==============================================================================

// Simple kernel to convert FP4 output with scale factor to BF16
__global__ void convert_fp4_sfd_to_bf16(
    const uint8_t* __restrict__ fp4_output,      // [M/2] packed FP4
    const uint8_t* __restrict__ sfd,             // [M/16] FP8 scale factors
    nv_bfloat16* __restrict__ bf16_output,       // [M] BF16
    int M)
{
    // Each thread handles 2 FP4 values (1 byte)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int byte_idx = tid;
    
    if (byte_idx >= M / 2) return;
    
    // Get the packed FP4 byte
    uint8_t packed = fp4_output[byte_idx];
    
    // Extract nibbles
    uint8_t nibble_lo = packed & 0x0F;
    uint8_t nibble_hi = packed >> 4;
    
    // FP4 E2M1 lookup table (same as llama.cpp)
    // Values are: 0, 0.5, 1, 1.5, 2, 3, 4, 6 for positive
    //            -0, -0.5, -1, -1.5, -2, -3, -4, -6 for negative
    static constexpr float fp4_lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    
    // Get scale factor index (16 elements per scale block)
    int sfd_idx = (byte_idx * 2) / 16;  // 2 elements per byte, 16 elements per scale
    
    // Decode FP8 E4M3 scale factor
    // Simplified: interpret as unsigned and scale
    uint8_t sfd_byte = sfd[sfd_idx];
    float scale = ldexpf(1.0f, (int)(sfd_byte >> 3) - 7);  // Simplified FP8 decode
    
    // Convert and write
    float val_lo = fp4_lut[nibble_lo] * scale;
    float val_hi = fp4_lut[nibble_hi] * scale;
    
    bf16_output[byte_idx * 2 + 0] = __float2bfloat16(val_lo);
    bf16_output[byte_idx * 2 + 1] = __float2bfloat16(val_hi);
}

extern "C" __attribute__((visibility("default")))
int gemv_fp4_to_bf16(
    int M,  // Number of rows (output dimension)
    int K,  // Number of columns (reduction dimension)
    int batch_count,
    const void* ptr_A,      // [M, K] packed FP4
    const void* ptr_SFA,    // FP8 scales
    const void* ptr_B,      // [K] packed FP4
    const void* ptr_SFB,    // FP8 scales
    void* ptr_bf16_out,     // [M] BF16 output
    float alpha,
    cudaStream_t stream)
{
    // Allocate temporary FP4 output and scale factor buffers
    void* ptr_D_temp;
    void* ptr_SFD_temp;
    
    size_t d_size = M / 2;  // FP4 packed
    size_t sfd_size = (M + 15) / 16;
    
    cudaMalloc(&ptr_D_temp, d_size);
    cudaMalloc(&ptr_SFD_temp, sfd_size);
    
    // Run GEMV
    float epilogue_st = 1.0f;
    int result = test_cutlass_gemv(
        M, K, batch_count,
        ptr_A, ptr_SFA, ptr_B, ptr_SFB,
        ptr_D_temp, ptr_SFD_temp,
        alpha, 0.0f, epilogue_st,
        stream);
    
    if (result != 0) {
        cudaFree(ptr_D_temp);
        cudaFree(ptr_SFD_temp);
        return result;
    }
    
    // Convert FP4+SFD to BF16
    int threads = 256;
    int blocks = (M / 2 + threads - 1) / threads;
    convert_fp4_sfd_to_bf16<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ptr_D_temp),
        reinterpret_cast<const uint8_t*>(ptr_SFD_temp),
        reinterpret_cast<nv_bfloat16*>(ptr_bf16_out),
        M);
    
    cudaFree(ptr_D_temp);
    cudaFree(ptr_SFD_temp);
    
    return 0;
}

}  // namespace gemv
}  // namespace flashinfer

