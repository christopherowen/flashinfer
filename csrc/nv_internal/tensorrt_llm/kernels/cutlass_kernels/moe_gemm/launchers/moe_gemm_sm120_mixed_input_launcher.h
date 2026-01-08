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

#pragma once

#include <cuda_runtime_api.h>

#include "../../include/moe_gemm_kernels.h"
#include "cutlass_extensions/gemm_configs.h"
#include "cutlass_extensions/weight_only_quant_op.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels_oss {

using tensorrt_llm::kernels::cutlass_kernels::GroupedGemmInput;
using tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;

// SM120 Mixed-Input Grouped GEMM Launcher
// This launcher supports mixed-precision operations on SM120/SM121 (Blackwell) architecture:
// - FP8 activations x FP4 weights (FP8xFP4)
// - BF16/FP16 activations x FP4 weights (MXFP4/W4A16) - uses FP8xFP4 path with quantized activations
//
// Key features:
// - Uses block-scaled collective builder for SM120/SM121
// - Supports grouped GEMM for MoE workloads
// - Integrates with CUTLASS SM120 block-scaled infrastructure
//
// IMPORTANT: A-side scaling factors are mandatory for SM120 block-scaled kernels.
// The SM120 mma.mx_f4f6f8 instructions only support FP8/FP6/FP4 inputs, so BF16/FP16
// activations must be pre-quantized to FP8 before calling this launcher.
//
// For MXFP4 (W4A16) workloads where you want to preserve activation accuracy:
//   - Pre-quantize BF16/FP16 -> FP8 (float_e4m3_t)
//   - Use IDENTITY A-SCALES to avoid accuracy loss from block scaling
//
// Identity Scale Implementation:
//   - Scale factor type: float_ue8m0_t (unsigned 8-bit exponent, uses FP32 bias=127)
//   - Identity scale value: raw byte = 0x7F (127) → 2^(127-127) = 2^0 = 1.0
//   - Layout: MUST use proper CUTLASS LayoutSFA, NOT stride=0 broadcast
//
// IMPORTANT: TMA scale loads require real tiles with valid access patterns.
// Do NOT use stride=0 to broadcast a single scale value - this may fail or
// force slow paths on some driver versions. Instead:
//   1. Allocate SFA buffer matching CUTLASS's LayoutSFA for (M, K)
//   2. Fill entire buffer with 0x7F using cudaMemset
//   3. Cache buffer per (M, K, tile_shape) to avoid re-allocation
//   4. Pass buffer with correct strides from LayoutSFA
//
// Example identity scale setup:
//   #include "sm12x_activation_quantizer.cuh"
//   Sm12xIdentityScaleBufferManager& mgr = getIdentityScaleBufferManager();
//   uint8_t* sfa = mgr.getOrCreate(M, K, /*sf_vec_size=*/32, stream);
//   int sfa_stride = mgr.getScaleStride(M, K);
//
// WARNING: CUTLASS header says "exp_bias: 8" but convert_to_float uses FP32's
// bias of 127. Always use 0x7F for identity, NOT 8!
//

// =============================================================================
// Identity SFA Buffer Acquisition API
// =============================================================================
//
// Callers MUST use these functions to acquire identity SFA buffers before
// calling the launcher. The launcher does NOT own SFA allocation.
//
// GROUPED GEMM L SEMANTICS:
// For grouped GEMM with PtrArray, each problem has its own (M_i, N, K) shape.
// The TMA descriptor is set up per-problem, so L=1 for each problem's SFA layout.
// The "grouping" is via pointer arrays, NOT via an L dimension in the SFA tensor.
//
// Therefore: computeSm120IdentitySFABufferSize(M_max, N, K, L=1) gives the buffer
// size needed for the LARGEST single problem. All problems can share this buffer.
//
// POINTER ARRAY MEMORY SPACE:
// CUTLASS grouped GEMM expects ElementSF const** (device pointer to device pointers).
// The pointer array MUST be:
//   - Allocated on device (cudaMalloc)
//   - Lifetime outlives gemm.run()
//   - Contains num_groups pointers (all pointing to the same identity buffer is OK)
//
// Usage pattern (in the dispatch code that prepares hopper_inputs):
//
//   // Step 1: Compute size for largest problem (L=1 for grouped GEMM)
//   size_t sfa_bytes = computeSm120IdentitySFABufferSize(M_max, N, K, /*L=*/1);
//
//   // Step 2: Acquire pre-filled identity buffer (no hot-path alloc/memset)
//   uint8_t* identity_sfa = acquireSm120IdentitySFABuffer(sfa_bytes);
//
//   // Step 3: Build DEVICE-SIDE per-group pointer array
//   // All pointers point to the same identity buffer (safe because all 0x7F)
//   std::vector<uint8_t const*> h_sfa_ptrs(num_groups, identity_sfa);
//   uint8_t const** d_sfa_ptrs;
//   cudaMalloc(&d_sfa_ptrs, num_groups * sizeof(uint8_t*));
//   cudaMemcpyAsync(d_sfa_ptrs, h_sfa_ptrs.data(), num_groups * sizeof(uint8_t*),
//                   cudaMemcpyHostToDevice, stream);
//   hopper_inputs.fpX_block_scaling_factors_act = d_sfa_ptrs;
//
//   // Step 4: Call launcher
//   sm120_mixed_input_moe_gemm_kernelLauncher(...);
//
//   // NOTE: d_sfa_ptrs lifetime must outlive gemm.run()!
//   // Consider caching/pooling the pointer array.
//

// Compute required SFA buffer size for identity scales
// This is kernel-derived: uses the same Sm1xxBlockScaledConfig as the kernel
// M_max: Maximum M across all groups (use largest problem size)
// N, K: Problem dimensions
// L: Should be 1 for grouped GEMM (grouping is via pointer arrays, not L dimension)
size_t computeSm120IdentitySFABufferSize(int64_t M_max, int64_t N, int64_t K, int64_t L = 1);

// Acquire identity SFA buffer of the given size
// Uses Sm12xIdentityScaleBufferManager::getOrCreateWithSize internally
// Returns pre-filled buffer (all 0x7F), cached per (device, size)
// No allocation or memset in steady-state (after first call for each size)
uint8_t* acquireSm120IdentitySFABuffer(size_t required_bytes);

// Prewarm identity SFA buffers for common MoE shapes
// Call at engine initialization to avoid first-call allocation latency
void prewarmSm120IdentitySFABuffers(const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>& shapes);

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          typename CTAShape, typename ClusterShape, bool IsMXFP4 = false>
void sm120_mixed_input_moe_gemm_kernelLauncher(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);

}  // namespace cutlass_kernels_oss
}  // namespace kernels
}  // namespace tensorrt_llm

