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

#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"

namespace tensorrt_llm::kernels::cutlass_kernels_oss {

using tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;

template <typename T, typename WeightType, typename OutputType, typename EpilogueTag,
          typename TileShape, typename ClusterShape, bool IsMXFP4>
void sm120_mixed_input_moe_gemm_kernelLauncher(
    TmaWarpSpecializedGroupedGemmInput tma_inputs, int num_experts,
    int multi_processor_count, cudaStream_t stream, int* occupancy,
    size_t* workspace_size);

}  // namespace tensorrt_llm::kernels::cutlass_kernels_oss
