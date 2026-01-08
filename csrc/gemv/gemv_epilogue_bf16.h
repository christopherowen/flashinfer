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

/*! \file gemv_epilogue_bf16.h
 *  \brief Custom epilogue for GEMV with BF16 output.
 *
 *  This epilogue is designed for MoE decode where:
 *  - Input: FP4 activations and weights
 *  - Accumulator: FP32
 *  - Output: BF16
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

namespace flashinfer {
namespace epilogue {

/*!
 * \brief Simple GEMV epilogue that outputs BF16.
 *
 * This epilogue performs:
 *   D = alpha * accumulator + beta * C
 *
 * where accumulator is FP32, C is BF16, D is BF16.
 *
 * Template parameters match what GemvBlockScaled kernel expects.
 */
template <
    int kVectorSize_,
    typename ThreadShape_,
    typename ElementCompute_,
    typename ElementAccumulator_,
    typename ElementC_,
    typename ElementD_,
    typename LayoutOutput_
>
class GemvEpilogueBF16 {
public:
    using ThreadShape = ThreadShape_;
    using ElementCompute = ElementCompute_;      // float
    using ElementAccumulator = ElementAccumulator_;  // float
    using ElementC = ElementC_;                  // bfloat16_t
    using ElementD = ElementD_;                  // bfloat16_t
    using LayoutOutput = LayoutOutput_;
    using TensorRefD = cutlass::TensorRef<ElementD, LayoutOutput_>;

    static constexpr int kVectorSize = kVectorSize_;
    static constexpr int kThreadsPerCol = ThreadShape::kM;  // 16
    static constexpr int kThreadsPerRow = ThreadShape::kN;  // 8
    static constexpr int kThreadCount = kThreadsPerCol * kThreadsPerRow;  // 128

    // Static checks
    static_assert(kVectorSize == kThreadsPerCol, "vector size and threads per col should match");
    static_assert(std::is_same_v<ElementCompute, float>, "ElementCompute must be float");
    static_assert(std::is_same_v<ElementD, cutlass::bfloat16_t>, "ElementD must be bfloat16");
    static_assert(std::is_same_v<LayoutOutput, cutlass::layout::ColumnMajor>,
                  "Only column-major output supported");

    struct Params {
        TensorRefD tensor_d;
        ElementCompute alpha{1.0f};
        ElementCompute beta{0.0f};
        int64_t batch_stride_d{0};
        int64_t stride_d{0};
    };

    /// Shared storage (minimal for simple epilogue)
    struct SharedStorage {
        // No shared storage needed for simple linear combination
    };

private:
    Params const& params_;
    SharedStorage& shared_storage_;

public:
    CUTLASS_HOST_DEVICE
    GemvEpilogueBF16(Params const& params, SharedStorage& shared_storage)
        : params_(params), shared_storage_(shared_storage) {}

    /*!
     * \brief Apply epilogue operation.
     *
     * \param frag_acc Accumulated value (FP32)
     * \param frag_c Source value for beta term (BF16)
     * \param batch_idx Batch index for batched GEMV
     */
    CUTLASS_DEVICE
    void operator()(ElementAccumulator frag_acc, ElementC frag_c, int batch_idx) {
        const int block_idx = blockIdx.x;
        const int thread_idx_col = threadIdx.x;  // Which row within the block
        const int thread_idx_row = threadIdx.y;  // Which thread in K reduction

        // Only one thread per row writes output (after reduction)
        if (thread_idx_row != 0) return;

        // Compute output index
        int output_idx = block_idx * kThreadsPerCol + thread_idx_col;

        // Compute D pointer with batch offset
        ElementD* ptr_D = params_.tensor_d.data() + batch_idx * params_.batch_stride_d;

        // Linear combination: D = alpha * acc + beta * C
        ElementCompute result = params_.alpha * frag_acc;

        if (params_.beta != 0) {
            // Convert BF16 to float, apply beta, add
            cutlass::NumericConverter<ElementCompute, ElementC> c_to_compute;
            result += params_.beta * c_to_compute(frag_c);
        }

        // Convert result to BF16 and store
        cutlass::NumericConverter<ElementD, ElementCompute> compute_to_d;
        ptr_D[output_idx] = compute_to_d(result);
    }
};

}  // namespace epilogue
}  // namespace flashinfer

