/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "moe_gemm_template_dispatch.h"

// ============================================================================
// DEBUG: Verify ENABLE_FP4 is defined in this translation unit
// This file should ALWAYS have ENABLE_FP4 defined, otherwise FP4 kernels won't compile
// ============================================================================
#ifndef ENABLE_FP4
#error "ENABLE_FP4 is not defined in moe_gemm_kernels_fp4_fp4.cu - FP4 kernels will not be compiled!"
#endif

// DEBUG: Verify SM12x support is enabled (if targeting SM120/SM121)
#ifdef DEBUG_SM120_K_CONV
#ifndef CUTLASS_ARCH_MMA_SM12x_SUPPORTED
#warning "CUTLASS_ARCH_MMA_SM12x_SUPPORTED is not defined - SM120/SM121 kernels will not be available"
#endif
#ifndef COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
#warning "COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS is not defined - SM120 grouped GEMM will throw at runtime"
#endif
#endif

namespace tensorrt_llm::kernels::cutlass_kernels {
#ifdef ENABLE_FP4
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half>;
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;
#endif
#endif
}  // namespace tensorrt_llm::kernels::cutlass_kernels
