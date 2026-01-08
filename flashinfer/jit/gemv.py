"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

JIT compilation module for FP4 GEMV kernels.
"""

from typing import List

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    current_compilation_context,
)


def gen_gemv_fp4_sm120_module() -> JitSpec:
    """
    Generate a JitSpec for the FP4 GEMV kernel module for SM120 (Blackwell).
    
    This module provides:
    - gemv_fp4_blockscaled: Single GEMV with FP4 block scaling
    - batched_gemv_fp4: Batched GEMV for multiple experts
    """
    nvcc_flags = [
        "-DCOMPILE_BLACKWELL_TMA_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
    ]

    nvcc_flags += current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )

    return gen_jit_spec(
        "gemv_fp4_120",
        [
            jit_env.FLASHINFER_CSRC_DIR / "gemv/gemv_fp4_blockscaled.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
        extra_cflags=[],
        extra_ldflags=[],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "cutlass_extensions"
            / "include",
            jit_env.FLASHINFER_DATA / "cutlass" / "include",
            jit_env.FLASHINFER_DATA / "cutlass" / "tools" / "util" / "include",
        ],
    )


# Cached module instance
_gemv_module = None


def get_gemv_module():
    """
    Get the compiled GEMV module, compiling it if necessary.
    
    Returns
    -------
    module
        The compiled GEMV module with functions:
        - gemv_fp4_blockscaled(m, k, alpha, beta, A, B, C, D, SFA, SFB)
        - batched_gemv_fp4(num_experts, m, k, alpha, A, B, D, SFA, SFB)
    """
    global _gemv_module
    
    if _gemv_module is None:
        spec = gen_gemv_fp4_sm120_module()
        _gemv_module = spec.build_and_load()
    
    return _gemv_module

