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
"""

import os
from typing import List

from . import env as jit_env
from ..artifacts import ArtifactPath, CheckSumHash
from .core import (
    JitSpec,
    gen_jit_spec,
    current_compilation_context,
    sm90a_nvcc_flags,
    sm89_nvcc_flags,
)
from .cpp_ext import is_cuda_version_at_least
from .cubin_loader import get_cubin, get_meta_hash
from .gemm.cutlass.generate_kernels import generate_gemm_operations


_FUSED_MOE_BUILD_PROFILE_ENV = "FLASHINFER_FUSED_MOE_BUILD_PROFILE"


def _get_fused_moe_build_profile() -> str:
    """Return fused-MoE JIT build profile.

    This is modeled after other JIT generators (e.g. `jit/gemm/core.py`) where
    we compile only the dtypes/kernels needed for the current experiment.

    Supported values:
    - "full" (default): build all fused-MoE kernels (many dtypes/paths).
    - "mxfp4_minimal": build only the MXFP4 (FP8×FP4) path needed for SM120/121.
    """
    return os.getenv(_FUSED_MOE_BUILD_PROFILE_ENV, "full").strip().lower()


def gen_cutlass_fused_moe_sm120_module(
    use_fast_build: bool = False,
    tile_mn: tuple[int, int] = (128, 128),
) -> JitSpec:
    """Generate SM120 MoE module with configurable logical GEMM tile (M,N).

    Args:
        use_fast_build: Enable fast build mode (reduced optimizations).
        tile_mn: Logical (M,N) tile for the MoE GEMM (K is fixed by kernel family).
            This is *logical* in the sense of D = A @ W (M=tokens, N=output feature dim).
            Some logical tiles may be implemented via internal swap/transpose tricks.

    Returns:
        JitSpec for the configured MoE module.
    """
    logical_m, logical_n = tile_mn

    # swap_ab (transposed mode) for logical M < 64:
    # The tcgen05 hardware requires physical M >= 64. For logical M < 64 (decode batches),
    # we use swap_ab to transpose the problem: physical (N, M) -> logical (M, N).
    # For logical M >= 64, we use the native (non-swapped) path.
    swap_ab = logical_m < 64

    nvcc_flags = [
        "-DCOMPILE_BLACKWELL_TMA_GEMMS",
        "-DCOMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
        f"-DLOGICAL_TILE_M={logical_m}",
        f"-DLOGICAL_TILE_N={logical_n}",
        f"-DSWAP_AB={1 if swap_ab else 0}",
    ]

    nvcc_flags += current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )

    # Include logical tile in module name for separate caching
    module_suffix = ""
    if (logical_m, logical_n) != (128, 128):
        module_suffix = f"_M{logical_m}N{logical_n}"

    # Optional: reduce compilation surface area for SM120/121 MXFP4 iteration.
    # Default to minimal for SM120/121 unless explicitly overridden.
    env_val = os.getenv(_FUSED_MOE_BUILD_PROFILE_ENV)
    build_profile = _get_fused_moe_build_profile() if env_val is not None else "mxfp4_minimal"

    if build_profile == "mxfp4_minimal":
        nvcc_flags += ["-DFLASHINFER_FUSED_MOE_MXFP4_MINIMAL"]
        module_suffix += "_mxfp4min"

    return gen_cutlass_fused_moe_module(
        nvcc_flags, f"120{module_suffix}", use_fast_build
    )


def gen_cutlass_fused_moe_sm103_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = [
        "-DCOMPILE_BLACKWELL_TMA_GEMMS",
        "-DCOMPILE_BLACKWELL_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
        "-DCOMPILE_BLACKWELL_SM103_TMA_GROUPED_GEMMS",
    ]

    nvcc_flags += current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )

    return gen_cutlass_fused_moe_module(nvcc_flags, "103", use_fast_build)


def gen_cutlass_fused_moe_sm100_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = [
        "-DCOMPILE_BLACKWELL_TMA_GEMMS",
        "-DCOMPILE_BLACKWELL_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
    ]

    nvcc_flags += current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11]
    )

    return gen_cutlass_fused_moe_module(nvcc_flags, "100", use_fast_build)


def gen_cutlass_fused_moe_sm90_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = sm90a_nvcc_flags + [
        "-DCOMPILE_HOPPER_TMA_GEMMS",
        "-DCOMPILE_HOPPER_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP8_BLOCK_SCALE" if is_cuda_version_at_least("12.8") else "",
        "-DENABLE_FP4" if is_cuda_version_at_least("12.8") else "",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
    ]
    return gen_cutlass_fused_moe_module(nvcc_flags, "90", use_fast_build)


def gen_cutlass_fused_moe_sm89_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = sm89_nvcc_flags + [
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP8_BLOCK_SCALE" if is_cuda_version_at_least("12.8") else "",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
    ]
    return gen_cutlass_fused_moe_module(nvcc_flags, "89", use_fast_build)


def gen_cutlass_fused_moe_module(
    nvcc_flags: List[str], device_arch: str, use_fast_build: bool = False
) -> JitSpec:
    """
    Generate a JitSpec for the cutlass fused moe module.
    """
    build_profile = _get_fused_moe_build_profile()
    output_dir = (
        jit_env.FLASHINFER_CSRC_DIR
        / f"nv_internal/tensorrt_llm/cutlass_instantiations/{device_arch}"
    )

    # NOTE: The "full" fused-MoE build compiles a large matrix of dtypes/kernels.
    # For SM120/121 MXFP4 tile experiments (e.g. TILE_M=32), compiling unrelated
    # translation units can fail and/or significantly slow iteration.
    #
    # In "mxfp4_minimal" we:
    # - skip CUTLASS kernel generation (not used by the SM120 fixed-kernel launcher)
    # - compile only the MXFP4 (FP8×FP4) MoE runner + required plumbing
    output_dir.mkdir(parents=True, exist_ok=True)
    # SM120/121 fused-MoE uses fixed MXFP4 kernels (see SM120 launcher) and does
    # not rely on the large generated CUTLASS kernel matrix. Avoid generating
    # and compiling those translation units to keep JIT iteration fast/stable.
    is_sm120_family = device_arch.startswith("120") or device_arch.startswith("121")

    generated_sources: List[object] = []
    if (not is_sm120_family) and build_profile != "mxfp4_minimal":
        try:
            generate_gemm_operations(
                output_dir,
                f"{device_arch};{device_arch}-real",
            )
            generated_sources = [output_dir / k for k in output_dir.rglob("*.generated.cu")]
        except Exception as e:
            raise RuntimeError(f"Failed to generate Cutlass kernels: {e}") from e

    # Base sources needed for SM120/121 MXFP4 path and fused-MoE binding.
    # Keep these shared across profiles unless explicitly trimmed.
    base_sources: List[object] = [
        jit_env.FLASHINFER_CSRC_DIR
        / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu",
        # FP8×FP4 MoE runner instantiation (covers MXFP4 MoE entrypoint)
        jit_env.FLASHINFER_CSRC_DIR
        / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp4.cu",
        jit_env.FLASHINFER_CSRC_DIR
        / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu",
        jit_env.FLASHINFER_CSRC_DIR
        / "fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu",
        # Provides module method(s) like set_deepgemm_jit_include_dirs used by Python glue.
        jit_env.FLASHINFER_CSRC_DIR
        / "fused_moe/cutlass_backend/deepgemm_jit_setup.cu",
        # NOTE: keep this in full builds; in minimal build it is compiled with
        # FLASHINFER_FUSED_MOE_MXFP4_MINIMAL to avoid instantiating unused dtypes.
        jit_env.FLASHINFER_CSRC_DIR
        / "fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
        jit_env.FLASHINFER_CSRC_DIR
        / "nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu",
    ]

    if is_sm120_family:
        # Always keep SM120/121 source list minimal: only the MXFP4 path.
        sources = base_sources
    elif build_profile == "mxfp4_minimal":
        sources = base_sources + generated_sources
    else:
        sources = [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp4_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp32_fp32.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp16.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_bf16.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/deepgemm_jit_setup.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
            # Add all generated kernels
            *generated_sources,
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/lora/lora.cpp",
        ]

    return gen_jit_spec(
        f"fused_moe_{device_arch}",
        sources,
        extra_cuda_cflags=nvcc_flags,
        extra_cflags=["-DFAST_BUILD"] if use_fast_build else [],
        extra_ldflags=["-lnvrtc"],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "cutlass_extensions"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels",
        ],
    )


def gen_trtllm_gen_fused_moe_sm100_module() -> JitSpec:
    # Fetch "flashinferMetaInfo.h" from the online kernel cache. This file
    # contains the `tllmGenBatchedGemmList` as the list of available kernels
    # online. It is included when compiling `trtllm_fused_moe_runner.cu`, etc.
    include_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/include"
    header_name = "flashinferMetaInfo"

    # Check if checksums.txt exists in the cubin directory
    checksum_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/checksums.txt"
    checksum = get_cubin(checksum_path, CheckSumHash.TRTLLM_GEN_BMM)
    assert checksum, f"Failed to get checksums.txt from {checksum_path}"
    meta_hash = get_meta_hash(checksum)

    # use `get_cubin` to get "flashinferMetaInfo.h"
    metainfo = get_cubin(
        f"{include_path}/{header_name}.h",
        meta_hash,
    )
    # make sure "flashinferMetaInfo.h" is downloaded or cached
    assert metainfo, f"{header_name}.h not found"

    # currently only support Blackwell
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )

    return gen_jit_spec(
        "fused_moe_trtllm_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_kernel_launcher.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_runner.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_deepseek.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_llama4.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_renormalize.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_dev_kernel.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_batched_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_GEN_EXPORT_FLASHINFER",
            "-DTLLM_ENABLE_CUDA",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
            f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{ArtifactPath.TRTLLM_GEN_BMM}\\"',
        ]
        + nvcc_flags,
        extra_include_paths=[
            # link "include" sub-directory in cache
            jit_env.FLASHINFER_CUBIN_DIR / include_path,
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/include",
        ],
    )
