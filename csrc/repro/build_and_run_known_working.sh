#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

SRC="${HERE}/repro_sm12x_group_gemm_fp8_groupwise_sm120.cu"
OUT="${HERE}/repro_sm12x_group_gemm_fp8_groupwise_sm120"

CUDA_ARCH="${CUDA_ARCH:-sm_121}"

echo "[build] ROOT=${ROOT}"
echo "[build] CUDA_ARCH=${CUDA_ARCH}"

nvcc -std=c++17 -O2 -lineinfo \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -arch="${CUDA_ARCH}" \
  -DCUTLASS_ARCH_MMA_SM121_SUPPORTED -DCUTLASS_ARCH_MMA_SM12x_SUPPORTED \
  -I"${ROOT}/include" \
  -I"${ROOT}/3rdparty/cutlass/include" \
  -I"${ROOT}/3rdparty/cutlass/tools/util/include" \
  -I"${ROOT}/3rdparty/cutlass/include/cute" \
  "${SRC}" -o "${OUT}"

echo "[run] group=1 (M=256 N=5888 K=2944)"
"${OUT}" 1 256 5888 2944 || true

echo "[run] group=8 (M=256 N=5888 K=2944)"
"${OUT}" 8 256 5888 2944 || true

echo "[run] group=1 sanity (M=N=K=128)"
"${OUT}" 1 128 128 128 || true

