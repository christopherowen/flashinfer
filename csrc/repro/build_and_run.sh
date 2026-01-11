#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

SRC="${HERE}/repro_sm12x_mxf8_mxf4_grouped_init.cu"
OUT="${HERE}/repro_sm12x_mxf8_mxf4_grouped_init"

CUDA_ARCH="${CUDA_ARCH:-sm_121}"

echo "[build] ROOT=${ROOT}"
echo "[build] CUDA_ARCH=${CUDA_ARCH}"

nvcc -std=c++17 -O2 -lineinfo \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -arch="${CUDA_ARCH}" \
  -DCUTLASS_ARCH_MMA_SM121_SUPPORTED -DCUTLASS_ARCH_MMA_SM12x_SUPPORTED \
  -I"${ROOT}/3rdparty/cutlass/include" \
  -I"${ROOT}/3rdparty/cutlass/tools/util/include" \
  -I"${ROOT}/3rdparty/cutlass/include/cute" \
  "${SRC}" -o "${OUT}"

echo "[run] group=1"
"${OUT}" 1 || true
echo "[run] group=128"
"${OUT}" 128 || true

echo "[run] sanity tiny (group=1, M=N=K=128)"
"${OUT}" 1 128 128 128 || true

echo "[run] sanity tiny2 (group=1, M=128 N=256 K=128)"
"${OUT}" 1 128 256 128 || true

