#!/usr/bin/env bash
set -euo pipefail

STEP_CU="${1:-}"
if [[ -z "${STEP_CU}" ]]; then
  echo "usage: $0 <stepXX_*.cu> [run args...]"
  exit 2
fi
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

SRC="${HERE}/${STEP_CU}"
OUT="${HERE}/${STEP_CU%.cu}"

CUDA_ARCH="${CUDA_ARCH:-sm_121}"

echo "[build] SRC=${SRC}"
echo "[build] OUT=${OUT}"
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

echo "[run] ${OUT} $*"
"${OUT}" "$@"

