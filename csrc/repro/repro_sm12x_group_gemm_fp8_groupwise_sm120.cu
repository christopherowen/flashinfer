#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/numeric_types.h"

// FlashInfer "known working" SM12x grouped GEMM
#include "flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh"

static inline void ck(cudaError_t st, char const* what) {
  if (st != cudaSuccess) {
    std::fprintf(stderr, "CUDA ERROR: %s: %s\n", what, cudaGetErrorString(st));
    std::exit(1);
  }
}

template <typename T>
static inline T* dmalloc_elems(size_t n) {
  void* p = nullptr;
  ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
  return reinterpret_cast<T*>(p);
}

static inline void fill_f32(float* d, size_t n, float v) {
  std::vector<float> h(n, v);
  ck(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D float fill");
}

int main(int argc, char** argv) {
  // Defaults modeled after your MoE dimensions, but small M for faster runtime
  int num_groups = 1;
  int M = 256;     // rows per group (tokens per expert)
  int N = 5888;    // out features
  int K = 2944;    // in features

  if (argc >= 2) num_groups = std::atoi(argv[1]);
  if (argc >= 5) {
    M = std::atoi(argv[2]);
    N = std::atoi(argv[3]);
    K = std::atoi(argv[4]);
  }

  // This kernel supports only these granularities (see header static_asserts)
  constexpr int Gm = 128;
  constexpr int Gn = 128;
  constexpr int Gk = 128;
  constexpr bool ScaleMajorK = true;

  using DTypeIn = cutlass::float_e4m3_t;   // FP8 input
  using DTypeOut = cutlass::bfloat16_t;    // BF16 output

  // Total rows is sum across groups. Here we make each group exactly M rows.
  int total_M = num_groups * M;

  // Device buffers
  auto* A = dmalloc_elems<DTypeIn>(static_cast<size_t>(total_M) * K);
  auto* B = dmalloc_elems<DTypeIn>(static_cast<size_t>(num_groups) * N * K);
  auto* D = dmalloc_elems<DTypeOut>(static_cast<size_t>(total_M) * N);

  // m_indptr: prefix sum of M per group (length num_groups + 1)
  std::vector<int> h_m_indptr(num_groups + 1, 0);
  for (int i = 0; i < num_groups; ++i) h_m_indptr[i + 1] = (i + 1) * M;
  auto* m_indptr = dmalloc_elems<int>(static_cast<size_t>(num_groups + 1));
  ck(cudaMemcpy(m_indptr, h_m_indptr.data(), h_m_indptr.size() * sizeof(int),
                cudaMemcpyHostToDevice),
     "H2D m_indptr");

  // Scale factor buffers (float scales, not ue8m0). Size per header’s compute kernel.
  int sf_n = (N + Gn - 1) / Gn;
  int sf_k = (K + Gk - 1) / Gk;
  int sf_m_total = (total_M + Gm - 1) / Gm;

  size_t sfa_elems = static_cast<size_t>(sf_m_total) * sf_k;
  size_t sfb_elems = static_cast<size_t>(num_groups) * sf_n * sf_k;
  auto* SFA = dmalloc_elems<float>(sfa_elems);
  auto* SFB = dmalloc_elems<float>(sfb_elems);
  fill_f32(SFA, sfa_elems, 1.0f);
  fill_f32(SFB, sfb_elems, 1.0f);

  // Internal buffers for the FlashInfer helper (must be device memory)
  // These sizes are intentionally generous; if too small, the function will assert/fail.
  size_t int_bytes = 64 * 1024 * 1024;   // 64MB
  size_t float_bytes = 64 * 1024 * 1024; // 64MB
  void* int_buf = nullptr;
  void* float_buf = nullptr;
  ck(cudaMalloc(&int_buf, int_bytes), "cudaMalloc int_buf");
  ck(cudaMalloc(&float_buf, float_bytes), "cudaMalloc float_buf");

  cudaStream_t stream{};
  ck(cudaStreamCreate(&stream), "cudaStreamCreate");

  auto st = flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM120<
      Gm, Gn, Gk, ScaleMajorK, DTypeIn, DTypeOut>(
      int_buf, int_bytes, float_buf, float_bytes,
      A, B, SFA, SFB, D,
      m_indptr,
      /*max_m=*/M,  // each group has M
      /*n=*/N, /*k=*/K,
      /*num_groups=*/num_groups,
      stream);

  // Force sync to surface async failures (illegal instruction, device asserts, etc.)
  cudaError_t st_sync = cudaStreamSynchronize(stream);
  cudaError_t st_last = cudaGetLastError();

  std::printf(
      "[known-working repro] num_groups=%d M=%d N=%d K=%d -> call=%s (%d) sync=%s (%d) last=%s (%d)\n",
      num_groups, M, N, K,
      cudaGetErrorString(st), (int)st,
      cudaGetErrorString(st_sync), (int)st_sync,
      cudaGetErrorString(st_last), (int)st_last);

  // Cleanup
  cudaStreamDestroy(stream);
  cudaFree(int_buf);
  cudaFree(float_buf);
  cudaFree(A);
  cudaFree(B);
  cudaFree(D);
  cudaFree(m_indptr);
  cudaFree(SFA);
  cudaFree(SFB);

  return (st == cudaSuccess) ? 0 : 1;
}

