#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

// CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

using namespace cute;

static inline void ck(cudaError_t st, char const* what) {
  if (st != cudaSuccess) {
    std::fprintf(stderr, "CUDA ERROR: %s: %s\n", what, cudaGetErrorString(st));
    std::exit(1);
  }
}

static inline char const* cutlass_status(cutlass::Status st) {
  return cutlassGetStatusString(st);
}

template <typename T>
static inline void* cuda_malloc_bytes(size_t bytes) {
  void* p = nullptr;
  ck(cudaMalloc(&p, bytes), "cudaMalloc");
  return p;
}

template <typename T>
static inline T* cuda_malloc_elems(size_t elems) {
  void* p = cuda_malloc_bytes<T>(elems * sizeof(T));
  return reinterpret_cast<T*>(p);
}

static inline void fill_u8(void* ptr, size_t bytes, uint8_t v) {
  ck(cudaMemset(ptr, v, bytes), "cudaMemset");
}

template <int SFVectorSize>
int run_once(int num_groups, int64_t M, int64_t N, int64_t K) {
#if !(defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED))
  std::fprintf(stderr, "CUTLASS_ARCH_MMA_SM12x_SUPPORTED not enabled in this build.\n");
  return 2;
#else
  using ElementInputA = cutlass::float_e4m3_t;
  using ElementInputB = cutlass::float_e2m1_t;
  using ElementA = cutlass::mx_float8_t<ElementInputA>;
  using ElementB = cutlass::mx_float4_t<ElementInputB>;

  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;

  using ElementAccumulator = float;
  using ElementCompute = float;

  using ElementSF = cutlass::float_ue8m0_t;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  constexpr int AlignmentB = 128;  // MXFP4 special-case
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using TileShape_MNK = Shape<_128, _128, _128>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape_MNK, ClusterShape_MNK,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
          ElementC, LayoutC*, AlignmentC, ElementD, LayoutC*, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

  // Critical change vs step01: use CUTLASS’ SM120 blockscaled ptr-array schedule tag.
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
          ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCount,
          cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120>::CollectiveOp;

  using ProblemShape =
      cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename CollectiveMainloop::StrideA;
  using StrideB = typename CollectiveMainloop::StrideB;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename CollectiveMainloop::LayoutSFB;

  auto* A = cuda_malloc_elems<ElementInputA>(static_cast<size_t>(M * K));
  size_t b_bytes = static_cast<size_t>(N * K) / 2;
  auto* B = reinterpret_cast<ElementInputB*>(cuda_malloc_bytes<void>(b_bytes));
  auto* D = cuda_malloc_elems<ElementD>(static_cast<size_t>(M * N));

  auto sfa_layout0 = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>::tile_atom_to_shape_SFA(
      cute::make_shape((int)M, (int)N, (int)K, 1));
  auto sfb_layout0 = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>::tile_atom_to_shape_SFB(
      cute::make_shape((int)M, (int)N, (int)K, 1));
  size_t sfa_bytes = static_cast<size_t>(cute::cosize(sfa_layout0)) * sizeof(ElementSF);
  size_t sfb_bytes = static_cast<size_t>(cute::cosize(sfb_layout0)) * sizeof(ElementSF);

  auto* SFA = reinterpret_cast<ElementSF*>(cuda_malloc_bytes<void>(sfa_bytes));
  auto* SFB = reinterpret_cast<ElementSF*>(cuda_malloc_bytes<void>(sfb_bytes));
  fill_u8(SFA, sfa_bytes, 0x7f);
  fill_u8(SFB, sfb_bytes, 0x7f);

  auto* problem_sizes = cuda_malloc_elems<typename ProblemShape::UnderlyingProblemShape>(num_groups);
  auto* A_ptr = cuda_malloc_elems<ElementInputA const*>(num_groups);
  auto* B_ptr = cuda_malloc_elems<ElementInputB const*>(num_groups);
  auto* D_ptr = cuda_malloc_elems<ElementD*>(num_groups);
  auto* stride_A = cuda_malloc_elems<std::remove_pointer_t<StrideA>>(num_groups);
  auto* stride_B = cuda_malloc_elems<std::remove_pointer_t<StrideB>>(num_groups);
  auto* stride_D = cuda_malloc_elems<std::remove_pointer_t<StrideD>>(num_groups);
  auto* layout_SFA = cuda_malloc_elems<std::remove_pointer_t<LayoutSFA>>(num_groups);
  auto* layout_SFB = cuda_malloc_elems<std::remove_pointer_t<LayoutSFB>>(num_groups);
  auto* SFA_ptr = cuda_malloc_elems<ElementSF const*>(num_groups);
  auto* SFB_ptr = cuda_malloc_elems<ElementSF const*>(num_groups);

  std::vector<typename ProblemShape::UnderlyingProblemShape> h_shapes(num_groups);
  std::vector<ElementInputA const*> h_A_ptr(num_groups);
  std::vector<ElementInputB const*> h_B_ptr(num_groups);
  std::vector<ElementD*> h_D_ptr(num_groups);
  std::vector<std::remove_pointer_t<StrideA>> h_stride_A(num_groups);
  std::vector<std::remove_pointer_t<StrideB>> h_stride_B(num_groups);
  std::vector<std::remove_pointer_t<StrideD>> h_stride_D(num_groups);
  std::vector<std::remove_pointer_t<LayoutSFA>> h_layout_SFA(num_groups);
  std::vector<std::remove_pointer_t<LayoutSFB>> h_layout_SFB(num_groups);
  std::vector<ElementSF const*> h_SFA_ptr(num_groups);
  std::vector<ElementSF const*> h_SFB_ptr(num_groups);

  for (int i = 0; i < num_groups; ++i) {
    h_shapes[i] = typename ProblemShape::UnderlyingProblemShape(M, N, K);
    h_A_ptr[i] = A;
    h_B_ptr[i] = B;
    h_D_ptr[i] = D;
    h_stride_A[i] = cutlass::make_cute_packed_stride(
        std::remove_pointer_t<StrideA>{},
        cute::Shape<int, int, int>{(int)M, (int)K, 1});
    h_stride_B[i] = cutlass::make_cute_packed_stride(
        std::remove_pointer_t<StrideB>{},
        cute::Shape<int, int, int>{(int)N, (int)K, 1});
    h_stride_D[i] = cutlass::make_cute_packed_stride(
        std::remove_pointer_t<StrideD>{},
        cute::Shape<int, int, int>{(int)M, (int)N, 1});
    h_layout_SFA[i] = sfa_layout0;
    h_layout_SFB[i] = sfb_layout0;
    h_SFA_ptr[i] = SFA;
    h_SFB_ptr[i] = SFB;
  }

  ck(cudaMemcpy(problem_sizes, h_shapes.data(), sizeof(h_shapes[0]) * h_shapes.size(),
                cudaMemcpyHostToDevice),
     "H2D problem_sizes");
  ck(cudaMemcpy(A_ptr, h_A_ptr.data(), sizeof(h_A_ptr[0]) * h_A_ptr.size(), cudaMemcpyHostToDevice),
     "H2D A_ptr");
  ck(cudaMemcpy(B_ptr, h_B_ptr.data(), sizeof(h_B_ptr[0]) * h_B_ptr.size(), cudaMemcpyHostToDevice),
     "H2D B_ptr");
  ck(cudaMemcpy(D_ptr, h_D_ptr.data(), sizeof(h_D_ptr[0]) * h_D_ptr.size(), cudaMemcpyHostToDevice),
     "H2D D_ptr");
  ck(cudaMemcpy(stride_A, h_stride_A.data(), sizeof(h_stride_A[0]) * h_stride_A.size(),
                cudaMemcpyHostToDevice),
     "H2D stride_A");
  ck(cudaMemcpy(stride_B, h_stride_B.data(), sizeof(h_stride_B[0]) * h_stride_B.size(),
                cudaMemcpyHostToDevice),
     "H2D stride_B");
  ck(cudaMemcpy(stride_D, h_stride_D.data(), sizeof(h_stride_D[0]) * h_stride_D.size(),
                cudaMemcpyHostToDevice),
     "H2D stride_D");
  ck(cudaMemcpy(layout_SFA, h_layout_SFA.data(), sizeof(h_layout_SFA[0]) * h_layout_SFA.size(),
                cudaMemcpyHostToDevice),
     "H2D layout_SFA");
  ck(cudaMemcpy(layout_SFB, h_layout_SFB.data(), sizeof(h_layout_SFB[0]) * h_layout_SFB.size(),
                cudaMemcpyHostToDevice),
     "H2D layout_SFB");
  ck(cudaMemcpy(SFA_ptr, h_SFA_ptr.data(), sizeof(h_SFA_ptr[0]) * h_SFA_ptr.size(),
                cudaMemcpyHostToDevice),
     "H2D SFA_ptr");
  ck(cudaMemcpy(SFB_ptr, h_SFB_ptr.data(), sizeof(h_SFB_ptr[0]) * h_SFB_ptr.size(),
                cudaMemcpyHostToDevice),
     "H2D SFB_ptr");

  cutlass::KernelHardwareInfo hw{};
  int device_id = 0;
  ck(cudaGetDevice(&device_id), "cudaGetDevice");
  int sm_count = 0;
  ck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id),
     "cudaDevAttrMultiProcessorCount");
  hw.device_id = device_id;
  hw.sm_count = sm_count;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_groups, problem_sizes, nullptr},
      {A_ptr,
       reinterpret_cast<StrideA>(stride_A),
       B_ptr,
       reinterpret_cast<StrideB>(stride_B),
       SFA_ptr,
       reinterpret_cast<LayoutSFA>(layout_SFA),
       SFB_ptr,
       reinterpret_cast<LayoutSFB>(layout_SFB)},
      {{}, nullptr, nullptr, D_ptr, reinterpret_cast<StrideD>(stride_D)},
      hw};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;

  auto can = Gemm{}.can_implement(args);
  std::printf("[step06] num_groups=%d M=%lld N=%lld K=%lld can_implement=%s\n",
              num_groups, (long long)M, (long long)N, (long long)K, cutlass_status(can));

  Gemm gemm;
  size_t ws_bytes = gemm.get_workspace_size(args);
  void* ws = cuda_malloc_bytes<void>(ws_bytes);
  auto init = gemm.initialize(args, ws, /*stream=*/nullptr);
  std::printf("[step06] initialize=%s (ws_bytes=%zu)\n", cutlass_status(init), ws_bytes);

  cudaFree(ws);
  cudaFree(problem_sizes);
  cudaFree(A_ptr);
  cudaFree(B_ptr);
  cudaFree(D_ptr);
  cudaFree(stride_A);
  cudaFree(stride_B);
  cudaFree(stride_D);
  cudaFree(layout_SFA);
  cudaFree(layout_SFB);
  cudaFree(SFA_ptr);
  cudaFree(SFB_ptr);
  cudaFree(SFA);
  cudaFree(SFB);
  cudaFree(A);
  cudaFree(B);
  cudaFree(D);
  return (init == cutlass::Status::kSuccess) ? 0 : 1;
#endif
}

int main(int argc, char** argv) {
  int num_groups = 1;
  int64_t M = 256;
  int64_t N = 5888;
  int64_t K = 2944;
  if (argc >= 2) num_groups = std::atoi(argv[1]);
  if (argc >= 5) {
    M = std::atoll(argv[2]);
    N = std::atoll(argv[3]);
    K = std::atoll(argv[4]);
  }
  return run_once<32>(num_groups, M, N, K);
}

