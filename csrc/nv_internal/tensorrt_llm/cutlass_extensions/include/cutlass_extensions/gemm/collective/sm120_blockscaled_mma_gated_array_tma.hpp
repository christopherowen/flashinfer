/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Sequential SMEM Reuse Gated Mainloop for SM120 block-scaled MMA.
 *
 * Key design: Instead of allocating 6 SMEM arrays (A, B, Aux, SFA, SFB, SFAux),
 * we allocate only 4 (A, B, SFA, SFB) and reuse smem_B/smem_SFB for both linear
 * and gate weights by running two sequential K-reduction passes:
 *
 *   Phase 1: Load A + B_linear + SFA + SFB_linear → accum0 (linear GEMM)
 *   Phase 2: Load A + B_gate + SFA + SFB_gate → accum1 (gate GEMM)
 *   Then:    accum = SiLU(accum1) * accum0  (SwiGLU in registers)
 *
 * This reduces SMEM from ~112KB (6 arrays @ 128x128) to ~66KB (4 arrays @ 128x128),
 * fitting within SM121's 101KB limit and satisfying the N%128 hardware constraint.
 *
 * The kernel doubles k_tile_count for this mainloop (via IsGated flag), so load()
 * and mma() each receive 2 * real_k_tile_count and split into two phases internally.
 *
 **************************************************************************************************/

#pragma once

#include <limits>

// Must include primary CollectiveMma template declaration BEFORE any specializations
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Dispatch policy for gated SM120 block-scaled mainloop
template <int Stages_, int SchedulerPipelineStageCount_, class ClusterShape_, class KernelSchedule_>
struct MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated {
  static constexpr int Stages = Stages_;
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using KernelSchedule = KernelSchedule_;
  using Schedule = KernelSchedule_;  // Alias required by GemmUniversal kernel
  using ArchTag = arch::Sm120;
  static constexpr bool IsGated = true;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Sequential SMEM Reuse Gated Collective MMA for SM120 Block-Scaled Array TMA
///
/// Uses 4 SMEM arrays (same as unfused kernel), runs 2 sequential K passes.
/// The kernel provides 2*k_tile_count pipeline stages; this mainloop splits them
/// into two phases: linear GEMM then gate GEMM, with SwiGLU applied in registers.
///
template <
  int Stages,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class KernelScheduleType,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomsA_,
  class SmemCopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomsB_,
  class SmemCopyAtomsB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomsA_,
    SmemCopyAtomsA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomsB_,
    SmemCopyAtomsB_,
    TransformB_> {

  //
  // Base class for type reuse (NOT inheritance for Params!)
  //
  using BaseDispatchPolicy = MainloopSm120ArrayTmaWarpSpecializedBlockScaled<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
  
  using Base = CollectiveMma<
    BaseDispatchPolicy,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomsA_,
    SmemCopyAtomsA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomsB_,
    SmemCopyAtomsB_,
    TransformB_>;

  // Re-export all base type aliases
  using DispatchPolicy = MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
  
  using TileShape = typename Base::TileShape;
  using ElementPairA = typename Base::ElementPairA;
  using ElementPairB = typename Base::ElementPairB;
  using StridePairA = typename Base::StridePairA;
  using StridePairB = typename Base::StridePairB;
  
  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using StrideA = typename Base::StrideA;
  using StrideB = typename Base::StrideB;
  using InternalStrideA = typename Base::InternalStrideA;
  using InternalStrideB = typename Base::InternalStrideB;
  
  using ElementSF = typename Base::ElementSF;
  using LayoutSFA = typename Base::LayoutSFA;
  using LayoutSFB = typename Base::LayoutSFB;
  using InternalLayoutSFA = typename Base::InternalLayoutSFA;
  using InternalLayoutSFB = typename Base::InternalLayoutSFB;
  
  using TiledMma = typename Base::TiledMma;
  using ElementAccumulator = typename Base::ElementAccumulator;
  
  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutB = typename Base::SmemLayoutB;
  using SmemLayoutSFA = typename Base::SmemLayoutSFA;
  using SmemLayoutSFB = typename Base::SmemLayoutSFB;
  
  using SmemCopyAtomA = typename Base::SmemCopyAtomA;
  using SmemCopyAtomB = typename Base::SmemCopyAtomB;
  using SmemCopyAtomSFA = typename Base::SmemCopyAtomSFA;
  using SmemCopyAtomSFB = typename Base::SmemCopyAtomSFB;
  
  using MainloopPipeline = typename Base::MainloopPipeline;
  using PipelineState = typename Base::PipelineState;
  using PipelineParams = typename Base::PipelineParams;
  
  using ArchTag = typename Base::ArchTag;
  
  static constexpr int K_PIPE_MAX = Base::K_PIPE_MAX;
  static constexpr int K_PIPE_MMAS = Base::K_PIPE_MMAS;
  static constexpr int K_BLOCK_MAX = Base::K_BLOCK_MAX;
  static constexpr int NumProducerThreadEvents = Base::NumProducerThreadEvents;

  // Gated-specific: Aux uses same types as B (for TMA descriptors and pointers)
  using ElementAux = ElementB;
  using StrideAux = StrideB;
  using InternalStrideAux = InternalStrideB;
  using LayoutSFAux = LayoutSFB;
  using InternalLayoutSFAux = InternalLayoutSFB;

  // SMEM allocation types
  using SmemAllocTypeA = typename Base::SmemAllocTypeA;
  using SmemAllocTypeB = typename Base::SmemAllocTypeB;

  static constexpr bool IsGated = true;

  //
  // SharedStorage - ONLY 4 SMEM arrays (same as base/unfused kernel)
  //
  // smem_B and smem_SFB are REUSED for both linear and gate weights
  // across two sequential K passes.
  //
  struct TensorStorage : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
    alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
    alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    // NO smem_Aux or smem_SFAux - they are eliminated by sequential SMEM reuse

    // Expert index for the current CTA, written by load() and read by mma()
    // to index per-expert SwiGLU parameter arrays and FC1 bias.
    int expert_idx;
    // Tile N offset (global column index of this CTA's tile), written by load()
    // and read by mma() to index per-column FC1 bias vectors.
    int n_tile_offset;
  };

  using SharedStorage = TensorStorage;

  // Alignment validation
  static_assert(offsetof(TensorStorage, smem_A) % 1024 == 0,
                "smem_A must be 1024-byte aligned within TensorStorage");
  static_assert(offsetof(TensorStorage, smem_B) % 1024 == 0,
                "smem_B must be 1024-byte aligned within TensorStorage");
  static_assert(offsetof(TensorStorage, smem_SFA) % 16 == 0,
                "smem_SFA must be 16-byte aligned within TensorStorage");
  static_assert(offsetof(TensorStorage, smem_SFB) % 16 == 0,
                "smem_SFB must be 16-byte aligned within TensorStorage");

  // SMEM budget check: must fit within SM121's 101,376 bytes
  static_assert(sizeof(TensorStorage) <= 101376,
                "TensorStorage exceeds SM121's 101KB SMEM limit");

  // Pipeline storage (from base)
  using PipelineStorage = typename Base::PipelineStorage;
  
  // TensorMap storage - 6 tensormaps (A, B, SFA, SFB + Aux, SFAux descriptors)
  // We still need TMA descriptors for Aux/SFAux even though they share smem_B/smem_SFB
  struct TensorMapStorage : cute::aligned_struct<128, cute::Int<0>> {
    cute::TmaDescriptor smem_tensormap_A;
    cute::TmaDescriptor smem_tensormap_B;
    cute::TmaDescriptor smem_tensormap_SFA;
    cute::TmaDescriptor smem_tensormap_SFB;
    cute::TmaDescriptor smem_tensormap_Aux;
    cute::TmaDescriptor smem_tensormap_SFAux;
  };

  //
  // Arguments - extended for Aux pointers
  //
  struct Arguments {
    ElementA const** ptr_A = nullptr;
    StrideA dA{};
    ElementB const** ptr_B = nullptr;
    StrideB dB{};
    ElementSF const** ptr_SFA = nullptr;
    LayoutSFA layout_SFA{};
    ElementSF const** ptr_SFB = nullptr;
    LayoutSFB layout_SFB{};
    
    // Gated extension: Aux (gate weights)
    ElementAux const** ptr_Aux = nullptr;
    StrideAux dAux{};
    ElementSF const** ptr_SFAux = nullptr;
    LayoutSFAux layout_SFAux{};

    // Per-expert SwiGLU activation parameters (device pointers).
    // gate_clamped = min(gate, limit)
    // linear_clamped = clamp(linear, -limit, limit)
    // output = gate_clamped * sigmoid(gate_clamped * alpha) * (linear_clamped + beta)
    //
    // nullptr = use defaults (alpha=1, beta=0, limit=inf) = standard SwiGLU.
    float const* swiglu_alpha = nullptr;
    float const* swiglu_beta = nullptr;
    float const* swiglu_limit = nullptr;

    // Per-expert FC1 bias, shape [num_experts, 2*inter_size].
    // Layout per expert: [bias_linear(inter_size) | bias_gate(inter_size)].
    // nullptr = no bias. Stored as bfloat16, converted to float in mma().
    void const* fc1_bias = nullptr;
    int64_t bias_inter_size = 0;  // inter_size for offset computation
  };

  //
  // AuxParams - aux-specific device params (for gate weight TMA descriptors)
  //
  struct AuxParams {
    typename Base::Params::TMA_B tma_load_aux;
    typename Base::Params::TMA_SFB tma_load_sfaux;
    ElementAux const** ptr_Aux;
    StrideAux dAux;
    ElementSF const** ptr_SFAux;
    LayoutSFAux layout_SFAux;
  };

  //
  // Params - COMPOSITION pattern (base + aux)
  //
  // CRITICAL: tma_transaction_bytes is now STANDARD 4-plane (MK + NK per stage).
  // Each phase loads 4 TMA arrays into the same 4 SMEM slots; the kernel doubles
  // the total stage count to accommodate two phases.
  //
  struct Params {
    typename Base::Params base;
    AuxParams aux;
    
    // Standard 4-plane transaction bytes (same as unfused kernel)
    uint32_t tma_transaction_bytes = Base::TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = Base::TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = Base::TmaTransactionBytesNK;

    // Per-expert SwiGLU parameter arrays (device pointers, nullptr = defaults)
    float const* swiglu_alpha = nullptr;
    float const* swiglu_beta = nullptr;
    float const* swiglu_limit = nullptr;

    // Per-expert FC1 bias (device pointer, nullptr = no bias)
    void const* fc1_bias = nullptr;
    int64_t bias_inter_size = 0;
  };

  static_assert(std::is_trivially_copyable_v<Params>, 
                "Gated mainloop Params must be trivially copyable");

  //
  // to_underlying_arguments - builds composed Params
  //
  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {
    
    // Build base args
    typename Base::Arguments base_args{
      args.ptr_A, args.dA,
      args.ptr_B, args.dB,
      args.ptr_SFA, args.layout_SFA,
      args.ptr_SFB, args.layout_SFB
    };
    
    auto base_params = Base::to_underlying_arguments(problem_shape, base_args, workspace);
    
    // Build aux TMA using same pattern as B
    typename Base::Arguments aux_as_b_args{
      args.ptr_A, args.dA,
      args.ptr_Aux, args.dAux,
      args.ptr_SFA, args.layout_SFA,
      args.ptr_SFAux, args.layout_SFAux
    };
    auto aux_params = Base::to_underlying_arguments(problem_shape, aux_as_b_args, workspace);
    
    AuxParams aux;
    aux.tma_load_aux = aux_params.tma_load_b;
    aux.tma_load_sfaux = aux_params.tma_load_sfb;
    aux.ptr_Aux = reinterpret_cast<ElementAux const**>(args.ptr_Aux);
    aux.dAux = aux_params.dB;
    aux.ptr_SFAux = reinterpret_cast<ElementSF const**>(args.ptr_SFAux);
    aux.layout_SFAux = aux_params.layout_SFB;
    
    Params result;
    result.base = base_params;
    result.aux = aux;
    result.swiglu_alpha = args.swiglu_alpha;
    result.swiglu_beta = args.swiglu_beta;
    result.swiglu_limit = args.swiglu_limit;
    result.fc1_bias = args.fc1_bias;
    result.bias_inter_size = args.bias_inter_size;
    return result;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    // 6 tensormaps (even though only 4 SMEM slots): A, B, SFA, SFB + Aux, SFAux
    constexpr uint32_t NumInputTensors = 6;
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    return (NumInputTensors * SizeOfCuTensorMap * sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape, Arguments const& args, 
      void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(
      ProblemShape problem_shapes,
      [[maybe_unused]] Arguments const& args) {
    typename Base::Arguments base_args{
      args.ptr_A, args.dA, args.ptr_B, args.dB,
      args.ptr_SFA, args.layout_SFA, args.ptr_SFB, args.layout_SFB
    };
    return Base::can_implement(problem_shapes, base_args);
  }

  //
  // Tensormap methods - initialize 6 tensormaps
  //
  CUTLASS_DEVICE auto
  tensormaps_init(Params const& params, TensorMapStorage& shared_tensormaps, 
                  int32_t sm_count, int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap = params.base.tensormaps;
    
    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];
    cute::TmaDescriptor* tma_desc_sfa = &gmem_tensormap[sm_idx + 2 * sm_count];
    cute::TmaDescriptor* tma_desc_sfb = &gmem_tensormap[sm_idx + 3 * sm_count];
    cute::TmaDescriptor* tma_desc_aux = &gmem_tensormap[sm_idx + 4 * sm_count];
    cute::TmaDescriptor* tma_desc_sfaux = &gmem_tensormap[sm_idx + 5 * sm_count];
    
    if (cute::elect_one_sync()) {
      Tensor pA_tensormap = make_tensor(params.base.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
      Tensor pB_tensormap = make_tensor(params.base.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});
      Tensor pSFA_tensormap = make_tensor(params.base.tma_load_sfa.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sSFA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_SFA), Int<1>{}, Int<1>{});
      Tensor pSFB_tensormap = make_tensor(params.base.tma_load_sfb.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sSFB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_SFB), Int<1>{}, Int<1>{});
      
      Tensor pAux_tensormap = make_tensor(params.aux.tma_load_aux.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sAux_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_Aux), Int<1>{}, Int<1>{});
      Tensor pSFAux_tensormap = make_tensor(params.aux.tma_load_sfaux.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sSFAux_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_SFAux), Int<1>{}, Int<1>{});
      
      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
      copy(recast<uint128_t>(pSFA_tensormap), recast<uint128_t>(sSFA_tensormap));
      copy(recast<uint128_t>(pSFB_tensormap), recast<uint128_t>(sSFB_tensormap));
      copy(recast<uint128_t>(pAux_tensormap), recast<uint128_t>(sAux_tensormap));
      copy(recast<uint128_t>(pSFAux_tensormap), recast<uint128_t>(sSFAux_tensormap));
    }
    __syncwarp();
    
    return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_sfa, tma_desc_sfb, tma_desc_aux, tma_desc_sfaux);
  }

  template <class TensorMapTuple, class ProblemShape>
  CUTLASS_DEVICE void
  tensormaps_perform_update(TensorMapStorage& shared_tensormaps, Params const& params,
                            TensorMapTuple const& input_tensormaps,
                            ProblemShape const& problem_shape, int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replace global addresses for all 6 tensormaps
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                      params.base.ptr_A[next_batch]);
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                      params.base.ptr_B[next_batch]);
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFA,
                                                      params.base.ptr_SFA[next_batch]);
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFB,
                                                      params.base.ptr_SFB[next_batch]);
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_Aux,
                                                      params.aux.ptr_Aux[next_batch]);
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFAux,
                                                      params.aux.ptr_SFAux[next_batch]);
      
      if constexpr (Base::IsGroupedGemmKernel) {
        tensormaps_replace_global_tensor_properties(shared_tensormaps, params, next_batch, problem_shape);
      }
    }
  }
  
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    
    constexpr int MaxTensorRank = 5;
    auto [M, N, K, L] = problem_shape_mnkl;
    
    cute::array<uint32_t, MaxTensorRank> prob_shape_A = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_SFA = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_SFA = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_SFB = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_SFB = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_Aux = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_Aux = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_SFAux = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_SFAux = {0,0,0,0,0};
    
    using TmaInternalElementA = typename Base::TmaInternalElementA;
    using TmaInternalElementB = typename Base::TmaInternalElementB;
    
    TmaInternalElementA const* ptr_A = nullptr;
    Tensor tensor_a = make_tensor(ptr_A, make_shape(M, K, Int<1>{}), params.base.dA[next_group]);
    ElementSF const* ptr_SF = nullptr;
    Tensor tensor_sfa = make_tensor(ptr_SF, params.base.layout_SFA[next_group]);
    TmaInternalElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(ptr_B, make_shape(N, K, Int<1>{}), params.base.dB[next_group]);
    Tensor tensor_sfb = make_tensor(ptr_SF, params.base.layout_SFB[next_group]);
    Tensor tensor_aux = make_tensor(ptr_B, make_shape(N, K, Int<1>{}), params.aux.dAux[next_group]);
    Tensor tensor_sfaux = make_tensor(ptr_SF, params.aux.layout_SFAux[next_group]);
    
    cute::detail::fill_tma_gmem_shape_stride(params.base.tma_load_a, tensor_a, prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(params.base.tma_load_sfa, tensor_sfa, prob_shape_SFA, prob_stride_SFA);
    cute::detail::fill_tma_gmem_shape_stride(params.base.tma_load_b, tensor_b, prob_shape_B, prob_stride_B);
    cute::detail::fill_tma_gmem_shape_stride(params.base.tma_load_sfb, tensor_sfb, prob_shape_SFB, prob_stride_SFB);
    cute::detail::fill_tma_gmem_shape_stride(params.aux.tma_load_aux, tensor_aux, prob_shape_Aux, prob_stride_Aux);
    cute::detail::fill_tma_gmem_shape_stride(params.aux.tma_load_sfaux, tensor_sfaux, prob_shape_SFAux, prob_stride_SFAux);
    
    for (uint64_t& stride : prob_stride_A) { stride = (stride * sizeof_bits_v<TmaInternalElementA>) / 8; }
    for (uint64_t& stride : prob_stride_SFA) { stride = (stride * sizeof_bits_v<ElementSF>) / 8; }
    for (uint64_t& stride : prob_stride_B) { stride = (stride * sizeof_bits_v<TmaInternalElementB>) / 8; }
    for (uint64_t& stride : prob_stride_SFB) { stride = (stride * sizeof_bits_v<ElementSF>) / 8; }
    for (uint64_t& stride : prob_stride_Aux) { stride = (stride * sizeof_bits_v<TmaInternalElementB>) / 8; }
    for (uint64_t& stride : prob_stride_SFAux) { stride = (stride * sizeof_bits_v<ElementSF>) / 8; }
    
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A, prob_shape_A, prob_stride_A);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_SFA, prob_shape_SFA, prob_stride_SFA);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B, prob_shape_B, prob_stride_B);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_SFB, prob_shape_SFB, prob_stride_SFB);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_Aux, prob_shape_Aux, prob_stride_Aux);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_SFAux, prob_shape_SFAux, prob_stride_SFAux);
  }

  template <class TensorMapTuple>
  CUTLASS_DEVICE void
  tensormaps_cp_fence_release(TensorMapStorage& shared_tensormaps, 
                              TensorMapTuple const& input_tensormaps) {
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    cute::tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
    cute::tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
    cute::tma_descriptor_cp_fence_release(get<2>(input_tensormaps), shared_tensormaps.smem_tensormap_SFA);
    cute::tma_descriptor_cp_fence_release(get<3>(input_tensormaps), shared_tensormaps.smem_tensormap_SFB);
    cute::tma_descriptor_cp_fence_release(get<4>(input_tensormaps), shared_tensormaps.smem_tensormap_Aux);
    cute::tma_descriptor_cp_fence_release(get<5>(input_tensormaps), shared_tensormaps.smem_tensormap_SFAux);
  }

  template <class TensorMapTuple>
  CUTLASS_DEVICE void
  tensormaps_fence_acquire(TensorMapTuple const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<2>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<3>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<4>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<5>(input_tensormaps));
  }

  template <class LoadTuple, class ProblemShape>
  CUTLASS_DEVICE auto
  tensors_perform_update(LoadTuple const& load_inputs, Params const& params,
                         ProblemShape const& problem_shape, int32_t batch_idx) {
    Base base_collective;
    auto base_tuple = cute::make_tuple(get<0>(load_inputs), get<1>(load_inputs),
                                        get<3>(load_inputs), get<4>(load_inputs));
    auto updated_base = base_collective.tensors_perform_update(base_tuple, params.base, 
                                                                problem_shape, batch_idx);
    return cute::make_tuple(
      get<0>(updated_base),   // A
      get<1>(updated_base),   // B
      get<2>(load_inputs),    // Aux (unchanged)
      get<2>(updated_base),   // SFA
      get<3>(updated_base),   // SFB
      get<5>(load_inputs)     // SFAux (unchanged)
    );
  }

  //
  // load_init - returns 6-tuple (A, B, Aux, SFA, SFB, SFAux)
  //
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& params) {
    using X = cute::Underscore;
    
    auto [M, N, K, L] = problem_shape_MNKL;
    const int32_t init_L = 1;
    
    Tensor mA_mkl = params.base.tma_load_a.get_tma_tensor(cute::make_shape(M, K, init_L));
    Tensor mB_nkl = params.base.tma_load_b.get_tma_tensor(cute::make_shape(N, K, init_L));
    Tensor mAux_nkl = params.aux.tma_load_aux.get_tma_tensor(cute::make_shape(N, K, init_L));
    
    InternalLayoutSFA layout_SFA{};
    InternalLayoutSFB layout_SFB{};
    InternalLayoutSFAux layout_SFAux{};
    if constexpr (Base::IsGroupedGemmKernel) {
      layout_SFA = params.base.layout_SFA[0];
      layout_SFB = params.base.layout_SFB[0];
      layout_SFAux = params.aux.layout_SFAux[0];
    } else {
      layout_SFA = params.base.layout_SFA;
      layout_SFB = params.base.layout_SFB;
      layout_SFAux = params.aux.layout_SFAux;
    }
    
    Tensor mSFA_mkl = params.base.tma_load_sfa.get_tma_tensor(cute::shape(layout_SFA));
    Tensor mSFB_nkl = params.base.tma_load_sfb.get_tma_tensor(cute::shape(layout_SFB));
    Tensor mSFAux_nkl = params.aux.tma_load_sfaux.get_tma_tensor(cute::shape(layout_SFAux));
    
    Tensor gA_mkl = cute::local_tile(mA_mkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<cute::_1, X, cute::_1>{});
    Tensor gB_nkl = cute::local_tile(mB_nkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<X, cute::_1, cute::_1>{});
    Tensor gAux_nkl = cute::local_tile(mAux_nkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<X, cute::_1, cute::_1>{});
    
    Tensor gSFA_mkl = cute::local_tile(mSFA_mkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<cute::_1, X, cute::_1>{});
    Tensor gSFB_nkl = cute::local_tile(mSFB_nkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<X, cute::_1, cute::_1>{});
    Tensor gSFAux_nkl = cute::local_tile(mSFAux_nkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<X, cute::_1, cute::_1>{});
    
    return cute::make_tuple(gA_mkl, gB_nkl, gAux_nkl, gSFA_mkl, gSFB_nkl, gSFAux_nkl);
  }

  //
  // load - TWO-PHASE loading with sequential SMEM reuse
  //
  // The kernel provides k_tile_count = 2 * real_k_tile_count.
  // Phase 1 (k_real stages): Load A + B_linear + SFA + SFB into smem_A/smem_B/smem_SFA/smem_SFB
  // Phase 2 (k_real stages): Load A + B_gate + SFA + SFAux into smem_A/smem_B/smem_SFA/smem_SFB
  //
  // Both phases load into the SAME 4 SMEM arrays. This is safe because the pipeline
  // ensures the consumer finishes reading before the producer overwrites.
  //
  template <class LoadTuple, class TensorMapTuple, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void
  load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      LoadTuple const& load_inputs,
      TensorMapTuple const& input_tensormaps,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      // Create SMEM tensors - only 4 arrays
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
      Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
      Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

      // Extract from 6-tuple: (A, B, Aux, SFA, SFB, SFAux)
      auto gA_mkl = get<0>(load_inputs);
      auto gB_nkl = get<1>(load_inputs);
      auto gAux_nkl = get<2>(load_inputs);
      auto gSFA_mkl = get<3>(load_inputs);
      auto gSFB_nkl = get<4>(load_inputs);
      auto gSFAux_nkl = get<5>(load_inputs);
      
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      // Store expert index and tile N offset in shared memory for mma() to read
      shared_tensors.expert_idx = l_coord;
      shared_tensors.n_tile_offset = n_coord * cute::size<1>(TileShape{});

      // TMA slices for base (A, B, SFA, SFB)
      auto block_tma_a = params.base.tma_load_a.get_slice(0);
      auto block_tma_b = params.base.tma_load_b.get_slice(0);
      auto block_tma_sfa = params.base.tma_load_sfa.get_slice(0);
      auto block_tma_sfb = params.base.tma_load_sfb.get_slice(0);

      // TMA slices for aux (gate weights)
      auto block_tma_aux = params.aux.tma_load_aux.get_slice(0);
      auto block_tma_sfaux = params.aux.tma_load_sfaux.get_slice(0);

      // Partition gmem
      Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);
      Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);
      Tensor gAux = gAux_nkl(_, _, n_coord, _, l_coord);
      Tensor gSFA = gSFA_mkl(_, _, m_coord, _, l_coord);
      Tensor gSFB = gSFB_nkl(_, _, n_coord, _, l_coord);
      Tensor gSFAux = gSFAux_nkl(_, _, n_coord, _, l_coord);

      // Partition for TMA - base operands
      Tensor tAgA = block_tma_a.partition_S(gA);
      Tensor tAsA = block_tma_a.partition_D(sA);
      Tensor tBgB = block_tma_b.partition_S(gB);
      Tensor tBsB = block_tma_b.partition_D(sB);
      Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);
      Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);
      Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);
      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);

      // Partition for TMA - aux operands (source from gate weights, destination into smem_B/smem_SFB!)
      Tensor tAuxgAux = block_tma_aux.partition_S(gAux);
      Tensor tAuxsB = block_tma_aux.partition_D(sB);     // Gate data goes into same smem_B
      Tensor tAuxgSFAux = block_tma_sfaux.partition_S(gSFAux);
      Tensor tAuxsSFB = block_tma_sfaux.partition_D(sSFB); // Gate SF goes into same smem_SFB

      // Two-phase load in a single loop.  The kernel provides k_tile_count = 2 * k_real.
      // Phase 1 (first k_real iters): load A + B_linear + SFA + SFB_linear
      // Phase 2 (last  k_real iters): load A + B_gate  + SFA + SFAux
      //
      // ForwardCoordIterator does NOT wrap modulo shape — it increments past end.
      // We explicitly reset the coord at the phase boundary so phase 2 re-reads
      // A/SFA from the same K positions and loads gate B/SFAux at matching coords.
      // NOTE: These invariants are guaranteed by the kernel's k_tile_count doubling
      // logic in sm90_gemm_array_tma_warpspecialized_pingpong.hpp.  The checks are
      // compiled out in release (NDEBUG) to avoid device-side trap instructions.
#ifndef NDEBUG
      if (!(k_tile_count % 2 == 0 && k_tile_count > 0)) { return; }
#endif
      int k_real = k_tile_count / 2;

      // Save starting coord so we can reset the iterator for phase 2.
      auto k_start_coord = *k_tile_iter;

      CUTLASS_PRAGMA_NO_UNROLL
      for (int k = 0; k < k_tile_count; ++k) {
        // Reset K iterator at phase boundary: phase 2 re-reads the same K coords
        if (k == k_real) {
          k_tile_iter.coord = k_start_coord;
        }

        pipeline.producer_acquire(smem_pipe_write);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        int write_stage = smem_pipe_write.index();

        // A + SFA loaded every iteration (same for both phases)
        copy(params.base.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier), 
             tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
        copy(params.base.tma_load_sfa.with(get<2>(input_tensormaps), *tma_barrier),
             tAgSFA(_, _, _, *k_tile_iter), tAsSFA(_, _, _, write_stage));
        
        if (k < k_real) {
          // Phase 1: Load B_linear + SFB_linear into smem_B / smem_SFB
          copy(params.base.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier),
               tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
          copy(params.base.tma_load_sfb.with(get<3>(input_tensormaps), *tma_barrier),
               tBgSFB(_, _, _, *k_tile_iter), tBsSFB(_, _, _, write_stage));
        } else {
          // Phase 2: Load B_gate + SFAux INTO same smem_B / smem_SFB
          copy(params.aux.tma_load_aux.with(get<4>(input_tensormaps), *tma_barrier),
               tAuxgAux(_, _, _, *k_tile_iter), tAuxsB(_, _, _, write_stage));
          copy(params.aux.tma_load_sfaux.with(get<5>(input_tensormaps), *tma_barrier),
               tAuxgSFAux(_, _, _, *k_tile_iter), tAuxsSFB(_, _, _, write_stage));
        }

        ++k_tile_iter;
        ++smem_pipe_write;
      }
    }
    __syncwarp();
  }

  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  //
  // mma - TWO-PHASE compute with inline SwiGLU (single accumulator interface)
  //
  // The kernel provides k_tile_count = 2 * real_k_tile_count.
  // Phase 1 (k_real stages): accum = A @ B_linear  (linear GEMM)
  // Phase 2 (k_real stages): accum_gate = A @ B_gate (gate GEMM, B_gate was loaded into smem_B)
  // Then: accum = SiLU(accum_gate) * accum  (SwiGLU in registers)
  //
  // This provides a single-accumulator output, compatible with standard GemmUniversal epilogue.
  //
  template <class FrgTensorC>
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& params) {

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    // SMEM tensors - only 4 arrays
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));

    Base base_helper;
    Tensor tCrSFA = base_helper.partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
    Tensor tCrSFB = base_helper.partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(
      as_position_independent_swizzle_tensor(sA));
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(
      as_position_independent_swizzle_tensor(sB));
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);

    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                    base_helper.get_layoutSFA_TV(tiled_mma),
                                                    make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFA   = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA            = smem_thr_copy_SFA.partition_S(
        as_position_independent_swizzle_tensor(sSFA));
    Tensor tCrSFA_copy_view  = smem_thr_copy_SFA.retile_D(tCrSFA);

    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                    base_helper.get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFB   = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB            = smem_thr_copy_SFB.partition_S(
      as_position_independent_swizzle_tensor(sSFB));
    Tensor tCrSFB_copy_view  = smem_thr_copy_SFB.retile_D(tCrSFB);

    auto K_BLOCK_MAX = size<2>(tCrA);

    // Split k_tile_count into two phases.
    // Same invariant as load() — guaranteed by kernel k_tile_count doubling.
#ifndef NDEBUG
    if (!(k_tile_count % 2 == 0 && k_tile_count > 0)) { return; }
#endif
    int k_real = k_tile_count / 2;

    // Helper lambdas for copy and gemm (reused by both phases)
    auto update_stage_views = [&](int read_stage, auto& tCsA_stage, auto& tCsB_stage,
                                   auto& tCsSFA_stage, auto& tCsSFB_stage) {
      tCsA_stage = tCsA(_,_,_,read_stage);
      tCsB_stage = tCsB(_,_,_,read_stage);
      tCsSFA_stage = tCsSFA(_,_,_,read_stage);
      tCsSFB_stage = tCsSFB(_,_,_,read_stage);
    };

    // Run one phase of the pipelined MMA loop over k_phase_count tiles
    auto run_mma_phase = [&](int k_phase_count, FrgTensorC& phase_accum) {
      int read_stage = smem_pipe_read.index();
      auto tCsA_stage = tCsA(_,_,_,read_stage);
      auto tCsB_stage = tCsB(_,_,_,read_stage);
      auto tCsSFA_stage = tCsSFA(_,_,_,read_stage);
      auto tCsSFB_stage = tCsSFB(_,_,_,read_stage);

      auto copy_kblock = [&](auto k_block) {
        copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
        copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));

        using MMAOp = typename TiledMma::MMA_Op;
        fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
        fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));

        copy(tCsSFA_stage(_,_,k_block), tCrSFA_copy_view(_,_,k_block));
        copy(tCsSFB_stage(_,_,k_block), tCrSFB_copy_view(_,_,k_block));
      };

      auto gemm_kblock = [&](auto k_block) {
        cute::gemm(tiled_mma,
                   make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block)),
                   make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)),
                   phase_accum);
      };

      pipeline.consumer_wait(smem_pipe_read);
      copy_kblock(_0{});

      CUTLASS_PRAGMA_NO_UNROLL
      for (int k = k_phase_count; k > 1; --k) {
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
          auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

          if (k_block == K_BLOCK_MAX - 1) {
            cutlass::arch::NamedBarrier::sync(
              thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
            pipeline.consumer_release(smem_pipe_read);
            ++smem_pipe_read;
            read_stage = smem_pipe_read.index();
            update_stage_views(read_stage, tCsA_stage, tCsB_stage, tCsSFA_stage, tCsSFB_stage);
            pipeline.consumer_wait(smem_pipe_read);
          }

          copy_kblock(k_block_next);
          gemm_kblock(k_block);
        });
      }

      // Last k_tile
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
            thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
        }

        if (k_block_next > 0) {
          copy_kblock(k_block_next);
        }
        gemm_kblock(k_block);
      });
    };

    // ================================================================
    // PHASE 1: Linear GEMM → accum
    // ================================================================
    run_mma_phase(k_real, accum);

    // ================================================================
    // PHASE 2: Gate GEMM → accum_gate
    // ================================================================
    // smem_B/smem_SFB now contain gate weights (loaded by phase 2 of load())
    FrgTensorC accum_gate;
    clear(accum_gate);
    run_mma_phase(k_real, accum_gate);

    // ================================================================
    // FC1 bias + per-expert SwiGLU: matches doActivationKernel in unfused path
    // ================================================================
    // Bias is added to both accumulators BEFORE SwiGLU (same order as unfused):
    //   linear += bias_linear[n], gate += bias_gate[n]
    // Then SwiGLU:
    //   gate_clamped = min(gate, limit)
    //   linear_clamped = clamp(linear, -limit, limit)
    //   output = gate_clamped * sigmoid(gate_clamped * alpha) * (linear_clamped + beta)
    //
    // With nullptr params (defaults: alpha=1, beta=0, limit=inf, no bias) this
    // reduces to standard SwiGLU: gate * sigmoid(gate) * linear.
    int const eidx = shared_tensors.expert_idx;
    float const sw_alpha = params.swiglu_alpha ? params.swiglu_alpha[eidx] : 1.0f;
    float const sw_beta  = params.swiglu_beta  ? params.swiglu_beta[eidx]  : 0.0f;
    float const sw_limit = params.swiglu_limit ? params.swiglu_limit[eidx] : 1e30f;

    // FC1 bias: fragment-to-N coordinate mapping (compile-time resolved by CuTe).
    // bias layout per expert: [bias_linear(inter_size) | bias_gate(inter_size)]
    auto const* bias_bf16 = reinterpret_cast<cutlass::bfloat16_t const*>(params.fc1_bias);
    int64_t const bias_inter_sz = params.bias_inter_size;
    int const n_offset = shared_tensors.n_tile_offset;
    // Identity tensor maps fragment element index → (M, N) tile coordinate
    auto cD = make_identity_tensor(make_shape(size<0>(TileShape{}), size<1>(TileShape{})));
    auto tCcD = thread_mma.partition_C(cD);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      float gate = float(accum_gate(i));
      float linear = float(accum(i));

      // Add FC1 bias before SwiGLU (matches unfused doActivationKernel order)
      if (bias_bf16) {
        int n_global = n_offset + get<1>(tCcD(i));
        linear += float(bias_bf16[eidx * 2 * bias_inter_sz + n_global]);
        gate   += float(bias_bf16[eidx * 2 * bias_inter_sz + bias_inter_sz + n_global]);
      }

      float gate_clamped = fminf(gate, sw_limit);
      float linear_clamped = fmaxf(fminf(linear, sw_limit), -sw_limit);
      float sigmoid_gate = 1.0f / (1.0f + expf(-gate_clamped * sw_alpha));
      accum(i) = gate_clamped * sigmoid_gate * (linear_clamped + sw_beta);
    }
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
