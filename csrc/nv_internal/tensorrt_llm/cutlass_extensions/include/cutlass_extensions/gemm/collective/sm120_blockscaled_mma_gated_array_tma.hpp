/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Gated (dual-accumulator) extension of SM120 block-scaled MMA.
 *
 * Key design decisions (following TRT-LLM SM90 pattern):
 *   1. LoadState struct for load_init() return (not a tuple) - decouples kernel/mainloop ABI
 *   2. Params uses composition (base + aux) - trivially copyable, no inheritance surprises
 *   3. Dual accumulator mma() with inline SiLU - standard epilogue can be used
 *   4. Static asserts for compile-time validation
 *
 **************************************************************************************************/

#pragma once

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

/// Gated Collective MMA for SM120 Block-Scaled Array TMA
///
/// Key patterns (from user guidance):
///   1. LoadState struct - avoids tuple-size coupling with kernel
///   2. Params composition - base + aux, trivially copyable
///   3. Dual accumulator mma() - standard for gated
///   4. Single accumulator mma() - applies SiLU inline, allows standard epilogue
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

  // Gated-specific: Aux uses same types as B
  using ElementAux = ElementB;
  using StrideAux = StrideB;
  using InternalStrideAux = InternalStrideB;
  using LayoutSFAux = LayoutSFB;
  using InternalLayoutSFAux = InternalLayoutSFB;
  using SmemLayoutAux = SmemLayoutB;
  using SmemLayoutSFAux = SmemLayoutSFB;
  using SmemCopyAtomAux = SmemCopyAtomB;
  using SmemCopyAtomSFAux = SmemCopyAtomSFB;

  static constexpr bool IsGated = true;

  //
  // Extended SharedStorage - base storage + Aux operand
  //
  struct TensorStorage : cute::aligned_struct<128, _0> {
    // Base operands (same as Base::TensorStorage)
    cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>, 128> smem_A;
    cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>, 128> smem_B;
    cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFA>, 128> smem_SFA;
    cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFB>, 128> smem_SFB;
    
    // Gated extension: Aux operand (gate weights)
    cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutAux>, 128> smem_Aux;
    cute::array_aligned<ElementSF, cute::cosize_v<SmemLayoutSFAux>, 128> smem_SFAux;
  };

  using SharedStorage = TensorStorage;

  // ==========================================================================
  // SMEM Diagnostic - prints breakdown of shared memory usage
  // ==========================================================================
  static void print_smem_breakdown() {
    // Individual array sizes (element count * element size)
    constexpr size_t smem_A_elements = cute::cosize_v<SmemLayoutA>;
    constexpr size_t smem_B_elements = cute::cosize_v<SmemLayoutB>;
    constexpr size_t smem_Aux_elements = cute::cosize_v<SmemLayoutAux>;
    constexpr size_t smem_SFA_elements = cute::cosize_v<SmemLayoutSFA>;
    constexpr size_t smem_SFB_elements = cute::cosize_v<SmemLayoutSFB>;
    constexpr size_t smem_SFAux_elements = cute::cosize_v<SmemLayoutSFAux>;
    
    constexpr size_t sizeof_A = sizeof(typename TiledMma::ValTypeA);
    constexpr size_t sizeof_B = sizeof(typename TiledMma::ValTypeB);
    constexpr size_t sizeof_SF = sizeof(ElementSF);
    
    constexpr size_t smem_A_bytes = smem_A_elements * sizeof_A;
    constexpr size_t smem_B_bytes = smem_B_elements * sizeof_B;
    constexpr size_t smem_Aux_bytes = smem_Aux_elements * sizeof_B;
    constexpr size_t smem_SFA_bytes = smem_SFA_elements * sizeof_SF;
    constexpr size_t smem_SFB_bytes = smem_SFB_elements * sizeof_SF;
    constexpr size_t smem_SFAux_bytes = smem_SFAux_elements * sizeof_SF;
    
    // Raw total (without alignment)
    constexpr size_t raw_total = smem_A_bytes + smem_B_bytes + smem_Aux_bytes +
                                 smem_SFA_bytes + smem_SFB_bytes + smem_SFAux_bytes;
    
    // Actual struct size (with alignment)
    constexpr size_t tensor_storage_size = sizeof(TensorStorage);
    constexpr size_t alignment_overhead = tensor_storage_size - raw_total;
    
    printf("\n");
    printf("====================================================================\n");
    printf("GATED FC1 SMEM BREAKDOWN\n");
    printf("====================================================================\n");
    printf("Component       | Elements    | Elem Size | Bytes      | Type\n");
    printf("-----------------------------------------------------------------\n");
    printf("smem_A          | %10zu  | %4zu B    | %8zu B | FP8 activations\n", 
           smem_A_elements, sizeof_A, smem_A_bytes);
    printf("smem_B          | %10zu  | %4zu B    | %8zu B | FP4 linear weights\n",
           smem_B_elements, sizeof_B, smem_B_bytes);
    printf("smem_Aux        | %10zu  | %4zu B    | %8zu B | FP4 gate weights\n",
           smem_Aux_elements, sizeof_B, smem_Aux_bytes);
    printf("smem_SFA        | %10zu  | %4zu B    | %8zu B | E8M0 scale factors\n",
           smem_SFA_elements, sizeof_SF, smem_SFA_bytes);
    printf("smem_SFB        | %10zu  | %4zu B    | %8zu B | E8M0 scale factors\n",
           smem_SFB_elements, sizeof_SF, smem_SFB_bytes);
    printf("smem_SFAux      | %10zu  | %4zu B    | %8zu B | E8M0 scale factors\n",
           smem_SFAux_elements, sizeof_SF, smem_SFAux_bytes);
    printf("-----------------------------------------------------------------\n");
    printf("Raw total       |             |           | %8zu B |\n", raw_total);
    printf("Alignment pad   |             |           | %8zu B | (128-byte aligned)\n", alignment_overhead);
    printf("TensorStorage   |             |           | %8zu B | sizeof(TensorStorage)\n", tensor_storage_size);
    printf("====================================================================\n");
    printf("\n");
    printf("SmemLayout shapes:\n");
    printf("  SmemLayoutA:    cosize=%zu\n", smem_A_elements);
    printf("  SmemLayoutB:    cosize=%zu\n", smem_B_elements);
    printf("  SmemLayoutAux:  cosize=%zu (same as B)\n", smem_Aux_elements);
    printf("  SmemLayoutSFA:  cosize=%zu\n", smem_SFA_elements);
    printf("  SmemLayoutSFB:  cosize=%zu\n", smem_SFB_elements);
    printf("  SmemLayoutSFAux: cosize=%zu (same as SFB)\n", smem_SFAux_elements);
    printf("====================================================================\n");
  }

  // Pipeline storage (from base)
  using PipelineStorage = typename Base::PipelineStorage;
  
  // TensorMap storage - 6 tensormaps for gated GEMM
  // A, B, SFA, SFB (same as base) + Aux, SFAux (gated extension)
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
    // Base arguments
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
  };

  //
  // AuxParams - aux-specific device params (separated for composition)
  // Note: Uses StrideAux/LayoutSFAux (public types) to match Base::Params
  //
  struct AuxParams {
    typename Base::Params::TMA_B tma_load_aux;
    typename Base::Params::TMA_SFB tma_load_sfaux;
    ElementAux const** ptr_Aux;
    StrideAux dAux;             // Public stride type (matches Base::Params::dB)
    ElementSF const** ptr_SFAux;
    LayoutSFAux layout_SFAux;   // Public layout type (matches Base::Params::layout_SFB)
  };

  //
  // Params - COMPOSITION pattern (not inheritance!)
  // This ensures trivial copyability and avoids assignment surprises.
  // Note: We mirror key Base::Params members for kernel compatibility.
  //
  struct Params {
    typename Base::Params base;  // Base params for A, B, SFA, SFB
    AuxParams aux;               // Aux params for gate weights
    
    // Mirrored members from Base::Params for kernel compatibility
    // The kernel accesses params.mainloop.tma_transaction_bytes directly
    uint32_t tma_transaction_bytes = Base::TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = Base::TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = Base::TmaTransactionBytesNK;
  };

  // Static assert: Params must be trivially copyable (passed by value to kernel)
  static_assert(std::is_trivially_copyable_v<Params>, 
                "Gated mainloop Params must be trivially copyable");

  //
  // LoadState - encapsulates all load tensors
  //
  // Note: For compatibility with GemmUniversal which expects tuple-like load_init() return,
  // we return a tuple from load_init(). LoadState is used internally for documentation.
  // The tuple is: (gA, gB, gAux, gSFA, gSFB, gSFAux) - 6 elements
  //
  // The kernel accesses get<0>, get<1> for A, B which works.
  // Our load() method extracts all 6 elements.
  //

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
    
    // Get base params (includes TMA setup)
    auto base_params = Base::to_underlying_arguments(problem_shape, base_args, workspace);
    
    // Build aux TMA using same pattern as B (Aux has same layout as B)
    typename Base::Arguments aux_as_b_args{
      args.ptr_A, args.dA,           // A stays the same
      args.ptr_Aux, args.dAux,       // B is replaced with Aux
      args.ptr_SFA, args.layout_SFA, // SFA stays the same
      args.ptr_SFAux, args.layout_SFAux  // SFB is replaced with SFAux
    };
    auto aux_params = Base::to_underlying_arguments(problem_shape, aux_as_b_args, workspace);
    
    // Build AuxParams (use member-by-member assignment for complex TMA types)
    AuxParams aux;
    aux.tma_load_aux = aux_params.tma_load_b;     // TMA for Aux (using B slot)
    aux.tma_load_sfaux = aux_params.tma_load_sfb; // TMA for SFAux (using SFB slot)
    aux.ptr_Aux = reinterpret_cast<ElementAux const**>(args.ptr_Aux);
    aux.dAux = aux_params.dB;                     // dB slot was populated with aux stride
    aux.ptr_SFAux = reinterpret_cast<ElementSF const**>(args.ptr_SFAux);
    aux.layout_SFAux = aux_params.layout_SFB;     // layout_SFB slot was populated with aux layout
    
    // Build composed Params
    Params result;
    result.base = base_params;
    result.aux = aux;
    return result;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    // 6 tensormaps instead of 4 (add Aux and SFAux)
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
    // Delegate to base implementation
    typename Base::Arguments base_args{
      args.ptr_A, args.dA, args.ptr_B, args.dB,
      args.ptr_SFA, args.layout_SFA, args.ptr_SFB, args.layout_SFB
    };
    return Base::can_implement(problem_shapes, base_args);
  }

  //
  // Tensormap methods - initialize 6 tensormaps (A, B, SFA, SFB, Aux, SFAux)
  // Note: These are non-const to match Base's signatures
  //
  CUTLASS_DEVICE auto
  tensormaps_init(Params const& params, TensorMapStorage& shared_tensormaps, 
                  int32_t sm_count, int32_t sm_idx) {
    // Get global memory tensormap array (6 tensormaps per SM)
    cute::TmaDescriptor* gmem_tensormap = params.base.tensormaps;
    
    // Compute pointers for each tensormap in global memory
    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];
    cute::TmaDescriptor* tma_desc_sfa = &gmem_tensormap[sm_idx + 2 * sm_count];
    cute::TmaDescriptor* tma_desc_sfb = &gmem_tensormap[sm_idx + 3 * sm_count];
    cute::TmaDescriptor* tma_desc_aux = &gmem_tensormap[sm_idx + 4 * sm_count];
    cute::TmaDescriptor* tma_desc_sfaux = &gmem_tensormap[sm_idx + 5 * sm_count];
    
    if (cute::elect_one_sync()) {
      // Copy TMA descriptors from params to shared memory
      // A, B, SFA, SFB (from base params)
      Tensor pA_tensormap = make_tensor(params.base.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
      Tensor pB_tensormap = make_tensor(params.base.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});
      Tensor pSFA_tensormap = make_tensor(params.base.tma_load_sfa.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sSFA_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_SFA), Int<1>{}, Int<1>{});
      Tensor pSFB_tensormap = make_tensor(params.base.tma_load_sfb.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sSFB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_SFB), Int<1>{}, Int<1>{});
      
      // Aux, SFAux (from aux params)
      Tensor pAux_tensormap = make_tensor(params.aux.tma_load_aux.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sAux_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_Aux), Int<1>{}, Int<1>{});
      Tensor pSFAux_tensormap = make_tensor(params.aux.tma_load_sfaux.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sSFAux_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_SFAux), Int<1>{}, Int<1>{});
      
      // Copy all 6 tensormaps to shared memory
      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
      copy(recast<uint128_t>(pSFA_tensormap), recast<uint128_t>(sSFA_tensormap));
      copy(recast<uint128_t>(pSFB_tensormap), recast<uint128_t>(sSFB_tensormap));
      copy(recast<uint128_t>(pAux_tensormap), recast<uint128_t>(sAux_tensormap));
      copy(recast<uint128_t>(pSFAux_tensormap), recast<uint128_t>(sSFAux_tensormap));
    }
    __syncwarp();
    
    // Return 6-tuple of global memory tensormap pointers
    return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_sfa, tma_desc_sfb, tma_desc_aux, tma_desc_sfaux);
  }

  template <class TensorMapTuple, class ProblemShape>
  CUTLASS_DEVICE void
  tensormaps_perform_update(TensorMapStorage& shared_tensormaps, Params const& params,
                            TensorMapTuple const& input_tensormaps,
                            ProblemShape const& problem_shape, int32_t next_batch) {
    // Replace global addresses for all 6 tensormaps for the next batch
    // A, B, SFA, SFB (from base params)
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                    params.base.ptr_A[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    params.base.ptr_B[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFA,
                                                    params.base.ptr_SFA[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFB,
                                                    params.base.ptr_SFB[next_batch]);
    // Aux, SFAux (from aux params)
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_Aux,
                                                    params.aux.ptr_Aux[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_SFAux,
                                                    params.aux.ptr_SFAux[next_batch]);
    
    // For grouped GEMM, also update tensor dimensions and strides for all 6 tensormaps
    if constexpr (Base::IsGroupedGemmKernel) {
      tensormaps_replace_global_tensor_properties(shared_tensormaps, params, next_batch, problem_shape);
    }
  }
  
  // Update TMA descriptor dimensions and strides for grouped GEMM
  // This is critical for correct TMA access when expert problem shapes vary.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    
    constexpr int MaxTensorRank = 5;
    auto [M, N, K, L] = problem_shape_mnkl;
    
    // Arrays for TMA shape/stride (A, SFA, B, SFB, Aux, SFAux)
    cute::array<uint32_t, MaxTensorRank> prob_shape_A    = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A   = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_SFA  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_SFA = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B    = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B   = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_SFB  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_SFB = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_Aux    = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_Aux   = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_SFAux  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_SFAux = {0,0,0,0,0};
    
    // Construct tensors with proper shapes/strides for the next group
    using TmaInternalElementA = typename Base::TmaInternalElementA;
    using TmaInternalElementB = typename Base::TmaInternalElementB;
    
    TmaInternalElementA const* ptr_A = nullptr;
    Tensor tensor_a = make_tensor(ptr_A, make_shape(M, K, Int<1>{}), params.base.dA[next_group]);
    
    ElementSF const* ptr_SF = nullptr;
    Tensor tensor_sfa = make_tensor(ptr_SF, params.base.layout_SFA[next_group]);
    
    TmaInternalElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(ptr_B, make_shape(N, K, Int<1>{}), params.base.dB[next_group]);
    Tensor tensor_sfb = make_tensor(ptr_SF, params.base.layout_SFB[next_group]);
    
    // Aux uses same shape/stride as B (gate weights = same layout as linear weights)
    Tensor tensor_aux = make_tensor(ptr_B, make_shape(N, K, Int<1>{}), params.aux.dAux[next_group]);
    Tensor tensor_sfaux = make_tensor(ptr_SF, params.aux.layout_SFAux[next_group]);
    
    // Fill TMA shape/stride arrays
    cute::detail::fill_tma_gmem_shape_stride(params.base.tma_load_a, tensor_a, prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(params.base.tma_load_sfa, tensor_sfa, prob_shape_SFA, prob_stride_SFA);
    cute::detail::fill_tma_gmem_shape_stride(params.base.tma_load_b, tensor_b, prob_shape_B, prob_stride_B);
    cute::detail::fill_tma_gmem_shape_stride(params.base.tma_load_sfb, tensor_sfb, prob_shape_SFB, prob_stride_SFB);
    cute::detail::fill_tma_gmem_shape_stride(params.aux.tma_load_aux, tensor_aux, prob_shape_Aux, prob_stride_Aux);
    cute::detail::fill_tma_gmem_shape_stride(params.aux.tma_load_sfaux, tensor_sfaux, prob_shape_SFAux, prob_stride_SFAux);
    
    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_A) { stride = (stride * sizeof_bits_v<TmaInternalElementA>) / 8; }
    for (uint64_t& stride : prob_stride_SFA) { stride = (stride * sizeof_bits_v<ElementSF>) / 8; }
    for (uint64_t& stride : prob_stride_B) { stride = (stride * sizeof_bits_v<TmaInternalElementB>) / 8; }
    for (uint64_t& stride : prob_stride_SFB) { stride = (stride * sizeof_bits_v<ElementSF>) / 8; }
    for (uint64_t& stride : prob_stride_Aux) { stride = (stride * sizeof_bits_v<TmaInternalElementB>) / 8; }
    for (uint64_t& stride : prob_stride_SFAux) { stride = (stride * sizeof_bits_v<ElementSF>) / 8; }
    
    // Update all 6 tensormaps
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
    // Commit any pending TMA descriptor modifications and wait for them to complete
    // This is required before copying modified descriptors from SMEM to GMEM
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    // Entire warp must do this (i.e. the instructions are aligned)
    // Copy from shared memory to global memory with fence for all 6 tensormaps
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
    // Acquire fence for all 6 tensormaps
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
    // For gated, we need to update all 6 tensors, but the base only handles 4.
    // Simplification: Only update A, B, SFA, SFB via base. Aux uses static pointers.
    // This works for grouped GEMM where Aux pointers are pre-computed in params.
    Base base_collective;
    
    // Update base tensors (A, B, SFA, SFB) via indices 0, 1, 3, 4 of our 6-tuple
    auto base_tuple = cute::make_tuple(get<0>(load_inputs), get<1>(load_inputs),
                                        get<3>(load_inputs), get<4>(load_inputs));
    auto updated_base = base_collective.tensors_perform_update(base_tuple, params.base, 
                                                                problem_shape, batch_idx);
    
    // For Aux (indices 2, 5): keep original tensors - they're updated via params
    // This is a simplification that works because load() uses params.aux directly
    return cute::make_tuple(
      get<0>(updated_base),   // A (updated)
      get<1>(updated_base),   // B (updated)
      get<2>(load_inputs),    // Aux (unchanged - uses params.aux in load())
      get<2>(updated_base),   // SFA (updated)
      get<3>(updated_base),   // SFB (updated)
      get<5>(load_inputs)     // SFAux (unchanged - uses params.aux in load())
    );
  }

  //
  // load_init - returns tuple (compatible with GemmUniversal)
  //
  // Returns: (gA, gB, gAux, gSFA, gSFB, gSFAux) - 6 elements
  // The kernel accesses get<0> (A) and get<1> (B) which works for standard code.
  // Our gated load() extracts all 6 elements for the dual GEMM.
  //
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& params) {
    using X = cute::Underscore;
    
    // Extract problem shape
    auto [M, N, K, L] = problem_shape_MNKL;
    const int32_t init_L = 1;
    
    // Get base tensors (A, B, SFA, SFB) using TMA descriptors
    Tensor mA_mkl = params.base.tma_load_a.get_tma_tensor(cute::make_shape(M, K, init_L));
    Tensor mB_nkl = params.base.tma_load_b.get_tma_tensor(cute::make_shape(N, K, init_L));
    
    // Get Aux tensor (gate weights) using its TMA descriptor
    // Aux has same shape as B (N,K,L)
    Tensor mAux_nkl = params.aux.tma_load_aux.get_tma_tensor(cute::make_shape(N, K, init_L));
    
    // Get scale factor layouts
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
    
    // Get scale factor tensors using TMA descriptors
    Tensor mSFA_mkl = params.base.tma_load_sfa.get_tma_tensor(cute::shape(layout_SFA));
    Tensor mSFB_nkl = params.base.tma_load_sfb.get_tma_tensor(cute::shape(layout_SFB));
    Tensor mSFAux_nkl = params.aux.tma_load_sfaux.get_tma_tensor(cute::shape(layout_SFAux));
    
    // Make tiled views using TileShape, defer the slice
    Tensor gA_mkl = cute::local_tile(mA_mkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<cute::_1, X, cute::_1>{});
    Tensor gB_nkl = cute::local_tile(mB_nkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<X, cute::_1, cute::_1>{});
    Tensor gAux_nkl = cute::local_tile(mAux_nkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<X, cute::_1, cute::_1>{});
    
    Tensor gSFA_mkl = cute::local_tile(mSFA_mkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<cute::_1, X, cute::_1>{});
    Tensor gSFB_nkl = cute::local_tile(mSFB_nkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<X, cute::_1, cute::_1>{});
    Tensor gSFAux_nkl = cute::local_tile(mSFAux_nkl, TileShape{}, cute::make_coord(X{}, X{}, X{}), cute::Step<X, cute::_1, cute::_1>{});
    
    return cute::make_tuple(gA_mkl, gB_nkl, gAux_nkl, gSFA_mkl, gSFB_nkl, gSFAux_nkl);
  }

  //
  // load - takes 6-tuple from load_init
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
      // Create SMEM tensors (base + Aux)
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});
      Tensor sAux = make_tensor(make_smem_ptr(shared_tensors.smem_Aux.data()), SmemLayoutAux{});
      Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.data()), SmemLayoutSFA{});
      Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.data()), SmemLayoutSFB{});
      Tensor sSFAux = make_tensor(make_smem_ptr(shared_tensors.smem_SFAux.data()), SmemLayoutSFAux{});

      // Extract from 6-tuple: (A, B, Aux, SFA, SFB, SFAux)
      auto gA_mkl = get<0>(load_inputs);
      auto gB_nkl = get<1>(load_inputs);
      auto gAux_nkl = get<2>(load_inputs);
      auto gSFA_mkl = get<3>(load_inputs);
      auto gSFB_nkl = get<4>(load_inputs);
      auto gSFAux_nkl = get<5>(load_inputs);
      
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      // Get TMA slices
      auto block_tma_a = params.base.tma_load_a.get_slice(0);
      auto block_tma_b = params.base.tma_load_b.get_slice(0);
      auto block_tma_aux = params.aux.tma_load_aux.get_slice(0);
      auto block_tma_sfa = params.base.tma_load_sfa.get_slice(0);
      auto block_tma_sfb = params.base.tma_load_sfb.get_slice(0);
      auto block_tma_sfaux = params.aux.tma_load_sfaux.get_slice(0);

      // Partition gmem
      Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);
      Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);
      Tensor gAux = gAux_nkl(_, _, n_coord, _, l_coord);
      Tensor gSFA = gSFA_mkl(_, _, m_coord, _, l_coord);
      Tensor gSFB = gSFB_nkl(_, _, n_coord, _, l_coord);
      Tensor gSFAux = gSFAux_nkl(_, _, n_coord, _, l_coord);

      // Partition for TMA
      Tensor tAgA = block_tma_a.partition_S(gA);
      Tensor tAsA = block_tma_a.partition_D(sA);
      Tensor tBgB = block_tma_b.partition_S(gB);
      Tensor tBsB = block_tma_b.partition_D(sB);
      Tensor tAuxgAux = block_tma_aux.partition_S(gAux);
      Tensor tAuxsAux = block_tma_aux.partition_D(sAux);
      Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);
      Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);
      Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);
      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);
      Tensor tAuxgSFAux = block_tma_sfaux.partition_S(gSFAux);
      Tensor tAuxsSFAux = block_tma_sfaux.partition_D(sSFAux);

      // TMA copy loop - loads A, B, Aux, and all scale factors
      CUTLASS_PRAGMA_NO_UNROLL
      for (; k_tile_count > 0; --k_tile_count) {
        pipeline.producer_acquire(smem_pipe_write);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();

        // Copy A and SFA (indices 0, 2 in tensormap tuple)
        copy(params.base.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier), 
             tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
        copy(params.base.tma_load_sfa.with(get<2>(input_tensormaps), *tma_barrier),
             tAgSFA(_, _, _, *k_tile_iter), tAsSFA(_, _, _, write_stage));
        
        // Copy B and SFB (indices 1, 3 in tensormap tuple)
        copy(params.base.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier),
             tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
        copy(params.base.tma_load_sfb.with(get<3>(input_tensormaps), *tma_barrier),
             tBgSFB(_, _, _, *k_tile_iter), tBsSFB(_, _, _, write_stage));
        
        // Copy Aux and SFAux (gated extension, indices 4, 5 in tensormap tuple)
        copy(params.aux.tma_load_aux.with(get<4>(input_tensormaps), *tma_barrier),
             tAuxgAux(_, _, _, *k_tile_iter), tAuxsAux(_, _, _, write_stage));
        copy(params.aux.tma_load_sfaux.with(get<5>(input_tensormaps), *tma_barrier),
             tAuxgSFAux(_, _, _, *k_tile_iter), tAuxsSFAux(_, _, _, write_stage));

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
  // mma - Dual accumulator pattern (for GemmUniversalGated kernel)
  //
  template <class FrgTensorC>
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum0,     // Linear accumulator (A @ B)
      FrgTensorC& accum1,     // Gate accumulator (A @ Aux)
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& params) {
    
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum0);
    clear(accum1);

    // SMEM tensors
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});
    Tensor sAux = make_tensor(make_smem_ptr(shared_tensors.smem_Aux.data()), SmemLayoutAux{});
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.data()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.data()), SmemLayoutSFB{});
    Tensor sSFAux = make_tensor(make_smem_ptr(shared_tensors.smem_SFAux.data()), SmemLayoutSFAux{});

    // Partition for MMA
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));
    Tensor tCrAux = thread_mma.partition_fragment_B(sAux(_, _, Int<0>{}));
    
    // Scale factor fragments - use base class helper functions
    // Need to call on an instance since these are non-static members
    Base base_helper;
    Tensor tCrSFA = base_helper.partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
    Tensor tCrSFB = base_helper.partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);
    Tensor tCrSFAux = base_helper.partition_fragment_SFB(sSFAux(_, _, Int<0>{}), thread_mma);

    // SMEM copy setup (same as base)
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    Tensor tCsAux = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sAux));
    Tensor tCrAux_copy_view = smem_thr_copy_B.retile_D(tCrAux);

    // Scale factor copy setup (reuse base patterns)
    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                     base_helper.get_layoutSFA_TV(tiled_mma),
                                                     make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA = smem_thr_copy_SFA.partition_S(as_position_independent_swizzle_tensor(sSFA));
    Tensor tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);

    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                     base_helper.get_layoutSFB_TV(tiled_mma),
                                                     make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB = smem_thr_copy_SFB.partition_S(as_position_independent_swizzle_tensor(sSFB));
    Tensor tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);
    Tensor tCsSFAux = smem_thr_copy_SFB.partition_S(as_position_independent_swizzle_tensor(sSFAux));
    Tensor tCrSFAux_copy_view = smem_thr_copy_SFB.retile_D(tCrSFAux);

    //
    // PIPELINED MAIN LOOP - dual accumulator pattern
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      pipeline.consumer_wait(smem_pipe_read);
      int read_stage = smem_pipe_read.index();

      // Copy A, B, Aux and scale factors from SMEM to registers
      copy(smem_tiled_copy_A, tCsA(_, _, _, read_stage), tCrA_copy_view);
      copy(smem_tiled_copy_SFA, tCsSFA(_, _, _, read_stage), tCrSFA_copy_view);
      copy(smem_tiled_copy_B, tCsB(_, _, _, read_stage), tCrB_copy_view);
      copy(smem_tiled_copy_SFB, tCsSFB(_, _, _, read_stage), tCrSFB_copy_view);
      copy(smem_tiled_copy_B, tCsAux(_, _, _, read_stage), tCrAux_copy_view);
      copy(smem_tiled_copy_SFB, tCsSFAux(_, _, _, read_stage), tCrSFAux_copy_view);

      // Dual GEMM: accum0 += A @ B, accum1 += A @ Aux
      cute::gemm(tiled_mma,
                 make_zip_tensor(tCrA, tCrSFA),
                 make_zip_tensor(tCrB, tCrSFB),
                 accum0);
      
      cute::gemm(tiled_mma,
                 make_zip_tensor(tCrA, tCrSFA),
                 make_zip_tensor(tCrAux, tCrSFAux),
                 accum1);

      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }
  }

  //
  // mma - Single accumulator pattern (for standard GemmUniversal kernel)
  // 
  // This allows using the STANDARD kernel and epilogue by applying SiLU INLINE.
  // This is the TRT-LLM pattern: SwiGLU belongs in the consumer loop, not the epilogue.
  //
  // TEMPORARY DEBUG: Simplified to only do linear GEMM (no gate) to isolate issue
  //
  template <class FrgTensorC>
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,     // Single output: already has SwiGLU applied
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& params) {
    
    // TEMPORARY: Just call base class mma() for linear only (skip gate)
    Base base_helper;
    base_helper.mma(pipeline, smem_pipe_read, accum, k_tile_count, thread_idx,
                    reinterpret_cast<typename Base::TensorStorage&>(shared_tensors), params.base);
    return;
    
    // Allocate internal accumulators for dual GEMM
    FrgTensorC accum_linear;
    FrgTensorC accum_gate;
    
    // Call the dual-accumulator version
    mma(pipeline, smem_pipe_read, accum_linear, accum_gate, 
        k_tile_count, thread_idx, shared_tensors, params);
    
    // Validate accumulator layout match (same shape for element-wise ops)
    static_assert(cute::rank(FrgTensorC{}) >= 1, "Accumulator must have at least rank 1");
    
    // Apply SwiGLU: output = linear * silu(gate)
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_linear); ++i) {
      float linear_val = float(accum_linear(i));
      float gate_val = float(accum_gate(i));
      // SiLU activation on gate
      float silu_gate = gate_val / (1.0f + expf(-gate_val));
      // SwiGLU = linear * silu(gate)
      accum(i) = typename FrgTensorC::value_type(linear_val * silu_gate);
    }
  }

  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    pipeline.consumer_release(smem_pipe_release);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
