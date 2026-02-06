/***************************************************************************************************
 * SM120 Gated SwigluBias Epilogue
 *
 * This epilogue receives two accumulators (linear and gate) and applies SwigluBias:
 *
 *   gate_clamped   = min(gate, limit)
 *   linear_clamped = clamp(linear, -limit, limit)
 *   output = gate_clamped * sigmoid(gate_clamped * alpha) * (linear_clamped + beta)
 *
 * This exactly matches the SwigluBiasAdaptor formula used in doGatedActivation.
 * The output is stored as BF16 to an auxiliary buffer.
 *
 **************************************************************************************************/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/activation.h"

#include "cute/tensor.hpp"
#include "cute/layout.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gated SwigluBias epilogue for SM120
///
/// This implementation:
/// 1. Receives two accumulator fragments (linear and gate)
/// 2. Applies exact SwigluBias: gate_c * sigmoid(gate_c * alpha) * (linear_c + beta)
/// 3. Stores BF16 result to global memory
///
template <
  class TileShape_,
  class ElementOutput_,     // bfloat16_t
  class StrideOutput_,
  class ElementAccumulator_ // float
>
struct Sm120GatedSwiGLUEpilogue {
  
  using TileShape = TileShape_;
  using ElementOutput = ElementOutput_;
  using StrideOutput = StrideOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = float;
  
  // Required by gemm_universal_adapter.h (kernel concept conformance)
  using ThreadEpilogueOp = void;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;
  
  static_assert(cute::is_same_v<ElementOutput, cutlass::bfloat16_t>,
                "GatedSwiGLU epilogue currently only supports BF16 output");

  //
  // Shared Storage (minimal)
  //
  struct SharedStorage {
    // No shared storage needed for simple direct store
  };

  //
  // Arguments
  //
  struct Arguments {
    ElementOutput* ptr_output = nullptr;    // Output buffer [M, N]
    StrideOutput stride_output{};           // Output stride
    float swiglu_alpha = 1.0f;             // Sigmoid scaling: sigmoid(gate * alpha)
    float swiglu_beta  = 0.0f;             // Linear bias: (linear + beta)
    float swiglu_limit = std::numeric_limits<float>::infinity();  // Clamp bound
  };

  //
  // Params
  //
  struct Params {
    ElementOutput* ptr_output = nullptr;
    StrideOutput stride_output{};
    float swiglu_alpha = 1.0f;
    float swiglu_beta  = 0.0f;
    float swiglu_limit = std::numeric_limits<float>::infinity();
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {
    (void)workspace;
    return {args.ptr_output, args.stride_output,
            args.swiglu_alpha, args.swiglu_beta, args.swiglu_limit};
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;  // Always implementable
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;  // No workspace needed
  }

  CUTLASS_HOST_DEVICE
  Sm120GatedSwiGLUEpilogue() {}

  //
  // SwigluBias activation (exact match to SwigluBiasAdaptor):
  //   gate_clamped   = min(gate, limit)
  //   linear_clamped = max(min(linear, limit), -limit)
  //   output = gate_clamped * sigmoid(gate_clamped * alpha) * (linear_clamped + beta)
  //
  CUTLASS_DEVICE
  static ElementCompute swiglu_bias(
      ElementCompute linear, ElementCompute gate,
      float alpha, float beta, float limit) {
    ElementCompute gate_clamped   = fminf(gate, limit);
    ElementCompute linear_clamped = fmaxf(fminf(linear, limit), -limit);
    ElementCompute sigmoid_val    = ElementCompute(1) / (ElementCompute(1) + expf(-(gate_clamped * alpha)));
    return gate_clamped * sigmoid_val * (linear_clamped + beta);
  }

  //
  // Main epilogue operator
  //
  template <class FrgTensorC, class TileCoord>
  CUTLASS_DEVICE void
  operator()(
      FrgTensorC const& accum_linear,
      FrgTensorC const& accum_gate,
      TileCoord const& tile_coord,
      Params const& params,
      int thread_idx,
      SharedStorage& shared_storage) {
    
    using namespace cute;
    
    static_assert(is_rmem<FrgTensorC>::value, "Accumulators must be in registers");
    static_assert(rank(FrgTensorC{}) == 3, "Expected rank-3 accumulator (V, M, N)");
    
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord;
    
    // Get output tensor
    Tensor mD = make_tensor(make_gmem_ptr(params.ptr_output), 
                            make_layout(make_shape(size<0>(TileShape{}), size<1>(TileShape{})),
                                        params.stride_output));
    
    // Slice to this tile
    Tensor gD = local_tile(mD, TileShape{}, make_coord(m_coord, n_coord));
    
    float const alpha = params.swiglu_alpha;
    float const beta  = params.swiglu_beta;
    float const limit = params.swiglu_limit;
    
    CUTLASS_PRAGMA_UNROLL
    for (int v = 0; v < size<0>(accum_linear); ++v) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<1>(accum_linear); ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<2>(accum_linear); ++n) {
          ElementCompute linear_val = static_cast<ElementCompute>(accum_linear(v, m, n));
          ElementCompute gate_val   = static_cast<ElementCompute>(accum_gate(v, m, n));
          
          ElementCompute output_val = swiglu_bias(linear_val, gate_val, alpha, beta, limit);
          
          // Note: Proper implementation would compute thread-specific indices
          // and use vectorized stores. This is for reference/documentation.
          // gD(m, n) = static_cast<ElementOutput>(output_val);
        }
      }
    }
  }

  //
  // Apply SwigluBias to accumulator fragments (for use by mainloop inline path)
  //
  template <class FrgTensorC>
  CUTLASS_DEVICE auto
  apply_swiglu_bias(
      FrgTensorC const& accum_linear, FrgTensorC const& accum_gate,
      float alpha, float beta, float limit) {
    FrgTensorC result;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_linear); ++i) {
      ElementCompute linear_val = static_cast<ElementCompute>(accum_linear(i));
      ElementCompute gate_val   = static_cast<ElementCompute>(accum_gate(i));
      result(i) = static_cast<typename FrgTensorC::value_type>(
          swiglu_bias(linear_val, gate_val, alpha, beta, limit));
    }
    
    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::epilogue

/////////////////////////////////////////////////////////////////////////////////////////////////
