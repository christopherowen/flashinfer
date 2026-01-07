# SM121 Cluster Shape Investigation

## Summary

This document records the investigation into why FlashInfer/CUTLASS kernels on SM121 (Blackwell Thorough) are restricted to 1×1×1 cluster shapes, and whether this is a hardware limitation or a library design choice.

**Conclusion**: The CUDA runtime successfully launches 2×1×1 clusters on SM121 hardware. The 1×1×1 restriction is a **CUTLASS library design choice**, not a hardware limitation. CUTLASS treats SM12x Thorough as lacking programmatic multicast, and therefore disables multi-CTA cluster shapes in its SM120/SM121 builders.

---

## Test Results

### Runtime Cluster Capability Test

**Test file**: `tests/sm121_runtime_cluster_test.cu`

```
Device: NVIDIA GB10
Compute Capability: 12.1
Number of SMs: 48

=== Cluster Capability Attributes ===
  Max blocks per SM: 24
  Cluster launch supported: YES

=== Testing cluster launch 2x1x1 ===
  cudaLaunchKernelEx with clusterDim=(2,1,1)...
  2x1x1 cluster kernel: cluster_dim_x = 2
  Reported cluster_dim_x from device: 2
  Result: SUCCESS - cluster launch completed!
```

**Finding**: The CUDA runtime reports cluster launch support and successfully launches a 2×1×1 cluster on GB10 (SM121). The device correctly reports `cluster_dim_x = 2`, confirming the cluster was actually scheduled.

### Compile-Time Static Assert Test

**Test file**: `tests/sm121_cluster_static_assert_test.cu`

When attempting to instantiate a CUTLASS SM120 collective builder with a 2×1×1 cluster:

```
sm120_mma_builder.inl(86): error: static assertion failed with 
    "no programmatic multicast on this arch"
    
    static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, 
                  "no programmatic multicast on this arch");
```

**Finding**: CUTLASS enforces `cluster_size == 1` at compile time in all SM120 builders:
- `sm120_mma_builder.inl`
- `sm120_blockscaled_mma_builder.inl`
- `sm120_sparse_mma_builder.inl`
- `sm120_blockscaled_sparse_mma_builder.inl`

---

## Analysis

### What We Proved

| Layer | Result |
|-------|--------|
| **Hardware/Runtime** | SM121 supports cluster launches > 1×1×1 |
| **CUTLASS** | SM120/SM121 builders block clusters > 1×1×1 via static_assert |

### Why CUTLASS Blocks Multi-CTA Clusters

CUTLASS' SM120 builders are designed around a **TMA multicast strategy**:

1. **SM100 (Blackwell Ultra)** has "programmatic TMA multicast"
   - One SM loads data via TMA and broadcasts to cluster peers
   - This enables efficient 2SM/4SM cooperative execution
   - Shared data means lower aggregate memory bandwidth

2. **SM12x Thorough** lacks programmatic TMA multicast
   - Without multicast, CUTLASS' cluster strategy would cause each CTA to load independently
   - This creates redundant TMA loads with no bandwidth benefit
   - CUTLASS prevents this inefficient pattern at compile time

### Important Nuance

Even without TMA multicast, clusters can still provide other benefits:
- Scheduling/placement locality
- Cluster-level barriers
- Shared memory across cluster (on supporting architectures)

However, CUTLASS forbids them because **its current cluster strategy** would be inefficient without multicast. Alternative kernel designs could potentially use clusters differently, but CUTLASS SM120 builders don't implement such alternatives.

---

## Implications for FlashInfer on SM121

### What This Means

CUTLASS' SM120/SM121 builders hard-restrict cluster shape to 1×1×1 via `static_assert` ("no programmatic multicast on this arch"), so multi-CTA cluster shapes are **not available** in the current CUTLASS-backed FlashInfer kernels.

**Don't spend time trying to enable clusters in CUTLASS/FlashInfer SM12x today**—you'd need a different kernel strategy than CUTLASS' multicast-oriented one.

### Focus Areas for SM121 Performance

For SM121 "native FP4 peak performance," focus on:

| Priority | Area | Rationale |
|----------|------|-----------|
| **High** | Tile shapes | Better coverage for MoE decode (small-M), avoid tails for dims like 2880 |
| **High** | SM12x block-scaled FP4 MMA | Verify SASS includes the correct instructions |
| **Medium** | Activation/scale preparation | Minimize overhead, consider identity scales |
| **Medium** | Pipelining/schedule choices | Optimize for SM121's characteristics |
| **Medium** | Heuristics | Kernel selection for various problem shapes |
| **Low** | Cluster shapes | Not available via CUTLASS SM120 builders |

---

## Performance Verification (Recommended Next Steps)

### 1. Nsight Compute Analysis

Run Nsight Compute on the best SM121 FP4 kernel to verify:

```bash
ncu --set full -o profile_sm121_fp4 \
    python -c "import flashinfer; # your kernel invocation"
```

Check for:
- [ ] Not bandwidth-bound on scale-factor loads
- [ ] Memory throughput matches expectations
- [ ] Occupancy is reasonable

### 2. SASS Instruction Verification

Verify SASS includes SM12x block-scaled FP4 MMA instructions:

```bash
cuobjdump --dump-sass <compiled_kernel.cubin> | grep -i "mma\|fp4\|scale"
```

Expected: Should see SM12x UMMA (Unified Matrix Multiply-Accumulate) instructions with block-scaled operands.

### 3. Tile Coverage Analysis

For MoE workloads, verify tile shapes cover:
- [ ] Small-M decode (M=64, M=128 for per-expert batches)
- [ ] N dimensions that divide common model dims (192 for 2880, 256 for power-of-2)
- [ ] K dimensions appropriate for FP4 (K=128, K=256)

---

## Test Files

| File | Purpose |
|------|---------|
| `tests/sm121_cluster_test.cu` | Basic CUTLASS macro verification |
| `tests/sm121_cluster_static_assert_test.cu` | CUTLASS static_assert demonstration |
| `tests/sm121_runtime_cluster_test.cu` | **Runtime hardware capability proof** |

### Building and Running Tests

```bash
# Compile all tests
cd /path/to/flashinfer
nvcc -arch=sm_121a -std=c++17 --expt-relaxed-constexpr \
     -I3rdparty/cutlass/include \
     tests/sm121_cluster_test.cu -o tests/sm121_cluster_test

nvcc -arch=sm_121a -std=c++17 --expt-relaxed-constexpr \
     -I3rdparty/cutlass/include \
     tests/sm121_cluster_static_assert_test.cu -o tests/sm121_cluster_static_assert_test

nvcc -arch=sm_121a -std=c++17 \
     tests/sm121_runtime_cluster_test.cu -o tests/sm121_runtime_cluster_test

# Run tests
./tests/sm121_cluster_test
./tests/sm121_cluster_static_assert_test
./tests/sm121_runtime_cluster_test
```

---

## References

- CUTLASS SM120 builders: `3rdparty/cutlass/include/cutlass/gemm/collective/builders/sm120_*.inl`
- CUTLASS dispatch policies: `3rdparty/cutlass/include/cutlass/gemm/dispatch_policy.hpp`
- CUDA cluster launch API: `cudaLaunchKernelEx` with `cudaLaunchAttributeClusterDimension`

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-01-08 | AI-assisted | Initial investigation and documentation |

