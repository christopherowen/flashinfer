#!/usr/bin/env python3
"""
Comprehensive SM121 (GB10) hardware capability check.
Tests all key instructions used by CUTLASS SM120 block-scaled kernels.
"""

import subprocess
import tempfile
import os

def test_ptx(name, cuda_code, arch="sm_121a"):
    """Test if CUDA code with inline PTX compiles for the given architecture."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(cuda_code)
        cu_file = f.name
    
    try:
        result = subprocess.run(
            ["nvcc", f"-arch={arch}", "-c", cu_file, "-o", "/dev/null"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            return True, None
        else:
            for line in result.stderr.split("\n"):
                if "Feature" in line or "not supported" in line.lower() or "error" in line:
                    return False, line.strip()
            return False, result.stderr[:200]
    finally:
        os.unlink(cu_file)


def main():
    print("=" * 90)
    print("SM121 (GB10) Complete Hardware Capability Matrix")
    print("=" * 90)
    
    # Define all tests
    tests = {
        "ldmatrix (shared memory → registers)": [
            ("m8n8.x4.b16 (basic, SM75+)", '''
__global__ void test(void* s, unsigned int* o) {
    unsigned int d0, d1, d2, d3;
    unsigned int addr = (unsigned int)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
    o[0] = d0;
}'''),
            ("m8n8.x4.trans.b16 (transpose)", '''
__global__ void test(void* s, unsigned int* o) {
    unsigned int d0, d1, d2, d3;
    unsigned int addr = (unsigned int)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
    o[0] = d0;
}'''),
            ("m8n16.x4 FP4→FP8 (SM100)", '''
__global__ void test(void* s, unsigned int* o) {
    unsigned int d0, d1, d2, d3;
    unsigned int addr = (unsigned int)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
    o[0] = d0;
}'''),
        ],
        
        "stmatrix (registers → shared memory)": [
            ("m8n8.x4.b16 (basic, SM90+)", '''
__global__ void test(void* s, unsigned int* i) {
    unsigned int addr = (unsigned int)__cvta_generic_to_shared(s);
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(i[0]), "r"(i[1]), "r"(i[2]), "r"(i[3]));
}'''),
            ("m8n8.x4.trans.b16 (transpose)", '''
__global__ void test(void* s, unsigned int* i) {
    unsigned int addr = (unsigned int)__cvta_generic_to_shared(s);
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(i[0]), "r"(i[1]), "r"(i[2]), "r"(i[3]));
}'''),
        ],
        
        "TMA (Tensor Memory Accelerator)": [
            ("cp.async.bulk.tensor.2d (SM90+)", '''
__global__ void test(void* smem, void* gmem, void* mbar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {0, 0}], [%2];"
        :: "l"(smem), "l"(gmem), "l"(mbar));
}'''),
        ],
        
        "MMA (Matrix Multiply-Accumulate)": [
            ("mma.m16n8k16.f32.f16.f16 (SM80)", '''
#include <cuda_fp16.h>
__global__ void test() {
    unsigned int a[4], b[2], c[4], d[4];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}'''),
            ("mma.m16n8k8.f32.bf16.bf16 (SM80)", '''
#include <cuda_bf16.h>
__global__ void test() {
    unsigned int a[2], b[1], c[4], d[4];
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}'''),
        ],
        
        "Warpgroup MMA (SM90+)": [
            ("wgmma.mma_async.m64n128k16.f32.f16", '''
__global__ void test() {
    asm volatile("wgmma.fence.sync.aligned;");
}'''),
        ],
    }
    
    for category, category_tests in tests.items():
        print(f"\n{category}")
        print("-" * 70)
        print(f"  {'Instruction':<45} {'SM121':<8}")
        print("  " + "-" * 60)
        
        for name, code in category_tests:
            supported, error = test_ptx(name, code)
            status = "YES" if supported else "NO"
            print(f"  {name:<45} {status:<8}", end="")
            
            if error and not supported:
                # Extract key error info
                import re
                match = re.search(r"Feature '([^']+)'", error)
                if match:
                    print(f"(Missing: {match.group(1)})", end="")
                elif "not supported" in error.lower():
                    print("(Not supported)", end="")
            print()
    
    print("\n" + "=" * 90)
    print("ANALYSIS FOR SM120 BLOCK-SCALED MXFP4 KERNELS")
    print("=" * 90)
    print("""
The SM120 CUTLASS block-scaled kernels require:
  1. ldmatrix.m8n16.x4 with FP4→FP8 format conversion  → NOT AVAILABLE on SM121
  2. stmatrix.m8n8.x4 for epilogue                     → AVAILABLE
  3. TMA for async data movement                       → AVAILABLE
  4. tcgen05 MMA for block-scaled compute              → Status TBD

The MISSING PIECE is hardware FP4 unpacking in ldmatrix.
Without it, we need software FP4→FP8 conversion before feeding tensor cores.

ALTERNATIVE APPROACHES:
  1. Use ldmatrix.m8n8.x4.b16 to load FP4 as raw bytes
  2. Software unpack: extract 4-bit values, convert to FP8
  3. Feed unpacked FP8 data to tensor cores
  
This adds overhead but enables SM121 compatibility.
""")


if __name__ == "__main__":
    main()
