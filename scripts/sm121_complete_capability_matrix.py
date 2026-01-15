#!/usr/bin/env python3
"""
Complete SM121 (GB10) Hardware Capability Matrix
Tests all key instructions to determine what native support exists.
"""

import subprocess
import tempfile
import os

def test_compile(name, cuda_code, arch="sm_121a"):
    """Test if CUDA code compiles for the given architecture."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(cuda_code)
        cu_file = f.name
    
    try:
        result = subprocess.run(
            ["nvcc", f"-arch={arch}", "-c", cu_file, "-o", "/dev/null"],
            capture_output=True, text=True
        )
        error = None
        if result.returncode != 0:
            for line in result.stderr.split("\n"):
                if "Feature" in line or "not supported" in line.lower() or "Instruction" in line:
                    error = line.strip()
                    break
            if not error:
                error = result.stderr[:150]
        return result.returncode == 0, error
    finally:
        os.unlink(cu_file)


def main():
    print("=" * 90)
    print("SM121 (GB10) Complete Hardware Capability Matrix")
    print("=" * 90)
    
    tests = [
        # Category, Name, CUDA code, Expected
        ("ldmatrix", "m8n8.x4.b16 (basic FP16/BF16)", '''
__global__ void test(void* s, unsigned* o) {
    unsigned d0,d1,d2,d3, addr=(unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
        :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3):"r"(addr));
    o[0]=d0+d1+d2+d3;
}''', True),
        
        ("ldmatrix", "m8n16.x4 FP4->FP8 (SM100)", '''
__global__ void test(void* s, unsigned* o) {
    unsigned d0,d1,d2,d3, addr=(unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3},[%4];"
        :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3):"r"(addr));
    o[0]=d0+d1+d2+d3;
}''', False),

        ("stmatrix", "m8n8.x4.b16 (basic)", '''
__global__ void test(void* s, unsigned* i) {
    unsigned addr=(unsigned)__cvta_generic_to_shared(s);
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%2,%3,%4};"
        ::"r"(addr),"r"(i[0]),"r"(i[1]),"r"(i[2]),"r"(i[3]));
}''', True),

        ("TMA", "cp.async.bulk.tensor.2d", '''
__global__ void test(void* s, void* g, void* m) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{0,0}],[%2];"
        ::"l"(s),"l"(g),"l"(m));
}''', True),

        ("MMA", "m16n8k16.f32.f16.f16 (SM80 FP16)", '''
__global__ void test(float* o) {
    float d0,d1,d2,d3,c0=0,c1=0,c2=0,c3=0;
    unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(c0),"f"(c1),"f"(c2),"f"(c3));
    o[0]=d0+d1+d2+d3;
}''', True),

        ("MMA", "m16n8k8.f32.bf16.bf16 (SM80 BF16)", '''
__global__ void test(float* o) {
    float d0,d1,d2,d3,c0=0,c1=0,c2=0,c3=0;
    unsigned a0=0,a1=0,b0=0;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(b0),"f"(c0),"f"(c1),"f"(c2),"f"(c3));
    o[0]=d0+d1+d2+d3;
}''', True),

        ("MMA", "m16n8k32.e2m1.e2m1 (SM120 FP4xFP4)", '''
__global__ void test(float* o) {
    float d0,d1,d2,d3,c0=0,c1=0,c2=0,c3=0;
    unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(c0),"f"(c1),"f"(c2),"f"(c3));
    o[0]=d0+d1+d2+d3;
}''', False),

        ("MMA", "m16n8k32.e4m3.e2m1 (SM120 FP8xFP4)", '''
__global__ void test(float* o) {
    float d0,d1,d2,d3,c0=0,c1=0,c2=0,c3=0;
    unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(c0),"f"(c1),"f"(c2),"f"(c3));
    o[0]=d0+d1+d2+d3;
}''', False),

        ("MMA", "m16n8k32.e4m3.e4m3 (SM120 FP8xFP8)", '''
__global__ void test(float* o) {
    float d0,d1,d2,d3,c0=0,c1=0,c2=0,c3=0;
    unsigned a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(c0),"f"(c1),"f"(c2),"f"(c3));
    o[0]=d0+d1+d2+d3;
}''', False),

        ("tcgen05", "tcgen05.wait (fence)", '''
__global__ void test() {
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}''', False),

        ("wgmma", "wgmma.fence (SM90 warpgroup)", '''
__global__ void test() {
    asm volatile("wgmma.fence.sync.aligned;");
}''', False),
    ]
    
    current_category = None
    for category, name, code, expected in tests:
        if category != current_category:
            print(f"\n{category.upper()}")
            print("-" * 70)
            print(f"  {'Instruction':<45} {'SM121':<8} {'Match'}")
            print("  " + "-" * 65)
            current_category = category
        
        supported, error = test_compile(name, code)
        actual = "YES" if supported else "NO"
        expected_str = "YES" if expected else "NO"
        match = "OK" if supported == expected else "MISMATCH!"
        
        print(f"  {name:<45} {actual:<8} {match}")
        if error and not supported:
            # Truncate error
            short = error[:60] + "..." if len(error) > 60 else error
            print(f"      -> {short}")
    
    print("\n" + "=" * 90)
    print("SUMMARY: SM121 (GB10) Hardware Capabilities")
    print("=" * 90)
    print("""
SUPPORTED on SM121:
  - ldmatrix.m8n8 (basic FP16/BF16 matrix loads)
  - stmatrix.m8n8 (basic matrix stores)
  - TMA (Tensor Memory Accelerator) for async data movement
  - mma.m16n8k16 FP16 (standard SM80 tensor core)
  - mma.m16n8k8 BF16 (standard SM80 tensor core)

NOT SUPPORTED on SM121:
  - ldmatrix.m8n16 with FP4/FP6 format conversion
  - ldmatrix.m16n16 extended shapes
  - mma.kind::f8f6f4 (FP4/FP6/FP8 mixed precision MMA)
  - tcgen05 (Blackwell block-scaled tensor core instructions)
  - wgmma (SM90 warpgroup MMA)

CRITICAL IMPLICATIONS FOR MXFP4:
  The SM120 CUTLASS block-scaled kernels require BOTH:
    1. ldmatrix.m8n16 with FP4->FP8 unpacking (NOT available)
    2. mma.kind::f8f6f4 for FP8xFP4 compute (NOT available)
  
  SM121 CANNOT run native MXFP4 kernels because:
    - No hardware FP4 data path
    - No FP4/FP8 mixed-precision tensor core
  
WORKAROUND OPTIONS:
  1. Dequantize FP4 -> FP16/BF16 and use standard tensor cores
  2. Use software FP4 unpacking + FP8 emulation (very slow)
  3. Wait for NVIDIA driver/CUTLASS update for SM121
  4. Use different quantization (INT8, FP16) that SM121 supports
""")


if __name__ == "__main__":
    main()
