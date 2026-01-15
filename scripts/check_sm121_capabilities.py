#!/usr/bin/env python3
"""
Check SM121 (GB10) hardware capabilities vs what CUTLASS/FlashInfer enables.
"""

import subprocess
import tempfile
import os

def test_instruction(name, ptx_template, arch="sm_121a"):
    """Test if a PTX instruction compiles for the given architecture."""
    
    cuda_code = f'''
__device__ unsigned int cast_smem(void const* p) {{ 
    return (unsigned int)__cvta_generic_to_shared(p); 
}}

__global__ void test_kernel(void* smem, unsigned int* out) {{
    unsigned int d0=0, d1=0, d2=0, d3=0;
    unsigned int addr = cast_smem(smem);
    {ptx_template}
    out[0] = d0 + d1 + d2 + d3;
}}
'''
    
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
            # Extract error
            for line in result.stderr.split("\n"):
                if "Feature" in line or "not supported" in line.lower():
                    return False, line.strip()
            return False, result.stderr[:200]
    finally:
        os.unlink(cu_file)


def main():
    print("=" * 80)
    print("SM121 (GB10) Hardware Capability Check")
    print("=" * 80)
    
    tests = [
        # (Name, PTX template, Expected on SM121)
        ("ldmatrix.m8n8.x4.b16 (basic)", 
         'asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3) : "r"(addr));',
         True),
        
        ("ldmatrix.m8n16 FP4->FP8 (SM100)", 
         'asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];" : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3) : "r"(addr));',
         False),
        
        ("ldmatrix.m16n16.trans (SM100)", 
         'asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];" : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3) : "r"(addr));',
         False),
         
        ("stmatrix.m8n8.x4.b16 (basic)",
         'asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%4], {%0,%1,%2,%3};" :: "r"(d0),"r"(d1),"r"(d2),"r"(d3), "r"(addr));',
         True),
    ]
    
    print(f"\n{'Instruction':<45} {'SM121 HW':<12} {'Expected':<10} {'Status'}")
    print("-" * 80)
    
    all_pass = True
    for name, ptx, expected in tests:
        supported, error = test_instruction(name, ptx)
        expected_str = "YES" if expected else "NO"
        actual_str = "YES" if supported else "NO"
        
        if supported == expected:
            status = "OK"
        else:
            status = "MISMATCH!"
            all_pass = False
        
        print(f"{name:<45} {actual_str:<12} {expected_str:<10} {status}")
        if error and not supported:
            # Truncate error for display
            short_error = error[:70] + "..." if len(error) > 70 else error
            print(f"    Error: {short_error}")
    
    print("\n" + "=" * 80)
    print("CUTLASS Configuration Issue:")
    print("=" * 80)
    print("""
In cute/arch/config.hpp lines 128-133, CUTLASS incorrectly enables
CUTE_ARCH_LDSM_SM100A_ENABLED for SM121:

    #if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || ... ||
         defined(CUTLASS_ARCH_MMA_SM121A_ENABLED))  // <-- BUG: SM121 included
    #  define CUTE_ARCH_LDSM_SM100A_ENABLED        // <-- Enables SM100-only ldmatrix
    #  define CUTE_ARCH_STSM_SM100A_ENABLED
    #endif

This causes the SM120 block-scaled kernels to use SM100_SU4_DU8x16_x4_LDSM_N
which emits ldmatrix.m8n16 with format conversion - an instruction that
SM121 hardware does NOT support.

FIX: Remove SM121A from this condition, or create SM121-specific copy atoms.
""")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
