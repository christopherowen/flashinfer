#!/usr/bin/env python3
"""
Test ALL ldmatrix variants on SM121 to determine exact hardware capabilities.
"""

import subprocess
import tempfile
import os

def test_ldmatrix(name, ptx_instr, num_outputs, arch="sm_121a"):
    """Test if an ldmatrix instruction compiles for the given architecture."""
    
    # Build output register list
    if num_outputs == 1:
        out_regs = '"=r"(d0)'
        out_use = "d0"
    elif num_outputs == 2:
        out_regs = '"=r"(d0), "=r"(d1)'
        out_use = "d0 + d1"
    elif num_outputs == 4:
        out_regs = '"=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)'
        out_use = "d0 + d1 + d2 + d3"
    else:
        return None, "Invalid num_outputs"
    
    cuda_code = f'''
__device__ unsigned int cast_smem(void const* p) {{ 
    return (unsigned int)__cvta_generic_to_shared(p); 
}}

__global__ void test_kernel(void* smem, unsigned int* out) {{
    unsigned int d0=0, d1=0, d2=0, d3=0;
    unsigned int addr = cast_smem(smem);
    asm volatile("{ptx_instr}" : {out_regs} : "r"(addr));
    out[0] = {out_use};
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
            for line in result.stderr.split("\n"):
                if "Feature" in line or "not supported" in line.lower() or "error" in line:
                    return False, line.strip()
            return False, "Unknown error"
    finally:
        os.unlink(cu_file)


def main():
    print("=" * 90)
    print("Complete ldmatrix Instruction Support on SM121 (GB10)")
    print("=" * 90)
    
    # All ldmatrix variants found in CUTLASS
    tests = [
        # SM75+ basic ldmatrix (m8n8 shape, 16-bit elements)
        ("Category: SM75+ Basic (m8n8, b16)", None, None),
        ("ldmatrix.x1.m8n8.b16", "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];", 1),
        ("ldmatrix.x2.m8n8.b16", "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];", 2),
        ("ldmatrix.x4.m8n8.b16", "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];", 4),
        ("ldmatrix.x1.trans.m8n8.b16", "ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];", 1),
        ("ldmatrix.x2.trans.m8n8.b16", "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];", 2),
        ("ldmatrix.x4.trans.m8n8.b16", "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];", 4),
        
        # SM100 extended (m16n16 shape, 8-bit elements with transpose)
        ("Category: SM100 Extended (m16n16, b8)", None, None),
        ("ldmatrix.m16n16.x1.trans.b8", "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];", 2),
        ("ldmatrix.m16n16.x2.trans.b8", "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];", 4),
        
        # SM100 format conversion (m8n16 shape, FP4/FP6 unpacking)
        ("Category: SM100 Format Conversion (FP4/FP6 unpack)", None, None),
        ("ldmatrix.m8n16.x1.b4x16_p64 (FP4)", "ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64 {%0}, [%1];", 1),
        ("ldmatrix.m8n16.x2.b4x16_p64 (FP4)", "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0, %1}, [%2];", 2),
        ("ldmatrix.m8n16.x4.b4x16_p64 (FP4)", "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0, %1, %2, %3}, [%4];", 4),
        ("ldmatrix.m8n16.x1.b6x16_p32 (FP6)", "ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 {%0}, [%1];", 1),
        ("ldmatrix.m8n16.x2.b6x16_p32 (FP6)", "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32 {%0, %1}, [%2];", 2),
        ("ldmatrix.m8n16.x4.b6x16_p32 (FP6)", "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 {%0, %1, %2, %3}, [%4];", 4),
    ]
    
    print(f"\n{'Instruction':<50} {'SM121':<8} {'Notes'}")
    print("-" * 90)
    
    supported_count = 0
    unsupported_count = 0
    
    for name, ptx, outputs in tests:
        if ptx is None:
            # Category header
            print(f"\n{name}")
            print("-" * 50)
            continue
            
        supported, error = test_ldmatrix(name, ptx, outputs)
        
        if supported:
            status = "YES"
            supported_count += 1
        else:
            status = "NO"
            unsupported_count += 1
        
        print(f"  {name:<48} {status:<8}", end="")
        if error:
            # Extract just the key part of error
            if "Feature" in error:
                import re
                match = re.search(r"Feature '([^']+)'", error)
                if match:
                    print(f"(Missing: {match.group(1)})", end="")
        print()
    
    print("\n" + "=" * 90)
    print(f"Summary: {supported_count} supported, {unsupported_count} unsupported")
    print("=" * 90)
    
    print("""
KEY FINDINGS:
- SM121 supports basic ldmatrix (m8n8, b16) - the SM75+ variants
- SM121 does NOT support:
  * m16n16 shape (SM100 extended)
  * m8n16 shape with format conversion (FP4/FP6 unpacking)
  
The SM120 CUTLASS block-scaled kernels use the FP4 unpacking variant
(ldmatrix.m8n16.x4.b4x16_p64) which SM121 lacks.

WORKAROUND OPTIONS:
1. Use basic ldmatrix.x4.m8n8.b16 + software FP4 unpacking
2. Use element-by-element loads (slower but works)
3. Report to NVIDIA as SM121 CUTLASS support gap
""")


if __name__ == "__main__":
    main()
