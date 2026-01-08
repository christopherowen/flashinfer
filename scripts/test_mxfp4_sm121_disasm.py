#!/usr/bin/env python3
"""
Test script to compile SM120 MXFP4 kernels and disassemble them.

Usage:
    python scripts/test_mxfp4_sm121_disasm.py

This will:
1. Trigger JIT compilation of the fused_moe_120 module
2. Find the generated .so file
3. Use cuobjdump to extract SASS and look for tcgen05.mma instructions
"""

import os
import subprocess
import sys
from pathlib import Path

def find_nvcc_tools():
    """Find CUDA tools directory."""
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuobjdump = Path(cuda_home) / "bin" / "cuobjdump"
    nvdisasm = Path(cuda_home) / "bin" / "nvdisasm"
    return cuobjdump, nvdisasm

def main():
    print("=" * 60)
    print("SM120/SM121 MXFP4 Kernel Disassembly Test")
    print("=" * 60)
    
    # Check CUDA arch
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    cc = torch.cuda.get_device_capability()
    print(f"GPU Compute Capability: {cc[0]}.{cc[1]}")
    
    if cc[0] != 12 or cc[1] not in [0, 1]:
        print(f"WARNING: This test is designed for SM120/SM121, got SM{cc[0]}{cc[1]}")
    
    # Step 1: Try to import and trigger compilation
    print("\nStep 1: Triggering kernel compilation...")
    try:
        from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
        module = get_cutlass_fused_moe_module(backend="120", use_fast_build=False)
        print(f"  Module loaded: {module}")
    except Exception as e:
        print(f"  ERROR: Failed to load module: {e}")
        print("  This is expected if compilation hasn't completed yet.")
    
    # Step 2: Find compiled .so files
    print("\nStep 2: Looking for compiled kernel files...")
    cache_base = Path.home() / ".cache" / "flashinfer"
    so_files = list(cache_base.glob("*/121a/cached_ops/fused_moe_120/*.so"))
    
    if not so_files:
        so_files = list(cache_base.glob("*/12*/cached_ops/fused_moe_120/*.so"))
    
    if not so_files:
        print("  No compiled .so files found in cache.")
        print(f"  Cache directory: {cache_base}")
        print("  Please run vLLM with MXFP4 to trigger compilation first.")
        
        # Show what's in the cache
        print("\n  Contents of cache directory:")
        try:
            for item in cache_base.iterdir():
                print(f"    {item}")
        except Exception as e:
            print(f"    Cannot read cache: {e}")
        sys.exit(1)
    
    print(f"  Found {len(so_files)} .so file(s):")
    for f in so_files[:5]:
        print(f"    {f}")
    
    # Step 3: Disassemble
    print("\nStep 3: Disassembling kernel...")
    cuobjdump, nvdisasm = find_nvcc_tools()
    
    target_so = so_files[0]
    print(f"  Target: {target_so}")
    
    if not cuobjdump.exists():
        print(f"  ERROR: cuobjdump not found at {cuobjdump}")
        sys.exit(1)
    
    # Run cuobjdump to extract SASS
    print(f"\n  Running: {cuobjdump} --dump-sass {target_so}")
    result = subprocess.run(
        [str(cuobjdump), "--dump-sass", str(target_so)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  ERROR: cuobjdump failed: {result.stderr}")
        sys.exit(1)
    
    sass_output = result.stdout
    
    # Step 4: Look for key instructions
    print("\nStep 4: Searching for tensor core instructions...")
    
    # Key patterns to look for
    patterns = [
        "tcgen05",      # Tensor core generation 05 (Blackwell)
        "mma",          # Matrix multiply-accumulate
        "mxf4",         # MXFP4 format
        "block_scale",  # Block scaling
        "e2m1",         # FP4 E2M1 format
        "f8f6f4",       # FP8/FP6/FP4 MMA kind
    ]
    
    found_any = False
    for pattern in patterns:
        matches = [line for line in sass_output.split('\n') if pattern.lower() in line.lower()]
        if matches:
            found_any = True
            print(f"\n  Found '{pattern}' ({len(matches)} matches):")
            for match in matches[:5]:  # Show first 5 matches
                print(f"    {match.strip()}")
            if len(matches) > 5:
                print(f"    ... and {len(matches) - 5} more")
    
    if not found_any:
        print("  WARNING: No tensor core instructions found!")
        print("  This might indicate the kernel is not using expected MMA instructions.")
        print("\n  Dumping first 100 lines of SASS for inspection:")
        for line in sass_output.split('\n')[:100]:
            print(f"    {line}")
    
    # Save full SASS output
    sass_file = Path("/tmp/mxfp4_sm121_kernel.sass")
    sass_file.write_text(sass_output)
    print(f"\n  Full SASS saved to: {sass_file}")
    print(f"  To inspect: less {sass_file}")
    print(f"  To search: grep -i 'mma\\|tcgen05' {sass_file}")

if __name__ == "__main__":
    main()


