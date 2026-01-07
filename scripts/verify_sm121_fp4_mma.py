#!/usr/bin/env python3
"""
SM121 FP4 MMA Instruction Verification Script

This script verifies that FlashInfer's SM121 MoE GEMM kernels use native
Blackwell FP4 tensor core instructions (tcgen05 block-scaled MMA).

It:
1. Builds a minimal SM121 grouped GEMM kernel
2. Disassembles the resulting binary using nvdisasm/cuobjdump
3. Asserts the SASS contains SM12x block-scaled FP4 MMA instruction signatures
4. Fails if it sees only BF16/FP16 MMA sequences

Usage:
    python scripts/verify_sm121_fp4_mma.py [--kernel-type fp8xfp4|mxfp4|nvfp4]
    
Requirements:
    - CUDA Toolkit (nvdisasm or cuobjdump in PATH)
    - NVIDIA GPU with SM121 compute capability
    - FlashInfer installed with SM121 support
"""

import argparse
import subprocess
import tempfile
import os
import sys
import re


# =============================================================================
# SASS Instruction Patterns for SM12x FP4 MMA
# =============================================================================
#
# SM12x block-scaled FP4 uses "tcgen05" (tensor core gen 5) instructions.
# Key patterns to look for in SASS:
#
# Block-scaled MMA instructions:
#   HMMA.TT.F32.F8F4.BLOCK_SCALE  - FP8 x FP4 with block scaling
#   HMMA.*.*.MXF4*                 - MXFP4 variants
#   TCGEN05.*                      - Gen5 tensor core operations
#   MMA.*F4*                       - FP4 MMA operations
#
# If we see these BF16/FP16 patterns WITHOUT FP4 patterns, the kernel is wrong:
#   HMMA.*.BF16.*
#   HMMA.*.F16.*
#   without corresponding FP4/MXF4/BLOCK_SCALE instructions

# Patterns that indicate native FP4 MMA is being used
FP4_MMA_PATTERNS = [
    r'TCGEN05',              # Gen5 tensor core (Blackwell)
    r'UMMA',                 # Unified MMA (SM100+)
    r'\.MXF4',               # MXFP4 operations
    r'\.F4',                 # FP4 operations
    r'\.E2M1',               # E2M1 format (FP4)
    r'BLOCK_SCALE',          # Block-scaled operations
    r'UBLK',                 # Unified block operations
    r'mma\.sync.*f4',        # MMA with FP4
]

# Patterns that indicate fallback to BF16/FP16 MMA (without FP4)
FALLBACK_PATTERNS = [
    r'HMMA\..*\.BF16',       # BF16 half MMA
    r'HMMA\..*\.F16',        # FP16 half MMA
    r'mma\.sync.*bf16',      # MMA with BF16
    r'mma\.sync.*f16',       # MMA with FP16
]


def find_cuda_tools():
    """Find nvdisasm or cuobjdump in PATH or CUDA installation."""
    tools = ['nvdisasm', 'cuobjdump']
    
    # Check PATH first
    for tool in tools:
        try:
            result = subprocess.run([tool, '--version'], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return tool
        except FileNotFoundError:
            pass
    
    # Check common CUDA paths
    cuda_paths = [
        '/usr/local/cuda/bin',
        '/opt/cuda/bin',
        os.environ.get('CUDA_HOME', '') + '/bin',
    ]
    
    for cuda_path in cuda_paths:
        for tool in tools:
            tool_path = os.path.join(cuda_path, tool)
            if os.path.exists(tool_path):
                return tool_path
    
    return None


def get_sass_dump(cubin_path, tool):
    """Get SASS dump from a cubin file."""
    if 'nvdisasm' in tool:
        cmd = [tool, '-c', cubin_path]
    else:  # cuobjdump
        cmd = [tool, '--dump-sass', cubin_path]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to disassemble: {result.stderr}")
    
    return result.stdout


def analyze_sass(sass_output):
    """Analyze SASS output for FP4 MMA instructions."""
    fp4_matches = []
    fallback_matches = []
    
    for line in sass_output.split('\n'):
        # Check for FP4 patterns
        for pattern in FP4_MMA_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                fp4_matches.append((pattern, line.strip()))
        
        # Check for fallback patterns
        for pattern in FALLBACK_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                fallback_matches.append((pattern, line.strip()))
    
    return fp4_matches, fallback_matches


def create_test_kernel_source(kernel_type):
    """Create a minimal test kernel source file."""
    
    source = f'''
// Minimal SM121 FP4 MMA Test Kernel
// This file is auto-generated for SASS verification

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/float8.h"

#include "cute/tensor.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

// Element types for {kernel_type}
#if "{kernel_type}" == "nvfp4"
using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
#elif "{kernel_type}" == "fp8xfp4"
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e2m1_t;
#else  // mxfp4
using ElementA = cutlass::float_e4m3_t;  // BF16 quantized to FP8
using ElementB = cutlass::float_e2m1_t;
#endif

using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementSF = cutlass::float_ue8m0_t;

// Block-scaled element types
using ElementAMX = cutlass::mx_float8_t<ElementA>;
using ElementBMX = cutlass::mx_float4_t<ElementB>;

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Tile shape - use a known-good SM120 configuration
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// Architecture
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Alignments
constexpr int AlignmentA = 16;  // 128 bits / 8 bits per element
constexpr int AlignmentB = 32;  // 128 bits / 4 bits per element
constexpr int AlignmentC = 8;   // 128 bits / 16 bits per element
constexpr int AlignmentD = 8;

// Schedules
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

// Problem shape for grouped GEMM
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

// Build epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC*, AlignmentC,
    ElementD, LayoutD*, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

// Build mainloop with block-scaled types
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementAMX, LayoutA*, AlignmentA,
    ElementBMX, LayoutB*, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

// GEMM kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Force template instantiation
__global__ void force_instantiation() {{
    // Just reference the type to force instantiation
    Gemm::Arguments args;
    (void)args;
}}

#endif  // SM120/SM121 supported

int main() {{
    return 0;
}}
'''
    return source


def build_test_kernel(kernel_type, flashinfer_root):
    """Build a minimal test kernel and return the cubin path."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write source
        source_path = os.path.join(tmpdir, 'test_kernel.cu')
        with open(source_path, 'w') as f:
            f.write(create_test_kernel_source(kernel_type))
        
        # Cubin path
        cubin_path = os.path.join(tmpdir, 'test_kernel.cubin')
        
        # Build command
        cutlass_include = os.path.join(flashinfer_root, '3rdparty/cutlass/include')
        cutlass_tools = os.path.join(flashinfer_root, '3rdparty/cutlass/tools/util/include')
        
        cmd = [
            'nvcc',
            '-arch=sm_121a',  # Target SM121
            '-std=c++17',
            '--expt-relaxed-constexpr',
            '-DENABLE_FP4=1',
            f'-I{cutlass_include}',
            f'-I{cutlass_tools}',
            '-cubin',
            source_path,
            '-o', cubin_path,
        ]
        
        print(f"Building test kernel ({kernel_type})...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return None
        
        print(f"Build succeeded: {cubin_path}")
        
        # Copy cubin to persistent location
        persistent_cubin = os.path.join(flashinfer_root, f'test_sm121_{kernel_type}.cubin')
        subprocess.run(['cp', cubin_path, persistent_cubin])
        
        return persistent_cubin


def verify_kernel(cubin_path, tool):
    """Verify a kernel uses native FP4 MMA instructions."""
    
    print(f"\n=== Analyzing SASS for {cubin_path} ===")
    
    sass = get_sass_dump(cubin_path, tool)
    fp4_matches, fallback_matches = analyze_sass(sass)
    
    print(f"\nFP4 MMA pattern matches: {len(fp4_matches)}")
    for pattern, line in fp4_matches[:10]:  # Show first 10
        print(f"  [{pattern}] {line[:100]}...")
    if len(fp4_matches) > 10:
        print(f"  ... and {len(fp4_matches) - 10} more")
    
    print(f"\nBF16/FP16 fallback pattern matches: {len(fallback_matches)}")
    for pattern, line in fallback_matches[:5]:
        print(f"  [{pattern}] {line[:100]}...")
    if len(fallback_matches) > 5:
        print(f"  ... and {len(fallback_matches) - 5} more")
    
    # Decision logic
    uses_fp4 = len(fp4_matches) > 0
    has_fallback = len(fallback_matches) > 0
    
    print("\n=== Verification Result ===")
    
    if uses_fp4:
        print("✅ PASS: Native FP4 MMA instructions detected!")
        if has_fallback:
            print("   Note: Some BF16/FP16 MMA also present (may be for epilogue/aux ops)")
        return True
    elif has_fallback:
        print("❌ FAIL: Only BF16/FP16 MMA found, no native FP4!")
        print("   The kernel is NOT using native FP4 tensor cores.")
        return False
    else:
        print("⚠️ WARNING: No recognized MMA patterns found.")
        print("   Could not determine if kernel uses FP4 or fallback.")
        print("   Manual SASS inspection may be needed.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Verify SM121 kernels use native FP4 MMA instructions'
    )
    parser.add_argument(
        '--kernel-type',
        choices=['fp8xfp4', 'mxfp4', 'nvfp4'],
        default='fp8xfp4',
        help='Kernel type to verify (default: fp8xfp4)'
    )
    parser.add_argument(
        '--cubin',
        type=str,
        help='Path to pre-built cubin file (skip build step)'
    )
    parser.add_argument(
        '--flashinfer-root',
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help='Path to FlashInfer root directory'
    )
    
    args = parser.parse_args()
    
    # Find disassembly tool
    tool = find_cuda_tools()
    if not tool:
        print("ERROR: Could not find nvdisasm or cuobjdump")
        print("Please ensure CUDA Toolkit is installed and in PATH")
        sys.exit(1)
    print(f"Using disassembly tool: {tool}")
    
    # Get or build cubin
    if args.cubin:
        cubin_path = args.cubin
    else:
        cubin_path = build_test_kernel(args.kernel_type, args.flashinfer_root)
        if not cubin_path:
            print("ERROR: Failed to build test kernel")
            sys.exit(1)
    
    # Verify
    success = verify_kernel(cubin_path, tool)
    
    # Cleanup temp cubin if we built it
    if not args.cubin and os.path.exists(cubin_path):
        os.remove(cubin_path)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
'''
)

