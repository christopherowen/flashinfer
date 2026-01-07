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
# Priority: patterns earlier in the list are stronger indicators
FP4_MMA_PATTERNS = [
    # STRONGEST INDICATORS (block-scaled FP4 MMA)
    r'TCGEN05\.MMA.*BLOCK_SCALE',  # Gen5 TC with block scaling
    r'HMMA\..*\.F4.*BLOCK_SCALE',  # Half MMA with FP4 + block scale
    r'UMMA\..*MXF4',               # Unified MMA with MXFP4
    r'UMMA\..*BLOCK_SCALE',        # Unified MMA with block scaling
    r'\.MXF4.*BLOCK_SCALE',        # MXFP4 with block scaling
    r'\.E2M1.*BLOCK_SCALE',        # E2M1 (FP4) with block scaling
    
    # SECONDARY INDICATORS (FP4 operations without explicit block_scale)
    r'TCGEN05',                    # Gen5 tensor core (Blackwell)
    r'UMMA',                       # Unified MMA (SM100+)
    r'\.MXF4',                     # MXFP4 operations
    r'\.E2M1',                     # E2M1 format (FP4)
    
    # WEAK INDICATORS (may appear in non-FP4 code too)
    r'BLOCK_SCALE',                # Block-scaled operations (could be FP8)
    r'UBLK',                       # Unified block operations
]

# DEFINITIVE patterns - these PROVE block-scaled FP4 MMA is used
# Must match: (TCGEN05 or UMMA) AND BLOCK_SCALE AND (FP4 format indicator)
# We check for combinations, not individual patterns
FP4_DEFINITIVE_PATTERNS = [
    # TCGEN05 with block scaling AND FP4 format
    r'TCGEN05.*BLOCK_SCALE.*(?:MXF4|E2M1|\.F4)',
    r'TCGEN05.*(?:MXF4|E2M1|\.F4).*BLOCK_SCALE',
    # UMMA with MXFP4 and block scaling
    r'UMMA.*MXF4.*BLOCK_SCALE',
    r'UMMA.*BLOCK_SCALE.*MXF4',
    # HMMA with FP4 and block scaling  
    r'HMMA\..*(?:MXF4|E2M1|\.F4).*BLOCK_SCALE',
    r'HMMA\..*BLOCK_SCALE.*(?:MXF4|E2M1|\.F4)',
]

# Strong patterns - good indicators but not definitive alone
FP4_STRONG_PATTERNS = [
    r'TCGEN05.*(?:MXF4|E2M1|\.F4)',  # TCGEN05 with FP4 format
    r'TCGEN05.*BLOCK_SCALE',          # TCGEN05 with block scaling
    r'UMMA.*(?:MXF4|E2M1)',           # UMMA with FP4 format
    r'HMMA\..*(?:MXF4|E2M1)\..*BLOCK', # HMMA with FP4 and block
]

# Weak patterns - may appear but need context
FP4_WEAK_PATTERNS = [
    r'TCGEN05',                        # Gen5 TC (could be other ops)
    r'\.MXF4',                         # MXFP4 mention
    r'\.E2M1',                         # E2M1 format
    r'BLOCK_SCALE',                    # Block scaling (could be FP8)
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
    """Analyze SASS output for FP4 MMA instructions.
    
    Returns:
        tuple: (definitive_matches, strong_matches, weak_matches, fallback_matches)
        - definitive_matches: Patterns that PROVE block-scaled FP4 MMA
        - strong_matches: Good indicators requiring context
        - weak_matches: May appear in non-FP4 code
        - fallback_matches: BF16/FP16 fallback patterns
    """
    definitive_matches = []
    strong_matches = []
    weak_matches = []
    fallback_matches = []
    
    for line in sass_output.split('\n'):
        # Check for definitive FP4 patterns (PROVE block-scaled FP4)
        for pattern in FP4_DEFINITIVE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                definitive_matches.append((pattern, line.strip()))
        
        # Check for strong FP4 patterns
        for pattern in FP4_STRONG_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                strong_matches.append((pattern, line.strip()))
        
        # Check for weak FP4 patterns
        for pattern in FP4_WEAK_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                weak_matches.append((pattern, line.strip()))
        
        # Check for fallback patterns
        for pattern in FALLBACK_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                fallback_matches.append((pattern, line.strip()))
    
    return definitive_matches, strong_matches, weak_matches, fallback_matches


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
    """Verify a kernel uses native FP4 MMA instructions.
    
    Verification hierarchy:
    1. DEFINITIVE: Patterns proving TCGEN05 AND BLOCK_SCALE AND FP4 format → PASS
    2. STRONG: Good indicators (TCGEN05+FP4 or TCGEN05+BLOCK_SCALE) → LIKELY PASS
    3. WEAK: Individual patterns → needs manual review
    4. FALLBACK only: BF16/FP16 MMA without FP4 → FAIL
    """
    
    print(f"\n=== Analyzing SASS for {cubin_path} ===")
    
    sass = get_sass_dump(cubin_path, tool)
    definitive, strong, weak, fallback = analyze_sass(sass)
    
    print(f"\n=== Pattern Match Summary ===")
    print(f"DEFINITIVE (TCGEN05 + BLOCK_SCALE + FP4): {len(definitive)}")
    for pattern, line in definitive[:3]:
        print(f"  ✓✓ {line[:90]}")
    if len(definitive) > 3:
        print(f"  ... and {len(definitive) - 3} more")
    
    print(f"\nSTRONG (TCGEN05+FP4 or TCGEN05+BLOCK_SCALE): {len(strong)}")
    for pattern, line in strong[:3]:
        print(f"  ✓  {line[:90]}")
    if len(strong) > 3:
        print(f"  ... and {len(strong) - 3} more")
    
    print(f"\nWEAK (individual patterns): {len(weak)}")
    
    print(f"\nFALLBACK (BF16/FP16 MMA): {len(fallback)}")
    for pattern, line in fallback[:2]:
        print(f"  ⚠  {line[:90]}")
    if len(fallback) > 2:
        print(f"  ... and {len(fallback) - 2} more")
    
    # Decision logic - STRICT verification
    has_definitive = len(definitive) > 0
    has_strong = len(strong) > 0
    has_weak = len(weak) > 0
    has_fallback = len(fallback) > 0
    
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    
    if has_definitive:
        print("✅ PASS: DEFINITIVE block-scaled FP4 MMA instructions detected!")
        print(f"   Found {len(definitive)} patterns matching:")
        print("   TCGEN05/UMMA/HMMA + BLOCK_SCALE + (MXF4/E2M1/F4)")
        if has_fallback:
            print(f"   Note: {len(fallback)} BF16/FP16 MMA also present (likely epilogue)")
        return True
    elif has_strong:
        print("✓ LIKELY PASS: Strong FP4 indicators found but no definitive proof.")
        print(f"   Found {len(strong)} strong patterns.")
        print("   This is probably correct but manual SASS review recommended.")
        if has_fallback:
            print(f"   Warning: {len(fallback)} BF16/FP16 MMA also present.")
        return True  # Accept strong patterns
    elif has_weak and not has_fallback:
        print("⚠️ UNCERTAIN: Only weak FP4 patterns found.")
        print(f"   Found {len(weak)} weak patterns (TCGEN05, MXF4, E2M1, BLOCK_SCALE alone).")
        print("   Manual SASS inspection required to confirm FP4 MMA.")
        return False
    elif has_fallback and (has_weak or has_strong):
        print("⚠️ UNCERTAIN: Mixed patterns - some FP4, some BF16/FP16.")
        print("   Cannot definitively confirm block-scaled FP4 MMA is the main path.")
        print("   Manual SASS inspection required.")
        return False
    elif has_fallback:
        print("❌ FAIL: Only BF16/FP16 MMA found, NO FP4 indicators!")
        print("   The kernel is NOT using native FP4 tensor cores.")
        print("   Check: Is CUTLASS using SM120 block-scaled builders?")
        return False
    else:
        print("❌ FAIL: No MMA patterns found at all.")
        print("   The compiled binary may not contain the expected kernel.")
        print("   Check: Is the kernel instantiated? Is target arch SM121?")
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
