#!/bin/bash
# Verify SM121 MMA instructions for MXFP4
# Usage: ./scripts/verify_sm121_mma.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_CU="$SCRIPT_DIR/compile_mxfp4_kernel_test.cu"
OUTPUT_DIR="/tmp/sm121_mma_test"

echo "=============================================="
echo "SM120/SM121 MXFP4 MMA Instruction Verification"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# Step 1: Compile for SM121a
echo ""
echo "Step 1: Compiling test kernel for SM121a..."
nvcc -arch=sm_121a -c "$TEST_CU" -o "$OUTPUT_DIR/test_mxfp4.o" 2>&1 && \
    echo "  ✓ Compilation successful" || \
    { echo "  ✗ Compilation failed - SM121 f8f6f4 MMA may not be supported"; exit 1; }

# Step 2: Extract PTX
echo ""
echo "Step 2: Generating PTX..."
nvcc -arch=sm_121a -ptx "$TEST_CU" -o "$OUTPUT_DIR/test_mxfp4.ptx" 2>&1

echo "  Searching PTX for MMA instructions..."
if grep -q "mma.sync.aligned.kind::f8f6f4" "$OUTPUT_DIR/test_mxfp4.ptx"; then
    echo "  ✓ Found f8f6f4 MMA in PTX:"
    grep "mma.sync.aligned.kind::f8f6f4" "$OUTPUT_DIR/test_mxfp4.ptx" | head -5
else
    echo "  ✗ No f8f6f4 MMA found in PTX"
    echo "  PTX saved to: $OUTPUT_DIR/test_mxfp4.ptx"
fi

# Step 3: Generate cubin and extract SASS
echo ""
echo "Step 3: Generating cubin and SASS..."
nvcc -arch=sm_121a -cubin "$TEST_CU" -o "$OUTPUT_DIR/test_mxfp4.cubin" 2>&1

echo "  Disassembling cubin..."
cuobjdump --dump-sass "$OUTPUT_DIR/test_mxfp4.cubin" > "$OUTPUT_DIR/test_mxfp4.sass" 2>&1

echo ""
echo "Step 4: Searching SASS for tensor core instructions..."

# Search patterns
patterns=("IMMA" "HMMA" "DMMA" "BMMA" "MMA" "tcgen05" "QMMA")
found_any=false

for pattern in "${patterns[@]}"; do
    count=$(grep -c "$pattern" "$OUTPUT_DIR/test_mxfp4.sass" 2>/dev/null || echo "0")
    if [ "$count" -gt 0 ]; then
        found_any=true
        echo "  Found $count '$pattern' instructions:"
        grep "$pattern" "$OUTPUT_DIR/test_mxfp4.sass" | head -3
        echo ""
    fi
done

if [ "$found_any" = false ]; then
    echo "  No tensor core instructions found in SASS"
    echo "  This may be expected for a minimal test kernel"
fi

echo ""
echo "Files generated:"
echo "  PTX:   $OUTPUT_DIR/test_mxfp4.ptx"
echo "  cubin: $OUTPUT_DIR/test_mxfp4.cubin"
echo "  SASS:  $OUTPUT_DIR/test_mxfp4.sass"

echo ""
echo "To inspect the full SASS output:"
echo "  less $OUTPUT_DIR/test_mxfp4.sass"

echo ""
echo "=============================================="
echo "For the actual CUTLASS fused MoE kernel:"
echo "=============================================="
echo ""
echo "1. Run vLLM with MXFP4 to trigger JIT compilation"
echo "2. Find the compiled .so:"
echo "   find ~/.cache/flashinfer -name '*.so' -path '*fused_moe_120*'"
echo "3. Disassemble:"
echo "   cuobjdump --dump-sass <path_to_so> | grep -i 'mma\\|tcgen05'"


