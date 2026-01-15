#!/usr/bin/env python3
"""Test SM120 MoE tile expansion compilation.

This script validates all valid tile configurations for SM120/SM121 (GB10).

Tile configurations are constrained by:
1. Shared memory capacity: 101KB (need >= 2 pipeline stages)
2. MMA hardware: M must be >= 64 (or use swap_ab for logical M < 64)
3. N must be power of 2 (MMA constraint)

Total configurations: 23 native + 6 swapped = 29 tiles
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Force minimal fused-MoE build: compile only MXFP4 path.
os.environ["FLASHINFER_FUSED_MOE_BUILD_PROFILE"] = "mxfp4_minimal"

# Detect environment and set paths
if Path("/workspace/flashinfer").exists():
    # Docker environment
    FLASHINFER_PATH = "/workspace/flashinfer"
    VLLM_PATH = "/workspace/vllm"
else:
    # Local environment (assume sibling directories)
    project_root = Path(__file__).parent.parent.resolve()
    potential_root_1 = project_root.parent.parent
    if (potential_root_1 / "flashinfer").exists():
        workspace_root = potential_root_1
    else:
        workspace_root = project_root.parent

    FLASHINFER_PATH = str(workspace_root / "flashinfer")
    VLLM_PATH = str(workspace_root / "vllm")

print(f"Using FlashInfer path: {FLASHINFER_PATH}")
print(f"Using vLLM path: {VLLM_PATH}")

os.environ["PYTHONPATH"] = f"{FLASHINFER_PATH}:{VLLM_PATH}:{os.environ.get('PYTHONPATH', '')}"
sys.path.insert(0, FLASHINFER_PATH)
sys.path.insert(0, VLLM_PATH)


# =============================================================================
# Tile Configuration Definitions
# =============================================================================

SMEM_CAPACITY = 101376  # bytes (101KB on SM121)
MIN_STAGES = 2
EPILOGUE_CARVEOUT = 4096  # bytes reserved for epilogue


def calc_stage_bytes(m: int, n: int, k: int = 128) -> int:
    """Calculate bytes per pipeline stage for a tile configuration."""
    a_bytes = m * k  # FP4 stored as uint8
    b_bytes = n * k
    sf_a = (m // 32 + 1) * (k // 32) * 4  # Scale factors for A
    sf_b = (n // 32 + 1) * (k // 32) * 4  # Scale factors for B
    pipeline = 1024  # Pipeline overhead
    return a_bytes + b_bytes + sf_a + sf_b + pipeline


def calc_stages(m: int, n: int) -> float:
    """Calculate number of pipeline stages for a tile configuration."""
    stage_bytes = calc_stage_bytes(m, n)
    available = SMEM_CAPACITY - EPILOGUE_CARVEOUT
    return available / stage_bytes


@dataclass
class TileConfig:
    """A tile configuration for testing."""
    logical_m: int
    logical_n: int
    physical_m: int  # Actual CTA tile M (may differ if swapped)
    physical_n: int  # Actual CTA tile N
    swap_ab: bool
    stages: float
    category: str  # 'native' or 'swapped'
    
    @property
    def name(self) -> str:
        if self.swap_ab:
            return f"({self.logical_m},{self.logical_n}) [swap from ({self.physical_m},{self.physical_n})]"
        return f"({self.logical_m},{self.logical_n})"
    
    @property
    def tile_mn(self) -> Tuple[int, int]:
        """The tile_mn tuple to pass to get_cutlass_fused_moe_module."""
        return (self.logical_m, self.logical_n)


def generate_all_tile_configs() -> List[TileConfig]:
    """Generate all valid tile configurations.
    
    Constraints:
    - M must be power-of-2 multiple of 64 (CUTE shape divisibility)
    - N must be power of 2 (MMA constraint)
    - smem capacity must allow >= 2 pipeline stages
    - For M < 64, swap_ab is used (physical tile is swapped)
    """
    configs = []
    
    # Native tiles: M >= 64, no swap needed
    # M must be 64, 128, or 256 (192, 320 fail CUTE divisibility)
    m_values = [64, 128, 256]
    n_values = [8, 16, 32, 64, 128, 256]
    
    for m in m_values:
        for n in n_values:
            stages = calc_stages(m, n)
            if stages >= MIN_STAGES:
                configs.append(TileConfig(
                    logical_m=m,
                    logical_n=n,
                    physical_m=m,
                    physical_n=n,
                    swap_ab=False,
                    stages=stages,
                    category='native'
                ))
    
    # Swapped tiles: logical M < 64, uses swap_ab
    # Physical (M, N) -> Logical (N, M) after swap
    swap_phys_m = [64, 128]  # Physical M must be >= 64
    swap_phys_n = [8, 16, 32]  # Physical N becomes logical M
    
    for phys_m in swap_phys_m:
        for phys_n in swap_phys_n:
            stages = calc_stages(phys_m, phys_n)
            if stages >= MIN_STAGES:
                configs.append(TileConfig(
                    logical_m=phys_n,  # After swap
                    logical_n=phys_m,  # After swap
                    physical_m=phys_m,
                    physical_n=phys_n,
                    swap_ab=True,
                    stages=stages,
                    category='swapped'
                ))
    
    return configs


# All valid configurations
ALL_TILE_CONFIGS = generate_all_tile_configs()

# Group by category
NATIVE_TILES = [c for c in ALL_TILE_CONFIGS if c.category == 'native']
SWAPPED_TILES = [c for c in ALL_TILE_CONFIGS if c.category == 'swapped']


# =============================================================================
# Test Functions
# =============================================================================

def test_module_loading():
    """Test that FlashInfer modules can be imported."""
    print("=" * 70)
    print("Testing FlashInfer module import...")
    print("=" * 70)
    
    import flashinfer
    print(f"FlashInfer path: {flashinfer.__file__}")
    
    from flashinfer.fused_moe.core import (
        get_cutlass_fused_moe_module,
        select_tile_mn_for_sm120,
    )
    
    print(f"select_tile_mn_for_sm120(1) = {select_tile_mn_for_sm120(1)}")
    print(f"select_tile_mn_for_sm120(32) = {select_tile_mn_for_sm120(32)}")
    print(f"select_tile_mn_for_sm120(128) = {select_tile_mn_for_sm120(128)}")
    print()


def test_single_tile(config: TileConfig, verbose: bool = True) -> Tuple[bool, float, str]:
    """Test compilation of a single tile configuration.
    
    Returns: (success, elapsed_time, error_message)
    """
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    
    if verbose:
        swap_info = " [swap_ab]" if config.swap_ab else ""
        print(f"  Testing {config.name}{swap_info}...", end=" ", flush=True)
    
    start = time.time()
    try:
        module = get_cutlass_fused_moe_module(
            backend="121",  # SM121 for GB10
            use_fast_build=False,
            tile_mn=config.tile_mn,
        )
        elapsed = time.time() - start
        if verbose:
            print(f"OK ({elapsed:.1f}s, {config.stages:.1f} stages)")
        return True, elapsed, ""
    except Exception as e:
        elapsed = time.time() - start
        error_msg = str(e)
        if verbose:
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"    Error: {error_msg[:100]}...")
        return False, elapsed, error_msg


def test_all_tiles(configs: List[TileConfig], category_name: str) -> Tuple[int, int]:
    """Test all tiles in a category.
    
    Returns: (passed, failed)
    """
    print()
    print("=" * 70)
    print(f" {category_name} ({len(configs)} tiles)")
    print("=" * 70)
    
    passed = 0
    failed = 0
    failed_configs = []
    
    for config in configs:
        success, elapsed, error = test_single_tile(config)
        if success:
            passed += 1
        else:
            failed += 1
            failed_configs.append((config, error))
    
    if failed_configs:
        print()
        print("  Failed tiles:")
        for config, error in failed_configs:
            print(f"    - {config.name}: {error[:80]}")
    
    return passed, failed


def print_tile_matrix():
    """Print a matrix showing all tile configurations."""
    print()
    print("=" * 70)
    print(" Tile Configuration Matrix")
    print("=" * 70)
    
    # Native tiles
    print()
    print("NATIVE TILES (M >= 64, no swap):")
    print("-" * 70)
    print(f"{'M \\ N':<8}", end="")
    n_vals = [8, 16, 32, 64, 128, 256]
    for n in n_vals:
        print(f"{n:>8}", end="")
    print()
    print("-" * 70)
    
    m_vals = [64, 128, 256]  # Must be power-of-2 multiple of 64
    for m in m_vals:
        print(f"{m:<8}", end="")
        for n in n_vals:
            stages = calc_stages(m, n)
            if stages >= MIN_STAGES:
                print(f"{stages:>7.1f}s", end="")
            else:
                print(f"{'---':>8}", end="")
        print()
    
    # Swapped tiles
    print()
    print("SWAPPED TILES (logical M < 64, uses swap_ab):")
    print("-" * 70)
    print(f"{'Logical':<15} {'Physical':<15} {'Stages':<10} {'Use Case':<20}")
    print("-" * 70)
    for config in SWAPPED_TILES:
        print(f"({config.logical_m:>3},{config.logical_n:>3})       "
              f"({config.physical_m:>3},{config.physical_n:>3})       "
              f"{config.stages:>6.1f}     "
              f"Decode batch={config.logical_m}")
    
    print()
    print(f"Total: {len(NATIVE_TILES)} native + {len(SWAPPED_TILES)} swapped = {len(ALL_TILE_CONFIGS)} tiles")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SM120 tile configurations")
    parser.add_argument("--quick", action="store_true", 
                        help="Quick test: only test a subset of tiles")
    parser.add_argument("--native-only", action="store_true",
                        help="Only test native tiles (no swap)")
    parser.add_argument("--swapped-only", action="store_true",
                        help="Only test swapped tiles")
    parser.add_argument("--matrix", action="store_true",
                        help="Just print the tile matrix, don't test")
    parser.add_argument("--tile", type=str, default=None,
                        help="Test a specific tile, e.g., '128,64'")
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print(" SM120/SM121 Tile Configuration Test Suite")
    print("=" * 70 + "\n")
    
    print(f"SMEM capacity: {SMEM_CAPACITY:,} bytes")
    print(f"Minimum stages: {MIN_STAGES}")
    print(f"Epilogue carveout: {EPILOGUE_CARVEOUT:,} bytes")
    
    if args.matrix:
        print_tile_matrix()
        return 0
    
    # Test module loading first
    try:
        test_module_loading()
    except Exception as e:
        print(f"Failed to load FlashInfer module: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test specific tile if requested
    if args.tile:
        m, n = map(int, args.tile.split(","))
        # Find the config
        matching = [c for c in ALL_TILE_CONFIGS 
                    if c.logical_m == m and c.logical_n == n]
        if not matching:
            # Try to create a config anyway
            stages = calc_stages(m, n) if m >= 64 else calc_stages(n, m)
            swap = m < 64
            config = TileConfig(m, n, n if swap else m, m if swap else n, 
                               swap, stages, 'custom')
            matching = [config]
        
        print(f"\nTesting specific tile: ({m}, {n})")
        success, elapsed, error = test_single_tile(matching[0])
        if success:
            print(f"\n✓ Tile ({m},{n}) compiled successfully in {elapsed:.1f}s")
            return 0
        else:
            print(f"\n✗ Tile ({m},{n}) FAILED: {error}")
            return 1
    
    # Determine which tiles to test
    if args.quick:
        # Quick test: representative subset
        quick_tiles = [
            (64, 8), (64, 128),  # Small M native
            (128, 32), (128, 128),  # Medium M native
            (256, 32), (256, 64),  # Large M native
            (8, 64), (32, 128),  # Swapped
        ]
        test_configs = [c for c in ALL_TILE_CONFIGS 
                        if (c.logical_m, c.logical_n) in quick_tiles]
        category_name = "Quick Test (representative subset)"
    elif args.native_only:
        test_configs = NATIVE_TILES
        category_name = "Native Tiles Only"
    elif args.swapped_only:
        test_configs = SWAPPED_TILES
        category_name = "Swapped Tiles Only"
    else:
        test_configs = ALL_TILE_CONFIGS
        category_name = "All Tile Configurations"
    
    # Run tests
    total_start = time.time()
    
    if args.native_only or args.swapped_only or args.quick:
        passed, failed = test_all_tiles(test_configs, category_name)
    else:
        # Test native and swapped separately for clarity
        native_passed, native_failed = test_all_tiles(NATIVE_TILES, "Native Tiles")
        swapped_passed, swapped_failed = test_all_tiles(SWAPPED_TILES, "Swapped Tiles")
        passed = native_passed + swapped_passed
        failed = native_failed + swapped_failed
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print()
    print("=" * 70)
    print(" Test Summary")
    print("=" * 70)
    print(f"  Total tiles tested: {passed + failed}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print()
    
    if failed == 0:
        print("✓ All tile configurations compiled successfully!")
        return 0
    else:
        print(f"✗ {failed} tile configuration(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
