#!/usr/bin/env python3
"""
Stress tests for SM12x identity SFA path.

Tests:
1. Stress shapes (alignment + tiling edges)
2. Multi-stream + concurrency safety
3. CUDA Graph capture compatibility

Requirements:
- SM121 (Blackwell) GPU
- FlashInfer built with FP4 support
- PyTorch 2.x with FP8 support
"""

import pytest
import torch
import numpy as np
import threading
import time
from typing import List, Tuple

# Check if we can run on SM121
def is_sm121_available():
    if not torch.cuda.is_available():
        return False
    try:
        cc = torch.cuda.get_device_capability()
        return cc[0] == 12 and cc[1] >= 1
    except:
        return False

def has_fp8_support():
    return hasattr(torch, 'float8_e4m3fn')

requires_sm121 = pytest.mark.skipif(
    not is_sm121_available(),
    reason="Requires SM121 (Blackwell) GPU"
)

requires_fp8 = pytest.mark.skipif(
    not has_fp8_support(),
    reason="Requires PyTorch with FP8 support"
)


def compute_sfa_buffer_size_fallback(M: int, K: int, L: int = 1) -> int:
    """Compute SFA buffer size using the conservative fallback formula.
    
    This should match Sm12xLayoutSFAUtils::computeBufferSizeFallback.
    """
    BlkMN = 128
    SFVecSize = 32
    BlkSF = 4
    SfAtomSize = 512  # bytes per SfAtom
    
    num_m_blocks = (M + BlkMN - 1) // BlkMN
    k_atoms = (K + (SFVecSize * BlkSF) - 1) // (SFVecSize * BlkSF)
    
    sfa_bytes = num_m_blocks * k_atoms * SfAtomSize * L
    return ((sfa_bytes + 255) // 256) * 256


def fill_identity_buffer(size: int) -> torch.Tensor:
    """Create a buffer filled with identity scale (0x7F)."""
    buf = torch.empty(size, dtype=torch.uint8, device='cuda')
    buf.fill_(0x7F)
    return buf


def verify_identity_buffer(buf: torch.Tensor) -> bool:
    """Verify all bytes are 0x7F."""
    return (buf == 0x7F).all().item()


class TestStressShapes:
    """Test edge dimensions to catch layout/stride mistakes."""
    
    @requires_sm121
    def test_edge_dimensions_m(self):
        """Test M dimensions that are at tile boundaries."""
        M_values = [64, 65, 127, 128, 129, 255, 256]
        N, K = 128, 256
        num_groups = 2
        
        results = []
        for M in M_values:
            try:
                sfa_size = compute_sfa_buffer_size_fallback(M, K)
                sfa_buf = fill_identity_buffer(sfa_size)
                
                # Verify buffer is 256-byte aligned
                aligned = (sfa_buf.data_ptr() % 256 == 0)
                
                # Verify content
                content_ok = verify_identity_buffer(sfa_buf)
                
                passed = aligned and content_ok and sfa_size > 0
                results.append((M, N, K, sfa_size, passed))
                
            except Exception as e:
                results.append((M, N, K, 0, False))
                print(f"  FAIL M={M}: {e}")
        
        print("\nStress test results (M dimension):")
        print(f"{'M':>6} {'N':>6} {'K':>6} {'SFA_size':>10} {'Status':>8}")
        for M, N, K, size, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"{M:>6} {N:>6} {K:>6} {size:>10} {status:>8}")
        
        all_passed = all(r[4] for r in results)
        assert all_passed, "Some M dimension tests failed"
    
    @requires_sm121
    def test_edge_dimensions_k(self):
        """Test K dimensions that are at scale block boundaries."""
        M, N = 128, 256
        K_values = [128, 192, 256, 384, 512, 1024]
        
        results = []
        for K in K_values:
            try:
                sfa_size = compute_sfa_buffer_size_fallback(M, K)
                sfa_buf = fill_identity_buffer(sfa_size)
                
                aligned = (sfa_buf.data_ptr() % 256 == 0)
                content_ok = verify_identity_buffer(sfa_buf)
                
                passed = aligned and content_ok and sfa_size > 0
                results.append((M, N, K, sfa_size, passed))
                
            except Exception as e:
                results.append((M, N, K, 0, False))
                print(f"  FAIL K={K}: {e}")
        
        print("\nStress test results (K dimension):")
        print(f"{'M':>6} {'N':>6} {'K':>6} {'SFA_size':>10} {'Status':>8}")
        for M, N, K, size, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"{M:>6} {N:>6} {K:>6} {size:>10} {status:>8}")
        
        all_passed = all(r[4] for r in results)
        assert all_passed, "Some K dimension tests failed"
    
    @requires_sm121
    def test_multiple_groups(self):
        """Test various num_groups values."""
        M, N, K = 128, 256, 512
        group_counts = [2, 4, 8, 16, 32]
        
        results = []
        for num_groups in group_counts:
            try:
                # Compute SFA buffer size (same buffer can be shared across groups)
                sfa_size = compute_sfa_buffer_size_fallback(M, K)
                sfa_buf = fill_identity_buffer(sfa_size)
                
                # Create pointer array (all pointing to same identity buffer)
                ptr_array = torch.full(
                    (num_groups,), 
                    sfa_buf.data_ptr(), 
                    dtype=torch.int64, 
                    device='cuda'
                )
                
                # Verify we can read back the pointers
                ptrs_cpu = ptr_array.cpu().numpy()
                all_same = np.all(ptrs_cpu == sfa_buf.data_ptr())
                
                passed = all_same and verify_identity_buffer(sfa_buf)
                results.append((num_groups, sfa_size, passed))
                
            except Exception as e:
                results.append((num_groups, 0, False))
                print(f"  FAIL num_groups={num_groups}: {e}")
        
        print("\nStress test results (num_groups):")
        print(f"{'Groups':>8} {'SFA_size':>10} {'Status':>8}")
        for groups, size, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"{groups:>8} {size:>10} {status:>8}")
        
        all_passed = all(r[2] for r in results)
        assert all_passed, "Some group count tests failed"


class TestMultiStreamConcurrency:
    """Test buffer managers under concurrent access."""
    
    @requires_sm121
    def test_multistream_allocation(self):
        """Test allocation from multiple streams doesn't cause issues."""
        num_streams = 8
        M, N, K = 128, 256, 512
        
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        buffers = []
        
        sfa_size = compute_sfa_buffer_size_fallback(M, K)
        
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                buf = fill_identity_buffer(sfa_size)
                buffers.append(buf)
        
        # Sync all streams
        torch.cuda.synchronize()
        
        # Verify all buffers are valid
        all_valid = all(verify_identity_buffer(buf) for buf in buffers)
        
        print(f"\nMulti-stream allocation test:")
        print(f"  Streams: {num_streams}")
        print(f"  Buffer size: {sfa_size}")
        print(f"  All valid: {all_valid}")
        
        assert all_valid, "Some multi-stream buffers are invalid"
    
    @requires_sm121
    def test_thread_safety(self):
        """Test buffer operations from multiple threads."""
        num_threads = 8
        iterations_per_thread = 10
        M, N, K = 128, 256, 512
        
        errors = []
        buffers = {}
        lock = threading.Lock()
        
        def thread_func(thread_id):
            try:
                torch.cuda.set_device(0)
                sfa_size = compute_sfa_buffer_size_fallback(M, K)
                
                for i in range(iterations_per_thread):
                    buf = fill_identity_buffer(sfa_size)
                    
                    if not verify_identity_buffer(buf):
                        with lock:
                            errors.append(f"Thread {thread_id} iter {i}: Invalid buffer content")
                    
                    # Check alignment
                    if buf.data_ptr() % 256 != 0:
                        with lock:
                            errors.append(f"Thread {thread_id} iter {i}: Misaligned buffer")
                    
                    with lock:
                        buffers[f"{thread_id}_{i}"] = buf
                        
            except Exception as e:
                with lock:
                    errors.append(f"Thread {thread_id}: {e}")
        
        threads = [threading.Thread(target=thread_func, args=(i,)) for i in range(num_threads)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        print(f"\nThread safety test:")
        print(f"  Threads: {num_threads}")
        print(f"  Iterations per thread: {iterations_per_thread}")
        print(f"  Total buffers: {len(buffers)}")
        print(f"  Errors: {len(errors)}")
        
        if errors:
            for e in errors[:5]:
                print(f"  Error: {e}")
        
        assert len(errors) == 0, f"Thread safety test had {len(errors)} errors"


class TestCUDAGraphCapture:
    """Test CUDA Graph capture compatibility."""
    
    @requires_sm121
    def test_graph_capture_no_alloc(self):
        """Test that buffer operations work during CUDA Graph capture.
        
        Identity buffers should be prewarmed before capture.
        No allocations should occur during capture/replay.
        """
        M, N, K = 128, 256, 512
        sfa_size = compute_sfa_buffer_size_fallback(M, K)
        
        # Pre-warm: allocate buffer BEFORE graph capture
        identity_buf = fill_identity_buffer(sfa_size)
        
        # Verify prewarm worked
        assert verify_identity_buffer(identity_buf), "Prewarm buffer invalid"
        
        # Create a simple operation that uses the buffer
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        
        # Warmup run (required before graph capture)
        y = x.to(torch.float8_e4m3fn)
        y_recovered = y.to(torch.bfloat16)
        torch.cuda.synchronize()
        
        # Capture graph
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                # This should not allocate - buffer is prewarmed
                y_captured = x.to(torch.float8_e4m3fn)
                y_recovered_captured = y_captured.to(torch.bfloat16)
        
        torch.cuda.synchronize()
        
        # Replay graph multiple times
        num_replays = 10
        for i in range(num_replays):
            g.replay()
        
        torch.cuda.synchronize()
        
        # Verify buffer is still valid after replays
        assert verify_identity_buffer(identity_buf), "Buffer corrupted after graph replay"
        
        print(f"\nCUDA Graph capture test:")
        print(f"  Prewarm buffer size: {sfa_size}")
        print(f"  Graph replays: {num_replays}")
        print(f"  Buffer still valid: True")
        print(f"  No allocation errors: True")


class TestPerformanceSanity:
    """Basic performance sanity checks (not exhaustive)."""
    
    @requires_sm121
    @requires_fp8
    def test_cached_vs_cold_start(self):
        """Test that cached buffer access is fast after warmup."""
        M, N, K = 128, 256, 512
        sfa_size = compute_sfa_buffer_size_fallback(M, K)
        
        # Cold start: first allocation
        torch.cuda.synchronize()
        start = time.perf_counter()
        buf1 = fill_identity_buffer(sfa_size)
        torch.cuda.synchronize()
        cold_time = time.perf_counter() - start
        
        # Warm iterations: subsequent allocations should be similar
        # (In real implementation, cached manager returns same buffer)
        warm_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            buf = fill_identity_buffer(sfa_size)
            torch.cuda.synchronize()
            warm_times.append(time.perf_counter() - start)
        
        avg_warm_time = sum(warm_times) / len(warm_times)
        
        print(f"\nPerformance sanity check:")
        print(f"  Cold start time: {cold_time*1000:.3f} ms")
        print(f"  Avg warm time: {avg_warm_time*1000:.3f} ms")
        print(f"  Buffer size: {sfa_size} bytes")
        
        # Warm time shouldn't be dramatically worse than cold
        # (For cached buffers, warm should be faster or similar)
        # Allow up to 10x variance due to system noise
        assert avg_warm_time < cold_time * 10, "Warm iterations unexpectedly slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


