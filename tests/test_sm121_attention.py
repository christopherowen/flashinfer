"""
Test FlashAttention backends on SM121 (GB10 / DGX Spark).

This test verifies which attention backends work on SM121 architecture.
"""
import pytest
import torch

from flashinfer.utils import (
    get_compute_capability,
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm120a_supported,
    is_sm121a_supported,
)


def is_sm121():
    """Check if current device is SM121."""
    if not torch.cuda.is_available():
        return False
    cc = get_compute_capability(torch.device("cuda:0"))
    return cc == (12, 1)


@pytest.mark.skipif(not is_sm121(), reason="Requires SM121 GPU")
class TestSM121Attention:
    """Test attention backends on SM121."""

    @pytest.fixture
    def test_tensors(self):
        """Create test tensors for attention."""
        device = torch.device("cuda:0")
        num_heads = 4
        head_dim = 128
        seq_len_q = 16
        seq_len_kv = 64

        q = torch.randn(seq_len_q, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(seq_len_kv, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(seq_len_kv, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        return q, k, v

    def test_architecture_detection(self):
        """Test that SM121 is correctly detected."""
        device = torch.device("cuda:0")
        cc = get_compute_capability(device)
        
        assert cc == (12, 1), f"Expected SM121, got SM{cc[0]}{cc[1]}"
        assert is_sm121a_supported(device), "SM121 should be supported"
        # SM121 is different from SM90a (Hopper) and SM100a (Blackwell B100)
        assert not is_sm90a_supported(device), "SM121 should not report as SM90a"

    def test_fa3_not_available(self, test_tensors):
        """FA3 should NOT work on SM121 (compiled for SM90a only)."""
        from flashinfer.prefill import single_prefill_with_kv_cache
        
        q, k, v = test_tensors
        
        with pytest.raises(RuntimeError, match="no kernel image"):
            single_prefill_with_kv_cache(q, k, v, backend="fa3")

    def test_fa2_works(self, test_tensors):
        """FA2 should work on SM121."""
        from flashinfer.prefill import single_prefill_with_kv_cache
        
        q, k, v = test_tensors
        
        out = single_prefill_with_kv_cache(q, k, v, backend="fa2")
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_auto_backend_works(self, test_tensors):
        """Auto backend should work on SM121 (falls back to FA2)."""
        from flashinfer.prefill import single_prefill_with_kv_cache
        
        q, k, v = test_tensors
        
        out = single_prefill_with_kv_cache(q, k, v, backend="auto")
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_cutlass_fmha_module_loads(self):
        """CUTLASS FMHA module should load on SM121."""
        from flashinfer.prefill import get_fmha_module
        
        device = torch.device("cuda:0")
        
        mod = get_fmha_module(
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
            dtype_o=torch.bfloat16,
            dtype_idx=torch.int32,
            head_dim_qk=128,
            head_dim_vo=128,
            pos_encoding_mode=0,
            use_sliding_window=False,
            use_logits_soft_cap=False,
            device=device,
        )
        
        assert hasattr(mod, "run"), "Module should have run method"
        assert hasattr(mod, "plan"), "Module should have plan method"


if __name__ == "__main__":
    # Quick test when run directly
    if not is_sm121():
        print("This test requires SM121 GPU. Skipping.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        cc = get_compute_capability(torch.device("cuda:0"))
        print(f"Compute capability: {cc[0]}.{cc[1]}")
        print()
        
        test = TestSM121Attention()
        test.test_architecture_detection()
        print("✅ Architecture detection passed")
        
        tensors = (
            torch.randn(16, 4, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 4, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 4, 128, device="cuda", dtype=torch.bfloat16),
        )
        
        try:
            test.test_fa3_not_available(tensors)
            print("✅ FA3 correctly fails on SM121")
        except AssertionError:
            print("❌ FA3 unexpectedly worked on SM121")
        
        test.test_fa2_works(tensors)
        print("✅ FA2 works on SM121")
        
        test.test_auto_backend_works(tensors)
        print("✅ Auto backend works on SM121")
        
        test.test_cutlass_fmha_module_loads()
        print("✅ CUTLASS FMHA module loads on SM121")
        
        print()
        print("All SM121 attention tests passed!")

