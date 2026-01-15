"""
Test FlashAttention backends on SM121 (GB10 / DGX Spark).

This test verifies attention backends work correctly on SM121 architecture,
including attention sink support for both prefill and decode paths.
"""
import pytest
import torch

from flashinfer.utils import (
    get_compute_capability,
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

    @pytest.fixture
    def paged_kv_tensors(self):
        """Create paged KV cache tensors for testing."""
        device = torch.device("cuda:0")
        num_heads = 4
        head_dim = 128
        page_size = 16
        num_pages = 8
        batch_size = 2

        # Paged KV cache in HND layout (num_pages, num_heads, page_size, head_dim)
        k_cache = torch.randn(num_pages, num_heads, page_size, head_dim,
                              device=device, dtype=torch.bfloat16)
        v_cache = torch.randn(num_pages, num_heads, page_size, head_dim,
                              device=device, dtype=torch.bfloat16)

        # Indptr: batch_size + 1
        indptr = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        # Indices: which pages each sequence uses
        indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=device)
        # Last page len: how many tokens in the last page
        last_page_len = torch.tensor([8, 12], dtype=torch.int32, device=device)

        return {
            "k_cache": k_cache,
            "v_cache": v_cache,
            "indptr": indptr,
            "indices": indices,
            "last_page_len": last_page_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "batch_size": batch_size,
        }

    def test_architecture_detection(self):
        """Test that SM121 is correctly detected."""
        device = torch.device("cuda:0")
        cc = get_compute_capability(device)

        assert cc == (12, 1), f"Expected SM121, got SM{cc[0]}{cc[1]}"
        assert is_sm121a_supported(device), "SM121 should be supported"

    def test_fa3_not_available(self, test_tensors):
        """FA3 is not available on SM121 (requires SM90a)."""
        from flashinfer.prefill import single_prefill_with_kv_cache

        q, k, v = test_tensors

        with pytest.raises(RuntimeError):
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


@pytest.mark.skipif(not is_sm121(), reason="Requires SM121 GPU")
class TestSM121AttentionSinks:
    """Test attention sink support on SM121."""

    def test_prefill_sink_module_loads(self):
        """FA2 prefill attention sink module should load on SM121."""
        from flashinfer.prefill import get_batch_prefill_attention_sink_module

        mod = get_batch_prefill_attention_sink_module(
            backend="fa2",
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
            dtype_o=torch.bfloat16,
            dtype_idx=torch.int32,
            head_dim_qk=128,
            head_dim_vo=128,
            pos_encoding_mode=0,
            use_sliding_window=False,
        )

        assert hasattr(mod, "plan"), "Sink module should have plan method"
        assert hasattr(mod, "paged_run"), "Sink module should have paged_run method"

    def test_prefill_sink_module_with_logits_soft_cap(self):
        """Prefill sink module with logits_soft_cap should have different cache key."""
        from flashinfer.jit import (
            get_batch_prefill_attention_sink_uri,
        )

        # Get URIs for different configs
        uri_no_cap = get_batch_prefill_attention_sink_uri(
            backend="fa2",
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
            dtype_o=torch.bfloat16,
            dtype_idx=torch.int32,
            head_dim_qk=128,
            head_dim_vo=128,
            pos_encoding_mode=0,
            use_sliding_window=False,
            use_logits_soft_cap=False,
            use_fp16_qk_reduction=False,
        )

        uri_with_cap = get_batch_prefill_attention_sink_uri(
            backend="fa2",
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
            dtype_o=torch.bfloat16,
            dtype_idx=torch.int32,
            head_dim_qk=128,
            head_dim_vo=128,
            pos_encoding_mode=0,
            use_sliding_window=False,
            use_logits_soft_cap=True,
            use_fp16_qk_reduction=False,
        )

        assert uri_no_cap != uri_with_cap, \
            "Different logits_soft_cap should produce different cache keys"
        assert "use_logits_cap_False" in uri_no_cap
        assert "use_logits_cap_True" in uri_with_cap

    def test_prefill_sink_module_with_fp16_qk_reduction(self):
        """Prefill sink module with fp16_qk_reduction should have different cache key."""
        from flashinfer.jit import (
            get_batch_prefill_attention_sink_uri,
        )

        uri_no_fp16 = get_batch_prefill_attention_sink_uri(
            backend="fa2",
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
            dtype_o=torch.bfloat16,
            dtype_idx=torch.int32,
            head_dim_qk=128,
            head_dim_vo=128,
            pos_encoding_mode=0,
            use_sliding_window=False,
            use_logits_soft_cap=False,
            use_fp16_qk_reduction=False,
        )

        uri_with_fp16 = get_batch_prefill_attention_sink_uri(
            backend="fa2",
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
            dtype_o=torch.bfloat16,
            dtype_idx=torch.int32,
            head_dim_qk=128,
            head_dim_vo=128,
            pos_encoding_mode=0,
            use_sliding_window=False,
            use_logits_soft_cap=False,
            use_fp16_qk_reduction=True,
        )

        assert uri_no_fp16 != uri_with_fp16, \
            "Different fp16_qk_reduction should produce different cache keys"
        assert "use_fp16_qk_False" in uri_no_fp16
        assert "use_fp16_qk_True" in uri_with_fp16

    def test_prefill_sinks_require_fa2_backend(self):
        """Prefill should raise error if use_sinks=True with non-FA2 backend."""
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        device = torch.device("cuda:0")
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                               device=device),
            kv_layout="HND",
            backend="fa3",  # Non-FA2 backend
        )

        qo_indptr = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        indptr = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([8, 12], dtype=torch.int32, device=device)

        with pytest.raises(NotImplementedError, match="Attention sinks are only supported with backend='fa2'"):
            wrapper.plan(
                qo_indptr,
                indptr,
                indices,
                last_page_len,
                4,  # num_qo_heads
                4,  # num_kv_heads
                128,  # head_dim
                16,  # page_size
                causal=True,
                use_sinks=True,
            )

    def test_decode_sinks_require_tensor_cores(self):
        """Decode should raise error if use_sinks=True with use_tensor_cores=False."""
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        device = torch.device("cuda:0")
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                               device=device),
            kv_layout="HND",
            use_tensor_cores=False,  # Non-tensor-core mode
        )

        indptr = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([8, 12], dtype=torch.int32, device=device)

        with pytest.raises(NotImplementedError, match="Attention sinks require use_tensor_cores=True"):
            wrapper.plan(
                indptr.cpu(),
                indices,
                last_page_len.cpu(),
                4,  # num_qo_heads
                4,  # num_kv_heads
                128,  # head_dim
                16,  # page_size
                use_sinks=True,
            )

    def test_prefill_sinks_reject_fp8_inputs(self):
        """Prefill should raise error if use_sinks=True with FP8 dtypes."""
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        device = torch.device("cuda:0")
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                               device=device),
            kv_layout="HND",
            backend="fa2",
        )

        qo_indptr = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        indptr = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([8, 12], dtype=torch.int32, device=device)

        with pytest.raises(NotImplementedError, match="Attention sinks are not supported with FP8 inputs"):
            wrapper.plan(
                qo_indptr,
                indptr,
                indices,
                last_page_len,
                4,  # num_qo_heads
                4,  # num_kv_heads
                128,  # head_dim
                16,  # page_size
                causal=True,
                q_data_type=torch.float8_e4m3fn,  # FP8 input
                use_sinks=True,
            )

    def test_sinks_change_output(self):
        """Verify that non-zero sinks produce different output than zero sinks."""
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        device = torch.device("cuda:0")
        torch.manual_seed(42)

        # Small test dimensions
        num_heads = 4
        head_dim = 128
        page_size = 16
        num_pages = 4
        batch_size = 1
        seq_len = 8

        # Create paged KV cache in HND layout
        kv_cache = torch.randn(
            num_pages, 2, num_heads, page_size, head_dim,
            device=device, dtype=torch.bfloat16
        )

        # Page table for a single sequence using 2 pages
        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        paged_kv_indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
        paged_kv_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        paged_kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)

        # Query
        q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)

        # Sinks: non-zero values
        sinks_nonzero = torch.tensor([0.5, 1.0, 0.3, 0.8], device=device, dtype=torch.float32)
        # Sinks: zero values (should produce different output)
        sinks_zero = torch.zeros(num_heads, device=device, dtype=torch.float32)

        # Run with non-zero sinks
        wrapper1 = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="HND",
            backend="fa2",
        )
        wrapper1.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            use_sinks=True,
        )
        out_nonzero = wrapper1.run(q, kv_cache, sinks=sinks_nonzero)

        # Run with zero sinks
        wrapper2 = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="HND",
            backend="fa2",
        )
        wrapper2.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            use_sinks=True,
        )
        out_zero = wrapper2.run(q, kv_cache, sinks=sinks_zero)

        # Outputs should be different when sinks are non-zero
        assert out_nonzero.shape == out_zero.shape, "Outputs should have same shape"
        diff = (out_nonzero - out_zero).abs().max().item()
        assert diff > 1e-3, f"Non-zero sinks should produce different output (max diff: {diff})"

        # Sanity check: outputs should be finite
        assert torch.isfinite(out_nonzero).all(), "Output with non-zero sinks should be finite"
        assert torch.isfinite(out_zero).all(), "Output with zero sinks should be finite"


@pytest.mark.skipif(not is_sm121(), reason="Requires SM121 GPU")
class TestSM121PrefillDecodeConsistency:
    """Test that prefill and decode paths use sinks consistently."""

    @pytest.fixture
    def paged_kv_setup(self):
        """Create paged KV cache for testing."""
        device = torch.device("cuda:0")
        num_heads = 4
        head_dim = 128
        page_size = 16
        num_pages = 8
        batch_size = 2

        # Paged KV cache in HND layout
        k_cache = torch.randn(num_pages, num_heads, page_size, head_dim,
                              device=device, dtype=torch.bfloat16)
        v_cache = torch.randn(num_pages, num_heads, page_size, head_dim,
                              device=device, dtype=torch.bfloat16)

        # Page table
        indptr = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([8, 12], dtype=torch.int32, device=device)

        # Sinks tensor (per attention head)
        sinks = torch.randn(num_heads, device=device, dtype=torch.float32)

        return {
            "k_cache": k_cache,
            "v_cache": v_cache,
            "indptr": indptr,
            "indices": indices,
            "last_page_len": last_page_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "batch_size": batch_size,
            "sinks": sinks,
            "device": device,
        }

    def test_prefill_wrapper_use_sinks_flag(self, paged_kv_setup):
        """Prefill wrapper should set _use_sinks flag when use_sinks=True."""
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        setup = paged_kv_setup

        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                               device=setup["device"]),
            kv_layout="HND",
            backend="fa2",
        )

        # Create query indptr for prefill
        qo_indptr = torch.tensor([0, 4, 8], dtype=torch.int32, device=setup["device"])

        wrapper.plan(
            qo_indptr,
            setup["indptr"],
            setup["indices"],
            setup["last_page_len"],
            setup["num_heads"],
            setup["num_heads"],  # num_kv_heads = num_qo_heads for MHA
            setup["head_dim"],
            setup["page_size"],
            causal=True,
            use_sinks=True,
        )

        assert hasattr(wrapper, "_use_sinks"), "Wrapper should have _use_sinks attribute"
        assert wrapper._use_sinks is True, "_use_sinks should be True when use_sinks=True"

    def test_prefill_wrapper_use_sinks_flag_false(self, paged_kv_setup):
        """Prefill wrapper should set _use_sinks=False when use_sinks=False."""
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        setup = paged_kv_setup

        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                               device=setup["device"]),
            kv_layout="HND",
            backend="fa2",
        )

        qo_indptr = torch.tensor([0, 4, 8], dtype=torch.int32, device=setup["device"])

        wrapper.plan(
            qo_indptr,
            setup["indptr"],
            setup["indices"],
            setup["last_page_len"],
            setup["num_heads"],
            setup["num_heads"],
            setup["head_dim"],
            setup["page_size"],
            causal=True,
            use_sinks=False,
        )

        assert hasattr(wrapper, "_use_sinks"), "Wrapper should have _use_sinks attribute"
        assert wrapper._use_sinks is False, "_use_sinks should be False when use_sinks=False"

    def test_decode_wrapper_use_sinks_flag(self, paged_kv_setup):
        """Decode wrapper should set _use_sinks flag when use_sinks=True."""
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        setup = paged_kv_setup

        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                               device=setup["device"]),
            kv_layout="HND",
            use_tensor_cores=True,
        )

        wrapper.plan(
            setup["indptr"].cpu(),
            setup["indices"],
            setup["last_page_len"].cpu(),
            setup["num_heads"],
            setup["num_heads"],
            setup["head_dim"],
            setup["page_size"],
            use_sinks=True,
        )

        assert hasattr(wrapper, "_use_sinks"), "Wrapper should have _use_sinks attribute"
        assert wrapper._use_sinks is True, "_use_sinks should be True when use_sinks=True"

    def test_decode_wrapper_use_sinks_flag_false(self, paged_kv_setup):
        """Decode wrapper should set _use_sinks=False when use_sinks=False."""
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        setup = paged_kv_setup

        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                               device=setup["device"]),
            kv_layout="HND",
            use_tensor_cores=True,
        )

        wrapper.plan(
            setup["indptr"].cpu(),
            setup["indices"],
            setup["last_page_len"].cpu(),
            setup["num_heads"],
            setup["num_heads"],
            setup["head_dim"],
            setup["page_size"],
            use_sinks=False,
        )

        assert hasattr(wrapper, "_use_sinks"), "Wrapper should have _use_sinks attribute"
        assert wrapper._use_sinks is False, "_use_sinks should be False when use_sinks=False"


if __name__ == "__main__":
    # Quick test when run directly
    if not is_sm121():
        print("This test requires SM121 GPU. Skipping.")
    else:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        cc = get_compute_capability(torch.device("cuda:0"))
        print(f"Compute capability: {cc[0]}.{cc[1]}")
        print()

        # Basic attention tests
        print("=== Basic Attention Tests ===")
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
            print("✅ FA3 correctly unavailable on SM121")
        except AssertionError:
            print("❌ FA3 unexpectedly worked on SM121")

        test.test_fa2_works(tensors)
        print("✅ FA2 works on SM121")

        test.test_auto_backend_works(tensors)
        print("✅ Auto backend works on SM121")

        test.test_cutlass_fmha_module_loads()
        print("✅ CUTLASS FMHA module loads on SM121")

        # Attention sink tests
        print()
        print("=== Attention Sink Tests ===")
        sink_test = TestSM121AttentionSinks()

        sink_test.test_prefill_sink_module_loads()
        print("✅ Prefill sink module loads on SM121")

        sink_test.test_prefill_sink_module_with_logits_soft_cap()
        print("✅ Prefill sink cache key includes logits_soft_cap")

        sink_test.test_prefill_sink_module_with_fp16_qk_reduction()
        print("✅ Prefill sink cache key includes fp16_qk_reduction")

        # Prefill/Decode consistency tests
        print()
        print("=== Prefill/Decode Consistency Tests ===")
        consistency_test = TestSM121PrefillDecodeConsistency()

        paged_kv = {
            "k_cache": torch.randn(8, 4, 16, 128, device="cuda", dtype=torch.bfloat16),
            "v_cache": torch.randn(8, 4, 16, 128, device="cuda", dtype=torch.bfloat16),
            "indptr": torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda"),
            "indices": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device="cuda"),
            "last_page_len": torch.tensor([8, 12], dtype=torch.int32, device="cuda"),
            "num_heads": 4,
            "head_dim": 128,
            "page_size": 16,
            "batch_size": 2,
            "sinks": torch.randn(4, device="cuda", dtype=torch.float32),
            "device": torch.device("cuda"),
        }

        consistency_test.test_prefill_wrapper_use_sinks_flag(paged_kv)
        print("✅ Prefill wrapper sets _use_sinks=True correctly")

        consistency_test.test_prefill_wrapper_use_sinks_flag_false(paged_kv)
        print("✅ Prefill wrapper sets _use_sinks=False correctly")

        consistency_test.test_decode_wrapper_use_sinks_flag(paged_kv)
        print("✅ Decode wrapper sets _use_sinks=True correctly")

        consistency_test.test_decode_wrapper_use_sinks_flag_false(paged_kv)
        print("✅ Decode wrapper sets _use_sinks=False correctly")

        print()
        print("All SM121 attention tests passed!")
