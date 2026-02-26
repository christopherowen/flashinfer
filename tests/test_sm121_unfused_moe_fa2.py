import pytest
import torch
from torch.nn import functional as F

from flashinfer.fused_moe.core import (
    ActivationType,
    cutlass_fused_moe,
    get_cutlass_fused_moe_module,
)
from flashinfer.prefill import single_prefill_with_kv_cache
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper


def is_sm121() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (12, 1)


def _compute_routing(router_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.to(torch.float32), selected_experts.to(torch.int32)


def _reference_unfused_moe(
    x: torch.Tensor,
    w31_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    num_tokens = x.shape[0]
    top_k = selected_experts.shape[1]
    out = torch.zeros((num_tokens, w2_weight.shape[1]), device=x.device, dtype=x.dtype)
    for token_idx in range(num_tokens):
        token_state = x[token_idx]
        token_out = torch.zeros((w2_weight.shape[1],), device=x.device, dtype=x.dtype)
        for k_idx in range(top_k):
            expert = int(selected_experts[token_idx, k_idx].item())
            weight = routing_weights[token_idx, k_idx].to(x.dtype)
            w3_expert, w1_expert = torch.chunk(w31_weight[expert], 2, dim=0)
            inter = F.silu(token_state @ w1_expert.t()) * (token_state @ w3_expert.t())
            token_out += weight * (inter @ w2_weight[expert].t())
        out[token_idx] = token_out
    return out


@pytest.mark.skipif(not is_sm121(), reason="Requires SM121 GPU")
def test_sm121_unfused_moe_modules_load_for_production_tiles():
    for tile_mn in ((64, 128), (128, 128)):
        module = get_cutlass_fused_moe_module("121", tile_mn=tile_mn, fuse_activation=False)
        assert hasattr(module, "cutlass_fused_moe")


@pytest.mark.skipif(not is_sm121(), reason="Requires SM121 GPU")
@pytest.mark.parametrize("num_tokens", [16, 96])
def test_sm121_unfused_moe_matches_reference_for_decode_prefill_tiles(num_tokens: int):
    torch.manual_seed(0)
    hidden_size = 128
    intermediate_size = 128
    num_experts = 2
    top_k = 1

    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16) / 8
    router_logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    w31_weight = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16)
        / 8
    )
    w2_weight = (
        torch.randn(num_experts, hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16)
        / 8
    )

    routing_weights, selected_experts = _compute_routing(router_logits, top_k)
    ref = _reference_unfused_moe(x, w31_weight, w2_weight, selected_experts, routing_weights)

    out = cutlass_fused_moe(
        x,
        selected_experts,
        routing_weights,
        w31_weight,
        w2_weight,
        output_dtype=x.dtype,
        quant_scales=None,
        auto_tile_select=True,
        fuse_activation=False,
        activation_type=ActivationType.Swiglu,
    )
    torch.testing.assert_close(out, ref, rtol=6e-2, atol=6e-2)


@pytest.mark.skipif(not is_sm121(), reason="Requires SM121 GPU")
def test_sm121_fa2_prefill_smoke():
    torch.manual_seed(1)
    q = torch.randn(16, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(64, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(64, 4, 128, device="cuda", dtype=torch.bfloat16)
    out = single_prefill_with_kv_cache(q, k, v, backend="fa2")
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not is_sm121(), reason="Requires SM121 GPU")
def test_sm121_fa2_decode_wrapper_smoke():
    torch.manual_seed(2)
    kv_layout = "HND"
    num_qo_heads = 4
    num_kv_heads = 4
    page_size = 16
    head_dim = 128
    batch_size = 2
    num_pages = 8

    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_data = torch.randn(
        num_pages, 2, num_kv_heads, page_size, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_indptr = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.tensor([8, 12], dtype=torch.int32, device="cuda")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    wrapper.plan(
        kv_indptr.cpu(),
        kv_indices,
        kv_last_page_len.cpu(),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.bfloat16,
        q_data_type=torch.bfloat16,
    )
    out = wrapper.run(q, kv_data)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
