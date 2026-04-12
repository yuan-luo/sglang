import sys

import pytest
import torch
from sgl_kernel import kimi_linear_fused_gate

from sglang.srt.layers.moe.topk import biased_grouped_topk_impl


NUM_EXPERTS = 128
HIDDEN_DIM = 5120


def reference_impl(
    hidden_states,
    router_weights,
    correction_bias,
    topk,
    num_expert_group,
    topk_group,
    renormalize,
):
    """Reference: matmul -> biased_grouped_topk_impl"""
    # Step 1: Router GEMM via torch matmul
    logits = torch.matmul(
        hidden_states.float(), router_weights.float().t()
    )

    # Step 2: biased_grouped_topk_impl (sigmoid + bias + grouped topk)
    topk_weights, topk_ids = biased_grouped_topk_impl(
        hidden_states,
        logits,
        correction_bias,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("num_expert_group,topk_group", [(1, 1)])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("use_correction_bias", [True, False])
def test_kimi_linear_fused_gate(
    num_tokens, topk, num_expert_group, topk_group, renormalize, use_correction_bias
):
    torch.manual_seed(num_tokens + int(renormalize) * 100 + int(use_correction_bias) * 200)

    hidden_states = torch.randn(
        num_tokens, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    router_weights = torch.randn(
        NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )

    if use_correction_bias:
        correction_bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda") * 0.1
    else:
        correction_bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda")

    # Fused kernel
    fused_weights, fused_ids = kimi_linear_fused_gate(
        hidden_states,
        router_weights,
        correction_bias,
        topk=topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        renormalize=renormalize,
    )

    # Reference
    ref_weights, ref_ids = reference_impl(
        hidden_states,
        router_weights,
        correction_bias,
        topk=topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        renormalize=renormalize,
    )

    # Check shapes
    assert fused_weights.shape == (num_tokens, topk), f"Weight shape mismatch: {fused_weights.shape}"
    assert fused_ids.shape == (num_tokens, topk), f"ID shape mismatch: {fused_ids.shape}"
    assert fused_weights.dtype == torch.float32
    assert fused_ids.dtype == torch.int32

    # Check that the same experts are selected (order may differ)
    for m in range(num_tokens):
        fused_expert_set = set(fused_ids[m].tolist())
        ref_expert_set = set(ref_ids[m].tolist())
        assert fused_expert_set == ref_expert_set, (
            f"Token {m}: expert set mismatch. "
            f"Fused: {sorted(fused_expert_set)}, Ref: {sorted(ref_expert_set)}"
        )

    # Check weights match (sort by expert id for comparison)
    for m in range(num_tokens):
        fused_sorted_idx = fused_ids[m].argsort()
        ref_sorted_idx = ref_ids[m].argsort()
        fused_w = fused_weights[m][fused_sorted_idx]
        ref_w = ref_weights[m][ref_sorted_idx]
        assert torch.allclose(fused_w, ref_w, rtol=1e-3, atol=1e-3), (
            f"Token {m}: weight mismatch. "
            f"Max diff: {(fused_w - ref_w).abs().max().item()}"
        )

    # Renormalization check
    if renormalize:
        weight_sums = fused_weights.sum(dim=-1)
        assert torch.allclose(
            weight_sums, torch.ones_like(weight_sums), rtol=1e-3, atol=1e-4
        ), f"Renormalization failed: sums = {weight_sums}"


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_kimi_linear_fused_gate_no_bias(num_tokens):
    """Test with correction_bias=None."""
    torch.manual_seed(42)
    topk = 8

    hidden_states = torch.randn(
        num_tokens, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    router_weights = torch.randn(
        NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )

    # Fused kernel with None bias
    fused_weights, fused_ids = kimi_linear_fused_gate(
        hidden_states,
        router_weights,
        None,  # No bias
        topk=topk,
        num_expert_group=1,
        topk_group=1,
        renormalize=True,
    )

    # Reference with zero bias
    ref_weights, ref_ids = reference_impl(
        hidden_states,
        router_weights,
        torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda"),
        topk=topk,
        num_expert_group=1,
        topk_group=1,
        renormalize=True,
    )

    # Check expert sets match
    for m in range(num_tokens):
        fused_expert_set = set(fused_ids[m].tolist())
        ref_expert_set = set(ref_ids[m].tolist())
        assert fused_expert_set == ref_expert_set, (
            f"Token {m}: expert set mismatch (no bias). "
            f"Fused: {sorted(fused_expert_set)}, Ref: {sorted(ref_expert_set)}"
        )


@pytest.mark.parametrize("num_tokens", [1, 8, 16])
def test_kimi_linear_fused_gate_output_deterministic(num_tokens):
    """Test that repeated calls give the same result."""
    torch.manual_seed(123)
    topk = 8

    hidden_states = torch.randn(
        num_tokens, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    router_weights = torch.randn(
        NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    correction_bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda") * 0.1

    w1, ids1 = kimi_linear_fused_gate(
        hidden_states, router_weights, correction_bias,
        topk=topk, num_expert_group=1, topk_group=1, renormalize=True,
    )
    w2, ids2 = kimi_linear_fused_gate(
        hidden_states, router_weights, correction_bias,
        topk=topk, num_expert_group=1, topk_group=1, renormalize=True,
    )

    assert torch.equal(ids1, ids2), "Non-deterministic expert IDs"
    assert torch.equal(w1, w2), "Non-deterministic weights"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
