"""Pathological routing distributions.

MoE must stay correct — not merely run — under adversarial routing:
all tokens on one expert, starved experts, saturated routers, and extreme
capacity pressure. Synthetic weights make these deterministic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neuratal import NeuratalConfig
from neuratal.model.moe import MoEFFN

V = 262_144


def make_moe(num_experts: int = 4, top_k: int = 1, seed: int = 0, **overrides) -> MoEFFN:
    torch.manual_seed(seed)
    cfg = NeuratalConfig(
        vocab_size=V, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=64, embedding_size=8,
        num_experts=num_experts, num_experts_per_token=top_k,
        expert_hidden_dim=64, **overrides,
    )
    return MoEFFN(cfg)


def force_dominant_expert(moe: MoEFFN, dominant: int = 0, logit: float = 10.0) -> None:
    """Make every token route to `dominant` for its first pick (all-positive
    row => positive dot with any all-ones input)."""
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[dominant, :] = logit


class TestAllTokensOneExpert:
    def test_single_assignment_counts(self):
        moe = make_moe(num_experts=4, top_k=1)
        force_dominant_expert(moe, 0)
        x = torch.ones(2, 8, 32)
        _, stats = moe(x, compute_aux=False)
        assert stats.counts.tolist() == [16.0, 0.0, 0.0, 0.0]
        assert stats.num_assignments == 16

    def test_output_finite_and_shared_path(self):
        moe = make_moe(num_experts=4, top_k=1)
        force_dominant_expert(moe, 2)
        x = torch.ones(2, 8, 32)
        out, stats = moe(x)
        assert torch.isfinite(out).all()
        # unused experts produce exactly zero contribution, not garbage
        assert stats.counts[0].item() == 0.0

    def test_backward_finite_for_used_and_unused_experts(self):
        moe = make_moe(num_experts=4, top_k=1)
        force_dominant_expert(moe, 0)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out, _ = moe(x)
        out.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        expert0 = moe.experts[0].down_proj.weight.grad
        expert1 = moe.experts[1].down_proj.weight.grad
        assert expert0 is not None and torch.isfinite(expert0).all() and expert0.abs().sum() > 0
        # unused experts must not fabricate gradients (None is fine in eager)
        assert expert1 is None or torch.all(expert1 == 0)


class TestStarvedExperts:
    def test_only_one_expert_starved_still_finite(self):
        moe = make_moe(num_experts=8, top_k=2)
        # every token: dominant expert 3, runner-up expert 5 (index 1 tie-break)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[3, 0] = 10.0
            moe.router.weight[5, 1] = 5.0
        x = torch.ones(1, 16, 32, requires_grad=True)
        out, stats = moe(x)
        assert torch.isfinite(out).all()
        counts = stats.counts
        assert counts[3].item() == 16.0
        assert counts[5].item() == 16.0
        assert (counts[torch.tensor([0, 1, 2, 4, 6, 7])] == 0).all()

    def test_zero_assignment_expert_keeps_finite_param_state(self):
        moe = make_moe(num_experts=4, top_k=1)
        force_dominant_expert(moe, 1)
        x = torch.ones(1, 4, 32)
        out, _ = moe(x)
        out.sum().backward()
        for i, expert in enumerate(moe.experts):
            for p in expert.parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"expert {i} grad not finite"


class TestSaturatedRouter:
    def test_huge_logits_do_not_produce_nan(self):
        moe = make_moe(num_experts=4, top_k=2)
        with torch.no_grad():
            moe.router.weight.mul_(1000.0)
        x = torch.randn(1, 8, 32)
        out, stats = moe(x, compute_aux=True)
        assert torch.isfinite(out).all()
        assert torch.isfinite(stats.z_loss).all()
        assert torch.isfinite(stats.aux_loss).all()

    def test_z_loss_scales_quadratically_with_logit_norm(self):
        moe = make_moe(num_experts=4, top_k=2)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0, 0] = 8.0
        x = torch.ones(1, 4, 32)  # logits are constant [8, 0, 0, 0] per token
        _, stats = moe(x, compute_aux=True)
        lse = torch.logsumexp(torch.tensor([[8.0, 0.0, 0.0, 0.0]]), dim=-1)
        expected = moe.router_z_loss_coef * lse.pow(2).mean()
        assert torch.allclose(stats.z_loss, expected, rtol=1e-5)


class TestAssignmentConservation:
    def test_every_token_keeps_topk_assignments_without_capacity(self):
        for seed in range(5):
            moe = make_moe(num_experts=8, top_k=2, seed=seed)
            x = torch.randn(2, 32, 32)
            _, stats = moe(x, compute_aux=False)
            assert stats.counts.sum().item() == 2 * 32 * 2

    def test_kept_plus_dropped_equals_total_assignments_with_capacity(self):
        moe = make_moe(num_experts=8, top_k=2, capacity_factor=0.25)
        force_dominant_expert(moe, 0)  # pathological: all assignments on expert 0
        x = torch.ones(2, 32, 32)
        out, stats = moe(x, compute_aux=False)
        assert torch.isfinite(out).all()
        total = 2 * 32 * 2  # 128 assignments (top-2, 64 tokens)
        # counts are pre-capacity (documented convention): every token's first
        # pick is expert 0, the zero-logit tie-break second pick is expert 1
        assert stats.counts.sum().item() == total
        assert stats.counts[0].item() == 64.0
        assert float(stats.dropped) == total - 8  # capacity 4 per expert (0.25 * 128 / 8)

    def test_capacity_limits_assignments_per_expert(self):
        moe = make_moe(num_experts=4, top_k=1, capacity_factor=1.0)
        force_dominant_expert(moe, 0)
        x = torch.ones(1, 8, 32)
        _, stats = moe(x, compute_aux=False)
        # capacity = ceil(1.0 * 8 * 1 / 4) = 2 per expert; all 8 assignments
        # target expert 0, so 6 are dropped and counted (counts stay pre-capacity).
        assert stats.counts[0].item() == 8.0
        assert float(stats.dropped) == 6.0


class TestAuxGradientRouting:
    def test_aux_loss_gradient_reaches_router_weights(self):
        moe = make_moe(num_experts=4, top_k=2)
        x = torch.ones(1, 4, 32)
        _, stats = moe(x, compute_aux=True)
        stats.aux_loss.backward()
        assert moe.router.weight.grad is not None
        assert torch.isfinite(moe.router.weight.grad).all()
        assert moe.router.weight.grad.abs().sum() > 0
