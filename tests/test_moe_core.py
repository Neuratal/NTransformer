# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""MoE core: config, router math, dispatch, capacity, padding, dtype, parameters."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuratal import NeuratalConfig, NeuratalModel
from neuratal.model.moe import MoEFFN


def tiny_config(**overrides) -> NeuratalConfig:
    base = dict(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        embedding_size=8,
        intermediate_size=64,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_token=2,
    )
    base.update(overrides)
    return NeuratalConfig(**base)


class TestMoEConfig:
    def test_default_is_dense_and_has_no_moe_state(self):
        cfg = NeuratalConfig()
        assert cfg.is_moe is False
        assert cfg.num_moe_layers == 0
        assert cfg.moe_layer_indices == []
        assert cfg.num_experts == 0

    def test_all_layers_default_when_enabled(self):
        cfg = tiny_config()
        assert cfg.is_moe
        assert cfg.moe_layer_indices == [0, 1]
        assert cfg.expert_hidden_dim == cfg.intermediate_size

    def test_frequency_and_explicit_indices(self):
        cfg = tiny_config(num_hidden_layers=6, moe_layer_frequency=2)
        assert cfg.moe_layer_indices == [0, 2, 4]
        cfg = tiny_config(num_hidden_layers=6, moe_layer_indices=[3, 1, 3])
        assert cfg.moe_layer_indices == [1, 3]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"num_experts": 1}, "num_experts"),
            ({"num_experts_per_token": 0}, "num_experts_per_token"),
            ({"num_experts_per_token": 5}, "num_experts_per_token"),
            ({"router_type": "sigmoid"}, "router_type"),
            ({"capacity_factor": 0.0}, "capacity_factor"),
            ({"capacity_factor": float("inf")}, "capacity_factor"),
            ({"min_capacity": -1}, "min_capacity"),
            ({"router_aux_loss_coef": -1}, "router_aux_loss_coef"),
            ({"router_z_loss_coef": float("nan")}, "router_z_loss_coef"),
            ({"moe_layer_frequency": 0, "num_hidden_layers": 2}, "moe_layer_frequency"),
            ({"moe_layer_indices": [9], "num_hidden_layers": 2}, "out of range"),
            ({"moe_layer_indices": [0], "moe_layer_frequency": 2}, "not both"),
            ({"shared_expert": "yes"}, "shared_expert"),
        ],
    )
    def test_invalid_moe_config_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            tiny_config(**kwargs)

    def test_moe_placement_rejected_on_dense(self):
        with pytest.raises(ValueError, match="num_experts"):
            NeuratalConfig(moe_layer_indices=[0])
        with pytest.raises(ValueError, match="num_experts"):
            NeuratalConfig(moe_layer_frequency=2)
        with pytest.raises(ValueError, match="shared_expert"):
            NeuratalConfig(shared_expert=True)

    def test_roundtrip_preserves_moe_fields(self):
        cfg = tiny_config(moe_layer_indices=[0], shared_expert=True, capacity_factor=1.5)
        rebuilt = NeuratalConfig.from_dict(cfg.to_dict())
        assert rebuilt.to_dict() == cfg.to_dict()
        assert rebuilt.is_moe and rebuilt.moe_layer_indices == [0]


class TestRouterMath:
    def test_top1_selects_expected_expert(self):
        cfg = tiny_config(num_experts_per_token=1)
        moe = MoEFFN(cfg).eval()
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0].fill_(100.0)
            moe.router.weight[1:].fill_(-100.0)
        x = torch.rand(2, 5, cfg.hidden_size) + 1.0
        out, stats = moe(x)
        expected = moe.experts[0](x.reshape(-1, cfg.hidden_size)).reshape_as(x)
        assert torch.allclose(out, expected, atol=1e-5)
        assert int(stats.counts[0]) == 10
        assert int(stats.counts.sum()) == 10

    def test_top2_combines_experts_with_normalized_weights(self):
        cfg = tiny_config()
        moe = MoEFFN(cfg).eval()
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0].fill_(100.0)
            moe.router.weight[1].fill_(100.0)
            moe.router.weight[2:].fill_(-100.0)
        x = torch.rand(1, 4, cfg.hidden_size) + 1.0
        out, _ = moe(x)
        flat = x.reshape(-1, cfg.hidden_size)
        e0 = moe.experts[0](flat)
        e1 = moe.experts[1](flat)
        expected = (0.5 * e0 + 0.5 * e1).reshape_as(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_router_aux_and_z_loss_match_formula(self):
        cfg = tiny_config(router_aux_loss_coef=0.03, router_z_loss_coef=0.007)
        moe = MoEFFN(cfg)
        x = torch.randn(2, 6, cfg.hidden_size)
        _, stats = moe(x, compute_aux=True)
        logits = F.linear(x.reshape(-1, cfg.hidden_size).float(), moe.router.weight.float())
        probs = torch.softmax(logits, dim=-1)
        _, idx = torch.topk(probs, moe.top_k, dim=-1)
        counts = torch.bincount(idx.reshape(-1), minlength=moe.num_experts).float()
        f = counts / counts.sum()
        p = probs.mean(dim=0)
        expected_aux = cfg.router_aux_loss_coef * moe.num_experts * (f * p).sum()
        expected_z = cfg.router_z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()
        assert torch.allclose(stats.aux_loss, expected_aux, atol=1e-6)
        assert torch.allclose(stats.z_loss, expected_z, atol=1e-6)

    def test_aux_loss_builds_router_gradient_only(self):
        moe = MoEFFN(tiny_config())
        x = torch.randn(2, 4, moe.hidden_size)
        _, stats = moe(x, compute_aux=True)
        stats.aux_loss.backward()
        assert moe.router.weight.grad is not None
        assert moe.router.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in moe.experts.parameters())

    def test_main_loss_reaches_router_and_every_expert(self):
        cfg = tiny_config()
        moe = MoEFFN(cfg)
        # Construct a routing pattern that makes every expert top-1 for some token.
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[: cfg.num_experts, : cfg.num_experts] = (
                torch.eye(cfg.num_experts) * 10.0
            )
        x = torch.zeros(8, 1, cfg.hidden_size)
        for i in range(8):
            x[i, 0, i % 4] = 1.0
        out, stats = moe(x)
        out.sum().backward()
        assert moe.router.weight.grad is not None
        assert moe.router.weight.grad.abs().sum() > 0
        assert int((stats.counts > 0).sum()) == cfg.num_experts
        for i, expert in enumerate(moe.experts):
            assert expert.gate_up_proj.weight.grad is not None, i
            assert expert.gate_up_proj.weight.grad.abs().sum() > 0, i

    def test_inference_does_not_build_aux_graph(self):
        moe = MoEFFN(tiny_config())
        x = torch.randn(2, 3, moe.hidden_size)
        _, stats = moe(x, compute_aux=False)
        assert stats.aux_loss is None and stats.z_loss is None
        assert stats.counts.numel() == moe.num_experts


class TestDispatch:
    def test_capacity_drops_and_marks(self):
        cfg = tiny_config(capacity_factor=0.1, min_capacity=0)
        moe = MoEFFN(cfg)
        x = torch.randn(16, 4, cfg.hidden_size)
        out, stats = moe(x)
        assert torch.isfinite(out).all()
        assert stats.dropped > 0
        assert stats.num_assignments == 16 * 4 * moe.top_k

    def test_padding_mask_excludes_and_zeroes_positions(self):
        cfg = tiny_config()
        moe = MoEFFN(cfg)
        x = torch.randn(2, 5, cfg.hidden_size)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        out, stats = moe(x, token_mask=mask)
        assert stats.num_tokens == 5
        assert int(stats.counts.sum()) == 5 * moe.top_k
        assert out[0, 3:].abs().sum() == 0
        assert out[1, 2:].abs().sum() == 0

    def test_variable_batch_and_sequence_lengths(self):
        moe = MoEFFN(tiny_config()).eval()
        for batch, seq in ((1, 1), (3, 7), (2, 13)):
            out, stats = moe(torch.randn(batch, seq, moe.hidden_size))
            assert out.shape == (batch, seq, moe.hidden_size)
            assert int(stats.counts.sum()) == batch * seq * moe.top_k

    def test_bf16_forward_finite_and_dtypes(self):
        cfg = tiny_config()
        moe = MoEFFN(cfg).to(torch.bfloat16)
        x = torch.randn(2, 5, cfg.hidden_size, dtype=torch.bfloat16)
        out, stats = moe(x)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()
        assert torch.isfinite(stats.aux_loss.float())
        assert torch.isfinite(stats.z_loss.float())

    def test_shared_expert_is_always_active(self):
        cfg = tiny_config(shared_expert=True)
        moe = MoEFFN(cfg)
        assert moe.shared_expert is not None
        x = torch.randn(2, 3, cfg.hidden_size)
        out, _ = moe(x)
        assert torch.isfinite(out).all()


class TestModelIntegration:
    def test_dense_model_has_zero_moe_parameters(self):
        model = NeuratalModel(NeuratalConfig(vocab_size=256, hidden_size=32, num_hidden_layers=2,
                                            num_attention_heads=4, num_key_value_heads=2,
                                            embedding_size=8, intermediate_size=64))
        report = model.parameter_report()
        assert report["moe_router"] == report["moe_experts"] == 0
        assert all(not getattr(layer, "is_moe", False) for layer in model.layers)

    def test_all_moe_and_mixed_placement(self):
        all_moe = NeuratalModel(tiny_config())
        assert all(layer.is_moe for layer in all_moe.layers)
        mixed = NeuratalModel(tiny_config(num_hidden_layers=4, moe_layer_indices=[1, 3]))
        assert [layer.is_moe for layer in mixed.layers] == [False, True, False, True]
        assert sum(1 for n, _ in mixed.named_parameters() if ".mlp.router." in n) == 2

    def test_total_and_active_parameter_accounting(self):
        cfg = tiny_config()
        model = NeuratalModel(cfg)
        report = model.parameter_report()
        assert sum(v for k, v in report.items() if k != "total") == report["total"] == model.num_parameters()
        assert report["moe_experts"] == cfg.num_hidden_layers * cfg.num_experts * 3 * cfg.hidden_size * cfg.expert_hidden_dim
        assert report["moe_router"] == cfg.num_hidden_layers * cfg.hidden_size * cfg.num_experts
        active = model.active_parameters_per_token()
        expert_total = report["moe_experts"]
        expected = model.num_parameters() - expert_total + (
            cfg.num_hidden_layers * cfg.num_experts_per_token * (expert_total // (cfg.num_hidden_layers * cfg.num_experts))
        )
        assert active == expected
        assert active < model.num_parameters()

    def test_moe_forward_backward_and_aux_through_model(self):
        model = NeuratalModel(tiny_config())
        ids = torch.randint(0, model.config.vocab_size, (2, 7))
        hidden, stats = model.forward_hidden(ids, return_aux=True)
        assert hidden.shape == (2, 7, model.config.hidden_size)
        assert stats is not None and stats.aux_loss is not None
        loss = hidden.square().mean() + stats.aux_loss + stats.z_loss
        loss.backward()
        assert model.layers[0].mlp.router.weight.grad is not None
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    def test_model_hidden_only_forward_matches_forward_hidden(self):
        model = NeuratalModel(tiny_config()).eval()
        ids = torch.randint(0, model.config.vocab_size, (1, 5))
        with torch.no_grad():
            a = model.forward_hidden(ids)
            b = model(ids, output_hidden_only=True)
        assert torch.allclose(a, b, atol=0, rtol=0)
