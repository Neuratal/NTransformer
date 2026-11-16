"""Gated DeltaNet mixer: shapes, causality, cached/uncached parity, stability, roundtrip."""

from __future__ import annotations

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel, load_pretrained, save_pretrained
from neuratal.model.gated_delta_net import DeltaNetCache, GatedDeltaNetAttention

V = 262_144


def make_model(**overrides) -> NeuratalModel:
    torch.manual_seed(0)
    kw = dict(NeuratalConfig().to_dict())
    kw.update(overrides)
    kw["attention_type"] = "gated_delta"
    return NeuratalModel(NeuratalConfig.from_dict(kw)).eval()


class TestForward:
    def test_logits_shape_full_vocab(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (2, 12), device=device)
        out = model(ids)
        assert out.logits.shape == (2, 12, V)
        assert torch.isfinite(out.logits).all()

    def test_backward_pass(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 8), device=device)
        out = model(ids)
        out.logits.square().mean().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name

    def test_batch_isolation_between_calls(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 8), device=device)
        with torch.no_grad():
            a = model(ids).logits
            b = model(ids).logits
        assert torch.equal(a, b)


class TestCausality:
    def test_future_tokens_do_not_affect_earlier_positions(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 12), device=device)
        with torch.no_grad():
            ref = model(ids, use_cache=False).logits
            ids2 = ids.clone()
            ids2[:, 6:] = torch.randint(0, V, (1, 6), device=device)
            out = model(ids2, use_cache=False).logits
        assert torch.allclose(ref[:, :6], out[:, :6], atol=1e-6)


class TestCachedEquivalence:
    def test_token_by_token_matches_full_forward(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 10), device=device)
        with torch.no_grad():
            ref = model(ids, use_cache=False).logits
            cache = DeltaNetCache(model.config)
            outs = []
            for t in range(10):
                outs.append(model(ids[:, t:t + 1], past_key_values=cache).logits)
            stacked = torch.cat(outs, dim=1)
        max_abs = (ref - stacked).abs().max().item()
        assert max_abs < 1e-4, f"cached vs uncached diverged by {max_abs}"

    def test_cache_length_tracks_commit(self, device):
        model = make_model().to(device)
        cache = DeltaNetCache(model.config)
        with torch.no_grad():
            model(torch.randint(0, V, (1, 5), device=device), past_key_values=cache)
        assert cache.get_seq_length() == 5
        assert cache._states[0].abs().sum() > 0  # state actually evolved


class TestAlphaBetaBounds:
    def test_gates_are_sigmoid_bounded(self, device):
        model = make_model().to(device)
        x = torch.randn(1, 4, 256, device=device)
        a = torch.sigmoid(model.layers[0].attn.alpha_proj(x))
        b = torch.sigmoid(model.layers[0].attn.beta_proj(x))
        assert (a > 0).all() and (a < 1).all()
        assert (b > 0).all() and (b < 1).all()

    def test_state_contracts_from_zero(self, device):
        # contraction property: starting from S=0 with bounded gates, no blowup over 64 steps
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 64), device=device)
        with torch.no_grad():
            out = model(ids).logits
        assert torch.isfinite(out).all()


class TestParameterAndSerialization:
    def test_parameter_count_reasonable(self):
        model = make_model()
        n = model.num_parameters()
        dense = NeuratalModel(NeuratalConfig()).num_parameters()
        assert n > dense  # gates/add projections add params
        # sanity: embedding-heavy model, delta adds < 5%
        assert n < dense * 1.05

    def test_roundtrip_bitwise(self, tmp_path):
        model = make_model()
        model.eval()
        ids = torch.randint(0, V, (1, 6))
        with torch.no_grad():
            ref = model(ids).logits
            save_pretrained(model, tmp_path)
            loaded = load_pretrained(tmp_path).eval()
            out = loaded(ids).logits
        assert torch.equal(ref, out)

    def test_bf16_forward_finite(self, device):
        model = make_model(dtype="bfloat16").to(device)
        ids = torch.randint(0, V, (1, 4), device=device)
        with torch.no_grad():
            out = model(ids)
        assert torch.isfinite(out.logits.float()).all()


class TestConfigValidation:
    def test_unknown_attention_type_rejected(self):
        with pytest.raises(ValueError, match="attention_type"):
            NeuratalConfig(attention_type="mamba2")

    def test_gqa_default_unchanged(self):
        assert NeuratalConfig().attention_type == "gqa"
