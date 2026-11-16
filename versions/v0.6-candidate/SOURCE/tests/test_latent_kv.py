"""Latent-KV (MLA-style) attention: correctness, cache parity, serialization."""

from __future__ import annotations

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel, load_pretrained, save_pretrained
from neuratal.model.latent_kv import LatentKVCache

V = 262_144


def make_model(**overrides) -> NeuratalModel:
    torch.manual_seed(0)
    kw = dict(NeuratalConfig().to_dict())
    kw.update(overrides)
    kw["attention_type"] = "mla"
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

    def test_bf16_finite(self, device):
        model = make_model(dtype="bfloat16").to(device)
        ids = torch.randint(0, V, (1, 4), device=device)
        with torch.no_grad():
            out = model(ids)
        assert torch.isfinite(out.logits.float()).all()


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
            cache = LatentKVCache(model.config, model.config.mla_latent_dim)
            outs = []
            for t in range(10):
                outs.append(model(ids[:, t:t + 1], past_key_values=cache).logits)
            stacked = torch.cat(outs, dim=1)
        max_abs = (ref - stacked).abs().max().item()
        assert max_abs < 1e-4, f"cached vs uncached diverged by {max_abs}"

    def test_chunked_prefill_matches_single_token(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 9), device=device)
        with torch.no_grad():
            cache_a = LatentKVCache(model.config, model.config.mla_latent_dim)
            prefill = model(ids[:, :5], past_key_values=cache_a).logits
            step = model(ids[:, 5:], past_key_values=cache_a).logits
            out_a = torch.cat([prefill, step], dim=1)

            cache_b = LatentKVCache(model.config, model.config.mla_latent_dim)
            outs = []
            for t in range(9):
                outs.append(model(ids[:, t:t + 1], past_key_values=cache_b).logits)
            out_b = torch.cat(outs, dim=1)
        assert torch.allclose(out_a, out_b, atol=1e-5)


class TestSerializationAndParams:
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

    def test_mla_has_fewer_kv_params_than_gqa(self):
        mla = NeuratalModel(NeuratalConfig(attention_type="mla"))
        dense = NeuratalModel(NeuratalConfig())
        kv_mla = sum(p.numel() for n, p in mla.named_parameters() if "attn" in n and "o_proj" not in n and "q_proj" not in n)
        kv_dense = sum(p.numel() for n, p in dense.named_parameters() if "attn" in n and "o_proj" not in n)
        # dense qkv is fused; compare approximate kv-side projections
        assert kv_mla < kv_dense

    def test_cache_bytes_per_token(self, device):
        cache = LatentKVCache(NeuratalConfig(), 64)
        cache.initialize(1, device, torch.float32)
        total = sum(t.numel() * t.element_size() for t in cache._buffers)
        bytes_per_token = total / 2 / cache.capacity  # layers counted in buffers; per layer per token
        assert total == 2 * 1 * cache.capacity * 64 * 4  # 2 layers x 1 x cap x latent x fp32


class TestConfigValidation:
    def test_mla_latent_dim_bounds(self):
        with pytest.raises(ValueError, match="mla_latent_dim"):
            NeuratalConfig(attention_type="mla", mla_latent_dim=0)
        with pytest.raises(ValueError, match="mla_latent_dim"):
            NeuratalConfig(attention_type="mla", mla_latent_dim=999)

    def test_default_stays_gqa(self):
        assert NeuratalConfig().attention_type == "gqa"

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="attention_type"):
            NeuratalConfig(attention_type="nsa")
