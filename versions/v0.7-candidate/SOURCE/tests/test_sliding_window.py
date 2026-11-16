"""Sliding-window attention: windowing semantics, causality, cache parity."""

from __future__ import annotations

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel, load_pretrained, save_pretrained
from neuratal.model.cache import KVCache
from neuratal.model.sliding_window import build_sliding_window_mask

V = 262_144


def make_model(**overrides) -> NeuratalModel:
    torch.manual_seed(0)
    kw = dict(NeuratalConfig().to_dict())
    kw.update(overrides)
    kw["attention_type"] = "sliding_window"
    kw.setdefault("swa_window", 64)
    return NeuratalModel(NeuratalConfig.from_dict(kw)).eval()


class TestWindowMask:
    def test_full_length_window_is_causal(self):
        mask = build_sliding_window_mask(8, 8, 64, "cpu")
        assert torch.equal(mask, torch.tril(torch.ones(8, 8, dtype=torch.bool)))

    def test_window_limits_attendance(self):
        mask = build_sliding_window_mask(6, 6, 3, "cpu")
        # token i attends keys j in [i-2, i]
        expected = torch.tensor([
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ], dtype=torch.bool)
        assert torch.equal(mask, expected)

    def test_offset_alignment_in_cache(self):
        mask = build_sliding_window_mask(2, 5, 3, "cpu")  # cached decoding: offset=3
        # query abs positions 3,4 attend j in [1,3] and [2,4]
        expected = torch.tensor([
            [False, True, True, True, False],
            [False, False, True, True, True],
        ])
        assert torch.equal(mask, expected)

    def test_window_one_attends_self_only(self):
        mask = build_sliding_window_mask(4, 4, 1, "cpu")
        assert torch.equal(mask, torch.eye(4, dtype=torch.bool))


class TestForward:
    def test_logits_shape_full_vocab(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (2, 12), device=device)
        out = model(ids)
        assert out.logits.shape == (2, 12, V)
        assert torch.isfinite(out.logits).all()

    def test_early_positions_unaffected_by_late_tokens_within_window(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 16), device=device)
        with torch.no_grad():
            ref = model(ids, use_cache=False).logits
            ids2 = ids.clone()
            ids2[:, 10:] = torch.randint(0, V, (1, 6), device=device)
            out = model(ids2, use_cache=False).logits
        # positions 0..9 cannot attend to 10.. anyway (window), but check causality:
        assert torch.allclose(ref[:, :10], out[:, :10], atol=1e-6)


class TestCachedEquivalence:
    def test_token_by_token_matches_full_forward(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 12), device=device)
        with torch.no_grad():
            ref = model(ids, use_cache=False).logits
            cache = KVCache(model.config)
            outs = []
            for t in range(12):
                outs.append(model(ids[:, t:t + 1], past_key_values=cache).logits)
            stacked = torch.cat(outs, dim=1)
        max_abs = (ref - stacked).abs().max().item()
        assert max_abs < 1e-4, f"cached vs uncached diverged by {max_abs}"


class TestSerializationAndValidation:
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

    def test_swa_window_bounds(self):
        with pytest.raises(ValueError, match="swa_window"):
            NeuratalConfig(swa_window=0)
        with pytest.raises(ValueError, match="swa_window"):
            NeuratalConfig(swa_window=999_999)

    def test_default_stays_gqa(self):
        assert NeuratalConfig().attention_type == "gqa"
        assert NeuratalConfig().swa_window == 512
