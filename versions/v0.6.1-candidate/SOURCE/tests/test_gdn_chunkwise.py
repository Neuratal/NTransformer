"""WY chunkwise GDN: correctness vs the recurrent oracle, causality, parity, roundtrip."""

from __future__ import annotations

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel, load_pretrained, save_pretrained
from neuratal.model.gated_delta_net import GatedDeltaNetAttention

V = 262_144


def make_gdn(chunk_size: int, **overrides) -> NeuratalModel:
    torch.manual_seed(0)
    kw = dict(NeuratalConfig().to_dict())
    kw.update(overrides)
    kw["attention_type"] = "gated_delta"
    kw["gdn_chunk_size"] = chunk_size
    return NeuratalModel(NeuratalConfig.from_dict(kw)).eval()


def load_into(target: NeuratalModel, source: NeuratalModel) -> NeuratalModel:
    target.load_state_dict(source.state_dict())
    return target


class TestChunkwiseParity:
    @pytest.mark.parametrize("chunk", [2, 16, 64])
    @pytest.mark.parametrize("T", [2, 7, 16, 37, 64, 100, 128, 200, 512])
    def test_chunkwise_matches_sequential_fp32(self, chunk, T):
        torch.manual_seed(0)
        seq = NeuratalModel(NeuratalConfig(attention_type="gated_delta", gdn_chunk_size=0))
        chk = load_into(
            NeuratalModel(NeuratalConfig(attention_type="gated_delta", gdn_chunk_size=chunk)),
            seq,
        )
        ids = torch.randint(0, V, (1, T))
        with torch.no_grad():
            a = seq(ids).logits
            b = chk(ids).logits
        max_abs = (a - b).abs().max().item()
        assert max_abs < 5e-3, f"chunked diverged by {max_abs} at C={chunk} T={T}"

    def test_chunkwise_t_2048_finite(self):
        torch.manual_seed(0)
        seq = NeuratalModel(NeuratalConfig(attention_type="gated_delta", gdn_chunk_size=0)).eval()
        chk = load_into(
            NeuratalModel(NeuratalConfig(attention_type="gated_delta", gdn_chunk_size=64)).eval(),
            seq,
        )
        ids = torch.randint(0, V, (1, 2048))
        with torch.no_grad():
            out = chk(ids)
        assert torch.isfinite(out.logits).all()

    def test_chunkwise_bf16_finite(self):
        chk = make_gdn(64, dtype="bfloat16")
        ids = torch.randint(0, V, (1, 100))
        with torch.no_grad():
            out = chk(ids)
        assert torch.isfinite(out.logits.float()).all()


class TestCausalityAndCache:
    @pytest.mark.parametrize("chunk", [16, 64])
    def test_chunkwise_future_tokens_do_not_affect_past(self, chunk, device):
        model = make_gdn(chunk).to(device)
        ids = torch.randint(0, V, (1, 16), device=device)
        with torch.no_grad():
            ref = model(ids, use_cache=False).logits
            ids2 = ids.clone()
            ids2[:, 8:] = torch.randint(0, V, (1, 8), device=device)
            out = model(ids2, use_cache=False).logits
        assert torch.allclose(ref[:, :8], out[:, :8], atol=1e-5)


class TestSerialization:
    def test_chunkwise_roundtrip_bitwise(self, tmp_path):
        model = make_gdn(64)
        model.eval()
        ids = torch.randint(0, V, (1, 8))
        with torch.no_grad():
            ref = model(ids).logits
            save_pretrained(model, tmp_path)
            loaded = load_pretrained(tmp_path).eval()
            out = loaded(ids).logits
        assert torch.equal(ref, out)

    def test_chunk_size_in_config(self):
        cfg = NeuratalConfig(attention_type="gated_delta", gdn_chunk_size=64)
        assert cfg.gdn_chunk_size == 64
        with pytest.raises(ValueError, match="gdn_chunk_size"):
            NeuratalConfig(attention_type="gated_delta", gdn_chunk_size=-1)

    def test_default_stays_gqa(self):
        assert NeuratalConfig().gdn_chunk_size == 0
        assert NeuratalConfig().attention_type == "gqa"
