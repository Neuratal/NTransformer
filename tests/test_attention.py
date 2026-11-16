"""GQA attention: causality, head mapping, SDPA-vs-reference equivalence, cache regimes."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from neuratal.config import NeuratalConfig
from neuratal.model.attention import GQAAttention, build_lower_right_mask
from neuratal.model.cache import KVCache


def make_cache() -> KVCache:
    return KVCache(NeuratalConfig())

D = 256


def make_attention(**overrides) -> GQAAttention:
    torch.manual_seed(0)
    cfg = NeuratalConfig(**overrides)
    return GQAAttention(cfg)


def sdpa_reference(q, k, v, groups, scaling):
    """Manual GQA + causal reference in fp32."""
    kk = k.repeat_interleave(groups, dim=1)
    vv = v.repeat_interleave(groups, dim=1)
    scores = torch.matmul(q.float(), kk.float().transpose(-2, -1)) * scaling
    q_len, kv_len = q.size(2), k.size(2)
    offset = kv_len - q_len
    i = torch.arange(q_len, device=q.device).unsqueeze(1)
    j = torch.arange(kv_len, device=q.device).unsqueeze(0)
    scores = scores.masked_fill(j > i + offset, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), vv)


class TestLowerRightMask:
    def test_mask_alignment(self):
        mask = build_lower_right_mask(2, 5, "cpu")
        expected = torch.tensor([
            [True, True, True, True, False],
            [True, True, True, True, True],
        ])
        assert torch.equal(mask, expected)

    def test_full_length_is_causal(self):
        mask = build_lower_right_mask(4, 4, "cpu")
        assert torch.equal(mask, torch.tril(torch.ones(4, 4, dtype=torch.bool)))

    def test_single_query_sees_all(self):
        mask = build_lower_right_mask(1, 7, "cpu")
        assert mask.all()

    def test_invalid_offset(self):
        with pytest.raises(ValueError):
            build_lower_right_mask(5, 4, "cpu")


class TestForwardShapes:
    def test_output_shape(self, device):
        attn = make_attention().to(device)
        hidden = torch.randn(2, 7, D, device=device)
        positions = torch.arange(7, device=device)
        out = attn(hidden, positions)
        assert out.shape == (2, 7, D)

    def test_non_3d_input_rejected(self):
        attn = make_attention()
        with pytest.raises(ValueError, match="3-D"):
            attn(torch.randn(4, D), torch.arange(4))

    def test_wrong_hidden_size_rejected(self):
        attn = make_attention()
        with pytest.raises(ValueError, match="hidden size"):
            attn(torch.randn(1, 2, 128), torch.arange(2))


class TestCausality:
    def test_future_tokens_do_not_affect_earlier_positions(self, device):
        attn = make_attention().to(device).eval()
        hidden = torch.randn(1, 10, D, device=device)
        positions = torch.arange(10, device=device)
        out_a = attn(hidden, positions)
        hidden_b = hidden.clone()
        hidden_b[:, 6:] = torch.randn_like(hidden_b[:, 6:])  # perturb the future
        out_b = attn(hidden_b, positions)
        assert torch.allclose(out_a[:, :6], out_b[:, :6], atol=1e-6)

    def test_gqa_mapping_matches_grouped_reference(self, device):
        attn = make_attention().to(device).eval()
        hidden = torch.randn(2, 6, D, device=device)
        positions = torch.arange(6, device=device)
        # manual path: repeat_interleave + explicit mask
        q = attn.q_proj(hidden).view(2, 6, 4, 64).transpose(1, 2)
        k = attn.k_proj(hidden).view(2, 6, 2, 64).transpose(1, 2)
        v = attn.v_proj(hidden).view(2, 6, 2, 64).transpose(1, 2)
        q, k = attn.rope(q, k, positions)
        ref = sdpa_reference(q, k, v, groups=2, scaling=attn.scaling)
        ref = attn.o_proj(ref.transpose(1, 2).reshape(2, 6, D))
        out = attn(hidden, positions)
        assert torch.allclose(out, ref, atol=1e-5)


class TestCachedDecoding:
    def test_single_token_decode_matches_full_forward(self, device):
        attn = make_attention().to(device).eval()
        hidden = torch.randn(1, 12, D, device=device)
        positions = torch.arange(12, device=device)

        full_out = attn(hidden, positions)

        cache = make_cache()
        outs = []
        for t in range(12):
            step = hidden[:, t:t + 1]
            out_t = attn(step, positions[t:t + 1], past_key_values=cache, layer_idx=0)
            cache.commit(1)
            outs.append(out_t)
        incremental = torch.cat(outs, dim=1)
        assert torch.allclose(full_out, incremental, atol=1e-5)

    def test_multi_token_chunk_matches_single_token_steps(self, device):
        attn = make_attention().to(device).eval()
        hidden = torch.randn(1, 9, D, device=device)
        positions = torch.arange(9, device=device)

        cache_a = make_cache()
        prefill_a = attn(hidden[:, :5], positions[:5], past_key_values=cache_a, layer_idx=0)
        cache_a.commit(5)
        step_a = attn(hidden[:, 5:], positions[5:], past_key_values=cache_a, layer_idx=0)
        cache_a.commit(4)
        out_a = torch.cat([prefill_a, step_a], dim=1)

        cache_b = make_cache()
        prefill_b = attn(hidden[:, :5], positions[:5], past_key_values=cache_b, layer_idx=0)
        cache_b.commit(5)
        chunks = [prefill_b]
        for t in range(5, 9):
            chunks.append(attn(hidden[:, t:t + 1], positions[t:t + 1],
                               past_key_values=cache_b, layer_idx=0))
            cache_b.commit(1)
        out_b = torch.cat(chunks, dim=1)
        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_chunked_prefill_uses_lower_right_mask(self, device):
        # 1 < q_len < kv_len: must not silently attend with an upper-left mask
        attn = make_attention().to(device).eval()
        hidden = torch.randn(1, 6, D, device=device)
        positions = torch.arange(6, device=device)

        cache = make_cache()
        _ = attn(hidden[:, :4], positions[:4], past_key_values=cache, layer_idx=0)
        cache.commit(4)
        chunk_out = attn(hidden[:, 4:], positions[4:], past_key_values=cache, layer_idx=0)

        full_out = attn(hidden, positions)
        assert torch.allclose(full_out[:, 4:], chunk_out, atol=1e-5)

    def test_cache_length_advances(self, device):
        attn = make_attention().to(device).eval()
        hidden = torch.randn(1, 3, D, device=device)
        cache = make_cache()
        attn(hidden, torch.arange(3, device=device), past_key_values=cache, layer_idx=0)
        cache.commit(3)
        assert cache.get_seq_length() == 3
        attn(hidden[:, :1], torch.arange(3, 4, device=device), past_key_values=cache, layer_idx=0)
        cache.commit(1)
        assert cache.get_seq_length() == 4

    def test_cache_requires_layer_idx(self):
        attn = make_attention()
        hidden = torch.randn(1, 2, D)
        cache = make_cache()
        with pytest.raises(ValueError, match="layer_idx"):
            attn(hidden, torch.arange(2), past_key_values=cache)


class TestReferenceAndFallback:
    def test_reference_matches_sdpa(self, device):
        attn = make_attention().to(device).eval()
        attn_ref = make_attention(attention_implementation="reference").to(device).eval()
        attn_ref.load_state_dict(attn.state_dict())
        hidden = torch.randn(2, 8, D, device=device)
        positions = torch.arange(8, device=device)
        out_sdpa = attn(hidden, positions)
        out_ref = attn_ref(hidden, positions)
        assert torch.allclose(out_sdpa, out_ref, atol=1e-5)

    def test_gqa_fallback_is_explicit_and_warns(self, device):
        attn = make_attention().to(device).eval()
        hidden = torch.randn(1, 5, D, device=device)
        positions = torch.arange(5, device=device)
        baseline = attn(hidden, positions)

        call_count = {"n": 0}
        real_sdpa = F.scaled_dot_product_attention

        def broken_sdpa(*args, **kwargs):
            if kwargs.get("enable_gqa"):
                call_count["n"] += 1
                raise RuntimeError("enable_gqa unsupported (test double)")
            return real_sdpa(*args, **kwargs)

        with pytest.warns(UserWarning, match="enable_gqa"):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(F, "scaled_dot_product_attention", broken_sdpa)
                out = attn(hidden, positions)
        assert call_count["n"] == 1
        assert torch.allclose(baseline, out, atol=1e-5)
        # fallback warning only fires once
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(F, "scaled_dot_product_attention", broken_sdpa)
            out2 = attn(hidden, positions)
        assert torch.allclose(baseline, out2, atol=1e-5)


class TestAutograd:
    def test_gradients_flow_to_all_projections(self):
        attn = make_attention()
        hidden = torch.randn(2, 6, D)
        out = attn(hidden, torch.arange(6))
        out.sum().backward()
        for name, param in attn.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name

    def test_attention_dropout_only_in_training(self):
        attn = make_attention(attention_dropout=0.5)
        hidden = torch.randn(1, 8, D)
        attn.eval()
        out_a = attn(hidden, torch.arange(8))
        out_b = attn(hidden, torch.arange(8))
        assert torch.equal(out_a, out_b)
        attn.train()
        outs = {tuple(attn(hidden, torch.arange(8))[0].flatten().tolist()[:8]) for _ in range(8)}
        assert len(outs) > 1
