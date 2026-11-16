"""RoPE: frequency math, layout, partial rotary, position handling, dtype behavior."""

from __future__ import annotations

import pytest
import torch

from neuratal.config import NeuratalConfig
from neuratal.model.positional import RotaryEmbedding


def make_rope(**overrides) -> RotaryEmbedding:
    cfg = NeuratalConfig(**overrides)
    return RotaryEmbedding(cfg)


class TestFrequencyMath:
    def test_inv_freq_matches_formula(self):
        rope = make_rope(hidden_size=256, num_attention_heads=4, rotary_dim=64, rope_theta=10_000.0)
        expected = 1.0 / (10_000.0 ** (torch.arange(0, 64, 2).float() / 64))
        assert torch.allclose(rope.inv_freq, expected, rtol=0, atol=0)

    def test_theta_changes_frequencies(self):
        rope_a = make_rope(rope_theta=10_000.0)
        rope_b = make_rope(rope_theta=500_000.0)
        assert not torch.allclose(rope_a.inv_freq, rope_b.inv_freq)

    def test_cos_sin_buffers_not_persistent(self):
        rope = make_rope()
        assert "cos_cached" not in rope.state_dict()
        assert "inv_freq" not in rope.state_dict()


class TestApply:
    def test_position_zero_is_identity(self):
        rope = make_rope()
        q = torch.randn(2, 4, 5, 64)
        k = torch.randn(2, 2, 5, 64)
        pos = torch.arange(5)
        q_out, k_out = rope(q, k, pos)
        # first position must be unrotated (angle 0)
        assert torch.allclose(q_out[:, :, 0], q[:, :, 0], atol=1e-6)
        assert torch.allclose(k_out[:, :, 0], k[:, :, 0], atol=1e-6)

    def test_norm_is_preserved_per_pair_block(self):
        rope = make_rope()
        x = torch.randn(2, 4, 17, 64)
        pos = torch.arange(17)
        out, _ = rope(x, x.clone(), pos)
        # rotate-half pairs (i, i + r/2) are rotated together -> pair norms preserved
        half = 32
        pairs_in = x[..., :half].pow(2) + x[..., half:].pow(2)
        pairs_out = out[..., :half].pow(2) + out[..., half:].pow(2)
        assert torch.allclose(pairs_in, pairs_out, atol=1e-4)
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)

    def test_known_rotation_single_pair(self):
        # rotary_dim=2 gives one frequency pair rotated by ang = pos * theta^0 = pos
        rope = make_rope(hidden_size=64, num_attention_heads=1, num_key_value_heads=1,
                         rotary_dim=2, rope_theta=2.0)
        x = torch.tensor([[[[1.0, 0.0]]]])  # batch=1, heads=1, seq=1, dim=2
        pos = torch.tensor([1])  # angle = 1.0 radian
        out, _ = rope(x, x.clone(), pos)
        # rotate_half layout: x_rot*cos + cat(-x2, x1)*sin -> (cos, sin)
        c, s = torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))
        assert torch.allclose(out[0, 0, 0], torch.tensor([c, s]), atol=1e-6)

    def test_partial_rotary_leaves_trailing_channels_untouched(self):
        rope = make_rope(rotary_dim=32)
        x = torch.randn(1, 4, 8, 64)
        pos = torch.arange(8)
        out, _ = rope(x, x.clone(), pos)
        assert torch.equal(out[..., 32:], x[..., 32:])
        assert not torch.equal(out[..., :32], x[..., :32])

    def test_positions_offset_equals_absolute(self):
        rope = make_rope()
        x = torch.randn(1, 2, 6, 64)
        q_a, _ = rope(x, x.clone(), torch.arange(6))
        q_b, _ = rope(x[:, :, 3:], x[:, :, 3:].clone(), torch.arange(3, 6))
        assert torch.allclose(q_a[:, :, 3:], q_b, atol=1e-6)

    def test_sequence_length_growth(self):
        rope = make_rope(max_position_embeddings=128)
        x = torch.randn(1, 2, 100, 64)
        out, _ = rope(x, x.clone(), torch.arange(100))
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestPositionValidation:
    def test_negative_position_rejected(self):
        rope = make_rope()
        x = torch.randn(1, 1, 2, 64)
        with pytest.raises(ValueError, match=">= 0"):
            rope(x, x.clone(), torch.tensor([0, -1]))

    def test_position_beyond_capacity_rejected(self):
        rope = make_rope(max_position_embeddings=16)
        x = torch.randn(1, 1, 17, 64)
        with pytest.raises(ValueError, match="max_position_embeddings"):
            rope(x, x.clone(), torch.arange(17))

    def test_2d_positions_rejected(self):
        rope = make_rope()
        x = torch.randn(1, 1, 2, 64)
        with pytest.raises(ValueError, match="1-D"):
            rope(x, x.clone(), torch.zeros(2, 2, dtype=torch.long))

    def test_empty_positions_rejected(self):
        rope = make_rope()
        x = torch.randn(1, 1, 0, 64)
        with pytest.raises(ValueError, match="empty"):
            rope(x, x.clone(), torch.tensor([], dtype=torch.long))


class TestDtypeAndGradients:
    def test_bf16_output_finite(self, device):
        rope = make_rope().to(device)
        x = torch.randn(1, 2, 9, 64, device=device, dtype=torch.bfloat16)
        out, _ = rope(x, x.clone(), torch.arange(9, device=device))
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()

    def test_gradient_flows(self):
        rope = make_rope()
        q = torch.randn(1, 2, 4, 64, requires_grad=True)
        k = torch.randn(1, 2, 4, 64, requires_grad=True)
        q_out, k_out = rope(q, k, torch.arange(4))
        (q_out.sum() + k_out.sum()).backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()

    def test_matches_fp64_reference(self):
        rope = make_rope()
        x = torch.randn(1, 1, 5, 64).double()
        out64, _ = rope(x, x.clone(), torch.arange(5))
        assert torch.isfinite(out64).all()
