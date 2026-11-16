"""Output projection: tied matrix usage, exact chunked computation, validation."""

from __future__ import annotations

import pytest
import torch

from neuratal.config import NeuratalConfig
from neuratal.model.output import OutputProjection

V = 262_144


def make_head(**overrides) -> OutputProjection:
    return OutputProjection(NeuratalConfig(**overrides))


class TestFactorizedHead:
    def test_out_proj_exists_and_shapes(self):
        head = make_head()  # F=32 < D=256
        assert head.out_proj is not None
        assert head.out_proj.weight.shape == (32, 256)

    def test_logits_shape_full_vocab(self):
        head = make_head()
        weight = torch.randn(V, 32)
        hidden = torch.randn(2, 5, 256)
        logits = head(hidden, weight)
        assert logits.shape == (2, 5, V)
        assert torch.isfinite(logits).all()

    def test_exact_math(self):
        head = make_head()
        weight = torch.randn(V, 32)
        hidden = torch.randn(1, 3, 256)
        expected = (hidden @ head.out_proj.weight.t()) @ weight.t()
        assert torch.allclose(head(hidden, weight), expected, atol=0, rtol=0)

    def test_full_width_skips_out_proj(self):
        head = make_head(embedding_size=None)  # F == D == 256
        assert head.out_proj is None
        weight = torch.randn(V, 256)
        hidden = torch.randn(1, 2, 256)
        assert torch.allclose(head(hidden, weight), hidden @ weight.t(), atol=0, rtol=0)

    def test_wrong_weight_shape_rejected(self):
        head = make_head()
        hidden = torch.randn(1, 2, 256)
        with pytest.raises(ValueError, match="shape mismatch"):
            head(hidden, torch.randn(V, 64))  # wrong F
        with pytest.raises(ValueError, match="shape mismatch"):
            head(hidden, torch.randn(1000, 32))  # wrong V
        with pytest.raises(ValueError, match="2-D"):
            head(hidden, torch.randn(V, 32, 1))

    def test_wrong_hidden_dim_rejected(self):
        head = make_head()
        with pytest.raises(ValueError, match="hidden size"):
            head(torch.randn(1, 2, 128), torch.randn(V, 32))


class TestChunkedComputation:
    def test_chunked_matches_unchunked_exact(self):
        # mathematically identical (per-row independent matmuls); GEMM tiling may differ by ulps
        full = make_head()
        chunked = make_head(logits_chunk_size=2)
        chunked.load_state_dict(full.state_dict())
        weight = torch.randn(V, 32)
        hidden = torch.randn(1, 7, 256)
        out_full = full(hidden, weight)
        out_chunked = chunked(hidden, weight)
        assert out_full.shape == out_chunked.shape
        # measured max |diff| ≈ 6e-6 (fp32 GEMM kernel-tiling ulps, mathematically identical)
        assert torch.allclose(out_full, out_chunked, atol=1e-5, rtol=1e-5)

    def test_chunked_matches_unchunked_fp32_logits(self):
        full = make_head(fp32_logits=True)
        chunked = make_head(fp32_logits=True, logits_chunk_size=3)
        chunked.load_state_dict(full.state_dict())
        weight = torch.randn(V, 32)
        hidden = torch.randn(1, 8, 256)
        assert torch.allclose(full(hidden, weight), chunked(hidden, weight), atol=1e-5, rtol=1e-5)

    def test_chunk_size_larger_than_sequence_is_fine(self):
        head = make_head(logits_chunk_size=128)
        weight = torch.randn(V, 32)
        hidden = torch.randn(1, 4, 256)
        out = head(hidden, weight)
        assert out.shape == (1, 4, V)

    def test_fp32_logits_dtype(self):
        head = make_head(fp32_logits=True)
        weight = torch.randn(V, 32, dtype=torch.bfloat16)
        hidden = torch.randn(1, 2, 256, dtype=torch.bfloat16)
        out = head(hidden, weight)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()


class TestParameterOwnership:
    def test_head_owns_no_vocabulary_parameter(self):
        head = make_head()
        for name, p in head.named_parameters():
            # only the out_proj projection; never a [V, F] matrix
            assert p.dim() == 2
            assert p.numel() < V
