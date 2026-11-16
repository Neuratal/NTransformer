"""Factorized embedding: full NTokenizer vocabulary, tie mechanics, ID validation."""

from __future__ import annotations

import pytest
import torch

from neuratal.config import NeuratalConfig
from neuratal.model.embeddings import FactorizedEmbedding

V = 262_144


class TestForward:
    def test_output_shape_factorized(self):
        emb = FactorizedEmbedding(NeuratalConfig())  # F=32 < D=256
        ids = torch.randint(0, V, (2, 7))
        out = emb(ids)
        assert out.shape == (2, 7, 256)

    def test_output_shape_full_width(self):
        emb = FactorizedEmbedding(NeuratalConfig(embedding_size=None))  # F == D
        ids = torch.randint(0, V, (2, 7))
        out = emb(ids)
        assert out.shape == (2, 7, 256)

    def test_all_special_token_ids_are_valid_inputs(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        specials = torch.tensor([[262137, 262138, 262139, 262140, 262141, 262142, 262143]])
        out = emb(specials)
        assert torch.isfinite(out).all()
        assert out.shape == (1, 7, 256)

    def test_boundary_ids(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        ids = torch.tensor([[0, 255, 256, V - 1]])
        out = emb(ids)
        assert torch.isfinite(out).all()

    def test_factorized_projection_matches_manual(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        ids = torch.randint(0, V, (1, 4))
        expected = emb.in_proj(emb.token_embedding(ids))
        assert torch.equal(emb(ids), expected)

    def test_in_proj_omitted_when_full_width(self):
        emb = FactorizedEmbedding(NeuratalConfig(embedding_size=None))
        assert emb.in_proj is None
        ids = torch.randint(0, V, (1, 4))
        assert torch.equal(emb(ids), emb.token_embedding(ids))


class TestInputValidation:
    def test_out_of_range_ids_rejected(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        with pytest.raises(ValueError, match="out of range"):
            emb(torch.tensor([[V]]))
        with pytest.raises(ValueError, match="out of range"):
            emb(torch.tensor([[-1]]))

    def test_float_ids_rejected(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        with pytest.raises(ValueError, match="integer dtype"):
            emb(torch.tensor([[1.5]]))

    def test_non_2d_rejected(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        with pytest.raises(ValueError, match="2-D"):
            emb(torch.tensor([1, 2, 3]))

    def test_non_tensor_rejected(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        with pytest.raises(TypeError, match="torch.Tensor"):
            emb([[1, 2, 3]])  # type: ignore[arg-type]


class TestParameterSharing:
    def test_embedding_weight_is_single_matrix(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        assert emb.embedding_weight is emb.token_embedding.weight
        assert emb.token_embedding.weight.shape == (V, 32)

    def test_parameter_count_boundary(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        expected = V * 32 + 32 * 256
        assert sum(p.numel() for p in emb.parameters()) == expected

    def test_gradient_reaches_shared_matrix(self):
        emb = FactorizedEmbedding(NeuratalConfig())
        ids = torch.randint(0, V, (1, 8))
        emb(ids).sum().backward()
        assert emb.token_embedding.weight.grad is not None
        assert torch.isfinite(emb.token_embedding.weight.grad.float()).all()
