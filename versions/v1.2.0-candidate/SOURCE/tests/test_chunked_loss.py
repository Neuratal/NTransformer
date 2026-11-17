"""Chunked cross-entropy: parity with standard CE + memory behavior."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuratal.model.model import NeuratalModel
from neuratal.config import NeuratalConfig
from neuratal.training.chunked_loss import chunked_cross_entropy, chunked_cross_entropy_batch

V = 262_144


def make_model(**overrides):
    torch.manual_seed(0)
    cfg = NeuratalConfig(vocab_size=V, hidden_size=64, num_hidden_layers=1,
                         num_attention_heads=1, num_key_value_heads=1,
                         embedding_size=8, intermediate_size=128, **overrides)
    return NeuratalModel(cfg).eval()


class TestParity:
    def test_matches_standard_ce_fp32(self):
        model = make_model()
        ids = torch.randint(0, V, (2, 16))
        targets = torch.randint(0, V, (2, 16))
        with torch.no_grad():
            out = model(ids)
            h = out.logits if hasattr(out, "logits") else out[0]
            # reconstruct hidden via model internals for direct CE comparison
            hidden = model.embed_tokens(ids)
            hidden = model.layers[0](hidden, torch.arange(ids.size(1)), past_key_values=None)
            hidden = model.final_norm(hidden)
            logits = model.output_projection(hidden, model.embed_tokens.embedding_weight)
            ref = F.cross_entropy(logits.float().reshape(-1, V), targets.reshape(-1), reduction="mean")

            chunked, _ = chunked_cross_entropy(hidden, targets,
                                               model.output_projection,
                                               model.embed_tokens.embedding_weight,
                                               chunk_size=4)
            ref_sum = F.cross_entropy(logits.float().reshape(-1, V), targets.reshape(-1), reduction="sum")
            assert torch.allclose(chunked, ref_sum, atol=1e-4, rtol=1e-4)

    def test_grad_flow_to_all_params(self):
        model = make_model()
        model.train()
        ids = torch.randint(0, V, (1, 8))
        targets = torch.randint(0, V, (1, 8))
        hidden = model.embed_tokens(ids)
        hidden = model.layers[0](hidden, torch.arange(8))
        hidden = model.final_norm(hidden)
        total, n = chunked_cross_entropy(hidden, targets,
                                         model.output_projection,
                                         model.embed_tokens.embedding_weight,
                                         chunk_size=4, training=True)
        loss = total / n.clamp_min(1)
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(p.grad).all(), name


class TestIgnoreIndex:
    def test_ignore_index_excluded_from_count(self):
        model = make_model()
        ids = torch.randint(0, V, (1, 8))
        targets = torch.tensor([[IGNORE := -100, 1, -100, 2, -100, 3, -100, 4]])
        hidden = model.embed_tokens(ids)
        hidden = model.layers[0](hidden, torch.arange(8))
        hidden = model.final_norm(hidden)
        total, n = chunked_cross_entropy(hidden, targets,
                                         model.output_projection,
                                         model.embed_tokens.embedding_weight,
                                         chunk_size=4, training=False)
        assert int(n.item()) == 4


class TestChunkSize:
    def test_chunk_size_one(self):
        model = make_model()
        ids = torch.randint(0, V, (1, 4))
        targets = torch.randint(0, V, (1, 4))
        hidden = model.embed_tokens(ids)
        hidden = model.layers[0](hidden, torch.arange(4))
        hidden = model.final_norm(hidden)
        total, n = chunked_cross_entropy(hidden, targets,
                                         model.output_projection,
                                         model.embed_tokens.embedding_weight,
                                         chunk_size=1, training=False)
        assert torch.isfinite(total).all()

    def test_chunk_size_larger_than_seq(self):
        model = make_model()
        ids = torch.randint(0, V, (1, 4))
        targets = torch.randint(0, V, (1, 4))
        hidden = model.embed_tokens(ids)
        hidden = model.layers[0](hidden, torch.arange(4))
        hidden = model.final_norm(hidden)
        total, n = chunked_cross_entropy(hidden, targets,
                                         model.output_projection,
                                         model.embed_tokens.embedding_weight,
                                         chunk_size=128, training=False)
        assert torch.isfinite(total).all()
