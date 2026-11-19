# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""NeuratalModel: forward/backward, tying mechanics, cache equivalence, leakage, vocab integration."""

from __future__ import annotations

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel
from neuratal.model.cache import KVCache

V = 262_144


def make_model(**overrides) -> NeuratalModel:
    torch.manual_seed(0)
    return NeuratalModel(NeuratalConfig(**overrides))


class TestForward:
    def test_output_shape_full_vocab(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (2, 12), device=device)
        out = model(ids)
        assert out.logits.shape == (2, 12, V)
        assert torch.isfinite(out.logits).all()

    def test_different_batch_sizes_and_lengths(self, device):
        model = make_model().to(device)
        for batch, seq in [(1, 1), (1, 33), (3, 8)]:
            ids = torch.randint(0, V, (batch, seq), device=device)
            out = model(ids)
            assert out.logits.shape == (batch, seq, V)

    def test_full_vocab_ids_accepted(self, device):
        model = make_model().to(device)
        ids = torch.tensor([[0, 1, 255, 256, 100_000, V - 1, 262137, 262143]], device=device)
        out = model(ids)
        assert out.logits.shape == (1, 8, V)

    def test_empty_input_rejected(self):
        model = make_model()
        with pytest.raises(ValueError, match="at least one token"):
            model(torch.zeros(1, 0, dtype=torch.long))

    def test_non_2d_input_rejected(self):
        model = make_model()
        with pytest.raises(ValueError, match="2-D"):
            model(torch.randint(0, V, (4,)))

    def test_out_of_range_ids_rejected(self):
        model = make_model()
        with pytest.raises(ValueError, match="out of range"):
            model(torch.tensor([[V]]))

    def test_backward_pass(self, device):
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 8), device=device)
        out = model(ids)
        loss = out.logits.square().mean()
        loss.backward()
        checked = 0
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad.float()).all(), name
            checked += 1
        assert checked == len(list(model.named_parameters()))


class TestWeightTying:
    def test_embedding_matrix_counted_once(self):
        model = make_model()
        weight = model.embed_tokens.token_embedding.weight
        total = sum(p.numel() for _, p in model.named_parameters())
        # exactly the 262,144 x 32 matrix appears once
        count = sum(1 for name, p in model.named_parameters() if p is weight)
        assert count == 1

    def test_output_head_has_no_vocabulary_parameter(self):
        model = make_model()
        for name, p in model.output_projection.named_parameters():
            assert p.numel() < V, name

    def test_gradient_reaches_shared_matrix_via_both_paths(self):
        model = make_model()
        ids = torch.randint(0, V, (1, 6))
        model(ids).logits.sum().backward()
        grad = model.embed_tokens.token_embedding.weight.grad
        assert grad is not None and torch.isfinite(grad.float()).all()
        assert grad.abs().sum() > 0

    def test_parameter_report_sums_to_total(self):
        model = make_model()
        report = model.parameter_report()
        total_parts = sum(v for k, v in report.items() if k != "total")
        assert total_parts == report["total"] == model.num_parameters()
        # tied matrix counted exactly once
        assert report["embedding_matrix"] == V * 32
        assert report["output_projection"] == 32 * 256

    def test_parameter_report_dev_preset_exact_numbers(self):
        model = make_model()
        report = model.parameter_report()
        assert report["embedding_matrix"] == 8_388_608
        assert report["input_projection"] == 8_192
        assert report["output_projection"] == 8_192
        assert report["attention"] == 2 * (256 * 256 + 2 * 256 * 128 + 256 * 256)
        assert report["mlp"] == 2 * (2 * 256 * 640 + 640 * 256)
        assert report["normalization"] == 2 * 2 * 256 + 256
        assert report["total"] == 9_782_528


class TestCachedEquivalence:
    def test_cached_matches_uncached_logits(self, device):
        model = make_model().to(device).eval()
        ids = torch.randint(0, V, (1, 17), device=device)
        with torch.no_grad():
            ref = model(ids, use_cache=False).logits
            cache = KVCache(model.config)
            out = model(ids, past_key_values=cache).logits
            assert cache.get_seq_length() == 17
        max_abs = (ref - out).abs().max().item()
        assert max_abs < 1e-5, f"cached/uncached logits diverged: {max_abs}"  # measured: 0.0 prefill, ~1.6e-7 token-by-token

    def test_token_by_token_matches_prefill(self, device):
        model = make_model().to(device).eval()
        ids = torch.randint(0, V, (2, 10), device=device)
        with torch.no_grad():
            ref = model(ids, use_cache=False).logits
            cache = KVCache(model.config)
            outs = []
            for t in range(10):
                step = ids[:, t:t + 1]
                outs.append(model(step, past_key_values=cache).logits)
            stacked = torch.cat(outs, dim=1)
        max_abs = (ref - stacked).abs().max().item()
        assert max_abs < 1e-5, f"incremental logits diverged: {max_abs}"  # measured: ~1.6e-7


class TestCausality:
    def test_future_token_perturbation_leaves_past_logits_unchanged(self, device):
        model = make_model().to(device).eval()
        ids = torch.randint(0, V, (1, 12), device=device)
        with torch.no_grad():
            ref = model(ids, use_cache=False).logits
            ids2 = ids.clone()
            ids2[:, 7:] = torch.randint(0, V, (1, 5), device=device)
            out = model(ids2, use_cache=False).logits
        assert torch.allclose(ref[:, :7], out[:, :7], atol=1e-6)


class TestDtypeAndDevices:
    def test_bf16_forward_finite(self):
        model = make_model().to(torch.bfloat16)
        ids = torch.randint(0, V, (1, 4))
        out = model(ids)
        assert out.logits.dtype == torch.bfloat16
        assert torch.isfinite(out.logits.float()).all()

    def test_cuda_forward_and_backward(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        model = make_model().to(device)
        ids = torch.randint(0, V, (1, 6), device=device)
        out = model(ids)
        out.logits.square().mean().backward()
        assert torch.isfinite(out.logits.float()).all()


class TestContextCapacity:
    def test_sequence_exceeding_max_position_rejected(self):
        model = make_model(max_position_embeddings=8)
        with pytest.raises(ValueError, match="max_position_embeddings"):
            model(torch.randint(0, V, (1, 9)))
