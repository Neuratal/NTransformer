# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Generation: greedy determinism, sampling, EOS stop, cache/no-cache parity."""

from __future__ import annotations

import types

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel, generate
from neuratal.model.model import CausalLMOutput

V = 262_144
EOS_ID = 262_138  # <|eos|> per NTokenizer v0.3


def make_model(**overrides) -> NeuratalModel:
    torch.manual_seed(0)
    return NeuratalModel(NeuratalConfig(**overrides))


class TestGreedy:
    def test_deterministic_across_calls(self, device):
        model = make_model().to(device).eval()
        ids = torch.randint(0, V, (1, 6), device=device)
        out_a = generate(model, ids, max_new_tokens=8)
        out_b = generate(model, ids, max_new_tokens=8)
        assert torch.equal(out_a, out_b)

    def test_output_length_and_prefix(self, device):
        model = make_model().to(device).eval()
        ids = torch.randint(0, V, (1, 6), device=device)
        out = generate(model, ids, max_new_tokens=8)
        assert out.shape == (1, 14)
        assert torch.equal(out[:, :6], ids)

    def test_batch_size_preserved(self, device):
        model = make_model().to(device).eval()
        ids = torch.randint(0, V, (3, 5), device=device)
        out = generate(model, ids, max_new_tokens=4)
        assert out.shape == (3, 9)


class TestCacheParity:
    def test_cache_matches_no_cache_greedy(self, device):
        model = make_model().to(device).eval()
        ids = torch.randint(0, V, (1, 7), device=device)
        model.config.use_cache = True
        out_cached = generate(model, ids, max_new_tokens=6)
        model.config.use_cache = False
        out_plain = generate(model, ids, max_new_tokens=6)
        model.config.use_cache = True
        assert torch.equal(out_cached, out_plain)


class TestSampling:
    def test_seeded_sampling_is_reproducible(self):
        model = make_model().eval()
        ids = torch.randint(0, V, (1, 5))
        gen = torch.Generator().manual_seed(1234)
        out_a = generate(model, ids, max_new_tokens=6, do_sample=True, generator=gen)
        gen = torch.Generator().manual_seed(1234)
        out_b = generate(model, ids, max_new_tokens=6, do_sample=True, generator=gen)
        assert torch.equal(out_a, out_b)

    def test_temperature_validation(self):
        model = make_model().eval()
        ids = torch.randint(0, V, (1, 3))
        with pytest.raises(ValueError, match="temperature"):
            generate(model, ids, max_new_tokens=2, do_sample=True, temperature=0.0)

    def test_top_k_validation(self):
        model = make_model().eval()
        ids = torch.randint(0, V, (1, 3))
        with pytest.raises(ValueError, match="top_k"):
            generate(model, ids, max_new_tokens=2, do_sample=True, top_k=0)

    def test_max_new_tokens_validation(self):
        model = make_model().eval()
        ids = torch.randint(0, V, (1, 3))
        with pytest.raises(ValueError, match="max_new_tokens"):
            generate(model, ids, max_new_tokens=0)

    def test_top_k_one_is_greedy(self):
        model = make_model().eval()
        ids = torch.randint(0, V, (1, 5))
        gen = torch.Generator().manual_seed(99)
        out = generate(model, ids, max_new_tokens=4, do_sample=True, top_k=1, generator=gen)
        greedy = generate(model, ids, max_new_tokens=4)
        assert torch.equal(out, greedy)

    def test_generator_device_mismatch_rejected(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = make_model().to("cuda").eval()
        ids = torch.randint(0, V, (1, 5), device="cuda")
        gen = torch.Generator().manual_seed(0)  # CPU generator
        with pytest.raises(ValueError, match="generator"):
            generate(model, ids, max_new_tokens=2, do_sample=True, generator=gen)


class TestEosStop:
    def test_stops_when_all_sequences_emit_eos(self):
        model = make_model()
        ids = torch.randint(0, V, (1, 3))


        def fake_forward(self_model, input_ids, **kwargs):
            logits = torch.full((input_ids.size(0), input_ids.size(1), V), -1000.0)
            logits[:, :, EOS_ID] = 1000.0  # always argmax to EOS
            return CausalLMOutput(logits=logits, past_key_values=None)

        model.forward = types.MethodType(fake_forward, model)
        out = generate(model, ids, max_new_tokens=16, eos_token_id=EOS_ID)
        assert out.shape[1] == ids.size(1) + 1  # stopped after the first EOS
        assert out[0, -1].item() == EOS_ID

    def test_stopped_rows_are_padded_while_others_continue(self):
        model = make_model()
        pad_id = 262139
        ids = torch.randint(0, V, (2, 2))

        def fake_forward(self_model, input_ids, **kwargs):
            logits = torch.full((input_ids.size(0), input_ids.size(1), V), -1000.0)
            logits[0, :, EOS_ID] = 1000.0  # row 0 always argmax EOS
            logits[1, :, 42] = 1000.0      # row 1 never emits EOS
            return CausalLMOutput(logits=logits, past_key_values=None)

        model.forward = types.MethodType(fake_forward, model)
        out = generate(model, ids, max_new_tokens=4, eos_token_id=EOS_ID, pad_token_id=pad_id)
        assert out.shape[1] == ids.size(1) + 4  # row 1 never finishes -> full horizon
        assert out[0, 2].item() == EOS_ID       # row 0's first generated token is EOS
        assert (out[0, 3:] == pad_id).all()     # row 0 padded with pad_token_id afterwards
        assert (out[1, 2:] == 42).all()         # row 1 kept generating token 42


class TestNoInjection:
    def test_does_not_inject_special_tokens(self, device):
        model = make_model().to(device).eval()
        ids = torch.randint(0, 100, (1, 5), device=device)
        out = generate(model, ids, max_new_tokens=3)
        assert torch.equal(out[:, :5], ids)
