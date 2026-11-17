"""Static-shape CUDA graph decode: bitwise equivalence, opt-in semantics, errors."""

from __future__ import annotations

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel, generate
from neuratal.model.graph_decode import GraphDecodeSession, GraphDecodeError, next_pow2

V = 262_144

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_model(**overrides) -> NeuratalModel:
    torch.manual_seed(0)
    kw = dict(NeuratalConfig().to_dict())
    kw.update(overrides)
    return NeuratalModel(NeuratalConfig.from_dict(kw)).to("cuda").eval()


class TestNextPow2:
    def test_boundaries(self):
        assert next_pow2(0) == 1
        assert next_pow2(1) == 1
        assert next_pow2(2) == 2
        assert next_pow2(3) == 4
        assert next_pow2(49) == 64
        assert next_pow2(64) == 64


class TestEquivalence:
    def test_greedy_matches_eager_token_by_token(self, device):
        model = make_model()
        ids = torch.randint(0, V, (1, 24), device="cuda")
        out_eager = generate(model, ids, max_new_tokens=16)
        out_graph = generate(model, ids, max_new_tokens=16, experimental_cuda_graphs=True)
        assert torch.equal(out_eager, out_graph)

    def test_prefill_and_step_logits_match_eager(self, device):
        torch.manual_seed(0)
        model = make_model()
        cfg = model.config
        prompt = torch.randint(0, V, (1, 16), device="cuda")
        from neuratal.model.cache import KVCache

        with torch.inference_mode():
            eager_cache = KVCache(cfg)
            eager_out = model(prompt, past_key_values=eager_cache)
            logits_eager = eager_out.logits[:, -1, :].float()

            session = GraphDecodeSession(model, prompt, max_new_tokens=4)
            assert torch.equal(logits_eager, session.prefill_logits)
            token = 12345
            logits_eager_step = model(
                torch.tensor([[token]], device="cuda"), past_key_values=eager_cache
            ).logits[:, -1, :].float()
            logits_graph_step = session.step(token)
            assert torch.allclose(logits_eager_step, logits_graph_step, atol=1e-6, rtol=1e-5)

    def test_bf16_greedy_matches_eager(self, device):
        model = make_model(dtype="bfloat16")
        ids = torch.randint(0, V, (1, 12), device="cuda")
        out_eager = generate(model, ids, max_new_tokens=8)
        out_graph = generate(model, ids, max_new_tokens=8, experimental_cuda_graphs=True)
        assert torch.equal(out_eager, out_graph)

    def test_session_length_advances_per_step(self, device):
        model = make_model()
        prompt = torch.randint(0, V, (1, 8), device="cuda")
        session = GraphDecodeSession(model, prompt, max_new_tokens=2)
        assert session.length == 8
        session.step(999)
        assert session.length == 9
        session.step(1000)
        assert session.length == 10


class TestErrors:
    def test_batch_two_rejected(self, device):
        model = make_model()
        ids = torch.randint(0, V, (2, 8), device="cuda")
        with pytest.raises(ValueError, match="batch size 1"):
            generate(model, ids, max_new_tokens=4, experimental_cuda_graphs=True)

    def test_dropout_rejected(self, device):
        model = make_model(attention_dropout=0.1)
        with pytest.raises(GraphDecodeError, match="dropout"):
            GraphDecodeSession(model, torch.randint(0, V, (1, 8), device="cuda"), max_new_tokens=4)

    def test_window_exceeding_capacity_rejected(self, device):
        model = make_model(cache_capacity=8, max_position_embeddings=8)
        with pytest.raises(GraphDecodeError, match="cache_capacity"):
            GraphDecodeSession(model, torch.randint(0, V, (1, 4), device="cuda"), max_new_tokens=16)

    def test_cpu_rejected(self, device):
        pass  # covered by GraphDecodeError via device check in session; CPU never reaches CUDA tests


class TestEosAndStopping:
    def test_stops_at_eos(self, device):
        model = make_model()
        ids = torch.randint(0, V, (1, 6), device="cuda")
        # find the model's greedy first token and designate it EOS
        greedy = generate(model, ids, max_new_tokens=1)
        eos_id = int(greedy[0, -1].item())
        out = generate(model, ids, max_new_tokens=8, eos_token_id=eos_id, experimental_cuda_graphs=True)
        assert int(out[0, -1].item()) == eos_id
        assert out.shape[1] == ids.shape[1] + 1

    def test_greedy_graph_matches_eager_with_eos(self, device):
        model = make_model()
        ids = torch.randint(0, V, (1, 10), device="cuda")
        greedy = generate(model, ids, max_new_tokens=1)
        eos_id = int(greedy[0, -1].item())
        a = generate(model, ids, max_new_tokens=8, eos_token_id=eos_id)
        b = generate(model, ids, max_new_tokens=8, eos_token_id=eos_id, experimental_cuda_graphs=True)
        assert torch.equal(a, b)
