# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""MoE checkpointing: single-file, sharded, dense/MoE mismatch, training resume."""

from __future__ import annotations

import json

import pytest
import torch

from neuratal import (
    NeuratalConfig,
    NeuratalModel,
    load_pretrained,
    load_pretrained_sharded,
    save_pretrained,
    save_pretrained_sharded,
)
from neuratal.serialization.sharded import INDEX_FILENAME, plan_shards
from neuratal.training.checkpoint import (
    load_training_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
)
from neuratal.training.chunked_linear_ce import chunked_linear_cross_entropy
from neuratal.training.optim import WarmupCosineSchedule, build_optimizer


def tiny_moe(**overrides) -> NeuratalConfig:
    base = dict(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        embedding_size=8,
        intermediate_size=64,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_token=2,
    )
    base.update(overrides)
    return NeuratalConfig(**base)


def make_model(cfg=None):
    torch.manual_seed(0)
    return NeuratalModel(cfg or tiny_moe())


class TestSingleFile:
    def test_moe_roundtrip_is_bitwise(self, tmp_path):
        model = make_model().eval()
        ids = torch.randint(0, 256, (1, 6))
        with torch.no_grad():
            ref = model(ids).logits
        save_pretrained(model, tmp_path)
        loaded = load_pretrained(tmp_path).eval()
        assert loaded.config.to_dict() == model.config.to_dict()
        assert loaded.config.is_moe
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor, loaded.state_dict()[key]), key
        with torch.no_grad():
            out = loaded(ids).logits
        assert torch.equal(ref, out)

    def test_dense_and_moe_checkpoints_do_not_cross_load(self, tmp_path):
        moe = make_model().eval()
        save_pretrained(moe, tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        payload["config"]["num_experts"] = 0
        payload["config"]["num_experts_per_token"] = 2
        payload["config"]["moe_layer_indices"] = []
        payload["config"]["expert_hidden_dim"] = 64
        (tmp_path / "config.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="architecture mismatch"):
            load_pretrained(tmp_path)

    def test_dense_checkpoint_with_moe_config_rejected(self, tmp_path):
        dense = make_model(NeuratalConfig(vocab_size=256, hidden_size=32, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2,
                                          embedding_size=8, intermediate_size=64)).eval()
        save_pretrained(dense, tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        payload["config"].update({
            "num_experts": 4, "num_experts_per_token": 2,
            "moe_layer_indices": [0, 1], "expert_hidden_dim": 64,
        })
        (tmp_path / "config.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="architecture mismatch"):
            load_pretrained(tmp_path)


class TestSharded:
    def test_plan_shards_bounds_and_no_tensor_split(self):
        state = {
            "a": torch.zeros(100, dtype=torch.float32),  # 400 B
            "b": torch.zeros(100, dtype=torch.float32),
            "c": torch.zeros(1000, dtype=torch.float32),  # 4000 B, own shard
        }
        shards = plan_shards(state, max_shard_bytes=500)
        flat = [k for s in shards for k in s]
        assert sorted(flat) == ["a", "b", "c"]
        assert any("c" in s for s in shards)
        assert all(sum(t.numel() * t.element_size() for t in s.values()) <= 500
                   or len(s) == 1 for s in shards)

    def test_moe_sharded_roundtrip(self, tmp_path):
        model = make_model().eval()
        ids = torch.randint(0, 256, (1, 7))
        with torch.no_grad():
            ref = model(ids).logits
        save_pretrained_sharded(model, tmp_path, max_shard_bytes=8 * 1024)
        index = json.loads((tmp_path / INDEX_FILENAME).read_text())
        assert index["metadata"]["num_shards"] > 1
        assert set(index["weight_map"]) == set(model.state_dict())
        loaded = load_pretrained_sharded(tmp_path).eval()
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor, loaded.state_dict()[key]), key
        with torch.no_grad():
            out = loaded(ids).logits
        assert torch.equal(ref, out)

    def test_missing_shard_rejected(self, tmp_path):
        model = make_model()
        save_pretrained_sharded(model, tmp_path, max_shard_bytes=8 * 1024)
        index = json.loads((tmp_path / INDEX_FILENAME).read_text())
        victim = sorted(set(index["weight_map"].values()))[0]
        (tmp_path / victim).unlink()
        with pytest.raises(ValueError, match="missing shard"):
            load_pretrained_sharded(tmp_path)


def _train_one_step(model, optimizer, ids, labels):
    hidden = model.forward_hidden(ids)
    total, n = chunked_linear_cross_entropy(
        hidden, labels, model.output_projection.out_proj.weight,
        model.embed_tokens.embedding_weight, chunk_size=16,
    )
    loss = total / n.clamp_min(1)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


class TestTrainingResume:
    def test_moe_optimizer_state_and_resume_parity(self, tmp_path):
        torch.manual_seed(3)
        cfg = tiny_moe()
        model = make_model(cfg)
        opt = build_optimizer(model, learning_rate=1e-3)
        sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=10)
        ids = torch.randint(0, 256, (2, 8))
        labels = torch.randint(0, 256, (2, 8))
        loss1 = _train_one_step(model, opt, ids, labels)
        sched.step()
        assert torch.isfinite(torch.tensor(loss1))

        torch.manual_seed(99)
        saved_rng = torch.get_rng_state().clone()
        save_training_checkpoint(model, opt, sched, tmp_path, global_step=1, tokens_seen=16,
                                 tokenizer_fingerprint="abc", precision="float32")
        meta = json.loads((tmp_path / "trainer_meta.json").read_text())
        assert meta["architecture"] == "moe"
        assert meta["num_experts"] == cfg.num_experts
        assert meta["moe_layers"] == cfg.moe_layer_indices

        # Corrupt the live RNG, then prove resume restores the saved state.
        torch.manual_seed(12345)
        model2, loaded_meta = load_training_checkpoint(tmp_path)
        opt2 = build_optimizer(model2, learning_rate=1e-3)
        sched2 = WarmupCosineSchedule(opt2, warmup_steps=1, total_steps=10)
        resumed = resume_training_checkpoint(opt2, sched2, tmp_path, model=model2)
        assert resumed["global_step"] == 1
        assert resumed["tokens_seen"] == 16
        assert sched2.current_step == 1
        assert torch.equal(torch.get_rng_state(), saved_rng)
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor, model2.state_dict()[key]), key

        # Expert and router optimizer state must be restored per parameter.
        checked = 0
        for name, p in model2.named_parameters():
            if ".mlp.router." in name or ".mlp.experts." in name:
                state = opt2.state.get(p)
                assert state is not None, name
                assert "exp_avg" in state and "exp_avg_sq" in state, name
                assert torch.isfinite(state["exp_avg"]).all()
                checked += 1
        assert checked > 0

        # Continue one identical step on both models; parameters must match.
        loss_a = _train_one_step(model, opt, ids, labels)
        loss_b = _train_one_step(model2, opt2, ids, labels)
        assert abs(loss_a - loss_b) < 1e-6
        for key, tensor in model.state_dict().items():
            assert torch.allclose(tensor, model2.state_dict()[key], atol=1e-6, rtol=1e-5), key

    def test_resume_rejects_dense_moe_mismatch(self, tmp_path):
        moe = make_model()
        opt = build_optimizer(moe)
        sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=10)
        save_training_checkpoint(moe, opt, sched, tmp_path, global_step=1)
        dense = make_model(NeuratalConfig(vocab_size=256, hidden_size=32, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2,
                                          embedding_size=8, intermediate_size=64))
        opt2 = build_optimizer(dense)
        sched2 = WarmupCosineSchedule(opt2, warmup_steps=1, total_steps=10)
        with pytest.raises(ValueError, match="architecture mismatch"):
            resume_training_checkpoint(opt2, sched2, tmp_path, model=dense)
