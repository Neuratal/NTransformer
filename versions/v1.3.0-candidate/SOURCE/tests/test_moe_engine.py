"""MoE training-engine integration: losses, metrics, gradients, checkpoints, BF16."""

from __future__ import annotations

import math

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel
from neuratal.training.data import SyntheticTokenDataset
from neuratal.training.engine import TrainConfig, train
from neuratal.training.checkpoint import load_training_checkpoint


V = 256


def tiny_cfg(**overrides):
    base = dict(
        vocab_size=V, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, embedding_size=8, intermediate_size=64,
        max_position_embeddings=64, num_experts=4, num_experts_per_token=2,
    )
    base.update(overrides)
    return NeuratalConfig(**base)


def run_training(tmp_path, *, cfg=None, **cfg_overrides):
    torch.manual_seed(0)
    model = NeuratalModel(cfg or tiny_cfg())
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    train_ds = SyntheticTokenDataset(8, 16, vocab_size=V, seed=11)
    val_ds = SyntheticTokenDataset(3, 16, vocab_size=V, seed=12)
    tc = TrainConfig(
        batch_size=2, sequence_length=16, learning_rate=1e-3, warmup_steps=1,
        grad_clip_norm=1.0, gradient_accumulation_steps=2, precision="float32",
        eval_every_steps=2, log_every_steps=1, checkpoint_every_steps=2,
        max_steps=2, out_dir=str(tmp_path), device="cpu",
    )
    for k, v in cfg_overrides.items():
        setattr(tc, k, v)
    trained, history = train(model, train_ds, val_ds, tc)
    return trained, history, before


class TestMoETraining:
    def test_two_steps_finite_metrics_and_parameter_updates(self, tmp_path):
        model, history, before = run_training(tmp_path)
        log = [h for h in history if "main_loss" in h]
        assert len(log) == 2
        for entry in log:
            assert math.isfinite(entry["main_loss"])
            assert math.isfinite(entry["grad_norm"])
            assert entry["active_experts"] >= 1
            assert len(entry["tokens_per_expert"]) == 4
            # counts are aggregated across the model's two MoE layers
            assert sum(entry["tokens_per_expert"]) == 2 * 2 * 16 * 2 * 2
            assert abs(sum(entry["expert_utilization"]) - 1.0) < 1e-5
            assert entry["routing_entropy"] > 0
            assert entry["dropped_tokens"] >= 0
            assert math.isfinite(entry["aux_loss"])
            assert math.isfinite(entry["router_z_loss"])
        assert any("val_loss" in h for h in history)

        changed = [name for name, p in model.named_parameters()
                   if not torch.equal(p.detach(), before[name])]
        assert changed
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
        # Expert and router parameters received real updates.
        assert any(".mlp.router." in n for n in changed)
        assert any(".mlp.experts." in n for n in changed)

        # Checkpoint written at step 2 and loadable.
        ckpt = tmp_path / "ckpt-2"
        assert (ckpt / "trainer_meta.json").is_file()
        loaded, meta = load_training_checkpoint(ckpt)
        assert meta["architecture"] == "moe"
        assert meta["global_step"] == 2
        assert loaded.config.is_moe

    def test_dense_training_history_has_no_moe_fields(self, tmp_path):
        dense = NeuratalConfig(vocab_size=V, hidden_size=32, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2,
                               embedding_size=8, intermediate_size=64,
                               max_position_embeddings=64)
        _, history, _ = run_training(tmp_path, cfg=dense)
        log = [h for h in history if "main_loss" in h]
        assert log
        assert all("aux_loss" not in h for h in log)
        assert all("active_experts" not in h for h in log)

    def test_no_moe_parameters_created_when_disabled(self):
        dense = NeuratalModel(NeuratalConfig(vocab_size=V, hidden_size=32, num_hidden_layers=2,
                                             num_attention_heads=4, num_key_value_heads=2,
                                             embedding_size=8, intermediate_size=64))
        assert not any(".mlp.router." in n or ".mlp.experts." in n
                       for n, _ in dense.named_parameters())

    def test_resume_from_checkpoint_restores_counters(self, tmp_path):
        model, _, _ = run_training(tmp_path)
        ckpt = tmp_path / "ckpt-2"
        loaded, meta = load_training_checkpoint(ckpt)
        assert meta["tokens_seen"] == 2 * 2 * 16 * 2  # microbatches: steps * accum * batch * seq
        # Resume the training engine for one more step from the saved state.
        model2, _ = load_training_checkpoint(ckpt)
        train_ds = SyntheticTokenDataset(8, 16, vocab_size=V, seed=11)
        tc = TrainConfig(
            batch_size=2, sequence_length=16, learning_rate=1e-3, warmup_steps=1,
            grad_clip_norm=1.0, gradient_accumulation_steps=2, precision="float32",
            eval_every_steps=99, log_every_steps=1, checkpoint_every_steps=99,
            max_steps=3, out_dir=str(tmp_path / "resume"), device="cpu",
        )
        _, history = train(model2, train_ds, None, tc, resume_from=str(ckpt))
        assert history and history[0]["step"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoEBF16:
    def test_bf16_training_step_finite(self, tmp_path):
        torch.manual_seed(0)
        model = NeuratalModel(tiny_cfg()).to(torch.bfloat16)
        train_ds = SyntheticTokenDataset(4, 16, vocab_size=V, seed=21)
        tc = TrainConfig(
            batch_size=1, sequence_length=16, learning_rate=1e-3, warmup_steps=0,
            gradient_accumulation_steps=1, precision="bfloat16", eval_every_steps=99,
            log_every_steps=1, checkpoint_every_steps=99, max_steps=1,
            out_dir=str(tmp_path), device="cuda",
        )
        model, history = train(model, train_ds, None, tc)
        assert history and math.isfinite(history[0]["main_loss"])
        assert math.isfinite(history[0]["aux_loss"])
        assert torch.isfinite(torch.tensor(history[0]["grad_norm"]))
