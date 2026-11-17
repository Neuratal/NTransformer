"""Regression tests for engine/checkpoint fixes.

Covers: full-width tied head (embedding_size == hidden_size, out_proj=None),
tokenizer-fingerprint validation on resume, and GradScaler save/restore.

Synthetic data only — no real training, no corpus downloads.
"""

from __future__ import annotations

import math

import pytest
import torch

from neuratal import NeuratalConfig, NeuratalModel
from neuratal.training.checkpoint import (
    resume_training_checkpoint,
    save_training_checkpoint,
)
from neuratal.training.data import SyntheticTokenDataset
from neuratal.training.engine import TrainConfig, train
from neuratal.training.optim import WarmupCosineSchedule, build_optimizer


def full_width_model():
    torch.manual_seed(0)
    cfg = NeuratalConfig(
        vocab_size=4096, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=1, num_key_value_heads=1,
        embedding_size=None,  # full-width: OutputProjection.out_proj is None
        intermediate_size=64, max_position_embeddings=64,
    )
    assert cfg.embedding_size == cfg.hidden_size
    return NeuratalModel(cfg)


def test_full_width_train_step_and_checkpoint(tmp_path):
    model = full_width_model()
    assert model.output_projection.out_proj is None
    ds = SyntheticTokenDataset(4, 8, vocab_size=4096, seed=5)
    tc = TrainConfig(
        batch_size=1, sequence_length=8, learning_rate=1e-3, warmup_steps=1,
        gradient_accumulation_steps=1, precision="float32",
        eval_every_steps=99, log_every_steps=1, checkpoint_every_steps=1,
        max_steps=2, out_dir=str(tmp_path), device="cpu",
    )
    trained, history = train(model, ds, None, tc)
    assert history and math.isfinite(history[0]["main_loss"])
    # checkpoint roundtrip works for full-width heads too
    ckpt = tmp_path / "ckpt-2"
    assert (ckpt / "model.safetensors").is_file()
    assert (ckpt / "trainer_state.safetensors").is_file()


class TestFingerprintValidation:
    def _save(self, tmp_path, fingerprint):
        model = full_width_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=4)
        save_training_checkpoint(model, opt, sched, tmp_path, global_step=1,
                                 tokens_seen=64, tokenizer_fingerprint=fingerprint)
        return tmp_path

    def test_matching_fingerprint_resumes(self, tmp_path):
        ckpt = self._save(tmp_path / "ckpt", "abc123")
        opt = build_optimizer(full_width_model(), learning_rate=1e-3)
        sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=4)
        meta = resume_training_checkpoint(opt, sched, ckpt, expected_tokenizer_fingerprint="abc123")
        assert meta["global_step"] == 1

    def test_mismatched_fingerprint_rejected(self, tmp_path):
        ckpt = self._save(tmp_path / "ckpt", "abc123")
        opt = build_optimizer(full_width_model(), learning_rate=1e-3)
        sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=4)
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            resume_training_checkpoint(opt, sched, ckpt, expected_tokenizer_fingerprint="deadbeef")

    def test_missing_fingerprint_rejected_when_expected(self, tmp_path):
        ckpt = self._save(tmp_path / "ckpt", None)  # checkpoint records none
        opt = build_optimizer(full_width_model(), learning_rate=1e-3)
        sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=4)
        with pytest.raises(ValueError, match="does not record"):
            resume_training_checkpoint(opt, sched, ckpt, expected_tokenizer_fingerprint="abc123")

    def test_fingerprint_check_skippable(self, tmp_path):
        ckpt = self._save(tmp_path / "ckpt", "abc123")
        opt = build_optimizer(full_width_model(), learning_rate=1e-3)
        sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=4)
        meta = resume_training_checkpoint(opt, sched, ckpt, expected_tokenizer_fingerprint=None)
        assert meta["global_step"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler path requires CUDA")
def tiny_cfg():
    return NeuratalConfig(
        vocab_size=4096, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=1, num_key_value_heads=1, embedding_size=8,
        intermediate_size=64, max_position_embeddings=64,
    )


def test_fp16_step_and_scaler_resume(tmp_path):
    torch.manual_seed(0)
    model = NeuratalModel(tiny_cfg()).to("cuda")
    ds = SyntheticTokenDataset(4, 8, vocab_size=4096, seed=5)
    tc = TrainConfig(
        batch_size=1, sequence_length=8, learning_rate=1e-3, warmup_steps=0,
        gradient_accumulation_steps=1, precision="float16",
        eval_every_steps=99, log_every_steps=1, checkpoint_every_steps=1,
        max_steps=1, out_dir=str(tmp_path), device="cuda",
    )
    trained, history = train(model, ds, None, tc)
    assert history and math.isfinite(history[0]["main_loss"])

    from safetensors.torch import load_file
    state = load_file(str(tmp_path / "ckpt-1" / "trainer_state.safetensors"))
    assert "trainer.grad_scaler.scale" in state
    saved_scale = float(state["trainer.grad_scaler.scale"].item())

    model2 = NeuratalModel(tiny_cfg()).to("cuda")
    opt2 = build_optimizer(model2, learning_rate=1e-3)
    sched2 = WarmupCosineSchedule(opt2, warmup_steps=1, total_steps=4)
    scaler2 = torch.amp.GradScaler("cuda")
    scaler2.scale(torch.zeros((), device="cuda"))  # materialize internal tensors
    scaler2._scale.fill_(1.0)  # prove resume overwrites the scale
    resume_training_checkpoint(opt2, sched2, tmp_path / "ckpt-1",
                               expected_tokenizer_fingerprint=None,
                               grad_scaler=scaler2)
    assert float(scaler2.state_dict()["scale"]) == pytest.approx(saved_scale)
    # optimizer state actually restored (AdamW step counter incremented at save time)
    steps = [float(opt2.state[p]["step"]) for p in opt2.param_groups[0]["params"] if p in opt2.state]
    assert steps and all(s > 0 for s in steps)
