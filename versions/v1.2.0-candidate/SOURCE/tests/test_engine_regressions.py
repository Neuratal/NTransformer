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


def tiny_cfg():
    return NeuratalConfig(
        vocab_size=4096, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=1, num_key_value_heads=1, embedding_size=8,
        intermediate_size=64, max_position_embeddings=64,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler path requires CUDA")
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


def test_resume_restores_rng_and_optimizer_state(tmp_path):
    from torch.utils.data import DataLoader

    from neuratal.training.chunked_linear_ce import chunked_linear_cross_entropy
    from neuratal.training.data import collate

    torch.manual_seed(123)
    model = full_width_model()
    opt = build_optimizer(model, learning_rate=1e-3)
    sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=8)
    ds = SyntheticTokenDataset(4, 8, vocab_size=4096, seed=9)
    batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=collate)))
    for _ in range(2):
        hidden = model.forward_hidden(batch["input_ids"])
        total, n_tok = chunked_linear_cross_entropy(
            hidden, batch["labels"], None, model.embed_tokens.embedding_weight)
        (total / n_tok.clamp_min(1)).backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    # reference map by (group_idx, param_idx) — the checkpoint's own addressing
    ref_state = {}
    for i, group in enumerate(opt.param_groups):
        for j, p in enumerate(group["params"]):
            st = opt.state.get(p)
            if st is not None:
                ref_state[(i, j)] = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                                     for k, v in st.items()}
    rng_before = torch.get_rng_state()
    save_training_checkpoint(model, opt, sched, tmp_path / "ckpt", global_step=2,
                             tokens_seen=16, tokenizer_fingerprint="fp-1")

    # corrupt the world, then resume
    torch.manual_seed(999)
    model2 = full_width_model()
    opt2 = build_optimizer(model2, learning_rate=1e-3)
    sched2 = WarmupCosineSchedule(opt2, warmup_steps=1, total_steps=8)
    assert sched2.current_step == 0
    meta = resume_training_checkpoint(opt2, sched2, tmp_path / "ckpt",
                                      expected_tokenizer_fingerprint="fp-1")
    assert meta["global_step"] == 2
    assert sched2.current_step == 2
    assert torch.equal(torch.get_rng_state(), rng_before)

    # optimizer tensors restored exactly (not merely "finite")
    matched = 0
    for i, group in enumerate(opt2.param_groups):
        for j, p in enumerate(group["params"]):
            st = opt2.state.get(p)
            if st is None:
                continue
            ref = ref_state[(i, j)]
            for k, v in st.items():
                if torch.is_tensor(v):
                    assert torch.equal(v.detach().cpu(), ref[k]), (i, j, k)
            matched += 1
    assert matched == len(ref_state)


def test_resume_then_continue_updates_step_counter(tmp_path):
    torch.manual_seed(7)
    model = full_width_model()
    opt = build_optimizer(model, learning_rate=1e-3)
    sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=8)
    save_training_checkpoint(model, opt, sched, tmp_path / "ckpt", global_step=3,
                             tokens_seen=24, tokenizer_fingerprint=None)

    # resume into a *fresh* engine and run two more steps
    model2 = full_width_model()
    ds = SyntheticTokenDataset(8, 8, vocab_size=4096, seed=3)
    tc = TrainConfig(batch_size=1, sequence_length=8, learning_rate=1e-3,
                     warmup_steps=1, gradient_accumulation_steps=1,
                     precision="float32", eval_every_steps=99, log_every_steps=1,
                     checkpoint_every_steps=99, max_steps=5,
                     out_dir=str(tmp_path / "run"), device="cpu")
    trained, history = train(model2, ds, None, tc, resume_from=str(tmp_path / "ckpt"))
    steps = [h["step"] for h in history if "main_loss" in h]
    assert steps == [4, 5]
    assert all(math.isfinite(h["main_loss"]) for h in history if "main_loss" in h)


def test_scheduler_fresh_applies_warmup_lr():
    model = full_width_model()
    opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    sched = WarmupCosineSchedule(opt, warmup_steps=10, total_steps=100)
    # step-0 LR is already the warmup LR, not the full base LR
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3 / 10)
    assert sched.current_step == 0


def test_scheduler_resume_applies_resumed_lr():
    model = full_width_model()
    opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    sched = WarmupCosineSchedule(opt, warmup_steps=10, total_steps=100)
    for _ in range(4):
        opt.step()
        sched.step()
    lr_after_4 = opt.param_groups[0]["lr"]
    assert lr_after_4 > 0

    # restore into a fresh optimizer/scheduler — LR must not snap back to base
    model2 = full_width_model()
    opt2 = build_optimizer(model2, learning_rate=1e-3, weight_decay=0.0)
    sched2 = WarmupCosineSchedule(opt2, warmup_steps=10, total_steps=100)
    base_lr = opt2.param_groups[0]["lr"]
    assert base_lr == pytest.approx(1e-3 / 10)  # warmup LR after fresh construction
    sched2.load_state_dict(sched.state_dict())
    assert sched2.current_step == 4
    # resume restores the exact LR the groups held at save time (lr_at(3))
    assert opt2.param_groups[0]["lr"] == pytest.approx(lr_after_4)
    assert opt2.param_groups[0]["lr"] == pytest.approx(sched.lr_at(3)[0])


def test_scheduler_base_lrs_length_mismatch_rejected():
    model = full_width_model()
    opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=10)
    state = sched.state_dict()
    state["base_lrs"] = [1e-3, 1e-3, 1e-3]  # optimizer has 2 groups (decay/no-decay)
    with pytest.raises(ValueError, match="base_lrs"):
        sched.load_state_dict(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_rng_restore_exact(tmp_path):
    model = full_width_model()
    opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=8)
    torch.cuda.manual_seed_all(123)
    a = torch.cuda.get_rng_state()
    save_training_checkpoint(model, opt, sched, tmp_path / "ckpt", global_step=0,
                             tokens_seen=0, tokenizer_fingerprint=None)
    # corrupt the RNG, then resume restores it exactly
    torch.cuda.manual_seed_all(999)
    resume_training_checkpoint(opt, sched, tmp_path / "ckpt")
    assert torch.equal(torch.cuda.get_rng_state().cpu(), a.cpu())
