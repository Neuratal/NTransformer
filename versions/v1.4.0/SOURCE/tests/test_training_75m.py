# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""75M dense training-chain validation on the project's 8 GB GPU.

Synthetic data only. This is a plumbing/optimizer validation, not pretraining.
"""

from __future__ import annotations

import math

import pytest
import torch

from neuratal import NeuratalModel, get_preset
from neuratal.training.checkpoint import load_training_checkpoint, resume_training_checkpoint
from neuratal.training.data import SyntheticTokenDataset
from neuratal.training.engine import TrainConfig, train
from neuratal.training.optim import WarmupCosineSchedule, build_optimizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="75M validation requires CUDA")
def test_75m_dense_forward_backward_optimizer_checkpoint_resume(tmp_path):
    preset = get_preset("75m")
    assert preset.parameter_count == 77_088_896
    torch.manual_seed(0)
    model = NeuratalModel(preset.config).to(torch.bfloat16)
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    train_ds = SyntheticTokenDataset(16, 32, vocab_size=262_144, seed=1)
    val_ds = SyntheticTokenDataset(4, 32, vocab_size=262_144, seed=2)
    tc = TrainConfig(
        batch_size=1, sequence_length=32, learning_rate=1e-4, warmup_steps=1,
        grad_clip_norm=1.0, gradient_accumulation_steps=2, precision="bfloat16",
        eval_every_steps=2, log_every_steps=1, checkpoint_every_steps=2,
        max_steps=2, out_dir=str(tmp_path), device="cuda",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with pytest.warns(UserWarning, match="dtype"):
        trained, history = train(model, train_ds, val_ds, tc)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    logs = [h for h in history if "main_loss" in h]
    assert logs and all(math.isfinite(h["main_loss"]) for h in logs)
    assert all(math.isfinite(h["grad_norm"]) for h in logs)
    assert any("val_loss" in h for h in history)
    assert peak_gib < 7.5, f"unexpected peak VRAM: {peak_gib:.2f} GiB"
    assert all(torch.isfinite(p).all() for p in trained.parameters())
    changed = [n for n, p in trained.named_parameters()
               if not torch.equal(p.detach().cpu(), before[n])]
    assert changed
    assert all(torch.isfinite(p.grad).all() for p in trained.parameters() if p.grad is not None)

    ckpt = tmp_path / "ckpt-2"
    loaded, meta = load_training_checkpoint(ckpt)
    assert meta["architecture"] == "dense"
    assert meta["global_step"] == 2
    assert meta["precision"] == "bfloat16"
    assert loaded.config.dtype == "bfloat16"

    opt = build_optimizer(loaded, learning_rate=1e-4)
    sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=2)
    resumed = resume_training_checkpoint(opt, sched, ckpt, model=loaded)
    assert resumed["global_step"] == 2
    assert sched.current_step == 2
    for name, p in loaded.named_parameters():
        state = opt.state.get(p)
        if state is not None:
            assert torch.isfinite(state["exp_avg"]).all(), name
            assert torch.isfinite(state["exp_avg_sq"]).all(), name
    assert peak_gib < 7.5
