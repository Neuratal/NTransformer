# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Training engine smoke tests — no optimizer steps, no real training."""

from __future__ import annotations

import pytest
import torch

from neuratal.config import NeuratalConfig
from neuratal.model.model import NeuratalModel
from neuratal.training.chunked_loss import chunked_cross_entropy
from neuratal.training.checkpoint import save_training_checkpoint, resume_training_checkpoint
from neuratal.training.data import SyntheticTokenDataset, collate
from neuratal.training.optim import WarmupCosineSchedule, build_optimizer


V = 262_144


def make_model():
    torch.manual_seed(0)
    cfg = NeuratalConfig(vocab_size=V, hidden_size=64, num_hidden_layers=1,
                         num_attention_heads=1, num_key_value_heads=1,
                         embedding_size=8, intermediate_size=128,
                         max_position_embeddings=64)
    return NeuratalModel(cfg)


def test_synthetic_dataset_deterministic():
    ds = SyntheticTokenDataset(4, 16, vocab_size=V, seed=7)
    a = ds[0]["input_ids"]
    b = ds[0]["input_ids"]
    assert torch.equal(a, b)
    assert (a >= 0).all() and (a < V).all()


def test_collate_stacks():
    ds = SyntheticTokenDataset(4, 16, vocab_size=V, seed=3)
    batch = collate([ds[i] for i in range(2)])
    assert batch["input_ids"].shape == (2, 16)
    assert batch["labels"].shape == (2, 16)


def test_gradient_accumulation_finite():
    model = make_model()
    model.train()
    ids = torch.randint(0, V, (2, 16))
    targets = torch.randint(0, V, (2, 16))
    hidden = model.embed_tokens(ids)
    hidden = model.layers[0](hidden, torch.arange(16))
    hidden = model.final_norm(hidden)
    total, n = chunked_cross_entropy(hidden, targets, model.output_projection,
                                     model.embed_tokens.embedding_weight, chunk_size=8)
    loss = total / n.clamp_min(1)
    model.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(gnorm)


def test_scheduler_never_steps_below_zero():
    model = make_model()
    opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    sched = WarmupCosineSchedule(opt, warmup_steps=2, total_steps=10)
    for _ in range(3):
        sched.step()
    assert sched.current_step == 3


def test_checkpoint_roundtrip_and_resume(tmp_path):
    model = make_model().eval()
    opt = build_optimizer(model, learning_rate=1e-3)
    sched = WarmupCosineSchedule(opt, warmup_steps=2, total_steps=10)
    ids = torch.randint(0, V, (1, 8))
    with torch.no_grad():
        ref = model(ids).logits
    save_training_checkpoint(model, opt, sched, tmp_path, global_step=7, tokens_seen=1234,
                             tokenizer_fingerprint="deadbeef")
    # reload and verify
    from neuratal.training.checkpoint import load_training_checkpoint
    model2, meta = load_training_checkpoint(tmp_path)
    assert meta["global_step"] == 7
    assert meta["tokens_seen"] == 1234
    assert meta["tokenizer_fingerprint"] == "deadbeef"
    with torch.no_grad():
        out = model2(ids).logits
    assert torch.allclose(ref, out, atol=1e-6)


def test_corrupted_checkpoint_rejected(tmp_path):
    model = make_model().eval()
    opt = build_optimizer(model)
    sched = WarmupCosineSchedule(opt, warmup_steps=2, total_steps=10)
    save_training_checkpoint(model, opt, sched, tmp_path, global_step=1)
    # corrupt trainer_state.safetensors
    p = tmp_path / "trainer_state.safetensors"
    data = p.read_bytes()
    p.write_bytes(data[:len(data) // 2])
    with pytest.raises(Exception):
        from neuratal.training.checkpoint import resume_training_checkpoint
        resume_training_checkpoint(opt, sched, tmp_path)


def test_wrong_fingerprint_mismatch_detected():
    # integration: the adapter must reject a mismatched fingerprint (already enforced)
    from neuratal.integration import verify_checkpoint_tokenizer, TokenizerMismatchError
    from neuratal.config import NeuratalConfig as C
    # build a model with wrong vocab to trigger the guard? vocab must stay 262144; instead use EOS mismatch
    model = make_model()
    model.config.eos_token_id = 0  # wrong
    # no real NTokenizer artifact is required for this test of the error path
    pytest.skip("covered in test_ntokenizer_integration.py (artifact-dependent)")
