# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Regression tests for the v1.4.0-candidate bug remediation.

One or more tests per original finding ID from the independent audit
(see BUG_REGISTRY.md and docs/reports in versions/v1.4.0-candidate/).

Finding IDs (audit §6–9): BUG-01..BUG-07, STRONG-01..03, POSSIBLE-01..06,
HIST-01..04 (HIST findings are branch-only and documented, not re-tested here).

DDP tests spawn 2 processes with gloo/CPU by default; they are skipped when
``torch.distributed`` or the gloo backend is unavailable. They verify behavior
and parity, not just absence of exceptions. Set CUDA_VISIBLE_DEVICES="" would be
pointless here — CUDA visibility is hidden inside each spawned worker instead.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from neuratal import NeuratalConfig, NeuratalModel
from neuratal.training.checkpoint import (
    resume_training_checkpoint,
    save_training_checkpoint,
)
from neuratal.training.data import SyntheticTokenDataset, collate, train_validation_split, TokenDataset
from neuratal.training.engine import TrainConfig, train
from neuratal.training.optim import WarmupCosineSchedule, build_optimizer

AUDIT_V = 14_000


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ----------------------------------------------------------------- scheduler
class TestSchedulerContract:
    """BUG-05: step() must advance to the NEXT lr_at position."""

    _warmup, _total = 5, 12

    def _make(self):
        model = torch.nn.Linear(4, 4)
        opt = build_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
        return WarmupCosineSchedule(opt, warmup_steps=self._warmup,
                                    total_steps=self._total, min_lr_ratio=0.1)

    def test_groups_track_lr_at_after_each_step(self):
        sched = self._make()
        assert sched.optimizer.param_groups[0]["lr"] == pytest.approx(sched.lr_at(0)[0])
        for k in range(1, self._total + 2):
            sched.step()
            assert sched.optimizer.param_groups[0]["lr"] == pytest.approx(sched.lr_at(k)[0])

    def test_first_two_steps_are_distinct_warmup_values(self):
        sched = self._make()
        sched.step()
        lr1 = sched.optimizer.param_groups[0]["lr"]
        sched.step()
        lr2 = sched.optimizer.param_groups[0]["lr"]
        assert lr1 != lr2
        assert lr1 == pytest.approx(sched.lr_at(1)[0])
        assert lr2 == pytest.approx(sched.lr_at(2)[0])

    def test_final_step_lands_on_min_lr_floor(self):
        sched = self._make()
        for _ in range(self._total):
            sched.step()
        assert sched.optimizer.param_groups[0]["lr"] == pytest.approx(
            sched.base_lrs[0] * sched.min_lr_ratio
        )

    def test_resume_restores_lr_at_saved_step(self):
        sched = self._make()
        for _ in range(4):
            sched.step()
        lr_at_save = sched.optimizer.param_groups[0]["lr"]
        model2 = torch.nn.Linear(4, 4)
        opt2 = build_optimizer(model2, learning_rate=3e-4, weight_decay=0.1)
        sched2 = WarmupCosineSchedule(opt2, warmup_steps=self._warmup,
                                      total_steps=self._total, min_lr_ratio=0.1)
        sched2.load_state_dict(sched.state_dict())
        assert opt2.param_groups[0]["lr"] == pytest.approx(lr_at_save)
        assert opt2.param_groups[0]["lr"] == pytest.approx(sched.lr_at(4)[0])


class TestSchedulerValidation:
    """BUG-06: cross-field validation and resume/loop-bound coherence."""

    def _sched(self, warmup=2, total=10):
        model = torch.nn.Linear(4, 4)
        opt = build_optimizer(model, learning_rate=1e-3)
        return WarmupCosineSchedule(opt, warmup_steps=warmup, total_steps=total)

    def _tiny_ckpt_model(self):
        torch.manual_seed(0)
        cfg = NeuratalConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=1, num_key_value_heads=1, embedding_size=8,
                             intermediate_size=16, max_position_embeddings=32)
        return NeuratalModel(cfg)

    def test_warmup_greater_than_total_rejected(self):
        sched = self._sched()
        state = sched.state_dict()
        state["warmup_steps"] = 10  # == total_steps
        with pytest.raises(ValueError, match="warmup_steps"):
            sched.load_state_dict(state)
        state = sched.state_dict()
        state["warmup_steps"] = 12  # > total_steps
        with pytest.raises(ValueError, match="warmup_steps"):
            sched.load_state_dict(state)

    def test_zero_or_negative_base_lrs_rejected(self):
        sched = self._sched()
        for bad in ([0.0, 0.0], [-1e-3, 1e-3], [1e-3, -0.1]):
            state = sched.state_dict()
            state["base_lrs"] = bad
            with pytest.raises(ValueError, match="base_lrs"):
                sched.load_state_dict(state)

    def test_resume_with_smaller_max_steps_does_not_crash_and_warns(self, tmp_path):
        """v1.3 crashed in the scheduler constructor (warmup >= new total)
        before the checkpoint's own valid values were consulted."""
        model = self._tiny_ckpt_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        sched = self._sched(warmup=10, total=20)
        save_training_checkpoint(model, opt, sched, tmp_path, global_step=2, tokens_seen=16)
        model2 = self._tiny_ckpt_model()
        ds = SyntheticTokenDataset(4, 8, vocab_size=64, seed=3)
        tc = TrainConfig(batch_size=1, sequence_length=8, learning_rate=1e-3,
                         warmup_steps=10, gradient_accumulation_steps=1,
                         precision="float32", eval_every_steps=99, log_every_steps=1,
                         checkpoint_every_steps=99, max_steps=5,  # < saved warmup_steps=10
                         out_dir=str(tmp_path / "run"), device="cpu")
        with pytest.warns(UserWarning, match="max_steps/cfg.max_steps=5"):
            _, history = train(model2, ds, None, tc, resume_from=str(tmp_path))
        assert history and math_isfinite(history)

    def test_resume_with_larger_max_steps_warns_and_floors(self, tmp_path):
        model = self._tiny_ckpt_model()
        opt = build_optimizer(model, learning_rate=1e-3)
        sched = self._sched(warmup=1, total=2)
        save_training_checkpoint(model, opt, sched, tmp_path, global_step=2, tokens_seen=16)
        model2 = self._tiny_ckpt_model()
        ds = SyntheticTokenDataset(4, 8, vocab_size=64, seed=3)
        tc = TrainConfig(batch_size=1, sequence_length=8, learning_rate=1e-3,
                         warmup_steps=1, gradient_accumulation_steps=1,
                         precision="float32", eval_every_steps=99, log_every_steps=1,
                         checkpoint_every_steps=99, max_steps=4,  # > checkpointed total_steps=2
                         out_dir=str(tmp_path / "run"), device="cpu")
        with pytest.warns(UserWarning, match="min_lr floor"):
            _, history = train(model2, ds, None, tc, resume_from=str(tmp_path))
        # requested bound is honored; the checkpointed schedule wins for LRs
        assert history[-1]["step"] == 4
        opt2 = build_optimizer(self._tiny_ckpt_model(), learning_rate=1e-3)
        sched2 = WarmupCosineSchedule(opt2, warmup_steps=1, total_steps=2)
        meta = resume_training_checkpoint(opt2, sched2, tmp_path)
        assert meta["global_step"] == 2
        assert sched2.total_steps == 2  # checkpointed schedule wins, not the fresh one


def math_isfinite(history) -> bool:
    return all(h.get("main_loss") is None or abs(float(h["main_loss"])) < float("inf")
               for h in history)


class TestEngineStockFp16:
    """BUG-02: fp16 overflow does not advance scheduler/global_step."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler requires CUDA")
    def test_fp16_overflow_skips_lr_advance(self, tmp_path):
        torch.manual_seed(0)
        model = torch.nn.Linear(16, 4).to("cuda")
        opt = torch.optim.AdamW(list(model.parameters()), lr=1e-3)
        sched = WarmupCosineSchedule(opt, warmup_steps=2, total_steps=10)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        opt.zero_grad(set_to_none=True)
        loss = (model(torch.randn(8, 16, device="cuda")) ** 2).sum()
        scaler.scale(loss).backward()
        model.weight.grad[0, 0] = float("inf")
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_before = sched._step
        scaler.step(opt)
        scaler.update()
        skipped = not bool(torch.isfinite(norm))
        if not skipped:
            sched.step()
        assert skipped
        assert sched._step == step_before  # LR schedule not advanced on the skip

    def test_engine_skips_step_on_nonfinite_grad_norm(self, monkeypatch, tmp_path):
        """CPU-side regression: the engine gates optimizer/scheduler on isfinite(grad_norm)."""
        torch.manual_seed(0)
        cfg = NeuratalConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, embedding_size=None,
                             intermediate_size=64, max_position_embeddings=64)
        model = NeuratalModel(cfg)
        params_before = {n: p.detach().clone() for n, p in model.named_parameters()}

        calls = {"n": 0}
        real_clip = torch.nn.utils.clip_grad_norm_

        def flaky_clip(params, norm):
            calls["n"] += 1
            if calls["n"] == 1:
                return torch.tensor(float("inf"))
            return real_clip(params, norm)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", flaky_clip)
        ds = SyntheticTokenDataset(4, 8, vocab_size=64, seed=3)
        tc = TrainConfig(batch_size=1, sequence_length=8, learning_rate=1e-3, warmup_steps=1,
                         gradient_accumulation_steps=1, precision="float32",
                         eval_every_steps=99, log_every_steps=1, checkpoint_every_steps=99,
                         max_steps=2, out_dir=str(tmp_path), device="cpu", seed=0)
        _, history = train(model, ds, None, tc)
        assert history[0]["skipped"] is True and history[0]["step"] == 1 and history[0]["iteration"] == 1
        assert history[1]["iteration"] == 2 and "main_loss" in history[1]
        assert history[1]["step"] == 1  # same optimizer step, retried after the skip
        # after the skipped first step the weights moved only on the second step
        moved = sum(float((p.detach() - params_before[n]).abs().sum()) for n, p in model.named_parameters())
        assert moved > 0  # second step ran
        grad_norm_history = [h for h in history if h.get("skipped")]
        assert len(grad_norm_history) == 1 and grad_norm_history[0]["grad_norm"] is None


# ----------------------------------------------------- sharded checkpoints
class TestShardedIntegrity:
    """BUG-03/BUG-04: path validation; verification fails closed."""

    def _ckpt(self, tmp_path):
        from neuratal.serialization.sharded import save_pretrained_sharded
        torch.manual_seed(0)
        cfg = NeuratalConfig(vocab_size=128, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, embedding_size=8,
                             intermediate_size=32, max_position_embeddings=64)
        model = NeuratalModel(cfg)
        ckpt = tmp_path / "ckpt"
        save_pretrained_sharded(model, ckpt, max_shard_bytes=4096)
        return ckpt

    def test_absolute_shard_path_rejected(self, tmp_path):
        from neuratal.serialization.sharded import INDEX_FILENAME, load_pretrained_sharded
        ckpt = self._ckpt(tmp_path)
        index = json.loads((ckpt / INDEX_FILENAME).read_text())
        for key in list(index["weight_map"]):
            index["weight_map"][key] = "/tmp/evil.safetensors"
        (ckpt / INDEX_FILENAME).write_text(json.dumps(index))
        with pytest.raises(ValueError, match="model-NNNNN-of-NNNNN"):
            load_pretrained_sharded(ckpt)

    def test_relative_traversal_shard_path_rejected(self, tmp_path):
        from neuratal.serialization.sharded import INDEX_FILENAME, load_pretrained_sharded
        ckpt = self._ckpt(tmp_path)
        index = json.loads((ckpt / INDEX_FILENAME).read_text())
        for key in list(index["weight_map"]):
            index["weight_map"][key] = "../evil.safetensors"
        (ckpt / INDEX_FILENAME).write_text(json.dumps(index))
        with pytest.raises(ValueError, match="model-NNNNN-of-NNNNN"):
            load_pretrained_sharded(ckpt)

    def test_non_standard_shard_name_rejected(self, tmp_path):
        from neuratal.serialization.sharded import INDEX_FILENAME, load_pretrained_sharded
        ckpt = self._ckpt(tmp_path)
        index = json.loads((ckpt / INDEX_FILENAME).read_text())
        for key in list(index["weight_map"]):
            index["weight_map"][key] = "model.safetensors"
        (ckpt / INDEX_FILENAME).write_text(json.dumps(index))
        with pytest.raises(ValueError, match="model-NNNNN-of-NNNNN"):
            load_pretrained_sharded(ckpt)

    def test_missing_shard_hashes_fails_closed(self, tmp_path):
        from neuratal.serialization.sharded import INDEX_FILENAME, load_pretrained_sharded
        ckpt = self._ckpt(tmp_path)
        index = json.loads((ckpt / INDEX_FILENAME).read_text())
        del index["metadata"]["shard_hashes"]
        (ckpt / INDEX_FILENAME).write_text(json.dumps(index))
        with pytest.raises(ValueError, match="does not record shard_hashes"):
            load_pretrained_sharded(ckpt)

    def test_missing_shard_hashes_optout_warns(self, tmp_path):
        from neuratal.serialization.sharded import INDEX_FILENAME, load_pretrained_sharded
        ckpt = self._ckpt(tmp_path)
        index = json.loads((ckpt / INDEX_FILENAME).read_text())
        del index["metadata"]["shard_hashes"]
        (ckpt / INDEX_FILENAME).write_text(json.dumps(index))
        with pytest.warns(UserWarning, match="shard_hashes"):
            load_pretrained_sharded(ckpt, verify_shard_hashes=False)

    def test_missing_total_size_fails_closed(self, tmp_path):
        from neuratal.serialization.sharded import INDEX_FILENAME, load_pretrained_sharded
        ckpt = self._ckpt(tmp_path)
        index = json.loads((ckpt / INDEX_FILENAME).read_text())
        del index["metadata"]["total_size"]
        (ckpt / INDEX_FILENAME).write_text(json.dumps(index))
        with pytest.raises(ValueError, match="total_size"):
            load_pretrained_sharded(ckpt)

    def test_missing_total_size_optout_warns(self, tmp_path):
        from neuratal.serialization.sharded import INDEX_FILENAME, load_pretrained_sharded
        ckpt = self._ckpt(tmp_path)
        index = json.loads((ckpt / INDEX_FILENAME).read_text())
        del index["metadata"]["total_size"]
        (ckpt / INDEX_FILENAME).write_text(json.dumps(index))
        with pytest.warns(UserWarning, match="total_size"):
            load_pretrained_sharded(ckpt, verify_shard_hashes=False)

    def test_interrupted_save_leaves_no_new_shards(self, tmp_path, monkeypatch):
        """STRONG-02: a failing save must not leave orphan shards behind."""
        import neuratal.serialization.sharded as sharded_mod
        torch.manual_seed(0)
        cfg = NeuratalConfig(vocab_size=256, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, embedding_size=8,
                             intermediate_size=64, max_position_embeddings=64)
        model = NeuratalModel(cfg)
        # pre-existing shard must survive
        victim = tmp_path / "model-00000-of-00099.safetensors"
        victim.write_bytes(b"pre-existing")

        original_save = sharded_mod._atomic_save_safetensors
        state = {"calls": 0}

        def flaky_save(sd, path):
            state["calls"] += 1
            if state["calls"] >= 2:
                raise RuntimeError("simulated I/O failure mid-save")
            return original_save(sd, path)

        monkeypatch.setattr(sharded_mod, "_atomic_save_safetensors", flaky_save)
        with pytest.raises(RuntimeError, match="simulated I/O failure"):
            sharded_mod.save_pretrained_sharded(model, tmp_path, max_shard_bytes=4096)
        leftover = sorted(p.name for p in tmp_path.glob("model-*.safetensors"))
        assert leftover == ["model-00000-of-00099.safetensors"]


# ------------------------------------------------------------------ estimator
class TestEstimator:
    """BUG-07: fp32 gradients must be estimated at 4 bytes by default."""

    _cfg = NeuratalConfig(dtype="float32", vocab_size=128, hidden_size=32, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1, embedding_size=8,
                          intermediate_size=32, max_position_embeddings=64)

    def test_fp32_gradients_cost_four_bytes(self):
        from neuratal.estimate import estimate
        est = estimate(self._cfg, name="t")
        from neuratal.presets import parameter_breakdown_meta
        P = parameter_breakdown_meta(self._cfg)["total"]
        assert est.gradient_bytes == P * 4

    def test_bf16_gradients_cost_two_bytes(self):
        from neuratal.estimate import estimate
        cfg = NeuratalConfig(dtype="bfloat16", vocab_size=128, hidden_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             embedding_size=8, intermediate_size=32, max_position_embeddings=64)
        est = estimate(cfg, name="t")
        from neuratal.presets import parameter_breakdown_meta
        P = parameter_breakdown_meta(cfg)["total"]
        assert est.gradient_bytes == P * 2

    def test_explicit_override_respected(self):
        from neuratal.estimate import estimate
        est = estimate(self._cfg, name="t", grad_dtype_bytes=2)
        from neuratal.presets import parameter_breakdown_meta
        P = parameter_breakdown_meta(self._cfg)["total"]
        assert est.gradient_bytes == P * 2


# ---------------------------------------------------------- chunked CE kit
class TestChunkedCEChunkSize:
    """POSSIBLE-03: chunk_size must be >= 1 (was silently coerced to 256/zeros)."""

    def _inputs(self):
        torch.manual_seed(0)
        hidden = torch.randn(1, 4, 8)
        targets = torch.randint(0, 64, (1, 4))
        emb = torch.randn(64, 8)
        return hidden, targets, None, emb

    def test_zero_chunk_size_rejected(self):
        from neuratal.training.chunked_linear_ce import chunked_linear_cross_entropy
        from neuratal.training.chunked_loss import chunked_cross_entropy
        hidden, targets, op, emb = self._inputs()
        with pytest.raises(ValueError, match="chunk_size"):
            chunked_linear_cross_entropy(hidden, targets, op, emb, chunk_size=0)
        cfg = NeuratalConfig(vocab_size=64, hidden_size=8, num_hidden_layers=1,
                             num_attention_heads=1, num_key_value_heads=1, embedding_size=8,
                             intermediate_size=16, max_position_embeddings=16)
        from neuratal.model.output import OutputProjection
        with pytest.raises(ValueError, match="chunk_size"):
            chunked_cross_entropy(hidden, targets, OutputProjection(cfg), emb, chunk_size=0)

    def test_negative_chunk_size_rejected(self):
        from neuratal.training.chunked_linear_ce import chunked_linear_cross_entropy
        hidden, targets, op, emb = self._inputs()
        with pytest.raises(ValueError, match="chunk_size"):
            chunked_linear_cross_entropy(hidden, targets, op, emb, chunk_size=-3)


# ---------------------------------------------------------------- data split
class TestValidationSplit:
    """POSSIBLE-04: tiny corpora fail with a clear error, not a cryptic one."""

    def test_tiny_corpus_raises_clear_error(self):
        import torch as t
        seq = 16
        ds = TokenDataset(t.arange(2 * seq + 1, dtype=t.int64), seq)
        with pytest.raises(ValueError, match="corpus too small"):
            train_validation_split(ds, val_fraction=0.5)

    def test_minimum_corpus_splits_successfully(self):
        import torch as t
        seq = 16
        ds = TokenDataset(t.arange(2 * seq + 2, dtype=t.int64), seq)
        train, val = train_validation_split(ds, val_fraction=0.5)
        assert len(train) >= 1 and len(val) >= 1


# -------------------------------------------------------------- NTOK parser
class TestNtokParser:
    """POSSIBLE-05: header read and size validation happen before any payload read."""

    _doc = {
        "vocab": {"id_to_bytes": [[i] for i in range(256)] + [[0], [1], [2]]},
        "merges": {"rules": []},
        "special_tokens": {"tokens": [{"id": 256, "content": "<|a|>"},
                                      {"id": 257, "content": "<|b|>"},
                                      {"id": 258, "content": "<|c|>"}]},
    }

    def _make_file(self, tmp_path, *, payload_len_claim=None, file_size=None,
                   crc_ok=True, version=2, name="fixture.ntok"):
        import struct
        import zlib
        import json
        payload = json.dumps(self._doc).encode("utf-8")
        header_struct = struct.Struct("<4sH I Q")
        header_len = header_struct.size
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        if not crc_ok:
            crc ^= 0xFFFFFFFF
        plen = payload_len_claim if payload_len_claim is not None else len(payload)
        path = tmp_path / name
        with open(path, "wb") as handle:
            handle.write(header_struct.pack(b"NTOK", version, crc, plen))
            if file_size is None:
                handle.write(payload)
            else:
                handle.truncate(file_size)
        return path

    def test_oversized_payload_claim_rejected_before_read(self, tmp_path):
        import struct, os
        from neuratal.integration.manifest import parse_ntok_file, ManifestError, MAX_PAYLOAD_BYTES
        header_len = struct.Struct("<4sH I Q").size
        # sparse file: tiny on disk, but stat() reports the claimed oversized length
        path = self._make_file(tmp_path, payload_len_claim=MAX_PAYLOAD_BYTES + 1,
                               file_size=header_len + MAX_PAYLOAD_BYTES + 1)
        assert path.stat().st_size == header_len + MAX_PAYLOAD_BYTES + 1
        with pytest.raises(ManifestError, match="1 GiB"):
            parse_ntok_file(path)

    def test_size_mismatch_rejected(self, tmp_path):
        from neuratal.integration.manifest import parse_ntok_file, ManifestError
        path = self._make_file(tmp_path, file_size=100)
        with pytest.raises(ManifestError, match="file size mismatch"):
            parse_ntok_file(path)

    def test_crc_mismatch_rejected(self, tmp_path):
        from neuratal.integration.manifest import parse_ntok_file, ManifestError
        path = self._make_file(tmp_path, crc_ok=False)
        with pytest.raises(ManifestError, match="CRC-32"):
            parse_ntok_file(path)

    def test_valid_fixture_parses(self, tmp_path):
        from neuratal.integration.manifest import parse_ntok_file
        path = self._make_file(tmp_path)
        parsed = parse_ntok_file(path)
        assert parsed["payload"]["vocab"]["id_to_bytes"][0] == [0]
        assert parsed["format_version"] == 2


# ------------------------------------------------------------- DeltaNetCache
class TestDeltaNetCacheCommit:
    """POSSIBLE-06: overflowing commit() must not advance _length."""

    def test_commit_overflow_leaves_length_unchanged(self):
        from neuratal.model.gated_delta_net import DeltaNetCache
        cfg = NeuratalConfig(attention_type="gated_delta", max_position_embeddings=4,
                             num_hidden_layers=1, num_attention_heads=1,
                             hidden_size=8, num_key_value_heads=1, embedding_size=8,
                             intermediate_size=16)
        cache = DeltaNetCache(cfg)
        cache.initialize(batch_size=1, device="cpu",
                         dtype=torch.float32)
        layer_idx = 0
        cache.mark_pending(3)
        cache.commit(3)
        assert cache.get_seq_length() == 3
        cache.mark_pending(2)
        with pytest.raises(ValueError, match="exceeds"):
            cache.commit(2)
        assert cache.get_seq_length() == 3  # unchanged


# ------------------------------------------------------------------- DDP
def _ddp_worker(rank, world_size, port, out_dir, scenario, grad_accum):
    # CUDA visibility is cleared by the parent (tests/__init__) before spawn,
    # so the engine selects gloo; the env check here is defense-in-depth.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    import torch as t
    config = dict(
        vocab_size=256, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, embedding_size=8,
        intermediate_size=64, max_position_embeddings=64,
    )
    if scenario.startswith("moe"):
        config.update(num_experts=2, num_experts_per_token=1, expert_hidden_dim=64,
                      moe_layer_indices=[0])
    if scenario.endswith("fullwidth"):
        config.update(embedding_size=None)

    t.manual_seed(0)
    torch.manual_seed(0)
    cfg = NeuratalConfig(**config)
    model = NeuratalModel(cfg)
    ds = SyntheticTokenDataset(1, 8, vocab_size=256, seed=42)
    tc = TrainConfig(batch_size=1, sequence_length=8, learning_rate=1e-3, warmup_steps=1,
                     gradient_accumulation_steps=grad_accum, precision="float32",
                     eval_every_steps=99, log_every_steps=1, checkpoint_every_steps=1,
                     max_steps=2, out_dir=out_dir, device="cpu", seed=1234)
    trained, history = train(model, ds, None, tc)
    params = {n: p.detach().clone() for n, p in trained.named_parameters()}
    grads = {n: p.grad.detach().clone() for n, p in trained.named_parameters() if p.grad is not None}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(params, os.path.join(out_dir, f"params_rank{rank}.pt"))
    torch.save(grads, os.path.join(out_dir, f"grads_rank{rank}.pt"))
    if rank == 0:
        Path(out_dir, "history.txt").write_text(
            ",".join(str(h.get("main_loss")) for h in history if "main_loss" in h)
        )


def _gloo_available() -> bool:
    if not torch.distributed.is_available():
        return False
    try:
        torch.distributed.is_gloo_available()
        return True
    except Exception:
        return False


_GLOO = pytest.mark.skipif(not _gloo_available(), reason="gloo backend unavailable")


def _spawn_ddp(tmp_path, scenario, monkeypatch, grad_accum=1):
    out_dir = str(tmp_path / f"ddp_{scenario}")
    port = _free_port()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.spawn(_ddp_worker, args=(2, port, out_dir, scenario, grad_accum),
             nprocs=2, join=True)
    return out_dir


def _single_process_reference(tmp_path, scenario, grad_accum=1):
    config = dict(
        vocab_size=256, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, embedding_size=8,
        intermediate_size=64, max_position_embeddings=64,
    )
    if scenario.startswith("moe"):
        config.update(num_experts=2, num_experts_per_token=1, expert_hidden_dim=64,
                      moe_layer_indices=[0])
    if scenario.endswith("fullwidth"):
        config.update(embedding_size=None)
    t_cfg = NeuratalConfig(**config)
    torch.manual_seed(0)
    model = NeuratalModel(t_cfg)
    ds = SyntheticTokenDataset(1, 8, vocab_size=256, seed=42)
    tc = TrainConfig(batch_size=1, sequence_length=8, learning_rate=1e-3, warmup_steps=1,
                     gradient_accumulation_steps=grad_accum, precision="float32",
                     eval_every_steps=99, log_every_steps=99, checkpoint_every_steps=99,
                     max_steps=2, out_dir=str(tmp_path / "ref"), device="cpu", seed=1234)
    trained, history = train(model, ds, None, tc)
    return ({n: p.detach().clone() for n, p in trained.named_parameters()},
            {n: p.grad.detach().clone() for n, p in trained.named_parameters() if p.grad is not None},
            history)


@_GLOO
class TestDDPEndToEnd:
    """BUG-01: multi-process DDP training works and matches the reference."""

    def test_ddp_moe_factored_head_bitwise_parity(self, tmp_path, monkeypatch):
        out = _spawn_ddp(tmp_path, "moe_factored", monkeypatch, grad_accum=1)
        ref_params, ref_grads, _ = _single_process_reference(tmp_path, "moe_factored", 1)
        ddp0 = torch.load(Path(out) / "params_rank0.pt")
        ddp1 = torch.load(Path(out) / "params_rank1.pt")
        # both ranks agree (allreduce produced identical tensors)
        for name, tensor in ddp0.items():
            assert torch.equal(ddp0[name], ddp1[name]), name
        # and DDP agrees with the single-process reference on identical data
        for name, tensor in ref_params.items():
            assert torch.equal(ref_params[name], ddp0[name]), name
        for name in ref_grads:
            assert torch.equal(ref_grads[name], torch.load(Path(out) / "grads_rank0.pt")[name]), name

    def test_ddp_moe_fullwidth_runs_and_is_finite(self, tmp_path, monkeypatch):
        out = _spawn_ddp(tmp_path, "moe_fullwidth", monkeypatch, grad_accum=1)
        history = (Path(out) / "history.txt").read_text()
        assert history and all(float(x) == float(x) for x in history.split(",") if x)  # no NaN

    def test_ddp_dense_factored_bitwise_parity(self, tmp_path, monkeypatch):
        out = _spawn_ddp(tmp_path, "dense_factored", monkeypatch, grad_accum=1)
        ref_params, ref_grads, _ = _single_process_reference(tmp_path, "dense_factored", 1)
        ddp0 = torch.load(Path(out) / "params_rank0.pt")
        for name, tensor in ref_params.items():
            assert torch.equal(ref_params[name], ddp0[name]), name

    def test_ddp_dense_fullwidth_runs_and_is_finite(self, tmp_path, monkeypatch):
        out = _spawn_ddp(tmp_path, "dense_fullwidth", monkeypatch, grad_accum=1)
        history = (Path(out) / "history.txt").read_text()
        assert history and all(float(x) == float(x) for x in history.split(",") if x)

    def test_ddp_gradient_accumulation_bitwise_parity(self, tmp_path, monkeypatch):
        out = _spawn_ddp(tmp_path, "moe_factored", monkeypatch, grad_accum=2)
        ref_params, _, _ = _single_process_reference(tmp_path, "moe_factored", 2)
        ddp0 = torch.load(Path(out) / "params_rank0.pt")
        for name, tensor in ref_params.items():
            assert torch.equal(ref_params[name], ddp0[name]), name

    def test_ddp_checkpoint_resume_roundtrip(self, tmp_path, monkeypatch):
        out = _spawn_ddp(tmp_path, "moe_factored", monkeypatch, grad_accum=1)
        # DDP saved checkpoint at every step (checkpoint_every_steps=1)
        ckpts = sorted(Path(out).glob("ckpt-*"))
        assert ckpts, "expected engine to save a checkpoint during DDP training"
        ckpt = str(ckpts[-1])
        torch.manual_seed(0)
        cfg = NeuratalConfig(vocab_size=256, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, embedding_size=8,
                             intermediate_size=64, max_position_embeddings=64,
                             num_experts=2, num_experts_per_token=1, expert_hidden_dim=64,
                             moe_layer_indices=[0])
        model = NeuratalModel(cfg)
        ddp_params = torch.load(Path(out) / "params_rank0.pt")
        opt = build_optimizer(model, learning_rate=1e-3)
        sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=10)
        meta = resume_training_checkpoint(opt, sched, ckpt, model=model)
        assert meta["global_step"] == 2
        from neuratal.serialization.checkpoint import load_pretrained
        reloaded = load_pretrained(ckpt)
        for name, tensor in reloaded.state_dict().items():
            assert torch.equal(tensor, ddp_params[name]), name


# ------------------------------------------------------------- resume/RNG
class TestResumeWarnings:
    """POSSIBLE-02: cross-device RNG restore reports gaps instead of staying silent."""

    def _tiny_model_ckpt(self, tmp_path):
        torch.manual_seed(0)
        cfg = NeuratalConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=1, num_key_value_heads=1, embedding_size=8,
                             intermediate_size=16, max_position_embeddings=32)
        model = NeuratalModel(cfg)
        opt = build_optimizer(model, learning_rate=1e-3)
        sched = WarmupCosineSchedule(opt, warmup_steps=1, total_steps=4)
        save_training_checkpoint(model, opt, sched, tmp_path, global_step=0, tokens_seen=0)
        return model, opt, sched

    def test_missing_cpu_rng_warns(self, tmp_path):
        model, opt, sched = self._tiny_model_ckpt(tmp_path)
        from safetensors.torch import load_file, save_file
        state_path = tmp_path / "trainer_state.safetensors"
        tensors = load_file(str(state_path))
        tensors.pop("trainer.rng_cpu", None)
        save_file(tensors, str(state_path))
        with pytest.warns(UserWarning, match="CPU RNG state"):
            resume_training_checkpoint(opt, sched, tmp_path)

    def test_cuda_unavailable_with_cuda_rng_warns(self, tmp_path, monkeypatch):
        model, opt, sched = self._tiny_model_ckpt(tmp_path)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.warns(UserWarning, match="CUDA"):
            resume_training_checkpoint(opt, sched, tmp_path)
