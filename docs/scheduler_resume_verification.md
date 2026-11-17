# Scheduler Resume Verification — v1.2.0-candidate → v1.3.0-candidate

Goal: prove, or refute, that an interrupted+resumed training run is
bit-for-bit identical to an uninterrupted run, at multiple checkpoint points,
with differential per-group learning rates.

## Setup

- Model: tiny NeuratalModel (V=512, D=16, L=1, H=2, KV=2, F=4, I=32, ctx=32),
  `torch.manual_seed(42)` before construction.
- Data: `SyntheticTokenDataset(num_sequences=64, sequence_length=8, seed=99)`,
  `DataLoader(..., shuffle=False)`, 32 fixed batches; same order for both paths.
- Optimizer: AdamW, **two parameter groups with different LRs**
  (≥2-D: 2e-3, <2-D: 1e-3), `betas=(0.9, 0.95)`, `weight_decay=0.1`.
- Scheduler: `WarmupCosineSchedule(warmup_steps=4, total_steps=32,
  min_lr_ratio=0.1)`.
- Machine: CPU, `torch.set_num_threads(1)` (pinned threads → deterministic
  CPU kernels); CUDA nondeterminism is irrelevant to this experiment.
- Script: `/tmp/opencode/scheduler_equivalence.py` (reproduced in §6).

## Method

For each checkpoint step K in {0, 1, 4, 5, 8, 15, 16}:
1. Run **uninterrupted** for N=16 steps; record per-step LRs, parameters,
   optimizer state, losses, scheduler counter.
2. Run to K steps; **save checkpoint**; destroy objects; rebuild model from
   the checkpoint (`load_training_checkpoint`), fresh optimizer/scheduler,
   resume state (`resume_training_checkpoint`); continue the same batches;
   record the same quantities.
3. Compare: LRs per step, final parameters (torch.equal), optimizer moments
   and step counters (torch.equal), scheduler `_step`, losses for the resumed
   segment.

## Results (v1.3.0-candidate)

| K | LRs identical | params bitwise equal | optimizer state equal | losses identical | final scheduler step |
|---:|:---:|:---:|:---:|:---:|---:|
| 0 | ✓ | ✓ | ✓ | ✓ | 16 |
| 1 | ✓ | ✓ | ✓ | ✓ | 16 |
| 4 (end of warmup) | ✓ | ✓ | ✓ | ✓ | 16 |
| 5 (early cosine) | ✓ | ✓ | ✓ | ✓ | 16 |
| 8 (mid cosine) | ✓ | ✓ | ✓ | ✓ | 16 |
| 15 (late cosine) | ✓ | ✓ | ✓ | ✓ | 16 |
| 16 (final step) | ✓ | ✓ | ✓ | ✓ | 16 |

LR trajectory (uninterrupted == interrupted@8):
`[0.0005, 0.0005, 0.001, 0.0015, 0.002, 0.002,
  0.001994, 0.001977, 0.001949, 0.001911, 0.001862, 0.001804,
  0.001736, 0.001661, 0.001579, 0.00149]`

## What this audit found (beyond the previous report)

The v1.2.0-candidate fix (apply `lr_at(step-1)` on load) made the LR
trajectory correct in semantics but **not bit-exact**: scheduler base LRs were
stored as **float32** (`torch.tensor(float(x))` default dtype), so after a
resume the entire cosine decay drifted by ~1e-7 relative to the uninterrupted
run — visible in the first experiment run, invisible to per-run callers.

Fix (v1.3.0-candidate, `neuratal/training/checkpoint.py`): scheduler and
optimizer scalar state are stored with exact dtypes — ints as `int64`,
floats as **float64**. Round-trip is now bit-exact (all 7 checkpoints show
`lrs equal: True`, not just "close").

Second hardening: `WarmupCosineSchedule.load_state_dict` now validates
required keys (`step`, `warmup_steps`, `total_steps`, `min_lr_ratio`,
`base_lrs`), checks integer/finite/range constraints, and enforces
`base_lrs` length == optimizer param-group count. Malformed checkpoints
raise `ValueError` with precise messages (previously: `KeyError` or silent
misassignment).

## Definition of the scheduler step counter (documented semantics)

- `scheduler.step()` is called **once per optimizer step, after it** (engine
  contract; never per micro-step). It applies `lr_at(current_step)` to the
  param groups and increments the counter.
- `state_dict()["step"]` is therefore the number of `step()` calls made —
  equal to the number of completed scheduler applications, and to the number
  of engine steps run. At save time (end of engine step K), `step == K`.
- `load_state_dict` reapplies the LR the param groups held at save time,
  `lr_at(step - 1)` (with `step == 0` → `lr_at(0)`). This is exactly what the
  groups held in a run started from scratch with v1.2+ semantics, so a
  resumed run's first optimizer step uses precisely the LR it would have
  used uninterrupted.
- `step > total_steps` is legal; the schedule sits at `min_lr_ratio * base`
  (cosine clamped) — documented as terminal steady state, not an error.

## Exact commands

```bash
cd ../NTransformer
.venv/bin/python /tmp/opencode/scheduler_equivalence.py   # full table above
.venv/bin/python -m pytest tests/test_engine_regressions.py -q
```

## Not verified

- Multi-GPU DDP resume equivalence (no multi-GPU on this machine).
- Resume with `shuffle=True` data replay (documented as re-permuted).
- GradScaler state bit-equivalence (fp16 path) — functional restore verified,
  bit-exactness not claimed.
