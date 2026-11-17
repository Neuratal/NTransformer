# Bug Registry — NTransformer (v1.1.0rc1 audit → v1.2.0-candidate)

Statuses: FIXED (with commit), VERIFIED-NOT-A-BUG, OPEN (documented).
Severity: critical · high · medium · low. Every entry cites its evidence.

## B-2022-1 — Post-resume LR silently reverts to construction LR
- **Severity:** high (training-correctness) · **Status:** FIXED (`27c7f78`)
- **File:** `neuratal/training/optim.py` — `WarmupCosineSchedule.load_state_dict`
- **Defect:** Restoring scheduler state set `step/warmup/total/base_lrs` but
  never wrote the resulting LR into the optimizer's param groups. The first
  `optimizer.step()` after `resume_training_checkpoint()` therefore used the
  LR baked in at `build_optimizer()` time (full base LR), even mid-cosine —
  a silent, potentially large LR jump on every resume.
- **Fix:** reapply `lr_at(max(step-1, 0))` (the exact LR groups held at save
  time); fresh construction applies `lr_at(0)` for consistency.
- **Regression tests:** `test_scheduler_fresh_applies_warmup_lr`,
  `test_scheduler_resume_applies_resumed_lr` (+ CUDA path test).

## B-2022-2 — Step-0 optimizer step ran at full base LR
- **Severity:** medium · **Status:** FIXED (`27c7f78`)
- **File:** `neuratal/training/optim.py` — `WarmupCosineSchedule.__init__`
- **Defect:** Fresh scheduler left param groups at `base_lr` for the first
  optimizer step; warmup never covered step 0 (for warmup_steps ≥ 2 the first
  step ran at 1.0× base instead of 1/warmup× base).
- **Fix:** construction applies `lr_at(0)`. warmup ≤ 1 unchanged; warmup ≥ 2
  now follows the intended ramp. Documented behavioral change.
- **Regression test:** `test_scheduler_fresh_applies_warmup_lr`.

## B-2022-3 — Optimizer-group LR misassignment on corrupt scheduler state
- **Severity:** medium · **Status:** FIXED (`27c7f78`)
- **File:** `neuratal/training/optim.py` — `load_state_dict`
- **Defect:** `base_lrs` was accepted at any length; the engine's two param
  groups (decay/no-decay) would then zip with a short/long LR list silently.
- **Fix:** length check → `ValueError` with a precise message.
- **Regression test:** `test_scheduler_base_lrs_length_mismatch_rejected`.

## B-2022-4 — Sharded checkpoint `total_size` never verified on load
- **Severity:** medium (integrity) · **Status:** FIXED (`eeed9b4`)
- **File:** `neuratal/serialization/sharded.py` — `load_pretrained_sharded`
- **Defect:** `metadata.total_size` recorded at save; load re-derived only
  weight_map/keys/shapes/dtypes. A stale-but-parseable shard set would pass
  silently.
- **Fix:** byte-count comparison of actually-loaded tensors vs metadata →
  `ValueError` on mismatch. (Safetensors CRC still guards bit-level corruption;
  this catches replaced-with-different-content.)
- **Regression test:** `test_sharded_total_size_integrity_check`.

## B-2022-5 — Duplicated `_atomic_save_safetensors` (sharded.py)
- **Severity:** low (maintainability/consistency) · **Status:** FIXED (`eeed9b4`)
- **Defect:** two copies of the atomic temp+rename writer with identical
  semantics; the sharded one was a drift risk.
- **Fix:** sharded.py imports the single-file writer's helper + the same
  `_check_no_shared_storage` anti-aliasing guard.

## B-2022-6 — `sorted_experts` computed even without capacity
- **Severity:** low (wasted gather) · **Status:** FIXED (v1.1 `d9bb7c1`)
- **File:** `neuratal/model/moe.py` — `_forward_tokens`
- **Fix:** computed only inside the `capacity_factor is not None` branch.

## B-2021-1 — Full-width tied head crashed the engine (v1.1)
- **Severity:** critical (API crash) · **Status:** FIXED (v1.1 `49cbbc6`)
- **File:** `neuratal/training/engine.py`
- **Defect:** `core.output_projection.out_proj.weight` dereferenced
  unconditionally; with `embedding_size=None` (full-width, `out_proj is None`)
  the engine crashed before the first step.
- **Fix:** `_head_out_proj_weight()` returns `None`; chunked CE accepts it.
- **Regression test:** `tests/test_engine_regressions.py::test_full_width_train_step_and_checkpoint`.

## B-2021-2 — fp16 AMP ran without a GradScaler (v1.1)
- **Severity:** high (silent gradient underflow/overflow) · **Status:** FIXED (v1.1 `49cbbc6`)
- **Fix:** autocast + `torch.amp.GradScaler`, unscale-before-clip, step skipped
  on non-finite grads; scaler state saved/restored.
- **Regression test:** `test_fp16_step_and_scaler_resume`.

## B-2021-3 — DDP MoE reducer wedges on zero-token experts (v1.1)
- **Severity:** high (distributed correctness) · **Status:** FIXED (v1.1 `49cbbc6`)
- **File:** `neuratal/training/engine.py` — `train()`
- **Defect:** a micro-batch may route zero tokens to some expert; DDP's
  reducer then never marks those params ready → hang/timeout.
- **Fix:** `find_unused_parameters=True` when `config.is_moe`. Dense keeps
  the fast default. **Not validated on multi-GPU** (none available) — labeled.

## B-2021-4 — Resume did not verify tokenizer fingerprint (v1.1)
- **Severity:** high (silent model↔tokenizer mismatch) · **Status:** FIXED (v1.1 `49cbbc6`)
- **Fix:** `resume_training_checkpoint(..., expected_tokenizer_fingerprint=...)`;
  mismatch/missing → `ValueError`; mismatch-by-architecture (dense↔MoE) → `ValueError`;
  optimizer-index out of range (mismatched model) → `ValueError`.

## B-2021-5 — `torch.tensor(0.0, device=cuda)` H2D scalar in MoE drop path (v1.1)
- **Severity:** low (sync/allocation) · **Status:** FIXED (v1.1 `b608fdf`) → `new_zeros`.

## VERIFIED — NOT A BUG (investigated, closed)

- **CUDA RNG restore:** works exactly (`torch.cuda.set_rng_state(state.cpu(), i)`
  verified byte-equal after corrupt→restore). Test gap only; new CUDA-gated
  test added. Note: an earlier probe by this audit had a test-order error
  (re-seeding inside the fixture) — the code was correct; the test was wrong.
- **All-ignored-target loss backward:** `F.cross_entropy(reduction='sum',
  ignore_index=-100)` backpropagates zeros without crashing.
- **`@torch.compile` on MoEFFN:** 15.07 vs 14.90 ms — no win (noise); not adopted.
- **`torch._grouped_mm`:** exact-match outputs, 0.601 vs 0.615 ms — adopted? NO.
- **RMSNorm under autocast:** fused weighted kernel cannot dispatch → old
  composite path was ~1.65× slower; replaced with fused normalize-without-weight
  (v1.2 `65830db`), ≤1-ulp worst case, 0.0 diff on common tensors.
- **SDPA backend:** re-measured flash/cuDNN/default over 3 interleaved rounds —
  no reproducible winner; default kept (non-adoption is the correct outcome).

## OPEN (documented, accepted, or blocked)

- **O-1 — `use_reentrant=True` in `training/chunked_loss.py`:** deprecated
  argument; works correctly in this build; kept for compatibility of the
  older public entry point. Revisit at next torch bump.
- **O-2 — Shuffle-order parity after resume:** data cursor advances by
  `tokens_seen // tokens_per_micro` micro-batches; exact for
  `shuffle=False`; re-permuted under `shuffle=True`. Documented, not claimed.
- **O-3 — Multi-GPU DDP MoE (`find_unused_parameters=True`):** correctness
  fix is untested on real multi-GPU; single-process gloo smoke test only.
- **O-4 — Checkpoint crash-consistency:** atomic rename protects against
  torn writes; no power-loss fault-injection campaign was run. Do not claim
  crash-proofing.
- **O-5 — activation memory estimator coefficient:** uncalibrated above
  dev scale; ±50 % guidance only; CE workspace now fp32-charged.
- **O-6 — `benchmark_model` noise:** runs after idle can read 10–25 % fast
  (SM clock 2565/3105 MHz); interleaved A/B is the only sound attribution
  method on this machine.

---

## v1.3.0-candidate additions (2026-11-17)

## B-2023-1 — Scheduler state stored as float32 (silent resume LR drift)
- **Severity:** high (training-correctness, silent) · **Status:** FIXED (`7939bbe`)
- **File:** `neuratal/training/checkpoint.py` — save path (`_scalar_to_tensor` introduced)
- **Defect:** `torch.tensor(float(x))` stores scheduler base LRs (and optimizer
  non-tensor scalars) as **float32**. After every resume, the restored base
  LRs differ from construction by ~1e-7 relative; the entire resumed cosine
  trajectory drifts with it. Bit-exactness of `lr_at()` between uninterrupted
  and resumed runs was impossible.
- **Fix:** ints stored as `int64`, floats as **`float64`** (`_scalar_to_tensor`).
  Round-trip is now bit-exact across all 7 tested resume points.
- **Evidence:** `docs/scheduler_resume_verification.md` —
  uninterrupted vs interrupted at K ∈ {0,1,4,5,8,15,16}: LR trajectories,
  parameters (bitwise), optimizer state, losses — all identical.
- **Note:** pre-v1.3 checkpoints keep their float32-rounded values
  (unrecoverable in hindsight) — backwards compatible for loading, but only
  checkpoints written by v1.3+ get exact round-trips.

## B-2023-2 — Scheduler `load_state_dict` accepted malformed state
- **Severity:** medium (robustness/clarity) · **Status:** FIXED (`7939bbe`)
- **File:** `neuratal/training/optim.py`
- **Defect:** missing keys produced raw `KeyError`; step/warmup/total could be
  non-integers or negatives; `min_lr_ratio` could be NaN/out-of-range;
  base_lrs items could be non-numeric.
- **Fix:** required-key validation + type/range/finiteness checks with precise
  `ValueError` messages; `base_lrs` length vs param groups enforced (v1.2).
- **Regression tests:** `test_scheduler_base_lrs_length_mismatch_rejected`,
  `test_scheduler_resume_applies_resumed_lr`,
  `test_scheduler_fresh_applies_warmup_lr` (v1.2/v1.3).

## B-2023-3 — Sharded checkpoint accepted replaced same-size stale shards
- **Severity:** medium (integrity) · **Status:** FIXED (`7939bbe`)
- **File:** `neuratal/serialization/sharded.py`
- **Defect:** v1.2's byte-count check passes when a stale shard has the exact
  same size as a healthy one.
- **Fix:** per-shard SHA-256 stored in `metadata.shard_hashes` at save time
  and verified at load (default on; `verify_shard_hashes=False` documented
  for enormous checkpoints with out-of-band verification); missing/entry-
  mismatched hashes are hard errors; legacy indexes without hashes still
  load (documented backwards compat; `total_size` still enforced).
- **Measured:** ~63 ms per 37 MB model (rc ≈ 50% of default load time; model
  init dominates anyway).
- **Regression tests:** `test_sharded_tampered_shard_rejected`,
  `test_sharded_missing_hash_entry_rejected`,
  `test_sharded_legacy_index_without_hashes_loads`,
  `test_sharded_save_records_and_verifies_hashes`.

## Not adopted / closed

- **Distributed checkpointing:** out of scope (local sharded only, by design).
- **Crash-consistency fault-injection campaign:** not performed; atomic rename
  is not power-loss proof (documented, not claimed).
- **C-level parallel Python bindings in NTransformer:** n/a (work is in
  NTokenizer's repo; see `../NTokenizer/docs/tokenizer_engineering_audit.md`).
