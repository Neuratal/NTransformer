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
