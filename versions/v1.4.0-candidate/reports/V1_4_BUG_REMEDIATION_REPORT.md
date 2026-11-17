# V1.4 — Bug Remediation Report (independent v1.3 audit follow-up)

- **Baseline (audited) commit:** `7d6bf9466171d2b7ab9ff2364890fa8ed1fb0dd9`
  (branch `feature/v1.3-enterprise-2026-11`)
- **Remediation branch:** `fix/v1.4-bug-remediation-2026-11`
- **Code commit under test:** `5d912b3b2157c0a1429f46a641b949dcd828ee16`
- **Snapshot:** `versions/v1.4.0-candidate/` (SOURCE = tree at that commit)
- **Source of truth for findings:** `/home/justerdem14/.cache/ntransformer_audit/NTRANSFORMER_BUG_HUNT_REPORT.md`

## 1. Executive summary

The independent v1.3 audit reported 7 confirmed defects (1 HIGH, 3 MEDIUM,
3 LOW), 3 STRONG findings, 6 POSSIBLE findings and 4 historical findings.
**Every confirmed defect is remediated and regression-tested in v1.4.0-candidate.**
The HIGH bug (DDP + factorized-tied-head MoE crashing at the first backward)
is root-caused and fixed; multi-process DDP training is now validated in this
environment. One MEDIUM (sharded verification) and one MEDIUM (sharded path)
plus one LOW (scheduler off-by-one) are fixed with fail-closed semantics and
explicitly updated tests. All 6 POSSIBLE findings are fixed and covered.
One STRONG finding (index not authenticated) is a documented design
limitation and is deliberately **not** fixed. Historical findings are
branch-only or fixed upstream and are recorded, not re-implemented.

**Result: 1 HIGH fixed, 3 MEDIUM fixed, 3 LOW fixed, 6 POSSIBLE fixed,
2 STRONG fixed (1 resolved by design, 1 security hardening), 1 STRONG
accepted as documented design limitation, 4 HIST documented.**

## 2. Finding-by-finding resolution (required table)

| Original ID | Severity | Original evidence | Investigation result | Fix commit/file | Regression test | Final status |
|---|---|---|---|---|---|---|
| BUG-01 | HIGH | CONFIRMED (2-proc gloo; deterministic; all 7 MoE presets) | Selector: `find_unused_parameters=True` + `output_projection.out_proj.weight` used **outside** the DDP-wrapped forward (external chunked CE) → reducer marked it "ready (unused)" at forward end; autograd hook fired in backward → "marked as ready twice" on both ranks. Verified again on unmodified v1.3 export before fixing | `5d912b3` — `neuratal/training/engine.py` (new `_DDPLossModule`; hidden-only + chunked-CE inside one wrapped forward; engine loop gated on finite grads) | `tests/test_v14_regressions.py::TestDDPEndToEnd` — 6 tests; 2-proc gloo bitwise parity vs single-process on identical data; grad accumulation ×2; checkpoint save/resume after DDP | **CONFIRMED AND FIXED** |
| BUG-02 | MEDIUM | CONFIRMED (scaler skip did not advance `scheduler.step`; diag script) | fp16 path skipped the optimizer update but still advanced scheduler + `global_step`. Additionally (same audit area, new): bf16/fp32 path had **no non-finite-grad guard** — AdamW silently absorbed NaN weights (verified: finite weights → NaN after a forced inf/nan grad step) | `5d912b3` — `neuratal/training/engine.py` (skip ⇒ no scheduler/`global_step` advance; skipped steps logged; loop extends to `total_steps` real updates with a hard iteration cap; non-finite grads → step refused) | `TestEngineStockFp16::test_engine_skips_step_on_nonfinite_grad_norm` (CPU), `test_fp16_overflow_skips_lr_advance` (CUDA), DDP parity tests | **CONFIRMED AND FIXED** |
| BUG-03 | MEDIUM | CONFIRMED (absolute + `../` paths opened external files) | `weight_map` filenames used verbatim as read paths; no containment/normalization check | `5d912b3` — `neuratal/serialization/sharded.py` (`_validated_shard_path`: strict regex `^model-\d{5}-of-\d{5}\.safetensors$`, resolved-path containment in checkpoint dir, regular-file check) | `TestShardedIntegrity::test_absolute_shard_path_rejected`, `test_relative_traversal_shard_path_rejected`, `test_non_standard_shard_name_rejected` | **CONFIRMED AND FIXED** |
| BUG-04 | MEDIUM | CONFIRMED (hash-less index loaded corrupted shard silently; total_size silently skipped) | `if verify_shard_hashes and recorded_hashes:` failed open; `isinstance(total_size, int)` failed open | `5d912b3` — `neuratal/serialization/sharded.py` (fail-closed raises for absent/empty `shard_hashes` or missing/non-integer `total_size`; `verify_shard_hashes=False` kept as documented opt-out and now warns loudly) | `TestShardedIntegrity` (5 tests) + updated `tests/test_serialization.py::test_sharded_legacy_index_without_hashes_rejected_or_warns` | **CONFIRMED AND FIXED** |
| BUG-05 | LOW | CONFIRMED (instrumented LR trajectory) | `step()` applied `lr_at(self._step)` without incrementing first; `lr_at(0)` applied twice; whole schedule retarded one optimizer step; `lr_at(total_steps)` (min_lr floor) never reached within a run; resume built around the retarded invariant kept resume bit-exact but the schedule was wrong | `5d912b3` — `neuratal/training/optim.py` (increment-then-apply; resume restores `lr_at(_step)` exactly) | `TestSchedulerContract` (4 tests) + one existing test expectation updated (`test_scheduler_resume_applies_resumed_lr`: restore point `lr_at(3)`→`lr_at(4)`, the new contract) | **CONFIRMED AND FIXED** |
| BUG-06 | LOW | CONFIRMED (fuzz of malformed state) | Missing cross-field validation (`warmup ≥ total_steps` accepted; non-positive `base_lrs` accepted); resume crashed in constructor when `max_steps < saved warmup` (before the checkpoint's own valid values were consulted); loop/schedule bound split | `5d912b3` — `neuratal/training/optim.py` (warmup < total, base_lrs strictly positive), `neuratal/training/engine.py` (resume constructs from a guaranteed-valid placeholder then restores the checkpointed schedule; mismatch ⇒ `UserWarning`; extra steps run at min-LR floor) | `TestSchedulerValidation` (4 tests) | **CONFIRMED AND FIXED** |
| BUG-07 | LOW | CONFIRMED (code read; fp32 configs) | Estimator defaulted `grad_dtype_bytes=2` even for `float32` models; the param/act widths adapt but grad width was fixed | `5d912b3` — `neuratal/estimate.py` (dtype-derived default: 4 for fp32, 2 otherwise; explicit override honored; invalid values rejected) | `TestEstimator` (3 tests) | **CONFIRMED AND FIXED** |
| STRONG-01 | — | STRONG (dense DDP worked by accident of hook firing; MoE warned "did not find any unused parameters") | **Resolved by construction** — the BUG-01 joint forward consumes every reducer-relevant parameter inside the wrapped forward; residual `find_unused_parameters=True` warning noise for MoE is benign and documented | `5d912b3` — same locus as BUG-01 | `TestDDPEndToEnd` (all 6) | **CONFIRMED AND FIXED (by design)** |
| STRONG-02 | — | STRONG (interrupted save leaves orphan shards) | No cleanup on failure path; pre-existing files must never be deleted | `5d912b3` — `neuratal/serialization/sharded.py` (save tracks only files it created; failure unlinks those only; pre-existing untouched) | `TestShardedIntegrity::test_interrupted_save_leaves_no_new_shards` | **CONFIRMED AND FIXED** |
| STRONG-03 | — | STRONG (index unauthenticated; attacker recomputing hashes passes) | Inherent to the local-format design; no signing key/material exists to pin; adding cryptographic identity would change the format contract | Not fixed (by design). Mitigations landed: fail-closed SHA-256 (BUG-04), path containment (BUG-03), atomic writes. Documented in LIMITATIONS.md and BUG_REGISTRY.md | — | **CONFIRMED BUT NOT FIXED (deliberate; documented)** |
| POSSIBLE-01 | — | POSSIBLE | Resume with a different `total_steps` than the checkpoint had unspecified semantics | **Resolved as part of BUG-06** (checkpointed schedule always wins; loop bound honored; `UserWarning`; extra steps at min-LR floor) | `test_resume_with_smaller_max_steps_does_not_crash_and_warns`, `test_resume_with_larger_max_steps_warns_and_floors` | **CONFIRMED AND FIXED / RESOLVED** |
| POSSIBLE-02 | — | POSSIBLE (cross-device RNG restore silently diverges) | Missing CPU RNG key, saved-but-unavailable CUDA devices, and missing per-device CUDA RNG were all silently ignored | `5d912b3` — `neuratal/training/checkpoint.py` (loud `UserWarning`s for each class of gap) | `TestResumeWarnings` (2 tests) | **CONFIRMED AND FIXED** |
| POSSIBLE-03 | — | POSSIBLE (chunk_size=0 silently became 256) | **Became a real silent defect**: `chunk_size=0` silently defaulted; **`chunk_size<0` produced a zero loss with zero gradients — no error at all** | `5d912b3` — `neuratal/training/chunked_linear_ce.py` + `chunked_loss.py` (raise `ValueError` for `< 1`) | `TestChunkedCEChunkSize` (2 tests) | **CONFIRMED AND FIXED** |
| POSSIBLE-04 | — | POSSIBLE (tiny corpus → cryptic crash in `TokenDataset`) | Boundary clamp could place the val split below one sequence; error message was misleading | `5d912b3` — `neuratal/training/data.py` (clear up-front error at the split point; correct `[seq+1, total-seq-1]` clamp) | `TestValidationSplit` (2 tests) | **CONFIRMED AND FIXED** |
| POSSIBLE-05 | — | POSSIBLE (whole-file read before size validation) | `path.read_bytes()` happened before any header/size check; unbounded memory exposure on hostile paths | `5d912b3` — `neuratal/integration/manifest.py` (header-only read; size/CRC/cap validation before any payload read; payload read bounded by declared `payload_len`; unused whole-file "raw" read dropped) | `TestNtokParser` (4 tests, incl. sparse-file fixture) | **CONFIRMED AND FIXED** |
| POSSIBLE-06 | — | POSSIBLE (DeltaNetCache.commit overshoot) | `_length` incremented before the `max_position_embeddings` check; error path left cache in an over-limit state | `5d912b3` — `neuratal/model/gated_delta_net.py` (check-then-increment) | `TestDeltaNetCacheCommit` | **CONFIRMED AND FIXED** |
| HIST-01 | MEDIUM (branch) | CONFIRMED on `cand/v0.7-sliding-window` | `reference_attention` ignores the sliding window (full causal attention); SDPA path correct; branch rejected, never merged; v1.4 carries no sliding-window code | Not fixed (obsolete code removed from scope) — documented for any future SWA revival: thread the window into `reference_attention` | — | **HISTORICAL ONLY — DOCUMENTED** |
| HIST-02 | LOW (branch) | CONFIRMED (documented limitation) | GDN WY-chunkwise cumprod underflows for chunk > 64 (registry documents "tested at C<=64 only"); branch rejected, never merged; not in v1.4 | Not fixed (branch-only; rejected candidate) — documented | — | **HISTORICAL ONLY — DOCUMENTED** |
| HIST-03 | LOW | CONFIRMED (fixed upstream in v0.7.1) | Default `swa_window=512` vs small `max_position_embeddings` crashed master/v0.7-candidate; fixed in v0.7.1 (`0b9945f`) on that branch; not applicable to v1.4 | Not applicable — documented | — | **HISTORICAL ONLY — FIXED UPSTREAM / DOCUMENTED** |
| HIST-04 | INFO | CONFIRMED | Snapshot/registry version drift (freeze README said 418; HEAD said 422; BUG_REGISTRY heading cited stale commit) | `5d912b3` + freeze commit — README/BUG_REGISTRY/CHANGELOG updated for v1.4 | snapshot-vs-HEAD byte-identical SOURCE check (see V1_4_REGRESSION_TEST_REPORT.md §6) | **CONFIRMED AND FIXED (docs)** |

## 3. New defect found and fixed during remediation (audit deep-dive)

| ID | Severity | Evidence | Resolution |
|---|---|---|---|
| V14-NEW-01 | MEDIUM | CONFIRMED empirically: finite `.grad` containing `inf` lands in `clip_grad_norm_` → `inf × 0 = NaN` mask; AdamW then **silently poisons all weights to NaN** (verified). bf16/fp32 path had no non-finite guard | Fixed as part of BUG-02 (finite gate; skip + loud log) |
| V14-NEW-02 | LOW | `chunk_size` negative → empty loop → **silent zero loss/gradient** (became POSSIBLE-03) | Fixed (see POSSIBLE-03) |

## 4. Verification commands

```
# full suite
.venv/bin/python -m pytest tests/ --basetemp=/home/justerdem14/.cache/pytest_tmp -q
#   -> exit 0; 460 passed, 3 skipped, 3 warnings (Appendix in test_results.txt
#      records the environment's intermittently truncated summary line)

# every BUG re-check on the fixed tree
.venv/bin/python /home/justerdem14/.cache/ntransformer_audit/diag_ddp_moe.py      # DDP now trains
.venv/bin/python /home/justerdem14/.cache/ntransformer_audit/diag_sharded.py      # tampers rejected
.venv/bin/python /home/justerdem14/.cache/ntransformer_audit/diag_sharded2.py     # traversal + fail-open closed
.venv/bin/python /home/justerdem14/.cache/ntransformer_audit/diag_clipnan.py      # non-finite guard evidence
```

## 5. Residual risks / what is not fixed

- STRONG-03 (index unauthenticated) — accepted design limitation, documented.
- DDP validated only on 2-process CPU/gloo; **no multi-GPU/NCCL** on this machine.
- fp16 end-to-end convergence — mechanics only verified.
- Cross-version resume (v1.3→v1.4) LR shifts one schedule position (documented).
- NTokenizer integration still not executed (policy).
