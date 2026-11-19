# NTransformer — Consolidated Bug Registry (v1.4.0-candidate @ HEAD of `fix/v1.4-bug-remediation-2026-11`)

All confirmed defects, their severity, status, and regression evidence, mapped
one-to-one to the independent v1.3 audit's original finding IDs. Complements
`docs/bug_registry.md` (v1.3) and `docs/ultimate_audit/bug_registry.md` (v1.1/v1.2).
Original audit report: `/home/justerdem14/.cache/ntransformer_audit/NTRANSFORMER_BUG_HUNT_REPORT.md`
(audited `7d6bf9466171d2b7ab9ff2364890fa8ed1fb0dd9`).

## v1.4 resolution of the independent v1.3 audit findings

| Original ID | Severity | Original evidence | Investigation result | Fix commit/file | Regression test | Final status |
|---|---|---|---|---|---|---|
| BUG-01 | HIGH | CONFIRMED (gloo, deterministic) | DDP `find_unused_parameters=True` marked factored-head `out_proj.weight` "ready (unused)"; hook fired later through the external chunked-CE → reducer raised "marked as ready twice" on step 1. Reproduced on v1.3 export | `neuratal/training/engine.py` (`_DDPLossModule`, joint hidden-only+chunked-CE inside the wrapped forward) | `tests/test_v14_regressions.py::TestDDPEndToEnd` (6 tests: factored/full-width × dense/MoE, grad accumulation, DDP checkpoint resume) + 2-proc CPU gloo bitwise parity vs single-process | **FIXED (v1.4)** |
| BUG-02 | MEDIUM | CONFIRMED | `scheduler.step()` and `global_step` advanced even when `scaler.step()` skipped the optimizer update (non-finite unscaled grads). Also: bf16/fp32 had NO non-finite guard — AdamW silently absorbed NaN grads (inf×0 masks) and poisoned weights; verified empirically | `neuratal/training/engine.py` (finite gate on unscaled norm; skipped steps logged, not scheduled; `iteration_cap`) | `tests/test_v14_regressions.py::TestEngineStockFp16::test_engine_skips_step_on_nonfinite_grad_norm`, `test_fp16_overflow_skips_lr_advance` | **FIXED (v1.4)** |
| BUG-03 | MEDIUM | CONFIRMED | `weight_map` shard filenames used verbatim as paths; absolute `/` paths and `..` traversal opened files outside the checkpoint dir | `neuratal/serialization/sharded.py` (`_validated_shard_path`: regex `^model-\d{5}-of-\d{5}\.safetensors$`, resolved-path containment, regular-file check) | `tests/test_v14_regressions.py::TestShardedIntegrity::test_absolute/relative/nonstandard_*` | **FIXED (v1.4)** |
| BUG-04 | MEDIUM | CONFIRMED | `verify_shard_hashes=True` silently skipped verification when `shard_hashes` absent/falsy (corrupted same-size shard loaded silently); `total_size` check silently skipped when absent | `neuratal/serialization/sharded.py` (fail-closed raises for absent/empty `shard_hashes` or missing/non-integer `total_size`; loud opt-out warning) | `tests/test_v14_regressions.py::TestShardedIntegrity::test_missing_shard_hashes_fails_closed/_optout_warns/test_missing_total_size_*` + updated `tests/test_serialization.py::test_sharded_legacy_index_without_hashes_rejected_or_warns` | **FIXED (v1.4)** |
| BUG-05 | LOW | CONFIRMED | `step()` applied `lr_at(self._step)` without incrementing first; `lr_at(0)` applied twice; whole schedule retarded one optimizer step; min-LR floor never reached | `neuratal/training/optim.py` (increment-then-apply; resume restores `lr_at(_step)`) | `tests/test_v14_regressions.py::TestSchedulerContract` (4 tests) | **FIXED (v1.4)** |
| BUG-06 | LOW | CONFIRMED (gaps) / POSSIBLE (impact) | `load_state_dict` accepted `warmup ≥ total_steps` and non-positive `base_lrs`; resume crashed in constructor with `max_steps < saved warmup`; loop bound vs scheduler `total_steps` split silently | `neuratal/training/optim.py` (cross-field + positivity validation), `neuratal/training/engine.py` (constructor built on a validated placeholder on resume; checkpointed schedule wins; warning on mismatch) | `tests/test_v14_regressions.py::TestSchedulerValidation` (4 tests) | **FIXED (v1.4)** |
| BUG-07 | LOW | CONFIRMED | estimator charged fp32 gradients at 2 bytes (default `grad_dtype_bytes=2` even for `float32` configs) | `neuratal/estimate.py` (dtype-derived default: 4 for fp32, 2 otherwise; explicit override respected; invalid values rejected) | `tests/test_v14_regressions.py::TestEstimator` (3 tests) | **FIXED (v1.4)** |
| STRONG-01 | — | STRONG | Dense DDP worked by incidental hook firing; MoE DDP warned "did not find any unused parameters" (false positive) | **Resolved by BUG-01's joint-forward wrapper** — all reducer-relevant params are consumed inside the wrapped forward; 2-proc validation matrix confirms; residual `find_unused_parameters=True` warning documented as benign (zero-token experts still need it) | `TestDDPEndToEnd` | **FIXED (v1.4, by construction)** |
| STRONG-02 | — | STRONG | Interrupted `save_pretrained_sharded` left orphan shards behind | `neuratal/serialization/sharded.py` (save tracks shards written by this call; on failure removes only those, never pre-existing files) | `tests/test_v14_regressions.py::TestShardedIntegrity::test_interrupted_save_leaves_no_new_shards` | **FIXED (v1.4)** |
| STRONG-03 | — | STRONG | Sharded index file itself unauthenticated (hashes can be recomputed by an attacker editing index+shards) | Not fixed by design — signing/MAC over metadata changes the format contract and is out of scope; residual risk documented; fail-closed SHA-256 (BUG-04) covers accidental corruption only | — | **CONFIRMED BUT NOT FIXED (documented design limitation; mitigations landed)** |
| POSSIBLE-01 | — | POSSIBLE | Resume with different `total_steps` than checkpoint — unspecified semantics | Resolved as part of BUG-06: checkpointed schedule always wins; requested loop bound honored; a `UserWarning` is issued; extra steps run at the min-LR floor | `test_resume_with_smaller_max_steps_does_not_crash_and_warns`, `test_resume_with_larger_max_steps_warns_and_floors` | **RESOLVED (v1.4)** |
| POSSIBLE-02 | — | POSSIBLE | `resume_training_checkpoint` silently skipped absent CUDA RNG keys; no message when devices differ from checkpoint | `neuratal/training/checkpoint.py` (loud warnings for missing CPU RNG, saved-but-now-unavailable CUDA devices, missing per-device RNG) | `tests/test_v14_regressions.py::TestResumeWarnings` | **FIXED (v1.4)** |
| POSSIBLE-03 | — | POSSIBLE | `chunk_size=0` silently became 256; **negative chunk sizes produced zero loss silently** | `neuratal/training/chunked_linear_ce.py` and `chunked_loss.py` (raise `ValueError` for `< 1`) | `tests/test_v14_regressions.py::TestChunkedCEChunkSize` | **FIXED (v1.4)** |
| POSSIBLE-04 | — | POSSIBLE | `train_validation_split` on tiny corpora crashed cryptically inside `TokenDataset`; boundary could be under-clamped | `neuratal/training/data.py` (clear up-front error at the split point; correct `[seq+1, total-seq-1]` clamp) | `tests/test_v14_regressions.py::TestValidationSplit` | **FIXED (v1.4)** |
| POSSIBLE-05 | — | POSSIBLE | `parse_ntok_file` read the whole file before size/CRC validation | `neuratal/integration/manifest.py` (header-only read; size/CRC/cap validation before payload read; payload read bounded by declared `payload_len`; unused whole-file "raw" read dropped) | `tests/test_v14_regressions.py::TestNtokParser` (sparse-fixture tests) | **FIXED (v1.4)** |
| POSSIBLE-06 | — | POSSIBLE | `DeltaNetCache.commit()` incremented `_length` before checking `max_position_embeddings`, leaving it over limit on failure | `neuratal/model/gated_delta_net.py` (check-then-increment) | `tests/test_v14_regressions.py::TestDeltaNetCacheCommit` | **FIXED (v1.4)** |
| HIST-01 | MEDIUM (branch) | CONFIRMED on `cand/v0.7-sliding-window` | `reference_attention` ignores the sliding window (full causal attention; silent); SDPA path correct | Branch-only (SWA rejected, never merged); v1.4 contains no sliding-window code; documented for future revival; NOT fixed (would reintroduce rejected-candidate code) | — | **BRANCH-ONLY / NOT APPLICABLE** |
| HIST-02 | LOW (branch) | CONFIRMED (documented) | GDN WY-chunkwise cumprod underflow for chunk > 64 (documented limitation, `registry/best_version.json`) | Branch-only; not in v1.4; documented | — | **BRANCH-ONLY / NOT APPLICABLE** |
| HIST-03 | LOW | CONFIRMED (fixed v0.7.1) | Master/v0.7-candidate crash: default `swa_window=512` vs small `max_position_embeddings` | Fixed upstream on `cand/v0.7-sliding-window`; not applicable to v1.4 (no `swa_window` field) | — | **FIXED UPSTREAM / NOT APPLICABLE** |
| HIST-04 | INFO | CONFIRMED | Snapshot/registry version drift (README 418 vs HEAD 422; registry heading citing stale commit) | Docs updated in v1.4 (README, this registry, CHANGELOG, snapshot) | `diff` of snapshot vs HEAD in freeze step | **FIXED (v1.4, docs)** |

## Historical fixed defects (carried forward; independently re-verified)

| ID | Severity | Status |
|---|---|---|
| B-2020-x (v1.0–1.2 audit family) | mixed | FIXED (see v1.2/v1.3 registries) |
| B-2021-1 … B-2021-5 | critical/high/low | FIXED, verified in both v1.3 and v1.4 (incl. residual fp16 gap → BUG-02) |
| B-2022-1 … B-2022-6 | medium/low | FIXED, verified in both v1.3 and v1.4 |
| B-2023-1 | high (silent) | FIXED, verified (bitwise resume re-verified in v1.3 and v1.4) |
| B-2023-2 | medium | FIXED with remaining gaps → BUG-06 fixed in v1.4 |
| B-2023-3 | medium (integrity) | FIXED, verified (same-size stale shard rejected; fail-open remainder → BUG-04 fixed in v1.4) |
| RMSNorm autocast | medium | FIXED, verified (bitwise equal to manual path under CUDA autocast) |

## Open / advisory (not defects, documented)

| ID | Severity | Note |
|---|---|---|
| O-1 | informational | `use_reentrant=True` in legacy `chunked_loss.py` — deprecated but correct in this build |
| O-2 | informational | shuffle-order data replay after resume is re-permuted (documented) |
| O-3 | medium | **DDP validated on 2-process CPU gloo only; no multi-GPU/NCCL hardware here — distributed throughput unproven** |
| O-4 | informational | no power-loss fault injection; atomic rename ≠ power-loss proof |
| O-5 | informational | activation memory estimator ±50 % beyond dev scale |
| O-6 | informational | benchmark noise ±10–20 % on this GPU |
| O-7 | informational | sharded index unauthenticated (STRONG-03) |

## Severity tally (v1.4)

- Fixed from independent audit: **1 HIGH (BUG-01), 3 MEDIUM (BUG-02/03/04), 3 LOW (BUG-05/06/07)**.
- STRONG findings: **2 fixed** (STRONG-01 via BUG-01's joint-forward design; STRONG-02 orphan-shard cleanup), **1 confirmed but not fixed by design** (STRONG-03, documented).
- POSSIBLE findings: **6 fixed** (POSSIBLE-01..06).
- Historical findings: 4 handled (HIST-01/02 documented branch-only; HIST-03 fixed upstream; HIST-04 docs drift fixed).
- Open critical: **0**. Open high: **0**. Open confirmed medium/low: **0**. Open advisory: **1** (O-3, environmental — no multi-GPU hardware available).
- Suite: **460 passed, 3 skipped, 3 warnings** (was 422 in v1.3).
