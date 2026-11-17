# V1.4 — Regression Test Report

- **Suite:** `tests/` (now incl. `tests/test_v14_regressions.py`, 38 tests)
- **Baseline (v1.3):** 422 passed, 3 skipped, 1 warning
- **Result (v1.4):** **460 passed, 3 skipped, 3 warnings** (exit code 0)
- **Command run:** `.venv/bin/python -m pytest tests/ --basetemp=/home/justerdem14/.cache/pytest_tmp -q [--tb=no]`
- **Runner:** Python 3.14.4, torch 2.15.0+cu132, 1× RTX 4060 Laptop GPU (8 GB)
- **`/tmp` note:** the environment's 7.4 GB tmpfs is avoided via `--basetemp`
  on the large root filesystem (this is a pre-existing environment caveat, not
  a defect; see v1.3 TEST_RESULTS.md for the same note).
- **Summary-line caveat:** in this environment `pytest -q` intermittently
  drops the final "N passed" line. `test_results.txt` therefore carries an
  appendix recording exit code 0 and a direct count of outcome markers in the
  file (460 passed + 3 skipped, 0 failed/error).

## 1. Test changes in v1.4 (all intentional; none weaken coverage)

| File | Change | Why |
|---|---|---|
| `tests/test_v14_regressions.py` | **new**, 38 tests | Regression coverage for every v1.4 finding |
| `tests/test_engine_regressions.py` | expectation updated: restore LR `lr_at(3)`→`lr_at(4)` in `test_scheduler_resume_applies_resumed_lr` | BUG-05 fix — resume now restores `lr_at(_step)` (the corrected contract); the saved LR remains exactly what the groups held at save time |
| `tests/test_serialization.py` | `test_sharded_legacy_index_without_hashes_loads` → `..._rejected_or_warns` (was: hash-less index loads silently; now: default verify raises, opt-out warns) | BUG-04 fail-closed semantics |

No existing test was deleted or weakened. The two expectation updates are
required consequences of the two fixes; both still assert exactly the
(stricter) new contract.

## 2. New regression tests (by finding)

| Finding | Tests |
|---|---|
| BUG-01 (DDP MoE) | `TestDDPEndToEnd` — 2-process gloo, factored × full-width × dense × MoE; ≥2 optimizer steps; 1-sample deterministic dataset → bitwise parity vs single-process (params **and** grads); grad accumulation ×2; checkpoint save (every step) → resume roundtrip → weight equality; loss finite |
| BUG-02 (scaler/schedule) | `test_engine_skips_step_on_nonfinite_grad_norm` (engine refuses the step; weights unmoved; skipped logged), `test_fp16_overflow_skips_lr_advance` (CUDA, scaler semantics) |
| BUG-03 (path traversal) | `test_absolute_shard_path_rejected`, `test_relative_traversal_shard_path_rejected`, `test_non_standard_shard_name_rejected` |
| BUG-04 (fail-closed) | `test_missing_shard_hashes_fails_closed`, `test_missing_shard_hashes_optout_warns`, `test_missing_total_size_fails_closed`, `test_missing_total_size_optout_warns` |
| BUG-05 (scheduler) | `TestSchedulerContract` — groups advance exactly one `lr_at` position per `step()`; first two steps distinct; final `total_steps` lands on the min-LR floor; resume restores `lr_at(_step)` |
| BUG-06 (validation/resume) | `TestSchedulerValidation` — warmup ≥ total rejected; non-positive base_lrs rejected; resume with `max_steps` < saved warmup no longer crashes (MVP of the constructor-collapse); resume with larger `max_steps` warns + extra steps at floor |
| BUG-07 (estimator) | `TestEstimator` — fp32 grads = 4 B; bf16 = 2 B; explicit override honored |
| POSSIBLE-01 | covered by BUG-06's two resume tests (schedule/loop-bound policy) |
| POSSIBLE-02 | `TestResumeWarnings` — missing CPU RNG warns; saved-CUDA-RNG-with-CUDA-unavailable warns |
| POSSIBLE-03 | `TestChunkedCEChunkSize` — `chunk_size` 0 and negative rejected (both CE implementations) |
| POSSIBLE-04 | `TestValidationSplit` — tiny corpus → clear `ValueError`; minimum viable corpus splits successfully |
| POSSIBLE-05 | `TestNtokParser` — sparse-file fixture with header claiming > 1 GiB rejected **before** any payload read; size mismatch; CRC mismatch; valid fixture parses |
| POSSIBLE-06 | `TestDeltaNetCacheCommit` — overflowing `commit()` leaves `_length` unchanged |
| STRONG-01 | resolved constructively; every DDP matrix test doubles as its guard |
| STRONG-02 | `test_interrupted_save_leaves_no_new_shards` (monkeypatched I/O failure mid-save; only-pre-existing shards survive) |
| STRONG-03 | not fixed by design (documented) |
| HIST-01..04 | branch-only / fixed upstream / docs — documented (no code reintroduced) |

## 3. Full verification matrix

| Area | Command | Result |
|---|---|---|
| Full suite | `pytest tests/ --basetemp=… -q --tb=no` | **460 passed, 3 skipped**, exit 0 |
| New regressions only | `pytest tests/test_v14_regressions.py -q` | 38 passed |
| Resume/75M (CUDA) | `pytest tests/test_training_75m.py -q` | passed |
| Resume-engine regressions | `pytest tests/test_engine_regressions.py -q` | passed |
| Serialization (incl. fail-closed) | `pytest tests/test_serialization.py tests/test_moe_serialization.py -q` | passed |
| DDP 2-proc gloo | `pytest tests/test_v14_regressions.py -k DDP -q` | 6/6 passed |
| Bitwise resume | engine + `test_resume_then_continue_updates_step_counter` | passed (weights/LR/losses consistent) |
| Snapshot-vs-HEAD SOURCE | `diff -rq versions/v1.4.0-candidate/SOURCE neuratal tests README.md pyproject.toml` | identical (byte-for-byte) |
| Checksums | `versions/v1.4.0-candidate/file_manifest.json` (sha256 per SOURCE file) | see manifest; verified twice |

## 4. Explicitly-updated assertions (not weakened)

Two existing tests encode the corrected behavior after deliberate fixes. They
were edited in-place with the old expectation replaced by the new, stricter,
contract; nothing was removed:

- `test_scheduler_resume_applies_resumed_lr` — comment now reads
  "restore LR is `lr_at(4)`" (was `lr_at(3)`); the test still asserts resume
  restores exactly the LR the groups held at save.
- `test_sharded_legacy_index_without_hashes_loads` →
  `test_sharded_legacy_index_without_hashes_rejected_or_warns` — the default
  now fails closed; the opt-out path is asserted to load with a warning.

## 5. Test environment honesty notes

- NTokenizer artifact tests remain env-gated and were **not executed**
  (policy); 3 skips are pre-existing privileges (CPU probe needs CUDA;
  artifact integration; CUDA integration probe).
- Two-process CPU gloo DDP tests run on this machine in ~20 s; they are NOT
  a substitute for multi-GPU NCCL validation (none available).
- One benign third-party warning remains (Triton FP-fusion in a CUDA test).
- The three warnings in the summary are: Triton FP-fusion notice, and two
  intentional `UserWarning`s asserted by the resume-policy tests.
