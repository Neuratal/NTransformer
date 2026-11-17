# Changelog

All notable changes to this project are recorded here. Versions are frozen
once released; history is never rewritten.

## [v1.4.0-candidate] — 2026-11-17

Bug-remediation release against the independent v1.3 audit (7 confirmed,
3 STRONG, 6 POSSIBLE, 4 historical findings). Correctness fixes for training,
checkpointing and the scheduler; DDP MoE training now works. No architecture
change. **Not promoted to stable** (registry rule unchanged; v0.4 remains best).

### Fixed
- **HIGH — DDP + factorized-tied-head MoE crashed at the first backward**
  ("Parameter … `output_projection.out_proj.weight` has been marked as ready
  twice"). Root cause: the engine applied the tied LM head *outside* the
  DDP-wrapped forward via raw parameter references, while
  `find_unused_parameters=True` marked that parameter ready (unused) at
  forward end. The DDP path now wraps the forward in an internal
  `_DDPLossModule` that computes hidden-only encoding **and** the chunked LM
  loss inside one wrapped forward, so every reducer-relevant parameter is
  consumed inside the graph. Verified with 2-process gloo (CPU): factored and
  full-width heads × dense and MoE × 2 steps, bitwise parity with the
  single-process reference on identical data, gradient accumulation ×2, and
  checkpoint save/resume after a DDP run. All seven shipped MoE presets are
  affected in v1.3 (crash) and work in v1.4.
- **MEDIUM — fp16 GradScaler skip advanced the LR scheduler anyway.**
  Skipped optimizer steps no longer advance `scheduler.step()`/`global_step`;
  skipped steps are recorded in `history` (`skipped: true`); skip decisions
  are consistent across DDP ranks (identical all-reduced grads). Additionally,
  **bf16/fp32 non-finite gradient norms now refuse the optimizer step** (same
  skip semantics) instead of silently poisoning weights — in torch 2.15,
  `clip_grad_norm_` maps inf norms to inf×0 masks (i.e., NaN grads) that
  AdamW would otherwise absorb.
- **MEDIUM — Sharded loaders opened index-controlled paths (path
  traversal).** Shard filenames in `weight_map` must match
  `^model-\d{5}-of-\d{5}\.safetensors$` and resolve inside the checkpoint
  directory; absolute paths, `..` traversal and non-standard names are
  refused before the file is opened (BUG-03).
- **MEDIUM — Sharded verification failed open.** With
  `verify_shard_hashes=True` (default), a missing/empty `shard_hashes` or a
  missing/non-integer `total_size` now raises instead of silently loading
  unverified; `verify_shard_hashes=False` remains as a documented opt-out and
  now warns loudly (BUG-04).
- **LOW — LR scheduler lagged by one optimizer step** (`lr_at(0)` applied
  twice; whole trajectory retarded; final min-LR never reached).
  `WarmupCosineSchedule.step()` now advances before applying `lr_at(_step)`;
  after `total_steps` steps the groups hold exactly the `min_lr_ratio ×
  base_lr` floor (BUG-05).
- **LOW — `load_state_dict` lacked cross-field validation** (warmup ≥ total
  accepted; zero/negative `base_lrs` accepted). Both are rejected; on resume
  the engine no longer crashes constructing the scheduler with a smaller
  `max_steps` than the checkpoint implies, and warns when the requested bound
  exceeds the checkpointed schedule (extra steps run at the min-LR floor)
  (BUG-06, resolves POSSIBLE-01).
- **LOW — memory estimator charged fp32 gradients at 2 bytes** when
  `config.dtype="float32"`; `grad_dtype_bytes` now defaults to the
  dtype-derived width (4 for fp32) (BUG-07).

### Investigated and closed (no code change)
- STRONG-01: resolved by the BUG-01 fix (all reducer-relevant parameters now
  consumed inside the wrapped forward; residual `find_unused_parameters`
  warning noise documented).
- STRONG-02: interrupted sharded saves now remove shards created by the
  failed call (pre-existing files untouched).
- STRONG-03: sharded index remains unauthenticated **by design** (no
  signing); documented as a residual integrity gap — per-shard SHA-256 now
  fails closed (BUG-04) mitigates accidental corruption.
- POSSIBLE-02: cross-device RNG restore gaps now warn loudly (missing CPU
  RNG, missing/absent CUDA devices, CUDA unavailable).
- POSSIBLE-03: `chunk_size=0`/negative now rejected in both chunked-CE
  implementations (was silently coerced / produced a zero loss).
- POSSIBLE-04: tiny-corpus `train_validation_split` fails with a clear
  message at the split point (was a cryptic crash inside `TokenDataset`).
- POSSIBLE-05: the NTOK parser reads only the 18-byte header, validates
  magic/version/`payload_len`/file size **before** any payload read, and
  drops the unused whole-file read.
- POSSIBLE-06: `DeltaNetCache.commit()` overflow leaves `_length` unchanged.

### Historical findings
- HIST-01 (sliding-window reference attention ignores the window): affects
  only `cand/v0.7-sliding-window` (never merged; branch discontinued). v1.4
  does not carry sliding-window code; documented for future revival.
- HIST-02 (GDN WY-chunkwise cumprod underflow >64): branch-only,
  documented limitation; not in v1.4.
- HIST-03 (SWA default-window bounds crash): fixed upstream in v0.7.1;
  not applicable to v1.4.
- HIST-04 (docs/snapshot version drift): fixed by this release's docs
  updates (README, registry, BUG_REGISTRY).

### Tests
- 38 new regression tests (`tests/test_v14_regressions.py`), including
  2-process gloo DDP validation (dense/MoE × factored/full-width heads,
  gradient accumulation, checkpoint save/resume after a DDP run). Suite:
  **460 passed, 3 skipped** (`python -m pytest tests`).

### Behavior changes (resume compatibility)
- The cosine schedule now lands exactly on the documented curve (one-step
  advance). A v1.3 checkpoint resumed under v1.4 restores `lr_at(step)`
  rather than the v1.3-retarded `lr_at(step-1)` — the first post-resume LR
  differs by one schedule position. Weight-level resume remains bit-exact.
- `train()` returns the unwrapped `NeuratalModel` in every configuration
  (previously the DDP-wrapped module on multi-process paths).
- Skipped optimizer steps (fp16 overflow / non-finite gradient norm) do not
  advance the scheduler; `history` may contain repeated step numbers with a
  monotonic `iteration` field.
- Sharded checkpoints written before v1.3 (no `shard_hashes`) no longer load
  by default (`verify_shard_hashes=False` to opt out with a warning).

### Registry
- v0.4 remains the registry "best" version; v1.4.0-candidate is recorded as a
  candidate only (not promoted).


## [v1.3.0-candidate] — 2026-11-17

Enterprise-audit hardening release (NT-ENGINEERING-2026-FINAL). Correctness
fixes in scheduler/checkpoint integrity; no architecture change.

### Fixed
- **Scheduler state round-trip is now bit-exact.** Scheduler/optimizer
  scalar state is stored as `int64`/`float64` (was `float32`); every resumed
  cosine trajectory previously drifted ~1e-7 from uninterrupted training.
  Verified bitwise at 7 resume points (0, 1, 4, 5, 8, 15, 16), including
  parameters, optimizer moments, scheduler counter, and losses.
- **`WarmupCosineSchedule.load_state_dict` validates malformed state**
  (required keys, integer/finite/range checks, base_lrs length vs param
  groups) — precise `ValueError` instead of `KeyError`/silent misassignment.
- **Sharded checkpoints verify per-shard SHA-256** (stored at save, verified
  at load; default on; `verify_shard_hashes=False` documented for enormous
  checkpoints with out-of-band verification). Catches same-size stale shard
  replacement that byte-count checks cannot. Measured 63 ms per 37 MB model.

### Tests
- 4 new integrity tests; suite: **422 passed, 3 skipped, 1 warning**
  (`python -m pytest tests`).

### Docs
- `docs/scheduler_resume_verification.md` (full experiment),
  `docs/checkpoint_integrity.md`, `docs/bug_registry.md` (v1.3 additions),
  `docs/final_engineering_audit.md` (this mission's report),
  `docs/ultimate_audit/*` retained.

## [v1.2.0-candidate] — 2026-11-17

Engineering-audit release: correctness fixes in the optimizer LR schedule and
checkpoint integrity, one op-level performance improvement, no architectural
change. **Not to be confused with a stable release; stable promotion requires
evidence under the registry rule.**

### Fixed
- **Scheduler LR off-by-ones** (`neuratal/training/optim.py`):
  fresh `WarmupCosineSchedule` ran the first optimizer step at the full base
  LR (warmup never covered step 0); `load_state_dict` never pushed the
  restored schedule into the param groups, so the first post-resume step ran
  at the construction LR (a silent mid-cosine jump on every resume);
  `load_state_dict` now also validates `base_lrs` length and reapplies the
  LR the groups held at save time (`lr_at(step-1)`).
- **Sharded checkpoints** (`neuratal/serialization/sharded.py`): load now
  verifies `metadata.total_size` against the bytes actually loaded (stale /
  replaced shard detection), and reuses the single-file atomic-save helper
  instead of a private copy.
- **RMSNorm under autocast**: the dominant training path (fp32 master weights
  + bf16 autocast) now uses the fused `F.rms_norm` normalize-without-weight
  kernel + explicit fp32 weight multiply (~1.65× faster at the op level,
  ≤1-ulp worst case). Previously it fell back to the manual fp32 path.

### Changed (behavior)
- First-step LR is now `lr_at(0)` = `base/warmup` for `warmup_steps ≥ 2`
  (`warmup_steps ≤ 1` unchanged). Resume now reproduces the exact LR trajectory
  of the interrupted run.

### Added
- Tests: scheduler fresh/resume/mismatch, CUDA RNG byte-exact restore,
  sharded `total_size` integrity, autocast-norm behavior. Suite:
  **418 passed, 3 skipped, 1 warning** (`python -m pytest tests`).
- Docs: `docs/ultimate_audit/` (file inventory, bug registry, performance
  research, rejected experiments, benchmark summary, version history,
  FINAL_AUDIT report), `docs/release_notes_v1.2.0.md`, this changelog.

### Not changed
- Architecture, checkpoint format (`format_version=1`), model semantics,
  NTokenizer integration and identity (fingerprint verified unchanged),
  `versions/` archives.

## [v1.1.0rc1] — 2026-11-17
See `docs/release_readiness.md`. Highlights: MoE gather/scatter-once dispatch;
fused `F.rms_norm` (dtype-match/no-autocast); meta-init skip + lazy presets;
fp16 GradScaler; DDP-MoE `find_unused_parameters`; full-width-head crash fix;
checkpoint fingerprint/index guards; preset honesty rename (`7b`→`5.6b`,
`7b-moe`→`5.7b-moe`); 413 tests; benchmark harness additions.

## [v1.0.0] — 2026-11-16
Training engine, chunked linear cross-entropy, dense presets with measured
counts, local sharded checkpoints, resume/RNG restore, memory estimator.

## [v0.4] — 2026-11-16
Opt-in CUDA-graph decode (batch=1, CUDA). Current registry "best".

## [v0.3] — 2026-11
Original baseline (fused QKV/MLP projections).
