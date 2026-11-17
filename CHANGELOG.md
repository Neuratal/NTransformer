# Changelog

All notable changes to this project are recorded here. Versions are frozen
once released; history is never rewritten.

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
