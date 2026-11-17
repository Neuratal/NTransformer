# Release Notes — v1.2.0-candidate

**Status:** release candidate (engineering audit). Not a stable release; not
promoted over v1.1.0rc1 or the registry's current best (v0.4). Synthetic data
only; NTokenizer untouched (fingerprint verified unchanged).

## What this release is

A deep, file-by-file engineering audit and correctness hardening of the
v1.1.0rc1 codebase. Headline: the LR schedule and resume path had two silent
defects — the first optimizer step ran at the full base LR, and a resumed run
restarted at the construction LR. Both are fixed with exact-restoration
semantics. Sharded checkpoints now verify byte counts on load. One op-level
performance improvement (RMSNorm under autocast) ships with honest
documentation that end-to-end effect is within noise.

## Correctness fixes (behavioral changes)

- **Warmup LR applies from step 0.** A fresh `WarmupCosineSchedule` now sets
  param groups to `lr_at(0)`. For `warmup_steps ≥ 2` this changes the first
  step's LR from `base` to `base/warmup` (more faithful to a warmup schedule);
  `warmup_steps ≤ 1` is unchanged.
- **Resume restores the exact saved LR.** `load_state_dict` reapplies
  `lr_at(step-1)` — the LR the param groups held at save time — instead of
  leaving construction LRs in place. Mid-cosine resumes no longer jump LR.
- **`base_lrs` length validation** on scheduler restore (corrupt checkpoint).
- **Sharded checkpoints:** `metadata.total_size` is compared against the
  bytes actually loaded; mismatch → `ValueError`.
- **RMSNorm under autocast:** fused normalize-without-weight + explicit fp32
  weight multiply. ~1.65× faster op; ≤1-ulp worst-case deviation (0.0 on
  common inputs); output dtype unchanged. End-to-end: within noise.

## Integrity / reliability

- New tests: fresh/resume/mismatch scheduler LR, CUDA RNG byte-exact restore,
  sharded `total_size` integrity, autocast RMSNorm behavior.
- Suite: **418 passed, 3 skipped, 1 warning** (exact commands in README).
- New docs: `docs/ultimate_audit/` (inventory, bug registry, performance
  research, rejected experiments, version history, benchmark summary,
  FINAL_AUDIT report) and this file; root `CHANGELOG.md` added.
- Registry: `registry/best_version.json` records the candidate (not promoted).

## Not in this release (explicit)

- No architectural change; checkpoint format unchanged (`format_version=1`).
- No stable-release promotion; v0.4 remains the registry's best under its
  promotion rule until reproducible evidence supports a move.
- No multi-GPU validation (DDP MoE correctness fix is untested on real
  multi-GPU; single-process gloo smoke test only).
- No real pretraining; no corpus handling.
- No distributed checkpointing (local sharded format only, by design).
- No claim of crash-proof checkpoints (atomic rename is not power-loss
  fault injection).

## Upgrade notes

- If you checkpoint with `WarmupCosineSchedule`, resume behavior now matches
  the interrupted run exactly — no action needed; if anything, long-run
  warmups are now correctly gentle at step 0.
- Parameter group LR is now applied at scheduler construction (fixes an
  unintended full-LR first step) — no config change.
- No changes required to checkpoints, configs, or the tokenizer.

## Known limitations (see FINAL_AUDIT.md §N)

Untrained framework; synthetic-data-only evidence; single-GPU 8 GB machine;
SDPA defaults retained; activation memory estimator is ±50 % at best;
every wall-clock number on this GPU is clock-state sensitive (±10–20 %).

Verdict: **research-ready. Development-ready. Not release-ready / not
production-ready** (see FINAL_AUDIT §P).
