# v1.4.0-candidate — changes vs v1.3.0-candidate

Bug-remediation release closing every confirmed independent-audit finding
against v1.3 (`7d6bf9466171d2b7ab9ff2364890fa8ed1fb0dd9`). **Not promoted.**

See the root `CHANGELOG.md` for the full narrative, `BUG_REGISTRY.md` for the
complete finding-by-finding resolution table, and `reports/` for the audit
trace.

### Highlights
- **HIGH — DDP + factorized-tied-head MoE trained again.** Root-caused to the
  LM head being consumed outside the DDP-wrapped forward while
  `find_unused_parameters=True` marked `output_projection.out_proj.weight`
  ready(unused) at forward end. Fix: internal `_DDPLossModule` computes the
  hidden-only encoding and the chunked LM loss inside one wrapped forward.
  Verified: 2-process gloo (CPU) parity (bitwise vs single-process on
  identical data), gradient accumulation ×2, checkpoint save/resume after a
  DDP run. All seven shipped MoE presets (all factorized heads) are affected
  in v1.3 and work in v1.4.
- **MEDIUM — fp16 GradScaler skip no longer advances the LR schedule**; and
  bf16/fp32 non-finite gradient norms now refuse the optimizer step (was:
  silent NaN weight poisoning — `clip_grad_norm_` maps inf to NaN masks).
- **MEDIUM — sharded loader path hardened** (traversal rejected) and **
  verification fails closed** (`verify_shard_hashes=True` now raises on a
  hash-less index or missing `total_size` instead of silently skipping).
- **LOW — LR scheduler one-step retard removed** (`lr_at(0)` double-applied;
  min-LR floor never reached); resume restores `lr_at(_step)` exactly.
- **LOW — scheduler state validation completed** (warmup < total_steps;
  positive base_lrs); resume-with-smaller-`max_steps` no longer crashes;
  schedule/loop-bound mismatch now warns.
- **LOW — estimator fp32 gradient width fixed** (4 B, was 2 B).
- All 6 POSSIBLE findings fixed (RNG restore warnings, chunk_size >= 1,
  tiny-corpus split error, NTOK header-first parsing, DeltaNetCache overflow).
- Both actionable STRONG findings resolved; STRONG-03 documented as a design
  limitation (index unauthenticated; mitigated fail-closed).

### Suite
- **460 passed, 3 skipped** (+38 new regression tests incl. a 2-proc gloo
  DDP matrix). v1.3 was 422 passed.

### Behavior changes
- `train()` returns the unwrapped `NeuratalModel` in every configuration.
- Resume of a v1.3 checkpoint under v1.4 restores weights bit-exactly but the
  LR trajectory shifts by one schedule position (v1.3's LR was retarded one
  step). Resume within v1.4 is bit-exact.
- Skipped steps are recorded in history with a monotonic `iteration` field.
- Pre-v1.3 sharded indexes no longer load by default (opt-out documented).
