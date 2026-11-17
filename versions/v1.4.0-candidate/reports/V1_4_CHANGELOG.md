# NTransformer v1.4.0-candidate — Changelog

## Summary

Bug-remediation release closing every confirmed independent-audit finding against
`7d6bf9466171d2b7ab9ff2364890fa8ed1fb0dd9` (v1.3.0-candidate). Multi-process DDP
training now works (see V1_4_DDP_VALIDATION_REPORT.md); checkpoint integrity fails
closed; the LR schedule matches its documented contract; 38 new regression tests.
**Not promoted to stable; v0.4 remains the registry's best version.**

## What changed (by subsystem)

### Training / optimizer / scheduler
- `WarmupCosineSchedule.step()` advances to the **next** `lr_at` position; the
  duplicate `lr_at(0)` at construction+first step is gone; after `total_steps`
  steps the groups hold exactly the `min_lr_ratio × base_lr` floor (BUG-05).
- `load_state_dict`: `warmup_steps >= total_steps` rejected; `base_lrs` must be
  finite **and strictly positive** (BUG-06).
- On resume, the checkpointed schedule (`warmup_steps`, `total_steps`) always
  wins; the loop bound honors the requested `max_steps`/`cfg.max_steps`; a
  `UserWarning` is issued on mismatch (extra steps run at the min-LR floor).
  Constructor collapse for `max_steps < saved warmup` eliminated (BUG-06).
- fp16 GradScaler overflow no longer advances the LR schedule / `global_step`.
  bf16/fp32 non-finite gradient norms now **refuse the optimizer step** (loud;
  previously NaN weights were absorbed silently).
- Skipped optimizer steps are recorded in `history` (`skipped: true`, monotonic
  `iteration` field added to all entries).
- Engine loop is `while global_step < total_steps` with a hard iteration cap
  (raises `RuntimeError` after excessive consecutive skips rather than hanging).

### Serialization / checkpoints
- Sharded load: shard filenames must match `^model-\d{5}-of-\d{5}\.safetensors$`,
  resolve inside the checkpoint directory, and be regular files (BUG-03).
- Sharded load: `verify_shard_hashes=True` (default) now **fails closed** —
  raises for absent/empty `shard_hashes` or missing/non-integer `total_size`;
  `verify_shard_hashes=False` remains as a documented opt-out and warns loudly
  (BUG-04).
- Interrupted `save_pretrained_sharded` now removes only shards it created this
  call; pre-existing files are never touched (STRONG-02).
- Resume (`resume_training_checkpoint`) warns loudly on RNG-restore gaps: missing
  CPU RNG, CUDA RNG missing for current devices, or CUDA unavailable with a
  CUDA-authored checkpoint (POSSIBLE-02).

### Model / MoE / attention / generation
- `DeltaNetCache.commit()` overflow now leaves `_length` unchanged (POSSIBLE-06).
- Residual DDP fact: **multi-process training was broken for factorized-head
  MoE at v1.3** — now fixed by construction (`_DDPLossModule` puts the LM head
  inside the wrapped forward). See V1_4_DDP_VALIDATION_REPORT.md (BUG-01).

### Data pipeline
- `train_validation_split`: clear up-front error for undersized corpora;
  correct boundary clamp so both sides always hold ≥ 1 sequence (POSSIBLE-04).

### Cross-entropy
- `chunk_size` must be ≥ 1 in both chunked-CE implementations — previously
  `0` silently defaulted and negative values produced **a zero loss with zero
  gradients** (POSSIBLE-03).

### Integration / parsing
- NTOK parser reads only the 18-byte header, validates magic/version/
  `payload_len`/total size **before** any payload read; payload read is
  bounded by the declared length; unused whole-file `.read_bytes()` dropped
  (POSSIBLE-05).

### Config / estimator
- Memory estimator charges fp32 gradients 4 bytes (`grad_dtype_bytes` now
  dtype-derived; explicit override honored) (BUG-07).

## Behavior-change notes
- Resume of a v1.3 checkpoint restores weights/optimizer/counters bit-exactly
  but the LR trajectory is shifted one schedule position (v1.3's restore point
  was one step retarded). Resume within v1.4 is bit-exact.
- `train()` returns the unwrapped `NeuratalModel` in every configuration.
- Sharded checkpoints written before v1.3 no longer load by default.
- `history` entries gained `iteration`; duplicate step numbers can appear on
  skipped iterations.

## Docs / registry
- README, CHANGELOG, BUG_REGISTRY rewritten for v1.4 (HIST-04 drift fixed).
- v1.3 and all historical snapshots are untouched; `versions/v1.4.0-candidate/`
  added (SOURCE + manifest + checksums + this report set).
- Registry: `best` remains `v0.4`; v1.4.0-candidate recorded as candidate only.

## Suite & evidence
- **460 passed, 3 skipped** (was 422). 38 new regression tests.
- DDP: 2-process CPU/gloo validation matrix (see V1_4_DDP_VALIDATION_REPORT.md).
- Performance: no regression beyond measurement noise (V1_4_PERFORMANCE_COMPARISON.md).
