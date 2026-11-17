# v1.2.0-candidate — changes vs v1.1.0rc1

Ultimate engineering audit (see docs/ultimate_audit/FINAL_AUDIT.md in the repo).

Fixed (correctness)
- LR scheduler off-by-ones: fresh warmup never covered step 0 (first step ran
  at full base LR); resume re-applied construction LR instead of the saved
  schedule (silent mid-cosine jump every resume); base_lrs length validated.
- Sharded checkpoints: metadata.total_size verified on load; duplicated
  atomic-save helper deduplicated (reuses serialization.checkpoint's).

Performance
- RMSNorm under autocast (fp32 master weights + bf16 autocast): fused
  normalize-without-weight + explicit fp32 weight multiply (~1.65x op;
  end-to-end within noise; <=1 ulp deviation class as accepted bf16 path).

Added
- tests: scheduler fresh/resume/mismatch, CUDA RNG byte-exact restore,
  sharded byte-count integrity. Suite: 418 passed, 3 skipped, 1 warning.

Unchanged
- architecture, config schema, checkpoint format (format_version=1),
  NTokenizer integration, versions/ archives, registry "best" (v0.4).
