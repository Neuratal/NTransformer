# v1.4.0-candidate — known limitations (honest)

- Untrained framework; synthetic-data evidence only. No quality claims.
- Two-process CPU gloo DDP is validated (see V1_4_DDP_VALIDATION_REPORT.md).
  **No multi-GPU/NCCL training has been performed or validated anywhere in
  this project** — all DDP verification is gloo/CPU, correctness-oriented.
  Distributed throughput, NCCL behavior, and multi-node training are unproven.
- Cross-version resume (v1.3 checkpoint → v1.4 runner) restores weights and
  optimizer state bit-exactly, but the LR trajectory shifts by one schedule
  position (the v1.3 resume LR was retarded one step; v1.4 is corrected).
  Resume within v1.4 is bit-exact.
- Skipped optimizer steps (fp16 overflow / non-finite gradient norm) do not
  advance the LR schedule; `history` may contain repeated step numbers with a
  monotonic `iteration` field. Runs that skip many steps extend the loop up to
  a hard cap and then abort with `RuntimeError` (rather than hang forever).
- No crash-consistency fault injection; atomic rename is not power-loss proof.
- NTokenizer integration tests skip (40+) in a moved checkout (machine-relative
  NTOKENIZER_ROOT default); integration correctness is not verified in this
  environment. The tokenizer fingerprint below is carried verbatim from the
  v1.3 snapshot (not re-verified).
- NTokenizer's repository has no git history at audit time (reported by the
  v1.3 audit; unchanged).
- Activation memory estimator still carries the documented ±50% uncertainty
  beyond development scale.
- MoE global load balance under DDP is not synchronized (per-rank router
  statistics).
- `find_unused_parameters=True` remains set for MoE (needed for zero-token
  experts); DDP emits a benign "did not find any unused parameters" warning
  on healthy steps when all experts are used.
- Sharded index is not authenticated (STRONG-03 from the v1.3 audit): hashes
  detect accidental corruption, not adversarial index+shard replacement.
- Pre-v1.3 sharded checkpoints (no `shard_hashes`) no longer load by default;
  `verify_shard_hashes=False` loads them unverified with a warning.
