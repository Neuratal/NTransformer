# v1.3.0-candidate — known limitations (honest)

- Untrained framework; synthetic-data evidence only. No quality claims.
- Multi-GPU DDP untested (single-process gloo smoke only).
- Scheduler step-0 LR is now lr_at(0) (warmup) for warmup >= 2.
- Resume LR trajectory is bit-exact only in runs checkpointed by v1.3.0+
  (float64 storage requires a post-v1.3 checkpoint for exactness).
- No crash-consistency fault injection; atomic rename is not power-loss proof.
- NTokenizer integration tests skip (40+) in a moved checkout (machine-relative
  NTOKENIZER_ROOT default); 3 skips in the canonical checkout.
- NTokenizer's repository has no git history at audit time (reported; see
  ../NTokenizer/docs/tokenizer_engineering_audit.md).
