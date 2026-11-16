# LIMITATIONS — v0.3

Inherits everything from v0.2-baseline. Additionally:
- Checkpoint key layout is fused; v0.2 checkpoints require the migration loader
  (transparent, warned). Downgrade to v0.2 requires manual key splitting.
- bf16 and decode show no measurable benefit on this machine — the fusion is a
  GEMM-efficiency/kernel-count optimization, not a latency miracle at tiny scale.
- Migration path untested against other unknown-era checkpoints (only v0.2 → v0.3
  roundtrip is tested).
