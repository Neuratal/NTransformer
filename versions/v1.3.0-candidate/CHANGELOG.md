# v1.3.0-candidate — changes vs v1.2.0-candidate

Enterprise-audit hardening (NT-ENGINEERING-2026-FINAL). Not promoted.

Fixed
- Scheduler state round-trip is now bit-exact (int64/float64 storage; was
  float32); uninterrupted == interrupted training verified bitwise at 7
  resume points (parameters, optimizer moments, losses, scheduler counter).
- WarmupCosineSchedule.load_state_dict validates required keys, types,
  ranges, and base_lrs length against the optimizer param groups.
- Sharded checkpoints: per-shard SHA-256 recorded at save, verified at load
  by default (verify_shard_hashes=False opt-out documented).

Added
- docs/final_engineering_audit.md, scheduler_resume_verification.md,
  checkpoint_integrity.md, bug_registry.md (v1.3 additions); 4 new tests;
  versions/v1.3.0-candidate freeze. Suite: 422 passed.
