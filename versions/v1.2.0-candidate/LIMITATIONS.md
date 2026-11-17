# v1.2.0-candidate — known limitations (honest)

- Untrained framework; synthetic-data evidence only. No quality claims.
- Multi-GPU DDP untested (single-process gloo smoke test only); the MoE
  find_unused_parameters fix is untested on real multi-GPU hardware.
- Scheduler behavior change: first-step LR is now lr_at(0) (base/warmup)
  for warmup >= 2 (warmup <= 1 unchanged); resume reproduces the saved LR
  trajectory exactly.
- Crash-consistency of checkpoints not fault-injection tested.
- Activation memory estimator uncalibrated above dev scale.
- End-to-end performance on this 45 W GPU is clock-state sensitive (±10-20%).
- bf16/fp16 only for FLASH SDPA; fp32 uses mem-efficient path.
- This is a release candidate, not a stable release; not promoted in the
  registry (v0.4 remains best).
