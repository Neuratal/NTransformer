# v1.4.0 — Known limitations

- Untrained; synthetic data only; no trained weights are distributed.
- No benchmarks across hardware beyond the dev machine.
- No cross-framework speed comparison is claimed anywhere.
- Long context (16K/32K) is experimental and memory-constrained.
- Multi-GPU/NCCL not verified (single GPU; 2-proc CPU gloo verified).
- NTokenizer integration verified at load/inference/training time with fingerprint checks; library integration test suite may skip on machines without the tokenizer artifact.
- No quality claims of any kind.
