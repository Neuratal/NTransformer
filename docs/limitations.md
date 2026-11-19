# NTransformer — Known Limitations & Honest Scope

Status: **matches `neuratal` 1.4.0 (2026-11-19).**

## What this package is not

- **Not a pretrained model.** No weights are distributed; the development
  preset produces gibberish.
- **Not production-ready for scale.** Nothing above ~100M has been run on
  hardware. All presets are configuration/counts validation.
- **Not verified on multi-GPU.** DDP is verified on 2-proc CPU `gloo` only.
  No multi-GPU/NCCL training has ever been performed here.

## Technical limitations

- **Untrained.** No corpus has been used. The 40M pilot (val loss
  10.84 → 6.24) is synthetic; it means nothing about model quality.
- **Long-context is experimental.** 16K/32K requires `gradient_checkpointing=True`
  and is for correctness, not throughput. Do not expect 32K throughput.
- **32K inference** fits in memory with headroom; decode still always does
  KV-cache reads — memory scales with context.
- **MoE is a reference implementation.** Python loop over experts; no kernel
  fusion; no capacity-aware CUDA kernel; no all-to-all. Dispatch optimizations
  are constant-factor, not asymptotic.
- **Chunked CE is ~35% slower** than standard CE (the point is memory, not speed).
- **Sharded checkpoints are local**, not distributed checkpointing.
- **DDP is on CPU `gloo`.** MoE DDP global router statistics are per-rank.

## NTokenizer boundary

NTokenizer is an independent project (github.com/Neuratal/neuratal-tokenizer,
CC-BY-SA-4.0). This package consumes it as a fixed artifact and does not
re-license or re-implement it. Integration is full and fingerprint-verified.

## NTokenizer integration is fully complete

The package is designed to operate **as the language model for NTokenizer
v0.3-polished**. Compatibility is not partial: it is the sole supported
tokenizer for this package, with contract-level checks on every operation
(loading, inference, training, resume).

## Benchmarks

All benchmarks are **development-machine numbers**. No cross-hardware
comparison exists; none is claimed. See `docs/benchmarking.md`.
