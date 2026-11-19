# Benchmarking NTransformer

Status: **matches `neuratal` 1.4.0 (2026-11-19).**

## Where the numbers come from

Measured on the development machine (RTX 4060 Laptop GPU, torch 2.15.0+cu132,
Python 3.14.4; 45 W mobile GPU — ±15% run-to-run variance is normal). Seed 42
everywhere. See `docs/benchmark_methodology.md`.

| Measurement | Baseline | Latest | Notes |
|---|---|---|---|
| Dense prefill fp32 T=512 | 4.121 ms | **3.926 ms** (-5%) | fused RMSNorm |
| Dense prefill bf16 T=512 | 3.032 ms | **2.725 ms** (-10%) | T=1024: 5.43→5.14 ms |
| MoE fwd (B2, T=128) | 1.927 ms | **1.611 ms** (-16%) | gather/scatter-once |
| Import `neuratal` | 13.4 s | **1.3-1.4 s** | meta-init skip + lazy presets |
| Chunked CE fwd+bwd (1024 tok) | 88.0 ms | 88.0 ms | 1.29 GB vs 4.29 GB std CE |

**No accuracy benchmark exists** — the model is untrained; claims are
infrastructure throughput only.

Docs: `docs/benchmark_methodology.md`, `docs/benchmark_results.md`,
per-release JSONs in `benchmarks/results/`.
