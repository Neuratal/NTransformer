# Benchmark Results (measured)

All numbers below were measured on **2026-11-16** on this machine (RTX 4060
Laptop GPU, sm_89; i7-12700H ×20; torch 2.15.0+cu132; python 3.14.4;
safetensors 0.8.0; NVIDIA driver 595.99.02/CUDA 13.2) with the scripts in
`benchmarks/`. Seed = 42 everywhere. Every table cites its JSON artifact in
`benchmarks/results/`. Rows fluctuate ±10–20% across runs on this laptop;
medians are reported. **Nothing here is an estimate of untested hardware.**

Legend: M = measured execution. T = theoretical calculation from shapes.

## 1. NTokenizer (M) — artifact load kept separate from steady state

Artifact: `ntokenizer-0.3-256k.ntok` (v3, SHA-256 `1f0d4ad7…6d9`, 23.4 MB).
Load (bindings init + parse + CRC + SHA-256 + cross-check): **0.574 s**.

| Corpus | Bytes | IDs | B/T | Encode | Encode tok/s | Decode | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_en ("Hello, world!") | 13 | 3 | 4.33 | 4.33 MiB/s | 1.05M | 5.19 MiB/s | 1.26M |
| short_tr | 16 | 5 | 3.20 | 4.95 MiB/s | 1.62M | 6.20 MiB/s | 2.03M |
| medium_en ×4 | 848 | 136 | 6.24 | 10.77 MiB/s | 1.81M | 193.20 MiB/s | 32.5M |
| medium_tr ×4 | 880 | 132 | 6.67 | 11.19 MiB/s | 1.76M | 183.12 MiB/s | 28.8M |
| code_python ×6 | 1,602 | 462 | 3.47 | 12.68 MiB/s | 3.83M | 197.71 MiB/s | 59.8M |
| multilingual ×30 | 5,700 | 1,230 | 4.63 | 13.74 MiB/s | 3.11M | 273.05 MiB/s | 61.8M |
| numbers ×2000 | 8,890 | 2,885 | 3.08 | 15.06 MiB/s | 5.12M | 216.61 MiB/s | 73.7M |
| batch(32 × medium_en) | 27,136 | — | — | 10.62 MiB/s | 1.79M | — | — |

Artifact: `tokenizer_20261116T145028Z.json`. Note: NTokenizer is
single-threaded per call; these are per-thread numbers. Input sizes are small
(13–8,890 B); microsecond-level latencies at the short end are dominated by
FFI call overhead, not throughput.

## 2. NTransformer dev preset — CUDA fp32 (M)

Config: V=262144, D=256, L=2, H=4, H_kv=2, F=32, I=640, ctx 2048; 9,782,528
params (37.3 MiB fp32). `model_20261116T145740Z.json` (latest):

| Shape | Median | P95 | tok/s | Logits (T) |
|---|---:|---:|---:|---:|
| B=1 T=32 | 1.71 ms | 1.95 ms | 18,704 | 32 MiB |
| B=1 T=128 | 2.23 ms | 2.68 ms | 57,304 | 128 MiB |
| B=1 T=256 | 2.95 ms | 4.33 ms | 86,712 | 256 MiB |
| B=1 T=512 | 4.04 ms | 4.90 ms | 126,785 | 512 MiB |
| B=1 T=1024 | 6.44 ms | 7.44 ms | 159,033 | 1,024 MiB |

CUDA bf16, same shapes (`model_20261116T145735Z.json`):

| Shape | Median | P95 | tok/s | Logits (T) |
|---|---:|---:|---:|---:|
| B=1 T=32 | 0.93 ms | 1.15 ms | 34,569 | 16 MiB |
| B=1 T=128 | 1.14 ms | 1.27 ms | 112,635 | 64 MiB |
| B=1 T=256 | 1.44 ms | 1.55 ms | 178,221 | 128 MiB |
| B=1 T=512 | 2.03 ms | 2.18 ms | 251,902 | 256 MiB |
| B=1 T=1024 | 4.46 ms | 5.88 ms | 229,545 | 512 MiB |

CPU fp32 (`model_20261116T145050Z.json`): B=1 T=32 8.5 ms · T=128 25.2 ms ·
T=256 57.3 ms. CPU is far behind GPU; usable for correctness, not serving.

Baseline→optimized (host-check removal, see docs/optimization_log.md):
fp32 T=128 median moved 2.33 → 2.10–2.23 ms across runs; decode 0.88 → 0.67 ms/tok
(below). bf16 within run-to-run spread.

## 3. KV cache (M) — fp32, fresh cache per repeat

`kv_cache_20261116T145555Z.json` (CUDA):

| seq | Uncached fwd | Cached prefill | Decode | Cached vs uncached Δlogit |
|---:|---:|---:|---:|---:|
| 64 | 1.08 ms | 1.17 ms | 0.67 ms/tok | 0.00e+00 |
| 128 | 1.39 ms | 1.45 ms | 0.67 ms/tok | 0.00e+00 |
| 256 | 1.94 ms | 2.03 ms | 0.67 ms/tok | 0.00e+00 |
| 512 | 3.22 ms | 3.33 ms | 0.68 ms/tok | 0.00e+00 |

Cache buffer (exact, T): 2 layers × 2 (K,V) × 1 batch × 2 kv_heads × 2048 × 64
× 4 B = **4,194,304 B (4 MiB)**. Cached equivalence is asserted inside the
benchmark (< 1e-5, fp32; measured 0.00e+00).

## 4. SDPA backends (M) — shape B=1, H=4, H_kv=2, T=256, d_h=64

`attention_20261116T145110Z.json`:

| Backend | fp32 | bf16 | fp16 |
|---|---|---|---|
| flash | unavailable (no fp32 kernel) | 0.039 ms | **0.022 ms** |
| efficient | 0.062 ms (Δ vs fp64 1.7e-06) | 0.041 ms | 0.027 ms |
| cudnn | unavailable (no fp32 kernel) | **0.023 ms** | 0.023 ms |
| math | 0.219 ms (Δ 4.2e-07) | 0.150 ms | 0.148 ms |

Default dispatch on this build picks MemEfficient for fp32 and flash for bf16
(verified via profiler); cudnn is faster than both for bf16 at this shape but
the global default is left as PyTorch set it — overriding backend priority is
shape-sensitive and was not justified. Δ vs fp64: fp32 ≤ 4.2e-06; bf16 ≈ 5.0e-03
(dtype rounding); fp16 ≈ 6.2e-04. bf16/fp16 error is dtype-limited, not
kernel-limited (all three cutlass/flash backends agree to the last printed
digit). flash/cudnn cannot serve fp32 on this build ("No available kernel").

## 5. End-to-end pipeline (M) — fp32, 5 repeats

`end_to_end_20261116T145649Z.json` (post-optimization):

| Prompt | Prompt tokens | Gen len | Encode | Prefill | Gen ms/tok | Decode |
|---|---:|---:|---:|---:|---:|---:|
| turkish | 15 | 16/32/64 | 0.022–0.043 ms | 0.97–0.99 ms | 0.89–0.91 | 0.005–0.006 ms |
| english | 16 | 16/32/64 | 0.022–0.023 ms | 0.95–1.00 ms | 0.87–0.91 | 0.005–0.006 ms |
| code | 25 | 16/32/64 | 0.021–0.030 ms | 1.03–1.06 ms | 0.88–0.90 | 0.005–0.006 ms |

Model decode dominates end-to-end latency; tokenizer costs are sub-millis at
these prompt sizes. Untrained weights produce gibberish (e.g., mixed scripts);
this is a pipeline validation, not a language-quality measurement.

## 6. Memory (M, with budget guard; skips would be recorded)

`memory_20261116T145156Z.json`; budget 4608 MiB — **all cells passed, none
skipped**:

| Cell | Peak allocated | Peak reserved |
|---|---:|---:|
| dev preset fp32, T=64 / T=512 | 183.5 / 1080.0 MiB | (recorded in JSON) |
| dev preset bf16, T=64 / T=512 | 96.8 / 545.1 MiB | |
| dev preset fp16, T=64 / T=512 | 96.8 / 545.1 MiB | |
| 50M ref fp32 (B=1), T=64 / T=128 | 373.8 / 502.0 MiB | |
| 50M ref bf16 (B=1), T=64 / T=128 | 196.5 / 260.5 MiB | |

Dev preset accounting (fp32): parameters 9,782,528 → **39,130,112 B**;
embedding matrix alone 33,554,432 B (V×F = 262144×32); KV cache at capacity
2048 = 4,194,304 B; logits at B=1 T=512 fp32 = 536,870,912 B (**theoretical**).
Theoretical logits-only table (fp32): B=1 T=1024 = 1.0 GiB; B=2 T=512 = 1.0 GiB;
B=8 T=512 = 4.0 GiB — reproduced in the JSON `theoretical` block; never
presented as measured.

## 7. What was NOT measured / cannot be claimed

- Nothing about CUDA graphs, torch.compile decode behavior (measured as a
  failure), or other GPUs/drivers.
- No training-phase costs (optimizer, gradients at full scale); fwd+bwd B=1
  T=64 peak measured earlier ≈ 366 MiB (7 MB gradients for 9.8M params is
  accounted, but nobody should extrapolate training memory from this).
- NTokenizer understood only through its public API + documented .ntok format;
  no Rust internals were inspected beyond the binding surfaces and docs.
- Disk I/O of checkpoint save/load (seconds-scale, unmeasured).
- Thread scaling (batch encode used 32 texts sequentially; rayon parallelism
  untested here).

## 8. Batch-size scaling (M) — added after v0.4 review

`model_20261116T160432Z.json` (fp32) / `model_20261116T160442Z.json` (bf16):

| B | T=64 | T=128 | tok/s |
|---:|---:|---:|---:|
| 1 | 1.21 ms | 1.56 ms | 82k |
| 2 | 1.66 ms | 2.21 ms | 116k |
| 4 | 3.00 ms | 4.57 ms | 112k |
| 8 | 3.53 ms | 6.31 ms | 162k |

bf16 (T=128): B=2 109k · B=4 230k · B=8 187k tok/s. Small-batch bf16 is noisy
(clock-state ±30%). B=8 T=128 fp32 logits = 1 GiB (theoretical from shape).

## 9. Peak host RAM (M)

CPU dev-preset forward (T=64) + 16-token generate: **+181 MB peak RSS** vs
post-import baseline. No meaningful swap on 14.6 GiB host.
