# NTransformer — Performance Audit Report

Date: 2026-11-17 · GPU: NVIDIA GeForce RTX 4060 Laptop (sm_89, 8 GB VRAM, 45 W cap; clocks 210 MHz idle → 2565 MHz under load; temp 41 → 46 °C)
Protocol: warmup 10, repeats 25, medians reported; **all deltas < 10 % are considered noise on this machine**; nothing herein is a production claim.
Environment: torch 2.15.0+cu132, CUDA 13.2, Python 3.14.4, driver 595.99.02.

## 1. Measured benchmarks (this machine, same protocol)

### Dense prefill — `benchmark_model.py` (dev preset, B=1)

| Seq | fp32 median | bf16 median | fp32 tok/s | bf16 tok/s |
|---:|---:|---:|---:|---:|
| 32 | 1.412 ms | 1.289 ms | 22,659 | 24,817 |
| 128 | 1.810 ms | 1.470 ms | 70,711 | 87,090 |
| 256 | 2.411 ms | 1.850 ms | 106,169 | 138,387 |
| 512 | 3.601 ms | 2.365 ms | 142,171 | 216,493 |
| 1024 | 6.083 ms | 4.179 ms | 168,342 | 245,035 |

Artifacts: `benchmarks/results/model_20261117T094907Z.json` (fp32), `model_20261117T094915Z.json` (bf16).

### MoE module-level (H=512, 8 experts, top-2, expert_hidden=1408)

| Shape | dense fwd | MoE fwd | MoE fwd+bwd | dense fwd+bwd |
|---|---:|---:|---:|---:|
| B=2 T=128 | 0.192 ms | 0.890 ms | 2.458 ms | 0.606 ms |
| B=2 T=512 | 0.657 ms | 2.153 ms | 5.565 ms | 2.141 ms |
| B=4 T=128 | 0.315 ms | 1.375 ms | 3.345 ms | 1.018 ms |
| B=4 T=512 | 1.258 ms | 3.385 ms | 10.216 ms | 4.190 ms |

MoE ≈ 3.8–4.6× slower than dense at these tiny-GEMM shapes (2× active expert FLOPs + dispatch launch overhead; expected at dev scale). Artifact: `benchmarks/results/moe_20261117T094*.json`.

### Optimizer-step / training throughput (bf16, autocast, B=2/T128)

| Config | median | mean | min |
|---|---:|---:|---:|
| dense dev | **23.2 ms** | 23.4 | 23.1 |
| small-MoE (hidden 256, 8 experts) | **24.5 ms** | 24.5 | 24.3 |

### Chunked linear cross-entropy vs standard CE (fp32, V=262,144, F=64)

| tokens | chunked (recompute) | standard | chunked alloc | standard alloc |
|---:|---:|---:|---:|---:|
| 512 | 38.4 ms | 26.8 ms | 1,284 MiB | 2,242 MiB |
| 1024 | 76.2 ms | 53.2 ms | 1,286 MiB | 4,291 MiB |

Gradient parity rel L2 = **1.03e-06** (fp32 accumulate). Chunked CE is ~35–44 % slower and allocates ~3.3× less — the intended memory enabler for 262K-vocab training, not a speed tool. Artifact: `chunked_ce_20261117T094931Z.json`.

### RMSNorm op-level ([4, 512, 512], fp32, no autocast)

manual 0.0407 ms vs fused `F.rms_norm` **0.0145 ms** (≈ 2.8×); under autocast the v1.2 fused normalize-without-weight path is used.

### Checkpoint I/O (75M model, 617 MB)

single-file save **0.071 s** / load **0.514 s**; sharded save **0.241 s** / hash-verified load **0.688 s**.

### Peak VRAM (B2/T256, bf16 autocast, optimizer included)

| Config | peak allocated |
|---|---:|
| dense dev preset | **1,035 MiB** |
| tiny MoE (hidden 256, 8 experts) | **1,083 MiB** |

## 2. Hardware envelope (what 8 GB × 45 W actually supports)

- ✅ Dense dev-preset train steps, 75M dense chain (2-step validated)
- ✅ Small MoE train steps
- ⚠️ Nothing above ~100M with full forward/backward; **no preset was benchmarked for trainability**; presets are config/count validations only
- ❌ Multi-GPU untested; ❌ 1B+ forward/backward not attempted (8 GB)

## 3. What actually moved the needle

- Fused RMSNorm (op-level 2.8–4×; e2e within noise)
- MoE gather/scatter-once dispatch (B2T128 −16 % module fwd vs v1.0 baseline)
- Lazy preset import (`import neuratal` 13.4 s → ~1.3 s)
- fp16 GradScaler (correctness, not perf)
- OOM-avoidant chunked CE (enabler; memory win, latency cost)
- Sharded hash-verified checkpoints (+~34 % load time; integrity win)

## 4. Noise & measurement caveats

- GPU clock state swings ±15–20 % between runs (observed 2565 MHz vs max 3105 MHz); benchmarks recorded with warmup and repeats — **no sub-10 % delta is claimed**.
- CE timings dominated by `[N, 262144]` GEMMs (cuBLAS SGEMM).
- The 45 W cap means sustained runs throttle; all short benchmarks are bursty.

## 5. Bottlenecks (honest)

1. **Vocabulary GEMM** (262,144 × F) — dominates device time in fp32.
2. **Kernel launch overhead** at dev scale (attention ~4–20 %).
3. **fp32 vs bf16** — bf16 prefill is ~1.7–2× faster; fp32 kept for stability.
4. **MoE dispatch** — constant-factor; GEMMs already near memory-bound optimum.

## 6. No scaling claims

Nothing here measures 1B–500B behavior, distributed training, or multi-GPU. The presets are parameter counts. A config existing or counting is **not** evidence of trainability, quality, or release readiness.
