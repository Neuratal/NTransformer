# Benchmark Summary — v1.1.0rc1 → v1.2.0-candidate

All numbers measured on this machine unless labeled otherwise. Seed 42, warmup
and repeats stated per run; medians reported. This GPU (45 W laptop) swings
±10–20 % between runs; treat < 10 % deltas as unproven. Nothing below is a
production or cross-hardware claim.

## Attribution protocol

Only interleaved A/B in the same process is treated as attributable (R1/R6;
v1.1's R4). Everything else is context: either re-run comparisons against a
frozen worktree at 196a613 (v1.1) or repeat medians.

## A. Attributable results

| Experiment | Measurement | Outcome |
|---|---|---|
| R1 RMSNorm autocast (v1.2) | 0.0627 → **0.0379 ms** @ [4,512,512], diff 0.0 | adopted (op-level; end-to-end within noise, honest) |
| R2 `_grouped_mm` experts | 0.615 → 0.601 ms | rejected |
| R3 `torch.compile(MoEFFN)` | 15.07 vs 14.90 ms | rejected |
| R4 MoE dispatch gather/scatter-once (v1.1) | B2T128 **1.927→1.611 ms**; others within noise | adopted (v1.1) |
| R5 cuDNN attention | no reproducible winner (3 rounds) | default kept |
| R6 fused `F.rms_norm` matched-dtype/no-autocast (v1.1) | fp32 op 4.0×; e2e fp32 T128/512 −11/−5 % | adopted (v1.1) |
| R7 CE workspace charged fp32 in estimator | docs-only | adopted (v1.1) |

## B. Re-run vs committed v1.1 baselines (clock-dominated, NOT attributable)

fp32 prefill T=128/512/1024: 1.942 / 3.273 / 6.093 ms (v1.1 committed: 1.854 /
3.926 / 7.675; deltas mixed sign, clock-dominated). bf16 T=128/512/1024:
0.972 / 1.971 / 4.344 ms (bf16 T=512 spread across 4 runs: 2.725 → 2.210 →
1.971 → 1.994 → 1.990). **No v1.2 performance win is claimed from these.**

## C. Unchanged, re-verified

| Item | Value |
|---|---|
| Chunked CE (1024 tokens) fwd+bwd | 88.0 ms, 1,286 MiB peak |
| Standard CE (1024 tokens) | 62.4 ms, 4,291 MiB peak |
| Chunked-CE gradient parity | rel L2 = 1.03e-06 |
| Dense train-step (2×256, bf16 autocast) | ≈ 40 ms (v1.2 norm change within noise) |
| Peak VRAM (train-step benchmarks) | unchanged |
| `import neuratal` | ~1.3 s (v1.1) |
| Test suite | **418 passed, 3 skipped, 1 warning** (28–46 s, machine/cache-dependent) |

## D. Environment

RTX 4060 Laptop (sm_89), torch 2.15.0+cu132, CUDA 13.2, Python 3.14.4,
i7-12700H, 8 GB, driver 595.99.02. Full fingerprints in
`benchmarks/results/environment_*.json`. NTokenizer fingerprint verified:
`1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9`.
