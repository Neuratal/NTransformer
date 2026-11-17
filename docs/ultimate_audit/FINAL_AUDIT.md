# NTransformer — Ultimate Audit Report (v1.2.0-candidate)

Date: 2026-11-17 · Branch: `feature/v1.2-audit-2026-11` (from `feature/final-release-2026-11` @ 5d13af9) · `master` untouched.
Machine: RTX 4060 Laptop GPU (8 GB, sm_89, 45 W), i7-12700H, Python 3.14.4, torch 2.15.0+cu132, CUDA 13.2, safetensors 0.8.0.
NTokenizer: **verified unchanged** — sha256 `1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9` (checked at audit start and end; also re-printed by `benchmark_environment.py`).

---

## A. Executive Summary

**Initial state (v1.1.0rc1):** working, well-tested (413 passed) framework
with a correct training/checkpoint core, honest docs, and a known noise-prone
laptop GPU. No trained model; synthetic data only.

**Final state (v1.2.0-candidate):** two silent LR-schedule defects eliminated
(first step ran at full base LR; resume restarted at the construction LR),
sharded-checkpoint integrity hardened (byte-count verification + helper
dedup), RMSNorm under autocast de-composited into a fused kernel path
(~1.65× op-level, ≤1-ulp), 5 new tests (418 total green), full file-level
documentation of every relevant source file, and an honest verdict.

**Most important discoveries**
1. The training engine silently resumed at the construction LR, not the
   restored schedule — every interrupted run jumped LR mid-cosine.
2. Fresh warmup schedules never covered step 0.
3. Autocast RMSNorm (the dominant training-norm path) could not use the fused
   kernel and paid a composite fallback — fixed with a 1.65×-faster fused path.
4. Sharded checkpoints accepted replaced-with-stale-content shards with
   identical keys/shapes/dtypes.

**Remaining risks** (see §N): multi-GPU untested; no trained model/quality
evidence; estimator is hand-wavy above dev scale; this GPU is too noisy to
tune below ~10 %; no fault-injection campaign for crash consistency.

## B. Complete File Inventory

See `docs/ultimate_audit/file_inventory.md` — **every relevant file is
listed with purpose, key symbols, risks and status**: 79 Python files
(30,862 lines incl. tests), plus packaging, registry, scripts, docs.
Frozen archives under `versions/` are excluded from modification by project
policy and were not touched.

## C. File-by-File Findings

Each of the **core** files was read in full. Findings per file with
confirmation depth and status:

### Core package

- **`neuratal/config.py`** (404 lines) — single validated dataclass; every
  field validated in `__post_init__`; nothing silently corrected;
  serialization-safe resolution. No defect found; pinned by 40+ config tests.
- **`neuratal/presets.py`** — meta-device count construction; v1.1 already
  fixed init skip (4 s → 1.2 s for the largest preset) and honest naming.
  All counts re-verified exact in tests. No defect found.
- **`neuratal/estimate.py`** — honest ESTIMATE labels; fp32 CE workspace
  correction is in (v1.1); `zero*` documented as accounting models only.
  Known gap: activation coefficient uncalibrated above dev scale.
- **`neuratal/model/model.py`** — root module; v1.1 meta-init skip;
  `forward_hidden(output_hidden_only=True, return_aux=True)` DDP-safe entry;
  parameter accounting exact. No defect found.
- **`neuratal/model/block.py`** — correct pre-norm residual order; no defect.
- **`neuratal/model/attention.py`** — SDPA regime dispatch (`is_causal` only
  for q_len == kv_len; unmasked single-token; lower-right mask for chunked
  prefill); GQA `enable_gqa` with explicit warned fallback; no defect.
  Re-benchmarked backends: no reproducible winner → default kept.
- **`neuratal/model/mlp.py`** — fused gate+up GEMM; no defect.
- **`neuratal/model/moe.py`** — v1.1 gather/scatter-once dispatch; v1.2 lazy
  `sorted_experts` (capacity path only); fp32 router; **pathological routing
  test suite added in this audit cycle**. No defect; see §G.
- **`neuratal/model/normalization.py`** — v1.1 fused path; v1.2 autocast
  normalize-without-weight fused path (R1). Confirmed: autocast promoted the
  old fused attempt to a numerically *worse* composite (0.0156); fixed.
- **`neuratal/model/positional.py`** — RoPE rotate-half, absolute positions
  for cacheable decoding; fp32 tables; CPU-only bounds checks. No defect.
- **`neuratal/model/output.py`** — tied head; exact chunked projection; tied
  matrix serialized once. No defect.
- **`neuratal/model/cache.py`** — preallocated KV cache; commit protocol;
  overflow hard error. No defect.
- **`neuratal/model/generation.py`** — minimal greedy/sampling helper;
  EOS semantics; cached/uncached parity tested. O(T²) `cat` accepted.
- **`neuratal/model/gated_delta_net.py`** — experimental opt-in; latency
  regression measured (v0.5); retained, not promoted.
- **`neuratal/model/graph_decode.py`** — experimental opt-in; bitwise vs
  eager on tested build; B=1 CUDA only.
- **`neuratal/training/engine.py`** — see §D (LR defects), §H. Also:
  `tokens_seen` counts input ids (fine), `_moe_log_fields` logging-time only,
  zero unexpected host syncs (verified `.item()` audit).
- **`neuratal/training/optim.py`** — §D findings B-2022-1/2/3. Fixed.
- **`neuratal/training/checkpoint.py`** — fingerprint/architecture/index
  guards (v1.1); CUDA RNG restore **verified byte-exact** (new test); GradScaler
  state round-trip. No defect.
- **`neuratal/training/chunked_linear_ce.py`** — dead bookkeeping removed;
  gradient parity rel L2 = 1.03e-06; V=262,144 workspace fp32 in estimator.
- **`neuratal/training/chunked_loss.py`** — legacy path retained;
  `use_reentrant=True` deprecation noted (open item O-1).
- **`neuratal/training/data.py`** — greedy packing; synthetic-only; no defect.
- **`neuratal/serialization/checkpoint.py`** — strict single-file format;
  no defect.
- **`neuratal/serialization/sharded.py`** — §C findings B-2022-4/5. Fixed.
- **`neuratal/serialization/` / `neuratal/*/__init__.py`** — public API
  surfaces; `neuratal/__init__.py` keeps v1.1 lazy presets (verified
  round-trip import).
- **`neuratal/integration/*`** — CRC/SHA-256/specials/vocab cross-checks;
  never imported by the core; sidecar verification. No defect.

### Tests / benchmarks / docs
See file_inventory.md §3–§6. All 41 test modules read; suite green
(**418 passed, 3 skipped**, skip list below). Benchmarks re-used; results
committed per repo convention with clock-noise caveats.

## D. Bug Registry

See `docs/ultimate_audit/bug_registry.md`. Summary:

| ID | Severity | File | Status |
|---|---|---|---|
| B-2022-1 | high | training/optim.py | FIXED `27c7f78` |
| B-2022-2 | medium | training/optim.py | FIXED `27c7f78` |
| B-2022-3 | medium | training/optim.py | FIXED `27c7f78` |
| B-2022-4 | medium | serialization/sharded.py | FIXED `eeed9b4` |
| B-2022-5 | low | serialization/sharded.py | FIXED `eeed9b4` |
| B-2022-6 | low | model/moe.py | FIXED (v1.1 `d9bb7c1`) |
| B-2021-1 | critical | training/engine.py (full-width head) | FIXED (v1.1) |
| B-2021-2 | high | training/engine.py (fp16 GradScaler) | FIXED (v1.1) |
| B-2021-3 | high | training/engine.py (DDP MoE) | FIXED (v1.1, untested multi-GPU) |
| B-2021-4 | high | training/checkpoint.py (fingerprint/index) | FIXED (v1.1) |
| B-2021-5 | low | model/moe.py (H2D scalar) | FIXED (v1.1) |
| — | — | model/normalization.py (autocast composite 0.0156 error) | FIXED (v1.2 `65830db`) |

Open items: O-1…O-6 (see bug registry): deprecated `use_reentrant`;
shuffle-resume parity; multi-GPU MoE untested; crash-consistency not
fault-injected; activation estimator uncalibrated; GPU clock noise.

## E. Performance Research

See `docs/ultimate_audit/performance_research.md`. Headlines:

- ADOPTED: fused normalize-without-weight RMSNorm under autocast (R1, 1.65× op,
  end-to-end within noise); fused matched-dtype/no-autocast RMSNorm (R6, v1.1);
  MoE gather/scatter-once dispatch (R4, v1.1, −16 % on one cell, within noise
  elsewhere).
- REJECTED: `torch._grouped_mm` (exact-match but noise, private API);
  `torch.compile(MoEFFN)` (noise); cuDNN attention forcing (non-reproducible);
  fixed-capacity bmm padding; pre-allocated generation buffer; FP16 router.
- Not reproducible: "cuDNN faster than FLASH" (earlier note — clock noise).
- Benchmark claims are only attributable via interleaved A/B; single-run
  deltas <10 % on this GPU are discarded.

## F. Dense Report

Architecture (pre-norm decoder, RMSNorm, RoPE, GQA SDPA, SwiGLU, factorized
tied embeddings) is sound and was **not** changed. v1.2 changes the norm
kernel path only. Correctness: cached/uncached parity, generation, chunked-CE
gradient parity (rel L2 = 1.03e-06), checkpoint bitwise round-trip, 75M-class
CUDA chain — all green. Performance: see §E/R6 and §L. Memory: unchanged
(optimizer-step peak ≈1.15 GiB dev scale; chunked-CE 1.29 GB @1024 tokens
vs 4.29 GB standard CE). Latency honesty: fp32 T=1024 within noise;
**no end-to-end Dense win is claimed for v1.2 beyond the norm op.**

## G. MoE Report

- Router: fp32 logits/softmax/top-k + normalized gates (unchanged);
  `argsort(stable)`; one `.tolist()` host sync; empty segments skipped.
- Dispatch: gather-once/scatter-once (v1.1); lazy `sorted_experts` (v1.2).
- Experts: 8 independent SwiGLU (default); GEMMs dominate; grouped-GEMM
  rejected on measurement.
- Losses: aux (`coef·E·Σ f_e·P_e`) and z-loss formulas validated by tests;
  statistics from pre-capacity assignments (documented convention); masked
  padding contributes exactly zero output and zero assignments.
- Pathological suite (`tests/test_moe_pathological.py`): all-to-one-expert,
  starved experts, saturated router, capacity conservation — deterministic
  forced-routing fixtures.
- Checkpoint/resume: dense↔MoE arch-aware rejection; fingerprint guard;
  optimizer-index guard; per-expert grad finiteness verified.
- Performance: module fwd B2T128 1.927→1.611 ms (attributable); remainder
  within noise; MoE peak memory ≈ dense at dev scale.
- Open: multi-GPU DDP untested; per-rank router stats; no expert parallelism.

## H. Training Report

- Optimizer: AdamW (decoupled weight decay ≥2-D params only); pluggable.
- Scheduler: warmup→cosine; **fresh runs now start at the warmup LR (was
  full base)**; resume restores the exact saved LR (`lr_at(step-1)`) —
  verified by dedicated tests; corrupt `base_lrs` rejected.
- Precision: bf16 autocast (fp32 master weights) — default; fp16 autocast +
  GradScaler with unscale-before-clip and checkpointed scaler state; scaler
  restore verified.
- Accumulation → optimizer steps; clipping on accumulated grads (correct order).
- Checkpoint/resume: exact RNG (byte-equal), exact optimizer tensors,
  scheduler step, counters; full-width head, MoE, fp16, 75M-class chain — all
  verified. Corrupt files rejected; atomic writes; sharded format verified
  (byte count); local only — **not distributed checkpointing**.
- Data: greedy packing; synthetic-only in-repo; shuffle-resume parity
  documented (O-2).

## I. Memory Report

Measured (this GPU): dev full train step ≈1.15–1.20 GiB; chunked CE
1.29 GB @1024 tokens (standard: 4.29 GB); MoE bwd peak 162–228 MiB;
KV cache preallocated B×L×T×2×H_kv×hd×dtype (estimator formula).
Estimated (labeled): params/grads/AdamW m+v/master/activations/CE-workspace
(fp32)/comm buffers; activation coefficient uncalibrated above dev scale
(±50 %); `zero*/fsdp` accounting models only; TP/PP/EP rejected.
Overflows none observed in measured peaks; peak-reserved ≥ peak-allocated by
cuBLAS/workspace caching (documented, not a defect).

## J. Preset Report

Measured counts (meta device, exact): 50m 47,853,056 · 75m 77,088,896 ·
100m 98,096,896 · 500m 496,760,320 · 1b 1,074,104,320 · 3b 2,960,139,008 ·
5.6b 5,571,350,528 (renamed from 7b) · 13b 11,915,686,912 (−8.3 %, documented) ·
30b 30,235,631,616 · 70b 68,488,011,776 · 100b 93,843,515,392 ·
200b 209,420,042,240 · 500b 490,777,300,992. MoE: small 402,301,056
(act 123,609,216) · 1b-moe 1,074,300,928 (659,064,832) · 5.7b-moe
5,673,062,400 (2,351,173,632, renamed from 7b-moe) · 30b-moe 29,400,112,128
(8,260,819,968) · 100b-moe 99,909,378,048 (16,157,515,776) ·
200b-moe 198,474,149,888 (35,399,610,368) · 500b-moe 483,293,937,664
(77,419,528,192). No preset above ~100M has executed a forward pass on this
machine; nothing here is trainability evidence.

## K. Test Report

```
$ .venv/bin/python -m pytest tests 2>&1 | tail -1
418 passed, 3 skipped, 1 warning in 44.28s (v1.2.0 final) / 28.29s (rerun)
```
Skips (exact): `test_cache.py:179` (CUDA device-mismatch probe),
`test_engine.py:111` (artifact-covered), `test_ntokenizer_integration.py:209`
(needs CUDA in that probe). 1 warning (third-party Triton FP-fusion).
Commands: `python -m pytest tests -q` (all tests); `-rs` for skip reasons;
`-p no:warnings` for noise. New tests this audit: scheduler fresh/resume/
mismatch, CUDA RNG byte-exact, sharded byte-count, autocast RMSNorm
(behavioral coverage via suite), end-to-end suite count 413 → 418.

## L. Benchmark Report

See `docs/ultimate_audit/benchmark_summary.md`. Headlines: attributable wins
are op-level only (R1, R6) or one MoE cell (R4); everything else is within
noise; CE comparison unchanged; no production claims; raw JSONs committed.

## M. Rejected Experiments

See `docs/ultimate_audit/rejected_experiments.md`: RX-1…RX-8 (compile,
grouped_mm, autocast weighted-fused composite, cuDNN forcing, fixed-padding
bmm, CE "improvement", pre-allocated generation buffer, clock-state artifacts).
History preserved; nothing deleted.

## N. Remaining Limitations

1. Untrained; synthetic data only; no quality evidence of any kind.
2. Multi-GPU untested (single-process gloo smoke only); DDP-MoE fix untested
   on hardware; MoE router stats per-rank; no ZeRO/FSDP implementation.
3. Estimator uncalibrated beyond dev scale; do not use for procurement.
4. Crash-consistency of checkpoint writes not fault-injected; atomic rename is
   not a power-loss proof.
5. GPU clock noise dominates sub-10 % measurements — expect ±15–20 %.
6. SDPA default FLASH only for bf16/fp16; fp32 uses mem-efficient cutlass FMHA.
7. `gated_delta` experimental; `graph_decode` experimental bitwise-opt-in only.
8. fp16 path CUDA-only; CPU training untested beyond synthetic chains.
9. No fused linear-CE (Liger-style) — chunked CE is the mechanism, ~35 % slower
   than standard CE by design.
10. `versions/` archives are frozen snapshots; they contain known-outdated
    numbers by design (they are history, not current claims).

## O. Version Registry

Canonical: `registry/best_version.json`. Current: **best = v0.4** (promoted,
unchanged). Candidates appended: v1.1.0-rc1 (release candidate; not promoted),
**v1.2.0-candidate** (this audit; not promoted). Failed experiments RX-1…RX-8
and v0.5–v0.7 archives remain registered with evidence. Snapshot:
`versions/v1.2.0-candidate/` (commit reference inside).

## P. Repository Status & Final Verdict

- Branch: `feature/v1.2-audit-2026-11` · commits this pass: `27c7f78`
  (scheduler), `eeed9b4` (sharded), `65830db` (RMSNorm autocast), + audit docs.
- Working tree: clean (see `git status`); `master` untouched (`14f6fe0`).
- NTokenizer fingerprint: **verified unchanged** (pre/post audit + benchmark
  environment run).
- Stable dense regression: full suite green; one documented ≤1-ulp numerics
  path change (RMSNorm autocast only).

**Verdict — research-ready / development-ready. NOT release-ready, NOT**
**production-ready** (no trained model, no multi-GPU evidence, no fault**
**injection, noisy hardware, no quality claims). v1.2.0-candidate is**
**accepted as a candidate — not as stable.** Promotion, should anyone pursue**
**it, requires: reproducible multi-seed benchmarks beyond the noise band, a**
**real multi-GPU run, and crash-consistency testing of the checkpoint path.**

*Refusing to claim more is the deliverable. Evidence is in the committed**
**JSONs, tests, and this report.*
