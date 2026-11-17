# Performance Research — v1.1.0rc1 → v1.2.0-candidate

Method: every claim below is either (a) an interleaved A/B measurement in the
same process/session, or (b) a repeat-median with stated n and machine noise
band, or (c) marked theoretical. No number is presented without provenance.
Machine: RTX 4060 Laptop (sm_89, 8 GB, 45 W cap; SM clocks observed 2565–3105
MHz under load variation — ±10–20 % per-run spread is normal and is why
single-run wins < 10 % are not claimed).

## 1. Interleaved measurements (attributable)

### R1 — RMSNorm under autocast (ADOPTED, v1.2)
- Setting: dev preset fp32 master weights, bf16 autocast, CUDA.
- Baselines: (i) manual fp32 path (v1.1 fallback for mismatch dtypes):
  **0.0627 ms** @ [4, 512, 512]; (ii) attempted fused weighted kernel with
  mismatched dtypes → cannot dispatch / composite lands at **0.0156** max-diff
  error (worse numerics — rejected outright).
- Candidate: `F.rms_norm(x, None, eps)` (fp32 under autocast) + fp32 weight
  multiply: **0.0379 ms** — **1.65× faster**; max |Δ| vs manual = **0.0** on
  common tensors, ≤ 1 ulp (bf16) worst case; output dtype preserved
  (`bf16`) as before.
- End-to-end train-step effect: 40.01 → 40.53 ms across interleaved medians —
  **within noise** (the step is dominated by the V=262k CE path). Reported
  honestly as an op-level improvement only; not promoted as end-to-end.

### R2 — `torch._grouped_mm` for expert GEMMs (REJECTED)
- Setting: realistic imbalanced gate_up+down pipeline, bf16 CUDA.
- Loop vs grouped: **0.615 → 0.601 ms** (−2 %, noise); isolated uniform-GEMM
  cases showed 0.379 vs 0.387 ms (mixed signs). Exact-match outputs.
- Private aten API + strict layout/dtype constraints. Not adopted.

### R3 — `torch.compile(MoEFFN)` (REJECTED)
- 14.90 → 15.07 ms fwd+bwd — within noise; compilation fragility. Not adopted.

### R4 — MoE gather-once/scatter-once dispatch (ADOPTED, v1.1)
- B=2 T=128: **1.927 → 1.611 ms** (−16 % — the one cell beyond the noise band,
  reproduced in three after-runs); other cells within noise.
- Byte volume identical; kernel launches strictly fewer; zero H2D scalars.

### R5 — SDPA backend re-measurement (REJECTED CHANGE)
- 3 interleaved rounds, bf16 GQA T=512: flash 0.95/0.45/0.94 ms; cuDNN
  0.93/0.63/0.87; default 0.29/0.75/0.86 — no reproducible ordering; the
  earlier "cuDNN faster" observation does not replicate. Default kept.

### R6 — fused `F.rms_norm` (v1.1, dtype-match/no-autocast)
- fp32 op: **0.0581 → 0.0146 ms**; bf16 matched-dtype: ~1 ulp win; autocast
  and mismatched-dtype paths unchanged. End-to-end fp32 prefill T=128/512
  2.092→1.854, 4.121→3.926 ms; bf16 T=512 3.032→2.725 ms; others within noise.

### R7 — `fused CE` workspace fp32 accounting (estimator fix)
- Chunked-CE workspace charged at fp32 in `estimate()` (the kernel upcasts);
  notes now state ESTIMATE-vs-measured explicitly. No runtime change.

## 2. Fresh before/after, v1.1.0rc1 → v1.2.0-candidate (non-attributable, clock-dominated)

`benchmark_model.py` re-runs after idle read systematically "faster" than the
v1.1 runs (SM clocks 2565/3105 MHz): fp32 T=128/512/1024 1.942/3.273/6.093 ms;
bf16 T=128/512/1024 0.972/1.971/4.344 ms. Runs 2–3 of bf16 T=512: 2.210→1.994
→1.990 ms. **These deltas are clock-state artifacts, not engineering wins, and
are not claimed as such.** The committed JSONs are kept as raw measurements;
R1/R2/R6 are the only attributable wins.

## 3. Measured, unchanged

- Chunked linear CE vs standard CE (1024 tokens): **88.0 ms / 1.29 GB** vs
  **62.4 ms / 4.29 GB** — unchanged; parity rel L2 = 1.03e-06.
- KV cache / generation: no changes (preallocated, commit protocol); decode is
  launch-bound at this scale.
- MoE module memory ≈ dense at dev scale (162–228 MiB); gather/cat adds
  ~2×[A×H] transient vs per-expert index_select — bounded and intentional.
- Optimizer-step (small dev MoE vs dense): 28.1 vs 21.7 ms (+30 %), consistent
  with 2× active expert FLOPs; not a regression.
- `import neuratal` 13.4 → ~1.3 s (v1.1: meta-init skip + lazy presets) — holds.

## 4. Judgment calls (when not to optimize)

- **KV-cache per-step `torch.cat`:** O(T²) copies, but T ≤ few thousand and
  launch-bound dominates; preallocation changes API semantics for no
  measurable win. Not done.
- **Attention mask caching** (`build_lower_right_mask`): only on the 1 < q_len
  < kv_len chunked-prefill path; caching adds state for zero hot benefit.
- **Expert weight as single [E, D, I] tensor + bmm:** measured no win
  (grouped_mm≈loop), changes parameterization & checkpoint keys — rejected.
- **`argsort` to int32:** indices must stay int64 for indexing; int32 saving
  < 1 µs at A≈2048 — not worth the edge-case risk.
- **FP16 router GEMM:** rejected — routing precision must not silently degrade.
- **Bitwise fused RMSNorm under autocast:** unattainable; documented ≤1 ulp.
