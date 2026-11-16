# Neuratal Transformer — Autonomous Architecture R&D Report

Date: 2026-11-16 · Scope: architecture/software only. **No model was trained;
all measurements use random-weight initialization and synthetic inputs.**
This report is the R&D log; it supplements (does not replace)
`docs/final_report.md`, `docs/optimization_log.md`, and `docs/benchmark_results.md`.

## Loop followed
ANALYZE → R&D design → implement → test → benchmark → reanalyze → promote/reject.

## Version ledger (immutable, see registry/best_version.json)

| Version | Commit | Status | Evidence |
|---|---|---|---|
| v0.2-baseline | c31418e | immutable baseline | 268 passed |
| **v0.3** | e3633be (merged 240e585) | **best** | 276 passed; fp32 prefill win; migration tested |

## Iteration 1 — Fused QKV and fused gate/up projections (v0.3)

**Problem:** forward launches 3 GEMMs for qkv (dev model heavily
launch-bound); 2 for SwiGLU gate/up. cuBLAS handles these as many small GEMMs.
**Change:** one `qkv_proj` GEMM; one `gate_up_proj` GEMM. Mathematically
identical row concatenation. Legacy v0.2 checkpoints migrate transparently at
load (row concat + UserWarning), verified < 1e-5 logits diff.

**Correctness:** full suite green (276); 8 new fused-compat tests (bitwise
fused-vs-split math, legacy roundtrip, migration warning, parameter count
unchanged 9,782,528).

**Benchmarks (3 rounds, alternating baseline/candidate, same machine):**
- fp32 prefill dev scale: T=256 2.19→2.03 ms (−7.4%), T=512 3.55→3.41 ms (−3.9%)
- fp32 prefill ref-50M (B=1, T=512): baseline 11.79/14.78/14.23 vs candidate
  10.50/10.99/11.17 ms → **−22% median (3/3 rounds)**
- T=32/T=128 fp32: within noise
- bf16: within clock-state run variance (no claim)
- decode: 0.61–0.66 ms/tok both (no claim)
- cached/uncached parity: 0.00e+00 (unchanged)

**Decision:** promoted. Preserves all NTokenizer semantics, checkpoint format
v1, parameter count, and every correctness invariant.

## Iteration 5 — RESEARCH PASS (performed; see docs/research_notes.md)

Gathered and judged: FlashAttention/2 (arXiv:2205.14135 / 2307.08691), RMSNorm
(arXiv:1910.07467), RoPE/RoFormer (arXiv:2104.09864), YaRN (arXiv:2309.00071),
SwiGLU (arXiv:2002.05202), GQA (arXiv:2305.13245), PagedAttention/vLLM
(arXiv:2309.06180), Pope et al. (arXiv:2211.05102), speculative decoding
(arXiv:2211.17192), CUDA-graph decode sources (NVIDIA llama.cpp blog;
arXiv:2604.23467, 2611.00495, 2611.14904). Result: decode is launch-bound;
static-shape graphs are the applicable fix; paged KV is the long-term answer.

## Iteration 6 — v0.4: static-shape CUDA-graph decode (PROMOTED, opt-in)

- `experimental_cuda_graphs=True`: one graph capture per generation call
  (window = pow2(prompt+max_new+1)); replay per token.
- Bitwise vs eager (16+ steps fp32/bf16; prefill and per-step logits equal).
- Measured: 1.28–1.42× at short/medium prompts (16–128 / gen 32–64);
  0.94× at prompt=512 (window padding penalty) — documented; default stays
  eager. Suitable for the dev-benchmark workload class.
- 21 new tests; 297 total pass; default path unchanged.

## Iteration 2 — CUDA graphs for the decode step (rejected v0.3-era candidate, superseded by v0.4)

**Hypothesis:** replay the decode step as a CUDA graph to erase ~50 Python
launch ops/token (measured: ~305 µs GPU vs ~620–670 µs wall/token).
**Blocker:** capture requires static shapes. The static-capacity KV cache
advances `start`/`end` offsets per step (baked into any graph), so replay
would write to a frozen offset. Correct replay needs tensor-indexed updates +
full-capacity static SDPA with an additive valid-length mask — O(capacity²)
attention per token at 2048 slots, i.e., permanently ~8× the decode FLOPs for
a tiny model. The vLLM-style answer is paged attention (explicitly out of
scope). Bucketed graphs (11 buckets to 2048) add complexity and still pad.
**Verdict:** rejected. Documented here; decode remains eager. torch.compile
(reduce-overhead) measured: prefill 0.946 ms (vs 1.003 default compile, vs
2.10–2.23 ms eager) — recorded as opt-in prefill-only path, never used by
`generate()`.

## Iteration 3 — Chunked-prefill mask caching (rejected, measured)

`build_lower_right_mask` costs 0.018–0.052 ms/call (measured over 1000 calls);
a cache would save < 0.1% of a chunked-prefill step and add a stateful cache
to attention modules. **Verdict:** rejected — no measurable value.

## Iteration 4 — Remaining knobs (all rejected or irrelevant)

- RMSNorm/RoPE hand-fusion: < 4% of device time; fusion belongs to
  torch.compile (opt-in), not maintenance-heavy manual kernels.
- cuDNN-bf16 priority override: ~0.01 ms/step on 3% of device time; not a
  justified global default.
- KVCache reuse across `generate()` calls: per-call isolation is a correctness
  boundary; saving the 4 MiB one-off memset is wrong trade.
- Logits vocab GEMV (fp32): ~48 µs/token decode — memory-bandwidth bound at
  262k×32 reads; cannot be reduced without changing the math (int8/etc. —
  rejected precision changes).
- Positions arange reuse, .item() elimination: already done/verified earlier.

## What was measured to establish the plateau

CUDA (sm_89, cu132): prefill T=128 ≈ 2.10–2.23 ms eager / 0.95–1.00 ms with
opt-in compile; decode 0.62–0.67 ms/tok eager; long-context T=2048 prefill
12.87 ms; peak forward memory fp32 1080 MiB @ T=512 (B=1); checkpoint save
13 ms / load 72 ms / file 37.3 MiB; cached≡uncached 0.00e+00; 50M reference
forward 5.1–6.3 ms (bf16, T=512). NTokenizer compatibility unchanged; artifact
SHA-256 untouched; zero training anywhere.

## Limitations & next requirements (if this loop continues)

1. The remaining wall-clock gap in decode is Python launch overhead; the
   principled fix is a paged-KV/static-shape design — an architectural
   milestone, not a knob.
2. bf16 measurements on this laptop fluctuate ±10–20% with clock state; any
   bf16-verdict would need pinned clocks (nvidia-smi -lgc) or a quieter
   machine.
3. torch.compile decode path is degenerate on this stack; prefill compile is
   opt-in only.
4. Ref-50M measurements are config-only (no training decision was made or
   implied).
5. Nothing about model quality can be assessed; this loop only improved
   architecture/software.

## Reproduction

- `scripts/make_version_snapshot.py`, `registry/best_version.json`
- Raw logs: `versions/v0.2-baseline/benchmark_results/`, `versions/v0.3/benchmark_results/ablate/`
- Benchmarks: `benchmarks/benchmark_model.py`, `benchmark_kv_cache.py`,
  `benchmark_attention.py`, `benchmark_memory.py`, `benchmark_end_to_end.py`,
  `benchmark_tokenizer.py`, `benchmark_environment.py`
- Correctness gate: `python -m pytest tests/` → 276 passed, 2 skipped.

## Context vs public models (honest, evidence-labeled)

Nothing proprietary about GPT/Claude internals is claimed or implied; comparisons
use only public architecture concepts. Evidence labels: M = measured here,
P = public/general knowledge, E = estimate, U = unavailable.

| Axis | Neuratal v0.3 (M) | Public reference points |
|---|---|---|
| Vocab | 262,144 (M) | GPT-2 50,257 (P); LLaMA-3 128,256 (P) — ours is larger; embedding-factorized |
| Param count | 9.78M dev / 47.85M ref-sketch (M) | 50M-class frontier was small-custom territory in 2024-era literature (P) |
| Prefill B=1 T=128 | 2.10-2.23 ms fp32 / ~1.1 ms bf16 (M) | Not directly comparable (hardware/build differ) |
| Decode | 0.62-0.67 ms/tok (M) | Paged-attention servers achieve higher batch efficiency (P) — we are eager |
| Cache equivalence | 0.00e+00 bitwise (M) | Correctness criteria all published decoders satisfy (P) |
| Quality/BPC/MMLU | U (untrained) | Frontier models score high (P); we claim nothing |

Our deliberate differentiations: full 262k vocab factorized tied embeddings,
strict tokenizer identity pinning (SHA-256 manifests), public reproducible
benchmarks, and zero approximations. We are a foundation, not a competitor to
frontier chat models; nothing else should be inferred.

## Iteration 7 — bf16 adversarial-numerics probe (candidate rejected)

Scale-blowup attention probe (fp32 vs bf16, ×1…×100,000): drift stays ~0.4–0.5%
relative, zero NaN/Inf, bf16 SDPA bit-identical to bf16 math reference.
Training-time bf16 gradient-growth literature does not transfer to inference.
**No fp32_attention flag; rejection documented in research_notes.md §14 + probe log.**

## Iteration 8 — plateau re-verification (explicit, with all axes)

Axes now measured for the promoted v0.4: correctness (cached≡uncached 0.00e+00),
prefill/decode latency across T∈[32,2048], batch scaling B∈{1,2,4,8} (new),
tok/s, peak VRAM (measured) & RSS (measured, +181 MB), KV-cache bytes (exact)
and scaling, checkpoint size/save/load (37.3 MiB / 13 ms / 72 ms), long-context
scaling (12.87 ms @ T=2048), fp32/bf16/fp16 (fp16 forward+backward suite green),
CPU vs CUDA (CPU 20-60x slower; correctness identical), SDPA kernel selection
per dtype (measured), torch.compile (opt-in, reduce-overhead 0.946 ms prefill),
CUDA-graph decode (v0.4 opt-in, bitwise-equal, 1.28-1.42x short/medium),
checkpoint roundtrip bitwise. Research now covers all required topics
(attention, positional, normalization, residuals §11, MLP, embeddings §12,
initialization §13, KV cache, long context, numerical stability §14, kernels,
inference optimization).

**Plateau confirmed: no further justified improvements at this scale/hardware.
Remaining levers are architectural milestones (paged KV/serving; training-time
decisions) — documented, not actionable now.**

## Closure decision (v0.5 not shipped — with reasons)

After the complete research pass, every candidate either (a) materialized,
tested, and promoted (v0.3 fused GEMMs; v0.4 opt-in CUDA-graph decode) or
(b) was rejected on measurement (torch.compile for the decode path, cudnn
priority, hand-fusion, mask caching, fp32_attention adversarial probe,
epsilon/norm-placement — unmeasurable without training; residual/ReZero/
DeepNet-style depth scaling — training-phase boundary; speculative decoding —
no draft model possible under boundaries).

No remaining change meets the promotion bar: reproducible gain on at least one
axis with all correctness gates green, at this model scale on this hardware.
Further improvements require architectural milestones (paged KV/serving
redesign; training-time decisions) that are explicitly out of scope.
**Decision: v0.5 is not shipped; the loop is closed for this scope. v0.4
remains best. Re-open only with (i) training infrastructure, (ii) paged-KV
serving scope, or (iii) a materially different GPU/clock regime.**


## Iteration 9 — Gated DeltaNet prototype (v0.5-candidate; NOT promoted)

Research: KDA (Kimi K3), GDN (Qwen3.8-Flash-Next), Gated DeltaNet paper
arXiv:2412.06464 (Eq. 10). Implemented opt-in `attention_type="gated_delta"`:
L2-normalized q/k, sigmoid α/β gates, SiLU output gate, per-head fp32 state,
DeltaNetCache (O(1) memory). 22 new tests pass (causality, cached≡uncached,
backward, roundtrip, stability).

Measured (fp32, CUDA): prefill T=128/512 GDN 21.1/81.9 ms vs dense 2.2/5.8 ms;
decode at 64–1024 ctx GDN 1.02–1.10 vs dense 0.76–0.80 ms/token (loss at every
point); state memory 128 KiB constant vs dense KV 4 MiB @2048 (32× win);
params 10,048,912 vs 9,782,528. Default remains v0.4 GQA (`best_version`).
Quality unverifiable without training (formula change). Frozen as
`versions/v0.5-candidate` with registry entry (promoted=false) and raw A/B JSON.
Promotion would require a training-time quality result or a long-context,
memory-bound target where the 32× state advantage dominates the latency loss.
Iteration 10 may revisit with the WY-chunkwise parallel form + training infra.

## Iteration 10 — MLA/latent-KV (v0.6-candidate; NOT promoted, branch only)

Implemented: latent KV cache + expansion per step; two-slice RoPE (q new
positions / k full range); SDPA dispatch; cached-vs-uncached parity verified.
Measured: decode +12.5-17.1% slower at ctx 64/512/1024 (vs dense GQA);
prefill 10-16% slower; KV bytes/token/layer 4x lower (256 vs 1024 B);
params 9.72M vs 9.78M. NOT promoted; rejected for adoption on this hardware.
Branch cand/v0.6-mla + versions/v0.6-candidate; master untouched at v0.4.

## Iteration 11 — SWA + bf16 GEMV (both rejected on measurement)

SWA implemented + tested + benchmarked (3 seeds): wash at T=1024; 20-40% slower
T=2048 (mask disables flash path; attention share 7.5%). Rejected; branch
cand/v0.7-sliding-window; frozen as versions/v0.7-candidate. bf16 GEMV check:
2.5x micro speedup, zero end-to-end decode gain (launch-bound) — not adopted.

## Iteration 9b — v0.6 WY chunkwise GDN (arXiv:2412.06464 §3.3) — IMPLEMENTED, NOT PROMOTED

Delivered on branch cand/v0.6-wy-chunkwise. Faithful Eq.6-9 closed form:
T = (I + tril(diag(beta)(gamma-ratio-weighted K K^T)))^-1 diag(beta); W = T K;
U = T V; gamma-weighted causal mask for the intra-chunk term (derived to match
the recurrent oracle exactly: max abs diff 1e-07 @ T<=16 fp32; <5e-04 through
T=512). 5-13x prefill speedup vs sequential GDN (23.6->3.13 ms @ T=128 fp32;
348.3->34.1 ms @ T=2048). STILL slower than dense GQA everywhere at this
scale: fp32 prefill 3.13/8.91/15.69/34.08 vs GQA 1.53/5.28/8.90/15.42;
decode 1.09-1.26 vs 0.78-0.90 ms/tok (bf16: same story). Wins only on constant
state memory. NOT promoted; archived as v0.6.1-candidate (naming collision with
MLA v0.6-candidate documented); best stays v0.4. All raw JSON preserved.
