# Research Notes — Neuratal Transformer (R&D Loop)

Date: 2026-11-16. Each entry records a source, its claim, and a measured-applicability
judgment for our architecture. Nothing below is copied into the codebase without
a tested, benchmarked gate. Evidence labels: P = public knowledge, M = measured
on this machine, E = estimate, U = unavailable.

## 1. FlashAttention (Dao, Fu, Ermon, Rudra, Ré — NeurIPS 2022)
- Source: arXiv:2205.14135 — "Fast and Memory-Efficient Exact Attention with IO-Awareness"
- Claim: IO-aware exact attention reduces HBM traffic via tiling + recomputation;
  O(N) memory vs O(N²); training speedups ~2–4x; enables longer context.
- Judgment: **Adopt via SDPA (done).** Our sm_89/torch 2.15 build runs flash for
  bf16/fp16 and MemEff (fmha_cutlassF) for fp32 — flash has no fp32 kernel on
  this build (verified). No custom kernels in scope (brief); PyTorch dispatches.
- Status: shipped (v0.1). Evidence: profiler shows flash_fwd_kernel (bf16),
  fmha_cutlassF_f32 (MemEff fp32). Benchmarks per attention_results.json.

## 2. FlashAttention-2 (Dao 2023)
- Source: arXiv:2307.08691 — "FlashAttention-2: Better parallelism…"
- Claim: ~2x over FA1 via work partitioning; up to 73% theoretical peak forward.
  Underpins PyTorch's flash backend and cuDNN's fmha.
- Judgment: **no action required** — we inherit FA2-class kernels through SDPA.
  Our benchmarks: cuDNN 0.023 ms, flash 0.039 ms, efficient 0.041 ms (bf16,
  B=1, H=4, H_kv=2, T=256, M). Judgment: backend choice left to PyTorch; a
  global priority override rejected as unjustified (see optimization_log.md).

## 3. RMSNorm (Zhang & Sennrich, NeurIPS 2019)
- Source: arXiv:1910.07467 — "Root Mean Square Layer Normalization"
- Claim: re-centering in LayerNorm is dispensable; RMSNorm (x / RMS(x) · w)
  matches performance with fewer ops; 10–40% faster than LayerNorm in RNNs;
  widely adopted (LLaMA, T5, etc.).
- Judgment: **already our normalization.** M: 2 elementwise passes in fp32 stats;
  measured share < 2% of device time (profile) — no further fusion needed.
  fp32-internal statistics retained deliberately (GQA01 of audit).

## 4. RoPE (Su et al. 2021 — RoFormer)
- Source: arXiv:2104.09864 — "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Claim: rotate Q/K by position-dependent block-diagonal rotation; dot products
  then depend on relative offset; strong extrapolation vs additive positional
  embeddings; per-axis extrapolation limitations analyzed (leading-frequency).
- Judgment: **already implemented** (GPT-NeoX rotate-half; partial rotary;
  offset positions for KV cache). Verified by unit tests (norm preservation,
  position-0 identity, cached-offset equivalence vs prefill).
- Note: YaRN (Peng 2023, arXiv:2309.00071) shows context beyond trained length
  needs scaling tricks + fine-tuning; we do not train, so >2048 context stays an
  explicit error. rope_theta=10000 default retained.

## 5. SwiGLU / GLU variants (Shazeer 2020)
- Source: arXiv:2002.05202 — "GLU Variants Improve Transformer"
- Claim: gated FFNs (SwiGLU) improve quality vs vanilla FFN at equal params;
  8/3×d_ff expansion for param parity.
- Judgment: **already shipped** (gate/up/down, bias-free, 8/3 rule → I=640 in
  dev preset). Parameter parity check: 3·D·I ≈ 8·D² at D=256, I=640 → 4.9e5 vs
  5.2e5 — slightly under parity, documented.

## 6. Grouped-Query Attention (Ainslie et al. 2023)
- Source: arXiv:2305.13245 — "GQA: Training Generalized Multi-Query Transformer Models"
- Claim: interpolate MHA↔MQA; KV cache shrinks ~4–8x with near-MHA quality;
  single-epoch uptraining from MHA checkpoints.
- Judgment: **already shipped** (H=4, H_kv=2 dev preset). enable_gqa=True path
  verified bit-identical to manual repeat_interleave on this build (M);
  fallback path tested. KV cache at capacity 2048 = 4 MiB fp32 (M).

## 7. PagedAttention (Kwon et al. 2023, vLLM)
- Source: arXiv:2309.06180 — "Efficient Memory Management for LLM Serving with PagedAttention"
- Claim: KV cache as paged blocks with block tables; 2–4x throughput vs
  contiguous serving systems; near-zero fragmentation waste; enables large
  batch decode.
- Judgment: **principled long-term fix for our static-cache decode overhead.**
  Not adoptable without a serving-style redesign (paged kernels) — explicitly
  out of scope for V1. Documents why our current cache must remain
  eager-shaped (dynamic kv_len). Revisit at serving milestone.

## 8. CUDA Graphs for decode (NVIDIA llama.cpp blog; Hybrid JIT-CUDA Graph arXiv:2604.23467; SubGPU arXiv:2611.14904)
- Sources: NVIDIA Technical Blog "Optimizing llama.cpp AI Inference with CUDA Graphs";
  arXiv:2604.23467; arXiv:2611.00495; arXiv:2611.14904 ("Kernel launch overhead…
  non-trivial fraction of total step time at small decode batch sizes").
- Claim: graph replay of the decode step cuts per-token dispatch latency
  measurably (≈1.26× on H100 ctx=2048, marginal on L4 in one study); all
  sources agree capture needs fixed shapes.
- Judgment: **feasible only via static-shape redesign of our decode path.**
  Requirements: tensor-indexed cache updates (index_copy_), positions as
  tensors, full-capacity masked SDPA (−inf invalid) — the mask-softmax identity
  makes this mathematically exact. Full 2048-slot attention per token costs
  O(capacity²·d_h) FLOPs vs eager's O(len²·d_h) — at our launch-bound scale
  FLOPs are already free (M: 305 µs GPU vs ~620–670 µs wall per token, i.e.,
  ~50% is Python dispatch). **Candidate v0.4: static-graph decode experiment.**

## 9. Efficiently Scaling Transformer Inference (Pope et al. 2022)
- Source: arXiv:2211.05102 — Google, inference FLOPs/memory trade-offs, batching.
- Claim: decode is memory-bandwidth/launch-bound; batching and fused kernels
  dominate gains; weights load per step; FLOPs are nearly free at low batch.
- Judgment: **consistent with our measurements** (logits GEMV ~48 µs vs
  ~620 µs wall). Supports: decode bottleneck = dispatch, not math; batch
  prefill as the lever; no architecture change indicated.

## 10. Speculative decoding (Leviathan et al. 2022)
- Source: arXiv:2211.17192 — "Fast Inference from Transformers via Speculative Decoding"
- Claim: draft+verify accelerates autoregressive decode ~2–3x without quality loss.
- Judgment: **not applicable** — requires a trained draft model; we train nothing.

## Net decision for v0.4 — IMPLEMENTED AND PROMOTED

v0.4 ships `generate(..., experimental_cuda_graphs=True)`: one CUDA graph
capture per generation call (window = next_pow2(prompt + max_new + 1)), replay
per token, batch=1, CUDA, dropout=0. Correctness gate (bitwise id match vs
eager over 16+ steps, fp32 and bf16) passed before any timing was recorded.
Measured end-to-end: 1.28–1.42× at short/medium prompts (16–128 tokens,
gen 32–64); 0.94× at prompt=512 (window padding > eager's growing window) —
documented; eager remains default. See docs/rnd_report.md, registry, and
versions/v0.4/.

## Rejected/deferred (unchanged)

MLA (attention interface stays the seam; revisit at long-context milestone),
FlexAttention, Mamba/DeltaNet hybrids, adaptive softmax, fused cross-entropy
(needs training), rope_theta=500000 (no training to validate), untied
embeddings (V1 mandate), any kernel-level rewrite (hand kernels out of scope).

## 11. Residual scaling & very deep Transformers (GPT-2 null init scaling; DeepNet/DeepNorm)
- Sources: GPT-2 (Radford et al. 2019, P) initializes residual-branch projections
  with std 0.02/sqrt(2L) to keep residual variance constant across depth;
  DeepNet (Wang et al. 2022, arXiv:2203.00555) — DeepNorm: scale sublayer outputs
  by α≈(2N)^{-1/4}+… with matched init; trains 1000-layer transformers.
- Claim: unscaled residuals make gradients amplify with depth; depth-dependent
  scaling + matched init stabilizes. GPT-2 init is inference-irrelevant but
  training-relevant.
- Judgment: **not adoptable as an untrained change.** Residual scaling is an
  init/training-stability lever; without training there is no quality signal,
  and 2-layer dev preset is far from depth-instability regimes. Architecture
  already pre-norms (the 2022+ standard; post-norm deliberately rejected).
  Revisit at training-phase architecture selection. A zero-training ablation
  would only shuffle random outputs — unmeasurable.
- Status: documented, not adopted (would need training to validate — by boundary).

## 12. ALBERT factorized embedding (Lan et al. 2020, ICLR; arXiv:1909.11942)
- Source: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
- Claim: decompose vocab embedding into V×E ∘ E×H; parameter count O(V·H) →
  O(V·E + E·H); separates hidden size from vocab size; enables larger hidden
  dims at equal parameter budget; cross-layer parameter sharing additional win.
- Judgment: **already our architecture** (mandatory factorized tied head):
  V×F + F·D (+ D×F out) vs V×D = saves 125.8M params at D=512, F=64 (M, registry
  evidence). ALBERT additionally shares across layers — rejected for us (would
  change capacity irreparably without training evidence; GQA/MLA are the
  attention-side analogs we track).
- Status: implemented, measured, promoted (v0.1+). No further change justified.

## 13. Initialization research (ReZero, Bachlechner et al. 2020; fixup, Zhang et al. 2019)
- Sources: ReZero (arXiv:2003.04887) — gate each residual branch with a γ=0
  scalar ⇒ exact identity at init (dynamical isometry); Fixup (arXiv:1901.09321) —
  depth-scaled residual-branch weights without normalization; ZerO (arXiv:2110.12661).
- Claim: at init, identity mapping ⇒ stable early training without
  normalization layers.
- Judgment: **training-stability research; not actionable without training.**
  Also architecturally foreign to our pre-norm + tied-matrix-at-F design. Init is
  already documented (normal 0.02 projections; ones for norm weights). Revisit
  only in the training phase; no inference or correctness benefit now.
- Status: documented; no adoption (training-phase boundary).

## 14. Numerical stability (fp32 softmax / upcasting; bf16 attention concerns)
- Sources: Dao's kernels accumulate attention in fp32 internally (M: SDPA
  flash/MemEff bf16 path); "Broken Symmetry in BF16 Attention" (arXiv:2609.34272)
  documents bf16 attention *gradient* growth in LATE TRAINING; community practice
  (unsloth-studio, anemll) casts attention/MLP to fp32 on overflow-prone hardware;
  fp32 softmax is the reference convention (our reference_attention upcasts);
  our RMSNorm already computes in fp32; our generate() upcasts logits before
  softmax sampling.
- Claim (measured): our bf16 inference tests pass (isfinite, allclose-1e-2);
  flash kernel accumulates in fp32; no NaN/Inf observed across the suite.
- Judgment: **mostly already addressed.** Training-time bf16 gradient growth is
  out of scope (no training). Inference-time fp32-accumulation is what SDPA
  already does. Residual risk: extreme-magnitude logits — requires a probe,
  not speculation. A controlled probe is the only justified candidate: measure
  logits drift vs fp32 reference under scale-blowup inputs; adopt an opt-in
  `fp32_attention` only if drift exceeds dtype rounding; otherwise record
  rejection. See iteration below.
- Status: partially shipped (fp32 stats in norm, fp32 logits upcast for sampling,
  fp32 scores in reference impl). Probe pending — see Iteration 7.

## Iteration 7 — fp32_attention candidate (REJECTED with measurement)

Follow-up to §14. Adversarial scale-blowup probe on the attention module
(same weights fp32 vs bf16, inputs ×{1,10,100,1000,10000,100000}):

| scale | max abs Δ (bf16 vs fp32) | output max |
|---|---:|---:|
| 1 | 3.4e-03 | 0.87 |
| 10 | 0.46 | 11.7 |
| 100 | 0.47 | 118 |
| 1,000 | 5.19 | 1,179 |
| 10,000 | 59.3 | 11,790 |
| 100,000 | 503 | 117,900 |

bf16 SDPA vs bf16 reference (math backend): Δ = 0.0 exactly. No NaN/Inf at any
scale; drift is ordinary dtype rounding (~0.4-0.5% relative). The training-time
gradient issue from arXiv:2609.34272 does not transfer to inference; our
bf16 path already accumulates in fp32 (flash/MemEff kernels). **Decision: no
`fp32_attention` flag added; rejection recorded. No code change.**

## CPU memory (measured, previously unrecorded)

CPU peak RSS delta for dev-preset forward (T=64, fp32) + 16-token generation:
**+181 MB** (interpreter baseline 1.28 GB; params 39 MB; logits dominate the delta).
No swap pressure on the 14.6 GiB host. Recorded as measured; per-call RSS
churn on CUDA is ~0 (activations live on GPU).

## Batch-size scaling (measured, closes last benchmark axis gap)

CUDA fp32, T=128: B=1 1.56 ms (82k tok/s) · B=2 2.21 ms (116k) · B=4 4.57 ms
(112k) · B=8 6.31 ms (162k). bf16: B=4 best at 230k tok/s; small-T bf16 noisy
(±30% clock). Scaling is sublinear-throughput beyond B=4 at this tiny model;
logits memory becomes the bound (B=8 T=128 fp32 = 1 GiB logits tensor).

# Reference architectures (2026 open-weight releases, verified primary sources)

All three are real, public, open-weight releases verified on 2026-11-16 via
official blogs/GitHub READMEs/technical reports. Facts below are quoted from
those sources; inferences are labeled as our engineering judgment. Nothing is
copied; every candidate must stand alone in our codebase with our own tests.

## Kimi K3 (Moonshot AI, 2026 — arXiv:2607.24653; kimi.ai/blog/kimi-k3; GitHub MoonshotAI/Kimi-K3)
Facts (source: official README/blog/tech report summary):
- 2.8T total / 104B activated; 93 layers; 1 dense + 92 hybrid? Attention: **69 KDA
  (Kimi Delta Attention) + 24 Gated MLA (NoPE)**; 3:1 KDA:MLA pattern; 1 final MLA;
  attention hidden 7168; 96 heads; latent MoE dim 3584; 1M context; AttnRes (attention
  residuals across depth); LatentMoE + quantile load balancing; native multimodal.
- KDA is a descendant of Gated DeltaNet (Kimi Linear 2025): linear attention with
  delta-rule update + data-dependent gating.
Judgment: the KDA/Gated-DeltaNet component is the only piece testable here
without training, is exactly their dominant compute pattern, and maps to a
measured axis (long-context cost, state memory). Selected as the candidate.
MLA (Gated MLA): latent KV compression — deferred (we already cache cheaply with
GQA; MLA's win is training params/FLOPs, unmeasurable here without full reimpl+training).
AttnRes: residual-branch recalibration — training-stability mechanism, no-op under random weights.
LatentMoE/quantile balancing: routing quality — training-gated. Out of scope.

## Qwen3.8-Flash-Next (Alibaba, 2026-08-26 — github QwenLM/Qwen3.8-Flash-Next; blog)
Facts (source: official README):
- 125B main + 51B n-gram embedding table; 6B activated; hybrid "GDN + QSA":
  Gated DeltaNet (48 V heads, 16 QK heads, head dim 128) : Qwen Sparse Attention
  (24 Q heads, 2 KV heads, head dim 256) in 3:1 ratio over 48 layers (12 × [3 GDN+MoE → 1 QSA+MoE]);
  1M-token context; 4-branch **Gated Residual (GR)** stream with dynamic gate;
  **N-gram Embedding** lookup table (51B params, host-offloaded); Muon-family optimizer.
Judgment: GDN is the shared lineage with Kimi K3 → same candidate. QSA (sparse
top-k block attention with indexer) needs a *trained* indexer to select blocks —
an untrained fixed indexer measures speed but fabricates a quality story; rejected
for promotion claims; noted for long-context milestone. N-gram embedding: 51B
table, pure capacity/training play — rejected (no value pre-training). GR: gate
on residual branches — training-stability mechanism, unmeasurable untrained.
Muon: optimizer — explicitly out of scope.

## DeepSeek-V4.1-Flash (DeepSeek, 2026 — arXiv:2606.19348 (V4); arXiv:2609.19969 (V4.1); HF)
Facts (source: official docs/reports):
- 552B total, 8B/16B activated; **Causal Encoder–Decoder (CED)** asymmetry —
  bidirectional? No: both stacks are causal (per card); replaces MLA with hybrid
  local + long-range design; **CSA2** (Compressed Sparse Attention v2) with three
  static modes Full/Reindex/Reuse; FP4 main KV cache (E2M1 + E4M3 scale/16ch);
  ~890 B/token KV (~1/4 of V4-Flash); mHC (manifold-constrained hyperconnections);
  6-of-384 routed experts + 1 shared expert per layer.
Judgment: FP4 KV cache — quantization territory; e2e FP4 kernels not available
early enough to be honest here; also changes numerics silently-ish; deferred.
CSA2 — needs trained indexer; same training-gate as QSA; deferred. CED — encoder-
decoder asymmetry changes model class fundamentally; no benefit measurable
untrained; rejected for this project. mHC — training-stability hyperconnections;
unmeasurable untrained; documented.

## Research matrix → candidate selection

| Idea | Source | Inference-only testable? | Measurable axis | Decision |
|---|---|---|---|---|
| Gated DeltaNet / KDA | K3, Qwen Flash-Next, GDN paper | YES (unit fwd/bwd, cached-vs-uncached parity) | decode latency at long context; bytes/token; state O(1) vs KV O(T); stability | **SELECTED v0.5 candidate** |
| Gated MLA / latent KV | K3, DeepSeek-V3→V4 lineage | partial | KV bytes/token | deferred (GQA already near MLA-class savings; no training to close quality gap) |
| QSA / CSA2 sparse | Qwen, DeepSeek | only with trained indexer | speed | deferred (fixed indexer = fabricated quality) |
| Sliding-window + dense hybrid | GDN papers | yes | speed at long ctx | feasible later; GDN already subsumes mechanism class |
| N-gram embedding | Qwen | yes but zero gain untrained | none | rejected |
| GR / AttnRes / mHC | all three | gates are random-init | none | rejected (training-phase) |
| FP4 KV | DeepSeek-V4.1 | numerics change | memory | deferred (FP4 kernel maturity; silent numerics risk) |
| MTP (inference only) | K3/DeepSeek lineage | NO (needs trained head) | — | not applicable |
| Muon | Qwen, DeepSeek | optimizer | — | out of scope |

**Selected single experiment:** opt-in `attention_type="gated_delta"` prototype —
Sequential Gated DeltaNet mixer (Eq. 10, arXiv:2412.06464) with L2-normalized q/k,
per-head sigmoid α/β gates, SiLU output gate + RMSNorm before o_proj, recurrent
state cache (O(1) per token), numerics stable by contraction construction.
Correctness gates: shapes/finite, causality, cached-vs-uncached parity vs
chunked-free sequential reference, backward, roundtrip, long-run stability.
Measurement vs v0.4 dense: prefill/decode latency, state bytes/token vs KV
bytes/token, peak memory, parity. Promotion only on repeatable win; else recorded.
