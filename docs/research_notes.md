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
