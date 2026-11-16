# Neuratal — Research Matrix (refreshed 2026-11-16)

Scope: inference-only architectural and systems techniques drawn from public,
verified 2026 sources. Training is out of scope until explicit user consent.

## Sources — verified 2026 landscape

### Closed frontier (capability ceiling; architecture undisclosed — cited for context, not for copying)

| System | Owner | Access | Context | Notes (source) |
|---|---|---|---|---|
| GPT-6 Astra | OpenAI | API, $10/$50 per MTok | 1,050,000 (922k max input) | Agentic/computer-use focus; 128K max output; KB cutoff 2026-04; OSWorld 72.6%; no fine-tuning (openai.com/index/gpt-6-astra-next-generation-work) |
| Claude Fable 5.2 / Opus 5.5 | Anthropic | API, $8/$40 ($4/$20) | 1M both | 128K output; adaptive thinking (always on); tool use + multimodal (platform.claude.com/docs/en/models/overview) |
| Gemini 4 Argon / Carbon | Google DeepMind | API ($4/$20 post-launch) | 1M | Trained JAX/ML Pathways; Argon cyber-defense rollout; Carbon Nov-2026 card (deepmind.google/models/model-cards/gemini-4-argon, -carbon) |
| Gemma 4 | Google DeepMind | Open weights (E2B/E4B/12B/26B-A4B/31B) | up to 256K | Open sibling of Gemini 3 research; MoE + dense; multimodal (deepmind.com/models/gemma) |

Frontier closed models are "mythos-like" capability references; none disclose
architecture. Parameter count ≠ intelligence; nothing here transfers without
their training data/compute. We do not claim parity — we adopt disclosed
*techniques* only.

### Open weights (disclosed architectures — directly relevant)

| System | Scale | Architecture core | Source |
|---|---|---|---|
| Kimi K3 | 2.8T/104B, 93 L | **69 KDA + 24 Gated MLA (NoPE), 3:1**, AttnRes, LatentMoE | arXiv:2607.24653; kimi.ai/blog/kimi-k3; GitHub MoonshotAI/Kimi-K3 |
| Qwen3.8-Flash-Next | 125B + 51B n-gram, 6B act | **GDN:QSA hybrid 3:1**, 4-branch Gated Residual, MoE, Muon | github QwenLM/Qwen3.8-Flash-Next; qwen.ai/blog?id=qwen3.8-flash-next |
| DeepSeek-V4.1-Flash | 552B, 8-16B act | **CED asymmetric**; CSA2 (Full/Reindex/Reuse); FP4 KV (E2M1); mHC; 6/384 routed experts | arXiv:2606.19348; arXiv:2609.19969; HF card |
| Qwen3.5/3.6/3.7 lineage | — | GDN hybrid (3 GDN : 1 Gated Attn) since Qwen3.5 | qwen.ai/blog?id=qwen3.8-flash-next |
| Gemma 4 (E2B..31B) | open | dense + 26B-A4B MoE | deepmind.com/models/gemma |

## Technique matrix (math · cost · training dependency · hardware fit · status in NTransformer)

### Attention family
1. **FlashAttention 2/3** (arXiv:2205.14135, 2307.08691) — IO-aware exact attention; O(N) memory; training kernels. Inference: adopted via SDPA (bf16 flash / fp32 mem-efficient; verified). Status: shipped v0.1.
2. **GQA** (arXiv:2305.13245) — KV-head sharing; ~4× KV shrink; no training to adopt (we ship H_kv=2). Status: shipped.
3. **MLA / latent-KV** (DeepSeek-V2 lineage; K3 Gated MLA) — low-rank latent compressed KV; absorption trick makes decode KV reads O(λ·d) with λ≪2·H_kv; inference-usable math, but quality requires training to close vs GQA. Status: deferred (GQA sufficient at our scale; no quality delta measurable untrained).
4. **Gated DeltaNet / KDA** (arXiv:2412.06464; K3 69 layers; Qwen hybrid) — `S_t = S_{t-1}(α_t(I−β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`; O(1) state, zero KV growth; sequential form = our v0.5-candidate (measured: memory win 32×, latency loss everywhere). **The WY chunkwise parallel form (Eqs. 6–9 of the paper) is the missing algorithm for fast prefill.** Status: prototype shipped (opt-in, not promoted); chunkwise form = proposed v0.6.
5. **Sparse attention / indexer** (QSA, CSA2) — top-k block selection by a *trained* indexer. Inference-only without training = fixed pattern; then it's just sliding-window attention (below), honestly labeled. Status: deferred (indexer is training-gated).
6. **Sliding-window attention** (Mistral/Gemma lineage) — fixed window; long-context quality tied to training; inference cost measurable. Status: deferred as research branch only if context target > 2048 with quality evidence (none without training).
7. **NSA (native sparse attention) lineage** — same indexer dependency. Deferred.

### State/memory family
8. **PagedAttention/vLLM** (arXiv:2309.06180) — paged KV, block tables, continuous batching; the *serving* answer. Status: documented; not applicable to single-request decode benchmark; revisit at serving milestone.
9. **CUDA graphs** — graph replay erases launch overhead; v0.4 shipped opt-in (1.3–1.4× decode speedup, bitwise-equal). Status: shipped opt-in.
10. **FP4 KV (DeepSeek-V4.1: E2M1+E4M3)** — ~890 B/token (≈¼ of V4); needs FP4 kernels (sm_100+; our sm_89 lacks native FP4). Status: deferred on hardware.
11. **FP8/int8 KV** — memory-bandwidth win on decode GEMV; quantization kernels not in scope (silent numerics risk without eval harness). Status: deferred.
12. **Quantized weights (GPTQ/ExLlama, AutoAWQ+Marlin)** — 3–4× decode speedup on memory-bound decode at batch-1 (cited public study); quality loss requires eval. Status: deferred (needs eval harness; unverifiable without training).

### Normalization / residual / MLP family
13. **RMSNorm** (arXiv:1910.07467) — shipped; fp32 stats.
14. **Residual scaling** (GPT-2 init, DeepNorm arXiv:2203.00555, ReZero arXiv:2003.04887) — training-stability; no inference effect; training-gated. Deferred.
15. **Gated/branched residuals** (Qwen GR 4-branch, DeepSeek mHC, K3 AttnRes) — training-stability; unmeasurable untrained. Deferred.
16. **ALBERT factorized embeddings** (arXiv:1909.11942) — shipped (mandatory); capacity↔param trade-off documented.

### Training/inference support
17. **MTP (Multi-Token Prediction)** — training objective; inference benefit via self-speculation requires trained heads. Not applicable without training.
18. **Speculative decoding** (arXiv:2211.17192) — draft model; none exists trained on our tokenizer. Not applicable.
19. **Muon / NorMuon** — optimizer; explicitly out of scope (no training without consent).
20. **MoE + load balancing (auxiliary-loss-free / quantile)** — training-gated routing quality; inference-routing from random-init = noise. Deferred.
21. **N-gram embedding** (Qwen) — 51B table; capacity lever; zero value untrained. Rejected.

### Long-context mechanisms
22. **RoPE scaling (YaRN arXiv:2309.00071)** — needs fine-tune; rejected (no training).
23. **KDA/GDN hybrid (3:1 with dense attention)** — K3's pattern; our GDN prototype proves the mixer; hybrid interleaving is the actual K3 design. Candidate for later evaluation when chunkwise form lands (v0.6) and if memory win scales to long contexts.
24. **Compressed/global attention (CSA2 reuse, HCA)** — training-gated indexer; deferred.

## Bottleneck ranking (measured on this machine; importance × measurability)

| # | Bottleneck | Evidence | Lever class |
|---|---|---|---|
| 1 | Python/dispatch overhead at decode | ~50% of decode wall time; v0.4 CUDA graphs: −24% latency | CUDA graphs / compile / fusion |
| 2 | Launch-overhead-bound prefill GEMM sequence | 63% device time in logits GEMM at T=128; fused QKV+gate_up (v0.3) | kernel fusion |
| 3 | Dense KV bytes O(T) at long context | 2 KiB/token/layer fp32; 4 MiB at 2048 (dev cap) | GDN/KDA (state O(1)); MLA |
| 4 | Memory-bandwidth-bound vocab GEMV (decode logits) | ~48 µs/step fp32 (264K×32) | quantization; bf16 GEMV |
| 5 | Sequential Gated-Delta recurrence at prefill | 9.6× slower prefill vs dense (v0.5-candidate); WY chunkwise form identified | algorithm (paper Eqs. 6–9) |
| 6 | fp32 logits memory at 262k vocab | fwd+bwd peak 5.2 GB @B=2 T=512 | training-phase fused CE (out of scope) |

## Proposed single experiment (awaits approval)

**v0.6-candidate: WY chunkwise parallel form of the gated delta rule**

- Hypothesis: implementing the paper's chunkwise parallel algorithm (C=64,
  WY representation of cumulative Householder products via UT transform,
  Eqs. 6–9 of arXiv:2412.06464) reduces GDN prefill latency from ~9.6× slower
  to ≤2× of v0.4 dense at T ≥ 256, while keeping O(1) state and the 32×
  memory advantage; cached≡uncached parity and all 319 tests stay green.
- What changes: only the prefill computation path for attention_type="gated_delta";
  sequential path retained as the parity reference. Default attention_type
  stays "gqa"; no training; no external data; deterministic seeds.
- Measurements (identical harness to v0.4 A/B): prefill/decode at
  T ∈ {128, 512, 1024, 2048}, fp32 + bf16, seed 42, warmup 5, 20 reps,
  median + p95; correctness gate: sequential-vs-chunked max|Δlogit| ≤ 1e-5
  (fp32) before any timing; state bytes/token; cached-vs-uncached parity.
- Effort/risk: ~250 lines; risk = subtle WY-form bug (mitigate: parity test
  against sequential form is bitwise-class). Memory win preserved.
- Promotion gate: all tests green AND measured GDN prefill ≤ 2× dense at
  T ≥ 256 AND decode latency not worse than v0.4 dense. Otherwise recorded
  as a rejected iteration.
- Training-gated? No. No external resources. No new dependencies.

**Training track (Köprü, not started):** data/licenses, tokenizer parity,
pretraining, evals, instruct/RLHF, safety — outlined for AŞAMA 3; requires
explicit user approval before any data download, GPU training, or optimizer
step. Until then model capability stays "ölçülemedi".

## Honesty ledger
- Closed frontier benchmark numbers (OSWorld, FrontierMath, etc.) are cited
  as claims by their owners, not re-verified here.
- Our dev preset is 9.8M params, untrained; no capability claims are possible.
- Every technique adopted is with our own measured evidence on this machine.
- Long-context beyond 2048 is unmeasured here (config cap); no extrapolation
  will be presented as measurement.

## Iteration 10 — MLA/latent-KV prototype (measured; NOT promoted)

DeepSeek-style latent KV (arXiv:2405.04434 core idea): cache one 64-d latent
per token per layer; K/V expanded from it per step (no absorption trick).
Measured identically vs v0.4: decode +12.5-17.1% slower at ctx 64/512/1024;
prefill 10-16% slower; KV bytes/token/layer 256 vs 1024 B (4x memory win).
Correctness: 338 passed; parity vs eager verified (token-by-token and chunked
prefill). Quality unverifiable without training. Kept as branch cand/v0.6-mla
+ versions/v0.6-candidate (frozen) + opt-in attention_type="mla" there;
NOT merged into master; best_version stays v0.4. Honest verdict: at our
launch-bound 9.8M scale the up-projection costs more than the KV broadcast
saves; MLA only pays at long-context memory-bound serving (unmeasurable here).

## Iteration 11 — SWA + bf16 GEMV (both rejected on measurement)

SWA (Mistral-style, swa_window=512, lower-right-aligned mask):
- correctness: window-mask semantics, causality, cache parity, roundtrip bitwise — all green
- measured 3 seeds: wash at T=1024; **20-40% slower at T=2048** (flash kernel
  not selectable under explicit masks; attention is only ~7.5% of forward)
- verdict: rejected. Branch cand/v0.7-sliding-window, frozen snapshot, not promoted.

bf16 decode GEMV check (bottleneck #4):
- micro-GEMV [1x32]@[32x262144]: fp32 146.7 us (229 GB/s) -> bf16 58.4 us (287 GB/s)
- end-to-end decode fp32 0.73-0.77 ms/tok vs bf16 0.74-0.82 — NO end-to-end win
  (launch-bound; upcast overheads eat the 88 us saving)
- verdict: not a viable architecture lever at this scale; bf16 stays a config
  option; bottleneck #4 fix remains launch-overhead reduction (v0.4 graphs).
