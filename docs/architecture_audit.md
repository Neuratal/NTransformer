# Neuratal Transformer — Architecture Audit (Phase 1)

Date: 2026-11-16
Author: Principal ML Systems Architect / Senior PyTorch Engineer
Status: **Approved with amendments (user decision, 2026-11-16) — implemented and tested**

> **Post-approval amendments applied (user decisions, superseding §6.3/§9/§12):**
> dev preset `max_position_embeddings=2048` (not 512); `rms_norm_eps=1e-5` (not 1e-6);
> factorized tied embeddings **mandatory** — the untied path is not silently
> supported (`tie_word_embeddings=False` raises an explicit error); the output
> head owns no vocabulary parameter and receives the shared matrix at call time;
> all tests use the full 262,144-token vocabulary (no reduced vocabularies);
> cached-attention masking uses an explicit lower-right-aligned mask outside the
> `q_len == kv_len` / `q_len == 1` regimes, with position-offset RoPE.
> See docs/final_report.md for the implemented state.

This document is the required Phase 1 audit. No implementation code existed at
the time of writing.
Every quantitative claim here was computed against the real NTokenizer v0.3
artifact and this machine (see §12). Items marked **verified** were executed;
items marked **estimated** are analytic.

---

## 1. Existing project inspection: NTokenizer

### 1.1 What exists

`NTokenizer` (github-style tree at `../NTokenizer`) is a from-scratch Rust
byte-level BPE tokenizer, v0.3-polished (`output/v0.3-v2/README.md` confirms the
model of record is the unchanged v0.3 checkpoint):

| Property | Value (verified on this machine by loading the model) |
|---|---|
| Model file | `output/v0.3/ntokenizer-0.3-256k.ntok` |
| SHA-256 | `1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9` |
| Format | NTOK v3 (magic `NTOK`, version 3, CRC-32 payload check, embedded `ModelMetadata`) |
| Algorithm | byte_level_bpe, pretokenizer `MathSymbolAware`, normalization `None` |
| vocab_size | **262,144** |
| Token ID layout | `0–255` base byte tokens · `256–262,136` learned merges (261,881) · `262,137–262,143` special tokens |
| Special token IDs (verified via Python bindings) | `<|bos|>`=262137, `<|eos|>`=262138, `<|pad|>`=262139, `<|unk|>`=262140, `<|system|>`=262141, `<|user|>`=262142, `<|assistant|>`=262143 |
| Python API | `neuratal_tokenizer.Tokenizer` (ctypes, zero deps): `Tokenizer.from_file(path)`, `.encode(str) -> list[int]`, `.encode_batch(...)`, `.decode(ids) -> str`, `.vocab_size`, `.special_tokens` |
| Encode sanity check (verified) | `encode("Hello, world!") == [168069, 5431, 33]`; `decode(encode(x)) == x` |
| Rust→Python bridge | `NTOKENIZER_LIB` env var or `target/release/libneuratal_tokenizer.so` (built and present) |

### 1.2 Compatibility requirements the Transformer must satisfy

1. `vocab_size = 262,144` exactly; token IDs are consumed verbatim; **no ID
   remapping anywhere in the model**.
2. The model never tokenizes, detokenizes, pads, or normalizes text; those are
   tokenizer/pipeline responsibilities. `forward(input_ids)` accepts the
   tokenizer's raw integer IDs, including special-token IDs.
3. The model must run the full embedding lookup at ID range `[0, 262143]`;
   indices ≥ vocab_size are an error, handled by PyTorch's embedding (index
   out of range → device assert). We additionally validate ID bounds in
   `generate()`/examples, not by clamping (clamping would silently corrupt).
4. The model must not implicitly insert BOS/EOS in `forward()`. Special-token
   handling in sequences is a pipeline concern. `generate()` accepts an
   explicit `eos_token_id` / `pad_token_id` argument (fallback: config fields,
   default `None` → generate until `max_new_tokens`).
5. No dependency on the tokenizer at model build/load time. The tokenizer is
   only used by `examples/` to demonstrate the end-to-end path. The model
   package must import and run with the tokenizer entirely absent.
6. The 7 special tokens occupy real vocabulary rows (they were allocated as
   ordinary BPE entries if their literal bytes occurred in the corpus, per the
   v0.3 README's known limitation). The model must not special-case or
   re-embed them; they are normal rows of the embedding matrix.
7. `<|pad|>` (262139) is a legitimate input ID for masked-loss pipelines; the
   model has no opinion about loss masking. Loss is out of scope (Phase 2+
   boundary, §11 of the brief).

### 1.3 Tokenizer artifacts that matter for sizing

`docs/vocab_size_analysis.md` in the NTokenizer repo recommended
V=32,768/D=512/L=10 for a strict 50M full-embedding model. **That
recommendation does not carry over**: the team has since committed to the
262,144-token vocabulary for the Neuratal foundation. Vocabulary-size trade-offs
below (§6) show a full-size embedding at V=262,144 is 134M parameters at
D=512 — impossible under a ~50M budget — which makes factorized embeddings a
hard requirement, not an optimization. The 256k→128k→64k export B/T curve from
that document (4.93 / 4.61 / 4.23 bytes-per-token, v0.2 family) stays the
justification for keeping V=262,144.

---

## 2. Target architecture (proposal)

A **decoder-only, autoregressive, pre-norm Transformer** with:

- **RMSNorm** (fp32-computed, configurable eps) instead of LayerNorm — pre-norm placement, no bias, no mean-centering.
- **RoPE** on Q/K (full or partial rotary dim via `rotary_dim`), `rope_theta` configurable, precomputed cos/sin cache as non-persistent buffers, offset-aware for KV-cache decoding.
- **GQA** (separate `num_attention_heads` / `num_key_value_heads`), no biases anywhere (modern default, LLaMA-style).
- **PyTorch SDPA** (`torch.nn.functional.scaled_dot_product_attention`) as the attention kernel, with an explicit, switchable pure-PyTorch reference implementation for testing. Backend limitations handled explicitly (§4.4, §5.2).
- **SwiGLU** MLP, intermediate size configurable (default `round(8/3 × hidden)` to a multiple of 256 — SwiGLU keeps ≈ the same parameter budget as a 4× GeLU MLP while using 3 matrices).
- **Factorized embeddings** (ALBERT-style): `V × F → F × D` input projection, `D × F` output projection, shared `V × F` matrix when tying is enabled. **Weight tying** via shared parameter *by construction* (§6.4), not by aliasing in the state dict.
- **KV cache** for incremental decoding, per layer, capacity-bounded, overflow = hard error.
- **safetensors** serialization, directory format (`config.json` + `model.safetensors`), strict key validation.
- **No MoE, no hybrid SSM, no custom kernels** in v1.

Everything is **config-driven**; no architecture hyperparameter is hardcoded
outside `config.py` defaults and documented presets.

---

## 3. Mathematical formulation

Notation: `x_t ∈ R^D` hidden state, `h_l` layer `l` input, `H` query heads,
`H_kv` KV heads, `g = H / H_kv`, `d_h = D / H` head dim, `F` embedding
(factorization) dim, `I` intermediate dim, `V = 262,144`.

### 3.1 RMSNorm (pre-norm)

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * w,     w ∈ R^D, init 1
```

Computed in fp32 (upcast on entry, downcast on exit), so fp16/bf16 training
does not overflow or lose precision in the mean-of-squares. `eps` configurable
(default 1e-6).

### 3.2 Factorized input embedding

```
e_t = E[token_t]                # E ∈ R^{V×F}, the only V×* matrix (stored once)
x_0 = e_t @ W_in^T              # W_in ∈ R^{F×D}
if F == D: W_in is identity (skipped entirely; "standard" mode)
```

### 3.3 RoPE (applied to Q, K only)

For rotary dim `r ≤ d_h` (even), frequency `i ∈ [0, r/2)`:

```
θ_i = rope_theta^{-2i / r}
[cos(pos·θ_i), sin(pos·θ_i)]    # pos is absolute position incl. cache offset
```

GPT-NeoX-style rotate-half layout (`cat(-x_2, x_1)`), applied on the first `r`
channels of each head; channels `[r, d_h)` pass through untouched (partial
rotary support). cos/sin precomputed to `max_position_embeddings` as fp32
non-persistent buffers (recomputed on demand if a longer context is requested;
raises a clear error beyond a configurable hard cap rather than silently
wrapping positions).

### 3.4 Grouped-query attention (causal)

```
q = W_q h,  W_q ∈ R^{D×(H·d_h)}      k = W_k h,  W_k ∈ R^{D×(H_kv·d_h)}
v = W_v h,  W_v ∈ R^{D×(H_kv·d_h)}   o = W_o a,  W_o ∈ R^{D×D}
a_h = softmax( q_h k_{h//g}^T / sqrt(d_h) + M ) v_{h//g} ,  M = strict-causal mask (-inf above diagonal)
```

Query head `h` maps to KV head `h // g`. No biases. Attention dropout on
weights (training only, default 0).

### 3.5 SwiGLU MLP

```
MLP(x) = W_down · ( SiLU(W_gate x) ⊙ W_up x )
W_gate, W_up ∈ R^{I×D},  W_down ∈ R^{D×I}
```

### 3.6 Residual structure (pre-norm)

```
h = h + Attn( RMSNorm(h) )
h = h + MLP( RMSNorm(h) )
out = RMSNorm(h)                # final norm before the head
```

Residual-branch output dropout = `hidden_dropout` (training only, default 0).

### 3.7 Output projection (standard or factorized, tied or untied)

```
standard, untied:   logits = out @ W_vocab^T,          W_vocab ∈ R^{V×D}
standard, tied:     logits = out @ E^T                 (E is the input matrix)
factorized, tied:   logits = (out @ W_out) @ E^T,      W_out ∈ R^{D×F}  ← recommended
factorized, untied: logits = (out @ W_out) @ E_out^T,  E_out ∈ R^{V×F} separate
```

All logits variants are mathematically exact projections — no approximation.
Optional `fp32_logits` upcast before the vocab matmul output is materialized
(default off; see §7.3). Optional `logits_chunk_size` projects the sequence in
position-chunks (reduces transient matmul workspace, **not** total logits
memory; §7.3).

### 3.8 KV cache

Per layer: preallocated `K,V ∈ R^{B×H_kv×C×d_h}` buffers (`C = cache_capacity`,
default `max_position_embeddings`), with a tracked length. `update()` appends,
`get_seq_length()` reads, overflow raises `CacheOverflowError`. RoPE positions
for a decode step are `arange(cached_len, cached_len + q_len)`. Cache is an
explicit object passed `forward(..., past_key_values=cache)`; it is never
implicitly shared between sequences; `generate()` creates/resets it per call.

---

## 4. Verified environment & measured facts (this machine)

| Item | Value | Method |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, compute capability (8,9), 8 GB | `nvidia-smi`, `torch.cuda.get_device_capability` |
| Driver / CUDA | 595.99.02 / CUDA 13.2 | `nvidia-smi` |
| PyTorch | **2.15.0+cu132** (PyPI default wheel, CUDA-bundled) | pip install |
| CUDA usable | yes — bf16 matmul and SDPA verified | direct test |
| SDPA `enable_gqa=True` | **works on all backends** (flash, efficient, cudnn, math); math-backend result **bit-identical** to `repeat_interleave` expansion (max abs diff = 0.0, fp32 and bf16) | direct test |
| SDPA `is_causal=True` with `q_len == kv_len` | correct causal masking; flash-eligible (no mask tensor) | direct test |
| SDPA `is_causal=True` with `q_len < kv_len` | **upper-left aligned — WRONG for cached decoding** (query row `i` attends keys `0..i`, not `0..C+i`). Diff vs correct bottom-right mask: ≈3.29 (not ≈0) | direct test, see §5.2 |
| `torch.nn.RMSNorm` exists | yes, but we implement our own (dtype semantics explicit) | — |
| safetensors `save_file` with shared (tied) tensors | **RuntimeError** ("Some tensors share memory…") — verified; design must avoid duplicate storage by construction (§6.4) | direct test |
| `safetensors.torch.save_model` with tied weights | silently writes **one** copy; naive reload breaks the tie with a `KeyError` — verified; we do not rely on it | direct test |
| Special-token IDs | 262137–262143 as listed in §1.1 | loaded real model |

**Consequence of row 4–6:** `use_cache=True` changes which code path SDPA
takes. The implementation must branch on `(q_len == kv_len)` / `(q_len == 1)`
/ `(else)`, never on a global flag, and tests must cover all three.

**CUDA testing status:** CUDA hardware is present and functional, and the full
test suite will be run on both CPU and CUDA. No claim of CUDA support will be
made without the run actually happening. Flash/cuDNN SDPA backend dispatch on
sm_89 will be observed, not assumed; where a fallback occurs it will be
reported, not hidden (SDPA itself decides backends; we will log which backend
is selected and will not force one unless measurements justify it).

---

## 5. Critical design decisions and known pitfalls (addressed by design)

### 5.1 Attention scale
`1/sqrt(d_h)`, fixed, standard. No soft-capping (Gemma-style) — not needed, and
untested caps add risk.

### 5.2 SDPA masking semantics (**the #1 correctness risk**)
Three regimes, decided per forward call from actual shapes:

| Regime | Mask strategy | Flash eligible |
|---|---|---|
| Prefill / training: `q_len == kv_len` | `is_causal=True`, no mask tensor | yes |
| Single-token decode: `q_len == 1` (any cache) | no mask (all past keys are legitimately visible) | yes |
| Chunked prefill: `1 < q_len < kv_len` | explicit bottom-right-aligned boolean mask `j ≤ C + i` | no (falls to efficient/math — identical math, verified above) |

The naive `is_causal=True` in regime 2/3 is demonstrably wrong on this PyTorch
build (measured, §4). Tests assert regime-3 equivalence against the reference
implementation to fp tolerance.

### 5.3 GQA expansion
`enable_gqa=True` (verified identical to manual `repeat_interleave` on this
build, all backends). Manual expansion retained as the reference path. If a
future PyTorch/backend rejects `enable_gqa`, we will detect and fall back
explicitly with a one-time warning — never silently.

### 5.4 RoPE + KV cache position offsets
`forward(..., past_key_values=cache)` computes `position_ids = arange(past_len,
past_len + q_len)`. RoPE is applied **before** appending to the cache (cached
K is already rotated) — the standard, numerically correct order. Tests compare
cached vs uncached logits for exact-logit paths.

### 5.5 Numerical stability
- RMSNorm in fp32 internally.
- Reference attention softmax in fp32 with max subtraction.
- RoPE cos/sin in fp32, cast to Q/K dtype at apply time.
- Logits: model-dtype by default; `fp32_logits=True` available; softmax/loss
  consumers should upcast (documented).

### 5.6 Weight tying × safetensors
Tied output head **does not own a parameter**: the top-level model passes the
embedding matrix to the head at call time. `state_dict()` therefore contains
the shared matrix exactly once, under the embedding key, in every tying
configuration — `save_file` never sees shared memory, load is trivially
`strict=True`, and the parameter budget is exact by construction. (Rejected:
"save then strip duplicates + rewrite on load" — extra failure modes, no benefit.)

### 5.7 Causality & future leakage
Strict causal masking; no token attends to any later token in any regime
(including chunked prefill and cross-layer). Tests: perturb a future token and
assert earlier-position logits are bit-identical-ish (within fp nondeterminism);
perturb padding/mask and assert no effect.

### 5.8 Initialization
All projection matrices `N(0, 0.02)` (LLaMA/GPT-2 convention), RMSNorm weights
`1`, embedding matrix `N(0, 0.02)`, factorized projections included. No
residual out-projection rescaling in v1 (GPT-2's `0.02/sqrt(2L)` helps very deep
nets; at 2–12 layers it is noise — deferred, documented).

### 5.9 Explicitly rejected shortcuts
No clamping of token IDs; no `torch.compile` in the core path (kept opt-in,
never required); no `scaled_dot_product_attention` with silently-incorrect
argument combos; no dtype gymnastics hidden inside modules; dropout defaults
0.0 and is inert outside training mode.

---

## 6. Parameter budget & factorization analysis (computed, not guessed)

Formulas (no biases; `g = H/H_kv`; factorized tied embeddings add
`V·F + F·D + D·F + D` = matrix + in/out projections + final norm; full tied
adds `V·D + D`):

```
per_layer = 2·D²·(1 + 1/g) + 3·D·I + 2·D
total     = V·F + 2·F·D + D  +  L · per_layer        (factorized, tied)
total     = V·D + D            +  L · per_layer        (full, tied)
```

### 6.1 Per-layer cost, D = 512, H = 8, H_kv = 2 (g = 4), I = 1408 (= 11×128, ≈ 8D/3)

`per_layer = 655,360 (attn) + 2,162,688 (MLP) + 1,024 (norms) = 2,819,072`. Blocks ×10/11 layers ≈ 28.2M / 31.0M.

### 6.2 Future ~50M sketch (D = 512, H = 8, H_kv = 2, I = 1408, L = 11 → blocks = 31,009,792)

| Embedding scheme | Embedding params | **Total** | Fits 50M? | Bottleneck |
|---|---:|---:|---|---|
| Full tied (F = D = 512) | 134,218,240 | **165,228,032** | ✗ | none |
| Factorized tied, F = 128 | 33,686,016 | **64,695,808** | ✗ | 128 |
| Factorized tied, F = 96 | 25,264,640 | **56,274,432** | ✗ | 96 |
| Factorized tied, **F = 80** | 21,053,952 | **52,063,744** (L=10 ⇒ 49.2M ✓) | L=10 only | 80 |
| **Factorized tied, F = 64** | **16,843,264** | **47,853,056** (L=11) | ✓ comfortable | 64 |
| Factorized untied, F = 32/32 | 16,778,752 | **47,820,288** | ✓ | 32 — dominated by tied F=64 |
| Factorized untied, F = 32 in / 128 out | 41,944,064 | **72,952,832** | ✗ | — |

**Reading:** at V = 262,144, (a) full embeddings are categorically impossible
at ~50M; (b) tied F = 64 with L = 11 fits with ~2.1M headroom; tied F = 80
needs L = 10; (c) untied variants are parameter-dominated by any tied config at
equal budget. The 256k→128k→64k B/T curve from §1.3 shows why shrinking the
tokenizer vocabulary instead is also not free — it buys embedding params with
+2.7–7% bytes/token loss.

### 6.3 Small dev/test configuration (recommended for Phase 2, CPU-friendly, full vocab)

`V = 262,144 · D = 256 · L = 2 · H = 4 · H_kv = 2 (g = 2) · d_h = 64 · I = 640 · F = 32 · ctx = 2048 · theta = 10,000`

- per_layer = 688,640; blocks = 1,377,280; embeddings = 8,405,248 → **total ≈ 9.78M** (≈18.2M with F = 64).
- Uses the **real 262,144-row embedding** (integration reality: embedding, head, and logits all exercise the true vocab dimension), so nothing about tests is a "toy" except layer count and width, which are what vary across unit tests anyway. Unit tests for math use tiny vocab; the model-level tests use this config; an optional test instantiates a full-width (F = D) model.
- Logits transient memory, B=2, T=64, fp32: 2·64·262,144·4 B = **134 MB** — fine on both targets.
- Rationale for F = 32 (not 64+): dev speed; architecture is identical, only the V×F row width differs.

### 6.4 Weight tying — investigated, and how it is implemented

- **Tied, factorized** (recommended default): input and output share the single `V×F` matrix `E`. Output path `out → W_out (D×F) → logits via E^T`. No shape transposition errors: `E` is `[V, F]`, logits = `h_F @ E.T → [.., V]` ✓. `W_in`/`W_out` are **not** shared (different shapes and roles — input up-projection vs output down-projection; sharing them would force `F == D` or an extra transpose+reflection, and no published architecture does this).
- **Untied**: separate `V×F` matrix for output (input path cheaper memory-wise per step but +16.8M params at F = 64 — measured above, not worth it by default).
- **Tied, full-width** (F = D): standard LLaMA tying, matrix shared directly.
- Mechanism: the output head owns **no** vocab parameter when tied (the matrix is passed at call time); state_dict has exactly one copy in all configurations (§5.6). Tests assert: same tensor object identity, parameter count delta = exactly `V×F` between tied/untied configs, gradient reaches the shared matrix through both paths.
- **Claim discipline:** I do not claim tying improves quality — the literature is mixed and it is untrainable-speculation here. I claim only: at V = 262,144, tying is what makes the embedding budget survivable (saves 8.4–33.6M params), costs no extra storage, and is implemented so that it cannot silently double-count. Quality is a training-time empirical question we will not answer in this phase.

### 6.5 Memory: logits with a 262k vocab

`B·T·V` floats. B=8, T=512, fp32 = **4.3 GB** — exceeds the 4060's usable memory for larger batches. Mitigations implemented now:
1. `logits_chunk_size`: project positions in chunks (peak transient workspace ÷ chunk factor; final logits tensor unchanged — documented honestly).
2. `fp32_logits` optional.
3. Deferred (documented): fused/chunked cross-entropy for training (needs the training pipeline, explicitly out of scope); adaptive softmax (§8).

Inference-time peak is dominated by logits for the *last* position only in incremental decoding (`B·1·V` = trivial), so generation is always cheap on the head; the chunking matters for prefill + training, and we implement it now because it is small and self-contained.

---

## 7. Advanced techniques — evaluation and verdicts

Legend: **NOW** = implement in this phase; **DEFERRED** = intentionally not implemented; each deferral states the trigger that would reopen it.

| # | Technique | What it solves | Compute/memory implications | Suitable for our arch now? | Dependencies / complexity | Verdict |
|---|---|---|---|---|---|---|
| 1 | **FlashAttention** | IO-aware exact attention; removes N² memory traffic | Attention memory O(N) instead of O(N²); big speedup on long seqs | Yes — via PyTorch SDPA flash/cuDNN backends (verified working on sm_89, CUDA 13.2, torch 2.15) | none (built-in) | **NOW (through SDPA), not a custom kernel** |
| 2 | **FlexAttention** | programmable masks / block-sparse score mods; document masking | compile-dependent; performance varies by shape/compile | Not needed for dense causal; maturity risk | torch.compile toolchain | **DEFERRED** — reopen if we need block-sparse/doc masks in training |
| 3 | **MLA (DeepSeek-style)** | KV-cache compression via low-rank latent; big decode-memory win | smaller KV cache; absorbed RoPE complication; changes attention semantics & init | Not yet — prefill/training gains unclear at ≤4k ctx; head structure change | none strictly, high design risk | **DEFERRED** — GQA (H_kv=2) already cuts KV 4×; reopen at long context (>8k) or when KV memory dominates serving |
| 4 | **Gated DeltaNet** | linear-complexity seq mixing via delta rule | recurrent scan; chunked training kernels needed; different stability profile | No — different architecture class | triton/custom kernels | **DEFERRED** — reopen only as a *research branch*, never as a silent default |
| 5 | **Kimi Delta Attention** | same family as 4 (KDA, delta-rule variants) | same | No | same | **DEFERRED** — same trigger |
| 6 | **Mamba-style hybrid** | SSM+attention hybrids for long-context efficiency | new layer type; kernel deps; training interference questions | No — we are building the dense transformer foundation first (brief §4.IMPORTANT) | mamba-ssm / triton | **DEFERRED** — keep block interface modular so a hybrid Block can be introduced later |
| 7 | **Multi-Token Prediction** | sample-efficiency & speculative-decoding option | 1–4 auxiliary heads; training-loss changes; inference unchanged | Training-time only; we are not training | none strictly | **DEFERRED** — reopen in the training-pipeline phase; keep `NeuratalModel` head-output pluggable |
| 8 | **Adaptive Softmax** | cuts output-vocab compute for large V | rare-token clustering; breaks clean tying; sampling bias risks; complexity | Marginal — factorized tied head is already cheap; memory, not FLOPs, is our constraint, and it is solved by chunking | moderate | **DEFERRED** — reopen only if profiling shows the head dominates *training* step time |
| 9 | **Muon / NorMuon** | better hidden-matrix optimizer | orthogonalized updates; per-matrix optimizer state | Training-only; brief forbids implementing an optimizer now | none | **DEFERRED** — optimizer phase, not architecture phase |
| 10 | **Sparse attention** (sliding window / block) | long-context compute reduction | quality trade-offs; backend-specific | No — context ≤ 4k initially | mask plumbing exists via our mask path | **DEFERRED** — reopen at long-context milestone; the §5.2 mask path is the seam |

**Standing decisions:** SDPA-with-flash is our FlashAttention. Everything else is deferred with named triggers. The attention and MLP are plain
modules behind narrow interfaces, so any of these can be introduced later without rewriting the model. Speculative components will not be added to the default path.

---

## 8. Proposed module structure (v1)

```
NTransformer/
├── pyproject.toml                     # metadata, deps: torch, safetensors; dev-extra: pytest; NO others
├── README.md
├── docs/architecture_audit.md         # this document
├── neuratal/
│   ├── __init__.py                    # exports: NeuratalConfig, NeuratalModel, save_pretrained/load_pretrained
│   ├── config.py                      # NeuratalConfig dataclass + validation + resolved defaults
│   ├── model/
│   │   ├── __init__.py                # re-exports
│   │   ├── embeddings.py              # TokenEmbedding / FactorizedEmbedding (V×F, in F→D projection)
│   │   ├── normalization.py            # RMSNorm (fp32 internal)
│   │   ├── positional.py               # RotaryEmbedding (+ rotate-half); cache registry
│   │   ├── attention.py                # GQAAttention (SDPA path + reference_attention for tests)
│   │   ├── mlp.py                     # SwiGLU
│   │   ├── block.py                    # NeuratalBlock (pre-norm, residual)
│   │   ├── output.py                   # OutputProjection (standard/factorized × tied/untied, chunked)
│   │   ├── cache.py                    # KVCache / CacheOverflowError
│   │   ├── generation.py               # generate(): greedy / temperature / top-k; EOS stop; cache-aware
│   │   └── model.py                    # NeuratalModel + forward(); CausalLMOutput dataclass
│   └── serialization/
│       ├── __init__.py
│       └── checkpoint.py               # save_pretrained / load_pretrained (config.json + model.safetensors)
├── tests/                             # pytest; CPU default, CUDA parametrized where hardware exists
│   ├── conftest.py                    # tiny-vocab config factory + seeded RNG fixtures; real-vocab config
│   ├── test_config.py                 # validation: every error branch, resolved defaults, serialization-safe
│   ├── test_embeddings.py             # shapes, ID bounds, factorization path, tying identity
│   ├── test_normalization.py           # fp32-vs-bf16 stability, eps, near-zero variance input
│   ├── test_rope.py                   # freq math, partial rotary, offset positions, long-context growth
│   ├── test_attention.py               # causal property, GQA head mapping, SDPA-vs-reference equivalence,
│   │                                 #   regime q_len==kv_len / q_len==1 / chunked, grad flow
│   ├── test_mlp.py                    # SwiGLU exact math, shapes, backward
│   ├── test_block.py                  # residual paths, shapes
│   ├── test_model.py                  # forward fwd/bwd, vocab 262144 integration, cache equivalence,
│   │                                 #   no-future-leak, tied/untied param sharing
│   ├── test_cache.py                  # overflow error, reset, position tracking, cross-sequence isolation
│   ├── test_generation.py             # greedy determinism, EOS stop, cache vs no-cache equality
│   ├── test_serialization.py           # roundtrip bitwise, tied-matrix single-copy check, corrupt/missing/unexpected keys
│   └── test_tokenizer_integration.py  # skipped unless NTokenizer bindings + model file present: real IDs → forward pass
├── examples/
│   └── inference.py                    # loads tokenizer (optional) + checkpoint; prints generated ids/text
└── .gitignore
```

Deviation from the brief's sketch, with reasons: `tests/` sits at repo root (not inside the package — standard Python packaging), and there is no `normalization`/`positional` merge — each component keeps single responsibility for isolated testing. `serialization/` is a subpackage but intentionally tiny. No `__init__` re-export soup: one public surface in `neuratal/__init__.py`.

---

## 9. Configuration system (design)

`NeuratalConfig` — a `@dataclass` with `__post_init__` validation. Nothing is
silently corrected; every violation raises `ValueError` with the offending
field and the constraint.

Fields (defaults = the §6.3 dev/test preset):

```
vocab_size                int    = 262_144   (≥ 1; IDs verbatim)
hidden_size               int    = 256       (≥ 1)
embedding_size            int|None = 32      # F; None ⇒ full-width (F = hidden_size, standard embedding)
intermediate_size         int|None = None    # None ⇒ round(8/3·hidden) to multiple of 256, resolved in post_init
num_hidden_layers         int    = 2
num_attention_heads       int    = 4         # hidden % heads == 0 required
num_key_value_heads       int|None = 2       # heads % kv == 0 required; None ⇒ num_attention_heads
max_position_embeddings   int    = 2048
rope_theta                float  = 10_000.0  (> 0)
rotary_dim                int|None = None    # None ⇒ head_dim; even, ≤ head_dim
rms_norm_eps              float  = 1e-5      (> 0)
attention_dropout         float  = 0.0       [0, 1)
hidden_dropout            float  = 0.0       [0, 1)
tie_word_embeddings       bool   = True
use_cache                 bool   = True
dtype                     str    = "float32" # one of float32/bfloat16/float16
fp32_logits               bool   = False
logits_chunk_size         int|None = None    # > 0 when set
cache_capacity            int|None = None    # None ⇒ max_position_embeddings
attention_implementation  str    = "sdpa"    # {"sdpa", "reference"} — reference is for tests, not training
bos_token_id / eos_token_id / pad_token_id : int|None = None   # not baked-in; generation args fall back to these
```

Validation rules (each with a distinct error message): head divisibility,
KV divisibility, RoPE evenness/bounds, theta/eps positivity, dropout range,
dtype enum, `vocab_size ≥ 1`, factorized `embedding_size` must be `1..hidden_size`
and — for tying to be exact — the same F is used on both paths by construction,
`rotary_dim ≤ head_dim` and even, resolved `intermediate_size ≥ 8`, config
`to_dict()/from_dict()` roundtrip with unknown-key rejection, and a `validate()`
re-run after any mutation. The dataclass stores the **resolved** values so that
saved `config.json` is self-describing and checkpoint loading never re-derives
(and possibly disagrees on) defaults.

The 50M sketch (§6.2, F = 64, D = 512, L = 11, H = 8, H_kv = 2) is documented
as a preset in the README/audit but is **not** a special code path; it is one
more `NeuratalConfig`.

---

## 10. Serialization format (design)

```
checkpoint_dir/
├── config.json        # resolved config + format_version = 1 + architectures = ["NeuratalModel"]
└── model.safetensors  # weights only; no shared memory possible (§5.6)
```

- `save_pretrained(model, dir)`: validates the model first (finite-check optional, off by default), writes atomically (tmp file + rename) to avoid torn checkpoints.
- `load_pretrained(dir, map_location=None)`: reads `config.json` → builds config (unknown keys rejected) → builds model → loads safetensors with **strict** key matching (missing or unexpected → error listing both, no silent tolerance) → verifies embedding rows == `config.vocab_size` → returns the model.
- Tied checkpoints store exactly one copy of the shared matrix (guaranteed by construction; test asserts file size ≈ `(total − V·F)·elt_size` and key absence).
- No pickle, no `.pt` weights, ever. `config.json` is data, not code.
- Portability: safetensors is endian/word-size safe; dtype recorded per-tensor in the file; `map_location` handles CPU↔CUDA.

---

## 11. Test plan summary (what will be run, and on what)

- **Unit tests** per component (shapes, math, gradient flow, edge cases), deterministic seeds, tiny vocab (fast) + dtype-parametrized (fp32, bf16 CPU, bf16/fp16 CUDA).
- **Property tests**: causality (future-token perturbation invariance), cached vs uncached logit equivalence (fp32 ≤ ~1e-6 abs, loose for bf16/fp16), GQA mapping vs explicit grouped reference, RoPE norm preservation (‖RoPE(x)‖ == ‖x‖ per pair-block), RMSNorm output RMS ≈ 1 before weighting.
- **Integration**: dev/test config (§6.3) on CPU **and CUDA**; full forward+backward; serialization roundtrip bitwise; generation with cache.
- **CUDA runs happen on this machine** (RTX 4060 Laptop, torch 2.15.0+cu132); results reported with hardware, torch/CUDA versions, dtype, and tolerances. No unverified "CUDA supported" claims.
- **Tokenizer integration** test: encode "Hello, world!" with the real v0.3 model via bindings → feed IDs into the model → forward succeeds and logits shape is `[B,T,262144]`; skipped gracefully (never failed) if bindings/model file are absent.
- **Warnings policy**: none suppressed; the known harmless one (missing NumPy inside torch) is eliminated by installing numpy in the dev venv, not by filtering.

Tolerance policy: fp32 exact-ish (rtol/atol ≈ 1e-5/1e-6); bf16/fp16 (rtol/atol ≈ 1e-2/1e-2 with relative-to-scale checks); no tolerance is loosened to make a failing test pass — a mismatch means a bug, investigated.

---

## 12. Risks & open questions (for the approver)

1. **Default `rope_theta`**: 10,000 (classic) vs 500,000 (LLaMA-3-style, better long-context). Proposal: 10,000 for ≤ 2k context; configurable; revisit with context milestone.
2. **`rms_norm_eps` default**: 1e-6 (torch convention) vs 1e-5 (LLaMA). Proposal: 1e-6.
3. **`hidden_dropout` semantics**: residual-branch output dropout (my proposal) vs embedding dropout. Proposal: residual-branch output; spec review welcome.
4. **`intermediate_size` resolution rule**: `round(8/3 × hidden)` to a multiple of 256 (proposal) — alternative is explicit always-set values in presets only. Proposal stands (deterministic, documented, serialized resolved).
5. **Dev/test preset** (§6.3): F = 32 factorized, 2 layers, D = 256. Final ~50M dimensions are **not** committed here — that decision belongs to the training phase with quality measurements.
6. **`attention_implementation="reference"`** ships in the package (test-only but useful for debugging on exotic hardware). Flagged as non-performance API.
7. Generation scope: greedy + temperature/top-k sampling only; no beam search, no KV-eviction, no serving batching.
8. Chunked logits reduce transient workspace only; **fused cross-entropy is explicitly out of scope** until the training phase.

---

## 13. Explicit non-goals (this phase)

No training, no optimizer/schedule, no dataset/tokenizer work, no 50M pipeline,
no MoE, no custom kernels, no compilation dependency, no serving stack, no
benchmark *claims* without runs. The deliverable is the correct, tested,
serializable architecture plus this audit.

---

## 14. Decision requested

Approve (or amend) the above: architecture set (§2), serialization design (§10),
dev/test preset (§6.3), configuration defaults (§9), and deferral list (§7).
On approval, Phase 2 implements components with tests alongside, then Phase 3
(cache/generation/serialization), Phase 4 (full audit + test suite on CPU and
CUDA), Phase 5 (final report with measured results).
