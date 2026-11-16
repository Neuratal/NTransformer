# Neuratal Transformer — Final Engineering Report (Phase 5)

Date: 2026-11-16
Project: Neuratal language model foundation — Transformer core (V1)
Status: **Implementation complete, tested on CPU and CUDA. Untrained model; no training artifacts exist.**

> **Update (2026-11-16):** A later engineering pass added a training engine,
> chunked cross-entropy, opt-in MoE, MoE presets, local sharded checkpoints, and
> measured dense/MoE smoke tests. See `docs/moe_integration.md` for that pass's
> final report, scale-readiness table, and limitations. Statements below such as
> "no training loop" and "no MoE" describe the earlier Phase 5 snapshot.

This report covers Phases 2–5 (implementation, tests, integration audit, final
report). Phase 1 is documented in `docs/architecture_audit.md`. All quantitative
claims below were measured on this machine or computed exactly by the
implementation; nothing is estimated where a measurement was possible, and
theoretical numbers are labeled as such.

---

## 1. What was built (and what was not)

**Built and tested**
- Decoder-only, autoregressive, pre-norm Transformer (NeuratalModel)
- RMSNorm · RoPE (rotate-half, partial-rotary support, offset-aware) · GQA · SwiGLU
- **Mandatory factorized + tied embeddings** at full NTokenizer vocab (262,144)
- SDPA attention with position-regime causal dispatch (no blind `is_causal`)
- Explicit pure-PyTorch reference attention (test/debug path)
- KV cache (preallocated, capacity-bounded, commit protocol, overflow = hard error)
- `generate()`: greedy + temperature/top-k sampling, EOS early stop, batch-aware padding
- Safetensors + JSON checkpoint format (atomic writes, strict validation, dtype recording)
- Full test suite: **218 collected; 217 passed, 1 skipped** (CPU); all 34 CUDA-parametrized/CUDA tests pass
- `examples/inference.py` (end-to-end NTokenizer → model → decode), `examples/benchmark.py` (measured performance)

**Not built (per project boundary)**
- No training loop, dataset pipeline, optimizer, or LR schedule
- No 50M training run; no model quality claims of any kind
- No MoE, MLA, Mamba/DeltaNet, FlexAttention, adaptive softmax, MTP, custom kernels
- No fused cross-entropy (logits storage reduction remains open work)
- V1 has no untied embedding path (mandatory tying; `tie_word_embeddings=False` raises `ValueError`)

---

## 2. Implemented architecture

### 2.1 Topology

```
ids → FactorizedEmbedding(vocab=262144, F) → in_proj (F→D)
    → L × NeuratalBlock:  h = h + drop(attn(RMSNorm(h))) ; h = h + drop(mlp(RMSNorm(h)))
    → RMSNorm → OutputProjection(h) → logits[vocab]
```

Attention: `W_q: D→H·d_h, W_k/W_v: D→H_kv·d_h, W_o: D→D`, no biases anywhere.
MLP: `W_gate, W_up: D→I, W_down: I→D` (SwiGLU: `W_down(SiLU(W_gate x) ⊙ W_up x)`).
Norms: `RMSNorm(x) = x/√(mean(x²)+eps)·w` computed in fp32 (`eps` default 1e-5).
Positions: RoPE, GPT-NeoX rotate-half layout, `θ_i = 10000^(-2i/r)`, partial rotary
via `rotary_dim` (default = `head_dim` = 64). Absolute positions only:
`positions = arange(past_len, past_len + T)`; cached K is rotated before storage.

### 2.2 The tied, factorized head (mandatory)

One trainable matrix `E ∈ R^{V×F}` exists, exactly once, in the module tree and
in the state dict:

```
Input:  x_ids = E[ids]                (lookup)
        h0    = x_ids @ W_in^T        (W_in ∈ R^{F×D})
Output: h     = RMSNorm(final)
        f     = h @ W_out^T           (W_out ∈ R^{D×F})
        logits = f @ E^T              (transpose of the SAME matrix)
```

The output head owns **no** vocabulary parameter; the model passes
`E` to `OutputProjection.forward(h, embedding_weight=E)` at call time. Tying is
structural by construction — there is no duplicate storage to "remember to strip"
at save time (safetensors would refuse the save anyway). Verified: checkpoint
stores exactly one `token_embedding` tensor; params bytes 39,130,112 vs file
39,132,280 (2,168 B JSON header); logits bitwise identical after save/load;
`parameter_report()` sums exactly to the model total with the matrix counted once.
Parameter capacity added by tying vs. untied: saves `V·F` (8.4M at F=32 dev
preset; 16.8M at F=64).

`embedding_size=None` resolves to full-width (F = hidden_size; W_in/W_out omitted).
`F > hidden_size` is rejected at config validation.

### 2.3 Position-regime-correct causal attention (the critical requirement)

Measured on this machine: SDPA `is_causal=True` with `q_len < kv_len` is
**upper-left aligned** (0-vs-N diff ≈ 3.29 vs the correct mask). V1 dispatches
on shapes, every call:

| Regime | Mask | Backend eligibility |
|---|---|---|
| `q_len == kv_len` (prefill/training) | `is_causal=True`, no mask tensor | flash/cuDNN-eligible |
| `q_len == 1` (decode) | none (all past visible) | flash/cuDNN-eligible |
| `1 < q_len < kv_len` (chunked prefill) | explicit boolean `j ≤ i + (kv_len − q_len)` — lower-right, absolute | efficient/math (identical math) |

GQA: `enable_gqa=True` (verified bit-identical to manual expansion with the
math backend on torch 2.15.0+cu132, and functional on flash, mem-efficient,
cuDNN backends on sm_89). If SDPA rejects it at runtime, V1 falls back
explicitly to `repeat_interleave` with a one-time `UserWarning` — never
silently; the fallback path is unit-tested. Backend behavior on other
hardware/PyTorch builds is tolerance-checked, not bit-identical.

### 2.4 KV cache

Per-layer preallocated `K,V ∈ [B, H_kv, C, d_h]` (`C = cache_capacity`, default
`max_position_embeddings`); views `K[:, :, :length]` are returned to callers.
Model calls `commit(T)` exactly once per forward; within a pass the offset is
constant across layers (tested). Overflow raises `CacheOverflowError` (tested at
the boundary, including partial fits). `reset()` rewinds length keeping buffers
(buffers reused, tested by data_ptr identity; cross-sequence isolation tested).
`RoPE` positions are absolute; single-token decode matches full prefill to
≤ 6e-7 (measured).

---

## 3. Parameters: measured, not estimated

Formulas (no biases): `per_layer = 2D²(1+1/g) + 3DI + 2D`,
`total = V·F + 2FD + D + L·per_layer`.

**Development/testing preset (as approved)** — `V=262144, D=256, L=2, H=4, H_kv=2, F=32, I=640, ctx=2048`:

| Component | Params |
|---|---:|
| Embedding matrix (V×F, tied, counted once) | 8,388,608 |
| Input projection (F×D) | 8,192 |
| Output projection (D×F) | 8,192 |
| Attention (2 layers: qkv + o) | 393,216 |
| MLP (2 layers: gate+up+down) | 983,040 |
| RMSNorms (2×2 + final) | 1,280 |
| **Total** | **9,782,528** |

Logits transient at B=1, T=8, fp32: ~8 MB; fwd peak measured 115.5 MiB (CUDA fp32, T=64, B=1).

**Reference only — possible ~50M configuration** (NOT a committed training config),
`F=64, D=512, L=11, H=8, H_kv=2, I=1408`, computed by the implementation:

| Component | Params |
|---|---:|
| Embedding matrix | 16,777,216 |
| Input + output projections | 65,536 |
| Attention | 7,208,960 |
| MLP | 23,789,568 |
| RMSNorms | 11,776 |
| **Total** | **47,853,056** |

Full-size tied embedding at D=512 alone would be 134,217,728 → factorization is
what makes a ~50M-parameter model with the 262k vocabulary feasible. The final
training architecture decision belongs to the training phase (with measurements);
V1 commits to nothing beyond this reference point.

---

## 4. Module inventory

| Module | Path | Responsibility |
|---|---|---|
| Config | `neuratal/config.py` | validated `NeuratalConfig`; nothing silently corrected; resolved values serialized |
| FactorizedEmbedding | `neuratal/model/embeddings.py` | V×F matrix + F→D projection; ID bounds/dtype validation; the single tied matrix |
| RMSNorm | `neuratal/model/normalization.py` | fp32 statistics, configurable eps, no bias |
| RotaryEmbedding | `neuratal/model/positional.py` | fp32 cos/sin buffers (non-persistent, `.to()`-protected), partial rotary, absolute positions |
| GQAAttention | `neuratal/model/attention.py` | q/k/v/o projections, RoPE, SDPA dispatch, explicit fallback, `reference_attention` |
| SwiGLU | `neuratal/model/mlp.py` | bias-free gated MLP |
| NeuratalBlock | `neuratal/model/block.py` | pre-norm block, residual, hidden_dropout placement |
| OutputProjection | `neuratal/model/output.py` | factorized head on the shared matrix; `logits_chunk_size`; `fp32_logits` |
| NeuratalModel | `neuratal/model/model.py` | topology, positions, cache wiring, `parameter_report()`, applies `config.dtype` |
| KVCache | `neuratal/model/cache.py` | preallocated, overflow-safe, commit protocol, reset/reuse |
| generate() | `neuratal/model/generation.py` | greedy/sampling, EOS stop, per-call fresh cache |
| checkpoint | `neuratal/serialization/checkpoint.py` | atomic safetensors+JSON save/load, strict keys/shapes/dtypes, tied-matrix exactly-once |

Dependencies: **torch ≥ 2.4, safetensors ≥ 0.4** (runtime); pytest, numpy (dev).
Nothing else. No tokenizer dependency (the model never imports it; only
`examples/inference.py` uses the NTokenizer bindings, and it degrades gracefully).

---

## 5. Test results (measured)

- **CPU suite:** 218 collected → 217 passed, 1 skipped (the CPU side of the
  CUDA-only device-mismatch probe). ~8 s total.
- **CUDA:** all 33 CUDA-marked tests pass on NVIDIA GeForce RTX 4060 Laptop
  GPU (sm_89, 8 GB), torch 2.15.0+cu132, driver 595.99.02, CUDA 13.2.
  - Includes: cached/uncached logit parity on GPU (bitwise 0.0 for full-sequence
    reuse; ≤6e-7 token-by-token), bf16/fp16 forward+backward, bf16/CUDA
    serialization roundtrip with dtype recorded, GQA mapping vs manual grouped
    reference, SDPA enable_gqa fallback path, tokenizer integration tests using
    the real NTokenizer v0.3 bindings (IDs verified verbatim; special-token IDs
    262137–262143 verified).
- **Full vocabulary everywhere:** no test uses a reduced vocab; embedding,
  head, and logits all run at 262,144.
- **Tolerance policy:** fp32 paths use 1e-5/1e-6-class tolerances (headroom over
  measured values, never "whatever passes"); bf16/fp16 use looser dtype-based
  bounds. Tightening performed during audit (cached/uncached 2e-4 → 1e-5;
  attention 2e-5 → 1e-5 after measuring 6e-7 worst case).
- **Warnings:** none suppressed. The packaging deprecation warning was fixed by
  using an SPDX `license` string; torch's “NumPy not installed” notice was
  eliminated by installing numpy, not by filtering.

**Test coverage map**: config validation (28 error branches + roundtrip),
RMSNorm (math/stability/autograd), RoPE (freq math, partial rotary, offsets,
capacity errors, norm preservation), attention (causality, GQA mapping, three
decoder regimes, fallback, reference equivalence, backward), block (residual
identity, gradients), cache (lazy init, commit protocol, overflow, reset/reuse,
batch/device isolation), generation (greedy determinism, EOS early stop incl.
batched partial stop, top-k/temperature, device-checked generator), serialization
(bitwise roundtrip, tied-once serialization, missing/unexpected/shape/dtype/
format-version/architecture/config-key errors, atomicity), tokenizer integration.

---

## 6. Measured performance (this machine)

Dev preset, full 262,144 vocab. Latency via `perf_counter` with CUDA
synchronization; memory via `torch.cuda.max_memory_allocated`. Rows are
measurements, not commitments.

**CUDA fp32 (RTX 4060 Laptop, torch 2.15.0+cu132)**

| Workload | Result |
|---|---|
| fwd B=1 T=64 | 2.00–2.11 ms (0.031–0.033 ms/token) |
| fwd B=4 T=64 | 2.84–2.94 ms |
| fwd B=1 T=512 | 3.63–4.14 ms |
| fwd B=2 T=512 | 6.07–6.44 ms |
| fwd+bwd B=1 T=64 | 6.71–6.86 ms |
| decode 16 tok, cache on (one 16-token call) | 3.44–3.59 ms/token |
| decode 16 tok, cache off | 1.11 ms/token |
| peak mem fwd B=1 T=64 / B=2 T=512 | 115.5 / 1080.6 MiB |
| peak mem fwd+bwd B=1 T=64 / B=2 T=512 | 365.8 / 5207.1 MiB |
| Attention kernels actually executed (profiler) | `fmha_cutlassF_f32_aligned_64x64_rf_sm80 (MemEfficient)`; GEMMs `ampere_sgemm_*` |

**CUDA bf16**

| Workload | Result |
|---|---|
| fwd B=1 T=64 | 2.04 ms |
| fwd B=4 T=64 | 2.53 ms |
| fwd B=1 T=512 | 3.08 ms |
| fwd B=2 T=512 | 5.39 ms |
| fwd+bwd B=1 T=64 | 6.63 ms |
| decode 16 tok, cache on / off | 7.67 / 1.95 ms/token |
| peak mem fwd B=1 T=64 / B=2 T=512 | 61.8 / 544.4 MiB |
| peak mem fwd+bwd B=1 T=64 / B=2 T=512 | 183.2 / 2608.6 MiB |
| Attention kernels (profiler) | `pytorch_flash::flash_fwd_kernel` observed in bf16 prefill |

**CPU fp32 (this machine, no GPU paths)**

| Workload | Result |
|---|---|
| fwd B=1 T=64 / B=4 T=64 | 18.0 / 60.6 ms |
| fwd B=1 T=512 / B=2 T=512 | 114.5 / 199.5 ms |
| fwd+bwd B=1 T=64 | 117.7 ms |
| decode 16 tok, cache on / off | 3.85 / 7.93 ms/token |

Honest readings:
- On this tiny (9.8M) dev model, the KV cache **loses** on CUDA for short
  generations (preallocated 2048-slot buffers + update copies cost more than
  recomputing 32 tokens of a tiny network) and **wins** on CPU (3.85 vs 7.93
  ms/token). Cache benefit scales with model size and context length; the dev
  preset demonstrates it on CPU and the crossover is expected to invert for a
  larger model. Not a defect; reported as measured.
- bf16 ≈ fp32 latency at this scale (GEMM-dominated by the V×F logits projection);
  bf16 fwd+bwd memory is ~2× lower (2609 vs 5207 MiB at B=2 T=512).
- fp32 logits backward is memory-heavy: at B=2, T=512 the full logits tensor
  (2·512·262144·4 B ≈ 1.07 GB) plus its gradient dominate peak memory. This is
  the strongest argument for a training-phase fused/chunked loss — out of scope
  here by instruction.
- `logits_chunk_size` reduces only transient projection workspace (measured
  output differences ≤ ~6e-6, GEMM tiling ulps); the materialized
  `[B,T,V]` logits tensor remains full-size either way.

---

## 7. Phase 4 audit — what was found and fixed

| # | Finding | Severity | Fix |
|---|---|---|---|
| 1 | `RotaryEmbedding._apply(x, cos, sin)` shadowed `nn.Module._apply(fn)` → `.to()` crashed (TypeError) | high (crash) | renamed `_apply_rotation`; regression covered by full suite |
| 2 | `reference_attention` scored in input dtype (bf16) | medium | scores in fp32 |
| 3 | `OutputProjection.forward(fp32_logits=True)` upcast after `out_proj` (mixed dtypes crash) | high (crash) | whole head computed in fp32 when set |
| 4 | `config.dtype` declared but inert; `.to(dtype)` would cast RoPE fp32 tables to bf16 | high (silent precision loss) | config dtype applied at construction (parameters only); RoPE tables pinned fp32 via `_apply` override; checkpoint records actual dtype, warns on declared≠actual, load rejects file/config dtype mismatch |
| 5 | Test-created heads with different RNG weights compared (false failure) | test bug | shared weights via `load_state_dict` |
| 6 | Tuple-unpack bugs in `num_parameters()` (library) and test | high (crash) | fixed |
| 7 | Cache test called `commit(5)` after two `update()` calls (pending=2) | test bug | proper commit sequence |
| 8 | Chunked-vs-unchunked logits asserted bitwise equality (GEMM tiling) | test bug | measured ≤6e-6; tolerance 1e-5 with justification |
| 9 | `sdpa_reference` mask created on CPU vs CUDA tensors | test bug | device-parameterized |
| 10 | `CacheOverflowError` on commit before any update path untested | low | tested |
| 11 | `_check_no_shared_storage` keyed on pointer+size (missed storage aliasing via views) | medium | storage-level identity check |
| 12 | Batched partial-EOS generation path untested | low | test added (row padded while others continue) |
| 13 | Tolerances 2e-4/2e-5 vs measured ≤6e-7 | hygiene | tightened to 1e-5/1e-6; comment-referenced measurements |

Post-fix status: 218 collected, 217 passed, 1 skip; CUDA suite fully green.

Standing residual notes: embedding ID bounds-check costs one reduction per
forward (deliberate V1 correctness choice); chunked-prefill mask is rebuilt per
call (documented, measure before optimizing); `.item()` syncs exist in
RMSNorm-style reductions only where dtype/size logic demands, none on the hot
logit path.

---

## 8. NTokenizer compatibility — verified evidence

- Real bindings (`../NTokenizer/python/neuratal_tokenizer`) load
  `output/v0.3/ntokenizer-0.3-256k.ntok`; `vocab_size == 262144` confirmed;
  special-token IDs verified exactly: bos 262137, eos 262138, pad 262139,
  unk 262140, system 262141, user 262142, assistant 262143.
- `encode("Hello, world!") == [168069, 5431, 33]`; roundtrip exact.
- Integration tests feed real tokenizer IDs (including special-token IDs) into
  the model on CPU and CUDA: forward succeeds, logits `[B, T, 262144]`, finite.
- The model never remaps IDs, never injects special tokens, never imports the
  tokenizer package (only `examples/inference.py` uses it).

## 9. Known limitations

1. V1 is **untrained**; no quality claims. Checkpoints of a trained model do not exist.
2. `enable_gqa` verified on one build (torch 2.15.0+cu132, sm_89); other builds
   fall back explicitly with a warning (math identical). Flash performance with
   GQA is backend-dependent; observed `flash_fwd_kernel` on CUDA bf16 prefill only.
3. Logits storage is full-size `[B,T,V]`; `logits_chunk_size` does not fix that
   (measured fwd+bwd peak 5.2 GB at B=2 T=512 fp32). Fused CE is training-phase work.
4. `hidden_dropout` applies to residual-branch outputs only; embedding dropout
   not implemented (choice, documented); default dropout is 0.0 everywhere.
5. `generate()` is greedy/sampling only; no beam search, no serving; generation
   with a cache is slower than cacheless at this dev-preset scale on CUDA (measured).
6. Embedding ID bounds-checking synchronizes (CPU↔GPU) per forward — deliberate
   V1 trade; make it a validated fast path in the training phase if profiling demands.
7. CUDA numbers are this machine's; other GPUs may dispatch different SDPA
   backends (cuDNN attention could be selected there).
8. `reference` attention implementation is a debugging aid, not a fast path;
   do not benchmark it as such.
9. fp16 tested on CUDA only; bf16 forward/backward tested; bf16 all-reduce/
   distributed paths untested (no distributed anything in V1).
10. Safetensors CPU load tested; other platforms/endianess are supported by
    format, but untested here.

## 10. Deferred techniques (with re-open triggers — unchanged from audit)

FlashAttention → **implemented via SDPA**. Deferred: FlexAttention (need custom
masks), MLA (long-context milestone), Gated/Kimi Delta attention (research
branch only), Mamba-style hybrids (architecture milestone), multi-token
prediction (training phase), adaptive softmax (if head FLOPs dominate training),
Muon/NorMuon (optimizer phase), sparse attention (long-context milestone),
fused cross-entropy (training pipeline), rope_theta=500000 long-context variant.
V1 keeps the attention and MLP behind narrow module interfaces so each can be
introduced without rewriting the model.

## 11. Future extension points

- `RotaryEmbeddingBase` (interface for future rope variants)
- `attention_implementation` registry seam (add `"flex"`, etc. later)
- `NeuratalConfig` is the single source of architecture truth; new sizes = new configs
- `OutputProjection` chunk/fp32 knobs; fused loss slots into the same call site
- `NeuratalModel.parameter_report()` for training-phase budget accounting
- Serialization `format_version=1`; backward-compat loaders gated on version

## 12. Amendment compliance checklist

| Amendment | Status |
|---|---|
| §1 defaults (rope_theta 1e4, rms_norm_eps 1e-5, dropouts 0) | implemented + tested |
| §2 dev preset (V 262144, D 256, L 2, H 4, Hkv 2, F 32, I 640, ctx 2048) | implemented, exact parameter count 9,782,528 |
| §3 factorized tied mandatory; single matrix; shared matrix passed at call time; no untied path | enforced by construction + validation error + tests (incl. checkpoint-once check) |
| §4 cached attention correctness (position-aware lower-right mask, offsets, overflow tests) | implemented; regime dispatch; bitwise/6e-7 parity measured and tested (prefill 0.0; token-by-token ≤6e-7; chunked-prefill equivalence; overflow, reset, batch isolation tested) |
| §5 GQA enable_gqa + divisibility validation + explicit tested fallback + documented tolerances | done; fallback unit-tested with a raising SDPA double; equivalence ≤1e-5 fp32 |
| §6 safetensors: tied matrix serialized once; tying effective after load; before/after equivalence | verified (keys, bytes, bitwise logits roundtrip) |
| §7 F=64/D=512/L=11 reference only; implemented model computes its own counts | yes — measured 47,853,056 |
| §8 chunked output exact; workspace-only; no approximations; no fused CE | yes; documented in module + README; tests 1e-5-tolerance vs measured ≤6e-6 |
| §9 boundaries (no training/optimizers/pipeline; phases 2–5 only) | respected |

## 13. How to reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python -m pytest tests/                  # full suite, CPU+CUDA if present
python -m pytest tests/ -k cuda          # CUDA-only
python examples/inference.py --prompt "Hello, world!"
python examples/benchmark.py --device cuda --dtype bfloat16
```

Environment recorded: Python 3.14.4 · torch 2.15.0+cu132 · safetensors 0.8.0 ·
pytest 9.2.0 · numpy 2.x · NVIDIA GeForce RTX 4060 Laptop GPU (sm_89) ·
NVIDIA driver 595.99.02 · CUDA 13.2 · torch built against cu132.
