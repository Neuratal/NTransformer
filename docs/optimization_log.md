# Optimization Log

Chronological record of bottleneck analysis and optimization attempts for the
Neuratal Transformer V1 integration phase. Every entry states the measured
basis, the change, and the measured outcome. "No change" is a legitimate,
recorded outcome. Nothing here was shipped without a measurement.

## Baseline profile (2026-11-16, pre-optimization)

Dev preset, fp32, B=1 T=128, CUDA, torch 2.15.0+cu132 (torch.profiler,
`Self CUDA time total` per forward ≈ 1.11 ms; 5 forwards shown):

| Op | CUDA total | Share |
|---|---:|---:|
| `aten::mm` (dominated by `ampere_sgemm_64x64_tn` = the V×F logits GEMM at ~705 µs/fwd) | 4.42 ms | ~80% |
| Elementwise (`aten::mul` + kernels) | 0.21 ms | ~3.8% |
| `_efficient_attention_forward` (MemEff, fp32 causal GQA) | 0.19 ms | ~3.4% |
| `aten::gather` (embedding lookup) | 0.14 ms | ~2.5% |
| `aten::mean` / reduce | 0.07 ms | ~1.3% |
| everything else | ~0.5 ms | ~9% |

CPU total 21.6 ms vs CUDA 5.6 ms (5 forwards) — **the model is launch-bound at
this scale**: GPU waits on Python far more than Python waits on GPU.

Cross-checks: ~80% of device time is cuBLAS; attention is 3–4%; optimizer-free
eager mode launches ~17 mm + ~40 elementwise/reduce ops per forward.

## Attempt 1 — `torch.compile` (default-mode) for prefill

- Hypothesis: eliminating Python launch overhead (the dominant term) ~2× the
  tiny model's wall latency.
- Measured: eager 2.19 → compiled 1.38 ms (−37%), synced compile 1.20 ms (−43%).
  Numerics vs eager: max |Δ| = 8.9e-08 (well inside fp32 expectations).
- Cache-decode path: **failure** — 2,884 ms for 16 decode tokens (recompile
  churn around the mutable KV cache + shape specialization). Not usable for
  generation.
- Decision: NOT adopted as default; never wired into `generate()` or config.
  Documented as an experimental prefill-only opt-in (`torch.compile(model)` at
  the call site, if desired). The graph break originally came from
  `_validate_token_ids`/`RotaryEmbedding._slice` (`.item()`), which attempt 2
  removes from the hot path.

## Attempt 2 — remove host-side value validation from the CUDA hot path (ADOPTED)

- Measured bottleneck: profiling showed no single op dominated; the cost was
  CPU-GPU serialization via two `.item()` calls per forward
  (`RotaryEmbedding._slice` positions min/max; `embeddings._validate_token_ids`
  min/max over ids), each forcing the CPU to wait for the GPU per layer per
  forward.
- Change: model.forward enforces `past_length + seq_len ≤ max_position_embeddings`
  with int arithmetic (no sync); position and id *value* checks now run only on
  CPU tensors (free there); CUDA callers still fail loudly — ids via the
  embedding's kernel bounds check, positions via index_select — though with
  kernel-level errors rather than Python `ValueError`s.
- Correctness: all 268 tests pass unchanged (CPU paths validate identically);
  new test `TestContextCapacity` pins the int-level limit error.
- Measured outcome (synced per-iteration):
  - fp32 B=1 T=128: 2.33 → 2.10–2.23 ms (≈ −10%, run-dependent)
  - cached decode: 0.88 → 0.67 ms/token (≈ −24%)
  - pipelined (non-synced) forward: 2.19 → 1.17 ms (context only; synced is
    the reported benchmark)
- Retained: yes.

## Attempt 3 — backend priority override for bf16 attention

- Measured (`benchmark_attention.py`): bf16 T=256 GQA prefill — cudnn 0.023 ms,
  flash 0.039 ms, efficient 0.041 ms. PyTorch default picks flash on this build.
- Considered forcing cuDNN via `sdpa_kernel` or backend priority flags.
- Decision: rejected as a global change — attention is ~3% of forward time;
  backend heuristics are shape-dependent; overriding global priority for 0.01
  ms/step is not justified and risks regressing other shapes. Recorded as a
  measured observation; users may opt in locally if bf16 attention dominates
  their profile.

## Attempt 4 — eliminate remaining reductions / elementwise ops

- RMSNorm does pow→mean→rsqrt→mul→mul→cast per call; RoPE does index_select +
  two cats. Fusion candidates are real but bounded (both < 4% of device time
  together).
- Decision: rejected — hand-fusion adds maintenance cost and numerical risk for
  a <4% ceiling; the correct fusion path is torch.compile (attempt 1), which is
  available opt-in. No change.

## Attempt 5 — KV cache reuse across generate() calls

- The preallocated 2048-slot cache zeros 16 MB at first use per KVCache; decode
  at T=1 is launch-bound.
- Considered: reusing one KVCache across generate() calls in the example.
- Decision: rejected — a fresh cache per generate() call is the safe contract
  (no cross-sequence leakage by construction); reuse is opt-in via explicit
  cache passing, already documented in KVCache's contract. No change.

## What the optimization hunt confirmed (negative results)

- Logits GEMM (~63% of device time at T=128) is cuBLAS-bound; no win available
  without changing the math.
- Attention is 3–4% of device time; backend tweaks are noise for the model.
- The integration boundary (tokenizer → ids → model) costs ~0.02–0.04 ms per
  prompt encode — below noise vs model time; no allocation or bridging win
  available at this scale.
- Memory: dev preset fp32 forward peaks 183.5 MiB @ T=64 and 1080 MiB @ T=512;
  the 47.85M reference fits in 502 MiB @ T=128 fp32 — no memory crisis to fix.
  The known spike remains the B×T×262144 logits tensor in backward (measured
  5.2 GB at B=2 T=512 training-like forward+backward) — explicitly out of scope
  (fused cross-entropy deferred with the rest of the training phase).

## Final state

Measured improvements over baseline (synced medians, this machine):
fp32 prefill T=128 −10%, cached decode −24%. Everything else measured within
run-to-run noise; three attempts rejected with reasons recorded. All 268 tests
pass; cached/uncached equivalence remains bitwise at 0.00e+00 in the KV-cache
benchmark.

---

# R&D loop (2026-11-16, after integration)

See `docs/rnd_report.md` for the full loop. Summary:
- v0.3 PROMOTED: fused qkv_proj + gate_up_proj (fp32 prefill win ~4-7% dev /
  ~22% ref-50M, T=512, 3/3 rounds); 276 tests pass; legacy migration tested.
- CUDA graphs for decode: rejected (static-shape incompatibility; analysis in rnd_report).
- Mask caching: rejected (0.018 ms/call, <0.1%).
- torch.compile reduce-overhead for prefill: measured 0.946 ms (vs 1.003 default,
  2.10-2.23 eager) — opt-in only, never in generate().
- bf16 wins unproven (clock variance); decode unchanged (~0.62-0.67 ms/tok).

## R&D loop additions (post-v0.4)
- fp32_attention candidate: rejected by adversarial probe (drift ~0.4–0.5%
  relative at all scales ≤ 1e5; bf16 SDPA == bf16 reference exactly).
- Batch scaling B∈{1,2,4,8} measured (T=64/128, fp32/bf16); plateau re-verified
  with all benchmark axes (see rnd_report Iteration 8).


## R&D loop additions (frontier 2026 references)
- v0.5-candidate (GDN): implemented, tested, benchmarked — rejected for general
  promotion (latency regression everywhere at dev scale; 32x memory win noted).
  Opt-in only. No default changes; v0.4 remains best.

## R&D loop additions (frontier 2026 references)
- v0.6-candidate (MLA): implemented + tested + benchmarked — rejected for
  general adoption (latency regression everywhere at dev scale; 4x KV memory
  win real but unexploitable here). Branch-only candidate; best stays v0.4.

## R&D loop additions (iteration 11)
- SWA (v0.7-candidate): rejected on 3-seed timings (wash T=1024, 20-40% slower T=2048)
- bf16 decode GEMV: micro 2.5x faster (229->287 GB/s), no end-to-end latency win
  at launch-bound scale — config-level recommendation only

## v0.6.1 WY chunkwise GDN
- faithful Eq.6-9 form; verified vs oracle; 5-13x over sequential GDN
- dense GQA still wins everywhere at dev scale (prefill and decode); memory is
  the only GDN axis (constant state); NOT promoted; archived
