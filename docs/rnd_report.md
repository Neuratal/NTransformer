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

## Iteration 2 — CUDA graphs for the decode step (rejected, analysis)

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
