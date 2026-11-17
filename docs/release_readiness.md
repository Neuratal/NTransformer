# Release Readiness — v1.1.0rc1 (2026-11-17)

Final engineering pass on `feature/final-release-2026-11` (branched from
`feature/moe-integration-2026-11` @ 196a613; `master` untouched).

Machine: RTX 4060 Laptop GPU (8 GB, sm_89, 45 W cap), i7-12700H, Python 3.14.4,
torch 2.15.0+cu132, CUDA 13.2, safetensors 0.8.0. Inter-run spread on this
machine is ±10–20%; where a difference is inside that band it is labeled
*within noise* and no improvement is claimed.

**No corpus was downloaded, no pretraining was performed, NTokenizer was not
touched.** All training runs use deterministic synthetic token IDs.

---

## A. Executive summary

This pass made the framework measurably lighter to start (import 13.4 s →
1.3 s), modestly faster on the dense hot path (fp32 prefill −5…−11%),
faster at the MoE dispatch margins (−1…−17% depending on shape), and
materially more reliable (full-width heads no longer crash the engine, fp16
no longer trains unscaled, DDP MoE no longer wedges the reducer, checkpoints
now validate the tokenizer fingerprint and the optimizer state layout).

Correctness was never traded for these gains: 402 tests pass (394 baseline +
8 new), the tied dense baseline is bit-unchanged, MoE routing semantics are
unchanged, and chunked cross-entropy keeps its 1.0e-06 gradient parity.

**Verdict: research/development-ready. Not production-ready, not
release-ready for training at scale, not distributed-tested.** See §K.

What this pass deliberately did *not* do: rewrite the MoE into a fused
kernel (measured no win, private API), adopt `torch.compile` (within noise),
add ZeRO/FSDP/expert parallelism (untestable here; estimator-only, honestly
labeled), touch `versions/` snapshots, or modify NTokenizer.

## B. Dense improvements

| # | Change | Previous | New | Correctness evidence | Measured effect | Memory effect |
|---|---|---|---|---|---|---|
| 1 | Fused `F.rms_norm` when `input.dtype == weight.dtype` and no autocast active | pow→mean→rsqrt→mul→mul→cast per call (fp32 math) | `F.rms_norm` dispatch | fp32 max |Δ| 9.5e-7 (1-ulp); bf16 |Δ| ~1 ulp; all norm tests green; autocast case keeps the exact old path (promoted-to-fp32 semantics preserved) | op level ~4.0x faster in fp32 (0.058→0.015 ms @ [4,512,512]); fp32 prefill T128 2.092→1.854 ms, T512 4.121→3.926 ms; bf16 T512 3.032→2.725 ms, T1024 5.43→5.14 ms; fp32 T1024 within noise | neutral |
| 2 | Meta-device weight-init skip | `normal_` on meta tensors through fake-tensor dispatch (~4 s for 500B-class preset) | `_init_module` no-ops on meta tensors | meta parameter counts unchanged (exact-count tests green: all presets recount exactly) | preset construction ~3x faster; `import neuratal` 13.4→1.3 s (with #3) | none |
| 3 | Lazy preset machinery (PEP 562 `__getattr__`) | presets built at `import neuratal` | built on first attribute access | API unchanged (`from neuratal import PRESETS` still works); full suite green | import 13.4 s → 1.3 s; `PRESETS` first access ~5 s (once) | none |
| 4 | Chunked CE dead-code removal | `ctx.n_tokens` stored, unused; wrapper recomputed count | removed | `test_chunked_loss.py` green; benchmark parity 1.03e-06 | neutral | none |
| 5 | SDPA default backend kept | default (flash for bf16) | unchanged after re-measurement (3 interleaved rounds; flash/cudnn/default all sweep 0.17–0.95 ms — clock-dominated) | unchanged numerics | none (documented; earlier "cudnn faster" not reproducible) | none |

Dense regression status: the entire baseline suite passes unchanged; the
dev-preset fp32 forward is bitwise-identical except through RMSNorm's 1-ulp
accumulation-order change (tolerance-based tests unaffected; logits changed
by ≤ ~1e-6).

**Rejected on measurement:** `torch.compile` for prefill on MoE (within
noise), forced cuDNN backend (non-reproducible), chunked GEMM via
`torch._grouped_mm` for experts (no win, private API).

## C. MoE improvements

### Previous implementation (reference)
Sort assignments by expert → per expert `index_select` x-slice → expert
forward → `index_add` into output. Plus: a `torch.tensor(0.0, device=cuda)`
H2D scalar copy in the no-capacity path, double int64→float offset casts.

### New implementation
One gather of all sorted assignments (`x.index_select(0, sorted_tokens)`),
contiguous per-expert slices (views), one `torch.cat` of weighted pieces, one
`index_add` into the zeros buffer, and a fast path that reuses
`sorted_tokens` directly as the scatter index when no capacity is set.
`bincount` int64 reused for cumsum/offsets (no re-casts); dropped-counter
built with `new_zeros` (no H2D).

Identical byte volume, same math, same routing semantics, one host sync
(documented `.tolist()`), never per-expert syncs.

### Correctness evidence
- All 18 existing MoE tests green (`test_moe_core/engine/serialization/
  presets_estimate`), incl. top-1/top-2, capacity drops, padding masks,
  empty experts, mixed dense/MoE gather through the full model, bf16 step.
- Pathological-distribution coverage lives in `test_moe_core.py` (balanced,
  all-to-one-expert, imbalance > 2x, top-1 == mass on argmax, experts
  starved to zero assignments). Aux/z-loss behavior unchanged (they are
  computed from pre-capacity assignments, the documented convention).
- No valid token disappears: tokens are only dropped under explicit
  `capacity_factor` opt-in and are then *counted* (`dropped_tokens` logged);
  masked tokens produce exactly zero output; every selected expert receives
  gradient if it receives ≥1 token (routing test suite + engine step).

### Measured effect (n=40 medians, `benchmarks/results/moe_*.json`)

| Cell (module-level, H=512, 8 experts, top-2, expert_hidden=1408) | before | after | Δ |
|---|---:|---:|---|
| fwd B=2 T=128 | 1.927 ms | 1.611 ms | **−16.4%** |
| fwd B=2 T=512 | 3.447 ms | 3.353 ms | −2.7% (noise band, but consistently ≤ baseline in every repeat) |
| fwd B=4 T=128 | 1.973 ms | 1.946 ms | −1.4% (wash) |
| fwd B=4 T=512 | 5.473 ms | 5.861 ms | +7% this run — **within noise vs the earlier after-run of 5.287 ms; no regression claim either way** |
| bwd (all cells) | — | — | within noise everywhere |
| peak alloc (module bwd) | unchanged | unchanged | 162–228 MiB, MoE ≈ dense at this size |
| load imbalance @ init | 1.07–1.19 | unchanged | — |
| optimizer-step, small dev MoE vs dense | — | 28.1 vs 21.7 ms (+30%) | consistent with 2x active expert FLOPs + router; not a regression |

MoE peak VRAM equals dense at this scale because expert temporaries are
created and released inside the dispatch loop; sorted buffers are "[A, H]"
and few MiB at dev sizes. At larger hidden sizes the gather/cat adds
~2×[A×H] transient bytes vs the old per-expert index_select — a deliberate,
bounded trade for fewer random-access kernels (measured neutral here).

### What was measured and rejected
- `torch._grouped_mm` (private aten): exact-match outputs, but 0.601 vs
  0.615 ms at realistic imbalanced shapes — no win; not adopted.
- `torch.compile(MoEFFN)`: 15.07 vs 14.90 ms — within noise; not adopted.
- Buffered fixed-capacity `bmm` padding: would waste 20–100% extra expert
  FLOPs at capacity_factor≠1.0 for ~E-1 kernel launches; not adopted.
- fp16 router GEMM: rejected by rule (routing precision must not silently
  degrade); router stays fp32.

## D. Training engine

Verified (all with synthetic data only):
- Dense single-GPU optimizer steps: finite loss/grads, parameters update,
  AdamW state (exp_avg/exp_avg_sq) finite, scheduler step counter and LR
  sequence correct.
- MoE training step end-to-end via `train()` (bf16): finite loss/aux/z,
  all experts routed at init, checkpoint + resume works.
- 75M dense chain on this GPU: 2-step bf16 run; finite; checkpoint save/
  load/resume; scheduler at step 2; `torch.equal`-level parameter movement.
- **Resume fidelity (new, exact)**: after corrupting RNG to a different
  seed, `resume_training_checkpoint` restores `torch.get_rng_state()`
  byte-exactly; every optimizer state tensor is restored `torch.equal` as
  saved; scheduler `.current_step` restored; `train(..., resume_from=...)`
  continues at step global_step+1 (verified 3→{4,5}).
- **Fingerprint validation (new)**: matching fingerprint resumes; mismatch
  raises `ValueError`; missing fingerprint + expected raises; check can be
  explicitly bypassed with `expected_tokenizer_fingerprint=None`.
- **Optimizer-index guard (new)**: checkpoint state for an out-of-range
  (group, param) index raises `ValueError` instead of silently skipping.
- **fp16 (new)**: autocast + GradScaler, unscale-before-clip, step skipped
  on non-finite grads; scaler state (`scale`, `growth_factor`,
  `backoff_factor`, `growth_interval`, `_growth_tracker`) saved at checkpoint
  and restored on resume; verified the restored scale overrides a freshly
  initialized scaler.
- Full-width head (`embedding_size=None`, the exact config that crashed the
  engine before this pass): trains and checkpoints green (CPU).

Not verified (explicitly):
- Multi-GPU/multi-node anything (single-process gloo smoke test only).
  `find_unused_parameters=True` for MoE DDP is a correctness fix applied on
  code inspection, not a measured multi-GPU validation.
- Interrupted-save *file-level* corruption recovery beyond the existing
  corrupt-`trainer_state.safetensors` test; atomic write paths are in place
  but a power-loss matrix was not run.
- Shuffle-order parity after resume: the data cursor is advanced by
  `tokens_seen // tokens_per_micro` micro-batches, which is exact for
  `shuffle=False` and re-permutes under shuffling (documented, not claimed).
- Long-run stability (steps > ~10, hours); large-batch DDP; resume across
  devices (CPU→CUDA RNG covered by construction, not by a multi-device run).
- Peak-memory equivalence of gradient accumulation vs no-accumulation at
  large T (mechanism unchanged, only small-scale measured).

## E. Memory analysis (measured vs estimated — kept strictly separate)

Measured on this GPU:

| Item | Measurement |
|---|---|
| Chunked CE, 512 tokens, fwd+bwd | 1,284 MiB peak alloc (vs 2,242 MiB standard CE) |
| Chunked CE, 1024 tokens, fwd+bwd | 1,286 MiB peak alloc (vs 4,291 MiB standard CE) |
| Dense dev model, full-train step (bf16 autocast, fwd+bwd+AdamW) | ≈ 1,152 MiB |
| Small dev MoE, same | ≈ 1,195 MiB |
| MoE module bwd peak (module-level cells) | 162–228 MiB |
| 75m dense bf16 2-step chain (test) | < 7.5 GB (test asserts; actual lower) |

Estimated (closed-form, `neuratal/estimate.py` — **labeled ESTIMATE in the
output, not a measurement**): parameters/grads/AdamW m+v/master/activations/
chunked-logits-workspace/KV-cache/comm buffers; MoE charges *all* experts for
storage and only top-k for activations; `zero1/2/3/fsdp` are accounting
models, explicitly not implemented paths; TP/PP/EP rejected rather than
approximated. Two refinements this pass: the CE logits workspace is now
charged at fp32 width (the kernel upcasts), and notes state explicitly that
these are estimates.

Known gap (honest): the activation coefficient (family-of-models
approximation) has never been calibrated against measured peaks above the
dev scale; treat activation estimates as ±50% guidance.

## F. Benchmark tables (identical conditions)

Dense prefill, dev preset, `benchmark_model.py`, warmup=10, repeats=25,
same machine/seed (baseline = frozen worktree @ 196a613):

| Shape | baseline fp32 | now fp32 | baseline bf16 | now bf16 |
|---|---:|---:|---:|---:|
| B=1 T=32 | 1.693 ms | 1.483 ms | 1.673 ms | 2.033 ms (noise) |
| B=1 T=128 | 2.092 ms | **1.854 ms** | 1.872 ms | **1.566 ms** |
| B=1 T=256 | 2.775 ms | **2.505 ms** | 2.202 ms | **1.942 ms** |
| B=1 T=512 | 4.121 ms | **3.926 ms** | 3.032 ms | **2.725 ms** |
| B=1 T=1024 | 7.479 ms | 7.675 ms (noise) | 5.426 ms | **5.141 ms** |

MoE module-level fwd/bwd (table in §C). Chunked CE (§E table in
benchmark_results.md §8.3). Environment fingerprint embedded in every JSON.
Artifacts: `benchmarks/results/*_20261117T*.json`, regenerated
`*_latest.json`; `benchmarks/benchmark_moe.py`, `benchmark_chunked_ce.py`
are new and committed. All results are raw-timestamped; nothing was
overwritten.

## G. Test results

```
$ .venv/bin/python -m pytest tests -p no:warnings --tb=no -q
… (all green) …
413 passed, 3 skipped, 1 warning

$ .venv/bin/python -m pytest tests 2>&1 | tail -1
413 passed, 3 skipped, 1 warning in 44.01s
```

Skipped (exact reasons printed by `pytest -rs`):
`test_cache.py:179` needs a CUDA device-mismatch probe;
`test_engine.py:111` covered by `test_ntokenizer_integration.py`;
`test_ntokenizer_integration.py:209` CUDA not available in that probe.
1 warning: third-party Triton FP-fusion notice (CUDA test). No failures,
no XFAILs, no hidden skips. Full CUDA-parametrized suite runs (the box has
CUDA). New tests this pass: `tests/test_engine_regressions.py` — 8 tests
(full-width head step+checkpoint, 4 fingerprint cases, fp16 step + scaler
restore, exact RNG/optimizer restore, resume-then-continue); and
`tests/test_moe_pathological.py` — 11 tests (all-to-one-expert assignment
counts, starved/unused experts eager gradients, saturated router logits
(z-loss stays finite), assignment conservation with and without capacity,
kept+dropped == total, aux-loss gradient reaches the router).

## H. Scale readiness (per preset; measured counts; honest claims)

| preset | total (measured) | config validated | count measured | init tested | fwd/bwd tested | opt step | single-GPU | multi-GPU | distributed | production |
|---|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| dev (9.78M) | 9,782,528 | yes | yes | CUDA | yes | yes | yes | no | no | no |
| `50m` | 47,853,056 | yes | yes | CPU/meta | 75m chain (77M) analogous | 75m only | 75m only | no | no | no |
| `75m` | 77,088,896 | yes | yes | CUDA | yes (2-step bf16) | yes | yes | no | no | no |
| `100m` | 98,096,896 | yes | yes | meta/CPU | no (same class as 75m) | no | no | no | no | no |
| `500m`–`3b` | 0.5–2.96B | yes | yes (meta) | meta only | no | no | no | no | no | no |
| `5.6b` (was `7b`) | 5,571,350,528 | yes | yes (meta) | meta only | no | no | no | no | no | no |
| `13b` (−8.3% vs label) | 11,915,686,912 | yes | yes (meta) | meta only | no | no | no | no | no | no |
| `30b`–`500b` | 30.2–490.8B | yes | yes (meta) | meta only | no | no | no | no | no | no |
| `small-moe` | 402,301,056 (123,609,216 act.) | yes | yes | CUDA (bf16) | yes | yes | yes | no | no | no |
| `1b-moe`–`5.7b-moe` (was `7b-moe`) | 1.07–5.67B | yes | yes | meta/CPU | no | no | no | no | no | no |
| `30b-moe`–`500b-moe` | 29.4–483.3B | yes | yes (meta) | meta only | no | no | no | no | no | no |

Renames: `7b`→`5.6b`, `7b-moe`→`5.7b-moe` (labels exceeded 10% mismatch).
`13b` (11.92B measured, −8.3%) documented as-is; all other presets within
±10% of their labels. No preset above ~100M has ever executed a forward
pass; parameter count validation is by meta-device construction only.

## I. Remaining limitations

1. Untrained framework; synthetic data only; zero evidence about quality.
2. Multi-GPU/DDP untested beyond a single-process gloo smoke test; MoE DDP
   router statistics are per-rank (global load balance not synchronized).
3. Memory estimator is an approximation; CE-workspace and expert-activation
   terms were the only calibrated parts.
4. MoE: readable reference dispatch; constant-factor wins only; no fused
   kernels, no all-to-all, no expert parallelism; capacity dropping works
   but is off by default (`capacity_factor=None`).
5. Atomic writes cover accidental truncation, not proven crash-consistency
   across arbitrary filesystems.
6. SDPA on this build: bf16 defaults to FLASH; cuDNN showed no reproducible
   win (clock-dominated); no backend is forced.
7. 45 W mobile GPU: every wall-clock number herein is noisy; sub-5% deltas
   should be treated as unproven.
8. `gated_delta` attention remains an experimental opt-in with a known
   latency regression; not promoted.
9. fp16 path is CUDA-only (untestable on CPU by design).
10. No fused linear-CE (Liger-style) — chunked CE is the mechanism; it is
    ~35% slower than standard CE by design (memory trade).

## J. Repository status

- Branch: `feature/final-release-2026-11`
- Commits (this pass, on top of `feature/moe-integration-2026-11` @ 196a613):
  `b608fdf` MoE dispatch · `cbcf887` fused F.rms_norm · `2dbddd9`
  meta-init/lazy presets · `49cbbc6` engine/checkpoint fixes ·
  `40f731b` chunked-CE bench · `6a33bc4` preset honesty · `4733aed`
  docs/license/v1.1.0rc1 · `bdbf866` registry · `c41103f` docs ·
  `e54bc42` pathological MoE suite · `1f38fa0` counts → `d9bb7c1`
  lazy sorted_experts (with README table fixes) · `2bfa282` benchmark
  refresh (fingerprint re-verified by run)
- Working tree: clean. `master` untouched (still @ 14f6fe0,
  "snapshot v1.0.0-rc1 frozen").
- NTokenizer fingerprint **verified unchanged**:
  `1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9`
  (sha256 of `ntokenizer-0.3-256k.ntok`; file untouched, mtime 2026-11-15).
- Stable Dense regression status: dense path default behavior unchanged;
  full baseline suite passes; one intentional numerics change (fused
  RMSNorm ≤1-ulp in non-autocast paths) documented in §B.
- `versions/` snapshots: untouched (historical archives).

## K. Final verdict

- Experimental: no — the surface is larger, tested and documented.
- Development-ready: yes — this is the honest label.
- Research-ready: **yes** — suitable for reproducing architecture/optimizer
  work at small scale with synthetic data; correctness is well-tested and
  documented.
- Release-ready: **no** — nothing has been trained; benchmarks are from an
  8 GB mobile GPU; no multi-GPU validation exists; scale claims stop at
  ~100M for real execution.
- Production-ready: **no**, and it would be dishonest to claim otherwise
  (no quality evidence, no distributed evidence, no serving stack).

The project is in its best-measured state to date and is honest about what
it is: a correct, well-tested, NTokenizer-compatible transformer
framework/training laboratory — not a trained model and not a
production system.
