# NTransformer — NT-ENGINEERING-2026-FINAL Report

Date: 2026-11-17 · Branch: `feature/v1.3-enterprise-2026-11` (from v1.2.0-candidate @ `3976a5d`)
Machine: RTX 4060 Laptop GPU (8 GB, sm_89, 45 W), i7-12700H, Python 3.14.4, torch 2.15.0+cu132, CUDA 13.2.
NTokenizer stable v0.3-polished: **fingerprint verified unchanged** (`1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9`) before and after this audit.

---

## A. Executive summary

Previous audit (v1.2.0-candidate) claims were independently re-verified:
scheduler resume semantics, sharded-checkpoint byte count, RMSNorm autocast
numerics, and the test suite. The verification found one **silent correctness
defect that the previous fix did not fully close** — scheduler base LRs were
stored as float32, so every resumed cosine trajectory drifted ~1e-7 from the
uninterrupted run. It is now fixed (float64 storage) with **bit-exact** resume
equivalence demonstrated at 7 resume points. Sharded checkpoints now verify
per-shard SHA-256 (byte count alone cannot catch same-size stale replacements).
422 tests are green (4 new).

The "428 → 418" claim from the mission brief does not exist anywhere in the
repository or in my own reports (see §C): the true history is
**394 (v1.1.0) → 413 (v1.1.0rc1) → 418 (v1.2.0-candidate) → 422 (v1.3.0-candidate)**,
with **zero removed or renamed tests** at any point.

## B. Previous-report verification (what held, what did not)

| Claim (v1.2.0-candidate) | Verification | Result |
|---|---|---|
| 418 passed, 3 skipped, 1 warning | re-collected and re-run at v1.2.0-candidate (`3976a5d`): 421 collected, 418 passed, 3 skipped | **holds** |
| Resume reapplies `lr_at(step-1)` | independently re-examined; code correct in semantics but **not bit-exact** (float32 storage) | **superseded by B-2023-1** |
| Sharded total_size check sufficient | **insufficient** for same-size stale shard replacement | **superseded by B-2023-3** (hashes added) |
| RMSNorm autocast path ≤1-ulp | re-verified fwd+bwd across CPU/CUDA × fp32/bf16/fp16; fp32 CUDA 2.4e-07, bf16/fp16 0.0 | **holds** |
| 3 skips stable | identical at every commit examined (in-repo); **but** the skipped count depends on the machine-relative `NTOKENIZER_ROOT` default (40+ tests skip in moved checkouts) — documented | **holds with caveat** |
| Frozen snapshot intact | `versions/v1.2.0-candidate` manifest hashes verified last run | **holds** |
| Scheduler equivalence | only approximate ("close") before; now bitwise | **upgraded** |

## C. Test-count investigation ("428")

The mission brief states the previous report printed `428 → 418 final tests`.
**The string `428` appears nowhere in the repository history, any doc, any
test file, or my own reports** (searched `docs/`, `CHANGELOG.md`, `README.md`,
`registry/`, `versions/`). The v1.2.0-candidate report says `418`; the value
`428` is a transcription error in the mission brief (likely `413`/`418`
combined).

Independent node-ID inventories (pytest `--collect-only`, node IDs sorted,
`LC_ALL=C`):

| Tree | Commit | Collected | Passed | Failed | Skipped | Removed/Renamed |
|---|---|---:|---:|---:|---:|---:|
| pre-v1.1 | `196a613` | 397 | 394 | 0 | 3 | — |
| v1.1.0rc1 | `5d13af9` | 416 | 413 | 0 | 3 | **0** |
| v1.2.0-candidate | `3976a5d` | 421 | 418 | 0 | 3 | **0** |
| v1.3.0-candidate (this) | HEAD | 425 | 422 | 0 | 3 | **0** |

- Removed or renamed: **none** (empty `comm -23` diffs at both transitions).
- Added: +19 (v1.1.0rc1), +5 (v1.2.0-candidate), +4 (v1.3.0-candidate).
- Skipped: identical 3 tests at every point in the real checkout
  (`test_cache.py:179` CUDA device-mismatch probe, `test_engine.py:111`
  covered elsewhere, `test_ntokenizer_integration.py:209` CUDA-dependent).
- Environment caveat: in a moved checkout where the machine-relative
  `NTOKENIZER_ROOT` default cannot resolve, all tokenizer-dependent tests
  skip and the count becomes 40+; that is correct behavior (skip with reason,
  never silently pass), not a coverage change.
- **Test collection coverage did not decrease at any step.** No test was
  deleted to inflate or shrink counts; assertions were strengthened, never
  weakened, in the v1.2/v1.3 scheduler work.

Commands used:

```bash
cd <worktree> && PYTHONPATH=<worktree> <venv-python> -m pytest tests --collect-only -q
cd <worktree> && LC_ALL=C sort nodes.txt
lc -l nodes_196a613.txt nodes_5d13af9.txt nodes_3976a5d.txt
LC_ALL=C comm -23 nodes_a.txt nodes_b.txt   # removed
LC_ALL=C comm -13 nodes_a.txt nodes_b.txt   # added
```

## D. Scheduler resume equivalence (full verification)

See `docs/scheduler_resume_verification.md`. Summary:

- Identical model, optimizer (**differential per-group LRs**, 2e-3 ≥2-D / 1e-3
  <2-D), scheduler (warmup 4 / total 32), 32 fixed batches, CPU pinned to 1 thread.
- Resume at K ∈ {0, 1, 4, 5, 8, 15, 16}: **LR trajectory, parameters (bitwise),
  optimizer moments/step counters, scheduler counter, and resumed-segment
  losses are all identical** to uninterrupted training.
- Two defects found and fixed: (1) v1.2's `lr_at(step-1)` apply (kept —
  correct semantics) but float32 storage (fixed → float64, now bit-exact);
  (2) `load_state_dict` accepted malformed state (fixed — required keys,
  type/range/finiteness, base_lrs length).
- Documented semantics: `scheduler.step()` is per optimizer step, after it;
  `step` = number of applied scheduler steps; `lr_at(step > total)` sits at
  `min_lr_ratio` (cosine clamp), not an error.
- Full table and exact reproduction commands in the linked doc.

## E. Checkpoint integrity

See `docs/checkpoint_integrity.md`. 20 failure cases catalogued with outcomes,
including: same-size replaced shards (byte count passed — now caught by
per-shard SHA-256), missing hashes, truncated safetensors, architecture
mismatch, optimizer-index mismatch, fingerprint mismatch, CUDA RNG restore
(byte-exact). Atomic rename guarantees; explicit **non-claims**: no power-loss
fault injection; CRC32 is accidental-corruption protection only.

## F. NTransformer bug registry

See `docs/bug_registry.md` (v1.2 + v1.3 additions) and `docs/ultimate_audit/bug_registry.md`
(v1.2 detail). New this cycle: **B-2023-1** (float32 scheduler state → drift;
fixed), **B-2023-2** (scheduler `load_state_dict` validation; fixed),
**B-2023-3** (sharded same-size stale shard replacement; fixed via
`shard_hashes`). All with regression tests.

## G. Dense engineering report

- Architecture unchanged (pre-norm decoder, RMSNorm, RoPE, GQA SDPA, SwiGLU,
  factorized tied embeddings).
- RMSNorm autocast path (fp32 master weights + bf16 autocast) now uses the
  fused normalize-without-weight kernel + explicit fp32 weight multiply
  (~1.65× op; end-to-end within noise — reported honestly).
- Fused `F.rms_norm` for matched-dtype/no-autocast (v1.1, holds; fp32 ~4× op).
- Scheduler/resume fixes (v1.2/v1.3) — training-correctness, not perf.
- Benchmark noise policy: only interleaved in-process A/B counts as evidence;
  single-run deltas < 10 % are discarded.
- No measurable end-to-end Dense change is claimed for v1.3 beyond the v1.2
  norm-path op win (which remains within noise at the train-step level).

## H. MoE engineering report

- No MoE dispatch change this cycle (v1.1 gather/scatter-once and lazy
  `sorted_experts` remain the implementation).
- Pathological suite (v1.1) unchanged: 11 fixtures (all-to-one-expert, starved
  experts, saturated router, capacity conservation).
- MoE peak memory ≈ dense at dev scale; module fwd B2T128 −16 % vs v1.0
  baseline stands.
- DDP MoE correctness fix (find_unused_parameters) remains untested on real
  multi-GPU — explicitly labeled.

## I. Memory report (measured vs estimated)

Measured: dev dense full train step ≈ 1.15 GiB; MoE ≈ 1.20 GiB; chunked CE
1.29 GB @1024 tokens (vs 4.29 GB standard); MoE module bwd peak 162–228 MiB.
Estimated (labeled ESTIMATE): params/grads/AdamW m+v/master/activations/CE
workspace (fp32)/KV-cache/comm; MoE charges all experts for storage, top-k for
activations; `zero*/fsdp` are accounting models only; TP/PP/EP rejected.
Activation coefficient uncalibrated beyond dev scale (±50 % guidance).

## J. Preset report

Counts unchanged from v1.1 (exact meta-device): 50m 47,853,056 · 75m
77,088,896 · 100m 98,096,896 · 500m 496,760,320 · 1b 1,074,104,320 · 3b
2,960,139,008 · 5.6b 5,571,350,528 · 13b 11,915,686,912 (−8.3 %, documented) ·
30b 30,235,631,616 · 70b 68,488,011,776 · 100b 93,843,515,392 ·
200b 209,420,042,240 · 500b 490,777,300,992; MoE 402,301,056 (act
123,609,216) … 500b-moe 483,293,937,664 (act 77,419,528,192). No preset above
~100M has ever executed a forward pass on this machine.

## K. Test results

```bash
cd ../NTransformer && .venv/bin/python -m pytest tests 2>&1 | tail -1
422 passed, 3 skipped, 1 warning in 44.28s
```

Skips (identical at every audited point): `test_cache.py:179` (CUDA-only
device-mismatch probe), `test_engine.py:111` (covered in
`test_ntokenizer_integration.py`), `test_ntokenizer_integration.py:209` (needs
CUDA in that probe). 1 warning (third-party Triton FP-fusion).
Commands to reproduce: `python -m pytest tests -q`; skip reasons with
`-rs`. New v1.3 tests: 4 (sharded hashes ×3, legacy-index compat ×1).

## L. Benchmark results

See `docs/benchmark_results.md` (§8–9 addenda) and
`docs/ultimate_audit/benchmark_summary.md`. Attributable wins are op-level
only (norm paths) and one MoE module cell; everything else within noise.
`benchmark_model.py` re-runs after idle read 5–25 % fast purely from SM clock
state (2565/3105 MHz) — discarded as evidence.

## M. Rejected experiments

Carried forward from v1.2 (`docs/ultimate_audit/rejected_experiments.md`):
`torch.compile(MoEFFN)`, `_grouped_mm`, autocast composite-fused weighted
RMSNorm (worse numerics), cuDNN attention forcing, fixed-capacity bmm padding,
pre-allocated generation buffer, and all clock-state-artifact "wins".

## N. Remaining limitations

1. Untrained; synthetic data only; no quality evidence.
2. Multi-GPU untested (single-process gloo smoke only).
3. Activation estimator ±50 % beyond dev scale; treat as guidance.
4. No crash-consistency fault-injection; atomic rename is not power-loss proof.
5. NTokenizer integration test skips are environment-dependent (moved checkouts
   skip 40+; in-repo skip count is 3).
6. fp16 path CUDA-only; CPU training untested beyond synthetic chains.
7. GPU wall-clock noise ±10–20 %; sub-10 % deltas are unproven.
8. **NTokenizer's own repository has no git history** (all files staged, zero
   commits at audit time) — the single worst reliability risk across the two
   projects (see NTokenizer report).

## O. Version registry

`registry/best_version.json`: best = **v0.4** (unchanged; promotion rule not
met). Candidates: … v1.1.0-rc1 (not promoted), v1.2.0-candidate (not promoted),
**v1.3.0-candidate** (not promoted). NTokenizer candidate **v0.4.0-candidate**
recorded in NTokenizer's CHANGELOG (bindings-only; model unchanged).

## P. Verdict

NTransformer v1.3.0-candidate: **research-ready / development-ready, same as
v1.2.0-candidate — not release-ready, not production-ready.** The improvements
are correctness fixes with exact-restoration evidence, not a capability jump.
NTokenizer v0.4.0-candidate: stable artifact unchanged; candidate adds a
binding-layer injection defense and parallel batch; not promoted.

Nothing beyond its evidence is claimed. Where verification is impossible on
this machine (multi-GPU, fault injection, real corpus training), the limitation
is stated instead of simulated.
