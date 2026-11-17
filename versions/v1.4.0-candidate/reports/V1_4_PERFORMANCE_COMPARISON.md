# V1.4 — Performance Comparison (v1.3.0-candidate → v1.4.0-candidate)

## 1. Methodology

- **Rig:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), CUDA 13.2, torch 2.15.0+cu132,
  Python 3.14.4, Linux 7.0.0-43, 1 GPU (no physical multi-GPU).
- **Comparison set:** identical closed-form harness for both versions.
  `v1.3` measured from an unmodified export of the audited commit
  (`/home/justerdem14/.cache/ntransformer_audit/v1.3_baseline/`, git-archive of
  `7d6bf94`), `v1.4` from the working tree at `5d912b3`. Same seeds, shapes,
  dtypes, iteration counts, warm-up, and GPU state (back-to-back runs).
- **Metrics:** wall-clock per iteration (CUDA synchronized), peak allocated
  bytes (`torch.cuda.max_memory_allocated`), spread = (max − min) / median
  across iterations (noise indicator). Medians reported; deltas within the
  interleaved spread are noise, not signal. Known caveat: this GPU's clock
  drift makes sub-5 % deltas unresolvable (documented benchmark noise ±10–20 %
  in the project's methodology notes).
- **Artifacts:** `versions/v1.4.0-candidate/benchmark_results/results_v1.3.json`,
  `results_v1.4.json`, raw logs `v1.3.log`, `v1.4.log`.

## 2. Results

| Benchmark | v1.3 | v1.4 | Δ (nominal) | Verdict |
|---|---|---|---|---|
| Dense forward B2 T512 (dev preset) | 6.15 ms | 6.45 ms | +4.6 % | within noise (spread 4–6 %) |
| Dense train step B2 T128 bf16 | 21.63 ms | 21.64 ms | +0.1 % | parity |
| MoE forward B2 T128 (4 experts) | 1.73 ms | 1.66 ms | −4.0 % | within noise |
| MoE train step B2 T128 bf16 | 6.67 ms | 6.56 ms | −1.7 % | parity |
| Chunked-CE 1024 tok: peak memory | 14.76 MB | 14.76 MB | 0.0 % | exact |
| Chunked-CE 1024 tok: step time | 3.36 ms | 3.44 ms | +2.3 % | within noise |
| Standard CE 1024 tok: peak memory (reference) | 68.84 MB | 68.84 MB | 0.0 % | exact |
| KV-cache decode (8 tokens) | 4.89 ms | 4.68 ms | −4.2 % | within noise |
| Checkpoint save | 1.01 ms | 1.05 ms | +3.6 % | within noise |
| Checkpoint load | 2.71 ms | 2.74 ms | +0.9 % | parity |

**No benchmark moved by more than the measurement noise band; no regression is
detectable at this resolution.** The correctness work does not carry a
throughput or memory tax on the single-process path (the DDP joint-loss module
only exists under multi-process execution; the single-process path executes
the identical op sequence as v1.3).

## 3. What *did* change, and by how much

- **`history.json` entries** gained one integer field (`iteration`) — no measurable cost.
- **Checkpoint load path** gained: shard-name regex + resolved-path containment
  check per shard (µs), and — on the fail-closed path — refusal to load
  rather than silently skip (not a hot path).
- **Resume path** gained: two warning branches and a constructor-clamp on the
  resume-only path. Not timed; negligible.
- **Manifest parser** now does a header-first two-pass read. For tiny test
  artifacts irrelevant; for 1 GiB artifacts same total bytes (header read
  skipped on second pass via seek) — actually **less memory**, since the
  whole-file `.read_bytes()` copy is gone.
- **Estimator**: one branch — unmeasurable.
- **Scheduler**: same op count; trajectory correct.

## 4. Chunked CE memory cross-check

The project's headline claim (peak memory of chunked CE ≪ standard CE) holds
unchanged on v1.4 in this 1024-token fixture (14.76 MB vs 68.84 MB allocated,
rel. ~4.7×). The full 262k-vocab peak comparison requires a full-scale run
beyond this 8 GB card; the ordering is preserved and the implementation is
unchanged mathematically (verified separately: relative gradient L2 ≈ 2.5e-7
vs standard CE, tolerance explained in V1_4_REGRESSION_TEST_REPORT.md).

## 5. DDP performance (CPU, 2 proc)

4 optimizer steps × 2 micro-batches of a tiny MoE on CPU/gloo complete in
≈ 1.84 s (≈ 230 ms/step including all-reduce). v1.3 crashed instead at step 1
(see V1_4_DDP_VALIDATION_REPORT.md). This is a correctness datum, not a
performance claim; nothing here predicts scaling beyond two CPU processes.

## 6. Verdict

No statistically meaningful performance or memory regression. The DDP MoE fix
is the expensive part of this release (a redesigned forward that keeps the
reducer honest); it costs single-GPU training nothing by construction.
