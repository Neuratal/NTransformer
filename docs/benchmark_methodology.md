# Benchmark Methodology

Every number in `docs/benchmark_results.md` comes from an actual execution of
the scripts in `benchmarks/`. Nothing is estimated, extrapolated, or invented.
Theoretical values (e.g., logits-tensor bytes from the shape formula) are
explicitly labeled *theoretical* and never mixed with measured timings.

## Hardware & environment

Recorded per run in `benchmarks/results/environment_<ts>.json` (created by
`benchmark_environment.py`). This run: NVIDIA GeForce RTX 4060 Laptop GPU
(sm_89, 7.8 GiB), Intel i7-12700H (20 cores), 15.3 GiB RAM, Ubuntu 26.04 LTS,
python 3.14.4, torch 2.15.0+cu132, safetensors 0.8.0, pytest 9.2.0,
NVIDIA driver 595.99.02, CUDA 13.2.

## Reproducibility contract

- Fixed seed 42 for all randomized tensors (`torch.manual_seed(42)` /
  explicit `torch.Generator(42)`); no other randomness on the hot path.
- Results are written to `benchmarks/results/<name>_<run_id>.json` and mirrored
  to `<name>_latest.json`; timestamped files are never overwritten, so
  repeated runs accumulate instead of clobber.
- Each result records the git-independent descriptor (machine fingerprint,
  configs, artifact SHA-256), never just "fast GPU".
- Benchmarks time with `time.perf_counter()` after `torch.cuda.synchronize()`;
  all model benchmarks use `torch.inference_mode()`; warm-up iterations are
  excluded from timing.
- Report latency as **median** plus p95 (never a cherry-picked best run);
  every cell records its sibling samples in the JSON.
- Machine load is not controlled beyond that; treat ±10–20% run-to-run spread
  as normal on this shared laptop GPU. Numbers are reproducible in
  distribution, not bit-exactly in wall time.
- Memory figures come from `torch.cuda.max_memory_allocated()` /
  `max_memory_reserved()` (allocated/reserved), while *theoretical* figures
  are computed from shapes and labeled as such.

## Procedure per benchmark

| Script | Measures | Method |
|---|---|---|
| `benchmark_environment.py` | OS/CPU/RAM/GPU/driver/torch/NTokenizer/NTransformer versions, dtype support matrix, SDPA backend availability | read-only probes; SDPA backends probed under `sdpa_kernel` for fp32/bf16/fp16 |
| `benchmark_tokenizer.py` | artifact load time (separate), encode/decode latency + throughput per corpus, batch(32) throughput | warmup per corpus; reps scaled to payload size; medians/p95 reported |
| `benchmark_model.py` | prefill latency + tok/s at T ∈ {32..1024}, B ∈ {1}; CUDA peak alloc/reserved | 10 warmup / 30 repeats; inference_mode |
| `benchmark_kv_cache.py` | uncached vs cached prefill, per-token decode, cache buffer bytes (exact), cached-vs-uncached max|Δlogit| | fresh KVCache per repeat; numerical assert |Δ| < 1e-5 (fp32) |
| `benchmark_attention.py` | per-backend availability + latency + |Δ| vs fp64 reference | backends forced via `sdpa_kernel`; fp16/bf16/fp32 |
| `benchmark_end_to_end.py` | encode/prefill/generate/decode per-stage times on 3 prompts × gen lens {16,32,64} | 5 repeats; fresh pipeline per prompt |
| `benchmark_memory.py` | dev preset + 47.85M reference: peak allocated/reserved forward at T ∈ {64,512}; theoretical tables | budget-guarded (skip if estimated > budget); explicit skip reasons |

## What "measured" means here — and caveats

- CUDA timings include kernel-launch overhead. Pipelined GPU work can look
  faster than a synchronize-per-iteration pattern; the scripts synchronize
  between iterations, so reported per-call times are honest wall latencies.
- `torch.cuda.synchronize()` after a timed call waits for all prior work, so
  per-iteration samples include all enqueued effects of that iteration.
- The KV-cache decode benchmark creates a fresh cache per repeat; cache
  creation time is included in prefill, not in decode.
- The dev preset is an untrained 9.8M-parameter model pinned to the real
  262,144-token vocabulary; decode speeds reflect launch-overhead-bound tiny
  models, not large-model decode characteristics.
- Peak memory numbers are for *this* workload (inference, no gradients).
  Training would add gradient and optimizer memory; none is measured or claimed.
- bf16/fp16 measurements use the model's parameter dtype; logits stay in
  parameter dtype (or fp32 when `fp32_logits=True`); softmax/loss consumers
  should upcast.
- Numerical equivalence checks (cached vs uncached) run inside the benchmarks;
  a failed check aborts the run, it does not print a warning and continue.

## Skipping policy

Any cell whose estimated footprint exceeds the configured budget is skipped
and recorded with its reason in the JSON (`skipped` array). No cell is allowed
to OOM-test the machine. CPU-only environments simply lack CUDA cells; the
tokenizer and model CPU benchmarks are unaffected.
