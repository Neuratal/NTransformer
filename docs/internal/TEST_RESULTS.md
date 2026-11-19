# NTransformer — Test Results (v1.3.0-candidate @ `d17391d`)

Machine: RTX 4060 Laptop · CUDA 13.2 · torch 2.15.0+cu132 · Python 3.14.4 · driver 595.99.02
Command: `.venv/bin/python -m pytest tests` (add `--basetemp=<dir>` if `/tmp` is a small tmpfs)

## 1. Headline

**422 passed, 3 skipped, 1 warning — 0 failed, 0 errors — in 44.28 s** (re-verified today; two earlier transient failures traced to `/tmp` disk-quota exhaustion, not code — see §5).

## 2. Collection history (independently verified via node-ID inventories)

| Version | Commit | Collected | Passed | Failed | Skipped | Removed/Renamed |
|---|---|---:|---:|---:|---:|---:|
| pre-v1.1 | `196a613` | 397 | 394 | 0 | 3 | — |
| v1.1.0rc1 | `5d13af9` | 416 | 413 | 0 | 3 | 0 |
| v1.2.0-candidate | `3976a5d` | 421 | 418 | 0 | 3 | 0 |
| **v1.3.0-candidate** | `d17391d` | **425** | **422** | **0** | **3** | **0** |

The `"428"` figure cited in earlier material does not exist anywhere in this repository (searched docs, registry, snapshots, README). True counts above; no test was ever removed or renamed.

## 3. Skips (full list, all with reasons)

| Test | Reason |
|---|---|
| `test_cache.py:179` | CUDA device-mismatch probe needs two CUDA contexts |
| `test_engine.py:111` | covered identically in artifact-dependent `test_ntokenizer_integration.py` |
| `test_ntokenizer_integration.py:209` | requires CUDA in this probe |

No hidden skips, no XFails, no `-k`-filtered gates.

## 4. Coverage map (module → tests)

| Area | Tests (files) |
|---|---|
| Config validation | `test_config.py` (205 lines, dozens of rules) |
| Model/block/attention | `test_model.py`, `test_block.py`, `test_attention.py` (incl. causality, GQA mapping, lower-right mask) |
| RoPE / embeddings / RMSNorm / SwiGLU / output | `test_rope.py`, `test_embeddings.py`, `test_normalization.py`, `test_mlp.py`, `test_output.py` |
| Generation / cache | `test_generation.py`, `test_cache.py`, `test_graph_decode.py` |
| Chunked CE | `test_chunked_loss.py`, `test_chunked_ce.py` (parity, memory) |
| Dense engine | `test_engine.py`, `test_training_75m.py` (75M bf16 chain, resume to step 2) |
| MoE | `test_moe_core.py`, `test_moe_engine.py`, `test_moe_pathological.py` (top-1/top-2, capacity, starved/empty experts, aux/z losses, bf16) |
| Checkpointing | `test_serialization.py` (roundtrip, tying, corruption, sharding, SHA-256), `test_engine_regressions.py` (fingerprint, GradScaler, RNG-exact restore, resume-then-continue) |
| Estimator / presets | `test_estimate.py`, `test_presets.py`, `test_moe_presets_estimate.py` |
| Integration | `test_ntokenizer_integration.py`, `test_tokenizer_integration.py` |
| Distributed | `test_distributed.py` (single-process gloo smoke; no multi-GPU) |

## 5. Flakiness encountered (both external, both understood)

Two runs failed with `SafetensorError: I/O error: Disk quota exceeded (os error 122)` — `/tmp` is a 7.4 GB tmpfs and pytest's default basetemp filled it (75M checkpoints ≈ 1.1 GB per run; pytest keeps 3 numbered dirs). **Not a code defect.** With `--basetemp=/home/justerdem14/.cache/pytest_tmp` (or any large FS), the suite passes repeatedly. README documents this. No test was modified to accommodate disk limits.

## 6. Warnings

1 third-party warning: Triton FP-fusion default notice (from CUDA kernel compilation inside a CUDA test; harmless, external).

## 7. Commands to reproduce

```bash
cd ../NTransformer
PYTHONPATH=. .venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest tests/test_serialization.py tests/test_engine_regressions.py tests/test_training_75m.py -q
.venv/bin/python -m pytest tests --collect-only   # counts
```

## 8. Honest gaps (not passing, not tested)

- Multi-GPU/multi-node execution (tests skip; no coverage)
- Tokenizer-heavy integration in a moved checkout (skips by design)
- GPU fault injection / power-loss (no destructive test executed)
- Enumeration of 100% of branches — branch-coverage not measured (no coverage tool configured; advisory, not claimed)
