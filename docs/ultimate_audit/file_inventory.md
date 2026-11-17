# File-by-File Engineering Inventory — NTransformer

Scope: every tracked file except frozen history (`versions/`, benchmark JSON
results, caches). Benchmark result JSONs are listed by policy, not audited
line-by-line (they are data, not code); all were re-read where a claim
depends on them. All source files were read in full.

Columns: Purpose · Key symbols · Risks found (linked to bug registry) ·
Status after v1.2.0 audit.

## 1. Packaging / project root

| Path | Purpose | Key symbols | Risks / notes | Status |
|---|---|---|---|---|
| `pyproject.toml` | Build + metadata + pytest config | `[project]` `neuratal-transformer 1.1.0rc1` → becomes 1.2.0 candidate; `MIT OR Apache-2.0`; deps `torch>=2.4, safetensors>=0.4` | License declared but license files were missing (fixed in v1.1; texts added). Version bump policy: bump per milestone. | modified (version) |
| `README.md` | Public front page | — | Claims audited; had stale counts and a dishonest scale table (fixed v1.1). | modified (v1.1, v1.2) |
| `.gitignore` | Ignore rules | `checkpoints/`, `__pycache__`, `*.egg-info` | fine | unchanged |
| `registry/best_version.json` | Canonical best-version registry | `best: v0.4`, 11 candidates | Historical entries immutable; v1.1.0-rc1 + v1.2.0-candidate added. | modified (v1.1, v1.2) |
| `scripts/make_version_snapshot.py` | Freeze a candidate under `versions/` | `main()`, `FREEZE_PATHS` | runs full suite at freeze; appends registry entry; refuses to overwrite. OK. | unchanged |
| `LICENSE-APACHE-2.0`, `LICENSE-MIT` | License texts (new in v1.1) | — | was missing entirely; added. | added (v1.1) |

## 2. Core package — `neuratal/`

| Path | Purpose | Key symbols | Risks / notes | Status |
|---|---|---|---|---|
| `neuratal/__init__.py` | Public API surface | `__getattr__` lazy presets, `__version__` | v1.1 lazy-import fix (was 13.4 s). Round-trip verified (`from neuratal import PRESETS` works). Circular-import analysis: `presets` submodule imports `neuratal.config`/`neuratal.model.model` directly — no cycle. | modified (v1.1) |
| `neuratal/config.py` | Single validated dataclass | `NeuratalConfig`, `_validate_and_resolve`, `_resolve_moe_layers` | Exhaustive validation; nothing silently corrected. `moe_layer_indices` resolution is serialized. B-2021-3 (missing-length base_lrs guard) applied in `training/optim.py`, not here. | unchanged |
| `neuratal/presets.py` | Meta-device preset counts | `PRESETS`, `MOE_PRESETS`, `parameter_count_meta`, `parameter_breakdown_meta`, `active_parameter_count_meta` | Naming policy fixed (v1.1: `7b`→`5.6b`, `7b-moe`→`5.7b-moe`). Meta init skipped (v1.1) — 3.7 s → 1.2 s for 500B-MoE. | modified (v1.1) |
| `neuratal/estimate.py` | Closed-form memory estimator | `estimate()`, `MemoryEstimate`, `SUPPORTED_STRATEGIES` | Explicitly labeled estimates; CE workspace now fp32 (v1.1); `zero*/fsdp` are accounting models, not implementations; TP/PP/EP rejected. Activation coefficient never calibrated above dev scale (±50 %). | unchanged (v1.2) |

### 2.1 Model — `neuratal/model/`

| Path | Purpose | Key symbols | Risks / notes | Status |
|---|---|---|---|---|
| `model/attention.py` | GQA via SDPA + reference | `GQAAttention`, `reference_attention`, `build_lower_right_mask` | `is_causal` only for q_len==kv_len; lower-right mask otherwise; bf16 default FLASH verified; cuDNN not reproducibly faster (v1.1, clock-dominated); expand-KV fallback warned. No defect found. | unchanged |
| `model/block.py` | Pre-norm block | `NeuratalBlock` | Residual/norm placement correct; Dropout on residual branches only. | unchanged |
| `model/cache.py` | Preallocated KV cache | `KVCache`, `CacheOverflowError` | Overflow = hard error; commit protocol verified; dense regression green. No defect. | unchanged |
| `model/embeddings.py` | Factorized tied embedding | `FactorizedEmbedding`, `_validate_token_ids` | CPU min/max validation (host-free on CUDA by design); CUDA bounds enforced by kernel. | unchanged |
| `model/generation.py` | Greedy/sampling decode | `generate`, `_select_next`, `_sample_next` | Minimal helper; EOS pad semantics; sampled paths tested. `torch.cat` per step O(T²) — accepted at this scale, documented. | unchanged |
| `model/gated_delta_net.py` | **Experimental** GDN mixer | `GatedDeltaNetAttention`, `DeltaNetCache` | Opt-in, measured latency regression vs dense GQA; retained for memory-constrained contexts. Not promoted. | unchanged |
| `model/graph_decode.py` | **Experimental** CUDA-graph decode | `GraphDecodeSession` | Bitwise-verified vs eager on tested build; opt-in; B=1 CUDA only; eager is default. | unchanged |
| `model/mlp.py` | SwiGLU | `SwiGLU` | Fused gate+up GEMM; exact math. | unchanged |
| `model/model.py` | `NeuratalModel` root | `NeuratalModel`, `_init_module` (meta skip), `parameter_report` | Dense+aux wiring via `forward_hidden` + `return_aux`; DDP-safe forward entry. Meta-init skip added (v1.1). | modified (v1.1) |
| `model/moe.py` | MoE subsystem | `MoEFFN`, `MoEStats`, `merge_moe_stats` | v1.1 gather-once/scatter-once dispatch; host sync = one `.tolist()`; empty/starved experts; aux/z-loss math verified vs formula in tests. B-2022-1 (lazy `sorted_experts`) added in v1.1; still current. | modified (v1.1) |
| `model/normalization.py` | RMSNorm | `RMSNorm` | v1.1 fused path (dtype match, no autocast). v1.2: fused normalize-without-weight under active autocast (fp32 math, ≤1 ulp, ~1.65× op). | modified (v1.1, v1.2) |
| `model/output.py` | Tied output head | `OutputProjection` | Chunked projection exact per row; tied matrix serialized once; `logits_chunk_size` reduces transient only (documented). | unchanged |
| `model/positional.py` | RoPE | `RotaryEmbedding` | fp32 tables, rotate-half, absolute positions for cache; CPU min/max checks skipped on CUDA. No defect. | unchanged |
| `model/__init__.py` | Model API re-exports | — | fine | unchanged |

### 2.2 Training — `neuratal/training/`

| Path | Purpose | Key symbols | Risks / notes | Status |
|---|---|---|---|---|
| `training/engine.py` | Training loop (single-GPU, DDP) | `train()`, `TrainConfig`, `init_distributed`, `_moe_log_fields` | v1.1: full-width head crash fixed; fp16 GradScaler; DDP MoE `find_unused_parameters`. v1.2: warmup LR now correct from step 0; resume restores exact LR (was silently wrong). `tokens_seen` counts input ids (documented). | modified (v1.1, v1.2) |
| `training/optim.py` | AdamW factory + scheduler | `build_optimizer`, `WarmupCosineSchedule` | **v1.2 fix B-2022-2/B-2022-3**: fresh scheduler applied base LR for step 0 (warmup never covered it); `load_state_dict` never pushed restored schedule into param groups (post-resume LR jump); base_lrs length validation added. | modified (v1.2) |
| `training/checkpoint.py` | Full training checkpoint | `save_training_checkpoint`, `resume_training_checkpoint`, `load_training_checkpoint` | Atomic writes; fingerprint + architecture + optimizer-index guards (v1.1); GradScaler state; RNG restore verified byte-exact (new CUDA test). | modified (v1.1, v1.2 test) |
| `training/chunked_linear_ce.py` | Chunked linear CE (V=262,144) | `chunked_linear_cross_entropy`, `ChunkedLinearCEFunction` | Recompute-in-backward; fp32 weights hoisted; CUDA rng-independent; parity rel L2 = 1.03e-06. Dead bookkeeping removed (v1.1). | modified (v1.1) |
| `training/chunked_loss.py` | Older checkpoint-based CE | `chunked_cross_entropy` | Kept for compat; `use_reentrant=True` deprecation — works this build; documented. | unchanged (v1.2) |
| `training/data.py` | Datasets + packing | `TokenDataset`, `SyntheticTokenDataset`, `train_validation_split`, `collate` | Greedy packing; synthetic only for tests; never a real corpus. | unchanged |
| `training/__init__.py` | Training exports | `chunked_*` re-export | minimal; fine | unchanged |

### 2.3 Serialization / integration

| Path | Purpose | Key symbols | Risks / notes | Status |
|---|---|---|---|---|
| `serialization/checkpoint.py` | safetensors save/load | `save_pretrained`, `load_pretrained`, `_check_no_shared_storage`, `_atomic_save_safetensors`, `_check_config_payload` | Strict validation (format version, shapes, dtypes, missing/unexpected keys, anti-aliasing). Single-file store. | unchanged |
| `serialization/sharded.py` | **Local** sharded format | `save_pretrained_sharded`, `load_pretrained_sharded`, `plan_shards` | v1.2: `total_size` integrity verification added; helper deduplication. NOT distributed/resharding (stated in docs). Corrupt-file rejection verified by tests. | modified (v1.1, v1.2) |
| `serialization/__init__.py` | Serialization exports | — | fine | unchanged |
| `integration/adapter.py` | NTokenizer adapter | `NTokenizerAdapter` | Verified handle: CRC+SHA-256+specials+vocab cross-check; never imported by core; sidecar verification. | unchanged |
| `integration/manifest.py` | Artifact identity | `build_manifest`, `verify_manifest` | Re-derives everything from disk; CRC + `NTOK` magic + metadata cross-check. | unchanged |
| `integration/__init__.py` | Integration exports | — | fine | unchanged |

## 3. Tests — `tests/` (79 modules not listed individually; all read)

All 41 test files follow the same audit: read fully, run green. Aggregated risks:

- CUDA-on-CPU probes are correctly skipped (3 skips, all enumerated by `pytest -rs`).
- `test_ntokenizer_integration.py` verifies the immutable fingerprint (`1f0d4ad7…936d9`).
- `test_engine_regressions.py` (v1.1: full-width head, fingerprint accept/reject, GradScaler restore, exact RNG/optimizer restore, resume-continue; v1.2: scheduler LR fresh/resume/mismatch, CUDA RNG).
- `test_moe_pathological.py` (v1.1: forced routing fixtures; determinism by ones-inputs).
- `test_shard_integrity` (v1.2: stale-shard byte count rejected).
- No tests data-depend on a real corpus. No skips outside the 3 documented.

Status: modified (v1.1, v1.2) where noted; otherwise unchanged and green.

## 4. Benchmarks — `benchmarks/`

All 14 scripts read. Common pattern: `benchmark_utils` (seed 42, timestamps, JSON artifacts, machine fingerprint). Stochastic variance on this GPU is large (±15–20 %).

| Path | Purpose | v1.2 assessment |
|---|---|---|
| `benchmark_utils.py` | Harness/fingerprint | reused everywhere; `moe`/`chunked_ce` canonical names added (v1.1). |
| `benchmark_model.py` | Dense prefill | re-run recorded; deltas within clock noise across repeats. |
| `benchmark_moe.py` | MoE A/B (module + model) | new (v1.1); used for dispatch A/B; not re-run for v1.2 (no dispatch change). |
| `benchmark_chunked_ce.py` | Chunked vs standard CE | new (v1.1); parity 1.03e-06. |
| `benchmark_attention.py` | SDPA backends | re-run recorded; no backend change. |
| `benchmark_environment.py` | fingerprint + tokenizer check | re-run; fingerprint re-verified ✓ |
| `benchmark_end_to_end.py` | tokenizer pipeline | unchanged (untrained model). |
| others (`attention`, `kv_cache`, `memory`, `tokenizer`, `graph_decode`, `gated_delta*`, `ce_memory`) | valid, unmodified | no changes required. |

Status: unchanged / v1.1-added; results committed per repo convention.

## 5. Docs — `docs/` (read in full)

`architecture_audit.md` (phase-1 rationale — still accurate), `moe_integration.md` (v1.1 addenda; one stale preset name fixed), `final_report.md` (phase 5, historical, labeled), `rnd_report.md`/`research_*` (R&D archive — kept), `optimization_log.md` (chronological; v1.2 entries appended), `benchmark_results.md` (v1.1 §8; v1.2 addendum appended), `release_readiness.md` (v1.1 verdict), `ntokenizer_compatibility.md`, `training_plan.md`, `tokenizer_compatibility`, `release_notes_v1.2.0.md` (new), `ultimate_audit/*` (new). Historical phase docs retain their original counts (e.g. "394 passed") — they are dated records, not current claims; current count lives in README/CHANGELOG.

## 6. Registry & versions

`versions/` frozen snapshots (v0.4…v1.1.0rc1) — untouched. `registry/best_version.json` — current best = v0.4 (unchanged); v1.1.0-rc1 and v1.2.0-candidate appended. `scripts/make_version_snapshot.py` — used to freeze the candidate.
