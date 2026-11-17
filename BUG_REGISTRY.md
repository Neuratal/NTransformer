# NTransformer — Consolidated Bug Registry (v1.3.0-candidate @ `d17391d`)

All confirmed defects, their severity, status, and regression evidence. Complements `docs/bug_registry.md` (v1.2) and `docs/ultimate_audit/bug_registry.md` (v1.1/v1.2).

## Registry

| ID | Severity | Component | Finding → Fix | Status | Regression test |
|---|---|---|---|---|---|
| B-2021-1 | critical (crash) | training engine | `core.output_projection.out_proj.weight` dereferenced unconditionally; full-width tied heads (`embedding_size=None`) crashed on first step → `_head_out_proj_weight()` returns `None`, chunked CE handles it | FIXED (v1.1 `49cbbc6`) | `test_full_width_train_step_and_checkpoint` |
| B-2021-2 | high | training engine | fp16 autocast ran **unscaled** (under/overflow risk) → GradScaler with unscale-before-clip, state checkpointed & restored | FIXED (v1.1 `49cbbc6`) | `test_fp16_step_and_scaler_resume` |
| B-2021-3 | high (distributed) | training engine | DDP MoE: zero-token experts never enter autograd → reducer wedges → `find_unused_parameters=True` for MoE (dense keeps fast default) | FIXED (v1.1 `49cbbc6`) | suite green; **multi-GPU untested (labeled)** |
| B-2021-4 | high | training/checkpoint.py | resume did not verify recorded tokenizer fingerprint → mismatch silently resumed against wrong tokenizer → validation + arch guard + optimizer-index guard | FIXED (v1.1 `49cbbc6`) | 4 fingerprint tests |
| B-2021-5 | low | model/moe.py | `torch.tensor(0.0, device=…)` H2D scalar in drop path → `new_zeros` | FIXED (v1.1 `b608fdf`) | suite green |
| B-2022-1 | medium | training/optim.py | fresh `WarmupCosineSchedule` left groups at full base LR for step 0 (warmup never covered it) → apply `lr_at(0)` at construction | FIXED (v1.2 `27c7f78`) | `test_scheduler_fresh_applies_warmup_lr` |
| B-2022-1b | **high** | training/optim.py | `load_state_dict` never pushed restored LR into param groups → first post-resume optimizer step ran at construction LR (silent mid-cosine jump) → reapply `lr_at(step-1)` | FIXED (v1.2 `27c7f78`) | `test_scheduler_resume_applies_resumed_lr` |
| B-2022-3 | medium | training/optim.py | `base_lrs` accepted at any length → silent group↔LR misassignment → length check → `ValueError` | FIXED (v1.2 `27c7f78`) | `test_scheduler_base_lrs_length_mismatch_rejected` |
| B-2022-4 | medium (integrity) | serialization/sharded.py | `total_size` recorded but never re-verified at load → same-size stale replacement passed → byte-count check added | FIXED (v1.2 `eeed9b4`) | `test_sharded_total_size_integrity_check` |
| B-2022-5 | low | serialization/sharded.py | duplicated `_atomic_save_safetensors` (drift risk) → deduplicated, anti-aliasing shared | FIXED (v1.2 `eeed9b4`) | sharded suite |
| B-2022-6 | low | model/moe.py | `sorted_experts` computed even without capacity → lazy | FIXED (v1.1 `d9bb7c1`) | MoE suite |
| **B-2023-1** | **high (silent)** | training/checkpoint.py | scheduler/optimizer scalar state stored as **float32** → entire resumed cosine trajectory drifted ~1e-7 from uninterrupted → int64/float64 storage, **bitwise-exact round-trip** | **FIXED (v1.3 `7939bbe`)** | `docs/scheduler_resume_verification.md` (7 resume points) |
| **B-2023-2** | medium | training/optim.py | `load_state_dict` accepted malformed state (missing keys → `KeyError`, NaN ratios, negative steps) → required-key + type/range/finiteness validation | **FIXED (v1.3 `7939bbe`)** | scheduler suite |
| **B-2023-3** | medium (integrity) | serialization/sharded.py | same-size stale shard replacement passed byte-count check → **per-shard SHA-256** recorded at save, verified at load (default on) | **FIXED (v1.3 `7939bbe`)** | `test_sharded_tampered_shard_rejected` |
| RMSNorm (aux) | medium (perf/numerics) | model/normalization.py | autocast-promoted fused weighted path was numerically worse (0.0156 err) → fused normalize-without-weight + explicit fp32 weight mul | FIXED (v1.2 `65830db`) | norm suite + parity checks |

## Closed / verified-not-a-bug

| Item | Resolution |
|---|---|
| CUDA RNG restore `torch.cuda.set_rng_state(state.cpu(), i)` | works byte-exactly (my initial probe was wrong; code was right) — regression test added (`test_cuda_rng_restore_exact`) |
| All-ignored-target loss backward (`ignore_index=-100`) | backpropagates zeros, no NaN/inf — verified |
| `torch.compile(MoE)` / `torch._grouped_mm` / cuDNN forcing | all measured, all within noise or worse — **not adopted** (see rejected_experiments.md) |
| `test_engine.py:111` skipped | covered by artifact-dependent integration suite — intentional |
| Two transient test failures | **disk quota** (`/tmp` tmpfs exhaustion), not code — `--basetemp` documented |

## Open / advisory (not defects, documented)

| ID | Severity | Note |
|---|---|---|
| O-1 | informational | `use_reentrant=True` in legacy `chunked_loss.py` — deprecated but correct in this build |
| O-2 | informational | shuffle-order data replay after resume is re-permuted (documented) |
| O-3 | medium (untested) | DDP MoE fix untested on multi-GPU — no hardware available |
| O-4 | informational | no power-loss fault injection; atomic rename ≠ power-loss proof |
| O-5 | informational | activation memory estimator ±50 % beyond dev scale |
| O-6 | informational | benchmark noise ±10–20 % on this GPU; sub-10 % deltas discarded |
| O-7 | informational | no hash-pinned pip requirements; documented |

## Severity tally

1 critical · 3 high · 4 medium · 2 low · rest informational. **None open critical/high — all fixed with tests.**
