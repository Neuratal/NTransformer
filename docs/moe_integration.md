# MoE Integration and Final Optimization Report

Date: 2026-11-16
Branch: `feature/moe-integration-2026-11`
Environment: Python 3.14.4, torch 2.15.0+cu132, RTX 4060 Laptop GPU
(7.62 GiB reported), 8 GB nominal VRAM.

This report covers the final engineering pass: general performance work, genuine
opt-in Mixture-of-Experts (MoE), MoE training/checkpoint integration, presets
and memory-estimator support, and a real dense-training validation at 75M
parameters. **No real corpus was downloaded, no pretraining was performed, and
no model-quality claim is made.** All quantitative statements below are either
measured on this machine or explicitly labeled as estimates.

---

## 1. What changed

### 1.1 General code / performance

- `training/chunked_linear_ce.py`: hoisted repeated fp32 weight conversions out
  of the chunk loop and removed a redundant `probs.clone()`. This reduces
  transient allocator pressure without changing the mathematics.
- Training forward now enters through the DDP-wrapped module's `forward`
  (`output_hidden_only=True, return_aux=True`) so the DDP reducer sees the whole
  graph; the previous hidden-state-only helper bypassed that preparation.
- Added local sharded safetensors checkpoints (`model.safetensors.index.json` +
  N shards) with `save_pretrained_sharded` / `load_pretrained_sharded` and
  `plan_shards`. This is a **local** format, not distributed checkpointing.
- Checkpoint restore now validates dense/MoE architecture, moves optimizer state
  tensors to each parameter's device, and restores scheduler length/step.
- Dense parameter counts are unchanged; all existing dense presets still count
  exactly as before.

### 1.2 MoE implementation

`neuratal/model/moe.py` implements a readable reference MoE:

- One independent bias-free SwiGLU per expert; optional shared SwiGLU expert.
- Router logits, softmax, and auxiliary losses are computed in **fp32**
  regardless of model dtype. Selected gates are normalized to sum to one.
- Dispatch is tensor-based: assignments are sorted by expert once, then one
  Python loop over experts processes contiguous slices. There is no per-token
  Python loop.
- Capacity dropping is opt-in (`capacity_factor=None` by default). When enabled,
  excess assignments per expert are dropped and counted; masked padding tokens
  are excluded from routing and produce exactly zero expert output.
- Auxiliary loss: `coef * E * sum_e(f_e * P_e)`; optional z-loss:
  `coef * mean(logsumexp(router_logits)^2)`. Both are built only when
  `compute_aux=True`, so inference and generation do not pay for training terms.
- `merge_moe_stats` aggregates per-layer counts/probabilities/entropy/dropped
  assignments for logging.

`neuratal/model/block.py` selects dense or MoE per layer, so the two can
coexist. `neuratal/model/model.py` exposes `forward_hidden`, hidden-only forward,
`return_aux`, optional `token_mask`, `active_parameters_per_token()`, and a
parameter breakdown that separates routers, experts, and shared experts.

### 1.3 Configuration

MoE is opt-in and never the default:

| Field | Meaning |
|---|---|
| `num_experts` | `0` = dense (v0.4 path); `>=2` = MoE; `1` rejected |
| `num_experts_per_token` | top-k, `1 <= k <= num_experts` |
| `expert_hidden_dim` | per-expert SwiGLU width; `None` → `intermediate_size` |
| `moe_layer_indices` / `moe_layer_frequency` | which layers are MoE; mutually exclusive; default all layers when MoE is enabled |
| `capacity_factor` / `min_capacity` | opt-in token dropping; `None` disables dropping |
| `router_aux_loss_coef` / `router_z_loss_coef` | auxiliary loss weights |
| `shared_expert` / `shared_expert_hidden_dim` | optional always-on expert |

Dense configs serialize the new fields with neutral values
(`num_experts=0`, `moe_layer_indices=[]`), so v0.4 dense checkpoints remain
loadable.

---

## 2. Measured results

### 2.1 Chunked cross-entropy

Same hardware, same shapes, 262,144-token vocabulary; five runs each after a
warm-up. Values are peak workspace and median time.

| Path | Before | After |
|---|---|---|
| fp32 | 22.57 ms · 385 MiB | ~19.8–20.0 ms · 321 MiB |
| bf16 | 23.99 ms · 401 MiB | ~20.3–20.4 ms · 337 MiB |

The improvement is modest but real: ~12–15% faster and ~16–17% less peak
workspace. It is a workspace/overhead optimization, not an algorithmic change.

### 2.2 75M dense end-to-end validation

Preset `75m`: **77,088,896 parameters** (unchanged). Synthetic token data,
batch 1 × 32 tokens, gradient accumulation 2, BF16, 2 optimizer steps.

| Measurement | Result |
|---|---|
| Peak training VRAM | **0.86 GiB** |
| Peak resume VRAM | **1.161 GiB** |
| Loss | finite (~12.47 at the logged step) |
| Checkpoint | step 2, architecture `dense`, precision `bfloat16` |
| Resume | counters, scheduler step, optimizer state restored; all finite |
| Parameter updates | at least one parameter changed; gradients finite |

This is a plumbing/optimizer/checkpoint validation on an 8 GB GPU, not a
pretraining result.

### 2.3 Small-MoE smoke test

Preset `small-moe`: 75M-class dense geometry, 8 experts, top-2, all layers MoE.
**402,301,056 total / 123,609,216 active per token.** BF16, 2 steps.

| Measurement | Result |
|---|---|
| Peak VRAM | **3.587 GiB** |
| Loss | finite |
| Aux loss / z-loss | ~0.22 / ~0.067 |
| Expert utilization | all 8 experts received tokens |
| Checkpoint | MoE metadata recorded; dense/MoE mismatch rejected on load |
| Resume | works with `model=` architecture validation |

This validates routing, dispatch, loss integration, and checkpoint/resume
plumbing at small scale. It does not validate convergence or quality.

### 2.4 MoE preset counts (meta-device measurement)

| Preset | Total parameters | Active / token |
|---|---:|---:|
| `small-moe` | 402,301,056 | 123,609,216 |
| `1b-moe` | 1,074,300,928 | 659,064,832 |
| `5.7b-moe` | 5,673,062,400 | 2,351,173,632 |
| `30b-moe` | 29,400,112,128 | 8,260,819,968 |
| `100b-moe` | 99,909,378,048 | 16,157,515,776 |
| `200b-moe` | 198,474,149,888 | 35,399,610,368 |
| `500b-moe` | 483,293,937,664 | 77,419,528,192 |

"Active" is an architecture-level per-token parameter count (top-k experts plus
all non-expert parameters). It is not a FLOP measurement. Every expert still
requires checkpoint and optimizer storage.

---

## 3. Compatibility evidence

- **NTokenizer unchanged.** Vocab remains 262,144; special IDs remain
  262137–262143; the v0.3 artifact SHA-256 is still
  `1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9`.
  No NTokenizer file was edited.
- **v0.4 stable dense path preserved.** `num_experts=0` constructs the same
  dense blocks and parameters as before. Dense preset counts are unchanged
  (75m = 77,088,896), and the existing dense/serialization/inference suites
  still pass.
- **Full suite:** 397 collected, **394 passed, 3 skipped** (the CUDA-on-CPU
  probes). This includes the pre-existing suite plus new MoE core,
  serialization, engine, preset/estimator, and 75M validation tests.

---

## 4. Scale readiness table (A–G)

Readiness labels are deliberately separate. "Config-ready" means a validated
config exists. "Architecture-ready" means the module tree/counts exist.
"Single-GPU-ready" means an end-to-end synthetic step was actually run here.
"Multi-GPU-ready" means an implemented path exists and was exercised.
"Production-distributed" means a credible large-scale training system; none of
these presets reaches that bar.

| | Scale | Total params | Active/token | Config | Architecture | Single 8 GB GPU | Multi-GPU | Production distributed |
|---|---|---:|---:|---|---|---|---|---|
| A | 75M dense | 77,088,896 | 77,088,896 | ✅ | ✅ | ✅ measured, 0.86 GiB | ⚠️ DDP path exists, not measured at scale | ❌ no |
| B | 1B-MoE | 1,074,300,928 | 659,064,832 | ✅ | ✅ | ❌ not run (optimizer/params alone exceed 8 GB) | ⚠️ DDP path only, untested | ❌ no |
| C | 5.7B-MoE (formerly "7b-moe") | 5,673,062,400 | 2,351,173,632 | ✅ | ✅ | ❌ no | ❌ no ZeRO/FSDP implementation | ❌ no |
| D | 30B-MoE | 29,400,112,128 | 8,260,819,968 | ✅ | ✅ | ❌ no | ❌ no | ❌ no |
| E | 100B-MoE | 99,909,378,048 | 16,157,515,776 | ✅ | ✅ | ❌ no | ❌ no | ❌ no |
| F | 200B-MoE | 198,474,149,888 | 35,399,610,368 | ✅ | ✅ | ❌ no | ❌ no | ❌ no |
| G | 500B-MoE | 483,293,937,664 | 77,419,528,192 | ✅ | ✅ | ❌ no | ❌ no | ❌ no |

`estimate()` provides closed-form per-rank memory accounting for
`single`/`ddp`/`zero1`/`zero2`/`zero3`/`fsdp`, but the ZeRO/FSDP numbers are
**accounting models, not implemented training paths**. Tensor, pipeline, and
expert parallelism are explicitly rejected by the estimator rather than
silently approximated.

---

## 5. The eight questions, answered

**Q1. What general code/performance improvements were made, and did they
measurably help?**
Chunked cross-entropy now hoists fp32 weight conversions and removes a redundant
clone; measured ~12–15% faster and ~16–17% less peak workspace (Section 2.1).
Training also enters through the DDP-wrapped forward, fixing reducer preparation.
Sharded local checkpoints and safer resume were added. These are engineering
improvements, not model-quality improvements.

**Q2. Is MoE genuinely integrated, or just a stub?**
It is genuinely integrated: real SwiGLU experts, an fp32 top-k router,
tensor-based dispatch, capacity handling, padding masks, aux/z losses, per-layer
dense/MoE placement, model-level active-parameter accounting, presets, estimator
support, and tests. It is not a fused production kernel, and expert parallelism
is not implemented.

**Q3. Are MoE training, checkpointing, and resume real and tested?**
Yes at small scale. `small-moe` ran two BF16 steps at 3.587 GiB peak with finite
loss, all 8 experts used, aux/z losses logged, MoE metadata saved, and resume
validated. Dense↔MoE architecture mismatches are rejected. DDP MoE router
statistics are per-rank; global load balance is not synchronized.

**Q4. Did 75M dense training validate on the 8 GB target GPU?**
Yes, for synthetic plumbing validation: 0.86 GiB peak training, 1.161 GiB peak
resume, finite loss/gradients, parameter updates, checkpoint at step 2, and
successful resume. This is not a pretraining or quality result.

**Q5. What remains before 7B–30B distributed training is credible?**
A real sharded optimizer (ZeRO-1/2/3 or FSDP), activation checkpointing,
distributed checkpoint save/load with resharding, global MoE load-balance
statistics, multi-node launch/observability, and actual multi-GPU measurements.
The current `zero*/fsdp` estimator output is accounting only; DDP alone would
replicate full parameters, gradients, and optimizer state per rank.

**Q6. What remains before 100B–500B?**
Everything in Q5 plus expert/tensor/pipeline parallelism, all-to-all
communication, capacity-aware fused dispatch kernels, hierarchical/async
checkpointing, and a datacenter fabric. At these sizes the presets are
**config/architecture targets only**; no training path exists in this
repository.

**Q7. Is NTokenizer compatibility unchanged?**
Yes. Vocab 262,144, special IDs 262137–262143, and the artifact SHA-256
`1f0d4ad7c7fb…6d9` are unchanged. No NTokenizer file was modified.

**Q8. Is v0.4 stable/dense behavior preserved?**
Yes. `num_experts=0` is the dense path; dense preset counts and the existing
dense architecture/serialization/inference behavior are unchanged. MoE is
opt-in only, and the full suite (including the pre-existing tests) passes.

---

## 6. Known limitations

- No real-corpus pretraining; no convergence, quality, or scaling-law claim.
- MoE dispatch is a readable reference loop over experts, not a fused kernel.
- DDP MoE router metrics/aux loss are computed per rank, not globally.
- No expert/tensor/pipeline parallelism; no distributed checkpoint resharding.
- Sharded safetensors are local-only; each rank owns a full state dict.
- Capacity dropping is implemented and tested, but no capacity-aware kernel.
- `estimate()` activation formulas are approximations; only
  `torch.cuda.max_memory_allocated` numbers in this report are measurements.
- Exact data replay on resume is guaranteed only with `shuffle=False`; the
  engine defaults to shuffling.

## 7. Reproduce

```bash
.venv/bin/python -m pytest tests/test_training_75m.py \
  tests/test_moe_core.py tests/test_moe_serialization.py \
  tests/test_moe_engine.py tests/test_moe_presets_estimate.py -q
.venv/bin/python -m pytest -q          # full suite
```

Preset/estimator inspection:

```python
from neuratal import get_moe_preset, list_moe_presets
from neuratal.estimate import estimate
p = get_moe_preset("small-moe")
print(p.parameter_count, p.active_parameter_count)
print(estimate(p.config, batch_size=1, seq_length=1024, strategy="single").report())
```
