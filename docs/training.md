# Training in NTransformer

Status: **matches `neuratal` 1.4.0 (2026-11-19).**

## Overview

The training engine (`neuratal/training/`) provides a deterministic, resumable
single-process training loop with DDP hooks.

- Data: pre-tokenized uint32/int64 streams packed greedily into fixed windows
  (`TokenDataset`); a synthetic generator (`SyntheticTokenDataset`) exists only
  for plumbing validation.
- Optimizer: AdamW with decoupled weight decay on ≥2D parameters; warmup →
  cosine decay (`WarmupCosineSchedule`), stepped per optimizer step.
- Gradient accumulation (micro → optimizer steps), gradient clipping on the
  accumulated gradient.
- AMP: bf16 autocast (fp32 master weights) or fp16 autocast + GradScaler.
- MoE auxiliary loss (load-balancing, router z-loss) is added once per
  micro-batch; inference never builds that graph.
- Validation on a held-out split, per-`log_every_steps` JSONL-style history,
  per-`checkpoint_every_steps` checkpoints.
- Resume restores model + optimizer + scheduler + RNG + step/token counters,
  GradScaler state, validates tokenizer fingerprint, validates optimizer state
  indices against the target model. Wrong-architecture resume is rejected
  explicitly (dense ↔ MoE). Mismatched fingerprints rejected.

## Quick start (synthetic data only)

```python
import torch
from neuratal import NeuratalModel, get_preset
from neuratal.training.data import SyntheticTokenDataset
from neuratal.training.engine import TrainConfig, train

model = NeuratalModel(get_preset("75m").config).to("cuda", torch.bfloat16)
ds = SyntheticTokenDataset(16, 32, vocab_size=262_144, seed=1)
cfg = TrainConfig(
    precision="bfloat16",
    max_steps=10,
    warmup_steps=2,
    out_dir="checkpoints/demo",
    device="cuda",
)
model, history = train(model, ds, None, cfg)
```

**SyntheticTokenDataset is plumbing validation only.** No real corpus has been
used for this project; no dataset is shipped or downloaded by this package.

## Checkpointing

```python
from neuratal.training.checkpoint import (
    load_training_checkpoint,
    resume_training_checkpoint,
)

ckpt = load_training_checkpoint("checkpoints/demo/ckpt-10/trainer_state.safetensors")
```

`config.json` + `model.safetensors`, atomic writes everywhere, strict
structural validation. Corrupt files are rejected with precise errors.

Training checkpoints add `trainer_state.safetensors` (optimizer/scheduler/
RNG/counters/scaler) and `trainer_meta.json` (step, tokens, fingerprint,
precision, architecture).

**Local sharded checkpoints** (`model.safetensors.index.json` + N shards)
exist for large models. This is a *local* format — **not** distributed
checkpointing; no resharding; no cross-rank coordination.

## Data

Pre-tokenized `uint32`/`int64` streams are packed greedily into fixed windows;
there is **no document-boundary masking** (standard packing behavior); synthetic
data is always same-vocab.

## GradScaler vs bf16

- bf16: autocast without a scaler; master weights stay fp32.
- fp16: autocast + GradScaler; unscale-before-clip; scaler state saved/restored.

## Honest scope

- **No real corpus has been used for pretraining this package.** The 40M pilot
  (val loss 10.84 → 6.24) used synthetic data; output quality is gibberish.
  Treat any such report as infrastructure validation, not quality evidence.
- **DDP is supported on CPU gloo only.** Dense and MoE were verified for
  numerical parity vs single-process training; multi-GPU/NCCL has not been run.
- **Long-context (16K/32K) training is experimental.** Requires
  `gradient_checkpointing=True`; no 32K benchmark claims are made.
- **MoE** is a reference implementation: Python loop over experts; no kernel
  fusion, no capacity-aware CUDA kernel, no all-to-all.
- **Chunked cross-entropy is ~35% slower** than standard CE but allocates
  ~3.3× less memory; the enabler for 262k-vocab training.

## What NOT to claim

- "Production-ready training framework."
- "Trained weights for model X."
- "Faster than PyTorch/TensorFlow/CUDA."
- "Megatron-level throughput."
