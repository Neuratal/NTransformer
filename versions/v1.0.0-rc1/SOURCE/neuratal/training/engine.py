"""Training engine — single-GPU and multi-GPU (DDP) training.

Design contract:
- Pure PyTorch, no Accelerate dependency.
- Single-GPU default; DDP via ``torchrun`` / ``init_process_group``.
- Gradient accumulation counted in *optimizer steps*; LR schedule steps on
  optimizer steps only (never micro-steps).
- Gradient clipping applied to the *accumulated* gradient before stepping.
- BF16 mixed precision via ``torch.autocast``; fp32 master weights handled by
  ``torch.optim.AdamW`` (it keeps fp32 state for bf16 params automatically).
- Loss normalization: average per-token loss over supervised tokens.
- Resume restores model + optimizer + scheduler + RNG + counters exactly.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from neuratal.config import NeuratalConfig
from neuratal.model.model import NeuratalModel
from neuratal.training.chunked_linear_ce import chunked_linear_cross_entropy
from neuratal.training.optim import WarmupCosineSchedule, build_optimizer
from neuratal.training.checkpoint import save_training_checkpoint, resume_training_checkpoint


@dataclass
class TrainConfig:
    # data
    batch_size: int = 2
    sequence_length: int = 1024
    num_workers: int = 0
    val_fraction: float = 0.05
    # optimization
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    # precision
    precision: str = "bfloat16"  # "float32" | "bfloat16" | "float16"
    # validation / logging
    eval_every_steps: int = 200
    log_every_steps: int = 20
    checkpoint_every_steps: int = 500
    # misc
    seed: int = 1234
    max_steps: int = 10_000
    out_dir: str = "checkpoints/run"
    tokenizer_fingerprint: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def init_distributed() -> tuple[int, int]:
    if not dist.is_available():
        return 0, 1
    if not dist.is_initialized():
        if "RANK" in __import__("os").environ:
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        else:
            return 0, 1
    return dist.get_rank(), dist.get_world_size()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    model: NeuratalModel,
    train_data,
    val_data,
    cfg: TrainConfig,
    *,
    max_steps: Optional[int] = None,
):
    """Run training. Returns (model, history) — no side effects on global state."""
    rank, world_size = init_distributed()
    is_main = rank == 0
    set_seed(cfg.seed + rank)
    model = model.to(cfg.device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank] if cfg.device == "cuda" else None)

    optimizer = build_optimizer(model, name="adamw", learning_rate=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)
    total_steps = max_steps or cfg.max_steps
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_steps,
                                     total_steps=total_steps, min_lr_ratio=cfg.min_lr_ratio)
    use_amp = cfg.precision in ("bfloat16", "float16") and cfg.device == "cuda"
    amp_dtype = torch.bfloat16 if cfg.precision == "bfloat16" else torch.float16

    out_dir = Path(cfg.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    t0 = time.perf_counter()
    global_step = 0
    tokens_seen = 0

    from torch.utils.data import DataLoader
    from neuratal.training.data import collate

    core = model.module if hasattr(model, "module") else model
    dl = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, collate_fn=collate, drop_last=True,
                    pin_memory=(cfg.device == "cuda"))
    data_iter = iter(dl)

    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        for _ in range(cfg.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            with torch.autocast(device_type=cfg.device, dtype=amp_dtype, enabled=use_amp):
                hidden = core.forward_hidden(ids)
                total_loss, n_tok = chunked_linear_cross_entropy(
                    hidden, labels,
                    core.output_projection.out_proj.weight,
                    core.embed_tokens.embedding_weight,
                )
                loss = total_loss / n_tok.clamp_min(1)
            (loss / cfg.gradient_accumulation_steps).backward()
            tokens_seen += ids.numel()
            accum += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if is_main and step % cfg.log_every_steps == 0:
            elapsed = time.perf_counter() - t0
            lr = scheduler.optimizer.param_groups[0]["lr"]
            entry = {"step": step, "tokens_seen": tokens_seen,
                     "lr": lr, "grad_norm": float(grad_norm),
                     "elapsed_s": round(elapsed, 2)}
            history.append(entry)
            print(f"[step {step}] tokens={tokens_seen} lr={lr:.2e} gnorm={float(grad_norm):.2f}")

        if is_main and step % cfg.checkpoint_every_steps == 0:
            save_training_checkpoint(model, optimizer, scheduler, out_dir / f"ckpt-{step}",
                                     global_step=step, tokens_seen=tokens_seen,
                                     tokenizer_fingerprint=cfg.tokenizer_fingerprint,
                                     precision=cfg.precision)

        if is_main and step % cfg.eval_every_steps == 0 and val_data is not None:
            model.eval()
            with torch.no_grad():
                from torch.utils.data import DataLoader as _DL
                vdl = _DL(val_data, batch_size=cfg.batch_size, shuffle=False,
                          collate_fn=collate, drop_last=True)
                tot, cnt = 0.0, 0
                for batch in vdl:
                    ids = batch["input_ids"].to(cfg.device)
                    labels = batch["labels"].to(cfg.device)
                    with torch.autocast(device_type=cfg.device, dtype=amp_dtype, enabled=use_amp):
                        hidden = core.forward_hidden(ids)
                        total_loss, n_tok = chunked_linear_cross_entropy(
                            hidden, labels,
                            core.output_projection.out_proj.weight,
                            core.embed_tokens.embedding_weight,
                        )
                        loss = total_loss / n_tok.clamp_min(1)
                    tot += float(loss) * ids.numel()
                    cnt += ids.numel()
                val_loss = tot / max(cnt, 1)
                history.append({"step": step, "val_loss": val_loss})
                print(f"[step {step}] val_loss={val_loss:.4f}")

    if is_main:
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, history
