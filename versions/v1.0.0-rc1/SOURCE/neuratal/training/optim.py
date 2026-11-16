"""Optimizer and LR schedule — pluggable factory.

Built-in: AdamW (default), SGD-momentum. Muon can be added by implementing the
same factory contract without touching the engine.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def build_optimizer(model: torch.nn.Module, name: str = "adamw",
                    learning_rate: float = 3e-4,
                    weight_decay: float = 0.1,
                    betas: tuple[float, float] = (0.9, 0.95),
                    eps: float = 1e-8) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, eps=eps)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=learning_rate, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer {name!r} (supported: adamw, sgd)")


class WarmupCosineSchedule:
    """Linear warmup -> cosine decay to ``min_lr_ratio * base_lr``.

    Driven by global optimizer step (not micro-step); scheduler state is
    saved/restored as plain ints via state_dict.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.1):
        if warmup_steps < 0 or total_steps < 1 or warmup_steps >= total_steps:
            raise ValueError(f"invalid schedule: warmup={warmup_steps}, total={total_steps}")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0

    def state_dict(self) -> dict[str, Any]:
        return {"step": self._step, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "min_lr_ratio": self.min_lr_ratio,
                "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step = int(state["step"])
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self.min_lr_ratio = float(state["min_lr_ratio"])
        self.base_lrs = list(state["base_lrs"])

    def lr_at(self, step: int) -> list[float]:
        if step < self.warmup_steps:
            scale = (step + 1) / self.warmup_steps if self.warmup_steps > 0 else 1.0
            return [base * scale for base in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr = self.min_lr_ratio
        return [base * (min_lr + (1.0 - min_lr) * cosine) for base in self.base_lrs]

    def step(self) -> None:
        lrs = self.lr_at(self._step)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self._step += 1

    @property
    def current_step(self) -> int:
        return self._step
