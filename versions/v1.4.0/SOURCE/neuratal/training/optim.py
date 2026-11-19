# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
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

    LR contract (v1.4): the param groups hold ``lr_at(0)`` after construction
    (the first optimizer step), and each ``step()`` call advances them to the
    next schedule position — after *k* calls groups hold ``lr_at(k)``. After
    ``total_steps`` scheduler steps the groups hold ``lr_at(total_steps)``,
    i.e. the ``min_lr_ratio * base_lr`` floor exactly.
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
        # Apply the schedule from step 0 so the first optimizer step already
        # runs at the warmup LR (lr_at(0)); otherwise the first step would
        # run at the full base LR even mid-warmup.
        self._apply(self.lr_at(0))

    def state_dict(self) -> dict[str, Any]:
        return {"step": self._step, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "min_lr_ratio": self.min_lr_ratio,
                "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        required = ("step", "warmup_steps", "total_steps", "min_lr_ratio", "base_lrs")
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(f"scheduler state is missing required keys {missing}; got keys {sorted(state)}")
        try:
            self._step = self._require_int("step", state["step"])
            self.warmup_steps = self._require_int("warmup_steps", state["warmup_steps"])
            self.total_steps = self._require_int("total_steps", state["total_steps"])
            self.min_lr_ratio = self._require_finite_float("min_lr_ratio", state["min_lr_ratio"], lo=0.0, hi=1.0)
            if self._step < 0 or self.warmup_steps < 0 or self.total_steps < 1:
                raise ValueError(
                    f"invalid scheduler state values: step={self._step}, "
                    f"warmup_steps={self.warmup_steps}, total_steps={self.total_steps}"
                )
            if self.warmup_steps >= self.total_steps:
                raise ValueError(
                    f"invalid scheduler state: warmup_steps ({self.warmup_steps}) must be "
                    f"< total_steps ({self.total_steps}); refusing to install a nonsense schedule"
                )
        except TypeError as exc:
            raise ValueError(f"scheduler state has invalid field types: {exc}") from exc
        restored_lrs = self._require_finite_float_list("base_lrs", state["base_lrs"], positive=True)
        if len(restored_lrs) != len(self.base_lrs):
            raise ValueError(
                f"scheduler state base_lrs has length {len(restored_lrs)} but this optimizer "
                f"has {len(self.base_lrs)} param group(s); refusing to restore a mismatched schedule"
            )
        self.base_lrs = restored_lrs
        # Reapply the LR the param groups held at save time: the groups always
        # hold lr_at(_step) (lr_at(0) when saved at step 0). This keeps a
        # resumed run's first optimizer step at exactly the LR the interrupted
        # run would have used.
        self._apply(self.lr_at(self._step))

    def _apply(self, lrs: list[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    @staticmethod
    def _require_int(name: str, value) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        ivalue = int(value)
        if float(ivalue) != float(value):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        return ivalue

    @staticmethod
    def _require_finite_float(name: str, value, *, lo: float, hi: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {value!r}")
        fvalue = float(value)
        if not math.isfinite(fvalue) or fvalue < lo or fvalue > hi:
            raise ValueError(f"{name} must be finite and in [{lo}, {hi}], got {value!r}")
        return fvalue

    @staticmethod
    def _require_finite_float_list(name: str, value, *, positive: bool = False) -> list[float]:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{name} must be a list of numbers, got {type(value).__name__}")
        out: list[float] = []
        for i, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"{name}[{i}] must be a number, got {item!r}")
            fvalue = float(item)
            if not math.isfinite(fvalue):
                raise ValueError(f"{name}[{i}] must be finite, got {item!r}")
            if positive and fvalue <= 0.0:
                raise ValueError(f"{name}[{i}] must be a positive learning rate, got {item!r}")
            out.append(fvalue)
        return out

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
        # Advance to the NEXT schedule position before computing the LR, so
        # that after k step() calls the param groups hold lr_at(k). With the
        # construction-time lr_at(0), the first optimizer step runs at lr_at(0),
        # the second at lr_at(1), ..., and the total_steps-th scheduler step
        # lands exactly on lr_at(total_steps) (the min_lr floor).
        self._step += 1
        lrs = self.lr_at(self._step)
        self._apply(lrs)

    @property
    def current_step(self) -> int:
        return self._step
