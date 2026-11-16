"""Rotary Position Embeddings (RoPE), GPT-NeoX / LLaMA rotate-half layout.

* Applied to Q and K only, on the first ``rotary_dim`` channels of each head;
  channels ``[rotary_dim, head_dim)`` pass through untouched (partial-rotary
  support).
* cos/sin are computed in float32, cached as non-persistent buffers (never
  serialized; they are pure functions of ``rope_theta`` and positions), and
  cast to the query/key dtype at apply time.
* Absolute positions are mandatory: for cached decoding the caller passes the
  absolute ``positions`` tensor (``arange(past_length, past_length + T)``), so
  rotating K before appending it to the cache is always correct.
* Requesting positions beyond ``max_position_embeddings`` is an explicit
  error, never a silent wrap-around.

Future RoPE variants should subclass :class:`RotaryEmbeddingBase` and keep the
``forward(q, k, positions)`` signature so call sites do not change.
"""

from __future__ import annotations

import torch
from torch import nn

from neuratal.config import NeuratalConfig


class RotaryEmbeddingBase(nn.Module):
    """Interface for rotary embeddings: apply rotation to Q and K given absolute positions."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        raise NotImplementedError


class RotaryEmbedding(RotaryEmbeddingBase):
    def __init__(self, config: NeuratalConfig):
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.head_dim = config.head_dim
        self.theta = float(config.rope_theta)
        self.max_position_embeddings = config.max_position_embeddings

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_pos, rotary_dim / 2]
        # rotate-half layout: [cos | cos] concatenated so the full rotary_dim is covered
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)  # [max_pos, rotary_dim]
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(self, fn, recurse=True):
        # RoPE tables are fp32 by design: keep them fp32 across .to(dtype)/.half() calls.
        ret = super()._apply(fn, recurse)
        self.cos_cached = self.cos_cached.float()
        self.sin_cached = self.sin_cached.float()
        self.inv_freq = self.inv_freq.float()
        return ret

    def _slice(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() != 1:
            raise ValueError(f"positions must be 1-D [sequence], got shape {tuple(positions.shape)}")
        if positions.numel() == 0:
            raise ValueError("positions must not be empty")
        if positions.device.type == "cpu":
            # Host-side validation is free on CPU and never executed on the CUDA hot path,
            # where it would force a device sync per layer per forward. On CUDA, the model's
            # int-level capacity check (model.forward) enforces the limit without syncs;
            # direct CUDA misuse still fails loudly via index_select bounds asserts.
            min_pos, max_pos = int(positions.min().item()), int(positions.max().item())
            if min_pos < 0:
                raise ValueError(f"positions must be >= 0, got {min_pos}")
            if max_pos >= self.max_position_embeddings:
                raise ValueError(
                    f"positions exceed max_position_embeddings ({self.max_position_embeddings}): {max_pos}"
                )
        cos = self.cos_cached.index_select(0, positions.to(torch.long))
        sin = self.sin_cached.index_select(0, positions.to(torch.long))
        # [1, 1, T, rotary_dim] for broadcasting over [batch, heads, T, rotary_dim]
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        r = self.rotary_dim
        x_rot, x_pass = x[..., :r], x[..., r:]
        half = r // 2
        x1, x2 = x_rot[..., :half], x_rot[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        out = x_rot * cos + rotated * sin
        return torch.cat((out, x_pass), dim=-1) if x_pass.numel() else out

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        if q.dim() != 4 or k.dim() != 4:
            raise ValueError(f"q and k must be 4-D [batch, heads, seq, head_dim], got {q.dim()}-D and {k.dim()}-D")
        if q.size(0) != k.size(0):
            raise ValueError(f"q and k batch mismatch: {q.size(0)} vs {k.size(0)}")
        if q.size(2) != k.size(2):
            raise ValueError(f"q and k sequence mismatch: {q.size(2)} vs {k.size(2)}")
        if q.size(-1) != k.size(-1):
            raise ValueError(f"q and k head_dim mismatch: {q.size(-1)} vs {k.size(-1)}")
        cos, sin = self._slice(positions)
        return self._apply_rotation(q, cos, sin), self._apply_rotation(k, cos, sin)

    def extra_repr(self) -> str:
        return f"rotary_dim={self.rotary_dim}, theta={self.theta}, max_position_embeddings={self.max_position_embeddings}"
