"""RMSNorm — root-mean-square normalization.

Design notes
------------
* Computes the mean of squares in float32 regardless of the input dtype
  (fp16/bf16 cannot reliably represent the reduction), then casts back.
* No bias, no mean-centering, no running statistics. Weight initialized to 1.
* ``eps`` is configurable via :class:`neuratal.config.NeuratalConfig`.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.hidden_size = hidden_size
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.hidden_size:
            raise ValueError(f"RMSNorm expected last dim {self.hidden_size}, got {x.size(-1)}")
        input_dtype = x.dtype
        x32 = x.float()
        mean_square = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(mean_square + self.eps)
        return (x32 * self.weight.float()).to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
