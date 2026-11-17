"""RMSNorm — root-mean-square normalization.

Design notes
------------
* Computes the mean of squares in float32 regardless of the input dtype
  (fp16/bf16 cannot reliably represent the reduction), then casts back.
* No bias, no mean-centering, no running statistics. Weight initialized to 1.
* ``eps`` is configurable via :class:`neuratal.config.NeuratalConfig`.
* When the weight dtype matches the input dtype *and no autocast region is
  active* (e.g. an fp32 model without autocast, or a fully bf16 model) this
  dispatches to the fused ``F.rms_norm`` kernel. Fused accumulation order
  differs from the manual path at the 1-ulp level (<= ~1e-6 for fp32, ~1 ulp
  for bf16); architecture, parameters and checkpoints are unaffected. When
  the dtypes differ — e.g. an fp32 model running under bf16 autocast, where
  the input is bf16 but the stored weight is fp32 — or when autocast is
  active (it promotes rms_norm to fp32, changing the output dtype), the
  explicit fp32 path below is kept, because the fused kernel cannot dispatch
  or would silently change the dtype contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _autocast_active(device: torch.device) -> bool:
    kind = "cuda" if device.type == "cuda" else device.type
    try:
        return bool(torch.is_autocast_enabled(kind))  # torch >= 2.4 device-aware API
    except TypeError:  # pragma: no cover - older torch fallback
        return bool(torch.is_autocast_enabled() if kind == "cuda" else torch.is_autocast_cpu_enabled())


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
        self._normalized_shape = (hidden_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.hidden_size:
            raise ValueError(f"RMSNorm expected last dim {self.hidden_size}, got {x.size(-1)}")
        if (
            x.dtype == self.weight.dtype
            and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and not _autocast_active(x.device)
        ):
            return F.rms_norm(x, self._normalized_shape, self.weight, self.eps)
        input_dtype = x.dtype
        x32 = x.float()
        mean_square = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(mean_square + self.eps)
        return (x32 * self.weight.float()).to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
