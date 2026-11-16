"""SwiGLU feed-forward network.

    MLP(x) = W_down @ ( SiLU(W_gate @ x) * (W_up @ x) )

Three bias-free matrices (LLaMA-style). No dropout inside the MLP — the block
applies ``hidden_dropout`` on the residual branch instead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from neuratal.config import NeuratalConfig


class SwiGLU(nn.Module):
    def __init__(self, config: NeuratalConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected 3-D input [batch, seq, hidden], got {x.dim()}-D")
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"
