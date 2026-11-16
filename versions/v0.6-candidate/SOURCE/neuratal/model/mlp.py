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
        # Fused gate+up projection: one GEMM instead of two; mathematically identical
        # (weight rows = cat([W_gate, W_up], dim=0)). Split after projection.
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected 3-D input [batch, seq, hidden], got {x.dim()}-D")
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"
