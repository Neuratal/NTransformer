# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""NeuratalBlock — pre-norm Transformer block (attention + SwiGLU MLP).

    h = h + hidden_dropout( Attn( RMSNorm(h) ) )
    h = h + hidden_dropout( MLP( RMSNorm(h) ) )

Residual connections, two RMSNorms per block, no post-norm. The block owns no
positional or cache state; RoPE positions and the KV cache are threaded through
from the model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from neuratal.config import NeuratalConfig
from neuratal.model.attention import GQAAttention
from neuratal.model.gated_delta_net import GatedDeltaNetAttention
from neuratal.model.mlp import SwiGLU
from neuratal.model.moe import MoEFFN, MoEStats
from neuratal.model.normalization import RMSNorm


class NeuratalBlock(nn.Module):
    def __init__(self, config: NeuratalConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = (
            GatedDeltaNetAttention(config)
            if config.attention_type == "gated_delta"
            else GQAAttention(config)
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_moe = bool(config.is_moe and layer_idx in config.moe_layer_indices)
        self.mlp = MoEFFN(config) if self.is_moe else SwiGLU(config)
        self.hidden_dropout = float(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        past_key_values=None,
        token_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Run the block.

        ``return_aux=False`` keeps the historical tensor-only return value.
        With ``return_aux=True`` the result is ``(hidden_states, MoEStats|None)``
        — dense layers return ``None`` for the stats, so mixed dense/MoE stacks
        aggregate exactly the MoE layers.
        """
        residual = hidden_states
        attn_out = self.attn(
            self.attention_norm(hidden_states), positions,
            past_key_values=past_key_values, layer_idx=self.layer_idx,
        )
        if self.hidden_dropout > 0.0 and self.training:
            attn_out = F.dropout(attn_out, p=self.hidden_dropout, training=True)
        hidden_states = residual + attn_out

        residual = hidden_states
        stats: MoEStats | None = None
        if self.is_moe:
            mlp_out, stats = self.mlp(
                self.mlp_norm(hidden_states),
                compute_aux=return_aux,
                token_mask=token_mask,
            )
        else:
            mlp_out = self.mlp(self.mlp_norm(hidden_states))
        if self.hidden_dropout > 0.0 and self.training:
            mlp_out = F.dropout(mlp_out, p=self.hidden_dropout, training=True)
        hidden_states = residual + mlp_out
        if return_aux:
            return hidden_states, stats
        return hidden_states
