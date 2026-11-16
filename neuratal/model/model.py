"""NeuratalModel — decoder-only autoregressive Transformer (pre-norm).

Topology per forward pass:

    ids -> FactorizedEmbedding (+ in_proj) -> N x NeuratalBlock -> RMSNorm
         -> OutputProjection (tied matrix, passed at call time) -> logits

The tied embedding matrix exists exactly once in the module tree, so it is
counted once by ``numel()``/``state_dict()``. ``parameter_report()`` attributes
every parameter to a role for auditing (embeddings / projections / attention /
MLP / normalization / other); the blocks sum exactly to the model total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from neuratal.config import NeuratalConfig
from neuratal.model.block import NeuratalBlock
from neuratal.model.cache import KVCache
from neuratal.model.embeddings import FactorizedEmbedding
from neuratal.model.gated_delta_net import DeltaNetCache
from neuratal.model.normalization import RMSNorm
from neuratal.model.output import OutputProjection


_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    past_key_values: Optional[KVCache]


class NeuratalModel(nn.Module):
    def __init__(self, config: NeuratalConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = FactorizedEmbedding(config)
        self.layers = nn.ModuleList(
            NeuratalBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_projection = OutputProjection(config)
        self.apply(self._init_module)
        self.to(_DTYPE_MAP[config.dtype])

    # ------------------------------------------------------------------
    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: Optional[KVCache] = None,
        use_cache: Optional[bool] = None,
    ) -> CausalLMOutput:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D [batch, sequence], got shape {tuple(input_ids.shape)}"
            )
        seq_len = input_ids.size(1)
        if seq_len < 1:
            raise ValueError("input_ids must contain at least one token")

        cfg = self.config
        effective_use_cache = cfg.use_cache if use_cache is None else use_cache
        if past_key_values is None and effective_use_cache:
            past_key_values = (
                DeltaNetCache(cfg) if cfg.attention_type == "gated_delta" else KVCache(cfg)
            )

        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_length + seq_len > cfg.max_position_embeddings:
            raise ValueError(
                f"sequence would exceed max_position_embeddings: past={past_length}, "
                f"new tokens={seq_len}, max_position_embeddings={cfg.max_position_embeddings}"
            )
        positions = torch.arange(past_length, past_length + seq_len, device=input_ids.device)

        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, positions, past_key_values=past_key_values)
        hidden = self.final_norm(hidden)

        logits = self.output_projection(hidden, self.embed_tokens.embedding_weight)
        if past_key_values is not None:
            past_key_values.commit(seq_len)
        return CausalLMOutput(
            logits=logits,
            past_key_values=past_key_values if past_key_values is not None else None,
        )

    # ------------------------------------------------------------------
    def parameter_report(self) -> dict[str, int]:
        """Per-role parameter counts. Sums exactly to the total (tied matrix counted once)."""
        report = {
            "embedding_matrix": 0,
            "input_projection": 0,
            "output_projection": 0,
            "attention": 0,
            "mlp": 0,
            "normalization": 0,
            "other": 0,
        }
        total = 0
        for name, param in self.named_parameters():
            n = param.numel()
            total += n
            if name.startswith("embed_tokens.token_embedding"):
                report["embedding_matrix"] += n
            elif name.startswith("embed_tokens.in_proj"):
                report["input_projection"] += n
            elif name.startswith("output_projection"):
                report["output_projection"] += n
            elif ".attn." in name:
                report["attention"] += n
            elif ".mlp." in name:
                report["mlp"] += n
            elif "norm" in name:
                report["normalization"] += n
            else:
                report["other"] += n
        report["total"] = total
        return report

    def num_parameters(self) -> int:
        """Total trainable parameters; shared tensors counted once."""
        return sum(p.numel() for _, p in self.named_parameters())
