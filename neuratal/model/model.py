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
from neuratal.model.moe import MoEStats, merge_moe_stats
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
            # Meta tensors carry no storage; running normal_ on them dispatches
            # through the slow fake-tensor paths (seconds per large preset) for
            # nothing. Skip initialization entirely in that case.
            if module.weight.device.type == "meta":
                return
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def _encode(
        self,
        input_ids: torch.Tensor,
        *,
        token_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """ids -> embeddings -> blocks -> final RMSNorm (pre-head)."""
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device)
        hidden = self.embed_tokens(input_ids)
        stats: list[MoEStats] = []
        for layer in self.layers:
            if return_aux:
                hidden, layer_stats = layer(
                    hidden, positions, past_key_values=None,
                    token_mask=token_mask, return_aux=True,
                )
                if layer_stats is not None:
                    stats.append(layer_stats)
            else:
                hidden = layer(hidden, positions, past_key_values=None)
        hidden = self.final_norm(hidden)
        if return_aux:
            return hidden, merge_moe_stats(stats)
        return hidden

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        token_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """Encode token ids into final-norm hidden states (pre-head).

        Used by training code that must avoid materializing [B, T, vocab]
        logits (chunked cross-entropy). Semantics: exactly the pre-head half
        of forward() — ids -> embeddings -> blocks -> final RMSNorm.

        With ``return_aux=True`` the return value is ``(hidden, MoEStats|None)``;
        the stats carry the differentiable router auxiliary/z losses plus
        detached utilization diagnostics. Dense models return ``None`` stats.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2-D [batch, sequence], got {tuple(input_ids.shape)}")
        if input_ids.size(1) < 1:
            raise ValueError("input_ids must contain at least one token")
        return self._encode(input_ids, token_mask=token_mask, return_aux=return_aux)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: Optional[KVCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_only: bool = False,
        return_aux: bool = False,
        token_mask: Optional[torch.Tensor] = None,
    ):
        """Model forward.

        ``output_hidden_only=True`` returns the pre-head hidden states (or a
        ``(hidden, MoEStats)`` pair with ``return_aux=True``).  This exists so
        training can keep the chunked cross-entropy path while still entering
        through ``DistributedDataParallel.forward`` (DDP's reducer preparation
        happens there; bypassing it by calling a submodule method directly is a
        real correctness bug for multi-GPU training).
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D [batch, sequence], got shape {tuple(input_ids.shape)}"
            )
        seq_len = input_ids.size(1)
        if seq_len < 1:
            raise ValueError("input_ids must contain at least one token")

        cfg = self.config
        if output_hidden_only:
            if seq_len > cfg.max_position_embeddings:
                raise ValueError(
                    f"sequence would exceed max_position_embeddings: new tokens={seq_len}, "
                    f"max_position_embeddings={cfg.max_position_embeddings}"
                )
            return self._encode(input_ids, token_mask=token_mask, return_aux=return_aux)
        if return_aux:
            raise ValueError("return_aux=True is only valid with output_hidden_only=True")
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
            "moe_router": 0,
            "moe_experts": 0,
            "moe_shared_expert": 0,
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
            elif ".mlp.router." in name:
                report["moe_router"] += n
            elif ".mlp.experts." in name:
                report["moe_experts"] += n
            elif ".mlp.shared_expert." in name:
                report["moe_shared_expert"] += n
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

    def active_parameters_per_token(self) -> int:
        """Parameters activated for one token (dense: equal to ``num_parameters``).

        All non-expert parameters are active.  For each MoE layer only ``top_k``
        of ``num_experts`` equally-sized experts act on a token; the router and
        the optional shared expert are always active.  This is an
        architecture-level count, not a measured FLOP count.
        """
        cfg = self.config
        if not cfg.is_moe:
            return self.num_parameters()
        total = self.num_parameters()
        expert_total = sum(
            p.numel() for name, p in self.named_parameters() if ".mlp.experts." in name
        )
        active = total - expert_total
        for layer in self.layers:
            if not getattr(layer, "is_moe", False):
                continue
            layer_expert = sum(p.numel() for p in layer.mlp.experts.parameters())
            per_expert = layer_expert // layer.mlp.num_experts
            active += layer.mlp.top_k * per_expert
        return active

    def parameter_breakdown(self) -> dict[str, int]:
        """Total/active MoE accounting plus the role report (honest, no estimates)."""
        report = self.parameter_report()
        breakdown = dict(report)
        breakdown["active_per_token"] = self.active_parameters_per_token()
        breakdown["moe_layers"] = int(self.config.num_moe_layers)
        breakdown["num_experts"] = int(self.config.num_experts)
        breakdown["top_k"] = int(self.config.num_experts_per_token) if self.config.is_moe else 0
        return breakdown
