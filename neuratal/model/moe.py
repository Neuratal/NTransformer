"""Mixture-of-Experts feed-forward network (opt-in reference implementation).

Design goals
------------
* **Correct and readable first.**  Expert dispatch is tensor-based: tokens are
  sorted by expert id once, then each expert processes a contiguous slice with
  a single batched GEMM.  The only Python loop is over experts (never over
  tokens), which is the standard reference-dispatch shape.
* **Dense path untouched.**  This module is instantiated only when
  ``config.num_experts >= 2`` and the layer is selected by
  ``config.moe_layer_indices``.
* **Router numerics in fp32.**  Router logits/softmax and both auxiliary losses
  are computed in fp32 regardless of the model dtype (bf16 router softmax is
  not trustworthy).  Gate weights are cast back to the expert activation dtype
  only for the final weighted sum.
* **No training loss at inference.**  ``compute_aux=False`` still returns
  utilization metrics but never builds the auxiliary-loss graph.  Generation
  and evaluation do not pay for training-only terms.
* **Capacity is explicit.**  ``capacity_factor=None`` means no dropping (the
  default reference behaviour).  When set, tokens beyond
  ``capacity_factor * tokens * top_k / num_experts`` assignments per expert are
  dropped and counted; they contribute no expert output and no main-loss
  gradient.  Padding-masked tokens are excluded before routing.

The auxiliary load-balancing loss is the Switch-Transformer form

    aux = coef * E * sum_e (f_e * P_e)

where ``f_e`` is the fraction of top-k assignments routed to expert ``e`` and
``P_e`` is the mean router probability of expert ``e`` over tokens.  The
optional z-loss is ``coef * mean(logsumexp(router_logits)^2)``.  Both are
computed from pre-capacity assignments, which is the usual convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from neuratal.config import NeuratalConfig
from neuratal.model.mlp import SwiGLU


@dataclass
class MoEStats:
    """Per-layer (or merged) MoE diagnostics.

    ``aux_loss`` / ``z_loss`` are differentiable scalars when ``compute_aux``
    was requested; otherwise they are ``None``.  All metric tensors are
    detached, so callers can log them without extending the autograd graph.
    """

    counts: torch.Tensor          # [E] float32: top-k assignments per expert
    probs: torch.Tensor           # [E] float32: mean router probability per expert
    entropy: torch.Tensor         # scalar: mean routing entropy (nats)
    dropped: torch.Tensor         # scalar: dropped assignments (capacity only)
    num_tokens: int
    num_assignments: int
    aux_loss: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None


def merge_moe_stats(stats: list[MoEStats]) -> Optional[MoEStats]:
    """Aggregate per-layer stats into one training-log record."""
    if not stats:
        return None
    aux = [s.aux_loss for s in stats if s.aux_loss is not None]
    z = [s.z_loss for s in stats if s.z_loss is not None]
    counts = torch.stack([s.counts for s in stats], dim=0).sum(dim=0)
    probs = torch.stack([s.probs for s in stats], dim=0).mean(dim=0)
    entropy = torch.stack([s.entropy for s in stats]).mean()
    dropped = torch.stack([s.dropped for s in stats]).sum()
    return MoEStats(
        counts=counts,
        probs=probs,
        entropy=entropy,
        dropped=dropped,
        num_tokens=sum(s.num_tokens for s in stats),
        num_assignments=sum(s.num_assignments for s in stats),
        aux_loss=torch.stack(aux).sum() if aux else None,
        z_loss=torch.stack(z).sum() if z else None,
    )


class MoEFFN(nn.Module):
    """Top-k routed SwiGLU experts with an fp32 router."""

    def __init__(self, config: NeuratalConfig):
        super().__init__()
        if not config.is_moe:
            raise ValueError("MoEFFN requires config.num_experts >= 2")
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.expert_hidden_dim = int(config.expert_hidden_dim)
        self.router_type = config.router_type
        self.capacity_factor = config.capacity_factor
        self.min_capacity = int(config.min_capacity)
        self.router_aux_loss_coef = float(config.router_aux_loss_coef)
        self.router_z_loss_coef = float(config.router_z_loss_coef)

        # No bias: bias shifts are not needed and keep the router auditable.
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(config, intermediate_size=self.expert_hidden_dim)
            for _ in range(self.num_experts)
        )
        self.shared_expert = (
            SwiGLU(config, intermediate_size=int(config.shared_expert_hidden_dim))
            if config.shared_expert
            else None
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        compute_aux: bool = True,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[MoEStats]]:
        """Apply routed experts.

        Args:
            x: ``[batch, seq, hidden]`` hidden states.
            compute_aux: build the differentiable auxiliary/z losses.
            token_mask: optional ``[batch, seq]`` boolean mask; ``True`` marks
                real tokens.  Masked positions are excluded from routing and
                their output is exactly zero (they still pass through the
                residual connection outside this module).
        """
        if x.dim() != 3:
            raise ValueError(f"expected 3-D input [batch, seq, hidden], got {x.dim()}-D")
        if x.size(-1) != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {x.size(-1)}")

        batch, seq, hidden = x.shape
        flat = x.reshape(-1, hidden)

        if token_mask is None:
            valid_index = None
            routed = flat
        else:
            if token_mask.dim() == 3 and token_mask.size(-1) == 1:
                token_mask = token_mask.squeeze(-1)
            if token_mask.shape != (batch, seq):
                raise ValueError(
                    f"token_mask must have shape {(batch, seq)}, got {tuple(token_mask.shape)}"
                )
            valid = token_mask.reshape(-1).to(torch.bool)
            valid_index = torch.nonzero(valid, as_tuple=False).squeeze(1)
            routed = flat.index_select(0, valid_index)

        routed_out, stats = self._forward_tokens(routed, compute_aux=compute_aux)

        if valid_index is None:
            out = routed_out
        else:
            # index_copy (out-of-place) keeps autograd well-defined and leaves
            # masked positions at exactly zero.
            out = flat.new_zeros((flat.size(0), hidden)).index_copy(0, valid_index, routed_out)
        return out.reshape(batch, seq, hidden), stats

    # ------------------------------------------------------------------
    def _forward_tokens(
        self, x: torch.Tensor, *, compute_aux: bool
    ) -> tuple[torch.Tensor, MoEStats]:
        num_tokens = x.size(0)
        device = x.device
        if num_tokens == 0:
            zeros = x.new_zeros((0, self.hidden_size))
            stats = MoEStats(
                counts=torch.zeros(self.num_experts, device=device),
                probs=torch.zeros(self.num_experts, device=device),
                entropy=torch.zeros((), device=device),
                dropped=torch.zeros((), device=device),
                num_tokens=0,
                num_assignments=0,
                aux_loss=torch.zeros((), device=device) if compute_aux else None,
                z_loss=torch.zeros((), device=device) if compute_aux else None,
            )
            return zeros, stats

        # --- router (always fp32) ---
        logits = F.linear(x.float(), self.router.weight.float())  # [N, E]
        probs = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        # Normalize the selected gates to sum to one.  Top-1 becomes exactly 1.
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        flat_indices = topk_indices.reshape(-1)          # [N * k]
        flat_weights = topk_weights.reshape(-1)          # [N * k]
        counts = torch.bincount(flat_indices, minlength=self.num_experts).to(torch.float32)

        # --- dispatch: sort assignments by expert once ---
        order = torch.argsort(flat_indices, stable=True)
        sorted_experts = flat_indices.index_select(0, order)
        sorted_tokens = torch.div(order, self.top_k, rounding_mode="floor")
        sorted_weights = flat_weights.index_select(0, order)

        offsets = torch.cumsum(counts.to(torch.long), dim=0) - counts.to(torch.long)
        offsets_list = offsets.tolist()  # one host sync, not one per expert

        keep = None
        if self.capacity_factor is not None:
            capacity = max(
                self.min_capacity,
                int(math.ceil(self.capacity_factor * num_tokens * self.top_k / self.num_experts)),
            )
            positions = torch.arange(
                num_tokens * self.top_k, device=device, dtype=torch.long
            ) - offsets.index_select(0, sorted_experts)
            keep = positions < capacity
        dropped = (
            torch.tensor(0.0, device=device)
            if keep is None
            else (num_tokens * self.top_k - keep.sum()).to(torch.float32)
        )

        out = x.new_zeros((num_tokens, self.hidden_size))
        total_assignments = num_tokens * self.top_k
        for expert_idx in range(self.num_experts):
            start = offsets_list[expert_idx]
            end = offsets_list[expert_idx + 1] if expert_idx + 1 < self.num_experts else total_assignments
            if end <= start:
                continue
            token_ids = sorted_tokens[start:end]
            weights = sorted_weights[start:end]
            if keep is not None:
                keep_slice = keep[start:end]
                token_ids = token_ids[keep_slice]
                weights = weights[keep_slice]
            if token_ids.numel() == 0:
                continue
            expert_out = self.experts[expert_idx](x.index_select(0, token_ids))
            contribution = expert_out * weights.to(expert_out.dtype).unsqueeze(-1)
            out = out.index_add(0, token_ids, contribution)

        if self.shared_expert is not None:
            out = out + self.shared_expert(x)

        # --- diagnostics / auxiliary losses ---
        assignments = counts.sum().clamp_min(1.0)
        f = counts / assignments
        mean_prob = probs.mean(dim=0)
        entropy = -(probs.clamp_min(1e-9).log() * probs).sum(dim=-1).mean()

        aux_loss = None
        z_loss = None
        if compute_aux:
            aux_loss = self.router_aux_loss_coef * self.num_experts * (f * mean_prob).sum()
            z_loss = self.router_z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()

        stats = MoEStats(
            counts=counts.detach(),
            probs=mean_prob.detach(),
            entropy=entropy.detach(),
            dropped=dropped.detach(),
            num_tokens=num_tokens,
            num_assignments=num_tokens * self.top_k,
            aux_loss=aux_loss,
            z_loss=z_loss,
        )
        return out, stats

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, expert_hidden_dim={self.expert_hidden_dim}, "
            f"router={self.router_type}, capacity_factor={self.capacity_factor}, "
            f"shared_expert={self.shared_expert is not None}"
        )
