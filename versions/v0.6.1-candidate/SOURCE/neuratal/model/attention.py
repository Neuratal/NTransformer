"""Grouped-Query Attention (GQA) with causal masking.

Correctness rules enforced here (see docs/architecture_audit.md §5.2)
--------------------------------------------------------------------
SDPA's ``is_causal=True`` is only valid when ``q_len == kv_len``. For cached
decoding (``q_len < kv_len``) PyTorch aligns the causal mask *upper-left*,
which is wrong — the mask must be aligned to the absolute cache offset
(*lower-right*). We therefore dispatch on shapes:

* ``q_len == kv_len``  -> ``is_causal=True`` (no mask tensor; flash eligible)
* ``q_len == 1``       -> no mask (every past key is legitimately visible)
* otherwise (1 < q_len < kv_len, i.e. chunked prefill) -> explicit
  lower-right-aligned boolean mask ``key_pos <= query_pos``; this path may
  fall back off the flash backend inside SDPA, with identical math.

GQA
---
Groups of ``num_query_groups = H // H_kv`` query heads share each KV head.
We pass ``enable_gqa=True`` to SDPA when available (bit-identical to manual
expansion on the tested build, torch 2.15.0+cu132, all backends). If SDPA
rejects it at runtime, we fall back explicitly to ``repeat_interleave`` with a
one-time warning — never silently. ``reference_attention`` (pure PyTorch,
fp32 softmax) is kept for correctness testing and debug deployments; select
it via ``config.attention_implementation="reference"``. It is not a fast path.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from neuratal.config import NeuratalConfig
from neuratal.model.positional import RotaryEmbedding


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_query_groups: int,
    scaling: float,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch attention (fp32 scores and softmax). A correctness reference, not a fast path."""
    if num_query_groups > 1:
        k = k.repeat_interleave(num_query_groups, dim=1)
        v = v.repeat_interleave(num_query_groups, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scaling
    q_len, kv_len = q.size(2), k.size(2)
    offset = kv_len - q_len
    i = torch.arange(q_len, device=q.device).unsqueeze(1)
    j = torch.arange(kv_len, device=q.device).unsqueeze(0)
    disallowed = j > i + offset  # strictly the future, in absolute positions
    scores = scores.float().masked_fill(disallowed, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        probs = F.dropout(probs, p=dropout_p, training=True)
    return torch.matmul(probs.to(v.dtype), v)


def build_lower_right_mask(q_len: int, kv_len: int, device: torch.device | str) -> torch.Tensor:
    """Boolean SDPA mask (True = attend) aligned to the cache offset, shape [q_len, kv_len]."""
    offset = kv_len - q_len
    if offset < 0:
        raise ValueError(f"kv_len ({kv_len}) must be >= q_len ({q_len})")
    i = torch.arange(q_len, device=device).unsqueeze(1)
    j = torch.arange(kv_len, device=device).unsqueeze(0)
    return j <= i + offset


class GQAAttention(nn.Module):
    def __init__(self, config: NeuratalConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_query_groups = self.num_heads // self.num_kv_heads
        self.hidden_size = config.hidden_size
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        qkv_out = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        # Fused QKV projection: one GEMM instead of three; mathematically identical
        # (weight rows = cat([W_q, W_k, W_v], dim=0)). Split after projection.
        self.qkv_proj = nn.Linear(self.hidden_size, qkv_out, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = RotaryEmbedding(config)
        self.attention_dropout = float(config.attention_dropout)
        self.attention_implementation = config.attention_implementation
        self._warned_gqa_fallback = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        past_key_values=None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be 3-D [batch, seq, hidden], got {hidden_states.dim()}-D")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_states.size(-1)}")
        batch_size, q_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim,
                  self.num_kv_heads * self.head_dim], dim=-1)
        q = q.reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, positions)

        if past_key_values is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when past_key_values is provided")
            k, v = past_key_values.update(layer_idx, k, v)

        kv_len = k.size(2)
        attn_output = self._core_attention(q, k, v, kv_len)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    # ------------------------------------------------------------------
    def _core_attention(self, q, k, v, kv_len: int) -> torch.Tensor:
        q_len = q.size(2)
        dropout_p = self.attention_dropout if self.training else 0.0

        if self.attention_implementation == "reference":
            return reference_attention(
                q, k, v,
                num_query_groups=self.num_query_groups,
                scaling=self.scaling,
                dropout_p=dropout_p,
                training=self.training,
            )

        # SDPA path — position-regime dispatch (never blind is_causal)
        kwargs: dict = {"dropout_p": dropout_p}
        if q_len == kv_len:
            kwargs["is_causal"] = True
        elif q_len == 1:
            pass  # single-token decode: all cached positions are visible
        else:
            kwargs["attn_mask"] = build_lower_right_mask(q_len, kv_len, q.device)

        needs_gqa = self.num_heads != self.num_kv_heads
        if needs_gqa:
            kwargs["enable_gqa"] = True
        try:
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
        except (RuntimeError, TypeError) as exc:
            if not needs_gqa:
                raise
            # Explicit, tested fallback: expand KV heads manually (identical math).
            self._warn_gqa_fallback(exc)
            kwargs.pop("enable_gqa")
            k_exp = k.repeat_interleave(self.num_query_groups, dim=1)
            v_exp = v.repeat_interleave(self.num_query_groups, dim=1)
            return F.scaled_dot_product_attention(q, k_exp, v_exp, **kwargs)

    def _warn_gqa_fallback(self, exc: Exception) -> None:
        if not self._warned_gqa_fallback:
            warnings.warn(
                f"SDPA rejected enable_gqa ({exc}); falling back to explicit repeat_interleave "
                "KV expansion. Math is identical, but this path is slower.",
                UserWarning,
                stacklevel=3,
            )
            self._warned_gqa_fallback = True

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"num_key_value_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"implementation={self.attention_implementation}"
        )
