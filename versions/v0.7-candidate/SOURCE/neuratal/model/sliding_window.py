"""Sliding-window attention (Mistral-style) — experimental opt-in.

Token i attends to keys j with absolute positions in [i - W + 1, i] (causal,
windowed). Everything else mirrors GQA: RoPE, GQA head-sharing, SDPA dispatch,
KV cache stores full K/V (no eviction — correctness first; eviction is a
memory research branch, not built here).

Why it exists here: the honest indexer-free subset of "sparse attention" from
the 2026 reference set. It bounds prefill/decode attention FLOPs for long
contexts. Quality vs dense is NOT measurable without training — candidate only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from neuratal.config import NeuratalConfig
from neuratal.model.attention import GQAAttention, reference_attention
from neuratal.model.positional import RotaryEmbedding


def build_sliding_window_mask(q_len: int, kv_len: int, window: int, device):
    """Boolean SDPA mask (True = attend) with lower-right-aligned window [i+offset-W+1, i+offset]."""
    offset = kv_len - q_len
    i = torch.arange(q_len, device=device).unsqueeze(1)
    j = torch.arange(kv_len, device=device).unsqueeze(0)
    pos = i + offset
    return (j <= pos) & (j > pos - window)


class SlidingWindowAttention(GQAAttention):
    def __init__(self, config: NeuratalConfig):
        super().__init__(config)
        self.swa_window = int(config.swa_window)

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
        kwargs: dict = {
            "dropout_p": dropout_p,
            "attn_mask": build_sliding_window_mask(q_len, kv_len, self.swa_window, q.device),
        }
        needs_gqa = self.num_heads != self.num_kv_heads
        if needs_gqa:
            kwargs["enable_gqa"] = True
        try:
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
        except (RuntimeError, TypeError) as exc:
            if not needs_gqa:
                raise
            if not self._warned_gqa_fallback:
                self._warned_gqa_fallback = True
                import warnings

                warnings.warn(
                    f"SDPA rejected enable_gqa ({exc}); falling back to repeat_interleave KV expansion",
                    UserWarning,
                    stacklevel=3,
                )
            k = k.repeat_interleave(self.num_query_groups, dim=1)
            v = v.repeat_interleave(self.num_query_groups, dim=1)
            kwargs.pop("enable_gqa")
            return F.scaled_dot_product_attention(q, k, v, **kwargs)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", swa_window={self.swa_window}"
