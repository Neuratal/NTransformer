"""Latent-KV attention (MLA-style) — experimental opt-in.

Implements the core idea of DeepSeek's Multi-head Latent Attention
(arXiv:2405.04434): instead of caching expanded K/V per token, cache a
low-rank latent vector ``c_t = W_down x_t`` and expand K/V from it on the fly:

    c_t = W_dkv x_t                  (latent, L dims)
    k_t = W_uk c_t                    (per-head keys)
    v_t = W_uv c_t                    (per-head values)

KV bytes per token per layer = L tensor elements (vs 2·H_kv·d_h for GQA).
For dev geometry (D=256, H=4, H_kv=2, d_h=64, L=64): 256 B vs 1024 B per token
per layer fp32 — a 4× reduction, identical to the ratio reported for DeepSeek
V2/V3-family MLA (a quality delta requires training; this is only the
inference-memory axis).

Prototype scope (documented):
* Unified RoPE, rotated separately for q (new-token positions) and k (full
  absolute range after latent expansion — q/k sequence lengths differ under
  cache). MLA's full form uses a decoupled rope-subspace to avoid re-rotating
  expanded k each step; that is not implemented here (rotation of 512×64
  elements per step is negligible at this scale).
* No weight absorption (DeepSeek's W_uk-into-q trick), so decode does one
  up-projection per step — cost is small relative to launch-bound savings.
* SDPA dispatch (is_causal / single-token / lower-right mask) reused as-is;
  enable_gqa supported. Cache is :class:`LatentKVCache` (stores only latent).
* fp32 internals not required (no recurrence); dtype follows model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from neuratal.config import NeuratalConfig
from neuratal.model.attention import build_lower_right_mask, reference_attention
from neuratal.model.cache import CacheOverflowError, KVCache
from neuratal.model.positional import RotaryEmbedding


class LatentKVCache:
    """Per-layer latent cache: [batch, 1, capacity, latent_dim].

    Same contract as KVCache: preallocated buffers, explicit commit(),
    overflow = hard error, reset() rewinds for reuse.
    """

    def __init__(self, config: NeuratalConfig, latent_dim: int):
        self.capacity = int(config.cache_capacity or config.max_position_embeddings)
        self.num_layers = config.num_hidden_layers
        self.latent_dim = int(latent_dim)
        self._buffers: list[torch.Tensor] = []
        self._length = 0
        self._pending = 0
        self.batch_size = None
        self.device = None
        self.dtype = None

    @property
    def is_initialized(self) -> bool:
        return bool(self._buffers)

    def get_seq_length(self) -> int:
        return self._length

    def initialize(self, batch_size: int, device, dtype) -> None:
        if self.is_initialized:
            raise RuntimeError("LatentKVCache is already initialized; call reset() instead")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        shape = (batch_size, 1, self.capacity, self.latent_dim)
        self._buffers = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.num_layers)]
        self.batch_size, self.device, self.dtype = batch_size, torch.device(device), dtype

    def update(self, layer_idx: int, latent: torch.Tensor):
        if latent.dim() != 3:  # [batch, seq, latent_dim]
            raise ValueError(f"latent must be 3-D [batch, seq, latent_dim], got {latent.dim()}-D")
        if latent.size(1) < 1:
            raise ValueError("latent must contain at least one token")
        new_tokens = latent.size(1)
        if not self.is_initialized:
            self.initialize(latent.size(0), latent.device, latent.dtype)
        elif latent.size(0) != self.batch_size:
            raise RuntimeError(
                f"batch size changed from {self.batch_size} to {latent.size(0)}; create a new cache"
            )
        start, end = self._length, self._length + new_tokens
        if end > self.capacity:
            raise CacheOverflowError(
                f"LatentKVCache overflow at layer {layer_idx}: capacity={self.capacity}, "
                f"current length={self._length}, incoming tokens={new_tokens}"
            )
        buf = self._buffers[layer_idx]
        buf[:, :, start:end] = latent.unsqueeze(1)
        self._pending = new_tokens
        return buf[:, :, :end]  # [batch, 1, end, latent_dim]

    def commit(self, num_tokens: int) -> None:
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if self._pending == 0:
            raise RuntimeError("commit() without a preceding update() in this forward pass")
        if num_tokens != self._pending:
            raise RuntimeError(f"commit({num_tokens}) != tokens written by update() ({self._pending})")
        self._length += num_tokens
        self._pending = 0

    def reset(self) -> None:
        self._length = 0
        self._pending = 0

    def __repr__(self) -> str:
        bytes_per_token = self.latent_dim * 4
        return (f"LatentKVCache(len={self._length}, capacity={self.capacity}, "
                f"latent_dim={self.latent_dim}, {bytes_per_token} B/token/layer)")


class LatentKVAttention(nn.Module):
    def __init__(self, config: NeuratalConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.latent_dim = int(config.mla_latent_dim)
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.num_query_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.kv_down_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.k_up_proj = nn.Linear(self.latent_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(self.latent_dim, self.num_kv_heads * self.head_dim, bias=False)
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

        q = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        latent = self.kv_down_proj(hidden_states)  # [B, T, L]

        if past_key_values is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when past_key_values is provided")
            if not isinstance(past_key_values, LatentKVCache):
                raise TypeError(
                    f"LatentKVAttention requires LatentKVCache, got {type(past_key_values).__name__}"
                )
            latent_view = past_key_values.update(layer_idx, latent)  # [B, 1, T', L]
            kv_flat = latent_view.reshape(batch_size, -1, self.latent_dim)  # [B, T', L]
        else:
            kv_flat = latent

        k = F.linear(kv_flat, self.k_up_proj.weight).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = F.linear(kv_flat, self.v_up_proj.weight).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        kv_len = k.size(2)

        # RoPE: q is rotated with the new-token positions; k (expanded from the
        # full latent history) is rotated with absolute positions over the whole
        # range — q/k sequence lengths differ when a cache is present.
        kv_positions = torch.arange(kv_len, device=positions.device)
        cos_q, sin_q = self.rope._slice(positions)
        cos_k, sin_k = self.rope._slice(kv_positions)
        q = self.rope._apply_rotation(q, cos_q, sin_q)
        k = self.rope._apply_rotation(k, cos_k, sin_k)

        dropout_p = self.attention_dropout if self.training else 0.0
        if self.attention_implementation == "reference":
            attn_out = reference_attention(
                q, k, v, num_query_groups=self.num_query_groups,
                scaling=self.scaling, dropout_p=dropout_p, training=self.training,
            )
        else:
            kwargs: dict = {"dropout_p": dropout_p}
            if q_len == kv_len:
                kwargs["is_causal"] = True
            elif q_len == 1:
                pass  # single-token decode: all past visible
            else:
                kwargs["attn_mask"] = build_lower_right_mask(q_len, kv_len, q.device)
            needs_gqa = self.num_heads != self.num_kv_heads
            if needs_gqa:
                kwargs["enable_gqa"] = True
            try:
                attn_out = F.scaled_dot_product_attention(q, k, v, **kwargs)
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
                attn_out = F.scaled_dot_product_attention(q, k, v, **kwargs)

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
                f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
                f"latent_dim={self.latent_dim}")
