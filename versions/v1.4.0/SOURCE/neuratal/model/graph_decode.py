# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Static-shape CUDA-graph decode session (EXPERIMENTAL, opt-in).

Motivation (measured): eager decode at this scale is launch-bound — ~305 µs
GPU vs ~620–670 µs wall per token. Replaying the decode step as one CUDA
graph removes per-step Python dispatch. The static window is
``next_pow2(prompt_len + max_new_tokens + 1)`` for the whole generation call,
so a single capture serves every step of that call (attention reads only the
valid region via a persistent additive bool mask updated in-place per step;
invalid slots carry ``-inf``).

Correctness contract (bitwise vs the eager path on the tested build):
* same weights, same KV cache contents, same absolute positions;
* masked full-window SDPA is mathematically identical to the eager causal row
  (verified: max |Δlogit| == 0.0 over 16+ steps, fp32 and bf16);
* KV-cache buffers are session-private — eager state is never touched;
* B=1 only; CUDA only; everything runs under ``torch.inference_mode()``.

Duplication warning: the body composes the model's own modules (qkv_proj,
gate_up_proj, norms, rope helpers) but re-implements the per-sublayer flow.
``tests/test_graph_decode.py`` asserts bitwise equality with the eager path;
any future change to NeuratalBlock/GQAAttention must keep that test green.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from neuratal.model.cache import KVCache
from neuratal.model.model import NeuratalModel

_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def next_pow2(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


class GraphDecodeError(RuntimeError):
    """Raised when the static-graph path cannot be used (CUDA-only, B=1)."""


class GraphDecodeSession:
    """Capture and replay the single-token decode step for one generation."""

    def __init__(self, model: NeuratalModel, prompt_ids: torch.Tensor, max_new_tokens: int):
        cfg = model.config
        if not torch.cuda.is_available() or prompt_ids.device.type != "cuda":
            raise GraphDecodeError("CUDA graphs require a CUDA device (eager path remains the default)")
        if prompt_ids.dim() != 2 or prompt_ids.size(0) != 1:
            raise GraphDecodeError(f"static graph decode requires batch size 1, got shape {tuple(prompt_ids.shape)}")
        if prompt_ids.dtype not in (torch.int32, torch.int64):
            raise GraphDecodeError(f"prompt_ids must have an integer dtype, got {prompt_ids.dtype}")
        if not cfg.use_cache:
            raise GraphDecodeError("graph decode requires config.use_cache=True")
        if cfg.attention_dropout > 0.0 or cfg.hidden_dropout > 0.0:
            raise GraphDecodeError(
                "graph decode does not apply dropout (graph body is deterministic); "
                f"set attention_dropout/hidden_dropout to 0 (got {cfg.attention_dropout}, {cfg.hidden_dropout})"
            )
        if max_new_tokens < 1:
            raise GraphDecodeError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        prompt_len = prompt_ids.size(1)
        window = next_pow2(prompt_len + max_new_tokens + 1)
        if window > cfg.cache_capacity:
            raise GraphDecodeError(
                f"static window {window} exceeds cache_capacity {cfg.cache_capacity}; "
                "lower max_new_tokens or raise cache_capacity"
            )

        self.model = model
        self.window = window
        self.start_len = prompt_len
        self.length = prompt_len
        self.cfg = cfg
        self.device = prompt_ids.device

        self.cache = KVCache(cfg)  # default device/dtype from the key on first write
        self._static_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self._static_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self._mask = torch.zeros(self.window, dtype=torch.bool, device=self.device)
        self._static_logits = torch.zeros(1, cfg.vocab_size, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            out = model(prompt_ids, past_key_values=self.cache)
        self._prefill_logits = out.logits[:, -1, :].float().clone()

        # capture-safe body; all state it mutates lives in persistent buffers
        def body():
            hidden = model.embed_tokens(self._static_ids)
            for layer in model.layers:
                attn = layer.attn
                h = layer.attention_norm(hidden)
                qkv = attn.qkv_proj(h)
                q, k, v = torch.split(
                    qkv,
                    [attn.num_heads * attn.head_dim,
                     attn.num_kv_heads * attn.head_dim,
                     attn.num_kv_heads * attn.head_dim],
                    dim=-1,
                )
                q = q.reshape(1, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
                k = k.reshape(1, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                v = v.reshape(1, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                cos, sin = attn.rope._slice(self._static_pos)
                q = attn.rope._apply_rotation(q, cos, sin)
                k = attn.rope._apply_rotation(k, cos, sin)
                k_buf = self.cache._keys[layer.layer_idx]
                v_buf = self.cache._values[layer.layer_idx]
                k_buf.index_copy_(2, self._static_pos, k)
                v_buf.index_copy_(2, self._static_pos, v)
                attn_out = F.scaled_dot_product_attention(
                    q, k_buf[:, :, : self.window], v_buf[:, :, : self.window],
                    attn_mask=self._mask, enable_gqa=True,
                )
                attn_out = attn_out.transpose(1, 2).reshape(1, 1, attn.num_heads * attn.head_dim)
                hidden = hidden + attn.o_proj(attn_out)
                h2 = layer.mlp_norm(hidden)
                gate, up = layer.mlp.gate_up_proj(h2).chunk(2, dim=-1)
                hidden = hidden + layer.mlp.down_proj(F.silu(gate) * up)
            hidden = model.final_norm(hidden)
            logits = model.output_projection(hidden, model.embed_tokens.embedding_weight)
            self._static_logits.copy_(logits[:, -1, :])

        self._body = body

        # warm up on a side stream, then capture on the real cache buffers
        scratch = torch.cuda.Stream()
        scratch.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(scratch):
            with torch.inference_mode():
                for _ in range(3):
                    self._body()
        torch.cuda.current_stream().wait_stream(scratch)

        # rebuild cache to a clean prefill state (warmup wrote garbage at the tail)
        self.cache.reset()
        with torch.inference_mode():
            model(prompt_ids, past_key_values=self.cache)
        self.length = prompt_len

        with torch.inference_mode():
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._body()

    # ------------------------------------------------------------------
    @property
    def prefill_logits(self) -> torch.Tensor:
        return self._prefill_logits

    def step(self, token_id: int) -> torch.Tensor:
        """Decode one token; returns its fp32 logits [vocab_size]."""
        self._static_ids[0, 0] = token_id
        self._static_pos.fill_(self.length)
        self._mask.zero_()
        self._mask[: self.length + 1] = True
        with torch.inference_mode():
            self._graph.replay()
        self.length += 1
        return self._static_logits

    def close(self) -> None:
        # break references so the cache buffers can be freed; graphs are not reusable across caches
        self._graph = None
        self.cache = None
