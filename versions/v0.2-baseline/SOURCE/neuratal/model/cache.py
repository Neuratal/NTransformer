"""KV cache — preallocated, per-layer key/value storage for incremental decoding.

Behavior contract
-----------------
* Buffers are preallocated to ``cache_capacity`` (defaults to
  ``max_position_embeddings``) on first use — no per-step reallocations.
* ``update(layer_idx, key, value)`` writes the new K/V at the current cache
  offset and returns the full ``[:, :, :length]`` view for that layer.
* Advancement is explicit: the model calls :meth:`commit` once per forward
  pass with the number of query tokens. Between :meth:`update` calls within one
  forward pass, the offset is constant, so every layer reads/writes the same
  logical position window.
* Overflow is a hard error (:class:`CacheOverflowError`), never truncation.
* :meth:`reset` reuses the buffers and rewinds the length to zero; the same
  cache object can safely serve a new, unrelated sequence afterwards. The
  buffers never leak content across sequences: positions beyond the current
  length are outside every view the model receives, and stale values are
  always overwritten before becoming visible.
* Not thread-safe. Not serialized (it is an inference-time artifact).
"""

from __future__ import annotations

import torch

from neuratal.config import NeuratalConfig


class CacheOverflowError(RuntimeError):
    """Raised when a prefill/decode step would exceed the cache capacity."""


class KVCache:
    def __init__(self, config: NeuratalConfig):
        self.capacity = int(config.cache_capacity or config.max_position_embeddings)
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.dtype = None  # resolved on initialize()
        self.device = None
        self.batch_size = None
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._length = 0
        self._pending = 0  # tokens written by update() awaiting commit()

    # ------------------------------------------------------------------
    @property
    def is_initialized(self) -> bool:
        return bool(self._keys)

    def get_seq_length(self) -> int:
        return self._length

    def initialize(self, batch_size: int, device: torch.device | str, dtype: torch.dtype) -> None:
        if self.is_initialized:
            raise RuntimeError("KVCache is already initialized; call reset() instead of re-initializing")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        shape = (batch_size, self.num_kv_heads, self.capacity, self.head_dim)
        self._keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.num_layers)]
        self._values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.num_layers)]
        self.batch_size, self.device, self.dtype = batch_size, torch.device(device), dtype

    # ------------------------------------------------------------------
    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        if key.shape != value.shape:
            raise ValueError(f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}")
        if key.dim() != 4:
            raise ValueError(f"key/value must be 4-D [batch, kv_heads, seq, head_dim], got {key.dim()}-D")
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError(f"layer_idx out of range: {layer_idx} (num_layers={self.num_layers})")
        if not self.is_initialized:
            self.initialize(key.size(0), key.device, key.dtype)
        elif key.size(0) != self.batch_size:
            raise RuntimeError(
                f"batch size changed from {self.batch_size} to {key.size(0)}; "
                "create a new KVCache (or reset and re-initialize) for a different batch"
            )
        elif key.device != self.device or key.dtype != self.dtype:
            raise RuntimeError(
                f"cache device/dtype mismatch: cache is ({self.device}, {self.dtype}), "
                f"got ({key.device}, {key.dtype})"
            )
        expected_heads = self.num_kv_heads
        if key.size(1) != expected_heads or key.size(3) != self.head_dim:
            raise ValueError(
                f"key shape must be [batch={self.batch_size}, kv_heads={expected_heads}, "
                f"seq, head_dim={self.head_dim}], got {tuple(key.shape)}"
            )

        new_tokens = key.size(2)
        start = self._length
        end = start + new_tokens
        if end > self.capacity:
            raise CacheOverflowError(
                f"KV cache overflow at layer {layer_idx}: capacity={self.capacity}, "
                f"current length={self._length}, incoming tokens={new_tokens}"
            )
        self._keys[layer_idx][:, :, start:end] = key
        self._values[layer_idx][:, :, start:end] = value
        self._pending = new_tokens  # constant across layers within one forward pass
        return self._keys[layer_idx][:, :, :end], self._values[layer_idx][:, :, :end]

    def commit(self, num_tokens: int) -> None:
        """Advance the cache by ``num_tokens``. Must be called exactly once per forward pass."""
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if self._pending == 0:
            raise RuntimeError("commit() without a preceding update() in this forward pass")
        if num_tokens != self._pending:
            raise RuntimeError(
                f"commit({num_tokens}) does not match the number of tokens written by update() "
                f"({self._pending})"
            )
        self._length += num_tokens
        self._pending = 0

    def reset(self) -> None:
        """Rewind length to zero, keeping the allocated buffers."""
        self._length = 0
        self._pending = 0

    def __repr__(self) -> str:
        return (
            f"KVCache(capacity={self.capacity}, length={self._length}, "
            f"layers={self.num_layers}, kv_heads={self.num_kv_heads}, initialized={self.is_initialized})"
        )
