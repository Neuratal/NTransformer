"""Local sharded safetensors checkpoints.

Why
---
A single ``model.safetensors`` is fine for the development preset and 75M/1B
models, but one-file writes become impractical for multi-hundred-billion
parameter MoE checkpoints: the writer needs one contiguous file, resuming a
partial upload is impossible, and filesystem/object-store limits start to bite.
This module implements a *local* sharded format (the same layout family used by
safetensors/Hugging Face indexes):

    ckpt_dir/
      config.json
      model.safetensors.index.json    # weight_map + metadata
      model-00001-of-00003.safetensors
      model-00002-of-00003.safetensors
      model-00003-of-00003.safetensors

What it is NOT
--------------
This is not a distributed checkpoint system. Every rank still owns a full
state dict; there is no cross-rank coordination, no resharding, and no
asynchronous upload. It makes local writing/loading modular and gives a clear
format to build distributed sharding on top of later.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file as _st_load_file
from safetensors.torch import save_file as _st_save_file

from neuratal.config import NeuratalConfig
from neuratal.model.model import NeuratalModel
from neuratal.serialization.checkpoint import (
    ARCHITECTURE_NAME,
    CONFIG_FILENAME,
    FORMAT_VERSION,
    _atomic_write_text,
    _check_config_payload,
    _has_moe_keys,
    build_config_payload,
)

INDEX_FILENAME = "model.safetensors.index.json"
SHARDED_FORMAT_VERSION = 1


def plan_shards(
    state_dict: dict[str, torch.Tensor],
    max_shard_bytes: int = 2 * 1024**3,
) -> list[dict[str, torch.Tensor]]:
    """Split a state dict into contiguous shards bounded by ``max_shard_bytes``.

    A single tensor larger than the bound gets its own shard (never split a
    tensor: that is a resharding/parallel-IO problem, deliberately out of scope).
    """
    if max_shard_bytes < 1:
        raise ValueError(f"max_shard_bytes must be >= 1, got {max_shard_bytes}")
    if not state_dict:
        raise ValueError("cannot shard an empty state dict")
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        nbytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + nbytes > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[name] = tensor
        current_bytes += nbytes
    if current:
        shards.append(current)
    return shards


def save_pretrained_sharded(
    model: NeuratalModel,
    output_dir: Union[str, Path],
    *,
    max_shard_bytes: int = 2 * 1024**3,
) -> Path:
    if not isinstance(model, NeuratalModel):
        raise TypeError(f"model must be a NeuratalModel, got {type(model).__name__}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    # Same anti-aliasing guarantee as the single-file writer (tied embedding).
    from neuratal.serialization.checkpoint import _check_no_shared_storage
    _check_no_shared_storage(state_dict)

    shards = plan_shards(state_dict, max_shard_bytes=max_shard_bytes)
    count = len(shards)
    weight_map: dict[str, str] = {}
    for idx, shard in enumerate(shards, start=1):
        filename = f"model-{idx:05d}-of-{count:05d}.safetensors"
        for name in shard:
            weight_map[name] = filename
        _atomic_save_safetensors(shard, output_dir / filename)

    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    index = {
        "metadata": {
            "format_version": SHARDED_FORMAT_VERSION,
            "architectures": [ARCHITECTURE_NAME],
            "total_size": total_size,
            "num_shards": count,
        },
        "weight_map": weight_map,
    }
    _atomic_write_text(output_dir / INDEX_FILENAME, json.dumps(index, indent=2, sort_keys=True))
    config_payload = build_config_payload(model)
    _atomic_write_text(output_dir / CONFIG_FILENAME, json.dumps(config_payload, indent=2, sort_keys=True))
    return output_dir


def load_pretrained_sharded(
    checkpoint_dir: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
) -> NeuratalModel:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")
    config_path = checkpoint_dir / CONFIG_FILENAME
    index_path = checkpoint_dir / INDEX_FILENAME
    if not config_path.is_file():
        raise ValueError(f"missing {CONFIG_FILENAME} in {checkpoint_dir}")
    if not index_path.is_file():
        raise ValueError(f"missing {INDEX_FILENAME} in {checkpoint_dir}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    _check_config_payload(payload)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError(f"{INDEX_FILENAME} must contain a JSON object")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise ValueError(f"{INDEX_FILENAME} requires 'metadata' and 'weight_map' objects")
    if metadata.get("format_version") != SHARDED_FORMAT_VERSION:
        raise ValueError(
            f"unsupported sharded format_version {metadata.get('format_version')!r} "
            f"(expected {SHARDED_FORMAT_VERSION})"
        )

    device = str(map_location) if map_location is not None else "cpu"
    shard_files = sorted(set(weight_map.values()))
    if not shard_files:
        raise ValueError(f"{INDEX_FILENAME} has an empty weight_map")
    state_dict: dict[str, torch.Tensor] = {}
    for filename in shard_files:
        shard_path = checkpoint_dir / filename
        if not shard_path.is_file():
            raise ValueError(f"missing shard {filename} referenced by {INDEX_FILENAME}")
        try:
            shard = _st_load_file(str(shard_path), device=device)
        except Exception as exc:
            raise ValueError(f"failed to read shard {shard_path}: {exc}") from exc
        duplicate = set(state_dict) & set(shard)
        if duplicate:
            raise ValueError(f"duplicate tensor keys across shards: {sorted(duplicate)}")
        state_dict.update(shard)

    if set(state_dict) != set(weight_map):
        missing = sorted(set(weight_map) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(weight_map))
        raise ValueError(
            f"shard contents do not match weight_map: missing={missing}, unexpected={unexpected}"
        )

    config = NeuratalConfig.from_dict(payload["config"])
    model = NeuratalModel(config)
    model_state = model.state_dict()
    missing = sorted(k for k in model_state if k not in state_dict)
    unexpected = sorted(k for k in state_dict if k not in model_state)
    if missing or unexpected:
        hint = ""
        if _has_moe_keys(state_dict) != _has_moe_keys(model_state):
            hint = " — dense/MoE architecture mismatch between config and shards"
        raise ValueError(
            f"sharded checkpoint weights do not match the model: missing={missing}, "
            f"unexpected={unexpected}{hint}"
        )
    mismatched = sorted(k for k in model_state if model_state[k].shape != state_dict[k].shape)
    if mismatched:
        details = {k: (tuple(model_state[k].shape), tuple(state_dict[k].shape)) for k in mismatched}
        raise ValueError(f"sharded checkpoint tensor shape mismatch: {details}")
    dtype_mismatch = sorted(k for k in model_state if model_state[k].dtype != state_dict[k].dtype)
    if dtype_mismatch:
        details = {k: (str(model_state[k].dtype), str(state_dict[k].dtype)) for k in dtype_mismatch}
        raise ValueError(
            f"sharded checkpoint tensor dtype mismatch: {details}. "
            "Edit config.json's dtype to match the shards rather than casting silently."
        )
    model.load_state_dict(state_dict, strict=True)
    return model


def _atomic_save_safetensors(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        _st_save_file(state_dict, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
