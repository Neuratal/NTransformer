# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
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

import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file as _st_load_file

from neuratal.config import NeuratalConfig
from neuratal.model.model import NeuratalModel
from neuratal.serialization.checkpoint import (
    ARCHITECTURE_NAME,
    CONFIG_FILENAME,
    FORMAT_VERSION,
    _atomic_save_safetensors,
    _atomic_write_text,
    _check_config_payload,
    _check_no_shared_storage,
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


_INDEX_FILENAME_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")


def _validated_shard_path(checkpoint_dir: Path, filename: object) -> Path:
    """Resolve a weight_map shard name to a contained, regular file.

    ``weight_map`` values are attacker-controlled when a checkpoint is loaded
    from an untrusted source: reject anything that is not a plain
    ``model-NNNNN-of-NNNNN.safetensors`` filename, any absolute path, any
    traversal/backslash trick, and any resolved path escaping the checkpoint
    directory or landing on a symlink outside it.
    """
    if not isinstance(filename, str):
        raise ValueError(f"shard filename must be a string, got {type(filename).__name__}")
    name = filename.strip()
    m = _INDEX_FILENAME_RE.fullmatch(name)
    if m is None:
        raise ValueError(
            f"shard filename {filename!r} is not a valid model-NNNNN-of-NNNNN.safetensors name; "
            "absolute paths, traversal segments and non-standard names are refused"
        )
    resolved = (checkpoint_dir / name).resolve()
    base = checkpoint_dir.resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(
            f"shard {filename!r} resolves outside the checkpoint directory ({base}); refused"
        ) from None
    if not resolved.is_file():
        raise ValueError(
            f"missing shard {filename!r} referenced by {INDEX_FILENAME} "
            f"(or not a regular file in {base})"
        )
    return resolved


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
    _check_no_shared_storage(state_dict)

    pre_existing = {p.name for p in output_dir.iterdir()} if output_dir.is_dir() else set()
    written: list[Path] = []
    try:
        shards = plan_shards(state_dict, max_shard_bytes=max_shard_bytes)
        count = len(shards)
        weight_map: dict[str, str] = {}
        shard_hashes: dict[str, str] = {}
        for idx, shard in enumerate(shards, start=1):
            filename = f"model-{idx:05d}-of-{count:05d}.safetensors"
            for name in shard:
                weight_map[name] = filename
            path = output_dir / filename
            _atomic_save_safetensors(shard, path)
            written.append(path)
            shard_hashes[filename] = _sha256(path)

        total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        index = {
            "metadata": {
                "format_version": SHARDED_FORMAT_VERSION,
                "architectures": [ARCHITECTURE_NAME],
                "total_size": total_size,
                "num_shards": count,
                "shard_hashes": shard_hashes,
            },
            "weight_map": weight_map,
        }
        _atomic_write_text(output_dir / INDEX_FILENAME, json.dumps(index, indent=2, sort_keys=True))
        config_payload = build_config_payload(model)
        _atomic_write_text(output_dir / CONFIG_FILENAME, json.dumps(config_payload, indent=2, sort_keys=True))
    except BaseException:
        # An interrupted save must not leave orphan shards behind. Only files
        # created by THIS call are removed; pre-existing shards are never
        # touched (they may come from a previous, valid checkpoint).
        for path in written:
            if path.name not in pre_existing:
                try:
                    path.unlink()
                except OSError:
                    pass
        raise
    return output_dir


def load_pretrained_sharded(
    checkpoint_dir: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    *,
    verify_shard_hashes: bool = True,
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
    shard_files = [
        _validated_shard_path(checkpoint_dir, name) for name in sorted(set(weight_map.values()))
    ]
    if not shard_files:
        raise ValueError(f"{INDEX_FILENAME} has an empty weight_map")
    shard_hashes = metadata.get("shard_hashes")
    if shard_hashes is not None and not isinstance(shard_hashes, dict):
        raise ValueError(f"{INDEX_FILENAME} metadata.shard_hashes must be an object")
    recorded_hashes = {str(k): str(v) for k, v in (shard_hashes or {}).items()}
    if verify_shard_hashes:
        if not recorded_hashes:
            raise ValueError(
                f"{INDEX_FILENAME} does not record shard_hashes; refusing to load an "
                f"unverified sharded checkpoint (the format written by this version always "
                f"records per-shard SHA-256). Pass verify_shard_hashes=False to load "
                f"unverified at your own risk, or re-save the checkpoint."
            )
        for shard_path in shard_files:
            filename = shard_path.name
            want = recorded_hashes.get(filename)
            if want is None:
                raise ValueError(f"{INDEX_FILENAME} shard_hashes is missing an entry for {filename!r}")
            got = _sha256(shard_path)
            if got != want:
                raise ValueError(
                    f"shard {filename} failed SHA-256 verification: metadata records "
                    f"{want}, file hashes to {got} (the shard was modified or replaced)"
                )
    elif not recorded_hashes:
        warnings.warn(
            f"{INDEX_FILENAME} does not record shard_hashes; loading shards WITHOUT "
            "any integrity verification (verify_shard_hashes=False).",
            UserWarning,
            stacklevel=2,
        )
    state_dict: dict[str, torch.Tensor] = {}
    for shard_path in shard_files:
        filename = shard_path.name
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

    # Integrity check: bytes on disk == bytes recorded at save time. A stale
    # or partially truncated shard file that still parses as safetensors
    # would otherwise load silently. A missing/non-integer total_size is a
    # malformed index and is rejected (fail closed).
    total_size = metadata.get("total_size")
    if not isinstance(total_size, bool) and isinstance(total_size, int):
        actual_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        if actual_size != total_size:
            raise ValueError(
                f"sharded checkpoint byte count mismatch: metadata records {total_size}, "
                f"loaded tensors hold {actual_size} (a shard was replaced with stale or "
                "different content)"
            )
    else:
        if verify_shard_hashes:
            raise ValueError(
                f"{INDEX_FILENAME} metadata.total_size is missing or not an integer; "
                "refusing to load a malformed sharded index. Pass verify_shard_hashes=False "
                "to load unverified at your own risk."
            )
        warnings.warn(
            f"{INDEX_FILENAME} metadata.total_size is missing or not an integer; "
            "skipping the byte-count check (verify_shard_hashes=False).",
            UserWarning,
            stacklevel=2,
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


def _sha256(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
