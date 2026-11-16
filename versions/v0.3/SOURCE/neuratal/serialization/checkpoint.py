"""Checkpoint serialization — safetensors + JSON config.

Layout of a checkpoint directory::

    ckpt_dir/
      config.json        # {format_version, architectures, config: NeuratalConfig dict}
      model.safetensors  # weights only

Guarantees
----------
* The tied embedding matrix is serialized exactly once: the tied head owns no
  parameter, and we additionally verify at save time that no two tensors in the
  state dict share storage (defense in depth — save fails loudly otherwise).
* Loading validates: config JSON structure, ``format_version``, architecture
  name, unknown config keys, missing keys, unexpected keys, and per-tensor
  shapes — each with a precise error. No partial/silent loads.
* Weight tying is restored by construction on load: the head references the
  embedding matrix at call time, so after ``load_state_dict`` there is exactly
  one copy, shared.
* Writes are atomic (temp file + ``os.replace``) so interrupted saves do not
  leave torn checkpoints behind.
* No pickle. ``config.json`` is data, not code.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file as _st_load_file
from safetensors.torch import save_file as _st_save_file

from neuratal.config import NeuratalConfig
from neuratal.model.model import NeuratalModel

FORMAT_VERSION = 1
ARCHITECTURE_NAME = "NeuratalModel"
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"


def save_pretrained(model: NeuratalModel, output_dir: Union[str, Path]) -> Path:
    if not isinstance(model, NeuratalModel):
        raise TypeError(f"model must be a NeuratalModel, got {type(model).__name__}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    _check_no_shared_storage(state_dict)

    # The config must describe the tensors actually written. If the caller cast the
    # model after construction, record the real dtype (with a warning — never silently).
    dtypes = {str(p.dtype).removeprefix("torch.") for p in model.parameters()}
    if len(dtypes) != 1:
        raise RuntimeError(f"mixed parameter dtypes are not supported for checkpointing: {sorted(dtypes)}")
    actual_dtype = dtypes.pop()
    config_dict = model.config.to_dict()
    if config_dict.get("dtype") != actual_dtype:
        warnings.warn(
            f"model parameters are {actual_dtype} but config.dtype is {config_dict.get('dtype')!r}; "
            f"recording dtype={actual_dtype!r} in the checkpoint config.",
            UserWarning,
            stacklevel=2,
        )
        config_dict["dtype"] = actual_dtype

    config_payload = {
        "format_version": FORMAT_VERSION,
        "architectures": [ARCHITECTURE_NAME],
        "config": config_dict,
    }
    _atomic_write_text(output_dir / CONFIG_FILENAME, json.dumps(config_payload, indent=2, sort_keys=True))
    _atomic_save_safetensors(state_dict, output_dir / WEIGHTS_FILENAME)
    return output_dir


def load_pretrained(
    checkpoint_dir: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
) -> NeuratalModel:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")
    config_path = checkpoint_dir / CONFIG_FILENAME
    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    if not config_path.is_file():
        raise ValueError(f"missing {CONFIG_FILENAME} in {checkpoint_dir}")
    if not weights_path.is_file():
        raise ValueError(f"missing {WEIGHTS_FILENAME} in {checkpoint_dir}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    _check_config_payload(payload)

    device = str(map_location) if map_location is not None else "cpu"
    try:
        state_dict = _st_load_file(str(weights_path), device=device)
    except Exception as exc:  # safetensors raises several exception types
        raise ValueError(f"failed to read safetensors file {weights_path}: {exc}") from exc

    config = NeuratalConfig.from_dict(payload["config"])
    model = NeuratalModel(config)

    state_dict = _migrate_legacy_state_dict(state_dict)

    model_state = model.state_dict()
    missing = sorted(k for k in model_state if k not in state_dict)
    unexpected = sorted(k for k in state_dict if k not in model_state)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint weights do not match the model: missing keys={missing}, "
            f"unexpected keys={unexpected}"
        )
    mismatched = sorted(k for k in model_state if model_state[k].shape != state_dict[k].shape)
    if mismatched:
        details = {k: (tuple(model_state[k].shape), tuple(state_dict[k].shape)) for k in mismatched}
        raise ValueError(f"checkpoint tensor shape mismatch (model vs file): {details}")
    dtype_mismatch = sorted(k for k in model_state if model_state[k].dtype != state_dict[k].dtype)
    if dtype_mismatch:
        details = {k: (str(model_state[k].dtype), str(state_dict[k].dtype)) for k in dtype_mismatch}
        raise ValueError(
            f"checkpoint tensor dtype mismatch (model vs file): {details}. "
            "Edit config.json's dtype to match the checkpoint rather than casting silently."
        )

    model.load_state_dict(state_dict, strict=True)
    return model


# ---------------------------------------------------------------------
_LEGACY_PROJECTION_GROUPS = (
    (r"(layers\.\d+)\.attn\.(q_proj|k_proj|v_proj)\.weight", ("q_proj", "k_proj", "v_proj"), "qkv_proj"),
    (r"(layers\.\d+)\.mlp\.(gate_proj|up_proj)\.weight", ("gate_proj", "up_proj"), "gate_up_proj"),
)


def _migrate_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Coalesce pre-v0.3 split projection weights into fused tensors.

    v0.3 fuses Q/K/V into ``qkv_proj`` and gate/up into ``gate_up_proj``
    (mathematically identical — one GEMM instead of several). Checkpoints
    written by v0.2 carry the split keys; they are concatenated deterministically
    (weights are row concatenations) and a warning is emitted. New checkpoints
    contain only the fused keys and pass through unchanged.
    """
    import re

    migrated = dict(state_dict)
    changed = False
    for pattern, names, fused in _LEGACY_PROJECTION_GROUPS:
        groups: dict[str, dict[str, str]] = {}
        for key in list(migrated):
            m = re.fullmatch(pattern, key)
            if m:
                groups.setdefault(m.group(1), {})[m.group(2)] = key
        for prefix, parts in groups.items():
            if set(parts) != set(names):
                continue  # partial legacy group: leave untouched; strict checks will report
            ordered = [parts[n] for n in names]
            stacked = torch.cat([migrated.pop(k) for k in ordered], dim=0)
            fused_key = f"{prefix}.{'attn' if fused == 'qkv_proj' else 'mlp'}.{fused}.weight"
            migrated[fused_key] = stacked
            changed = True
    if changed:
        warnings.warn(
            "migrated legacy split projection weights into fused qkv_proj/gate_up_proj "
            "(mathematically identical row concatenation)",
            UserWarning,
            stacklevel=2,
        )
    return migrated


def _check_config_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{CONFIG_FILENAME} must contain a JSON object, got {type(payload).__name__}")
    required = {"format_version": int, "architectures": list, "config": dict}
    for key, expected in required.items():
        if key not in payload:
            raise ValueError(f"{CONFIG_FILENAME} is missing required key {key!r}")
        if not isinstance(payload[key], expected):
            raise ValueError(f"{CONFIG_FILENAME} key {key!r} must be of type {expected.__name__}")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"unsupported format_version {payload['format_version']} "
            f"(this build supports {FORMAT_VERSION})"
        )
    archs = payload["architectures"]
    if not archs or archs[0] != ARCHITECTURE_NAME:
        raise ValueError(f"unsupported architecture(s) {archs!r}; expected {ARCHITECTURE_NAME!r}")


def _check_no_shared_storage(state_dict: dict[str, torch.Tensor]) -> None:
    # Storage-level (not pointer+size) identity: two distinct tensors must never
    # alias the same storage, even through different offsets/views.
    seen: dict[int, str] = {}
    for name, tensor in state_dict.items():
        if tensor.numel() == 0:
            continue
        storage_ptr = tensor.untyped_storage().data_ptr()
        other = seen.get(storage_ptr)
        if other is not None:
            raise RuntimeError(
                f"state dict tensors {other!r} and {name!r} share the same storage; "
                "refusing to write a checkpoint with aliased tensors"
            )
        seen[storage_ptr] = name


def _atomic_write_text(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_save_safetensors(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)  # safetensors wants a filename, not a fd
    try:
        _st_save_file(state_dict, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
