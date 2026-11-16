"""Full training-state checkpointing.

Format:
    <checkpoint_dir>/
        model.safetensors        (existing safetensors path — weights + config.json)
        config.json              (model config + architectures + format_version)
        trainer_state.safetensors (optimizer state, scheduler state, RNG, counters —
                                    flat tensor dict, safetensors, atomic rename)
        trainer_meta.json        (non-tensor metadata: global step, epoch, tokens,
                                    tokenizer fingerprint, scheduler name, precision)

All writes atomic (tmp file + os.replace). Corrupt files rejected with precise
errors. Resume restores model weights AND optimizer/scheduler/RNG/step/counters.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import torch
from safetensors.torch import load_file as _st_load, save_file as _st_save

from neuratal.config import NeuratalConfig
from neuratal.serialization.checkpoint import load_pretrained, save_pretrained

TRAINER_STATE_FILE = "trainer_state.safetensors"
TRAINER_META_FILE = "trainer_meta.json"
CHECKPOINT_FORMAT_VERSION = 1


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    output_dir: Union[str, Path],
    *,
    global_step: int,
    epoch: int = 0,
    tokens_seen: int = 0,
    tokenizer_fingerprint: Optional[str] = None,
    scheduler_name: str = "warmup_cosine",
    precision: str = "bfloat16",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, "module") else model  # DDP unwrap
    save_pretrained(core, output_dir)  # writes config.json + model.safetensors (atomic)

    # --- trainer state as a flat tensor dict ---
    state: dict[str, torch.Tensor] = {}
    for i, group in enumerate(optimizer.param_groups):
        for j, p in enumerate(group["params"]):
            st = optimizer.state.get(p)
            if st is None:
                continue
            for k, v in st.items():
                if torch.is_tensor(v):
                    state[f"optimizer.{i}.{j}.{k}"] = v.detach().cpu().contiguous()
                else:
                    state[f"optimizer.{i}.{j}.{k}"] = torch.tensor(v, dtype=torch.float32)
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        for k, v in scheduler.state_dict().items():
            if isinstance(v, (list, tuple)):
                state[f"scheduler.{k}.__len__"] = torch.tensor([len(v)], dtype=torch.long)
                for j, x in enumerate(v):
                    if isinstance(x, (int, float)):
                        state[f"scheduler.{k}.{j}"] = torch.tensor(float(x))
            elif isinstance(v, (int, float)):
                state[f"scheduler.{k}"] = torch.tensor(float(v))
    state["trainer.global_step"] = torch.tensor([int(global_step)])
    state["trainer.epoch"] = torch.tensor([int(epoch)])
    state["trainer.tokens_seen"] = torch.tensor([int(tokens_seen)])
    rng = torch.get_rng_state()
    state["trainer.rng_cpu"] = rng.to(torch.uint8)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            state[f"trainer.rng_cuda_{i}"] = torch.cuda.get_rng_state(i).to(torch.uint8)

    _atomic_save_safetensors(state, output_dir / TRAINER_STATE_FILE)
    model_cfg = core.config
    meta = {
        "trainer_format_version": CHECKPOINT_FORMAT_VERSION,
        "scheduler": scheduler_name,
        "precision": precision,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "tokens_seen": int(tokens_seen),
        "architecture": "moe" if model_cfg.is_moe else "dense",
        "num_experts": int(model_cfg.num_experts),
        "top_k": int(model_cfg.num_experts_per_token) if model_cfg.is_moe else 0,
        "moe_layers": list(model_cfg.moe_layer_indices or []),
    }
    _atomic_write_text(output_dir / TRAINER_META_FILE, json.dumps(meta, indent=2, sort_keys=True))
    return output_dir


def load_training_checkpoint(checkpoint_dir: Union[str, Path], map_location: str = "cpu"):
    """Restore (model, trainer_meta) from a training checkpoint directory.

    Optimizer/scheduler state is loaded into the provided objects by
    ``resume_training_checkpoint`` (kept explicit so optimizer class need not
    be known to this module).
    """
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / TRAINER_META_FILE
    if not meta_path.is_file():
        raise ValueError(f"missing {TRAINER_META_FILE} in {checkpoint_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("trainer_format_version")
    if fmt != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported trainer format_version {fmt}")
    model = load_pretrained(checkpoint_dir, map_location=map_location)
    expected_arch = "moe" if model.config.is_moe else "dense"
    recorded_arch = meta.get("architecture")
    if recorded_arch is not None and recorded_arch != expected_arch:
        raise ValueError(
            f"training checkpoint architecture mismatch: meta says {recorded_arch!r}, "
            f"model config builds {expected_arch!r}"
        )
    return model, meta


def resume_training_checkpoint(optimizer, scheduler, checkpoint_dir: Union[str, Path],
                               map_location: str = "cpu", *, model=None) -> dict[str, Any]:
    """Restore optimizer/scheduler/RNG state from a checkpoint directory.

    Optimizer tensors are moved to each parameter's device, so a checkpoint
    written on CPU resumes correctly on CUDA (and vice versa).  When ``model``
    is supplied, its dense/MoE architecture is checked against the recorded
    checkpoint metadata before any state is installed.
    """
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / TRAINER_STATE_FILE
    if not state_path.is_file():
        raise ValueError(f"missing {TRAINER_STATE_FILE} in {checkpoint_dir}")
    tensors = _st_load(str(state_path), device="cpu")

    meta_path = checkpoint_dir / TRAINER_META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if model is not None:
        core = model.module if hasattr(model, "module") else model
        expected_arch = "moe" if core.config.is_moe else "dense"
        recorded_arch = meta.get("architecture")
        if recorded_arch is not None and recorded_arch != expected_arch:
            raise ValueError(
                f"training checkpoint architecture mismatch: meta says {recorded_arch!r}, "
                f"target model is {expected_arch!r}"
            )

    # --- optimizer ---
    for i, group in enumerate(optimizer.param_groups):
        for j, p in enumerate(group["params"]):
            st_keys = [k for k in tensors if k.startswith(f"optimizer.{i}.{j}.")]
            if not st_keys:
                continue
            st = {}
            for k in st_keys:
                leaf = k.rsplit(".", 1)[-1]
                st[leaf] = tensors[k].to(device=p.device).clone()
            optimizer.state[p] = st

    # --- scheduler ---
    if scheduler is not None and hasattr(scheduler, "load_state_dict"):
        sched_state = {}
        seen = set()
        for k in list(tensors):
            if not k.startswith("scheduler."):
                continue
            rest = k[len("scheduler."):]
            parts = rest.split(".")
            name = parts[0]
            if name in seen:
                continue
            len_key = f"scheduler.{name}.__len__"
            if len_key in tensors:
                length = int(tensors[len_key].item())
                sched_state[name] = [
                    float(tensors[f"scheduler.{name}.{j}"].item())
                    for j in range(length) if f"scheduler.{name}.{j}" in tensors
                ]
                seen.add(name)
            elif len(parts) == 1:
                sched_state[name] = float(tensors[k].item())
                seen.add(name)
        if sched_state:
            scheduler.load_state_dict(sched_state)

    # --- RNG ---
    if "trainer.rng_cpu" in tensors:
        torch.set_rng_state(tensors["trainer.rng_cpu"].cpu())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            key = f"trainer.rng_cuda_{i}"
            if key in tensors:
                torch.cuda.set_rng_state(tensors[key].cpu(), i)

    meta["global_step"] = int(tensors["trainer.global_step"].item())
    meta["epoch"] = int(tensors["trainer.epoch"].item())
    meta["tokens_seen"] = int(tensors["trainer.tokens_seen"].item())
    return meta


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


def _atomic_save_safetensors(state: dict[str, torch.Tensor], path: Path) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        _st_save(state, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
