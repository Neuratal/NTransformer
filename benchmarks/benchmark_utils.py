# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Shared helpers for the Neuratal benchmark suite.

Reproducibility contract
------------------------
* Fixed seed (42) for all randomized tensors.
* Every result JSON embeds: timestamp, machine fingerprint, seeds, config, units.
* Results are written to benchmarks/results/<name>_<run_id>.json and a
  <name>_latest.json copy; previous runs are never overwritten.
* Measurements come from actual executions; theoretical values are labeled
  "theoretical" and never mixed with measured timings.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# canonical result filenames (in addition to timestamped archives + *_latest.json)
_CANONICAL = {
    "environment": "environment.json",
    "tokenizer": "tokenizer_results.json",
    "model": "model_results.json",
    "attention": "attention_results.json",
    "kv_cache": "kv_cache_results.json",
    "end_to_end": "end_to_end_results.json",
    "memory": "memory_results.json",
    "moe": "moe_results.json",
    "chunked_ce": "chunked_ce_results.json",
}


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("empty sample set")
    ordered = sorted(samples)
    n = len(ordered)
    p95 = ordered[min(n - 1, int(0.95 * (n - 1) + 0.999999))]
    return {
        "n": n,
        "min": min(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
        "max": max(ordered),
        "stdev": (statistics.pstdev(ordered) if n > 1 else 0.0),
    }


def timed(fn: Callable[[], Any], *, warmup: int, repeats: int) -> dict[str, Any]:
    """Time fn() with warmup; returns stats + raw samples (seconds)."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {"stats": stats(samples), "samples": samples}


def sync_device(device) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def machine_fingerprint() -> dict[str, Any]:
    fp: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _read_cpu_model(),
        "cpu_cores": os.cpu_count(),
        "ram_total_bytes": _read_ram_total(),
        "seed": SEED,
    }
    try:
        import torch

        fp["torch"] = torch.__version__
        fp["cuda_runtime"] = torch.version.cuda
        fp["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            fp["gpu"] = props.name
            fp["gpu_total_vram_bytes"] = props.total_memory
            fp["gpu_compute_capability"] = list(torch.cuda.get_device_capability(0))
            fp["gpu_sm_count"] = props.multi_processor_count
            free, _ = torch.cuda.mem_get_info(0)
            fp["gpu_free_vram_bytes"] = free
    except Exception as exc:  # fingerprinting must never break a run
        fp["torch_fingerprint_error"] = repr(exc)
    return fp


def _read_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _read_ram_total() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def save_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = RESULTS_DIR / f"{name}_{run_id()}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    (RESULTS_DIR / f"{name}_latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    canonical = _CANONICAL.get(name)
    if canonical:
        (RESULTS_DIR / canonical).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def estimated_tensor_bytes(rows: int, cols: int, dtype_bytes: int) -> int:
    return rows * cols * dtype_bytes


def dtype_nbytes(dtype) -> int:
    import torch

    return torch.empty(0, dtype=dtype).element_size()


def vram_free_bytes(device="cuda:0") -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info(device)[0]
    except Exception:
        return None
