# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Environment fingerprint -> benchmarks/results/environment_<ts>.json

Collects OS/CPU/RAM/GPU/driver/PyTorch/Python/NTokenizer/NTransformer details,
dtype support, and SDPA backend availability. Run before any other benchmark.
Never estimates; values that cannot be determined are recorded as null with a note.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import torch
from benchmark_utils import machine_fingerprint, run_id, save_result

import neuratal
from neuratal.integration import build_manifest

NTOKENIZER_ROOT = Path(os.environ.get("NTOKENIZER_ROOT", Path(__file__).resolve().parents[2] / "NTokenizer"))
NTOK = NTOKENIZER_ROOT / "output" / "v0.3" / "ntokenizer-0.3-256k.ntok"

_BINDINGS_DIR = Path(os.environ.get("NTOKENIZER_BINDINGS", NTOKENIZER_ROOT / "python"))
if _BINDINGS_DIR.is_dir() and str(_BINDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(_BINDINGS_DIR))


def nvidia_smi_field(query: str):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def dtype_support() -> dict:
    report = {}
    for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        entry = {"cpu_matmul": None, "cuda_matmul": None}
        try:
            a = torch.randn(64, 64, dtype=dtype)
            _ = (a @ a).sum().item()
            entry["cpu_matmul"] = True
        except Exception as exc:
            entry["cpu_matmul"] = f"unsupported: {exc}"
        if torch.cuda.is_available():
            try:
                a = torch.randn(64, 64, dtype=dtype, device="cuda")
                _ = (a @ a).sum().item()
                entry["cuda_matmul"] = True
            except Exception as exc:
                entry["cuda_matmul"] = f"unsupported: {exc}"
        report[name] = entry
    return report


def sdpa_backends() -> dict:
    from torch.nn.attention import SDPBackend

    report = {}
    if not torch.cuda.is_available():
        return {"note": "CUDA not available; backends not probed on GPU"}
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION),
                          ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("cudnn", SDPBackend.CUDNN_ATTENTION),
                          ("math", SDPBackend.MATH)):
        for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16), ("float16", torch.float16)):
            key = f"{name}[{dtype_name}]"
            try:
                from torch.nn.attention import sdpa_kernel
                q = torch.randn(1, 4, 32, 32, device="cuda", dtype=dtype)
                with sdpa_kernel(backend):
                    _ = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
                report[key] = "ok"
            except Exception as exc:
                report[key] = f"unavailable: {str(exc)[:100]}"
    return report


def main() -> None:
    manifest = build_manifest(NTOK) if NTOK.is_file() else {"error": f"artifact missing: {NTOK}"}
    try:
        import neuratal_tokenizer
        nt_version = neuratal_tokenizer.native_version()
    except Exception as exc:
        nt_version = f"unavailable: {exc}"
    import safetensors

    payload = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "nvidia_smi": {
            "driver_version": nvidia_smi_field("driver_version"),
            "cuda_version": nvidia_smi_field("cuda_version"),
            "gpu_name": nvidia_smi_field("name"),
            "memory_total_mib": nvidia_smi_field("memory.total"),
            "compute_cap": nvidia_smi_field("compute_cap"),
        },
        "python": sys.version,
        "safetensors_version": safetensors.__version__,
        "neuratal_transformer_version": neuratal.__version__,
        "ntokenizer_native_version": nt_version,
        "ntokenizer_manifest": manifest,
        "dtype_support": dtype_support(),
        "sdpa_backends": sdpa_backends(),
    }
    path = save_result("environment", payload)
    print(f"environment -> {path}")
    print(json.dumps({k: v for k, v in payload.items() if k != "ntokenizer_manifest"}, indent=2, sort_keys=True))
    print("tokenizer manifest:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
