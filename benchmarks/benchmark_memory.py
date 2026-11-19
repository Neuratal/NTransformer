# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Memory & scalability analysis -> results/memory_results_<ts>.json

Hard rules: never OOM on purpose; every cell estimates cost first, checks free
VRAM, applies a configurable budget, and records skips with reasons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from benchmark_utils import machine_fingerprint, run_id, save_result, SEED, dtype_nbytes

from neuratal import NeuratalConfig, NeuratalModel

V = 262_144


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--budget-mib", type=int, default=4608,
                        help="skip any cell whose estimated footprint exceeds this much free VRAM")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(0)
        print(f"free VRAM: {free/2**20:.0f} MiB / {total/2**20:.0f} MiB (budget {args.budget_mib} MiB)")
    else:
        free = None

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "budget_mib": args.budget_mib,
        "theoretical": {},
        "measured": [],
        "skipped": [],
    }

    def try_cell(name, config: NeuratalConfig, seq: int, batch: int, dtype, fn):
        dtype_bytes = dtype_nbytes(dtype)
        param_bytes = _estimate_params(config) * dtype_bytes
        logits_bytes = batch * seq * V * dtype_bytes
        # conservative activation estimate: ~4x logits for backward-free forward + 1.5x logits slack
        estimated = param_bytes + logits_bytes * 5
        if free is not None and estimated > (args.budget_mib * 2**20):
            reason = (f"estimated {estimated/2**20:.0f} MiB > budget {args.budget_mib} MiB "
                      f"(params {param_bytes/2**20:.1f} + logits {logits_bytes/2**20:.1f} MiB)")
            results["skipped"].append({"cell": name, "seq": seq, "batch": batch, "dtype": str(dtype), "reason": reason})
            print(f"SKIP {name} B={batch} T={seq} {dtype}: {reason}")
            return
        try:
            torch.cuda.empty_cache() if device.type == "cuda" else None
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            row = fn(config, seq, batch, dtype)
            row.update({"cell": name, "seq": seq, "batch": batch, "dtype": str(dtype),
                        "estimated_bytes": estimated,
                        "logits_bytes_theoretical": logits_bytes,
                        "params_bytes_theoretical": param_bytes})
            results["measured"].append(row)
            print(f"OK   {name} B={batch} T={seq} {dtype}: peak alloc {row.get('peak_allocated_bytes', 0)/2**20:.1f} MiB")
        except torch.cuda.OutOfMemoryError as exc:
            results["skipped"].append({"cell": name, "seq": seq, "batch": batch, "dtype": str(dtype),
                                        "reason": f"OOM despite estimate: {exc}"})
            print(f"OOM-SKIP {name}: recorded, VRAM left untouched")

    # --- theoretical table (clearly labeled) ---
    fp32, bf16 = 4, 2
    results["theoretical"] = {
        "logits_bytes_per_cell_fp32": {f"B={b},T={t}": b * t * V * fp32 for b in (1, 2, 8) for t in (128, 512, 1024)},
        "logits_bytes_per_cell_bf16": {f"B={b},T={t}": b * t * V * bf16 for b in (1, 2, 8) for t in (128, 512, 1024)},
        "note": "logits tensor alone; excludes parameters, activations, gradients, optimizer state",
    }

    # --- dev preset measurements ---
    dev = NeuratalConfig()
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        for seq in (64, 512):
            try_cell("dev_preset_forward", dev, seq, 1, dtype, _measure_forward)

    # --- 50M reference (scalability only) ---
    ref = NeuratalConfig(vocab_size=262144, hidden_size=512, num_hidden_layers=11,
                         num_attention_heads=8, num_key_value_heads=2,
                         embedding_size=64, intermediate_size=1408,
                         max_position_embeddings=2048)
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        for seq in (64, 128):
            try_cell("ref_50m_forward", ref, seq, 1, dtype, _measure_forward)

    # --- parameter/KV-cache accounting (measured buffers) ---
    dev_model = NeuratalModel(dev)
    kv_bytes = 2 * dev.num_hidden_layers * dev.num_key_value_heads * dev.cache_capacity * dev.head_dim * 4
    results["dev_preset"] = {
        "parameter_count": dev_model.num_parameters(),
        "parameter_bytes_fp32": sum(p.numel() * p.element_size() for p in dev_model.parameters()),
        "embedding_matrix_bytes_fp32": V * dev.embedding_size * 4,
        "kv_cache_bytes_fp32_capacity_2048": kv_bytes,
        "logits_bytes_B1_T512_fp32": 1 * 512 * V * 4,
    }
    path = save_result("memory", results)
    print(f"memory results -> {path}")


def _estimate_params(config: NeuratalConfig) -> int:
    model = NeuratalModel(config)
    n = model.num_parameters()
    del model
    return n


def _measure_forward(config: NeuratalConfig, seq: int, batch: int, dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = NeuratalModel(config).to(device, dtype).eval()
    ids = torch.randint(0, V, (batch, seq), generator=torch.Generator().manual_seed(SEED)).to(device)
    with torch.inference_mode():
        _ = model(ids)  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        _ = model(ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
    row = {"parameter_count": model.num_parameters()}
    if device.type == "cuda":
        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    else:
        row["peak_allocated_bytes"] = None
        row["note"] = "CPU device: VRAM accounting not applicable"
    del model
    return row


if __name__ == "__main__":
    main()
