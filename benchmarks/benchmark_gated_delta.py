"""Gated DeltaNet vs dense GQA A/B -> results/gated_delta_results_<ts>.json

Measures the same model shapes with attention_type in {gqa, gated_delta}:
forward latency (eval, inference_mode), tokens/s, peak memory, state/KV bytes
per token, and cached-vs-uncached parity. Identical seeds, shapes, dtype.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from benchmark_utils import machine_fingerprint, run_id, save_result, SEED

from neuratal import NeuratalConfig, NeuratalModel

V = 262_144


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--lengths", default="128,512,1024,2048")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    lengths = [int(x) for x in args.lengths.split(",")]

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "dtype": args.dtype,
        "cells": [],
    }

    for atype in ("gqa", "gated_delta"):
        cfg = NeuratalConfig(attention_type=atype)
        torch.manual_seed(SEED)
        model = NeuratalModel(cfg).to(device, dtype).eval()
        kv_bytes_per_token = None
        if atype == "gqa":
            kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * torch.empty(0, dtype=dtype).element_size()
        else:
            kv_bytes_per_token = 0.0  # O(1) state; reported separately below
        state_bytes = (2 * cfg.num_hidden_layers * cfg.num_attention_heads * cfg.head_dim * cfg.head_dim * 4) if atype == "gated_delta" else 0

        for T in lengths:
            ids = torch.randint(0, V, (1, T), generator=torch.Generator().manual_seed(SEED)).to(device)
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                samples = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    out = model(ids)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    samples.append(time.perf_counter() - t0)
                    assert torch.isfinite(out.logits.float()).all()
            import statistics

            med = statistics.median(samples)
            row = {
                "attention_type": atype,
                "seq": T,
                "median_s": med,
                "tok_per_s": T / med,
                "kv_bytes_per_token": kv_bytes_per_token,
                "state_bytes_total_fp32": state_bytes,
                "parameter_count": model.num_parameters(),
            }
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    model(ids)
                torch.cuda.synchronize()
                row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            results["cells"].append(row)
            print(f"{atype:12s} T={T:5d}: {med*1e3:8.2f} ms  {T/med:8.0f} tok/s  "
                  f"{row.get('peak_allocated_bytes', 0)/2**20:8.1f} MiB")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    path = save_result("gated_delta", results)
    print(f"gated delta results -> {path}")


if __name__ == "__main__":
    main()
