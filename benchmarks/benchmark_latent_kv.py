"""MLA vs GQA A/B -> results/latent_kv_results_<ts>.json

Same harness as other A/B benchmarks: fixed seed, identical shapes,
inference_mode, sync per iteration, median. Measures forward latency,
decode latency at context, tokens/s, exact KV/state bytes, peak memory.
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
from neuratal.model.latent_kv import LatentKVCache

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

    # exact byte accounting (fp32)
    cfg = NeuratalConfig()
    dsize = torch.empty(0, dtype=dtype).element_size()
    dense_kv_per_tok = 2 * cfg.num_key_value_heads * cfg.head_dim * dsize   # K+V per token per layer
    mla_kv_per_tok = cfg.mla_latent_dim * dsize                             # one latent per token per layer

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "dtype": args.dtype,
        "kv_bytes_per_token_per_layer_dense": dense_kv_per_tok,
        "kv_bytes_per_token_per_layer_mla": mla_kv_per_tok,
        "cells": [],
    }

    for atype in ("gqa", "mla"):
        torch.manual_seed(SEED)
        model = NeuratalModel(NeuratalConfig(attention_type=atype)).to(device, dtype).eval()
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
                "kv_bytes_per_token_per_layer": dense_kv_per_tok if atype == "gqa" else mla_kv_per_tok,
                "kv_bytes_total_at_capacity_2048": (dense_kv_per_tok if atype == "gqa" else mla_kv_per_tok)
                                                    * cfg.num_hidden_layers * 2048,
                "parameter_count": model.num_parameters(),
            }
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    model(ids)
                torch.cuda.synchronize()
                row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            results["cells"].append(row)
            print(f"{atype:4s} T={T:5d}: {med*1e3:7.2f} ms  {T/med:8.0f} tok/s  "
                  f"{row.get('peak_allocated_bytes', 0)/2**20:7.1f} MiB  KV/tok/layer={row['kv_bytes_per_token_per_layer']}B")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    path = save_result("latent_kv", results)
    print(f"latent KV results -> {path}")


if __name__ == "__main__":
    main()
