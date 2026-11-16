"""v0.6-candidate benchmark: GQA vs sequential GDN vs WY-chunkwise GDN.

Identical protocol for all three attention types (seeds 42/43/44):
forward latency (inference_mode, warmup, median+p95), decode latency per token
at context lengths 128/512/1024/2048, raw per-seed JSON saved.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from benchmark_utils import machine_fingerprint, run_id, save_result, SEED

from neuratal import NeuratalConfig, NeuratalModel
from neuratal.model.cache import KVCache
from neuratal.model.gated_delta_net import DeltaNetCache

V = 262_144
SEEDS = (42, 43, 44)


def bench_prefill(model, ids, warmup=5, repeats=20):
    with torch.inference_mode():
        for _ in range(warmup):
            model(ids)
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(ids)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
    return statistics.median(samples), sorted(samples)[int(0.95 * len(samples)) - 1]


def bench_decode(model, cfg, plen, steps=20):
    device = next(model.parameters()).device
    gdn = cfg.attention_type == "gated_delta"
    cache_factory = (lambda: DeltaNetCache(model.config)) if gdn else (lambda: KVCache(model.config))
    ids = torch.randint(0, V, (1, plen), generator=torch.Generator().manual_seed(SEED)).to(device)
    cur = torch.randint(0, V, (1, 1), generator=torch.Generator().manual_seed(SEED + 1)).to(device)
    with torch.inference_mode():
        cache = cache_factory()
        model(ids, past_key_values=cache)
        samples = []
        for _ in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(cur, past_key_values=cache)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--lengths", default="128,512,1024,2048")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    lengths = [int(x) for x in args.lengths.split(",")]
    configs = [("gqa", {"attention_type": "gqa"}),
               ("gdn_sequential", {"attention_type": "gated_delta", "gdn_chunk_size": 0}),
               ("gdn_chunkwise", {"attention_type": "gated_delta", "gdn_chunk_size": 64})]

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "dtype": args.dtype,
        "seeds": list(SEEDS),
        "cells": [],
    }

    for name, extra in configs:
        per_seed = {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = NeuratalModel(NeuratalConfig(dtype=args.dtype, max_position_embeddings=4096, **extra)).to(device).eval()
            for T in lengths:
                ids = torch.randint(0, V, (1, T), generator=torch.Generator().manual_seed(seed)).to(device)
                med, p95 = bench_prefill(model, ids)
                dec = bench_decode(model, model.config, T)
                per_seed.setdefault(T, {"prefill": [], "p95": [], "decode": []})
                per_seed[T]["prefill"].append(med)
                per_seed[T]["p95"].append(p95)
                per_seed[T]["decode"].append(dec)
            del model
            torch.cuda.empty_cache()

        cell = {"attention_type": name, "config": extra, "median_across_seeds": {}, "raw_per_seed": {}}
        for T in lengths:
            ms = statistics.median(per_seed[T]["prefill"])
            ds = statistics.median(per_seed[T]["decode"])
            p95 = statistics.median(per_seed[T]["p95"])
            cell["median_across_seeds"][str(T)] = {
                "prefill_ms": ms,
                "prefill_p95_ms": p95,
                "decode_ms_per_token": ds,
            }
            cell["raw_per_seed"][str(T)] = {
                "prefill_ms_per_seed": per_seed[T]["prefill"],
                "prefill_p95_ms_per_seed": per_seed[T]["p95"],
                "decode_ms_per_token_per_seed": per_seed[T]["decode"],
            }
        results["cells"].append(cell)
        for T in lengths:
            print(f"{name:16s} T={T:5d}: prefill {cell['median_across_seeds'][str(T)]['prefill_ms']*1e3:8.2f} ms  "
                  f"decode {cell['median_across_seeds'][str(T)]['decode_ms_per_token']*1e3:.3f} ms/tok")

    path = save_result("gdn_chunkwise", results)
    print(f"results -> {path}")


if __name__ == "__main__":
    main()
