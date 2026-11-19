# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""KV cache benchmark -> results/kv_cache_results_<ts>.json

Compares full-sequence uncached inference, cached prefill, single-token cached
decoding and multi-token incremental decoding across context lengths.
Verifies numerical equivalence of cached vs uncached paths (fp32: < 1e-5).
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
from neuratal.model.cache import KVCache

V = 262_144


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--lengths", default="64,128,256,512")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--decode-tokens", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    lengths = [int(x) for x in args.lengths.split(",")]

    torch.manual_seed(SEED)
    model = NeuratalModel(NeuratalConfig(cache_capacity=2048)).to(device, dtype).eval()

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "config": {**NeuratalConfig().to_dict(), "dtype": args.dtype, "device": str(device)},
        "decode_tokens_per_call": args.decode_tokens,
        "cells": [],
    }

    for seq in lengths:
        ids = torch.randint(0, V, (1, seq), generator=torch.Generator().manual_seed(SEED)).to(device)
        cell = {"seq": seq}

        # --- numerical equivalence (cached vs uncached), fp32 only
        with torch.inference_mode():
            ref = model(ids, use_cache=False).logits
            cache = KVCache(model.config)
            out = model(ids, past_key_values=cache).logits
        max_abs = (ref - out).abs().max().item()
        cell["cached_vs_uncached_max_abs_diff"] = max_abs
        if dtype == torch.float32 and max_abs >= 1e-5:
            raise AssertionError(f"cached path diverged beyond 1e-5: {max_abs}")

        # --- uncached full-sequence forward
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(ids, use_cache=False)
            torch.cuda.synchronize() if device.type == "cuda" else None
            samples = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                model(ids, use_cache=False)
                torch.cuda.synchronize() if device.type == "cuda" else None
                samples.append(time.perf_counter() - t0)
        import statistics

        cell["uncached_forward_median_s"] = statistics.median(samples)

        # --- cached prefill (fresh cache each iteration, long enough decode afterwards)
        def cached_prefill():
            cache = KVCache(model.config)
            with torch.inference_mode():
                model(ids, past_key_values=cache)
            return cache

        with torch.inference_mode():
            for _ in range(args.warmup):
                cached_prefill()
            samples = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                cached_prefill()
                torch.cuda.synchronize() if device.type == "cuda" else None
                samples.append(time.perf_counter() - t0)
        cell["cached_prefill_median_s"] = statistics.median(samples)

        # --- cached decode of decode-tokens additional tokens (fresh cache per repeat)
        def decode_run():
            cache = cached_prefill()  # prefill not timed
            torch.cuda.synchronize() if device.type == "cuda" else None
            cur = torch.randint(0, V, (1, 1), device=device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(args.decode_tokens):
                    model(cur, past_key_values=cache)  # model.commit() advances the cache internally
                    cur = torch.randint(0, V, (1, 1), device=device)
            torch.cuda.synchronize() if device.type == "cuda" else None
            return time.perf_counter() - t0, cache

        decode_samples, last_cache = [], None
        for _ in range(args.repeats):
            elapsed, last_cache = decode_run()
            decode_samples.append(elapsed)
        cache = last_cache
        per_token = statistics.median(decode_samples) / args.decode_tokens
        cell["decode_ms_per_token"] = per_token * 1e3
        cell["decode_tokens_per_second"] = 1.0 / per_token
        cell["cache_length_after_decode"] = cache.get_seq_length()

        # --- cache memory (exact buffer bytes)
        k_v_bytes = 2 * model.config.num_hidden_layers * 1 * model.config.num_key_value_heads \
            * model.config.cache_capacity * model.config.head_dim * torch.empty(0, dtype=dtype).element_size()
        cell["cache_buffer_bytes_theoretical"] = k_v_bytes
        cell["cache_length_after_decode"] = cache.get_seq_length()
        results["cells"].append(cell)
        print(f"seq={seq}: uncached {cell['uncached_forward_median_s']*1e3:.2f} ms  "
              f"cached_prefill {cell['cached_prefill_median_s']*1e3:.2f} ms  "
              f"decode {cell['decode_ms_per_token']:.2f} ms/tok  diff {max_abs:.2e}")

    path = save_result("kv_cache", results)
    print(f"kv cache results -> {path}")


if __name__ == "__main__":
    main()
