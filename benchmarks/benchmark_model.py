"""NTransformer dev-preset inference benchmark -> results/model_results_<ts>.json

Measures prefill latency (full-sequence forward) and tokens/s across sequence
lengths 32..1024, batch sizes that fit the memory budget, with inference_mode,
warmup, CUDA synchronization, median and P95.
"""

from __future__ import annotations

import argparse
import sys
import time
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
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seq-lengths", default="32,128,256,512,1024")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) if device.type == "cuda" else None

    model = NeuratalModel(NeuratalConfig()).to(device, dtype).eval()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"model params: {model.num_parameters():,} ({param_bytes / 2**20:.1f} MiB)  device={device}  dtype={dtype}")

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "config": {**NeuratalConfig().to_dict(), "dtype": args.dtype, "device": str(device)},
        "parameter_bytes_theoretical": param_bytes,
        "parameter_count": model.num_parameters(),
        "cells": [],
    }

    for batch in batch_sizes:
        for seq in seq_lengths:
            logits_bytes = batch * seq * V * dtype_nbytes(dtype)
            cell = {"batch": batch, "seq": seq, "logits_bytes_theoretical": logits_bytes}
            if device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(0)
                need = logits_bytes * 3 + 256 * 2**20  # logits + backward-free activations, conservative
                if need > free * 0.85:
                    cell["skipped"] = f"estimated need {need/2**20:.0f} MiB exceeds 85% of free VRAM ({free/2**20:.0f} MiB)"
                    results["cells"].append(cell)
                    print(f"B={batch} T={seq}: SKIPPED — {cell['skipped']}")
                    continue

            ids = torch.randint(0, V, (batch, seq), generator=torch.Generator().manual_seed(SEED)).to(device)
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

            ordered = sorted(samples)
            p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1) + 0.999999))]
            median = statistics.median(samples)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                with torch.inference_mode():
                    model(ids)
                torch.cuda.synchronize()
                cell["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                cell["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()

            cell.update({
                "median_s": median,
                "p95_s": p95,
                "mean_s": statistics.fmean(samples),
                "min_s": min(samples),
                "max_s": max(samples),
                "tokens_per_second": batch * seq / median,
                "repetitions": args.repeats,
                "warmup": args.warmup,
            })
            results["cells"].append(cell)
            print(f"B={batch} T={seq}: median {median*1e3:.3f} ms  p95 {p95*1e3:.3f} ms  "
                  f"{cell['tokens_per_second']:.1f} tok/s  logits {logits_bytes/2**20:.0f} MiB")

    path = save_result("model", results)
    print(f"model results -> {path}")


if __name__ == "__main__":
    main()
