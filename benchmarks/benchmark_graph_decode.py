"""Static-shape CUDA-graph decode benchmark -> results/graph_decode_results_<ts>.json

Compares eager decode vs the opt-in experimental CUDA-graph decode on identical
prompts, seeds and timing methodology. Correctness gate runs first (bitwise
id-match on greedy generation); benchmark runs only if it passes.
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

from neuratal import NeuratalConfig, NeuratalModel, generate

V = 262_144


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--prompts", default="16,64,128,512")
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    prompt_lens = [int(x) for x in args.prompts.split(",")]

    torch.manual_seed(SEED)
    model = NeuratalModel(NeuratalConfig(dtype=args.dtype)).to(device).eval()

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "config": {**model.config.to_dict(), "model_dtype": args.dtype, "device": str(device)},
        "gen_len": args.gen_len,
        "cells": [],
    }

    for plen in prompt_lens:
        torch.manual_seed(SEED)
        base_ids = torch.randint(0, V, (1, plen), device=device)
        cell = {"prompt_len": plen}

        with torch.inference_mode():
            out_eager = generate(model, base_ids, max_new_tokens=16)
            out_graph = generate(model, base_ids, max_new_tokens=16, experimental_cuda_graphs=True)
        correct = torch.equal(out_eager, out_graph)
        cell["bitwise_equal_greedy_16"] = correct
        if not correct:
            cell["skipped"] = "correctness gate failed (graph != eager ids)"
            results["cells"].append(cell)
            print(f"prompt={plen}: CORRECTNESS GATE FAILED — no timing recorded")
            continue

        # eager
        torch.cuda.synchronize()
        samples, samples_graph = [], []
        with torch.inference_mode():
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                generate(model, base_ids, max_new_tokens=args.gen_len)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - t0)
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                generate(model, base_ids, max_new_tokens=args.gen_len, experimental_cuda_graphs=True)
                torch.cuda.synchronize()
                samples_graph.append(time.perf_counter() - t0)

        import statistics

        cell.update({
            "eager_median_s": statistics.median(samples),
            "graph_median_s": statistics.median(samples_graph),
            "eager_min_s": min(samples),
            "graph_min_s": min(samples_graph),
            "speedup": statistics.median(samples) / statistics.median(samples_graph),
            "repeats": args.repeats,
        })
        results["cells"].append(cell)
        print(f"prompt={plen}: eager {cell['eager_median_s']:.3f}s  graph {cell['graph_median_s']:.3f}s  "
              f"speedup {cell['speedup']:.2f}x")

    path = save_result("graph_decode", results)
    print(f"graph decode results -> {path}")


if __name__ == "__main__":
    main()
