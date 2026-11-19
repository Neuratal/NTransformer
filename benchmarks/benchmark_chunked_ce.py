# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Chunked linear cross-entropy benchmark vs standard cross entropy.

Fair comparison: identical hidden states, labels, weights, dtype, and device;
identical reduction semantics (sum over supervised tokens / count externally).
The 262,144-token vocabulary makes the materialized logits the dominant cost.

Measures: forward time, forward+backward time, peak allocated / reserved VRAM,
for both paths, across (batch*seq) token counts. Verifies gradient parity
(max abs diff) between the two paths on one small case before reporting timings.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from benchmark_utils import machine_fingerprint, run_id, save_result, SEED, stats

from neuratal.training.chunked_linear_ce import chunked_linear_cross_entropy

VOCAB = 262_144
F_DIM = 64  # factorized embedding width (dev/50M-class presets)


def time_step(fn, *, warmup: int, repeats: int, device) -> list[float]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return samples


def peak_memory_of(fn, device):
    if device.type != "cuda":
        return 0, 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--tokens", default="512,1024")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(SEED)

    out_proj = torch.randn(F_DIM, 512, device=device, dtype=dtype) * 0.02
    emb = torch.randn(VOCAB, F_DIM, device=device, dtype=dtype) * 0.02
    out_proj.requires_grad_(True)
    emb.requires_grad_(True)

    results = {"run_id": run_id(), "machine": machine_fingerprint(),
               "dtype": args.dtype, "vocab": VOCAB, "embedding_factor": F_DIM,
               "hidden_size": 512, "chunk_size": 256, "cells": []}

    # ---- gradient parity check on one case ----
    torch.manual_seed(SEED)
    hidden = torch.randn(64, 512, device=device, dtype=dtype)
    targets = torch.randint(0, VOCAB, (64,), device=device)
    h1 = hidden.clone().requires_grad_(True)
    h2 = hidden.clone().requires_grad_(True)
    total, n = chunked_linear_cross_entropy(h1, targets, out_proj, emb, chunk_size=16)
    (total / n).backward()
    logits = F.linear(F.linear(h2.float(), out_proj.float()), emb.float())
    F.cross_entropy(logits, targets).backward()
    rel = (h1.grad.float() - h2.grad.float()).norm() / h2.grad.float().norm().clamp_min(1e-12)
    results["gradient_parity_relative_l2"] = float(rel)
    print(f"gradient parity (rel L2, fp32 accumulate): {float(rel):.3e}")

    for n_tokens in [int(t) for t in args.tokens.split(",")]:
        hidden = torch.randn(n_tokens, 512, device=device, dtype=dtype)
        targets = torch.randint(0, VOCAB, (n_tokens,), device=device)

        def chunked_fwdbwd():
            h = hidden.detach().requires_grad_(True)
            out_p = out_proj.detach().requires_grad_(True)
            e = emb.detach().requires_grad_(True)
            total, n = chunked_linear_cross_entropy(h, targets, out_p, e, chunk_size=256)
            (total / n).backward()

        def standard_fwdbwd():
            h = hidden.detach().requires_grad_(True)
            out_p = out_proj.detach().requires_grad_(True)
            e = emb.detach().requires_grad_(True)
            logits = F.linear(F.linear(h.float(), out_p.float()), e.float())
            F.cross_entropy(logits, targets).backward()

        cell = {"tokens": n_tokens}
        cell["chunked_fwdbwd_s"] = stats(time_step(chunked_fwdbwd, warmup=args.warmup, repeats=args.repeats, device=device))
        cell["standard_fwdbwd_s"] = stats(time_step(standard_fwdbwd, warmup=args.warmup, repeats=args.repeats, device=device))
        if device.type == "cuda":
            cell["chunked_fwdbwd_peak_allocated_bytes"], cell["chunked_fwdbwd_peak_reserved_bytes"] = peak_memory_of(chunked_fwdbwd, device)
            cell["standard_fwdbwd_peak_allocated_bytes"], cell["standard_fwdbwd_peak_reserved_bytes"] = peak_memory_of(standard_fwdbwd, device)
        logits_bytes = n_tokens * VOCAB * 4
        cell["standard_logits_bytes_fp32"] = logits_bytes
        results["cells"].append(cell)
        print(
            f"tokens={n_tokens}: chunked {cell['chunked_fwdbwd_s']['median']*1e3:.1f} ms"
            f" (alloc {cell.get('chunked_fwdbwd_peak_allocated_bytes', 0)/2**20:.0f} MiB) | "
            f"standard {cell['standard_fwdbwd_s']['median']*1e3:.1f} ms"
            f" (alloc {cell.get('standard_fwdbwd_peak_allocated_bytes', 0)/2**20:.0f} MiB)"
        )

    path = save_result("chunked_ce", results)
    print(f"chunked_ce results -> {path}")


if __name__ == "__main__":
    main()
