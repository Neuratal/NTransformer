# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""SDPA backend evaluation -> results/attention_results_<ts>.json

Forces each SDPA backend via sdpa_kernel for fp32/bf16/fp16 and records
whether it actually executes and how fast, plus numerical deltas vs fp64 math
and vs the reference implementation. Never assumes availability.
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
from benchmark_utils import machine_fingerprint, run_id, save_result, SEED

from neuratal.config import NeuratalConfig
from neuratal.model.attention import reference_attention


def sdpa_case(backend, dtype, q, k, v, groups):
    from torch.nn.attention import sdpa_kernel

    is_gqa = q.size(1) != k.size(1)
    try:
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  dropout_p=0.0,
                                                  enable_gqa=is_gqa)
    except Exception as exc:
        return exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        print("attention backend probing requires CUDA; running reference-only comparison on CPU")

    cfg = NeuratalConfig()
    H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    print(f"probing shape B={args.batch} heads={H} kv_heads={KV} T={args.seq} d_h={D}")

    torch.manual_seed(SEED)

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "shape": {"batch": args.batch, "heads": H, "kv_heads": KV, "seq": args.seq, "head_dim": D},
        "cases": [],
    }

    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        q = torch.randn(args.batch, H, args.seq, D, device=device, dtype=dtype)
        k = torch.randn(args.batch, KV, args.seq, D, device=device, dtype=dtype)
        v = torch.randn(args.batch, KV, args.seq, D, device=device, dtype=dtype)
        ref64 = reference_attention(q.double(), k.double(), v.double(),
                                    num_query_groups=H // KV, scaling=1.0 / (D ** 0.5)).float()

        from torch.nn.attention import SDPBackend

        for backend_name, backend in (
            ("flash", SDPBackend.FLASH_ATTENTION),
            ("efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("cudnn", SDPBackend.CUDNN_ATTENTION),
            ("math", SDPBackend.MATH),
        ):
            case = {"dtype": dtype_name, "backend": backend_name}
            out = sdpa_case(backend, dtype, q, k, v, H // KV)
            if isinstance(out, Exception):
                case["status"] = "unavailable"
                case["error"] = str(out)[:200]
            else:
                case["status"] = "ok"
                case["max_abs_vs_fp64"] = (out.float() - ref64).abs().max().item()
                # timing under the forced backend
                try:
                    from torch.nn.attention import sdpa_kernel

                    def call():
                        with sdpa_kernel(backend):
                            F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                           enable_gqa=(H != KV))

                    for _ in range(args.warmup):
                        call()
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    samples = []
                    for _ in range(args.repeats):
                        t0 = time.perf_counter()
                        call()
                        torch.cuda.synchronize() if device.type == "cuda" else None
                        samples.append(time.perf_counter() - t0)
                    import statistics

                    case["median_s"] = statistics.median(samples)
                except Exception as exc:
                    case["status"] = "failed_during_timing"
                    case["error"] = str(exc)[:200]
            results["cases"].append(case)
            print(f"{dtype_name:9s} {backend_name:9s} -> {case['status']:22s} "
                  f"diff_vs_fp64={case.get('max_abs_vs_fp64')}"
                  f" median={case.get('median_s', float('nan'))*1e3:.3f} ms")

    path = save_result("attention", results)
    print(f"attention results -> {path}")


if __name__ == "__main__":
    main()
