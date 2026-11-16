"""Measured performance checks for the development preset.

Not a production benchmark; it exists so the engineering report can quote
numbers measured on a specific machine with a specific build. It measures:

* forward latency (eval, no-grad)
* forward+backward latency (training-relevant shapes)
* incremental decode latency via generate() with and without the KV cache
* peak CUDA memory (torch.cuda.max_memory_allocated)
* which attention kernels actually execute (torch.profiler observation)

Every reported number carries its context (hardware, torch/CUDA versions,
dtype, batch, sequence length). Numbers are printed, never inferred.

Usage:  python examples/benchmark.py [--device cuda|cpu] [--dtype float32|bfloat16]
"""

from __future__ import annotations

import argparse
import time

import torch

from neuratal import NeuratalConfig, NeuratalModel, generate

V = 262_144


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_forward(model, ids, iters=50, warmup=10):
    device = ids.device
    with torch.no_grad():
        for _ in range(warmup):
            model(ids)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(ids)
        _sync(device)
    return (time.perf_counter() - t0) / iters * 1e3


def time_forward_backward(model, ids, iters=20, warmup=5):
    device = ids.device
    for _ in range(warmup):
        model(ids).logits.square().mean().backward()
        model.zero_grad(set_to_none=True)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(ids).logits.square().mean().backward()
        model.zero_grad(set_to_none=True)
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1e3


def time_generation(model, ids, max_new_tokens=16):
    _sync(ids.device)
    t0 = time.perf_counter()
    out = generate(model, ids, max_new_tokens=max_new_tokens)
    _sync(ids.device)
    total = time.perf_counter() - t0
    return total, (total / max_new_tokens) * 1e3


def peak_memory(model, ids, mode="forward"):
    if ids.device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    if mode == "forward":
        with torch.no_grad():
            model(ids)
    else:
        model(ids).logits.square().mean().backward()
        model.zero_grad(set_to_none=True)
    _sync(ids.device)
    return torch.cuda.max_memory_allocated() / 2**20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  device {device}  dtype {args.dtype}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}  cc {torch.cuda.get_device_capability(0)}")
    print(f"vocab_size {V} (full NTokenizer vocabulary)")

    torch.manual_seed(0)
    model = NeuratalModel(NeuratalConfig()).to(device, dtype).eval()

    for batch, seq in [(1, 64), (4, 64), (1, 512), (2, 512)]:
        ids = torch.randint(0, V, (batch, seq), device=device)
        fwd = time_forward(model, ids)
        ms_per_tok = fwd / seq
        print(f"[forward] B={batch} T={seq}: {fwd:.2f} ms/step, {ms_per_tok:.3f} ms/token")

    ids = torch.randint(0, V, (1, 64), device=device)
    fb = time_forward_backward(model, ids)
    print(f"[fwd+bwd] B=1 T=64: {fb:.2f} ms/step")

    prompt = torch.randint(0, V, (1, 16), device=device)
    for use_cache in (True, False):
        model.config.use_cache = use_cache
        total, per_tok = time_generation(model, prompt, max_new_tokens=16)
        print(f"[generate] cache={use_cache}: total {total:.2f}s, {per_tok:.2f} ms/token")
    model.config.use_cache = True

    if device.type == "cuda":
        for batch, seq in [(1, 64), (2, 512)]:
            ids = torch.randint(0, V, (batch, seq), device=device)
            fwd = peak_memory(model, ids, "forward")
            fb = peak_memory(model, ids, "forward_backward")
            print(f"[memory] B={batch} T={seq}: peak_forward {fwd:.1f} MiB, peak_fwd+bwd {fb:.1f} MiB")

    if device.type == "cuda":
        from torch.profiler import ProfilerActivity, profile

        ids = torch.randint(0, V, (1, 128), device=device)
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
                model(ids)
        print("[profiler] top CUDA kernels by time (one prefill):")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
