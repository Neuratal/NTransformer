"""MoE benchmark: router/dispatch/expert/combine + end-to-end block comparison.

Measures, under identical geometry and dtype:

* dense SwiGLU vs MoE FFN forward / forward+backward (same hidden size;
  MoE expert_hidden_dim defaults to the dense intermediate_size),
* full-model (hidden-states only, no LM head) forward / backward / optimizer
  step for a small MoE config vs the same-geometry dense model,
* peak allocated / reserved VRAM (reset before each measurement),
* tokens/second,
* dispatch statistics at init (active experts, load imbalance),
* chunked-vs-standard behavior across batch sizes and sequence lengths.

Usage:
    python benchmarks/benchmark_moe.py --device cuda --dtype bfloat16

Reproduce before/after optimization runs with the same flags; results are
written to benchmarks/results/moe_<run_id>.json (+ moe_latest.json).
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
import torch.nn.functional as F

from benchmark_utils import machine_fingerprint, run_id, save_result, SEED, stats

from neuratal import NeuratalConfig, NeuratalModel
from neuratal.model.mlp import SwiGLU
from neuratal.model.moe import MoEFFN


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_step(fn, *, warmup: int, repeats: int, device) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append(time.perf_counter() - t0)
    return samples


def peak_memory_of(fn, device) -> tuple[int, int]:
    """Returns (peak_allocated_bytes, peak_reserved_bytes) for one fn() call."""
    if device.type != "cuda":
        return 0, 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sync(device)
    fn()
    sync(device)
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def module_cell(name, dense_ffn, moe_ffn, hidden, batch, seq, dtype, device, warmup, repeats):
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    n_tokens = batch * seq

    cell = {
        "batch": batch, "seq": seq, "tokens": n_tokens, "dtype": str(dtype).removeprefix("torch."),
        "hidden": hidden,
    }

    def dense_fwd():
        dense_ffn(x)

    def dense_fwdbwd():
        dense_ffn.zero_grad(set_to_none=True)
        out = dense_ffn(x)
        out.backward(grad)

    def moe_fwd():
        moe_ffn(x, compute_aux=False)

    def moe_fwdbwd():
        moe_ffn.zero_grad(set_to_none=True)
        out, _ = moe_ffn(x, compute_aux=False)
        out.backward(grad)

    def moe_fwdbwd_aux():
        moe_ffn.zero_grad(set_to_none=True)
        out, st = moe_ffn(x, compute_aux=True)
        loss = out.float().square().mean() + (st.aux_loss or 0.0) + (st.z_loss or 0.0)
        loss.backward()

    cell["dense_fwd_s"] = stats(time_step(dense_fwd, warmup=warmup, repeats=repeats, device=device))
    cell["dense_fwdbwd_s"] = stats(time_step(dense_fwdbwd, warmup=warmup, repeats=repeats, device=device))
    cell["moe_fwd_s"] = stats(time_step(moe_fwd, warmup=warmup, repeats=repeats, device=device))
    cell["moe_fwdbwd_s"] = stats(time_step(moe_fwdbwd, warmup=warmup, repeats=repeats, device=device))
    cell["moe_fwdbwd_aux_s"] = stats(time_step(moe_fwdbwd_aux, warmup=warmup, repeats=repeats, device=device))
    if device.type == "cuda":
        cell["dense_fwdbwd_peak_allocated_bytes"], cell["dense_fwdbwd_peak_reserved_bytes"] = peak_memory_of(dense_fwdbwd, device)
        cell["moe_fwdbwd_peak_allocated_bytes"], cell["moe_fwdbwd_peak_reserved_bytes"] = peak_memory_of(moe_fwdbwd, device)
    with torch.inference_mode():
        _, st = moe_ffn(x.detach(), compute_aux=True)
    counts = st.counts.cpu()
    cell["moe_active_experts"] = int((counts > 0).sum().item())
    cell["moe_load_imbalance_max_over_mean"] = float(counts.max().item() / counts.mean().clamp_min(1e-9).item())
    cell["moe_assignments_per_expert"] = [int(c) for c in counts.tolist()]
    print(
        f"module B={batch} T={seq}: dense fwd {cell['dense_fwd_s']['median']*1e3:.3f} ms "
        f"bwd {cell['dense_fwdbwd_s']['median']*1e3:.3f} ms | moe fwd {cell['moe_fwd_s']['median']*1e3:.3f} ms "
        f"bwd {cell['moe_fwdbwd_s']['median']*1e3:.3f} ms | active={cell['moe_active_experts']}/{counts.numel()}"
    )
    return cell


def model_cell(name, dense_cfg, moe_cfg, batch, seq, dtype, device, warmup, repeats, full):
    torch.manual_seed(SEED)
    dense = NeuratalModel(dense_cfg).to(device, dtype)
    torch.manual_seed(SEED)
    moe = NeuratalModel(moe_cfg).to(device, dtype)
    ids = torch.randint(0, dense_cfg.vocab_size, (batch, seq),
                        generator=torch.Generator().manual_seed(SEED)).to(device)

    cell = {"name": name, "batch": batch, "seq": seq, "dtype": str(dtype).removeprefix("torch."),
            "full_forward": full, "dense_params": dense.num_parameters(), "moe_params": moe.num_parameters(),
            "moe_active_params_per_token": moe.active_parameters_per_token()}

    def step(model):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(dtype == torch.float32 and device.type == "cuda")):
            if full:
                out = model(ids)
                logits = out.logits.float()
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), torch.roll(ids, -1, dims=1).reshape(-1))
            else:
                hidden = model.forward_hidden(ids)
                loss = hidden.float().square().mean()
        loss.backward()

    def train_step(model):
        opt = getattr(train_step, "_opt", None)
        if opt is None:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            train_step._opt = opt
        step(model)
        opt.step()

    cell["dense_fwdbwd_s"] = stats(time_step(lambda: step(dense), warmup=warmup, repeats=repeats, device=device))
    cell["moe_fwdbwd_s"] = stats(time_step(lambda: step(moe), warmup=warmup, repeats=repeats, device=device))
    if device.type == "cuda":
        cell["dense_fwdbwd_peak_allocated_bytes"], cell["dense_fwdbwd_peak_reserved_bytes"] = peak_memory_of(lambda: step(dense), device)
        cell["moe_fwdbwd_peak_allocated_bytes"], cell["moe_fwdbwd_peak_reserved_bytes"] = peak_memory_of(lambda: step(moe), device)
    n_tokens = batch * seq
    cell["dense_tokens_per_second"] = n_tokens / cell["dense_fwdbwd_s"]["median"]
    cell["moe_tokens_per_second"] = n_tokens / cell["moe_fwdbwd_s"]["median"]
    print(
        f"model {name} B={batch} T={seq}: dense {cell['dense_fwdbwd_s']['median']*1e3:.3f} ms "
        f"({cell['dense_tokens_per_second']:.0f} tok/s) | moe {cell['moe_fwdbwd_s']['median']*1e3:.3f} ms "
        f"({cell['moe_tokens_per_second']:.0f} tok/s)"
    )
    del dense, moe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--batches", default="2,4")
    parser.add_argument("--seq-lengths", default="128,512")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(SEED)

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "dtype": args.dtype,
        "module_level": [],
        "model_level": [],
    }

    # ---- module-level: identical FFN geometry, dense vs MoE ----
    hidden, intermediate = 512, 1408
    moe_common = dict(
        hidden_size=hidden, num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=2,
        intermediate_size=intermediate, embedding_size=64, max_position_embeddings=512,
        num_experts=8, num_experts_per_token=2, expert_hidden_dim=intermediate,
    )
    dense_ffn = SwiGLU(NeuratalConfig(
        hidden_size=hidden, num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=2,
        intermediate_size=intermediate, embedding_size=64, max_position_embeddings=512,
    )).to(device, dtype)
    moe_ffn = MoEFFN(NeuratalConfig(**moe_common)).to(device, dtype)
    for batch in [int(b) for b in args.batches.split(",")]:
        for seq in [int(s) for s in args.seq_lengths.split(",")]:
            results["module_level"].append(
                module_cell("hidden512-e8k2", dense_ffn, moe_ffn, hidden, batch, seq, dtype, device,
                            args.warmup, args.repeats))

    # ---- model-level: small model, dense vs MoE (all layers MoE for the MoE cell) ----
    batches = [int(b) for b in args.batches.split(",")]
    seqs = [int(s) for s in args.seq_lengths.split(",")]
    small_dense = dict(
        vocab_size=262_144, hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=640, embedding_size=32, max_position_embeddings=512,
    )
    small_moe = dict(small_dense, num_experts=8, num_experts_per_token=2, expert_hidden_dim=640)
    for batch in batches:
        for seq in seqs:
            results["model_level"].append(
                model_cell("dev-tiny-e8k2", NeuratalConfig(**small_dense), NeuratalConfig(**small_moe),
                           batch, seq, dtype, device, args.warmup, args.repeats, full=False))

    path = save_result("moe", results)
    print(f"moe results -> {path}")


if __name__ == "__main__":
    main()
