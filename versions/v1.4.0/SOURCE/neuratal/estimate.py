# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Analytical memory estimator for Neuratal configurations.

All numbers are *estimates from closed-form formulas*, never measurements,
unless explicitly labeled otherwise. Formulas follow standard training
memory accounting for factorized tied-embedding Transformers:

    params       = P
    grads        = P * dtype_bytes(grad)
    AdamW state  = 2 * P * 4 (fp32 m + v)             [fused/flattened internally]
    master       = P * 4 (fp32 master copy, AMP)      [0 if pure-fp32 training]
    activations  = ~34 * B * T * H * L * bytes(act)    [mixed input-dependent]
    logits       = B * T * V * bytes(logits)            [materialized only with chunking -> ~B*T*V/c * bytes]
    KV cache     = B * L * T * 2 * kv_heads * head_dim * bytes(dtype)   [inference only]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from neuratal.config import NeuratalConfig
from neuratal.presets import parameter_breakdown_meta

SUPPORTED_STRATEGIES = ("single", "ddp", "zero1", "zero2", "zero3", "fsdp")


@dataclass
class MemoryEstimate:
    config_name: str
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    master_weight_bytes: int
    activation_bytes: int
    logits_bytes: int
    kv_cache_bytes: int
    total_bytes: int
    notes: list[str]
    # MoE / distributed breakdown (defaults keep dense call sites unchanged)
    total_parameter_count: int = 0
    active_parameter_count: int = 0
    active_parameter_bytes: int = 0
    router_parameter_bytes: int = 0
    expert_parameter_bytes: int = 0
    shared_expert_parameter_bytes: int = 0
    expert_activation_bytes: int = 0
    communication_bytes: int = 0
    world_size: int = 1
    strategy: str = "single"

    def gib(self, b: int) -> float:
        return b / 2**30

    def report(self) -> str:
        lines = [
            f"Memory estimate for {self.config_name}:",
            f"  strategy       : {self.strategy} (world_size={self.world_size}, per rank)",
            f"  params         : {self.gib(self.parameter_bytes):.3f} GiB",
            f"  gradients      : {self.gib(self.gradient_bytes):.3f} GiB",
            f"  AdamW(m+v,f32) : {self.gib(self.optimizer_state_bytes):.3f} GiB",
            f"  master fp32    : {self.gib(self.master_weight_bytes):.3f} GiB",
            f"  activations    : {self.gib(self.activation_bytes):.3f} GiB",
            f"  logits (CE)    : {self.gib(self.logits_bytes):.3f} GiB  (chunked; peak per chunk)",
            f"  KV cache       : {self.gib(self.kv_cache_bytes):.3f} GiB",
            f"  comm buffers   : {self.gib(self.communication_bytes):.3f} GiB",
            f"  TOTAL          : {self.gib(self.total_bytes):.3f} GiB",
        ]
        if self.total_parameter_count != self.active_parameter_count:
            lines.insert(2, f"  total params   : {self.total_parameter_count:,}")
            lines.insert(3, f"  active/token   : {self.active_parameter_count:,}")
            lines.insert(4, f"  active bytes   : {self.gib(self.active_parameter_bytes):.3f} GiB")
            lines.insert(5, f"  router bytes   : {self.gib(self.router_parameter_bytes):.3f} GiB")
            lines.insert(6, f"  expert bytes   : {self.gib(self.expert_parameter_bytes):.3f} GiB")
            if self.shared_expert_parameter_bytes:
                lines.insert(7, f"  shared expert  : {self.gib(self.shared_expert_parameter_bytes):.3f} GiB")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def estimate(
    config: NeuratalConfig,
    *,
    batch_size: int = 8,
    seq_length: int = 1024,
    grad_dtype_bytes: Optional[int] = None,
    act_dtype_bytes: int = 2,
    logits_chunk: int = 256,
    master_fp32: bool = True,
    kv_context: int = 0,
    name: str = "custom",
    world_size: int = 1,
    strategy: str = "single",
) -> MemoryEstimate:
    """Closed-form per-rank memory estimate.

    ``strategy`` is an *estimate* for a hypothetical sharding layout, not an
    implementation claim: ``single``/``ddp`` are the only paths this repository
    actually trains; ``zero1``/``zero2``/``zero3``/``fsdp`` are accounting
    models for future work. Tensor/pipeline/expert parallelism are deliberately
    rejected rather than silently approximated.

    ``grad_dtype_bytes`` defaults to the parameter dtype implied by
    ``config.dtype`` (4 for float32, 2 otherwise) — an explicit override is
    always honored.
    """
    if batch_size < 1 or seq_length < 1:
        raise ValueError("positive dims required")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {SUPPORTED_STRATEGIES}, got {strategy!r}; "
            "tensor/pipeline/expert parallelism are not implemented"
        )
    cfg = config
    breakdown = parameter_breakdown_meta(cfg)
    P = breakdown["total"]
    P_active = breakdown["active_per_token"]

    H, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    if cfg.dtype == "float32":
        act_dtype_bytes = 4
        param_dtype_bytes = 4
        if grad_dtype_bytes is None:
            grad_dtype_bytes = 4
    elif cfg.dtype in ("bfloat16", "float16"):
        act_dtype_bytes = 2
        param_dtype_bytes = 2
        if grad_dtype_bytes is None:
            grad_dtype_bytes = 2
    else:
        act_dtype_bytes = 2
        param_dtype_bytes = 2
        if grad_dtype_bytes is None:
            grad_dtype_bytes = 2
    if not isinstance(grad_dtype_bytes, int) or grad_dtype_bytes < 1:
        raise ValueError(f"grad_dtype_bytes must be a positive integer, got {grad_dtype_bytes!r}")

    # --- per-rank parameter/gradient/optimizer accounting ---
    shard = world_size if strategy in ("zero1", "zero2", "zero3", "fsdp") else 1
    if strategy in ("zero3", "fsdp"):
        param_count = P / shard
    else:
        param_count = P
    if strategy in ("zero2", "zero3", "fsdp"):
        grad_count = P / shard
    else:
        grad_count = P
    if strategy == "single":
        opt_count = P
        master_count = P if master_fp32 else 0
    elif strategy == "ddp":
        opt_count = P
        master_count = P if master_fp32 else 0
    else:
        opt_count = P / shard
        master_count = (P / shard) if master_fp32 else 0

    param_bytes = int(param_count * param_dtype_bytes)
    grad_bytes = int(grad_count * grad_dtype_bytes)
    optimizer_state = int(opt_count * 2 * 4)          # AdamW m, v in fp32
    master = int(master_count * 4)

    # --- activation approximation (active experts only) ---
    moe_layers = set(cfg.moe_layer_indices or []) if cfg.is_moe else set()
    dense_layers = L - len(moe_layers)
    top_k = cfg.num_experts_per_token if cfg.is_moe else 0
    expert_hidden = cfg.expert_hidden_dim or cfg.intermediate_size
    shared_hidden = cfg.shared_expert_hidden_dim or expert_hidden

    dense_per_token = dense_layers * (12 * H + 4 * cfg.intermediate_size)
    moe_per_token = 0
    expert_act = 0
    if moe_layers:
        # attention part + top-k expert MLP + optional shared expert + router logits
        per_layer = 12 * H + 4 * top_k * expert_hidden + (4 * shared_hidden if cfg.shared_expert else 0) + cfg.num_experts
        moe_per_token = len(moe_layers) * per_layer
        expert_act = (
            len(moe_layers) * (4 * top_k * expert_hidden
                               + (4 * shared_hidden if cfg.shared_expert else 0))
            * batch_size * seq_length * act_dtype_bytes
        )
    per_token = (dense_per_token + moe_per_token) * act_dtype_bytes
    activations = batch_size * seq_length * per_token

    # logits workspace for chunked CE (one chunk resident; fp32 because the
    # chunked kernel upcasts to fp32 internally, even for bf16 models)
    logits_bytes = batch_size * min(seq_length, logits_chunk) * V * 4

    kv_bytes = batch_size * L * kv_context * 2 * cfg.num_key_value_heads * cfg.head_dim * act_dtype_bytes

    # --- communication buffers (order-of-magnitude) ---
    if strategy in ("single",):
        comm = 0
    elif strategy in ("ddp", "zero1", "zero2"):
        comm = int(P * grad_dtype_bytes)
    else:  # zero3 / fsdp: parameter all-gather + gradient reduce-scatter
        comm = int(2 * P * grad_dtype_bytes)

    notes = [
        "ESTIMATE (closed-form), not a measurement; measure with torch.cuda.max_memory_allocated for ground truth",
        "activation estimate is a family-of-models approximation; it is not a substitute for profiling",
        "logits assume chunked cross-entropy (256-token chunks, fp32 workspace); full CE would be B*T*V*4 bytes (impractical at 262k vocab)",
        "optimizer state assumes AdamW with fp32 m/v; Muon/Lion differ",
        "activation estimate charges top_k experts per token; the 262144-row embedding gradient is included in grad bytes, not in activations",
    ]
    if cfg.is_moe:
        notes.append(
            "MoE: storage/optimizer state is charged to ALL experts; activation accounting charges only top_k experts per token"
        )
        notes.append(
            "expert parallelism (all-to-all) is NOT implemented; distributed MoE training is not claimed"
        )
    if strategy in ("zero1", "zero2", "zero3", "fsdp"):
        notes.append(f"{strategy} numbers are an accounting model, not an implemented training path")
    if strategy == "ddp" and cfg.is_moe:
        notes.append("DDP MoE: router statistics are per-rank; global load balance is not synchronized")

    total = param_bytes + grad_bytes + optimizer_state + master + activations + logits_bytes + kv_bytes + comm
    return MemoryEstimate(
        config_name=name,
        parameter_bytes=param_bytes,
        gradient_bytes=grad_bytes,
        optimizer_state_bytes=optimizer_state,
        master_weight_bytes=master,
        activation_bytes=activations,
        logits_bytes=logits_bytes,
        kv_cache_bytes=kv_bytes,
        total_bytes=total,
        notes=notes,
        total_parameter_count=P,
        active_parameter_count=P_active,
        active_parameter_bytes=int(P_active * param_dtype_bytes),
        router_parameter_bytes=int(breakdown["moe_router"] * param_dtype_bytes),
        expert_parameter_bytes=int(breakdown["moe_experts"] * param_dtype_bytes),
        shared_expert_parameter_bytes=int(breakdown["moe_shared_expert"] * param_dtype_bytes),
        expert_activation_bytes=expert_act,
        communication_bytes=comm,
        world_size=world_size,
        strategy=strategy,
    )
