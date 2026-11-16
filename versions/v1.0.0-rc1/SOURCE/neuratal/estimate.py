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

import torch

from neuratal.config import NeuratalConfig
from neuratal.presets import get_preset


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

    def gib(self, b: int) -> float:
        return b / 2**30

    def report(self) -> str:
        lines = [
            f"Memory estimate for {self.config_name}:",
            f"  params         : {self.gib(self.parameter_bytes):.3f} GiB",
            f"  gradients      : {self.gib(self.gradient_bytes):.3f} GiB",
            f"  AdamW(m+v,f32) : {self.gib(self.optimizer_state_bytes):.3f} GiB",
            f"  master fp32    : {self.gib(self.master_weight_bytes):.3f} GiB",
            f"  activations    : {self.gib(self.activation_bytes):.3f} GiB",
            f"  logits (CE)    : {self.gib(self.logits_bytes):.3f} GiB  (chunked; peak per chunk)",
            f"  KV cache       : {self.gib(self.kv_cache_bytes):.3f} GiB",
            f"  TOTAL          : {self.gib(self.total_bytes):.3f} GiB",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def estimate(
    config: NeuratalConfig,
    *,
    batch_size: int = 8,
    seq_length: int = 1024,
    grad_dtype_bytes: int = 2,
    act_dtype_bytes: int = 2,
    logits_chunk: int = 256,
    master_fp32: bool = True,
    kv_context: int = 0,
    name: str = "custom",
) -> MemoryEstimate:
    if batch_size < 1 or seq_length < 1:
        raise ValueError("positive dims required")

    cfg = config
    from neuratal.presets import parameter_count_meta
    P = parameter_count_meta(cfg)

    H, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
    if cfg.dtype == "float32":
        act_dtype_bytes = 4
        param_bytes = P * 4
    elif cfg.dtype in ("bfloat16", "float16"):
        act_dtype_bytes = 2
        param_bytes = P * 2
    else:
        param_bytes = P * 2

    grad_bytes = P * grad_dtype_bytes
    optimizer_state = P * 2 * 4          # AdamW m, v in fp32
    master = P * 4 if master_fp32 else 0

    # activation approximation: inputs/residuals + attention proj + MLP intermediates
    # ~ (12 * H + 4 * I) * bytes per token per layer; use 22*H dominance as proxy
    per_token_per_layer = (12 * H + 4 * cfg.intermediate_size) * act_dtype_bytes
    activations = batch_size * seq_length * L * per_token_per_layer

    # logits for CE (only one chunk resident at a time)
    logits_bytes = batch_size * min(seq_length, logits_chunk) * V * 2

    kv_bytes = batch_size * L * kv_context * 2 * cfg.num_key_value_heads * cfg.head_dim * act_dtype_bytes

    notes = [
        "activation estimate is a family-of-models approximation; measure with torch.cuda.max_memory_allocated for ground truth",
        "logits assume chunked cross-entropy (256-token chunks); full CE would be B*T*V*4 bytes (impractical at 262k vocab)",
        "optimizer state assumes AdamW with fp32 m/v; Muon/Lion differ",
    ]

    total = param_bytes + grad_bytes + optimizer_state + master + activations + logits_bytes + kv_bytes
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
    )
