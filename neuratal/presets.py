"""Canonical model scale presets, parameter counting and validation.

All presets map to real NeuratalConfig values. Parameter counts are computed
with meta-device construction of NeuratalModel (no formula drift) and stored
on the Preset dataclass so anything printed by this module is honest.
Numeric suffixes follow industry convention (e.g. "7b" designates a
7B-class design; the *measured* parameter count is printed, not the label).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neuratal.config import NeuratalConfig
from neuratal.model.model import NeuratalModel


@dataclass(frozen=True)
class Preset:
    name: str
    config: NeuratalConfig
    parameter_count: int
    notes: str = ""
    active_parameter_count: int = 0
    kind: str = "dense"
    num_experts: int = 0
    top_k: int = 0
    expert_hidden_dim: int = 0

    @property
    def parameter_bytes_2(self) -> int:
        return self.parameter_count * 2

    @property
    def is_moe(self) -> bool:
        return self.kind == "moe"


def parameter_count_meta(config: NeuratalConfig) -> int:
    with torch.device("meta"):
        model = NeuratalModel(config)
        total = sum(p.numel() for p in model.parameters())
    return total


def parameter_breakdown_meta(config: NeuratalConfig) -> dict[str, int]:
    """Total and active-per-token counts from a real meta-device model."""
    with torch.device("meta"):
        model = NeuratalModel(config)
        return model.parameter_breakdown()


def active_parameter_count_meta(config: NeuratalConfig) -> int:
    return parameter_breakdown_meta(config)["active_per_token"]


def _make(hidden, layers, heads, kv_heads, intermediate,
          embedding_size, context, name, notes="") -> Preset:
    cfg = NeuratalConfig(
        vocab_size=262_144,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=intermediate,
        embedding_size=embedding_size,
        max_position_embeddings=context,
    )
    n = parameter_count_meta(cfg)
    actual = f"measured={n:,} ({n/1e9:.2f}B)" if n >= 1e9 else f"measured={n:,} ({n/1e6:.1f}M)"
    if notes:
        actual += f" — {notes}"
    return Preset(name=name, config=cfg, parameter_count=n, notes=actual,
                  active_parameter_count=n, kind="dense")


PRESETS: dict[str, Preset] = {
    name: _make(*args)
    for name, args in {
        "50m":  (512, 11, 8, 2, 1408, 64, 2048, "D=512 L=11 H=8 KV=2 F=64"),
        "75m":  (640, 14, 10, 2, 1728, 64, 2048, "D=640 L=14 H=10 KV=2 F=64"),
        "100m": (768, 14, 12, 2, 1920, 64, 2048, "D=768 L=14 H=12 KV=2 F=64"),
        "500m": (1536, 20, 24, 2, 4096, 64, 4096, "D=1536 L=20 H=24 KV=2 F=64"),
        "1b":   (2048, 24, 32, 4, 5632, 64, 4096, "D=2048 L=24 H=32 KV=4 F=64"),
        "3b":   (2816, 36, 44, 4, 7600, 96, 4096, "D=2816 L=36 H=44 KV=4 F=96"),
        "7b":   (4096, 32, 64, 8, 11008, 128, 4096, "D=4096 L=32 H=64 KV=8 F=128 (LLaMA-2-7B class)"),
        "13b":  (5120, 44, 80, 8, 13824, 128, 4096, "D=5120 L=44 H=80 KV=8 F=128 (LLaMA-2-13B class)"),
        "30b":  (8192, 60, 128, 16, 14336, 128, 4096, "D=8192 L=60 H=128 KV=16 F=128"),
        "70b":  (8192, 80, 128, 16, 28672, 128, 8192, "D=8192 L=80 H=128 KV=16 F=128 (dense)"),
        "100b": (10240, 84, 160, 20, 28672, 128, 8192, "D=10240 L=84 H=160 KV=20 F=128 (dense; MoE preferred)"),
        "200b": (16384, 80, 256, 32, 40960, 128, 8192, "D=16384 L=80 H=256 KV=32 F=128 (dense; MoE preferred)"),
        "500b": (20480, 120, 320, 40, 51200, 128, 8192, "D=20480 L=120 H=320 KV=40 F=128 (dense; MoE required in practice)"),
    }.items()
}


def list_presets() -> list[str]:
    return list(PRESETS)


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]


# ---------------------------------------------------------------------
# MoE presets
#
# These are architecture targets, not training claims. Counts are measured by
# constructing the real module tree on the meta device (no weights allocated),
# and both total and active-per-token parameters are reported. "active" counts
# top_k experts per MoE layer plus router/shared-expert/non-expert parameters;
# every expert still needs checkpoint and optimizer storage.
# ---------------------------------------------------------------------
def _make_moe(hidden, layers, heads, kv_heads, intermediate, embedding_size,
              context, num_experts, top_k, expert_hidden_dim, notes="", *,
              name, moe_layer_frequency=None, shared_expert=False) -> Preset:
    cfg = NeuratalConfig(
        vocab_size=262_144,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=intermediate,
        embedding_size=embedding_size,
        max_position_embeddings=context,
        num_experts=num_experts,
        num_experts_per_token=top_k,
        expert_hidden_dim=expert_hidden_dim,
        moe_layer_frequency=moe_layer_frequency,
        shared_expert=shared_expert,
    )
    breakdown = parameter_breakdown_meta(cfg)
    total = breakdown["total"]
    active = breakdown["active_per_token"]
    placement = (
        "all layers" if cfg.num_moe_layers == cfg.num_hidden_layers
        else f"{cfg.num_moe_layers}/{cfg.num_hidden_layers} layers (freq={moe_layer_frequency})"
    )
    detail = (
        f"measured total={total:,} ({total/1e9:.2f}B), "
        f"active/token={active:,} ({active/1e9:.2f}B); "
        f"experts={num_experts}, top_k={top_k}, expert_hidden={expert_hidden_dim}, "
        f"placement={placement}"
    )
    if notes:
        detail += f" — {notes}"
    return Preset(
        name=name,
        config=cfg,
        parameter_count=total,
        notes=detail,
        active_parameter_count=active,
        kind="moe",
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden_dim=expert_hidden_dim,
    )


MOE_PRESETS: dict[str, Preset] = {
    name: _make_moe(*args, name=name)
    for name, args in {
        # Base geometry matches the dense 75M preset; 8 experts, top-2.
        "small-moe": (640, 14, 10, 2, 1728, 64, 2048, 8, 2, 1728, "75M-class dense geometry"),
        "1b-moe":    (2048, 24, 32, 4, 5632, 64, 4096, 4, 2, 1408, "1B-class"),
        "7b-moe":    (4096, 32, 64, 8, 11008, 128, 4096, 8, 2, 1408, "LLaMA-2-7B-class geometry"),
        "30b-moe":   (7168, 40, 56, 8, 18432, 128, 4096, 14, 2, 2048, "30B-class"),
        "100b-moe":  (8192, 64, 64, 8, 22016, 128, 8192, 28, 2, 2048, "100B-class"),
        "200b-moe":  (12288, 72, 96, 12, 32768, 128, 8192, 32, 2, 2048, "200B-class"),
        "500b-moe":  (16384, 96, 128, 16, 44032, 128, 8192, 44, 2, 2048, "500B-class"),
    }.items()
}


def list_moe_presets() -> list[str]:
    return list(MOE_PRESETS)


def get_moe_preset(name: str) -> Preset:
    if name not in MOE_PRESETS:
        raise KeyError(f"unknown MoE preset {name!r}; available: {sorted(MOE_PRESETS)}")
    return MOE_PRESETS[name]
