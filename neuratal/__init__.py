# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Neuratal Transformer — decoder-only, autoregressive Transformer core.

v1.1.0rc1: architecture, inference components, training engine and full
checkpoint management. Tokenization/detokenization are the responsibility of
NTokenizer; token IDs are consumed verbatim.
"""

from neuratal.config import NeuratalConfig
from neuratal.model.cache import CacheOverflowError, KVCache
from neuratal.model.generation import generate
from neuratal.model.model import CausalLMOutput, NeuratalModel
from neuratal.model.moe import MoEFFN, MoEStats
from neuratal.serialization.checkpoint import load_pretrained, save_pretrained
from neuratal.serialization.sharded import (
    load_pretrained_sharded,
    plan_shards,
    save_pretrained_sharded,
)

# Preset machinery constructs every scale preset (meta device) on first use.
# Defer it to first attribute access so `import neuratal` stays fast; the public
# names below are resolved lazily via PEP 562 __getattr__.
_PRESET_EXPORTS = {
    "MOE_PRESETS",
    "PRESETS",
    "active_parameter_count_meta",
    "get_moe_preset",
    "get_preset",
    "list_moe_presets",
    "list_presets",
    "parameter_breakdown_meta",
    "parameter_count_meta",
}


def __getattr__(name: str):
    if name in _PRESET_EXPORTS:
        from neuratal import presets

        return getattr(presets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PRESET_EXPORTS)

__version__ = "1.4.0"

__all__ = [
    "NeuratalConfig",
    "NeuratalModel",
    "CausalLMOutput",
    "KVCache",
    "CacheOverflowError",
    "generate",
    "save_pretrained",
    "load_pretrained",
    "save_pretrained_sharded",
    "load_pretrained_sharded",
    "plan_shards",
    "list_presets",
    "get_preset",
    "parameter_count_meta",
    "PRESETS",
    "MoEFFN",
    "MoEStats",
    "MOE_PRESETS",
    "list_moe_presets",
    "get_moe_preset",
    "parameter_breakdown_meta",
    "active_parameter_count_meta",
    "__version__",
]
