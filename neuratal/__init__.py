"""Neuratal Transformer — decoder-only, autoregressive Transformer core.

v1.0 engineering candidate: architecture, inference components, training engine
and full checkpoint management. Tokenization/detokenization are the
responsibility of NTokenizer; token IDs are consumed verbatim.
"""

from neuratal.config import NeuratalConfig
from neuratal.presets import (
    MOE_PRESETS,
    PRESETS,
    active_parameter_count_meta,
    get_moe_preset,
    get_preset,
    list_moe_presets,
    list_presets,
    parameter_breakdown_meta,
    parameter_count_meta,
)
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

__version__ = "1.0.0rc1"

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
