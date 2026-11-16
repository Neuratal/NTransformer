"""Neuratal Transformer — decoder-only, autoregressive Transformer core.

v1.0 engineering candidate: architecture, inference components, training engine
and full checkpoint management. Tokenization/detokenization are the
responsibility of NTokenizer; token IDs are consumed verbatim.
"""

from neuratal.config import NeuratalConfig
from neuratal.presets import PRESETS, get_preset, list_presets, parameter_count_meta
from neuratal.model.cache import CacheOverflowError, KVCache
from neuratal.model.generation import generate
from neuratal.model.model import CausalLMOutput, NeuratalModel
from neuratal.serialization.checkpoint import load_pretrained, save_pretrained

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
    "list_presets",
    "get_preset",
    "parameter_count_meta",
    "PRESETS",
    "__version__",
]
