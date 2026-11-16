"""Neuratal Transformer — decoder-only, autoregressive Transformer core.

V1 scope: architecture + inference components. No training, no optimizer, no
dataset code. Tokenization/detokenization are the responsibility of
NTokenizer; token IDs are consumed verbatim (vocab_size = 262,144).
"""

from neuratal.config import NeuratalConfig
from neuratal.model.cache import CacheOverflowError, KVCache
from neuratal.model.generation import generate
from neuratal.model.model import CausalLMOutput, NeuratalModel
from neuratal.serialization.checkpoint import load_pretrained, save_pretrained

__version__ = "0.1.0"

__all__ = [
    "NeuratalConfig",
    "NeuratalModel",
    "CausalLMOutput",
    "KVCache",
    "CacheOverflowError",
    "generate",
    "save_pretrained",
    "load_pretrained",
    "__version__",
]
