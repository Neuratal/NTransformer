"""Configuration system for the Neuratal Transformer.

A single :class:`NeuratalConfig` dataclass drives the whole architecture.
All values are validated in ``__post_init__``; invalid configurations raise
:class:`ValueError` with a precise message. Nothing is silently corrected.
Resolved defaults (e.g. ``intermediate_size=None``) are stored on the instance
so a serialized ``config.json`` is fully self-describing — loading never
re-derives values.

The dataclass defaults are the development/testing preset required by the
project (full NTokenizer vocabulary, 262,144 tokens). Alternative model sizes
are just different configurations; no architecture value is baked into code.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

_ALLOWED_DTYPES = ("float32", "bfloat16", "float16")
_ALLOWED_ATTENTION_IMPL = ("sdpa", "reference")
_ALLOWED_ATTENTION_TYPE = ("gqa", "gated_delta", "mla", "sliding_window")


def _round_to_multiple(value: float, multiple: int) -> int:
    return int(round(value / multiple) * multiple)


@dataclass
class NeuratalConfig:
    """Validated, serialized-safe configuration for a Neuratal decoder-only model.

    Defaults = project development/testing preset (small but full-vocabulary):
    ``V=262144, D=256, L=2, H=4, H_kv=2, F=32, I=640, ctx=2048``.
    """

    # --- Vocabulary / token compatibility (NTokenizer v0.3: strict layout) ---
    vocab_size: int = 262_144

    # --- Hidden dims ---
    hidden_size: int = 256
    # Factorization dimension F of the tied embedding. None => full-width (F = hidden_size).
    embedding_size: Optional[int] = 32
    # None => resolved to round(8/3 * hidden_size) rounded to a multiple of 256.
    intermediate_size: Optional[int] = 640

    # --- Depth ---
    num_hidden_layers: int = 2

    # --- Attention ---
    num_attention_heads: int = 4
    num_key_value_heads: Optional[int] = 2  # None => num_attention_heads
    max_position_embeddings: int = 2_048
    rope_theta: float = 10_000.0
    rotary_dim: Optional[int] = None  # None => head_dim; must be even and <= head_dim
    attention_dropout: float = 0.0
    attention_implementation: str = "sdpa"  # "reference" is a test/debug path, not for training
    attention_type: str = "gqa"             # experimental prototypes are opt-in; default is v0.4
    mla_latent_dim: Optional[int] = None    # only for attention_type="mla"; None -> 64
    swa_window: int = 512                   # only for attention_type="sliding_window"

    # --- Normalization / regularization ---
    rms_norm_eps: float = 1e-5
    hidden_dropout: float = 0.0

    # --- Embeddings / head ---
    # V1 mandates factorized tied embeddings (single shared matrix). The field exists so
    # serialized configs document the tying; setting False is an explicit error, never a
    # silent behavior change.
    tie_word_embeddings: bool = True

    # --- Runtime / inference ---
    use_cache: bool = True
    cache_capacity: Optional[int] = None  # None => max_position_embeddings

    # --- Output head memory controls ---
    fp32_logits: bool = False
    logits_chunk_size: Optional[int] = None  # project hidden states in position chunks

    # --- Precision ---
    dtype: str = "float32"

    # --- Optional generation hints (never implicit; used only as fallbacks by generate()) ---
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        self._validate_and_resolve()

    # ------------------------------------------------------------------ #
    # Validation & default resolution
    # ------------------------------------------------------------------ #
    def _validate_and_resolve(self) -> None:
        # --- integer-valued fields: exact ints only (no silent coercion of floats/bools) ---
        for name in (
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "max_position_embeddings",
        ):
            self._require_int(name, getattr(self, name))
        for name in ("embedding_size", "intermediate_size", "num_key_value_heads",
                     "rotary_dim", "cache_capacity", "logits_chunk_size",
                     "bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if value is not None:
                self._require_int(name, value)

        # --- vocab ---
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")

        # --- hidden / factorization ---
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.embedding_size is None:
            self.embedding_size = self.hidden_size  # full-width degenerate case (still tied)
        if not (1 <= self.embedding_size <= self.hidden_size):
            raise ValueError(
                f"embedding_size must satisfy 1 <= embedding_size <= hidden_size "
                f"({self.hidden_size}), got {self.embedding_size}"
            )

        # --- intermediate size ---
        if self.intermediate_size is None:
            self.intermediate_size = max(8, _round_to_multiple(8 / 3 * self.hidden_size, 256))
        if self.intermediate_size < 1:
            raise ValueError(f"intermediate_size must be >= 1, got {self.intermediate_size}")

        # --- depth ---
        if self.num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1, got {self.num_hidden_layers}")

        # --- attention geometry ---
        if self.num_attention_heads < 1:
            raise ValueError(f"num_attention_heads must be >= 1, got {self.num_attention_heads}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads "
                f"({self.num_attention_heads})"
            )
        head_dim = self.hidden_size // self.num_attention_heads

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_key_value_heads < 1:
            raise ValueError(f"num_key_value_heads must be >= 1, got {self.num_key_value_heads}")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot exceed "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        # --- context / RoPE ---
        if self.max_position_embeddings < 1:
            raise ValueError(f"max_position_embeddings must be >= 1, got {self.max_position_embeddings}")
        if not (isinstance(self.rope_theta, (int, float)) and math.isfinite(self.rope_theta)) or self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be a positive finite number, got {self.rope_theta}")
        if self.rotary_dim is None:
            self.rotary_dim = head_dim
        if self.rotary_dim < 2 or self.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even and >= 2, got {self.rotary_dim}")
        if self.rotary_dim > head_dim:
            raise ValueError(
                f"rotary_dim ({self.rotary_dim}) cannot exceed head_dim ({head_dim})"
            )

        # --- normalization ---
        if not (isinstance(self.rms_norm_eps, (int, float)) and math.isfinite(self.rms_norm_eps)) or self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be a positive finite number, got {self.rms_norm_eps}")

        # --- dropouts ---
        for name in ("attention_dropout", "hidden_dropout"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value)) or not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must satisfy 0.0 <= p < 1.0, got {value}")

        # --- tying (mandatory in V1) ---
        if not isinstance(self.tie_word_embeddings, bool):
            raise ValueError(f"tie_word_embeddings must be a bool, got {self.tie_word_embeddings!r}")
        if not self.tie_word_embeddings:
            raise ValueError(
                "tie_word_embeddings=False is not supported in this architecture: V1 mandates "
                "factorized tied embeddings with a single shared matrix (no untied path is "
                "silently provided). Set tie_word_embeddings=True."
            )

        if not isinstance(self.swa_window, int) or isinstance(self.swa_window, bool):
            raise ValueError(f"swa_window must be an integer, got {self.swa_window!r}")
        if self.swa_window < 1:
            raise ValueError(f"swa_window must be >= 1, got {self.swa_window}")
        if self.swa_window > self.max_position_embeddings:
            raise ValueError(
                f"swa_window ({self.swa_window}) cannot exceed max_position_embeddings "
                f"({self.max_position_embeddings})"
            )

        if self.attention_type not in _ALLOWED_ATTENTION_TYPE:
            raise ValueError(
                f"attention_type must be one of {_ALLOWED_ATTENTION_TYPE}, "
                f"got {self.attention_type!r}"
            )

        # --- attention implementation ---
        if self.attention_implementation not in _ALLOWED_ATTENTION_IMPL:
            raise ValueError(
                f"attention_implementation must be one of {_ALLOWED_ATTENTION_IMPL}, "
                f"got {self.attention_implementation!r}"
            )

        # --- precision ---
        if self.dtype not in _ALLOWED_DTYPES:
            raise ValueError(f"dtype must be one of {_ALLOWED_DTYPES}, got {self.dtype!r}")
        if not isinstance(self.fp32_logits, bool):
            raise ValueError(f"fp32_logits must be a bool, got {self.fp32_logits!r}")

        # --- cache ---
        if not isinstance(self.use_cache, bool):
            raise ValueError(f"use_cache must be a bool, got {self.use_cache!r}")
        if self.cache_capacity is None:
            self.cache_capacity = self.max_position_embeddings
        if self.cache_capacity < 1:
            raise ValueError(f"cache_capacity must be >= 1, got {self.cache_capacity}")

        # --- logits chunking ---
        if self.logits_chunk_size is not None and self.logits_chunk_size < 1:
            raise ValueError(f"logits_chunk_size must be >= 1 when set, got {self.logits_chunk_size}")

        # --- special-token hints ---
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if value is not None and not (0 <= value < self.vocab_size):
                raise ValueError(
                    f"{name} must lie in [0, vocab_size={self.vocab_size}), got {value}"
                )

    @staticmethod
    def _require_int(name: str, value: Any) -> None:
        # bool is a subclass of int; accept it for int fields only via explicit check here
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Resolved, JSON-serializable dict (round-trip safe via :meth:`from_dict`)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuratalConfig":
        if not isinstance(data, dict):
            raise ValueError(f"config payload must be a dict, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        cfg = cls(**data)  # __post_init__ revalidates everything
        return cfg

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_query_groups(self) -> int:
        assert self.num_key_value_heads is not None  # resolved in post_init
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_factorized(self) -> bool:
        """True when the tied embedding uses the low-rank (F < D) factorization."""
        assert self.embedding_size is not None  # resolved in post_init
        return self.embedding_size < self.hidden_size
