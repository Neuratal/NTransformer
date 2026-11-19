# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
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
_ALLOWED_ATTENTION_TYPE = ("gqa", "gated_delta")
_ALLOWED_ROUTER_TYPES = ("topk",)


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
    attention_type: str = "gqa"             # "gated_delta" is the experimental GDN prototype (opt-in)

    # --- Mixture-of-Experts (opt-in; num_experts=0 keeps the dense v0.4 path) ---
    # Every expert is an independent SwiGLU MLP. Only the layers listed in
    # ``moe_layer_indices`` (or selected by ``moe_layer_frequency``) are MoE;
    # the remaining layers stay dense, so dense/MoE mixtures are first-class.
    num_experts: int = 0
    num_experts_per_token: int = 2       # top-k routing; ignored when num_experts == 0
    expert_hidden_dim: Optional[int] = None  # None => intermediate_size
    router_type: str = "topk"            # softmax top-k (the reference implementation)
    capacity_factor: Optional[float] = None  # None => no token dropping
    min_capacity: int = 0
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    moe_layer_indices: Optional[list[int]] = None  # resolved at construction
    moe_layer_frequency: Optional[int] = None       # layer_idx % frequency == 0
    shared_expert: bool = False
    shared_expert_hidden_dim: Optional[int] = None  # None => expert_hidden_dim

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
    # Recompute block activations in backward instead of storing them.
    # Changes memory profile only, never the math (outputs and gradients are
    # identical up to floating-point nondeterminism of the recomputed ops).
    gradient_checkpointing: bool = False

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
            "max_position_embeddings", "num_experts", "num_experts_per_token",
            "min_capacity",
        ):
            self._require_int(name, getattr(self, name))
        for name in ("embedding_size", "intermediate_size", "num_key_value_heads",
                     "rotary_dim", "cache_capacity", "logits_chunk_size",
                     "expert_hidden_dim", "shared_expert_hidden_dim",
                     "moe_layer_frequency", "bos_token_id", "eos_token_id", "pad_token_id"):
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

        # --- Mixture-of-Experts ---
        # num_experts == 0 is the dense v0.4 path.  A one-expert "MoE" adds a
        # router with no routing choice; reject it instead of silently building
        # a slower dense layer.
        if self.num_experts != 0 and self.num_experts < 2:
            raise ValueError(
                f"num_experts must be 0 (dense) or >= 2 (MoE), got {self.num_experts}"
            )
        self._resolve_moe_layers()

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
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError(f"gradient_checkpointing must be a bool, got {self.gradient_checkpointing!r}")
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

    def _resolve_moe_layers(self) -> None:
        """Validate MoE knobs and resolve the per-layer MoE placement.

        Placement is deterministic and serialized: either an explicit
        ``moe_layer_indices`` list, or ``moe_layer_frequency=k`` (selects
        layers 0, k, 2k, ...), or — when neither is given — every layer.
        Dense configs store an empty resolved list, so ``config.json`` always
        says exactly which layers are MoE.
        """
        if self.router_type not in _ALLOWED_ROUTER_TYPES:
            raise ValueError(
                f"router_type must be one of {_ALLOWED_ROUTER_TYPES}, got {self.router_type!r}"
            )

        if not isinstance(self.shared_expert, bool):
            raise ValueError(f"shared_expert must be a bool, got {self.shared_expert!r}")

        if self.num_experts > 0:
            if not (1 <= self.num_experts_per_token <= self.num_experts):
                raise ValueError(
                    f"num_experts_per_token must satisfy 1 <= top_k <= num_experts "
                    f"({self.num_experts}), got {self.num_experts_per_token}"
                )
        elif self.num_experts_per_token < 1:
            raise ValueError(
                f"num_experts_per_token must be >= 1, got {self.num_experts_per_token}"
            )

        if self.capacity_factor is not None:
            cf = self.capacity_factor
            if not (isinstance(cf, (int, float)) and not isinstance(cf, bool)
                    and math.isfinite(cf) and cf > 0):
                raise ValueError(f"capacity_factor must be None or a positive finite number, got {cf!r}")

        if self.min_capacity < 0:
            raise ValueError(f"min_capacity must be >= 0, got {self.min_capacity}")

        for name in ("router_aux_loss_coef", "router_z_loss_coef"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")

        requested_indices = self.moe_layer_indices
        if self.moe_layer_frequency is not None and self.moe_layer_frequency < 1:
            raise ValueError(
                f"moe_layer_frequency must be >= 1 when set, got {self.moe_layer_frequency}"
            )
        if self.moe_layer_frequency is not None and requested_indices is not None:
            raise ValueError(
                "set either moe_layer_indices or moe_layer_frequency, not both"
            )

        if self.num_experts == 0:
            if requested_indices not in (None, [], ()):
                raise ValueError("moe_layer_indices is only valid when num_experts >= 2")
            if self.moe_layer_frequency is not None:
                raise ValueError("moe_layer_frequency is only valid when num_experts >= 2")
            if self.shared_expert:
                raise ValueError("shared_expert=True requires num_experts >= 2")
            self.moe_layer_indices = []
        else:
            if requested_indices is None:
                if self.moe_layer_frequency is None:
                    resolved = list(range(self.num_hidden_layers))
                else:
                    resolved = list(range(0, self.num_hidden_layers, self.moe_layer_frequency))
            else:
                if not isinstance(requested_indices, (list, tuple)):
                    raise ValueError(
                        "moe_layer_indices must be a list/tuple of layer indices, "
                        f"got {type(requested_indices).__name__}"
                    )
                resolved = []
                for idx in requested_indices:
                    self._require_int("moe_layer_indices entry", idx)
                    if not (0 <= idx < self.num_hidden_layers):
                        raise ValueError(
                            f"moe_layer_indices entry {idx} out of range "
                            f"[0, {self.num_hidden_layers})"
                        )
                    resolved.append(int(idx))
            resolved = sorted(set(resolved))
            if not resolved:
                raise ValueError("MoE is enabled but no MoE layer was selected")
            self.moe_layer_indices = resolved

        if self.expert_hidden_dim is None:
            self.expert_hidden_dim = self.intermediate_size
        if self.expert_hidden_dim < 1:
            raise ValueError(f"expert_hidden_dim must be >= 1, got {self.expert_hidden_dim}")
        if self.shared_expert_hidden_dim is None:
            self.shared_expert_hidden_dim = self.expert_hidden_dim
        if self.shared_expert_hidden_dim < 1:
            raise ValueError(
                f"shared_expert_hidden_dim must be >= 1, got {self.shared_expert_hidden_dim}"
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

    @property
    def is_moe(self) -> bool:
        """True when this config enables Mixture-of-Experts FFN layers."""
        return self.num_experts >= 2

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layer_indices or [])
