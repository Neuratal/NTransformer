# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Configuration validation: every rule raises a precise error; nothing is silently fixed."""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest

from neuratal.config import NeuratalConfig


class TestDevelopmentPreset:
    def test_defaults_are_the_approved_development_preset(self):
        cfg = NeuratalConfig()
        assert cfg.vocab_size == 262_144
        assert cfg.hidden_size == 256
        assert cfg.num_hidden_layers == 2
        assert cfg.num_attention_heads == 4
        assert cfg.num_key_value_heads == 2
        assert cfg.embedding_size == 32
        assert cfg.intermediate_size == 640
        assert cfg.max_position_embeddings == 2_048
        assert cfg.rope_theta == 10_000.0
        assert cfg.rms_norm_eps == 1e-5
        assert cfg.attention_dropout == 0.0
        assert cfg.hidden_dropout == 0.0
        assert cfg.tie_word_embeddings is True
        assert cfg.use_cache is True
        assert cfg.dtype == "float32"

    def test_resolved_values(self):
        cfg = NeuratalConfig()
        assert cfg.head_dim == 64
        assert cfg.num_query_groups == 2
        assert cfg.rotary_dim == 64
        assert cfg.cache_capacity == 2_048
        assert cfg.is_factorized is True


class TestValidationErrors:
    def test_vocab_zero(self):
        with pytest.raises(ValueError, match="vocab_size"):
            NeuratalConfig(vocab_size=0)

    def test_hidden_size_zero(self):
        with pytest.raises(ValueError, match="hidden_size"):
            NeuratalConfig(hidden_size=0)

    def test_embedding_size_above_hidden(self):
        with pytest.raises(ValueError, match="embedding_size"):
            NeuratalConfig(embedding_size=257)

    def test_embedding_size_zero(self):
        with pytest.raises(ValueError, match="embedding_size"):
            NeuratalConfig(embedding_size=0)

    def test_heads_not_dividing_hidden(self):
        with pytest.raises(ValueError, match="divisible"):
            NeuratalConfig(hidden_size=250, num_attention_heads=4)

    def test_kv_heads_exceeding_query_heads(self):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            NeuratalConfig(num_key_value_heads=5)

    def test_heads_not_divisible_by_kv_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            NeuratalConfig(num_attention_heads=4, num_key_value_heads=3)

    def test_zero_kv_heads(self):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            NeuratalConfig(num_key_value_heads=0)

    def test_odd_rotary_dim(self):
        with pytest.raises(ValueError, match="rotary_dim"):
            NeuratalConfig(rotary_dim=63)

    def test_rotary_dim_above_head_dim(self):
        with pytest.raises(ValueError, match="rotary_dim"):
            NeuratalConfig(rotary_dim=65)

    def test_nonpositive_rope_theta(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="rope_theta"):
                NeuratalConfig(rope_theta=bad)

    def test_nonpositive_rms_norm_eps(self):
        for bad in (0.0, -1e-6, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="rms_norm_eps"):
                NeuratalConfig(rms_norm_eps=bad)

    def test_dropout_out_of_range(self):
        for bad in (-0.1, 1.0, 1.5, float("inf")):
            with pytest.raises(ValueError, match="attention_dropout"):
                NeuratalConfig(attention_dropout=bad)
            with pytest.raises(ValueError, match="hidden_dropout"):
                NeuratalConfig(hidden_dropout=bad)

    def test_untied_embeddings_rejected_explicitly(self):
        with pytest.raises(ValueError, match="tie_word_embeddings"):
            NeuratalConfig(tie_word_embeddings=False)

    def test_unknown_attention_implementation(self):
        with pytest.raises(ValueError, match="attention_implementation"):
            NeuratalConfig(attention_implementation="flash3")

    def test_unknown_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            NeuratalConfig(dtype="float64")

    def test_cache_capacity_zero(self):
        with pytest.raises(ValueError, match="cache_capacity"):
            NeuratalConfig(cache_capacity=0)

    def test_logits_chunk_size_zero(self):
        with pytest.raises(ValueError, match="logits_chunk_size"):
            NeuratalConfig(logits_chunk_size=0)

    def test_special_token_out_of_range(self):
        with pytest.raises(ValueError, match="eos_token_id"):
            NeuratalConfig(eos_token_id=262_144)
        with pytest.raises(ValueError, match="bos_token_id"):
            NeuratalConfig(bos_token_id=-1)
        with pytest.raises(ValueError, match="pad_token_id"):
            NeuratalConfig(pad_token_id=262_144)

    def test_float_passed_for_int_field(self):
        with pytest.raises(ValueError, match="vocab_size"):
            NeuratalConfig(vocab_size=262144.0)  # type: ignore[arg-type]

    def test_bool_passed_for_int_field(self):
        with pytest.raises(ValueError, match="vocab_size"):
            NeuratalConfig(vocab_size=True)  # type: ignore[arg-type]

    def test_zero_layers(self):
        with pytest.raises(ValueError, match="num_hidden_layers"):
            NeuratalConfig(num_hidden_layers=0)

    def test_max_position_embeddings_zero(self):
        with pytest.raises(ValueError, match="max_position_embeddings"):
            NeuratalConfig(max_position_embeddings=0)

    def test_intermediate_size_zero(self):
        with pytest.raises(ValueError, match="intermediate_size"):
            NeuratalConfig(intermediate_size=0)


class TestDefaultResolution:
    def test_none_intermediate_size_resolves_to_multiple_of_256(self):
        cfg = NeuratalConfig(intermediate_size=None)
        assert cfg.intermediate_size is not None
        assert cfg.intermediate_size % 256 == 0
        assert cfg.intermediate_size >= 8
        # 8/3 * 256 = 682.67 -> round(682.67/256)=3 -> 768
        assert cfg.intermediate_size == 768

    def test_none_embedding_size_resolves_to_hidden(self):
        cfg = NeuratalConfig(embedding_size=None)
        assert cfg.embedding_size == cfg.hidden_size
        assert cfg.is_factorized is False

    def test_none_num_key_value_heads_resolves_to_num_heads(self):
        cfg = NeuratalConfig(num_key_value_heads=None)
        assert cfg.num_key_value_heads == cfg.num_attention_heads

    def test_none_cache_capacity_resolves_to_max_position(self):
        cfg = NeuratalConfig(cache_capacity=None, max_position_embeddings=777)
        assert cfg.cache_capacity == 777

    def test_none_rotary_dim_resolves_to_head_dim(self):
        cfg = NeuratalConfig(rotary_dim=None)
        assert cfg.rotary_dim == cfg.head_dim


class TestSerializationRoundTrip:
    def test_to_dict_from_dict_roundtrip(self):
        cfg = NeuratalConfig()
        rebuilt = NeuratalConfig.from_dict(cfg.to_dict())
        assert asdict(rebuilt) == asdict(cfg)

    def test_to_dict_contains_only_json_safe_scalars(self):
        import json

        payload = json.dumps(NeuratalConfig().to_dict())
        assert json.loads(payload)["vocab_size"] == 262_144

    def test_from_dict_rejects_unknown_keys(self):
        data = NeuratalConfig().to_dict()
        data["flash_attention"] = True
        with pytest.raises(ValueError, match="unknown config keys"):
            NeuratalConfig.from_dict(data)

    def test_from_dict_revalidates(self):
        data = NeuratalConfig().to_dict()
        data["vocab_size"] = 0
        with pytest.raises(ValueError, match="vocab_size"):
            NeuratalConfig.from_dict(data)

    def test_from_dict_rejects_non_dict(self):
        with pytest.raises(ValueError):
            NeuratalConfig.from_dict([("vocab_size", 262144)])  # type: ignore[arg-type]

    def test_nan_dropout_rejected(self):
        with pytest.raises(ValueError):
            NeuratalConfig(attention_dropout=math.nan)
