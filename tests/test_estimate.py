# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Memory estimator: formulas produce sane, monotone, documented outputs."""

from __future__ import annotations

from neuratal.estimate import estimate
from neuratal.presets import get_preset
from neuratal.config import NeuratalConfig


def test_estimator_monotone_in_params():
    small = get_preset("50m")
    big = get_preset("5.6b")
    e_small = estimate(small.config, batch_size=2, seq_length=128, name="50m")
    e_big = estimate(big.config, batch_size=2, seq_length=128, name="5.6b")
    assert e_big.parameter_bytes > e_small.parameter_bytes * 50


def test_logits_chunk_counted():
    cfg = NeuratalConfig(vocab_size=262_144, hidden_size=64, num_hidden_layers=1,
                         num_attention_heads=1, num_key_value_heads=1,
                         embedding_size=8, intermediate_size=128)
    e = estimate(cfg, batch_size=4, seq_length=512, logits_chunk=128, name="tiny")
    # chunked logits workspace is charged for one chunk in fp32 only:
    assert e.logits_bytes == 4 * 128 * 262_144 * 4
    full_fp32 = 4 * 512 * 262_144 * 4
    assert e.logits_bytes == full_fp32 // 4


def test_notes_present():
    cfg = NeuratalConfig(vocab_size=262_144, hidden_size=64, num_hidden_layers=1,
                         num_attention_heads=1, num_key_value_heads=1,
                         embedding_size=8, intermediate_size=128)
    e = estimate(cfg, batch_size=1, seq_length=16, name="x")
    assert e.notes
    assert any("AdamW" in n for n in e.notes)
    assert any("ESTIMATE" in n for n in e.notes)
