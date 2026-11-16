"""Presets: every preset builds, counts match, get_preset works, meta-device OK."""

from __future__ import annotations

import pytest

from neuratal import PRESETS, get_preset, list_presets, parameter_count_meta


EXPECTED_PARAMS = {
    "50m": 47_853_056,
    "75m": 77_088_896,
    "100m": 98_096_896,
    "500m": 496_760_320,
    "1b": 1_074_104_320,
    "3b": 2_960_139_008,
    "7b": 5_571_350_528,
    "13b": 11_915_686_912,
    "30b": 30_235_631_616,
    "70b": 68_488_011_776,
    "100b": 93_843_515_392,
    "200b": 209_420_042_240,
    "500b": 490_777_300_992,
}


def test_all_presets_build_on_meta():
    for name in list_presets():
        n = get_preset(name).parameter_count
        assert n > 1_000_000


def test_param_counts_exact():
    for name, expected in EXPECTED_PARAMS.items():
        actual = parameter_count_meta(get_preset(name).config)
        assert actual == expected, f"{name}: expected {expected:,}, got {actual:,}"


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError, match="unknown preset"):
        get_preset("999b")
