"""MoE presets and memory-estimator accounting."""

from __future__ import annotations

import pytest

from neuratal import (
    MOE_PRESETS,
    active_parameter_count_meta,
    get_moe_preset,
    get_preset,
    list_moe_presets,
    parameter_breakdown_meta,
)
from neuratal.estimate import estimate


EXPECTED_TOTAL = {
    "small-moe": 402_301_056,
    "1b-moe": 1_074_300_928,
    "5.7b-moe": 5_673_062_400,
    "30b-moe": 29_400_112_128,
    "100b-moe": 99_909_378_048,
    "200b-moe": 198_474_149_888,
    "500b-moe": 483_293_937_664,
}
EXPECTED_ACTIVE = {
    "small-moe": 123_609_216,
    "1b-moe": 659_064_832,
    "5.7b-moe": 2_351_173_632,
    "30b-moe": 8_260_819_968,
    "100b-moe": 16_157_515_776,
    "200b-moe": 35_399_610_368,
    "500b-moe": 77_419_528_192,
}


class TestMoEPresets:
    def test_expected_names(self):
        assert list_moe_presets() == list(EXPECTED_TOTAL)

    def test_total_and_active_counts_exact(self):
        for name in EXPECTED_TOTAL:
            preset = get_moe_preset(name)
            assert preset.parameter_count == EXPECTED_TOTAL[name], name
            assert preset.active_parameter_count == EXPECTED_ACTIVE[name], name
            assert preset.is_moe
            assert preset.active_parameter_count < preset.parameter_count

    def test_meta_breakdown_matches_preset(self):
        for name in ("small-moe", "1b-moe", "5.7b-moe"):
            preset = get_moe_preset(name)
            breakdown = parameter_breakdown_meta(preset.config)
            assert breakdown["total"] == preset.parameter_count
            assert breakdown["active_per_token"] == preset.active_parameter_count
            assert breakdown["moe_router"] > 0
            assert breakdown["moe_experts"] == (
                preset.config.num_moe_layers * preset.config.num_experts
                * 3 * preset.config.hidden_size * preset.config.expert_hidden_dim
            )

    def test_dense_presets_keep_active_equal_total(self):
        for name in ("50m", "75m", "5.6b"):
            p = get_preset(name)
            assert p.kind == "dense"
            assert p.active_parameter_count == p.parameter_count
            assert active_parameter_count_meta(p.config) == p.parameter_count

    def test_unknown_moe_preset(self):
        with pytest.raises(KeyError, match="unknown MoE preset"):
            get_moe_preset("42b-moe")


class TestMoEEstimate:
    def test_moe_accounting_splits_total_and_active(self):
        preset = get_moe_preset("small-moe")
        e = estimate(preset.config, batch_size=1, seq_length=128, name="small-moe")
        assert e.total_parameter_count == preset.parameter_count
        assert e.active_parameter_count == preset.active_parameter_count
        assert e.expert_parameter_bytes > 0
        assert e.router_parameter_bytes > 0
        assert e.active_parameter_bytes < e.parameter_bytes
        assert e.activation_bytes > 0
        assert e.expert_activation_bytes > 0
        assert any("ALL experts" in n for n in e.notes)
        assert any("expert parallelism" in n for n in e.notes)

    def test_expert_hidden_and_topk_reduce_activation_vs_all_experts(self):
        # Activation accounting charges top_k experts, not all num_experts.
        preset = get_moe_preset("small-moe")
        top2 = estimate(preset.config, batch_size=1, seq_length=64, name="top2")
        top1_cfg = preset.config.to_dict()
        top1_cfg["num_experts_per_token"] = 1
        from neuratal.config import NeuratalConfig
        top1 = estimate(NeuratalConfig.from_dict(top1_cfg), batch_size=1, seq_length=64, name="top1")
        assert top1.expert_activation_bytes < top2.expert_activation_bytes

    def test_distributed_strategies_are_accounting_models(self):
        preset = get_moe_preset("1b-moe")
        single = estimate(preset.config, batch_size=1, seq_length=32, name="single")
        zero3 = estimate(preset.config, batch_size=1, seq_length=32, name="zero3",
                         world_size=8, strategy="zero3")
        assert zero3.parameter_bytes < single.parameter_bytes
        assert zero3.optimizer_state_bytes < single.optimizer_state_bytes
        assert zero3.communication_bytes > 0
        assert any("not an implemented" in n for n in zero3.notes)
        with pytest.raises(ValueError, match="not implemented"):
            estimate(preset.config, strategy="tensor")
        with pytest.raises(ValueError, match="world_size"):
            estimate(preset.config, world_size=0)

    def test_report_contains_moe_lines(self):
        preset = get_moe_preset("small-moe")
        text = estimate(preset.config, batch_size=1, seq_length=16, name="moe").report()
        assert "total params" in text
        assert "active/token" in text
        assert "expert bytes" in text
