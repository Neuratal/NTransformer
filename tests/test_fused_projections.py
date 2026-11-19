# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""v0.3 fused projections: math equivalence + legacy checkpoint migration."""

from __future__ import annotations

import warnings

import pytest
import torch
from safetensors.torch import save_file

from neuratal import NeuratalConfig, NeuratalModel, load_pretrained, save_pretrained
from neuratal.serialization.checkpoint import _migrate_legacy_state_dict

V = 262_144
D = 256


class TestFusedMath:
    def test_qkv_fused_equals_separate(self):
        cfg = NeuratalConfig()
        attn = NeuratalModel(cfg).layers[0].attn
        qkv = attn.qkv_proj.weight
        q_w, k_w, v_w = qkv.split([4 * 64, 2 * 64, 2 * 64], dim=0)
        hidden = torch.randn(2, 5, D)
        qkv_full = attn.qkv_proj(hidden)
        q_sep = torch.nn.functional.linear(hidden, q_w)
        k_sep = torch.nn.functional.linear(hidden, k_w)
        v_sep = torch.nn.functional.linear(hidden, v_w)
        assert torch.allclose(qkv_full[..., :256], q_sep, atol=0, rtol=0)
        assert torch.allclose(qkv_full[..., 256:384], k_sep, atol=0, rtol=0)
        assert torch.allclose(qkv_full[..., 384:], v_sep, atol=0, rtol=0)

    def test_gate_up_fused_equals_separate(self):
        cfg = NeuratalConfig()
        mlp = NeuratalModel(cfg).layers[0].mlp
        gate_w, up_w = mlp.gate_up_proj.weight.chunk(2, dim=0)
        x = torch.randn(2, 5, D)
        fused = mlp.gate_up_proj(x)
        gate_sep = torch.nn.functional.linear(x, gate_w)
        up_sep = torch.nn.functional.linear(x, up_w)
        assert torch.allclose(fused[..., :640], gate_sep, atol=0, rtol=0)
        assert torch.allclose(fused[..., 640:], up_sep, atol=0, rtol=0)

    def test_parameter_count_unchanged(self):
        model = NeuratalModel(NeuratalConfig())
        assert model.num_parameters() == 9_782_528

    def test_state_dict_has_fused_keys_only(self):
        model = NeuratalModel(NeuratalConfig())
        keys = set(model.state_dict())
        for i in range(model.config.num_hidden_layers):
            for legacy in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
                assert f"layers.{i}.attn.{legacy}.weight" not in keys
                assert f"layers.{i}.mlp.{legacy}.weight" not in keys
        assert "layers.0.attn.qkv_proj.weight" in keys
        assert "layers.0.mlp.gate_up_proj.weight" in keys

    def test_gradients_flow_to_fused_weights(self):
        model = NeuratalModel(NeuratalConfig())
        ids = torch.randint(0, V, (1, 6))
        model(ids).logits.sum().backward()
        assert model.layers[0].attn.qkv_proj.weight.grad is not None
        assert model.layers[0].mlp.gate_up_proj.weight.grad is not None


class TestLegacyMigration:
    def _legacy_dict(self, model: NeuratalModel) -> dict[str, torch.Tensor]:
        """Split fused weights back into the pre-v0.3 layout (round-trip probe)."""
        sd = {}
        for k, v in model.state_dict().items():
            if k.endswith(".attn.qkv_proj.weight"):
                q, k_, v_ = v.split([256, 128, 128], dim=0)
                prefix = k.rsplit(".qkv_proj.weight", 1)[0]
                sd[prefix + ".q_proj.weight"] = q
                sd[prefix + ".k_proj.weight"] = k_
                sd[prefix + ".v_proj.weight"] = v_
            elif k.endswith(".mlp.gate_up_proj.weight"):
                gate, up = v.split([640, 640], dim=0)
                prefix = k.rsplit(".gate_up_proj.weight", 1)[0]
                sd[prefix + ".gate_proj.weight"] = gate
                sd[prefix + ".up_proj.weight"] = up
            else:
                sd[k] = v
        return sd

    def test_migrate_concatenates_deterministically(self):
        model = NeuratalModel(NeuratalConfig())
        legacy = self._legacy_dict(model)
        with pytest.warns(UserWarning, match="migrated"):
            migrated = _migrate_legacy_state_dict(legacy, expect_fused_att=True, expect_fused_mlp=True)
        assert "layers.0.attn.qkv_proj.weight" in migrated
        assert "layers.1.mlp.gate_up_proj.weight" in migrated
        assert not any(".q_proj.weight" in k for k in migrated)

    def test_legacy_checkpoint_loads_and_matches(self, tmp_path):
        torch.manual_seed(0)
        model = NeuratalModel(NeuratalConfig()).eval()
        ids = torch.randint(0, V, (1, 8))
        with torch.no_grad():
            ref = model(ids).logits

        legacy = self._legacy_dict(model)
        save_file(legacy, str(tmp_path / "model.safetensors"))
        import json

        (tmp_path / "config.json").write_text(json.dumps({
            "format_version": 1,
            "architectures": ["NeuratalModel"],
            "config": model.config.to_dict(),
        }))
        with pytest.warns(UserWarning, match="migrated"):
            loaded = load_pretrained(tmp_path)
        loaded.eval()
        with torch.no_grad():
            out = loaded(ids).logits
        max_abs = (ref - out).abs().max().item()
        assert max_abs < 1e-5, f"legacy migration changed numerics by {max_abs}"

    def test_new_checkpoint_still_loads_without_migration_warning(self, tmp_path):
        model = NeuratalModel(NeuratalConfig())
        save_pretrained(model, tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            loaded = load_pretrained(tmp_path)
        assert loaded.config.vocab_size == V
