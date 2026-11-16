"""SwiGLU MLP: exact math, shapes, gradients."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neuratal.config import NeuratalConfig
from neuratal.model.mlp import SwiGLU


class TestMath:
    def test_exact_swiglu_formula(self):
        cfg = NeuratalConfig(hidden_size=256, intermediate_size=640)
        torch.manual_seed(7)
        mlp = SwiGLU(cfg)
        x = torch.randn(2, 4, 256)
        expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
        assert torch.allclose(mlp(x), expected, atol=0, rtol=0)

    def test_silu_gating_not_relu(self):
        cfg = NeuratalConfig(hidden_size=256, intermediate_size=64)
        mlp = SwiGLU(cfg)
        x = torch.randn(1, 1, 256)  # random negatives exist with prob ~1
        wrong = mlp.down_proj(F.relu(mlp.gate_proj(x)) * mlp.up_proj(x))
        assert not torch.allclose(mlp(x), wrong, atol=1e-6)

    def test_shapes_and_intermediate_size(self):
        cfg = NeuratalConfig(hidden_size=256, intermediate_size=640)
        mlp = SwiGLU(cfg)
        assert mlp.gate_proj.weight.shape == (640, 256)
        assert mlp.up_proj.weight.shape == (640, 256)
        assert mlp.down_proj.weight.shape == (256, 640)
        out = mlp(torch.randn(3, 5, 256))
        assert out.shape == (3, 5, 256)

    def test_wrong_last_dim_rejected(self):
        cfg = NeuratalConfig(hidden_size=256, intermediate_size=640)
        mlp = SwiGLU(cfg)
        try:
            mlp(torch.randn(1, 2, 128))
        except Exception as exc:
            assert isinstance(exc, (ValueError, RuntimeError))
        else:
            raise AssertionError("expected an error for wrong input size")


class TestGradientsAndDevices:
    def test_gradients_flow(self, device):
        cfg = NeuratalConfig(hidden_size=256, intermediate_size=640)
        mlp = SwiGLU(cfg).to(device)
        x = torch.randn(2, 4, 256, device=device, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None
        for name, p in mlp.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name

    def test_no_biases_anywhere(self):
        cfg = NeuratalConfig(hidden_size=256, intermediate_size=640)
        mlp = SwiGLU(cfg)
        assert all(p.dim() == 2 for p in mlp.parameters())
