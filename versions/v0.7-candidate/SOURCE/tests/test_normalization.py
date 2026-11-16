"""RMSNorm: mathematical correctness, dtype stability, edge cases."""

from __future__ import annotations

import pytest
import torch

from neuratal.model.normalization import RMSNorm


class TestMath:
    def test_output_rms_is_one_with_unit_weight(self):
        torch.manual_seed(123)
        norm = RMSNorm(64, eps=1e-5)
        x = torch.randn(4, 7, 64) * 5.0
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_known_value(self):
        norm = RMSNorm(4, eps=1e-6)
        with torch.no_grad():
            norm.weight.fill_(2.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = norm(x)
        # reference: fp32 rms norm * weight
        rms = (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        expected = (x.float() * rms * 2.0)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_weight_scales_output(self):
        norm = RMSNorm(32)
        x = torch.randn(3, 32)
        base = norm(x)
        with torch.no_grad():
            norm.weight.mul_(3.0)
        assert torch.allclose(norm(x), base * 3.0, atol=1e-6)

    def test_wrong_last_dim_raises(self):
        norm = RMSNorm(8)
        with pytest.raises(ValueError, match="last dim"):
            norm(torch.randn(2, 4))

    def test_invalid_ctor_args(self):
        with pytest.raises(ValueError):
            RMSNorm(0)
        with pytest.raises(ValueError):
            RMSNorm(8, eps=0.0)


class TestStability:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_magnitude_inputs_do_not_overflow(self, dtype):
        norm = RMSNorm(256, eps=1e-5).to(dtype)
        x = (torch.randn(2, 5, 256) * 100.0).to(dtype)  # x^2 ~ 1e4; sum overflows fp16 easily
        out = norm(x)
        assert out.dtype == dtype
        assert torch.isfinite(out.float()).all()
        # matches fp32 reference
        ref = RMSNorm(256, eps=1e-5).to(torch.float32)(x.float())
        assert torch.allclose(out.float(), ref, atol=5e-2)  # bf16 has ~3 decimal digits

    def test_zero_input_is_finite_zero(self):
        norm = RMSNorm(16, eps=1e-5)
        out = norm(torch.zeros(2, 16))
        assert torch.isfinite(out).all()
        assert torch.equal(out, torch.zeros_like(out))

    def test_tiny_variance_input(self):
        norm = RMSNorm(16, eps=1e-5)
        x = torch.full((1, 16), 1e-4)
        out = norm(x)
        assert torch.isfinite(out).all()


class TestAutograd:
    def test_gradient_flows_to_weight_and_input(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 3, 32, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_gradient_is_finite_for_large_inputs_fp32(self):
        norm = RMSNorm(64)
        x = (torch.randn(2, 64) * 1e4).requires_grad_(True)
        norm(x).sum().backward()
        assert torch.isfinite(x.grad).all()


class TestDevices:
    def test_cuda_if_available(self, device):
        norm = RMSNorm(32).to(device)
        x = torch.randn(2, 32, device=device)
        out = norm(x)
        assert out.device == x.device
        assert torch.isfinite(out).all()
