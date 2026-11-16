"""NeuratalBlock: shapes, residual identity, dropout placement."""

from __future__ import annotations

import torch

from neuratal.config import NeuratalConfig
from neuratal.model.block import NeuratalBlock


class TestBlock:
    def test_output_shape(self, device):
        cfg = NeuratalConfig()
        block = NeuratalBlock(cfg, layer_idx=0).to(device)
        hidden = torch.randn(2, 5, 256, device=device)
        positions = torch.arange(5, device=device)
        out = block(hidden, positions)
        assert out.shape == hidden.shape

    def test_zero_sublayers_yield_identity(self, device):
        cfg = NeuratalConfig()
        block = NeuratalBlock(cfg, layer_idx=0).to(device).eval()
        with torch.no_grad():
            for p in block.attn.parameters():
                p.zero_()
            for p in block.mlp.parameters():
                p.zero_()
        hidden = torch.randn(2, 7, 256, device=device)
        out = block(hidden, torch.arange(7, device=device))
        assert torch.allclose(out, hidden, atol=1e-6)

    def test_residual_connections_present(self, device):
        cfg = NeuratalConfig()
        block = NeuratalBlock(cfg, layer_idx=0).to(device).eval()
        with torch.no_grad():
            block.mlp.gate_up_proj.weight.zero_()
            block.mlp.down_proj.weight.zero_()
        hidden = torch.randn(1, 4, 256, device=device)
        out = block(hidden, torch.arange(4, device=device))
        # MLP branch is zero; output = hidden + attn(...)
        attn_only = hidden + block.attn(block.attention_norm(hidden), torch.arange(4, device=device))
        assert torch.allclose(out, attn_only, atol=1e-6)

    def test_gradients_reach_both_sublayers(self):
        cfg = NeuratalConfig()
        block = NeuratalBlock(cfg, layer_idx=0)
        hidden = torch.randn(1, 3, 256, requires_grad=True)
        block(hidden, torch.arange(3)).sum().backward()
        assert all(p.grad is not None for p in block.attn.parameters())
        assert all(p.grad is not None for p in block.mlp.parameters())
        assert all(p.grad is not None for p in block.attention_norm.parameters())
        assert all(p.grad is not None for p in block.mlp_norm.parameters())
