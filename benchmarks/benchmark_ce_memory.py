# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Measure chunked-vs-standard CE memory on the 75M preset — PUBLISHED EVIDENCE.

Forward+backward only; NO optimizer step. Synthetic random ids.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from neuratal import NeuratalConfig, NeuratalModel
from neuratal.training.chunked_linear_ce import chunked_linear_cross_entropy
from neuratal.presets import get_preset


def standard_peak(model, ids, targets, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        hidden = model.forward_hidden(ids)
        logits = model.output_projection(hidden, model.embed_tokens.embedding_weight)
        loss = F.cross_entropy(logits.float().reshape(-1, 262144), targets.reshape(-1))
        torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    device = torch.device("cuda")
    preset = get_preset("75m")
    cfg = preset.config
    model = NeuratalModel(cfg).to(device)
    ids = torch.randint(0, 262144, (2, 512), device=device)
    targets = torch.randint(0, 262144, (2, 512), device=device)

    std = standard_peak(model, ids, targets, device)
    print(f"standard CE (no_grad) peak: {std:.2f} GiB")

    # grad-enabled path with our kernel
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    with torch.enable_grad():
        hidden = model.embed_tokens(ids)
        for layer in model.layers:
            hidden = cp.checkpoint(lambda h, l=layer: l(h, torch.arange(ids.size(1), device=device)), hidden, use_reentrant=False)
        hidden = model.final_norm(hidden)
        total, n = chunked_linear_cross_entropy(hidden, targets, model.output_projection.out_proj.weight,
                                                model.embed_tokens.embedding_weight, chunk_size=256)
        loss = total / n.clamp_min(1)
        loss.backward()
        torch.cuda.synchronize()
    print(f"chunked CE (train fwd+bwd) peak: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
