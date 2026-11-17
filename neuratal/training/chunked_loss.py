"""Chunked cross-entropy for very large vocabularies (262,144 tokens).

Motivation: at 262k vocabulary, materializing [batch*seq, vocab] logits in
memory costs gigabytes per training step (T=1024, B=8, fp32 → 8.6 GB). This
module computes cross-entropy in position-chunks using gradient checkpointing,
so peak memory scales with the chunk — not the full vocabulary pass.

IMPORTANT: the chunking is mathematically identical to computing the loss over
all positions at once (each position's loss is independent); only the memory
profile changes. Sum-reduction is used internally and normalized by the number
of supervised tokens, which is correct for token-level language modeling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from neuratal.config import NeuratalConfig
from neuratal.model.output import OutputProjection

_IGNORE_INDEX = -100


def project_logits(hidden: torch.Tensor, output_projection: OutputProjection, embedding_weight: torch.Tensor) -> torch.Tensor:
    """Project [N, hidden] -> logits [N, vocab] honoring the factorized tied head."""
    h = output_projection.out_proj(hidden) if output_projection.out_proj is not None else hidden
    return torch.matmul(h, embedding_weight.t())


def chunked_cross_entropy(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    output_projection: OutputProjection,
    embedding_weight: torch.Tensor,
    *,
    chunk_size: int = 256,
    ignore_index: int = _IGNORE_INDEX,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (sum_loss, supervised_token_count) for hidden states [batch, seq, hidden].

    hidden and targets must be on the same device. Returns a scalar loss
    tensor (requires_grad) and the supervised token count (detached). Use
    ``.backward()`` on (loss / token_count) or scale.
    """
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be [batch, seq, hidden], got {hidden.dim()}-D")
    batch, seq, hid = hidden.shape
    flat_h = hidden.reshape(batch * seq, hid)
    flat_t = targets.reshape(batch * seq).to(torch.long)

    if flat_h.numel() == 0:
        raise ValueError("empty input")

    supervised = (flat_t != ignore_index).sum()
    if supervised == 0:
        # no supervised tokens: return zero loss that still participates in autograd
        return flat_h.sum() * 0.0, torch.zeros((), dtype=torch.long, device=hidden.device)

    total_loss = flat_h.new_zeros(())
    chunk = int(chunk_size)
    for start in range(0, flat_h.size(0), chunk):
        h_chunk = flat_h[start:start + chunk]
        t_chunk = flat_t[start:start + chunk]
        if training:
            # checkpointing recomputes logits in backward; peak memory == one chunk
            logits = torch.utils.checkpoint.checkpoint(
                project_logits, h_chunk, output_projection, embedding_weight, use_reentrant=True
            )
        else:
            logits = project_logits(h_chunk, output_projection, embedding_weight)
        total_loss = total_loss + F.cross_entropy(
            logits.float(), t_chunk, ignore_index=ignore_index, reduction="sum"
        )

    return total_loss, supervised


def chunked_cross_entropy_batch(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    output_projection: OutputProjection,
    embedding_weight: torch.Tensor,
    *,
    chunk_size: int = 256,
    ignore_index: int = _IGNORE_INDEX,
):
    """Single-forward helper returning a token-normalized loss scalar (keeps autograd)."""
    total_loss, n_tokens = chunked_cross_entropy(
        hidden, targets, output_projection, embedding_weight,
        chunk_size=chunk_size, ignore_index=ignore_index, training=True,
    )
    return total_loss / n_tokens.clamp_min(1)
