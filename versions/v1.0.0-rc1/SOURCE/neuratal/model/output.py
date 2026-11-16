"""Output projection for the 262k-token vocabulary.

Tied design (mandatory in V1): this module owns **no** vocabulary parameter.
The shared embedding matrix is passed at call time by the model, so the
checkpoint and the parameter count always contain it exactly once.

Factorized path (embedding_size < hidden_size):

    h = hidden_states @ W_out^T          # W_out: hidden_size -> embedding_size
    logits = h @ E^T                     # E: [vocab_size, embedding_size] (the tied matrix)

Full-width path (embedding_size == hidden_size): ``W_out`` is omitted and
``logits = hidden_states @ E^T``.

Memory engineering (exact math, no approximation)
-------------------------------------------------
``logits_chunk_size`` projects the sequence dimension in independent chunks
and concatenates the results. Each chunk is an exact per-row matmul, so the
concatenated logits are mathematically identical to the unchunked
computation (bitwise equality is not guaranteed: GEMM kernels may tile
differently). This reduces only the
*transient* workspace inside the projection, **not** the memory required to
hold the fully materialized ``[batch, sequence, vocab_size]`` logits tensor —
at B=8, T=512, fp32 that tensor is ~4.3 GB. Reducing logits *storage* (fused
cross-entropy) is a training-pipeline concern and is deliberately out of scope.
``fp32_logits=True`` computes both projections in float32 (better precision,
2x transient memory); logits are otherwise returned in the model dtype.
"""

from __future__ import annotations

import torch
from torch import nn

from neuratal.config import NeuratalConfig


class OutputProjection(nn.Module):
    def __init__(self, config: NeuratalConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.factorized = self.embedding_size < self.hidden_size
        self.fp32_logits = bool(config.fp32_logits)
        self.logits_chunk_size = config.logits_chunk_size

        self.out_proj = (
            nn.Linear(self.hidden_size, self.embedding_size, bias=False)
            if self.factorized
            else None
        )

    def forward(self, hidden_states: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be 3-D [batch, seq, hidden], got {hidden_states.dim()}-D")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_states.size(-1)}")
        _validate_embedding_weight(embedding_weight, self.vocab_size, self.embedding_size)

        if self.fp32_logits:
            weight = embedding_weight.float()
            h = hidden_states.float()
            if self.out_proj is not None:
                h = torch.nn.functional.linear(h, self.out_proj.weight.float())
        else:
            weight = embedding_weight
            h = self.out_proj(hidden_states) if self.out_proj is not None else hidden_states

        chunk = self.logits_chunk_size
        if chunk is None or h.size(1) <= chunk:
            return torch.matmul(h, weight.t())

        pieces = [torch.matmul(part, weight.t()) for part in h.split(chunk, dim=1)]
        return torch.cat(pieces, dim=1)

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, embedding_size={self.embedding_size}, "
            f"fp32_logits={self.fp32_logits}, logits_chunk_size={self.logits_chunk_size}"
        )


def _validate_embedding_weight(weight: torch.Tensor, vocab_size: int, embedding_size: int) -> None:
    if weight.dim() != 2:
        raise ValueError(f"embedding_weight must be 2-D [vocab_size, embedding_size], got {weight.dim()}-D")
    if weight.shape != (vocab_size, embedding_size):
        raise ValueError(
            f"embedding_weight shape mismatch: expected ({vocab_size}, {embedding_size}), "
            f"got {tuple(weight.shape)}"
        )
