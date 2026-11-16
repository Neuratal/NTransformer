"""Token embeddings — factorized, tied design (NTokenizer-compatible).

Structure (V1, mandatory):

    token_embedding : Embedding(vocab_size, embedding_size)   <- ONE trainable matrix
    in_proj         : Linear(embedding_size -> hidden_size, bias=False)  (omitted when F == D)

The same ``token_embedding.weight`` is reused by the output head (passed at
call time — see :mod:`neuratal.model.output`), so the matrix exists exactly
once in memory, in the state dict, and in the checkpoint.

Token IDs are the NTokenizer IDs verbatim; no remapping occurs anywhere.
"""

from __future__ import annotations

import torch
from torch import nn

from neuratal.config import NeuratalConfig

_INTEGRAL_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


class FactorizedEmbedding(nn.Module):
    """``vocab_size x embedding_size`` tied matrix + optional projection to hidden size."""

    def __init__(self, config: NeuratalConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size  # resolved int by config validation
        self.hidden_size = config.hidden_size
        self.factorized = self.embedding_size < self.hidden_size

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.in_proj = (
            nn.Linear(self.embedding_size, self.hidden_size, bias=False)
            if self.factorized
            else None
        )

    @property
    def embedding_weight(self) -> nn.Parameter:
        """The single source of truth for the vocabulary matrix (shared with the head)."""
        return self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _validate_token_ids(input_ids, self.vocab_size)
        ids = input_ids.to(torch.long)
        x = self.token_embedding(ids)
        if self.in_proj is not None:
            x = self.in_proj(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, embedding_size={self.embedding_size}, "
            f"hidden_size={self.hidden_size}, factorized={self.factorized}"
        )


def _validate_token_ids(input_ids: torch.Tensor, vocab_size: int) -> None:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}")
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2-D [batch, sequence], got shape {tuple(input_ids.shape)}")
    if input_ids.dtype not in _INTEGRAL_DTYPES:
        # floats (and float-like) are rejected to avoid silent ID truncation
        raise ValueError(f"input_ids must have an integer dtype, got {input_ids.dtype}")
    if input_ids.numel() == 0:
        return
    min_id = int(input_ids.min().item())
    max_id = int(input_ids.max().item())
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"token id out of range: min={min_id}, max={max_id}, vocab_size={vocab_size}"
        )
