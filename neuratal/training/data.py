"""Dataset protocol for tokenized, packed sequences.

Does NOT download or read any real corpus by itself. Synthetic corpus
generators are provided for tests only (never for actual training).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class TokenDataset(Dataset):
    """Pre-tokenized uint32/int64 ids, packed into fixed-length sequences.

    Packs multiple documents into one sequence greedily (standard practice),
    does NOT pad (zero-token waste eliminated). BOS/EOS markers are the
    caller's responsibility to insert at pack time; this class only stores ids.
    """

    def __init__(self, token_ids: torch.Tensor, sequence_length: int):
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1-D tensor of token ids")
        if sequence_length < 2:
            raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")
        self.data = token_ids.to(torch.int64)
        self.sequence_length = sequence_length
        self.num_sequences = (len(self.data) - 1) // sequence_length
        if self.num_sequences < 1:
            raise ValueError("corpus too small for a single sequence")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= self.num_sequences:
            raise IndexError(idx)
        start = idx * self.sequence_length
        block = self.data[start:start + self.sequence_length + 1]
        return {
            "input_ids": block[:-1],
            "labels": block[1:],
        }


class SyntheticTokenDataset(Dataset):
    """Deterministic pseudo-random token stream — TESTS ONLY, never real training.

    Produces valid NTokenizer vocabulary ids, deterministic from (seed, idx).
    """

    def __init__(self, num_sequences: int, sequence_length: int, vocab_size: int = 262_144,
                 seed: int = 1234):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx)
        ids = torch.randint(0, self.vocab_size, (self.sequence_length + 1,), generator=gen)
        return {"input_ids": ids[:-1], "labels": ids[1:]}


def train_validation_split(dataset: TokenDataset, val_fraction: float = 0.05) -> tuple[TokenDataset, TokenDataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    seq = dataset.sequence_length
    total = dataset.data.numel()
    # A split only makes sense when BOTH sides can hold at least one sequence.
    # Each side needs seq + 1 tokens (inputs + the label-shifted next token).
    # Fail loudly here instead of inside TokenDataset with a confusing message.
    if total < 2 * seq + 2:
        raise ValueError(
            f"corpus too small for a train/validation split: {total} tokens < {2 * seq + 2} "
            f"(sequence_length={seq}); need at least {2 * seq + 2} tokens so both splits "
            f"can hold one full sequence"
        )
    val_tokens = int(total * val_fraction)
    # split at a sequence boundary: [train | val]
    boundary = (total - val_tokens) // seq * seq
    # keep at least one full sequence (plus label token) on each side
    boundary = max(seq + 1, min(boundary, total - seq - 1))
    train = TokenDataset(dataset.data[:boundary], dataset.sequence_length)
    val = TokenDataset(dataset.data[boundary:], dataset.sequence_length)
    return train, val


def collate(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
