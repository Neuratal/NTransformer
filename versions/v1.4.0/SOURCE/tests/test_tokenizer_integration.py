# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""End-to-end NTokenizer -> NeuratalModel compatibility.

Uses the real NTokenizer v0.3 Python bindings and the v0.3 model artifact when
present; skipped gracefully (never failed) otherwise. The Transformer never
modifies or remaps tokenizer IDs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

NTOKENIZER_ROOT = Path(__file__).resolve().parents[2] / "NTokenizer"
TOKENIZER_MODEL = NTOKENIZER_ROOT / "output" / "v0.3" / "ntokenizer-0.3-256k.ntok"

pytestmark = pytest.mark.skipif(
    not TOKENIZER_MODEL.is_file(),
    reason=f"NTokenizer v0.3 model not found at {TOKENIZER_MODEL}",
)


@pytest.fixture(scope="module")
def tokenizer():
    sys.path.insert(0, str(NTOKENIZER_ROOT / "python"))
    try:
        from neuratal_tokenizer import Tokenizer
    except Exception as exc:  # bindings unavailable/not built
        pytest.skip(f"NTokenizer bindings unavailable: {exc}")
    return Tokenizer.from_file(str(TOKENIZER_MODEL))


def test_tokenizer_vocab_matches_model(tokenizer):
    assert tokenizer.vocab_size == 262_144
    specials = tokenizer.special_tokens
    assert specials == {
        "<|bos|>": 262137,
        "<|eos|>": 262138,
        "<|pad|>": 262139,
        "<|unk|>": 262140,
        "<|system|>": 262141,
        "<|user|>": 262142,
        "<|assistant|>": 262143,
    }


def test_tokenizer_ids_roundtrip(tokenizer):
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    assert len(ids) > 0
    assert all(0 <= i < 262_144 for i in ids)
    assert tokenizer.decode(ids) == text


def test_model_consumes_real_tokenizer_ids(tokenizer, device):
    from neuratal import NeuratalConfig, NeuratalModel

    model = NeuratalModel(NeuratalConfig()).to(device).eval()
    ids = torch.tensor([tokenizer.encode("Hello, world!")], device=device)
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape == (1, ids.size(1), 262_144)
    assert torch.isfinite(out.logits.float()).all()


def test_model_consumes_special_token_ids(tokenizer, device):
    from neuratal import NeuratalConfig, NeuratalModel

    model = NeuratalModel(NeuratalConfig()).to(device).eval()
    # raw special-token IDs (system/user/assistant) — the model must accept them verbatim
    ids = torch.tensor([[262141, 262142, 262143]], device=device)
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape == (1, 3, 262_144)
    assert torch.isfinite(out.logits.float()).all()
