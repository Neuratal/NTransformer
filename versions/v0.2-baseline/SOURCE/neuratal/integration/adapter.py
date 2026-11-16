"""NTokenizer ↔ NeuratalTransformer adapter.

Optional, thin integration layer. Core (`neuratal.model`) never imports this.
Requires the NTokenizer Python bindings (ctypes, no new Python dependencies)
and the actual NTokenizer v0.3 artifact; importing this module does not load
either — construction is explicit.

Usage
-----
    adapter = NTokenizerAdapter("/path/to/ntokenizer-0.3-256k.ntok",
                                manifest_path="optional/tokenizer.json")
    # or: NTokenizerAdapter(ntok_path, expected=build_manifest(other_path))
    ids = adapter.encode("Hello, world!")
    ...
    report = verify_checkpoint_tokenizer(model, checkpoint_dir, ntok_path)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

import torch

from neuratal.config import NeuratalConfig
from neuratal.integration.manifest import (
    MANIFEST_FILENAME,
    NTokenizerIntegrationError,
    TokenizerMismatchError,
    build_manifest,
    load_manifest,
    save_manifest,
    verify_manifest,
)

NTOKENIZER_ROOT = Path(os.environ.get("NTOKENIZER_ROOT", Path(__file__).resolve().parents[3] / "NTokenizer"))
DEFAULT_TOKENIZER_PATH = NTOKENIZER_ROOT / "output" / "v0.3" / "ntokenizer-0.3-256k.ntok"
DEFAULT_BINDINGS_DIR = NTOKENIZER_ROOT / "python"

EXPECTED_SPECIAL_TOKENS = {
    "<|bos|>": 262137,
    "<|eos|>": 262138,
    "<|pad|>": 262139,
    "<|unk|>": 262140,
    "<|system|>": 262141,
    "<|user|>": 262142,
    "<|assistant|>": 262143,
}


class NTokenizerUnavailableError(NTokenizerIntegrationError):
    """Bindings or the tokenizer artifact are not available in this environment."""


def _load_bindings():
    bindings_dir = Path(os.environ.get("NTOKENIZER_BINDINGS", DEFAULT_BINDINGS_DIR))
    if str(bindings_dir) not in sys.path:
        sys.path.insert(0, str(bindings_dir))
    try:
        import neuratal_tokenizer  # noqa: F401
    except Exception as exc:
        raise NTokenizerUnavailableError(
            f"NTokenizer Python bindings unavailable from {bindings_dir}: {exc}"
        ) from exc
    return neuratal_tokenizer


class NTokenizerAdapter:
    """Verified handle to the exact tokenizer artifact.

    Parameters
    ----------
    ntok_path: path to the .ntok artifact.
    expected_manifest / manifest_path: optional manifest (dict or JSON path)
        to verify against. When provided, any mismatch raises TokenizerMismatchError.
        When ``verify_against_build`` is True (default when a manifest is given),
        the manifest is checked as-provided; building from the same file always
        matches. Cross-artifact verification means comparing against a manifest
        produced from a *different* file path.
    """

    def __init__(
        self,
        ntok_path: Union[str, Path],
        *,
        manifest_path: Optional[Union[str, Path]] = None,
        expected_manifest: Optional[dict[str, Any]] = None,
    ):
        self.ntok_path = Path(ntok_path)
        if not self.ntok_path.is_file():
            raise NTokenizerUnavailableError(f"tokenizer artifact not found: {self.ntok_path}")

        self.manifest = build_manifest(self.ntok_path)

        if manifest_path is not None:
            provided = load_manifest(manifest_path)
        elif expected_manifest is not None:
            provided = expected_manifest
        else:
            provided = None
        if provided is not None:
            verify_manifest(provided, self.ntok_path)
            self.verified_against = "manifest"
        else:
            self.verified_against = "embedded-metadata"

        bindings = _load_bindings()
        self._lib = bindings
        self._tokenizer = bindings.Tokenizer.from_file(str(self.ntok_path))
        # cross-check bindings vs manifest (defense in depth against a stale .so)
        if self._tokenizer.vocab_size != self.manifest["vocab_size"]:
            raise TokenizerMismatchError(
                f"native library reports vocab_size={self._tokenizer.vocab_size}, "
                f"manifest says {self.manifest['vocab_size']}"
            )
        if self._tokenizer.special_tokens != self.manifest["special_tokens"]:
            raise TokenizerMismatchError(
                f"native library special_tokens={self._tokenizer.special_tokens}, "
                f"manifest says {self.manifest['special_tokens']}"
            )

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.manifest["vocab_size"]

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self.manifest["special_tokens"])

    def special_token_id(self, content: str) -> Optional[int]:
        return self.manifest["special_tokens"].get(content)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.manifest.get("eos_token_id")

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.manifest.get("pad_token_id")

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.manifest.get("bos_token_id")

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def encode_bytes(self, data: bytes) -> list[int]:
        return self._tokenizer.encode_bytes(data)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self._tokenizer.encode_batch(texts)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def decode_bytes(self, ids: list[int]) -> bytes:
        return self._tokenizer.decode_bytes(ids)

    def assert_model_compatible(self, config: NeuratalConfig) -> None:
        if config.vocab_size != self.vocab_size:
            raise TokenizerMismatchError(
                f"model vocab_size={config.vocab_size} does not match tokenizer "
                f"vocab_size={self.vocab_size} (IDs are never remapped)"
            )

    def close(self) -> None:
        self._tokenizer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------
# Checkpoint association (sidecar; core serialization format untouched)
# ---------------------------------------------------------------------
def write_tokenizer_sidecar(ntok_path: Union[str, Path], checkpoint_dir: Union[str, Path]) -> Path:
    """Write <checkpoint_dir>/tokenizer.json describing the exact artifact."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")
    manifest = build_manifest(ntok_path)
    return save_manifest(manifest, checkpoint_dir / MANIFEST_FILENAME)


def verify_checkpoint_tokenizer(
    model_config: NeuratalConfig,
    checkpoint_dir: Union[str, Path],
    ntok_path: Union[str, Path],
    *,
    strict_sidecar: bool = True,
) -> dict[str, Any]:
    """Verify that a checkpoint, its tokenizer sidecar (if any), and the
    provided .ntok artifact all describe the same vocabulary.

    Returns a report dict; raises TokenizerMismatchError on any violation.
    """
    checkpoint_dir = Path(checkpoint_dir)
    report: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "ntok_path": str(ntok_path),
        "model_vocab_size": model_config.vocab_size,
        "checks": [],
    }
    artifact_manifest = build_manifest(ntok_path)
    art_vocab = artifact_manifest["vocab_size"]
    if model_config.vocab_size != art_vocab:
        raise TokenizerMismatchError(
            f"model vocab_size={model_config.vocab_size} != tokenizer vocab_size={art_vocab}"
        )
    report["checks"].append(("model_vocab_size", True))

    # special-token hints in config, when set, must agree with the artifact
    for attr, key in (("eos_token_id", "eos_token_id"), ("pad_token_id", "pad_token_id"), ("bos_token_id", "bos_token_id")):
        want = getattr(model_config, attr)
        got = artifact_manifest.get(key)
        if want is not None and want != got:
            raise TokenizerMismatchError(
                f"model config.{attr}={want} but tokenizer artifact says {key}={got}"
            )
        report["checks"].append((attr, True))

    sidecar = checkpoint_dir / MANIFEST_FILENAME
    if sidecar.is_file():
        sidecar_manifest = load_manifest(sidecar)
        verify_manifest(sidecar_manifest, ntok_path)
        report["sidecar"] = "verified"
        report["checks"].append(("sidecar_sha256", True))
    else:
        if strict_sidecar:
            raise TokenizerMismatchError(
                f"checkpoint {checkpoint_dir} has no tokenizer.json sidecar; "
                "write one with write_tokenizer_sidecar() or pass strict_sidecar=False"
            )
        report["sidecar"] = "absent (strict check disabled)"
    report["tokenizer_sha256"] = artifact_manifest["tokenizer_sha256"]
    report["verified"] = True
    return report


def load_checkpoint_with_tokenizer(
    checkpoint_dir: Union[str, Path],
    ntok_path: Union[str, Path] = DEFAULT_TOKENIZER_PATH,
    map_location: Optional[Union[str, torch.device]] = None,
    *,
    strict_sidecar: bool = True,
):
    """load an NTransformer checkpoint and verify the tokenizer artifact against it.

    Returns (model, adapter, report). Raises on any mismatch — never loads a
    model against a mismatched tokenizer silently.
    """
    from neuratal.serialization.checkpoint import load_pretrained

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")

    adapter = NTokenizerAdapter(ntok_path)
    model = load_pretrained(checkpoint_dir, map_location=map_location)
    report = verify_checkpoint_tokenizer(
        model.config, checkpoint_dir, ntok_path, strict_sidecar=strict_sidecar
    )
    return model, adapter, report
