"""Tokenizer artifact manifests — exact identity for a NTokenizer .ntok file.

A manifest (schema ``neuratal-tokenizer-manifest/1``) binds a checkpoint or
inference run to the *exact* tokenizer artifact:

* name, model version, crate version (from the v3 embedded ModelMetadata)
* format and format version (parsed + CRC-verified, not trusted blindly)
* vocab size, base-byte/merge/special counts (cross-checked against payload)
* special tokens (content → id map)
* SHA-256 of the exact artifact file

The manifest is data, never code. Verification re-derives everything from the
file on disk; a manifest that does not match is rejected with reasons. A
matching vocabulary size alone is NOT considered proof of identity —
SHA-256 equality plus special-token agreement are required.

Nothing in the Transformer core imports this module.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any, Optional, Union

MAGIC = b"NTOK"
HEADER_STRUCT = struct.Struct("<4sH I Q")  # magic, format_version, crc32, payload_len
SCHEMA = "neuratal-tokenizer-manifest/1"
MANIFEST_FILENAME = "tokenizer.json"
MAX_PAYLOAD_BYTES = 1 << 30


class NTokenizerIntegrationError(RuntimeError):
    """Base error for integration failures."""


class ManifestError(NTokenizerIntegrationError):
    """Raised when a .ntok artifact or manifest fails validation."""


class TokenizerMismatchError(NTokenizerIntegrationError):
    """Raised when the available tokenizer artifact does not match a manifest/checkpoint."""


# ---------------------------------------------------------------------
# .ntok parsing (documented NTOK layout only — NOT a tokenizer implementation)
# ---------------------------------------------------------------------
def parse_ntok_file(path: Union[str, Path]) -> dict[str, Any]:
    """Parse and validate an .ntok file's header + payload structure.

    Verifies: magic, format_version, CRC-32 of the payload, and total length.
    Returns a dict with the payload JSON and derived fields. Does not interpret
    the BPE table beyond structural validation.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"tokenizer file does not exist: {path}")
    data = path.read_bytes()
    if len(data) < HEADER_STRUCT.size:
        raise ManifestError(f"file too small for an NTOK header: {path} ({len(data)} bytes)")
    magic, version, crc, payload_len = HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC:
        raise ManifestError(f"bad magic {magic!r} in {path} (expected b'NTOK')")
    if version not in (1, 2, 3):
        raise ManifestError(f"unsupported NTOK format_version {version} in {path}")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ManifestError(f"payload_len {payload_len} exceeds 1 GiB in {path}")
    if len(data) != HEADER_STRUCT.size + payload_len:
        raise ManifestError(
            f"file size mismatch in {path}: header says {HEADER_STRUCT.size + payload_len} bytes, "
            f"file is {len(data)} bytes (truncation, splicing, or trailing garbage)"
        )
    payload = data[HEADER_STRUCT.size:]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ManifestError(f"CRC-32 mismatch in {path}: file is corrupt")
    try:
        doc = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"payload is not valid UTF-8 JSON in {path}: {exc}") from exc
    for key in ("vocab", "merges", "special_tokens"):
        if key not in doc:
            raise ManifestError(f"payload missing required section {key!r} in {path}")
    return {"path": path, "format_version": version, "payload": doc, "raw": data}


def _sha256(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _cross_check_metadata(payload: dict[str, Any], format_version: int) -> dict[str, Any]:
    """Verify v3 embedded metadata against the actual payload counts."""
    metadata = payload.get("metadata") or {}
    id_to_bytes = payload["vocab"]["id_to_bytes"]
    merges = payload["merges"]["rules"]
    specials = payload["special_tokens"]["tokens"]
    if len(id_to_bytes) < 256:
        raise ManifestError(f"vocabulary has {len(id_to_bytes)} entries (< 256 base byte tokens)")
    base = sum(1 for i in range(256) if id_to_bytes[i] == [i])
    vocab_size = len(id_to_bytes)
    seen: set[int] = set()
    for item in specials:
        sid = int(item["id"])
        if sid in seen:
            raise ManifestError(f"duplicate special token id {sid}")
        if not (256 <= sid < vocab_size):
            raise ManifestError(f"special token id {sid} out of range [256, {vocab_size})")
        seen.add(sid)
    report = {
        "vocab_size": len(id_to_bytes),
        "base_byte_tokens": base,
        "merges_count": len(merges),
        "special_tokens_count": len(specials),
    }
    if format_version == 3:
        expected = {
            "vocab_size": metadata.get("vocab_size"),
            "base_byte_tokens": metadata.get("base_byte_tokens"),
            "merges_count": metadata.get("merges_count"),
            "special_tokens_count": metadata.get("special_tokens_count"),
        }
        for key, want in expected.items():
            got = report[key]
            if want is not None and want != got:
                raise ManifestError(
                    f"embedded metadata {key}={want} disagrees with payload {key}={got}; refusing manifest"
                )
    return report


def _special_map(payload: dict[str, Any]) -> dict[str, int]:
    return {item["content"]: int(item["id"]) for item in payload["special_tokens"]["tokens"]}


def build_manifest(ntok_path: Union[str, Path]) -> dict[str, Any]:
    """Build a manifest for an .ntok artifact. Validates structure strictly."""
    parsed = parse_ntok_file(ntok_path)
    payload = parsed["payload"]
    counts = _cross_check_metadata(payload, parsed["format_version"])
    metadata = payload.get("metadata") or {}
    specials = _special_map(payload)
    manifest = {
        "schema": SCHEMA,
        "tokenizer_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "tokenizer_crate_version": metadata.get("tokenizer_crate_version"),
        "format": "NTOK",
        "format_version": parsed["format_version"],
        "pretokenizer": payload.get("pretokenizer"),
        "pretokenizer_version": metadata.get("pretokenizer_version"),
        "normalization": payload.get("normalization"),
        **counts,
        "special_tokens": specials,
        "bos_token_id": specials.get("<|bos|>"),
        "eos_token_id": specials.get("<|eos|>"),
        "pad_token_id": specials.get("<|pad|>"),
        "tokenizer_sha256": _sha256(parsed["path"]),
        "metadata_created_at": metadata.get("created_at"),
        "metadata_corpus_manifest_sha256": metadata.get("corpus_manifest_sha256"),
    }
    return manifest


def save_manifest(manifest: dict[str, Any], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest file does not exist: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if doc.get("schema") != SCHEMA:
        raise ManifestError(f"unsupported manifest schema {doc.get('schema')!r} in {path}")
    if "tokenizer_sha256" not in doc or "special_tokens" not in doc:
        raise ManifestError(f"manifest in {path} is missing required fields")
    return doc


def verify_manifest(manifest: dict[str, Any], ntok_path: Union[str, Path]) -> None:
    """Raise TokenizerMismatchError listing every discrepancy, or return None."""
    reasons: list[str] = []
    try:
        rebuilt = build_manifest(ntok_path)
    except ManifestError as exc:
        raise TokenizerMismatchError(f"tokenizer artifact failed validation: {exc}") from exc
    for key in (
        "format", "format_version", "vocab_size", "base_byte_tokens", "merges_count",
        "special_tokens_count", "tokenizer_sha256", "pretokenizer", "normalization",
    ):
        want, got = manifest.get(key), rebuilt.get(key)
        if want != got:
            reasons.append(f"{key}: manifest={want!r} vs artifact={got!r}")
    want_specials, got_specials = manifest.get("special_tokens"), rebuilt.get("special_tokens")
    if want_specials != got_specials:
        reasons.append(f"special_tokens: manifest={want_specials} vs artifact={got_specials}")
    if reasons:
        raise TokenizerMismatchError(
            "tokenizer artifact does not match manifest:\n  - " + "\n  - ".join(reasons)
        )
