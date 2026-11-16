# NTokenizer v0.3 ↔ NeuratalTransformer Compatibility Report

Status: **verified on the development machine (2026-11-16).**
Artifact: `output/v0.3/ntokenizer-0.3-256k.ntok` in the NTokenizer repository
(23,403,000 bytes; SHA-256
`1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9`).

This report documents how the integration works, what was verified, and the
compatibility contract. It preserves, rather than supersedes, earlier notes in
`docs/architecture_audit.md` §1 and `docs/final_report.md` §8 — those remain
valid.

## 1. How the tokenizer is loaded

- Bindings: `neuratal_tokenizer` (ctypes, stdlib-only) at
  `../NTokenizer/python/neuratal_tokenizer/`; `Tokenizer.from_file(path)`.
- The native library (`libneuratal_tokenizer.so`, release build) is loaded by
  the bindings; `NTOKENIZER_LIB` may override discovery.
- `.ntok` format v3: 18-byte header (`NTOK` magic, `format_version=3`,
  CRC-32 IEEE of the payload, `payload_len`), then a UTF-8 JSON payload
  containing `vocab.id_to_bytes` (262,144 entries, dense IDs from 0),
  `merges.rules` (261,881), `special_tokens.tokens` (7), and embedded
  `metadata` (model_name `ntokenizer-0.3-256k`, model_version `0.3.0`,
  crate version `0.3.0`, pretokenizer `MathSymbolAware`, normalization `None`).
- The v3 metadata claims are cross-checked against the actual payload on
  load by the Rust loader; our integration manifest repeats that verification
  in Python (magic/version/CRC/length, metadata↔payload agreement, special
  token uniqueness/range) and adds a SHA-256 of the file itself.

## 2. Token ID layout — verified against the artifact, not assumed

| IDs | Meaning |
|---|---|
| 0–255 | Base byte tokens; `decode_bytes([i]) == bytes([i])` for every i in 0..255 (tested) |
| 256–262136 | Learned merge targets |
| 262137 | `<|bos|>` |
| 262138 | `<|eos|>` |
| 262139 | `<|pad|>` |
| 262140 | `<|unk|>` |
| 262141 | `<|system|>` |
| 262142 | `<|user|>` |
| 262143 | `<|assistant|>` |

Verified programmatically (not read from docs): special literals encode to
their IDs (`encode("<|bos|>") == [262137]`, etc.) and decode back to their
literal content (`decode([262137]) == "<|bos|>"`, etc.).

## 3. Integration boundary

```
Raw Text → NTokenizer v0.3 → List[int] → NeuratalModel → logits → generate() → List[int] → NTokenizer → Text
```

- The Transformer core (`neuratal/model/*`, `neuratal/serialization/*`) never
  imports the tokenizer. Nothing in the core depends on it.
- `neuratal/integration/` is the only layer that knows about NTokenizer:
  - `manifest.py`: builds/verifies a `neuratal-tokenizer-manifest/1` JSON —
    tokenizer_name, versions, format, vocab/merge/base-byte/special counts,
    special-token map, and **SHA-256 of the exact artifact**. Verification
    re-derives everything from the file on disk and raises
    `TokenizerMismatchError` with per-field reasons on any discrepancy.
  - `adapter.py`: `NTokenizerAdapter` (verified handle; cross-checks native
    bindings vs manifest) and checkpoint association helpers:
    `write_tokenizer_sidecar()` writes `tokenizer.json` next to
    `config.json`/`model.safetensors`; `verify_checkpoint_tokenizer()` and
    `load_checkpoint_with_tokenizer()` enforce consistency.
  - The existing checkpoint format is **unchanged**. The sidecar is a new,
    optional file; checkpoints without it load exactly as before.
- Token IDs pass through verbatim in both directions; no remapping, no added
  BOS/EOS, no normalization anywhere.

## 4. Verification results (measured on 2026-11-16)

Artifact checks (via `neuratal.integration.build_manifest`):
magic/version/CRC/length OK · metadata↔payload cross-check OK ·
base bytes 0..255 correct in payload · special token IDs unique, ≥256, <vocab ·
SHA-256 matches the documented v0.3 model of record.

Test suite (`tests/test_ntokenizer_integration.py`, 51 tests): all pass on CPU
and CUDA —

- vocab sizes agree exactly (adapter 262144 == model config 262144);
- manifest SHA-256 matches the documented artifact;
- boundary IDs (0, 255, 256, 262136, 262137…262143) accepted in-range;
- GF: all 256 base byte IDs roundtrip exactly; special IDs decode to their
  literal content; special literals encode to their IDs;
- encode→decode roundtrip on 12 text classes (English, Turkish,
  multilingual, code, numbers, punctuation, whitespace, tabs/newlines, emoji
  incl. ZWJ sequences, combining marks, empty input) and a 40×-repeated long
  input; raw-byte roundtrip including invalid UTF-8 (`b"\xff\xfe\x00..."`);
- special-token EOS id is honored by `generate()` (unit-verified mechanism)
  and configured through the adapter, never hardcoded;
- end-to-end encode→forward→generate→decode for sample texts; logits finite
  and `[B, T, 262144]`;
- checkpoint compatibility: sidecar roundtrip verified; tampered SHA-256,
  v0.2-vs-v0.3 cross-artifact, and config-EOS mismatch all detected and
  rejected; checkpoint without sidecar loads via the core loader (backward
  compatibility) or with an explicitly relaxed check.

Not run / not claimed explicitly:
- NTokenizer was not modified in any way (no rebuilds, no flag changes).
- `decode()` (strict UTF-8) rejects byte streams that are not valid UTF-8 —
  observed on untrained model outputs; consumers must use `decode_bytes()`
  for arbitrary model output, exactly as the tokenizer documents.
- Special-token behavior beyond the seven registered tokens is tokenizer
  contract, not model behavior; untested configurations were not probed.

## 5. EOS/BOS/PAD wiring

`NeuratalConfig.eos_token_id` etc. are optional and never injected implicitly
by the core. The integration example and the checkpoint sidecar both carry
the artifact's actual ids (eos 262138, pad 262139, bos 262137), read from the
manifest at runtime — never hardcoded in the model.
