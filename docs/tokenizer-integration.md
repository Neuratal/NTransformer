# NTokenizer Integration

NTransformer is **NTokenizer-native by construction**. This document describes
exactly how the tokenizer and this package integrate, and how the boundary is
enforced.

## Compatibility contract (immutable)

- `vocab_size = 262,144` (256 base bytes + 261,881 learned merges + 7 special tokens).
- Token IDs are consumed **verbatim**. Never remapped, never translated.
- Special tokens: BOS 262137, EOS 262138, PAD 262139, UNK 262140,
  SYSTEM 262141, USER 262142, ASSISTANT 262143.
- Artifact: `ntokenizer-0.3-256k.ntok` (Neuratal NTokenizer v0.3-polished, `NTOK v3` format).
  SHA-256: `1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9`.
- This package **never** injects BOS/EOS implicitly. Callers supply full sequences.

## How it is loaded

Bindings: `neuratal_tokenizer` (ctypes, stdlib-only) at
`../NTokenizer/python/neuratal_tokenizer/`; `Tokenizer.from_file(path)`.
The native library (`libneuratal_tokenizer.so`, release build) is loaded by
the bindings; `NTOKENIZER_LIB` may override discovery.

The `.ntok` format is validated on load: magic/format-version/CRC/length, plus
cross-checks between the embedded metadata and the payload itself (special
tokens uniqueness/range, vocab size). Python repeats these checks and adds a
SHA-256 of the file itself.

## Integration module (`neuratal/integration/`)

A checkpoint produced by this package can be bound to the **exact** `.ntok`
artifact it was trained with or designed to run with. The integration manifest
records:

- exact tokenizer file SHA-256
- special token map (IDs)
- vocabulary size
- CRC/format sanity

Loading with a mismatched manifest is rejected explicitly; resume of a
training run validates the checkpoint fingerprint.

## NTokenizer links

- GitHub: https://github.com/Neuratal/neuratal-tokenizer
- Hugging Face: https://huggingface.co/Neuratal/ntokenizer
- Release: https://github.com/Neuratal/neuratal-tokenizer/releases/tag/v0.3.0

NTokenizer is an **independent project** — different repository, different
release cadence, different license (CC-BY-SA-4.0). This project does not
modify or re-license NTokenizer.
