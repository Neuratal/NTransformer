# Checkpoint Integrity — Verification Report (v1.3.0-candidate)

Scope: single-file safetensors checkpoints, training-state checkpoints,
local sharded checkpoints. All corruption cases below were tested on this
machine (RTX 4060 Laptop, CPU fallbacks exercised where applicable).

## 1. Tested failure cases and outcomes

| Case | Mechanism | Result |
|---|---|---|
| `trainer_state.safetensors` truncated | atomic temp+rename at save; load rejects (safetensors parse error) | rejected with precise error (`test_corrupted_checkpoint_rejected`) |
| `config.json` corrupt/invalid | `json.loads` + `_check_config_payload` | rejected |
| Model safetensors truncated | safetensors header/length check | rejected (`failed to read safetensors file`) |
| Missing shard / renamed shard | sharded loader: referenced filename absent | rejected (`missing shard ...`) |
| Extra/duplicate shard keys | weight_map cross-check | rejected |
| Shard **replaced with same keys, wrong byte count** | `metadata.total_size` vs loaded bytes | rejected (v1.3.0-candidate, `test_sharded_total_size_integrity_check`) |
| Shard **replaced with same-size, different bytes** | `metadata.shard_hashes` SHA-256 per shard (v1.3.0-candidate), default on; opt-out documented | rejected with hash mismatch (`test_sharded_tampered_shard_rejected`) |
| Shard hashes **missing from index** (legacy) | skip hash check; `total_size` still enforced | loads (backwards compat, documented) |
| Shard hashes **incomplete** (one entry missing) | per-file hash verification | rejected (`test_sharded_missing_hash_entry_rejected`) |
| Partial/interrupted save | atomic `mkstemp` + `os.replace` everywhere; no torn `.safetensors`/`trainer_state`/`tokenizer.json` can be observed | by construction (code review) |
| Wrong architecture (dense↔MoE) | `load_training_checkpoint` + `resume_training_checkpoint` arch guard | rejected |
| Optimizer state for missing param indices | index-range guard (v1.1) | rejected with precise message |
| Scheduler base_lrs length mismatch | v1.2 length validation | rejected |
| Tokenizer fingerprint mismatch on resume | `expected_tokenizer_fingerprint` comparison | rejected |
| RNG restore | CPU + CUDA byte-exact (new CUDA test) | verified byte-equal |
| Optimizer state round-trip | index-addressed tensors | verified `torch.equal` (v1.1 tests) |
| Scheduler counter/LRs round-trip | float64 storage (v1.3) + `lr_at(step-1)` reapply | verified exact (see `docs/scheduler_resume_verification.md`) |

## 2. Atomicity design

Every write path goes through `mkstemp` (unique temp name, `O_EXCL`) +
`os.replace`:
- `serialization/checkpoint.py::_atomic_save_safetensors` / `_atomic_write_text`
- `serialization/sharded.py::_atomic_save_safetensors` (shared helper)
- `training/checkpoint.py::_atomic_save_safetensors` / `_atomic_write_text`

Temp files are unlinked on any failure path; `missing` errors are explicit.
No temporary file can be promoted to the target name half-written: the
contents land on disk before `os.replace` (same-filesystem guarantee enforced
by `dir=path.parent`).

**Not claimed:** crash consistency across power loss or filesystem failures
(torn writes at the filesystem level). Atomic rename protects against
*partial file visibility*, not against storage-level corruption. CRC32/
SHA-256 checks cover load-time detection.

## 3. Strength assessment (honest)

| Layer | Detection guarantee |
|---|---|
| safetensors internal | malformed truncation caught; per-tensor CRC absent (the format does not checksum tensors) |
| `total_size` metadata | catches byte-count changes (but not same-size replacement) |
| `shard_hashes` | catches arbitrary content changes; ~63 ms per 37 MB model (measured); default **on**; `verify_shard_hashes=False` documented for huge checkpoints with out-of-band verification |
| `config.json` | structure + version + architecture + dtype + cross-checks |
| `.nvtok` (NTransformer-side manifest) | CRC32 + SHA-256 + vocab/specials cross-check |

## 4. What is not verified

- Power-fail/fsync ordering across filesystems (ext4 vs btrfs vs overlayfs).
- Adversarial artifacts specifically crafted to pass CRC32 + structural checks
  while carrying malicious-but-loadable content (format is validated, but a
  motivated attacker can always produce structurally valid files — provenance
  is fingerprint-based, not a trust chain).
- Distributed/sharded checkpoints across nodes (out of scope; local only).

## 5. Commands

```bash
.venv/bin/python -m pytest tests/test_serialization.py tests/test_engine_regressions.py -q
```
