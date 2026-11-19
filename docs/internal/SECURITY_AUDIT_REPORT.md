# NTransformer — Security Audit Report

Date: 2026-11-17 · Scope: source tree, checkpoint/deserialization paths, git history, dependencies, migration artifacts · NTokenizer excluded (untouched).

## 1. Secret & credential scan (HEAD + full history)

| Check | Result |
|---|---|
| Filenames matching `.env`, `*.key`, `*.pem`, `id_rsa`, `credentials`, `secret`, `token` | **none tracked** |
| History grep for `BEGIN … PRIVATE KEY`, `AKIA…`, `gh[pousr]_…`, `sk-…`, `xox…`, `gho_…` | **no matches** in any of the 72 commits |
| Benchmark JSONs / docs / tests scanned for `api_key|secret|password|bearer|authorization|credential` | **no matches** (only prose matches, none credential-like) |
| `.gitignore` coverage for local env files | hardened: `.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa*`, `id_ecdsa*` added this migration (pre-existing `.gitignore` also covered `.venv`, checkpoints, safetensors) |
| On-disk debris with ignored status (`.venv/` 5.4 GB, `.pytest_cache`, `*.pyc`) | untracked, not migrated ✔ |
| `.github/workflows/tests.yml` | tests-only; no secrets, no deployment keys, no `pull_request_target`, no code execution from forks |

**Secret leakage: NONE found.** No remediation required; exposure scope is zero (local repo, never pushed before today).

## 2. Deserialization safety

| Path | Mechanism | Risk |
|---|---|---|
| Model weights | safetensors (no code execution; size-checked, key/shape/dtype cross-validated) | none inherent |
| Checkpoint configs | `config.json` — JSON only, validated structurally, no code path | none inherent |
| Trainer state | safetensors + JSON | none inherent |
| `config.json` cross-check | `format_version`, `architectures`, unknown-key rejection, shape/dtype match | good |
| JSON parsing | `json.loads` (no `object_hook`, no `ast.literal_eval`) | none |
| **No pickle anywhere** (`git grep pickle` → only a comment in `checkpoint.py` docstring) | — | ✓ none |
| **No `eval`/`exec`** (only `model.eval()` matches, which are module calls) | — | ✓ none |
| **No `torch.load`** | — | ✓ none |

**Unsafe deserialization: NOT PRESENT.**

## 3. File handling & path safety

- Atomic saves everywhere: `mkstemp` (random name, mode `0600`) + `os.replace` in the same directory; temp files unlinked on failure; no world-readable checkpoints are created (they are owner-private `0600`).
- No writes outside caller-provided directories; no path traversal sinks identified (`checkpoint_dir`/`output_dir` are used directly as given — caller owns the path; standard save API contract).
- `epoch`-style file reads with size caps: safetensors header capped; `.ntok` integrity via CRC32 + `MAX_PAYLOAD_BYTES` (1 GiB) + magic checks.
- Symlink attacks: not applicable for trust-local usage; no follow-symlink behavior that maps a link outside the given dir.
- `os.replace` on cross-device fails loudly (no silent cross-fs copy) ✓.

## 4. Input validation at API boundaries

`NeuratalConfig` validates every field at construction; invalid values raise `ValueError` with precise messages; nothing silently corrected. Checkpoint loads reject unknown/missing keys, shape/dtype mismatch, format version, architecture mismatch, tokenizer fingerprint mismatch, optimizer-index out of range. Generation validates `max_new_tokens`/`temperature`/`top_k` and generator device match.

## 5. Dependency & supply-chain risks

| Dependency | Version | Notes |
|---|---|---|
| torch | 2.15.0+cu132 | current; official wheels; large attack surface but standard for ML |
| safetensors | 0.8.0 | Rust backend; no Python code execution from artifacts |
| safetensors dep chain | — | pinned wheels only |
| pytest / numpy (dev only) | 9.2.0 / 2.5.5 | dev-only |
| pip tree | minimal; no `transformers`/`accelerate`/`pickle`-libs pulled | good |

No known exploited CVEs are in the pinned set at time of audit (informational; no live CVE feed queried — flagged honestly). `nvidia-*` CUDA wheels come from the official PyPI index; no custom indexes.

## 6. Availability / DoS observations

- Allocation bomb via crafted `.ntok`/safetensors: mitigated by `MAX_PAYLOAD_BYTES` (1 GiB) and format CRC; model-building is caller-controlled (huge presets are meta-device only; real allocation is bounded by disk at save time and RAM at load).
- `plan_shards` never splits tensors; a tensor larger than `max_shard_bytes` gets its own shard (documented).
- No unbounded recursion/loops on untrusted data identified.

## 7. Severity-ranked findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| SEC-2022-1 | **Low** | Sharded format trusted byte count only (pre-hash) — same-size replacement passed | **FIXED (v1.3)** — per-shard SHA-256, default on |
| SEC-2022-2 | Low | Scheduler state accepted malformed/corrupt payloads silently | **FIXED (v1.2/v1.3)** — required-keys + type/range validation |
| SEC-2022-3 | Informational | History contains no secrets; `.gitignore` was missing `.env` patterns | FIXED (this migration) |
| SEC-2022-4 | Informational | `versions/v0.7-candidate` unverifiable (no manifest) | documented, not fixable without fabrication — left intact |
| SEC-2022-5 | Informational | No `integrity` pinning for the pip environment (no `requirements.txt` hash pinning); documented, not a vulnerability | open, advisory |
| SEC-2022-6 | Informational | Windows/macOS loaders untested | open, advisory |

**No critical or high-severity security findings.**

## 8. Practices NOT in scope / not claimed

No side-channel or memory-safety review of native extensions (torch/CUDA/safetensors are out of scope); no fuzzing of artifact loaders on this pass; no Windows/macOS verification; no license-compatibility scan of datasets/benchmarks (benchmark corpus licenses documented in `data/` of NTokenizer, not in this repo).

## 9. Migration-security conclusion

The repository is safe to host privately in GitHub: no secrets, no code-execution deserialization, no hidden payloads; `.gitignore` hardened; checkpoint integrity now hash-verified. **Public release (if ever desired) requires a fresh secrets review and removal of any private path references.**
