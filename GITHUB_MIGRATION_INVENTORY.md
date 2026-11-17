# NTransformer — GitHub Private Migration Inventory

Date: 2026-11-17 · Auditor: engineering pass on `feature/v1.3-enterprise-2026-11`
Machine: RTX 4060 Laptop (8 GB), Python 3.14.4, torch 2.15.0+cu132 · **NTokenizer repo intentionally excluded from all operations.**

---

## 1. Repository location and state

| Field | Value |
|---|---|
| Location | `/home/justerdem14/Masaüstü/Team N/NTransformer` |
| Active branch | `feature/v1.3-enterprise-2026-11` (HEAD = `d17391d`) |
| `master` | tip `14f6fe0` ("snapshot v1.0.0-rc1 frozen"); 36 commits; **behind** feature branches; NOT modified |
| Remotes | **none** (local-only repository; no `origin`) |
| Tags | **none** |
| Commits (HEAD) | 68; reachable from all refs: 72 |
| Tracked files | **1,629** (0 binary > 35 KiB; largest blob 35,532 bytes) |
| `.git` size | **4.9 MiB** |
| Working tree | **clean** (no staged/untracked/modified entries) |
| Largest on-disk dirs | `.venv/` 5.4 GB (**not tracked**), `versions/` 9.1 MB, `benchmarks/` 820 KB |

### Branches (all local, all to be migrated)

| Branch | Commits | Ahead of master | Tip |
|---|---:|---:|---|
| `master` | 36 | 0 | `14f6fe0` (frozen v1.0.0-rc1) |
| `cand/v0.6-mla` | 31 | +1 | latent-KV experiment (rejected/not promoted) |
| `cand/v0.6-wy-chunkwise` | 36 | +1 | WY-chunkwise GDN (rejected/not promoted) |
| `cand/v0.7-sliding-window` | 34 | +2 | sliding-window attention (rejected/not promoted) |
| `feature/v1.0-scaling-2026-11` | 38 | +2 | v1.0 training engine + presets |
| `feature/moe-integration-2026-11` | 42 | +6 | MoE integration |
| `feature/final-release-2026-11` | 56 | +20 | v1.1.0rc1 |
| `feature/v1.2-audit-2026-11` | 62 | +26 | v1.2.0-candidate |
| `feature/v1.3-enterprise-2026-11` | 68 | +32 | v1.3.0-candidate (HEAD) |

## 2. Release / version snapshots (`versions/` — 11 snapshots)

All rows: `manifest` = SHA-256 manifest of the frozen SOURCE tree; "tracked" = files tracked in git.

| Snapshot | Status | Files | Tracked | Manifest | commit.txt | Integrity |
|---|---|---:|---:|---|---|---|
| `v0.2-baseline` | baseline (frozen) | 95 (80 src+15 untracked `.pyc` debris — untracked, ignored) | 80 | ✓ (35 files re-verified) | `c31418e` | ✓ OK (pyc debris noted) |
| `v0.3` | candidate (promoted? registry: no) | 122 | 122 | ✓ (36) | `e3633be` | ✓ OK |
| `v0.4` | **stable best** (registry: best) | 113 | 113 | ✓ (38) | `a6b263c` | ✓ OK |
| `v0.5-candidate` | rejected (GDN) | 125 | 125 | ✓ (40) | `cdd5565` | ✓ OK |
| `v0.6-candidate` | rejected (MLA) | 137 | 137 | ✓ (42) | `d148e08` | ✓ OK |
| `v0.6.1-candidate` | rejected (WY-chunkwise GDN) | 137 | 137 | ✓ (41) | `f17a75f` | ✓ OK |
| `v0.7-candidate` | **rejected** (SWA) | 44 | 44 | **MISSING** (pre-existing; snapshot intentionally partial) | `03e4dd3` | **no manifest — documented gap, not fabricated** |
| `v0.7.1-candidate` | superseded (SWA bounds fix) | 135 | 135 | ✓ (42) | `0b9945f` | ✓ OK |
| `v1.0.0-rc1` | release candidate, not promoted | 150 | 150 | ✓ (54) | `6fe3b68` | ✓ OK |
| `v1.2.0-candidate` | candidate, not promoted | 181 | 181 | ✓ (63) | `f2b1d08…` | ✓ OK |
| `v1.3.0-candidate` | candidate, not promoted | 181 | 181 | ✓ (63) | `85ed0d0…` | ✓ OK |

**No `versions/v1.1.0-rc1` snapshot exists** — v1.1.0rc1 content is reachable via branch `feature/final-release-2026-11` (tip `5d13af9`) and full git history, but was never frozen under `versions/`. Recorded as a release-history gap; not fabricated.

`registry/best_version.json`: `best = v0.4` (unchanged) · 12 candidates, none promoted.

## 3. Secret / credential scan (HEAD + full history)

| Check | Result |
|---|---|
| Tracked filenames matching `.env`, keys, credentials, tokens | **none** |
| History grep for PEM/SSH private keys, AWS/GitHub/OpenAI/xox token patterns | **no matches** |
| Benchmark JSON scan for `api_key|secret|password|bearer|credential|authorization` | **none** |
| `.gitignore` coverage | `__pycache__`, `.venv/`, `*.egg-info`, checkpoints/, `*.safetensors` (tests excepted), `.pytest_cache`, build/dist — adequate |
| Tracked large/binary/model files | **none** (no `.safetensors` tracked; no checkpoints; largest file 35 KB) |

## 4. Risk register for migration

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | `v0.7-candidate` has no manifest — its content can't be integrity-verified by checksum | low (rejected candidate) | preserved as-is; documented; NOT re-frozen/fabricated |
| 2 | No `v1.1.0-rc1` snapshot directory | low | history preserved; documented |
| 3 | v0.2-baseline contains 15 stray `.pyc` files (untracked, ignored) | negligible | not tracked; not migrating junk |
| 4 | `.venv/` 5.4 GB and other local caches must not be pushed | — | verified untracked/ignored |
| 5 | No existing `origin`; pushing to brand-new repo = full history transfer (72 commits, tiny) | low | all refs pushed; no force |
| 6 | NTokenizer sibling repo exists at `../NTokenizer` and MUST NOT be pushed/touched | **critical boundary** | only NTransformer pushed; NTokenizer excluded by construction |

## 5. Files to be excluded from migration (untracked/ignored)

`.venv/`, `.pytest_cache/`, all `__pycache__/` trees, `neuratal_transformer.egg-info/`,
`versions/v0.2-baseline/**/__pycache__/*.pyc` (untracked), `/tmp` artifacts. None of these
is currently tracked (verified via `git status --porcelain` and `git ls-files`).

## 6. Verdict

Migration is safe: tiny repo, clean tree, zero secrets, zero large binaries, complete branch set, manifest-verified snapshots (except the two documented pre-existing gaps). Proceed to hygiene + GitHub migration.
