# NTransformer — GitHub Private Migration Report (FINAL)

Date: 2026-11-17 · Repository: **https://github.com/Neuratal/NTransformer** — **PRIVATE, verified**
Source: local checkout `/home/justerdem14/Masaüstü/Team N/NTransformer` (branch `feature/v1.3-enterprise-2026-11`)

---

## 1. Repository lifecycle

| Step | Status | Evidence |
|---|---|---|
| Target repository existence check | none existed | `gh api repos/Neuratal/NTransformer` → 404 pre-creation |
| Creation | created 2026-11-17T09:43:49Z via `gh repo create Neuratal/NTransformer --private` | API |
| Visibility | **PRIVATE** — `"visibility": "private"`, `"private": true` | API `repos/Neuratal/NTransformer` |
| Default branch | `master` | API `PATCH …/default_branch → master` |
| NTokenizer boundary | not accessed or modified at any point | — |

## 2. Branch migration — **9/9 complete, all verified**

Push used: `git push origin --all` (8 branches; then after the token received the `workflow` scope) and `git push origin feature/v1.3-enterprise-2026-11` (HEAD branch, containing `.github/workflows/tests.yml`). **No `--force`, no `--mirror`, no history rewrite.** `master` was never force-pushed.

| Branch | Head match | Tree match | Status |
|---|---|---|---|
| `master` | ✓ `fb6ccd708f58a...` | ✓ `55bb3e8` | migrated |
| `cand/v0.6-mla` | ✓ (d148e082) | ✓ `b1ccc78` | migrated |
| `cand/v0.6-wy-chunkwise` | ✓ (f17a75fe) | ✓ `6d78a8c` | migrated |
| `cand/v0.7-sliding-window` | ✓ (0b9945f9) | ✓ `84c9fc4` | migrated |
| `feature/moe-integration-2026-11` | ✓ (196a613d) | ✓ `e34035f` | migrated |
| `feature/v1.0-scaling-2026-11` | ✓ (14f6fe03) | ✓ `22e18e25` | migrated |
| `feature/v1.2-audit-2026-11` | ✓ (3976a5db) | ✓ `ca2ce63f` | migrated |
| `feature/final-release-2026-11` | ✓ (5d13af96) | ✓ `f4e8556f` | migrated |
| `feature/v1.3-enterprise-2026-11` | ✓ (b401a334) | ✓ `141f4977` | **migrated (after scope grant)** |

### 2a. master head note (corrected)
The local `master` branch is `fb6ccd7` (36 commits; v0.6.1-candidate state). `14f6fe0` ("snapshot v1.0.0-rc1 frozen") is a **descendant** of `fb6ccd7` that exists only on the feature branches, not on `master` — `master` was not updated to v1.0.0-rc1 by its previous maintainers. Remote `master` == local `master` (`fb6ccd708f58a919c2336e96a482d722f0b94f0e`); tree `55bb3e828e6334b08e0995625ad1519703db3db5` verified identical via the GitHub API.

## 3. Integrity verification (all 9 branches)

1. Remote head SHA == local head SHA ✔ (9/9)
2. Remote root tree SHA == local tree SHA ✔ (9/9, content byte-equality incl. filenames/modes/symlinks)
3. Spot blob comparisons ✔ (e.g. `README.md` on master; `neuratal/model/moe.py` on v1.2-audit; `versions/v1.3.0-candidate/file_manifest.json` on the freeze branch; `.github/workflows/tests.yml` on feature/v1.3-enterprise-2026-11 — all match `git hash-object`)
4. No large blobs on GitHub: `.git` < 5 MiB; largest tracked file 35 KB ✔
5. `versions/` snapshots: all manifests re-verified before push (v0.2-baseline … v1.3.0-candidate); `v0.7-candidate` has no manifest (pre-existing, documented) ✔

## 4. Releases / tags

No tags existed in the local repository (`git tag` → empty), so there was nothing to migrate. All release snapshots are on GitHub via the `versions/` directory in the relevant branches, with their SHA-256 manifests (except the documented `v0.7-candidate` gap).

## 5. Incidents (all surfaced, none hidden)

1. Initial push of HEAD branch rejected: token lacked the `workflow` scope (repo pushes need it for `.github/workflows/*`). **The account had been renamed (ronalixai → Neuratal)** so the stored gh credential was stale; initial device-flow approvals landed on the Neuratal account while the CLI looked for ronalixai.
2. Resolved by `gh auth login` re-authentication as Neuratal with `workflow` scope explicitly requested (`-s repo,workflow,read:org,gist`).
3. v1.3 branch pushed exactly as-is afterwards; all content (CI workflow included) verified by blob SHA.
4. Two transient test failures during check-up were disk-quota exhaustion on `/tmp` tmpfs (not code); suite passes repeatedly with `--basetemp` on a large filesystem.
5. Two pre-existing snapshot defects left as-is and documented: `v0.7-candidate` (no manifest) and missing `v1.1.0-rc1` snapshot.

## 6. What is NOT on GitHub (explicitly)

- No release **tags** (none existed locally).
- Local debris excluded by `.gitignore`: `.venv/` (5.4 GB), `.pytest_cache/`, `__pycache__/`, `*.pyc`, `*.egg-info/`, `/tmp` artifacts.
- `versions/v0.7-candidate/` has no manifest (pre-existing).
- No `v1.1.0-rc1` snapshot dir (pre-existing gap; content reachable via `feature/final-release-2026-11`).
- The 15 stray `.pyc` files inside `versions/v0.2-baseline/` were untracked and were not migrated.

## 7. Completion state

**Migration: 100% complete.** All 9 branches present on the PRIVATE GitHub repo, all tree/head SHAs verified byte-identical, privacy confirmed, default branch `master`, `versions/` history and snapshots migrated, NTokenizer untouched.
