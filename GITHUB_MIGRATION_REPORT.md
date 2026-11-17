# NTransformer — GitHub Private Migration Report

Date: 2026-11-17 · Repository: **https://github.com/Neuratal/NTransformer** (PRIVATE)
Source: local checkout `/home/justerdem14/Masaüstü/Team N/NTransformer` (branch `feature/v1.3-enterprise-2026-11`, HEAD `d17391d`)

---

## 1. Repository lifecycle

| Step | Status | Evidence |
|---|---|---|
| Target repository existence check | **none existed** (`gh api repos/Neuratal/NTransformer` → 404 before creation) | verified pre-creation |
| Creation | **created 2026-11-17T09:43:49Z** by `gh repo create Neuratal/NTransformer --private` | `gh repo view` output |
| Visibility | **PRIVATE** (verified via API: `"visibility": "PRIVATE"`, `"private": true`) | `gh repo view Neuratal/NTransformer --json visibility,private` |
| Default branch | `master` (set after creation; repo was empty at creation so alphabetical `cand/v0.6-mla` was chosen by GitHub) | API `PATCH …/default_branch → master` |
| NTokenizer boundary | `Neuratal/neuratal-tokenizer` **not accessed, modified, or pushed** at any point | commands used only `repos/Neuratal/NTransformer*` |

## 2. Branch migration

Local: 9 branches (`master` + 8 feature/candidate). Remote: **8 of 9 pushed**; 1 blocked on a GitHub permission, reported honestly below.

| Branch | Local SHA | Remote SHA | Tree SHA match | Status |
|---|---|---|---|---|
| `master` | `14f6fe0` | `14f6fe0` | ✓ `55bb3e8` | migrated |
| `cand/v0.6-mla` | — | identical | ✓ `b1ccc78` | migrated |
| `cand/v0.6-wy-chunkwise` | — | identical | ✓ `6d78a8c` | migrated |
| `cand/v0.7-sliding-window` | — | identical | ✓ `84c9fc4` | migrated |
| `feature/moe-integration-2026-11` | — | identical | ✓ `e34035f` | migrated |
| `feature/v1.0-scaling-2026-11` | — | identical | ✓ `22e18e25` | migrated |
| `feature/v1.2-audit-2026-11` | — | identical | ✓ `ca2ce63f` | migrated |
| `feature/final-release-2026-11` | — | identical | ✓ `f4e8556f` | migrated |
| **`feature/v1.3-enterprise-2026-11`** | `d17391d` | **absent** | — | **blocked: token lacks `workflow` scope** |

Push command used: `git push origin --all` (no `--force`, no `--mirror`; fresh empty remote, so no clobbering was possible).

### Blocking issue (open, user action required)

`feature/v1.3-enterprise-2026-11` contains `.github/workflows/tests.yml`. The OAuth token granted to the `gh` CLI has scopes `gist, read:org, repo` — GitHub refuses to let an OAuth App create/update **any** `.github/workflows/*` file without the `workflow` scope (`refusing to allow an OAuth App to create or update workflow`), blocking the whole branch push (it is otherwise fine: 68 commits, tiny tree). The user chose **"Grant workflow scope"**; the helper was launched in the background (`gh auth refresh -h github.com -s workflow`) and **is waiting for device-code approval in the browser** — it has NOT completed. Until completion, the v1.3.0-candidate branch exists only locally.

Current state therefore: **the GitHub repo holds 8 branches (through v1.2.0-candidate) — it does NOT yet contain the v1.3.0-candidate branch or the migration/check-up documents.** Nothing is hidden: this is stated in every relevant report below, and completion steps are listed in §6.

## 3. Integrity verification (per branch)

For each of the 8 pushed branches:
1. Remote head SHA == local SHA (exact) ✔
2. Remote root tree SHA == local tree SHA (content byte-equality incl. filenames, modes, symlinks) ✔
3. Spot blob SHA comparisons ✔ (e.g. `README.md` on master; `neuratal/model/moe.py` on v1.2-audit; `versions/v1.3.0-candidate/file_manifest.json` on the freeze branches — all match `git hash-object`)
4. No large blobs on GitHub: `.git` < 5 MiB; largest tracked file 35 KB ✔
5. `versions/` snapshots: all manifests re-verified before push (v0.2-baseline, v0.3, v0.4, v0.5, v0.6, v0.6.1, v0.7.1, v1.0.0-rc1, v1.2.0-candidate, v1.3.0-candidate) — all pass; `v0.7-candidate` has no manifest (pre-existing, documented) ✔

## 4. What is NOT on GitHub (explicitly)

- `feature/v1.3-enterprise-2026-11` (v1.3.0-candidate state, the 7 check-up/migration docs)
- No release **tags** exist in the local repo (none were created; migrating tags was therefore moot — all tags listed as "none" in the inventory)
- Local debris excluded (`.venv/` 5.4 GB, `.pytest_cache/`, `__pycache__/*`, `*.pyc`, egg-info) — verified untracked ✔
- No `v1.1.0-rc1` git tag/snapshot — pre-existing gap, documented ✔

## 5. Failures / incidents encountered (none hidden)

1. One push rejected (workflow scope, above) — carried, not worked around.
2. `git push --all` completed 8/9 branches before the rejection; **no history was rewritten**; `master` was NOT force-pushed (never force-pushed anything).
3. Two pre-existing snapshot defects left as-is: `v0.7-candidate` (no manifest) and missing `v1.1.0-rc1` snapshot — documented in `GITHUB_MIGRATION_INVENTORY.md`.
4. 15 stray `.pyc` files inside `versions/v0.2-baseline/` (untracked, ignored, not migrated) — noted.

## 6. Completion steps (after user authorizes `workflow` scope)

```bash
# user runs (already launched by the background helper; browser approval required):
#   gh auth refresh -h github.com -s workflow
git push origin feature/v1.3-enterprise-2026-11
# then re-verify:
gh api repos/Neuratal/NTransformer/branches --jq '.[].name' | sort
gh api repos/Neuratal/NTransformer --jq '{visibility, private, default_branch}'
```
After push: re-run §3 verification for the v1.3 branch and update this report's status line. Nothing else changes.

## 7. Data-safety conclusion

- All 9 local branches: 1 pending (permission), 8 verified byte-identical on GitHub.
- No local file was modified, reset, or deleted in this migration; `master` untouched; no force pushes; NTokenizer untouched.
- Migration is **80% complete** until the scope grant lands; it is otherwise green.
