# Version History — NTransformer

Chronological, immutable. Historical entries are not rewritten; only new
entries are appended. "Best" never moves automatically — see registry.

| Version | Date | Milestone | Status / outcome |
|---|---|---|---|
| v0.3 | 2026-11 | original baseline (fused QKV/MLP projections) | frozen; candidate |
| v0.4 | 2026-11 | CUDA-graph decode opt-in | **best (registry rule, promoted)** |
| v0.5/v0.5-candidate | 2026-11 | GDN mixer (opt-in) | rejected for promotion (latency regression); opt-in retained |
| v0.6-candidate | 2026-11 | MLA/latent-KV (opt-in) | rejected; branch-only |
| v0.6.1-candidate | 2026-11 | WY chunkwise GDN | rejected vs dense GQA; archived |
| v0.7-candidate | 2026-11 | sliding-window attention | rejected (20–40 % slower T=2048) |
| v0.7.1-candidate | 2026-11 | SWA bounds-check fix | superseded by v0.6.1; archived |
| v1.0.0-rc1 | 2026-11 | training engine, chunked CE, presets, checkpoints, estimator, DDP path | frozen; not promoted |
| v1.0.0 | — | first stable milestone (status: superseded by v1.1.0rc1 work on this branch) | — |
| v1.1.0rc1 | 2026-11-17 | MoE dispatch gather/scatter-once; fused F.rms_norm; lazy presets/imports; engine fixes (full-width head, fp16 GradScaler, DDP-MoE, checkpoint fingerprint/index guards); preset honesty rename (7b→5.6b, 7b-moe→5.7b-moe); benchmark infra; 413 tests | release candidate; not promoted |
| **v1.2.0-candidate** | 2026-11-17 | ultimate audit: scheduler LR fresh+resume fix (high-severity), sharded integrity check, atomic-save dedup, autocast fused-normalize RMSNorm, 5 new tests; 418 tests | release candidate; not promoted |
| v2.0.0 | future | reserved for a genuine architectural milestone | not scheduled |

## Naming rules (enforced)
- `-candidate`: not yet accepted as stable improvement.
- `-fail`: attempted, rejected with evidence (see `rejected_experiments.md`);
  the identifier is consumed, never recycled.
- `-rc`: passed verification, not yet stable.
- Stable requires correctness + compatibility + reproducibility evidence.
## Registry pointers
- Canonical: `registry/best_version.json`
- Frozen snapshots: `versions/v0.4` … `versions/v1.1.0rc1`
- Candidate freeze: `versions/v1.2.0-candidate/` (commit reference inside)
