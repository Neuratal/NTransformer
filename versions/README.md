# Versioned releases — NTransformer

Sequential, immutable snapshots of the implementation. **None are ever deleted or overwritten.**

| Version | Frozen at | Content | Status |
|---|---|---|---|
| v0.2-baseline | `c31418e` | V1 Transformer + NTokenizer integration phase (immutable baseline) | baseline; historical reference |
| v0.3 | `e3633be` | fused qkv_proj + gate_up_proj | promoted (superseded) |
| v0.4 | `a6b263c` | opt-in static-shape CUDA-graph decode (default eager unchanged) | **best** |

Versioning follows this repository's history: the package entered this phase
as v0.1.0 (Transformers) → integration work → frozen `v0.2-baseline`. Candidates
are named v0.3, v0.4, … in order of promotion attempts; a candidate is never
promoted to `best` without (a) all tests green and (b) measurable, reproducible
improvement in at least one benchmark axis without regression on correctness.

Each snapshot contains SOURCE (frozen copy), commit.txt, environment.json,
test_results.txt, file_manifest.json (SHA-256), CHANGELOG.md, LIMITATIONS.md,
benchmark_results/. Decisions live in `registry/best_version.json` and the
per-version CHANGELOG.
