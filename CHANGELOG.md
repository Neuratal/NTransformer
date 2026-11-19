# Changelog — NTransformer

## [1.4.0] — 2026-11-19

Stable software release.

- Internal audit reports retained at `docs/internal/` for transparency.
### Added
- **Long-context paths** (v1.4): gradient checkpointing for 16K/32K training;
  last-position-only forward pass for bounded memory.
- SPDX license identifiers and Apache-2.0 headers on all Python sources.
- `LICENSE` (Apache-2.0) at repository root; `NOTICE` added.
- `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`.
- `docs/architecture.md`, `docs/context-length.md`, `docs/inference.md`,
  `docs/training.md`, `docs/benchmarking.md`, `docs/comparison.md`,
  `docs/limitations.md`, `docs/tokenizer-integration.md`.
- Engine-final history entry with `{step, tokens_seen, final: true}` appended
  on loop exit.
- `TrainConfig` gains `deadline_unix`, `progress_log_seconds`,
  `save_final_checkpoint`.

### Changed
- License finalized to Apache-2.0 (dual MIT OR Apache removed).
- `pyproject.toml` metadata corrected and expanded (license, classifiers, URLs, keywords).
- README rewritten to honest-release quality: no quality claims on untrained
  weights, full feature matrix, complete NTokenizer boundary documentation.
- Benchmark numbers re-read from `benchmarks/results/`; no extrapolation.
- Examples tested end-to-end on v1.4.0 source.
- Version bumped 1.4.0rc1 → 1.4.0.

### Verified
- Full suite: 460 passed, 3 skipped (no GPU regression).
- 16K/32K correctness paths tested.
- NTokenizer end-to-end integration test green.

### Not included
- Trained weights. No quality claims are made.
- Multi-GPU/NCCL training evidence (CPU `gloo` DDP only).
- New benchmark hardware results (pending).

## [1.4.0rc1] — 2026-11-17

- Bug remediation release candidate; all tracked issues resolved.
- Long-context preparation; preliminary 16K/32K attention paths.
- Test suite: 460 passed, 3 skipped.

See `versions/v1.4.0-candidate/` for the frozen snapshot.

## [1.3.0] — 2026-11-16

- Scheduler resume verification (bit-exact), sharded safetensors atomic writes,
  checkpoint integrity reporting.

See `versions/v1.3.0-candidate/`.

## [1.2.0] — 2026-11-15

- Full file-by-file audit; bug registry confirmed; release notes.

See `versions/v1.2.0-candidate/`.

## [1.1.0rc1] — 2026-11-14

- v1.1 release candidate; registry audit and MoE path hardening.

## [1.0.0rc1] — 2026-11-13

- Initial stable-engine snapshot.

## [0.4] — 2026-11-12

- Static-shape CUDA-graph decode (opt-in).
