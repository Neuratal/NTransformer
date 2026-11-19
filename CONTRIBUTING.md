# Contributing to NTransformer

Thank you for your interest in contributing. NTransformer is a research and
engineering framework — correctness, reproducibility and honesty about what is
measured and what is not take priority over feature count.

## Ground rules

1. **Never claim what is not measured.** Benchmarks must be reproducible with
   the harness in `benchmarks/` and recorded with seed/date/hardware.
2. **Never fabricate results.** Failed experiments are documented in
   `docs/optimization_log.md` and rejected-experiments registers; keep that
   discipline.
3. **NTokenizer is a separate project.** Do not modify, vendor or re-implement
   NTokenizer here. Token IDs are consumed verbatim.
4. **Honesty over polish.** If a configuration has not been validated end to
   end, say so. If tests skip, say so.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python -m pytest tests/ -q
```

The full test suite (460 passed, 3 skipped on the development machine) is the
entry bar for any change. CPU-only environments run the same suite with
CUDA-dependent tests skipped automatically.

## Filing issues

- Bugs: attach the exact traceback, `python -V`, `pip show torch`, and the
  smallest reproducer. Include whether the issue reproduces on random init
  or only with a trained checkpoint.
- Proposals: open a discussion first for architecture changes; implementation
  PRs without prior discussion may be rejected.

## Pull requests

- One concern per PR.
- Add tests for every behavioral change (unit +, where applicable, CUDA-skip).
- Keep `neuratal` import light; heavy subsystems are imported lazily.
- Update docs when behavior changes; reference an issue when relevant.
- All PRs must pass CI (`.github/workflows/tests.yml`).

## Licensing

By submitting a contribution you agree that your contribution is licensed
under Apache License 2.0, the same license as the project.

## Architecture decisions

Track open proposals in `docs/rnd_report.md`. Experiments rejected for lack of
reproducible wins are recorded there; do not re-litigate them without new data.
