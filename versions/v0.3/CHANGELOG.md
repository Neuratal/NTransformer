# CHANGELOG — v0.3

**Frozen:** 2026-11-16, commit `e3633be` · **best:** yes (promoted over v0.2-baseline)

## Change
Fused projections, mathematically identical to v0.2:
- Attention: `qkv_proj` [D → (H + 2·H_kv)·d_h] replaces separate `q/k/v_proj`.
- MLP: `gate_up_proj` [D → 2I] replaces separate `gate_proj`/`up_proj`.
- Legacy v0.2 checkpoints migrate transparently at load (row concatenation +
  `UserWarning`); new checkpoints carry fused keys only.
- Parameter count unchanged: 9,782,528. Full test suite green (276 passed,
  2 CUDA-CPU skips, 8 new fused-compat tests: bitwise fused-vs-split math,
  legacy roundtrip < 1e-5 logits, migration idempotence/warning).

## Evidence
See `benchmark_results/` and registry. fp32 prefill: −4…−7% (dev scale,
T=256/512), ~−20% (ref-50M reference, T=512, 3/3 rounds). Decode ≈ wash;
bf16 ≈ wash (clock variance). Environment identical (sm_89, cu132).

## Known limitations
bf16 benefit unproven; decode unchanged; migration is one-way; checkpoint
key layout changed (semantic-identical tensors). Full report preserved in
registry/best_version.json and docs/optimization_log.md.
