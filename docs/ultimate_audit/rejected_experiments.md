# Rejected Experiments — NTransformer (v1.1 → v1.2.0-candidate)

Experiments that were attempted and deliberately not adopted. Recorded with
evidence. None of these are silently dropped from history.

## RX-1 — `torch.compile` on MoEFFN (and generally)
- Hypothesis: remove Python/launch overhead; fuse silu*up and dispatch.
- Measured: 15.07 vs 14.90 ms (fwd+bwd, B4/T512) — within noise; max |Δ| 8.9e-08.
- Rejected: no measurable benefit, environment fragility (Triton FP-fusion
  warnings, compile-time costs), and eager remains the supported path.
- Note: v0.6.1 GDN chunkwise was compiled-accepted then rejected for latency
  regression — different candidate, same caution.

## RX-2 — `torch._grouped_mm` (private aten op) for expert GEMMs
- Hypothesis: grouped GEMM beats the per-expert Python loop.
- Measured: exact-match outputs; realistic imbalanced gate_up+down 0.601 vs
  0.615 ms (−2 %, noise); uniform shapes mixed (some faster, some slower).
- Rejected: a private API with strict layout constraints (bf16, int32 offs,
  contiguous) and no robust win. Not adopted — not even behind a flag, to
  avoid a permanent "experimental" path with maintenance cost.

## RX-3 — Fused weighted RMSNorm under bf16 autocast
- Hypothesis: `F.rms_norm(x_bf16, w_fp32)` dispatches fused like elsewhere.
- Measured: cannot dispatch (dtype mismatch) → composite fallback with
  **0.0156** max error — worse numerics than v1.1's manual fp32 path.
- Rejected: the *reference implementation* in that regime is v1.1's manual
  path; anything else must beat it on numerics first.

## RX-4 — Forcing cuDNN attention over FLASH
- Hypothesis: early micro-measurement showed cuDNN faster (0.023 vs 0.039 ms).
- Re-measured (3 interleaved rounds): no reproducible ordering (spread
  0.17–0.95 ms across rounds for both). Rejected: no forcing, ever;
  SDPA default retained.

## RX-5 — Fixed-capacity `bmm` padding for expert GEMMs
- Hypothesis: pad segments to `capacity`, batched GEMM, fewer launches.
- Rejected without adoption: ≥20–100 % extra expert FLOPs at
  `capacity_factor ≠ 1.0` for ~E kernel launches; only relevant under
  capacity drop, which is off by default (`capacity_factor=None`).

## RX-6 — Chunked-CE "improvement" pass
- Reviewed for v1.2: nothing to adopt. The implementation already hoists
  fp32 weight conversions out of the chunk loop; gradient parity 1.03e-06;
  remaining GEMM cost is irreducible without fused kernels (rejected
  dependencies). `ctx.n_tokens` dead code removed in v1.1.

## RX-7 — Pre-allocated generation buffer (`generated` preallocated)
- Hypothesis: avoid per-step `torch.cat` O(T²) copies.
- Rejected: at this scale decode is launch-bound; copies are microseconds;
  scheme changes `generate()` semantics for no measurable win.

## RX-8 — Estimated "wins" that were clock-state artifacts
- Several benchmark re-runs read 5–25 % "faster" purely from SM clock state
  (2565 vs 3105 MHz). All such observations were discarded as evidence and
  are documented in `benchmark_summary.md` to prevent future false claims.

## Reconsider conditions
- RX-2 could be revisited if torch exposes a stable public grouped-GEMM.
- RX-1 if a stable, deterministic compile mode ships for this GPU family.
- RX-5 if capacity drop becomes the default for large-batch training.
