# LIMITATIONS — v0.2-baseline

Same as `docs/final_report.md` §9 plus integration-phase notes. In short:

1. Untrained; no language quality. Generated text is gibberish.
2. CPU launch-bound at dev scale; CUDA wins wall-clock.
3. Full [B, T, 262144] logits materialized; backward memory-heavy.
4. `enable_gqa` verified only on torch 2.15.0+cu132/sm_89; fallback explicit.
5. torch.compile decode path degenerate on this stack (rejected).
6. Checkpoint format frozen at format_version=1; sidecar is optional and
   additive, never a replacement.
7. Single machine (sm_89 laptop); portability of SDPA backend dispatch not
   guaranteed on other hardware.
8. No training path exists anywhere; benchmarks are architecture/software only.
9. Numerical validation of compile-generated code is opportunistic, not
   guaranteed across environments.
10. Variance run-to-run ±10–20% (shared laptop GPU).
