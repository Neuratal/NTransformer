# CHANGELOG — v0.2-baseline

**Frozen:** 2026-11-16, commit `c31418e`
**Role:** Immutable baseline (V1 Transformer + NTokenizer integration).

Content:
- Decoder-only pre-norm Transformer: RMSNorm (fp32 stats, eps=1e-5), RoPE
  (rotate-half, partial-rotary capable), GQA (H=4, H_kv=2) via PyTorch SDPA
  with position-regime causal dispatch, SwiGLU MLP, tied factorized embeddings
  (V=262144, F=32, D=256, I=640, L=2), dev preset ctx=2048.
- KV cache (preallocated, commit protocol), generate() (greedy/top-k),
  safetensors + JSON checkpoints (strict validation), NTokenizer v0.3 verified
  integration (adapter + sidecar manifest, SHA-256 pinned).
- Measured: 9,782,528 params; CUDA fp32 prefill T=128 ≈ 2.10–2.23 ms; cached
  decode ≈ 0.67 ms/token; 268 tests pass.
- Known accepted limitations: untrained; CPU launch-bound at this scale;
  full-size materialized logits; untested beyond this machine; audit found and
  fixed host-side validation syncs (undocumented-behavior-free, see
  optimization_log.md).
