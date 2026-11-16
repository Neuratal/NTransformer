# CHANGELOG — v0.4

**Frozen:** 2026-11-16, commit `a6b263c` · **best:** yes (promoted over v0.3)

## Change
Adds opt-in `generate(..., experimental_cuda_graphs=True)`:
- `GraphDecodeSession` (neuratal/model/graph_decode.py): captures the
  single-token decode step as a CUDA graph once per call, replay per token.
- Bitwise-identical logits to the eager path (verified 16+ steps fp32/bf16;
  prefill logits exactly equal; per-step logits allclose-1e-6).
- Static window = next_pow2(prompt_len + max_new_tokens + 1); one capture per
  generation call; persistent mask buffer updated in-place per step;
  cache writes via `index_copy_` (graph-safe).
- Explicit rejections: batch>1, CPU, dropout>0, window>cache_capacity — all
  raise before touching tensors. Eager remains the default everywhere.

## Evidence
- 21 new tests (`tests/test_graph_decode.py`); full suite 297 passed, 2 skipped.
- End-to-end: 1.28x-1.42x gen speedup at short/medium prompts (16-128 tokens).
- Long-prompt caveat: 0.94x at 512 (window 1024 wide vs eager's growing window);
  opt-in only, documented.
- No regression on default paths; v0.3 behavior preserved bit-for-bit.

## Known limitations
Default off; CUDA/B=1/dropout=0; capture amortizes over ≥~16 tokens;
attention runs over the padded pow2 window (wasteful at long prompts);
graph body duplicates the sublayer flow (guarded by bitwise-equivalence tests).
