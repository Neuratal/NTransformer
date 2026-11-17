# NTransformer — Engineering Check-Up Report (v1.3.0-candidate)

Date: 2026-11-17 · Branch `feature/v1.3-enterprise-2026-11` · Environment: RTX 4060 Laptop (sm_89, 8 GB, 45 W), CUDA 13.2, torch 2.15.0+cu132, Python 3.14.4 · NTokenizer v0.3-polished fingerprint verified unchanged.

Scope: full critical-path audit — architecture, training engine, checkpoint integrity, numerics. Every claim is backed by a command and a result. Honesty labels: **measured** vs **analyzed** vs **unverified**.

## A. Architecture audit

| Component | Assessment | Evidence |
|---|---|---|
| Decoder-only pre-norm topology | correct | `tests/test_block.py`, `test_model.py` green; residual order standard |
| RMSNorm (fused path, fp32 stats) | correct & faster | CUDA op-level: fused 0.0145 ms vs manual 0.0407 ms @ [4,512,512]; fwd/bwd parity CPU/CUDA × fp32/bf16/fp16 (Δ ≤ 2.4e-07 fp32, 0.0 bf16/fp16) — `docs/release_readiness.md` |
| RoPE (rotate-half, fp32 tables, partial-rotary) | correct | `test_rope.py` 17 tests; abs-positions for cached decode |
| GQA + SDPA (regime dispatch) | correct | flash chosen for bf16; no reproducible winner to change anything (3 interleaved rounds, clock noise); `test_attention.py` green; `enable_gqa` with explicit warned fallback |
| Causal masking | correct | is_causal only when q_len==kv_len; lower-right mask for chunked prefill; single-token decode unmasked |
| KV cache (preallocated, commit protocol) | correct | cached vs uncached parity: **exact token equality** (`generate`); overflow → hard `CacheOverflowError` on misuse |
| SwiGLU (fused gate+up GEMM) | correct | `test_mlp.py` 8 tests; byte-identical to unfused math |
| Factorized tied embeddings | correct | tied matrix serialized exactly once (round-trip tests assert) |
| Dense path | stable baseline | suite green; numerics changed only ≤1-ulp via fused RMSNorm (documented) |
| MoE: router (fp32 logits/softmax) | correct | fp32 routing preserved; no path narrows routing precision |
| MoE: dispatch (gather-once/scatter-once) | correct | v1.1 optimization; same math, fewer launches; module fwd B2T128 −16 % vs v1.0 |
| MoE: capacity | correct | pre-capacity counts; kept+dropped == total; padding masked exactly zero |
| MoE: aux loss + z-loss | correct | formulas verified in tests; both stable in saturated-router test |
| MoE: empty/starved experts | correct | no NaN; grads finite; eager leaves None (never garbage) |
| Mixed Dense/MoE layers | correct | `test_moe_engine.py`, `test_moe_core.py` |
| Generation (greedy + sampling) | correct | cached/uncached parity **exact**; EOS semantics; `test_generation.py` |

**No architectural change was made in this pass.** The only numeric path added (v1.2 fused autocast normalize) is documented and tolerance-tested.

## B. Training engine audit

| Item | Status | Evidence |
|---|---|---|
| LR scheduler warmup→cosine | **correct after v1.2/v1.3 fixes** | v1.2: step-0 LR = `lr_at(0)`, resume reapplies `lr_at(step-1)`; v1.3: **bitwise round-trip** — uninterrupted vs resumed bitwise identical at K ∈ {0,1,4,5,8,15,16} for LRs, params, optimizer moments, loss, scheduler counter — `docs/scheduler_resume_verification.md` |
| Scheduler state precision | **fixed** | was float32 (`~1e-7` drift per resume) → now int64/float64 (`7939bbe`) |
| Gradient accumulation | correct | per optimizer step; clipping after accumulation; AdamW ≥2-D weight decay split |
| AMP / bf16 | correct | fp32 master weights; chunked CE fp32 accumulation (parity rel L2 = 1.53e-07 vs standard CE, fp32) |
| FP16 GradScaler | correct | unscale-before-clip, scale/growth-tracker checkpoint round-trip tests |
| Validation loop | correct | non-interactive; finite metrics asserted |
| Resume determinism (data order) | documented: re-permuted under shuffle (not claimed bit-identical); exact for shuffle=False | documented in code + report |
| Optimizer state restore | exact | `torch.equal` across resume for every optimizer tensor (tests in `test_engine_regressions.py`) |
| RNG restore | exact | CPU & CUDA byte-exact |
| 75M dense chain | green | 2-step bf16 CUDA run, resume restores step 2, params finite & updated |
| Chunked CE | correct & memory-efficient | gradient parity rel L2 = 1.03e-06 (full test) / 1.53e-07 (fp32 spot check); 88 ms vs 62 ms standard CE for 1024 tokens — the intended memory trade (1.29 GB vs 4.29 GB) |

**No bug found on this pass beyond the four scheduler/integrity fixes already landed.**

## C. Checkpoint & data integrity audit

| Case | Result |
|---|---|
| Single-file save/load round-trip | bitwise equal weights, tied matrix serialized once, atomic temp+rename |
| Sharded save/load | byte-exact; `index.json` `total_size` verified; **per-shard SHA-256** recorded at save, verified at load (default on; `verify_shard_hashes=False` opt-out documented) |
| Same-size stale shard replacement | rejected with hash mismatch (v1.3, `test_sharded_tampered_shard_rejected`) |
| Missing shard / extra keys / duplicates / archive mismatch | rejected with precise messages |
| Truncated safetensors / config.json corrupt | rejected |
| Interrupted writes | atomic rename; temp files unlinked on failure; no torn file can be observed |
| Wrong architecture / wrong tokenizer / out-of-range optimizer index | rejected with specific errors |

**Measured:** 75M-model checkpoint: single save 0.071 s / load 0.514 s; sharded save 0.241 s / hash-verified load 0.688 s; total size 616.7 MB. **Hash overhead ≈ 63 ms per 37 MB model** (default on).

## D. Numerical anomalies hunted (none found)

Reductions reviewed: CE fp64 test showed every label-less case backpropagates zeros without NaN; RMSNorm fp32 CUDA 2.4e-07 (kernel reduction-order class); `topk_weights` normalize clamps at 1e-12 (top-1 gate exactly 1.0); router fp32 everywhere; z-loss finite under ×1000 logits.

## E. Performance measurements (protocol identical throughout)

Warmup 10, repeats 25, medians; GPU clocks observed 210 MHz idle → 2565 MHz under load (power-capped 45 W); all percent-level deltas treated as noise.

| Configuration | Result |
|---|---|
| Dev preset prefill fp32 (T=128/512/1024) | 1.810 / 3.601 / 6.083 ms |
| Dev preset prefill bf16 (T=128/512/1024) | 1.470 / 2.365 / 4.179 ms |
| Train step (dev preset, bf16 autocast, B2/T128) | dense **23.2 ms** · MoE-e8k2 **24.5 ms** |
| Fused RMSNorm op (D=512, [4,512,512]) | 0.0145 ms (manual 0.0407 ms, 2.8×) |
| MoE module fwd/bwd (H=512, E=8, top-2) | fwd 2.153 (B2T512) / 1.375–3.385 ms; bwd 5.565 / 3.345 / 10.2 ms |
| Peak VRAM B2/T256 (bf16, autocast) | dense **1035 MiB** · MoE **1083 MiB** |
| MoE routing efficiency | 2× active expert FLOPs → ~4–4.6× slower than dense at tiny GEMM scale (launch-bound); consistent with v1.1 ratios |
| Chunked CE vs standard (1024 tokens) | 76 ms vs 53 ms; 1.29 GB vs 4.29 GB — memory trade as designed |

**Bottleneck:** the 262,144-class logits GEMM (fp32) dominates device time; attention ~20 %; kernel launch overhead is significant at dev scale.

## F. CPU compatibility

CPU-parametrized suite passes (`tests/` runs CPU-only boxes via skip markers); development preset trains/validates on CPU with synthetic batches; integration tests requiring CUDA skip cleanly elsewhere.

## G. What was verified but not changed

- KV cache layout (preallocated, `commit()` protocol) — correct; no defect found.
- Sequence packing (`TokenDataset` greedy windows, labels=next-token) — correct.
- Generation `cat` O(T²) copies — accepted at this scale (launch-bound).
- `.gitignore`, packaging, presets (renamed honestly: `5.6b`, `5.7b-moe`).

## H. Explicitly unverified

- Multi-GPU/multi-node anything; DDP MoE `find_unused_parameters=True` fix is inspection-only (no multi-GPU on this machine).
- Checkpoint crash consistency under power loss (no fault injection).
- Shuffle-order data replay after resume (re-permuted, documented).
- >100M-parameter step at full scale; only 75M validated end-to-end.
- Language quality (synthetic data only).

## I. Verdict

Architecture correct; training engine deterministic-after-fix, exactly restorable; checkpointing robust (hash-verified); performance understood and honestly limited; **no production claim is made anywhere.**
