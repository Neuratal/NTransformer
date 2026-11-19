# NTransformer Architecture

Status: **matches `neuratal` 1.4.0 source tree (2026-11-19).**

## Design goals

1. **Legibility with correctness.** Every component is independently readable and
   tested; no component is magic.
2. **NTokenizer-native.** Token IDs are consumed verbatim (vocab 262,144); never
   remapped, never re-tokenized in-process; BOS/EOS handling stays the caller's
   responsibility.
3. **Explicit scale limits.** Nothing is trained past ~100M on this hardware;
   presets beyond that are validated as configurations, not as training runs.

## Topology

- Decoder-only, pre-norm residual network.
- `L` identical decoder blocks; residual path runs at the token embedding width.
- **Factorized tied embeddings**: one shared `[262144, F]` matrix feeds the input
  projection and the output head (logits softmax stays at `F` until the final
  GEMM to `262144`). This makes a ~50M V1 feasible (16.8M embedding params vs
  ~134M for a full-width tied matrix).

## Components

| Component | Choice | Notes |
|---|---|---|
| Normalization | RMSNorm (fp32 stats; fused `F.rms_norm` fast path when dtypes match and no autocast is active) | Fused path is opt-in via dtypes and environment; correctness path always available |
| Positions | RoPE (NeoX rotate-half), offset-aware | KV-cache aware; offset passed for decode |
| Attention | Grouped-query attention over PyTorch SDPA | Explicit regimes: `is_causal` only when q_len == kv_len; single-token decode unmasked; chunked prefill uses explicit lower-right mask |
| MLP | SwiGLU (bias-free); opt-in MoE with independent SwiGLU experts | Router = fp32 top-k softmax with normalized gates |
| Output head | Factorized tied → `F→V` GEMM after full-width window | `T → F` hidden state + `F → V` GEMM (never `T → V`) |
| Precision | fp32 reference; bf16/fp16 via autocast; fp16 + GradScaler for fp16 | fp32 router/loss paths are preserved |
| Optimizer | AdamW (decoupled weight decay on ≥2D params) | Warmup → cosine decay, per-step |
| Serialization | safetensors + `config.json` | Atomic writes; strict validation on load |

## Attention regimes

Three explicit regimes are used, never mixed implicitly:

1. **Causal full sequence** (`is_causal=True`) — q_len == kv_len.
2. **Single-token decode** — one query, attends to all previously cached keys.
3. **Chunked prefill** — explicit lower-right-aligned causal mask.

## Long-context support

- 4K–16K training on 8 GB GPU requires `gradient_checkpointing=True`.
- 32K training on the same device is feasible but slow (loss is
  ~`O(seq_len * vocab)` per chunk); no 32K benchmarks are claimed on this hardware.
- 32K inference with KV-cache fits in memory with headroom at prefill time.

## Factorized tied embeddings

Input projection maps `[B, T] → [B, T, F]`; logits path maps `[B, T, F] → [B, T, V]`.
The shared `[V, F]` matrix is the only large embedding parameter.

Recommended `F = 64` for 50M models (`NeuratalConfig.embedding_size=64`).

## MoE (opt-in)

Mixture-of-Experts is disabled by default. When enabled:

- Each expert is an independent bias-free SwiGLU feed-forward.
- Router is fp32 top-k softmax; selected gates are normalized.
- Dispatch: assignments are sorted by expert once; tokens gathered once;
  each expert processes a contiguous slice; single scatter-add recombines.
- Pathological cases (empty experts, starve/imbalance, capacity dropping) are
  all covered by tests.
- Grouped GEMM (`torch._grouped_mm`) was measured and rejected — no win at
  these sizes; private API.

## Scale readiness (honest)

| Size | Config validated | Count measured | fwd tested | bwd tested | single GPU | distributed | production |
|---|---|---|---|---|---|---|---|
| dev preset (9.8M) | yes | yes | yes (CUDA) | yes | yes | no | no |
| 50m–100m | yes | yes | yes (75m chain) | yes (75m) | yes (75m) | no | no |
| 500m–5.6b | yes | yes | no | no | no | no | no |
| 13b–500b | yes | yes | meta only | meta only | meta only | no | no |
| MoE small/75M-class | yes | yes | yes | yes | yes | no | no |
| MoE ≥1B | yes | yes | meta only | meta only | meta only | no | no |
| DDP (2-proc gloo, CPU) | yes | yes | yes | yes | yes (CPU) | yes (gloo) | no |

**No preset is trained or in production.** Sizes above ~100M are validated as
configurations/counts only.
