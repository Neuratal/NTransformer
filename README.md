# Neuratal Transformer

A decoder-only, autoregressive Transformer core for the **Neuratal** language
model project — designed to consume **NTokenizer v0.3** (byte-level BPE,
vocab **262,144**) token IDs directly.

**Status: v1.0 engineering candidate.** Architecture, inference, and a training
engine now exist. Dense training was validated end-to-end on the project's
8 GB GPU with synthetic data only; no real corpus pretraining has been run and
no quality claims are made. Opt-in Mixture-of-Experts (MoE) is implemented,
tested, checkpointed, and resume-tested at small scale. See
[Known limitations](#known-limitations), `docs/moe_integration.md`, and
`docs/final_report.md`.

## Architecture (V1)

| Component | Choice |
|---|---|
| Topology | Decoder-only, pre-norm, residual connections |
| Normalization | RMSNorm (fp32 statistics, no bias) |
| Positions | RoPE (GPT-NeoX rotate-half), offset-aware for KV-cache decoding |
| Attention | Grouped-query attention (separate Q/KV head counts) via PyTorch SDPA, explicit causal regimes |
| MLP | SwiGLU (bias-free); opt-in top-k MoE with independent SwiGLU experts |
| Embeddings | **Factorized + tied** (ALBERT-style): one shared `V×F` matrix feeds input projection (`F→D`) and output projection (`D→F`); logits = `(h @ W_outᵀ) @ Eᵀ` |
| Weight tying | Mandatory in V1; the tied matrix exists exactly once (the output head owns no vocabulary parameter) |
| Serialization | safetensors + `config.json` (strict key/shape validation) |
| Precision | fp32 reference path; bf16/fp16 storage supported; optional `fp32_logits` |

Design details, measured facts, and the full rationale: `docs/architecture_audit.md`.

## NTokenizer compatibility

- `vocab_size = 262144`; token IDs are consumed **verbatim**; IDs are never remapped.
- Special tokens (`<|bos|>`=262137 … `<|assistant|>`=262143) are ordinary rows of the embedding matrix; the model has no special casing for them.
- Tokenization/detokenization are the tokenizer's responsibility; this package imports with no tokenizer present.
- The model never injects BOS/EOS implicitly.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # runtime: torch>=2.4, safetensors>=0.4
pip install -e .[dev]     # + pytest, numpy for development
```

## Quickstart

```python
import torch
from neuratal import NeuratalConfig, NeuratalModel, generate

model = NeuratalModel(NeuratalConfig()).eval()   # development preset
ids = torch.tensor([[168069, 5431, 33]])          # NTokenizer("Hello, world!")
out = generate(model, ids, max_new_tokens=16)
```

End-to-end with the real tokenizer (encode → model → decode), with verified
tokenizer ↔ checkpoint compatibility:

```bash
python examples/neuratal_inference.py --prompt "Merhaba, dünya!" --max-new-tokens 16
python examples/neuratal_inference.py --checkpoint path/to/ckpt  # loads + verifies
python examples/neuratal_inference.py --save-checkpoint path/to/ckpt  # writes sidecar
```

**The development preset is untrained.** Its outputs are gibberish; the example
validates plumbing, not quality. No checkpoints of a trained model exist.

## Development / testing preset

The dataclass defaults are the approved development preset (full vocabulary, fast on CPU and CUDA):

| `vocab_size` | `hidden_size` | `layers` | `heads` | `kv_heads` | `embedding_size` | `intermediate_size` | `ctx` |
|---|---|---|---|---|---|---|---|
| 262,144 | 256 | 2 | 4 | 2 | 32 | 640 | 2,048 |

Parameter count (computed by the implementation, not estimated): **9,782,528**
(`model.parameter_report()`).

## Reference only: possible ~50M configuration

Not a training configuration. Measured from the implemented model:

```
vocab=262144  hidden=512  layers=11  heads=8  kv_heads=2  embedding_size=64
intermediate_size=1408    →  total = 47,853,056 parameters
```

Embedding matrix 16.8M · in/out projections 65,536 · attention 7.2M · MLP 23.8M · norms 11,776.
Full-size tied embedding at D=512 would be **134.2M** — factorization is what makes
a ~50M model with a 262k vocabulary feasible. The final training architecture is a
separate decision to be made with training-phase measurements.

## Configuration

`NeuratalConfig` validates everything on construction; invalid configs raise
`ValueError` with precise messages (nothing is silently corrected). Defaults:
`rope_theta=10000`, `rms_norm_eps=1e-5`, dropouts `0.0`. Highlights:

- `embedding_size=None` → full-width (F = hidden_size); `1 ≤ F ≤ hidden_size`
- `intermediate_size=None` → `round(8/3 · hidden)` to a multiple of 256 (resolved and serialized)
- `num_key_value_heads=None` → defaults to `num_attention_heads`
- `attention_implementation`: `"sdpa"` (default) or `"reference"` (pure-PyTorch correctness/debug path)
- `tie_word_embeddings=False` → **explicit error** in V1 (tying is mandatory)
- `logits_chunk_size`: exact chunked projection (see Memory notes)
- `fp32_logits`: compute the head in fp32
- `bos_token_id` / `eos_token_id` / `pad_token_id`: generation fallbacks; never
  injected implicitly
- `num_experts=0` keeps the dense path byte-for-byte; `num_experts>=2` enables
  MoE (`num_experts=1` is rejected). `moe_layer_indices` or
  `moe_layer_frequency` selects which layers are MoE; dense and MoE layers can
  coexist. `num_experts_per_token` is top-k; `capacity_factor=None` disables
  token dropping. See [Mixture of Experts](#mixture-of-experts-opt-in).

Serialized checkpoints store the **resolved** config; loading re-validates.

## Memory notes (262k-vocab logits)

`logits_chunk_size` reduces only the *transient* workspace inside the projection
(exact per-row math; bitwise equality is not guaranteed across chunk boundaries —
measured ≤ ~6e-6 fp32 ulps). It does **not** reduce the memory of a fully
materialized `[batch, sequence, vocab]` logits tensor (at B=8, T=512, fp32 that is
~4.3 GB). Training therefore uses `chunked_linear_cross_entropy`, which never
materializes the full `[B, T, V]` tensor. `fp32_logits=True` doubles transient
head memory for extra precision.

## KV cache and generation

- Per-layer preallocated cache (`cache_capacity`, default = `max_position_embeddings`); overflow raises `CacheOverflowError`.
- RoPE positions are absolute (`arange(past, past+T)`); cached K is rotated before storage.
- SDPA `is_causal` is used only when `query_len == kv_len`; single-token decode uses no mask; chunked prefill uses an explicit lower-right-aligned boolean mask.
- `generate()`: greedy or temperature/top-k sampling, EOS early stop (sequences stopping early are padded with `pad_token_id`/EOS), fresh cache per call.
- Cached vs. uncached inference equivalence is tested (token-by-token vs. prefill, multi-chunk, cache offsets).

## Tokenizer integration

The core never imports the tokenizer. `neuratal/integration/` is the only layer
that knows about NTokenizer:

```python
from neuratal.integration import NTokenizerAdapter, load_checkpoint_with_tokenizer

adapter = NTokenizerAdapter("../NTokenizer/output/v0.3/ntokenizer-0.3-256k.ntok")
ids = adapter.encode("Merhaba, dünya!")     # verified handle (CRC + SHA-256 + specials)
model, adapter, report = load_checkpoint_with_tokenizer("ckpt_dir/", ntok_path)
```

- Manifest (`neuratal-tokenizer-manifest/1`) binds a checkpoint to the exact
  artifact: SHA-256, special-token map, vocab size, versions.
- `write_tokenizer_sidecar()` writes `ckpt_dir/tokenizer.json`;
  `load_checkpoint_with_tokenizer()` verifies it on load.
- Checkpoints without a sidecar load exactly as before (backward compatible).
- Cross-artifact mismatches (e.g., v0.2 vs v0.3) and config/tokenizer EOS
  mismatches are detected and rejected. See `docs/ntokenizer_compatibility.md`.

## Serialization

```python
from neuratal import save_pretrained, load_pretrained
save_pretrained(model, "ckpt_dir/")
model = load_pretrained("ckpt_dir/", map_location="cpu")
```

- `config.json` + `model.safetensors`, atomic writes (temp + rename)
- Save: fails if any two tensors share storage (they never do by construction)
- Load: validates JSON structure, `format_version`, architecture name, unknown config keys, **missing** keys, **unexpected** keys, and per-tensor shapes
- The tied matrix is serialized exactly once
- Loading preserves stored dtypes (no silent casts)
- Local **sharded** checkpoints (`model.safetensors.index.json` + N shards) are
  available for large models: `save_pretrained_sharded` / `load_pretrained_sharded`.
  This is a local format, **not** a distributed/resharded checkpoint system.

## Mixture of Experts (opt-in)

MoE is disabled by default and never changes the dense path. Enabling it:

```python
from neuratal import NeuratalConfig, NeuratalModel

cfg = NeuratalConfig(
    num_experts=8,               # 0 = dense; 1 is rejected
    num_experts_per_token=2,     # top-k
    expert_hidden_dim=1728,      # per-expert SwiGLU width
    moe_layer_frequency=1,       # or moe_layer_indices=[...]
    shared_expert=False,
)
model = NeuratalModel(cfg)
print(model.active_parameters_per_token())   # architecture-level count
```

- Each expert is an independent bias-free SwiGLU; the router is fp32 top-k
  softmax with normalized selected gates.
- Dispatch sorts token assignments by expert once and runs one loop over
  experts (never over tokens). Capacity dropping is opt-in.
- Auxiliary load-balance and optional z-loss are added to the training loss only
  when `compute_aux=True`; inference/generation does not build that graph.
- The training engine logs `aux_loss`, `router_z_loss`, `tokens_per_expert`,
  routing entropy, and dropped assignments. MoE and dense layers may be mixed
  in one model.
- Training/optimizer/checkpoint/resume paths are architecture-aware; loading a
  dense checkpoint into a MoE config (or vice versa) fails with a specific
  error instead of silently mismatching.

Presets: `small-moe`, `1b-moe`, `5.7b-moe`, `30b-moe`, `100b-moe`, `200b-moe`,
`500b-moe`. Counts are measured from the real module tree on the meta device and
reported as **total** and **active per token**. Every expert still costs
checkpoint and optimizer storage.

Measured small-scale smoke test (RTX 4060 Laptop, 8 GB): `small-moe`
(402,301,056 total / 123,609,216 active) ran two BF16 steps at **3.587 GiB**
peak with finite loss and all 8 experts receiving tokens; checkpoint metadata
and resume worked. This validates plumbing, not model quality.

Honest boundaries: DDP MoE router statistics are per-rank (global load balance
is not synchronized), expert/tensor/pipeline parallelism is **not** implemented,
and `zero1/zero2/zero3/fsdp` in `estimate()` are accounting models only. Full
design, benchmark detail, and the scale-readiness table are in
`docs/moe_integration.md`.

## Experimental attention types

`attention_type="gated_delta"` (opt-in, default `"gqa"`) swaps GQA for a
Gated DeltaNet mixer (Kimi-K3/Qwen3.8-Flash-Next "KDA" lineage, arXiv:2412.06464).
Measured trade: **128 KiB O(1) state vs 4 MiB KV cache** at this dev scale —
a 32× memory win — but decode latency regresses at every measured context
(~1.0–1.1 vs 0.75–0.8 ms/token) and quality is unverifiable without training.
Not promoted; opt-in retained for memory-constrained long-context targets.
See `docs/rnd_report.md` and `versions/v0.5-candidate/`.

## Experimental decode path

`generate(model, ids, max_new_tokens=N, experimental_cuda_graphs=True)` replays
the decode step as a CUDA graph (CUDA, batch=1, dropout=0). Bitwise-identical
logits to the eager path; measured 1.28–1.42× at short/medium prompts. Default
stays eager. See `docs/rnd_report.md` and `versions/v0.4/`.

## Testing

```bash
python -m pytest tests/          # CPU + CUDA (params auto-enabled when CUDA exists)
python -m pytest tests/ -k cuda  # CUDA-only subset
python -m pytest tests/test_ntokenizer_integration.py  # integration suite
```

397 collected at time of writing (**394 passed, 3 skipped** — the CUDA-on-CPU
probes); CUDA results measured on an RTX 4060 Laptop GPU (torch 2.15.0+cu132)
— see `docs/moe_integration.md`, `docs/final_report.md`,
`docs/ntokenizer_compatibility.md`, and `docs/benchmark_results.md`. Tests use
the full 262,144-token vocabulary.

## Known limitations

- V1 is untrained; no quality claims are made anywhere. Dense/MoE training runs
  are synthetic plumbing validations only.
- `enable_gqa=True` on SDPA is verified on this machine's build (torch 2.15.0+cu132,
  RTX 4060, sm_89) and falls back explicitly (with a one-time warning) to KV-head
  expansion on builds that reject it. Cross-device behavior is tolerance-checked,
  not bit-identical.
- Chunked cross-entropy avoids the full `[B, T, V]` materialization; the logits
  chunk itself is still bounded by `logits_chunk_size` and vocab width.
- MoE is a readable reference implementation: a Python loop over experts, no
  fused all-to-all, no expert parallelism, no capacity-aware kernel. DDP router
  statistics are per-rank and not globally synchronized.
- Local sharded checkpoints are not distributed checkpoints: every rank owns a
  full state dict; there is no resharding or asynchronous upload.
- `hidden_dropout` applies to residual-branch outputs; dropout defaults to 0.0.
- `generate()` is a minimal greedy/sampling helper, not a serving stack.
- The `reference` attention implementation exists for correctness testing/debugging;
  it is not a performance path.
- No benchmarks are claimed as production numbers; see the reports for what was
  measured and how.

## Project boundaries (V1)

Implemented: architecture, inference, training engine, chunked cross-entropy,
dense presets 50M–500B, opt-in MoE with presets, local sharded checkpoints, and a
closed-form memory estimator. Not implemented: real-corpus pretraining,
production distributed training (ZeRO/FSDP are estimates only), expert/tensor/
pipeline parallelism, custom fused kernels, and MLA/Mamba variants. See
`docs/architecture_audit.md` and `docs/moe_integration.md` for deferrals and
re-open triggers.

## Benchmarks

```bash
python benchmarks/benchmark_environment.py   # fingerprint -> results/*.json
python benchmarks/benchmark_tokenizer.py     # encode/decode throughput
python benchmarks/benchmark_model.py --device cuda --dtype float32
python benchmarks/benchmark_kv_cache.py --device cuda
python benchmarks/benchmark_attention.py --device cuda
python benchmarks/benchmark_end_to_end.py --device cuda
python benchmarks/benchmark_memory.py --device cuda --budget-mib 4600
```

Reproducible (seed 42), timestamped JSON results in `benchmarks/results/`;
methodology in `docs/benchmark_methodology.md`; measured tables in
`docs/benchmark_results.md`; optimization attempts in `docs/optimization_log.md`.

## Layout

```
neuratal/            # core package: config, model/*, serialization/*
neuratal/integration/  # optional NTokenizer boundary (adapter, manifest, sidecar)
tests/               # pytest suite (CPU + CUDA)
benchmarks/          # benchmark suite + results/*.json
examples/inference.py          # minimal pipeline smoke demo
examples/neuratal_inference.py # full integration example (with verification)
examples/benchmark.py          # dev-preset spot benchmark
docs/architecture_audit.md    # Phase 1 audit
docs/final_report.md          # Phase 5 report (measured results)
docs/moe_integration.md       # MoE design, measured smoke tests, scale readiness
docs/ntokenizer_compatibility.md
docs/benchmark_methodology.md
docs/benchmark_results.md
docs/optimization_log.md
```
