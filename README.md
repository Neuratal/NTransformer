# NTransformer (Neuratal Transformer)

A decoder-only, autoregressive Transformer framework built around
**NTokenizer v0.3-polished** (byte-level BPE, vocabulary **262,144**). It
provides the model, inference, training engine, checkpointing and benchmarks —
no datasets are bundled and no model quality claims are made. Every claim in
this file is either measured on the development machine or explicitly labeled
as an estimate.

> **Status: v1.1.0rc1 (engineering release candidate).** The framework installs,
> tests green (413 passed, 3 skipped, 1 warning), and small
> dense/MoE training runs end-to-end on the 8 GB development GPU with
> synthetic data. **No real corpus has been used; no weights of a trained
> model exist. This is not a production-ready serving system and not a
> training recipe for any scale above ~100M.** See
> [Known limitations](#known-limitations) and
> [Scale readiness](#scale-readiness-honest).

## What this is (and is not)

| This is | This is not |
|---|---|
| A research/engineering framework: architecture, optimizer, checkpoints | A pretrained LLM (nothing has been trained) |
| Tested at dev/50–75M scale on 8 GB VRAM | Validated at 7B+ scale (impossible on this hardware) |
| NTokenizer-compatible by construction (vocab 262,144, IDs verbatim) | A tokenizer (NTokenizer is a separate project, untouched) |
| Honest about what is measured vs estimated | Production-ready distributed training |

## Architecture (V1)

| Component | Choice |
|---|---|
| Topology | Decoder-only, pre-norm, residual connections |
| Normalization | RMSNorm (fp32 statistics; fused `F.rms_norm` fast path when dtypes match and no autocast is active) |
| Positions | RoPE (GPT-NeoX rotate-half), offset-aware for KV-cache decoding |
| Attention | Grouped-query attention via PyTorch SDPA, explicit causal regimes (`is_causal` only when q_len == kv_len; single-token decode unmasked; chunked prefill uses an explicit lower-right-aligned mask) |
| MLP | SwiGLU (bias-free); opt-in top-k **Mixture-of-Experts** with independent SwiGLU experts |
| Embeddings | **Factorized + tied**: one shared `V×F` matrix feeds the input projection and the output head |
| Serialization | safetensors + `config.json` (strict validation); atomic writes |
| Precision | fp32 reference path; bf16/fp16 training via autocast (+ GradScaler for fp16) |

### Design rationale highlights

- **Factorized tied embeddings** make a ~50M model with a 262k vocabulary
  feasible (16.8M embedding params vs 134M for a full-width tied matrix).
- **Chunked linear cross-entropy** (recompute-in-backward) never materializes
  `[B, T, 262144]` logits; memory stays O(chunk) instead of tens of GB.
- **fp32 router, fp32 loss reductions** — numerics are not sacrificed for
  speed anywhere.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # runtime: torch>=2.4, safetensors>=0.4
pip install -e .[dev]       # + pytest, numpy
```

Python ≥ 3.10. Verified with Python 3.14.4, torch 2.15.0+cu132, CUDA 13.2.
No new runtime dependencies were added for the MoE work or the checkpoint
work; this project deliberately keeps its dependency surface minimal.

## Quick start

```python
import torch
from neuratal import NeuratalConfig, NeuratalModel, generate

model = NeuratalModel(NeuratalConfig()).eval()   # development preset
ids = torch.tensor([[168069, 5431, 33]])          # token IDs are consumed verbatim
out = generate(model, ids, max_new_tokens=16)
```

The development preset is **untrained**; its outputs are gibberish. The
example validates plumbing, not quality.

End-to-end with the real tokenizer (needs the NTokenizer v0.3 artifact; set
`NTOKENIZER_ROOT` or pass paths explicitly):

```bash
python examples/neuratal_inference.py --prompt "Merhaba, dünya!" --max-new-tokens 16
```

## Training quick start (synthetic data only)

```python
from neuratal import NeuratalModel, get_preset
from neuratal.training.data import SyntheticTokenDataset
from neuratal.training.engine import TrainConfig, train

model = NeuratalModel(get_preset("75m").config).to("cuda", torch.bfloat16)
ds = SyntheticTokenDataset(16, 32, vocab_size=262_144, seed=1)
cfg = TrainConfig(precision="bfloat16", max_steps=10, out_dir="checkpoints/demo", device="cuda")
model, history = train(model, ds, None, cfg)
```

`SyntheticTokenDataset` generates deterministic pseudo-random token IDs and
exists **only** for plumbing validation and benchmarks. No real corpus is
shipped, downloaded, or used anywhere in this repository.

## NTokenizer compatibility (immutable identity)

- `vocab_size = 262,144`; token IDs are consumed **verbatim** — never remapped.
- Special tokens: BOS 262137, EOS 262138, PAD 262139, UNK 262140,
  SYSTEM 262141, USER 262142, ASSISTANT 262143.
- Fingerprint: `1f0d4ad7c7fb609001b2af4d9741ec5b8a5d7686fa3b92eb8eefd057fea936d9`
  (verified unchanged at the time of this release; see
  `docs/ntokenizer_compatibility.md`).
- Tokenization/detokenization are the tokenizer's responsibility; this package
  imports with no tokenizer present. The core never injects BOS/EOS implicitly.
- `neuratal/integration/` binds a checkpoint to the exact `.ntok` artifact
  (SHA-256 + special-token map + vocab size + CRC) and rejects mismatches.
- **NTokenizer is a separate project and is not modified by this repository.**

## Training pipeline

- Data: pre-tokenized uint32/int64 streams packed greedily into fixed windows
  (`TokenDataset`); synthetic generator for tests. No document-boundary
  masking (packing is lossy by design, as in standard practice).
- Optimizer: AdamW (`weight_decay=0.1` on ≥2-D params), or SGD-momentum;
  warmup → cosine decay (`WarmupCosineSchedule`), stepped per optimizer step.
- Gradient accumulation (micro → optimizer steps), gradient clipping on the
  accumulated gradient, AMP: bf16 autocast (fp32 master weights) or fp16
  autocast + GradScaler with unscale-before-clip.
- MoE auxiliary load-balancing loss (`coef * E * Σ f_e·P_e`) and router
  z-loss are added to the LM loss once per micro-batch; inference never
  builds that graph.
- Validation on a held-out split, per-`log_every_steps` JSONL-style history,
  per-`checkpoint_every_steps` full checkpoints.
- Resume restores model + optimizer + scheduler + RNG + step/token counters,
  validates the recorded tokenizer fingerprint (opt-out), validates optimizer
  state indices against the target model, and restores the GradScaler state
  for fp16 runs. See `docs/training_plan.md`.

## Checkpointing

```python
from neuratal import save_pretrained, load_pretrained          # model weights
from neuratal.training.checkpoint import load_training_checkpoint, resume_training_checkpoint
from neuratal.training.engine import train                      # writes ckpt-<step>/
```

- `config.json` + `model.safetensors`, **atomic writes** everywhere, strict
  structural validation (format version, architecture name, missing/unexpected
  keys, per-tensor shapes, dtypes). Corrupt files are rejected with precise
  errors.
- Training checkpoints add `trainer_state.safetensors` (optimizer/scheduler/
  RNG/counters/scaler) and `trainer_meta.json` (step, tokens, fingerprint,
  precision, architecture). Written atomically as well.
- **Local sharded checkpoints** (`model.safetensors.index.json` + N shards)
  exist for large models: `save_pretrained_sharded` / `load_pretrained_sharded`.
  This is a *local* format — **not** distributed checkpointing, no resharding.
- Wrong-architecture resume is rejected explicitly (dense ↔ MoE); mismatched
  tokenizer fingerprints are rejected explicitly.

## Configuration

`NeuratalConfig` validates everything on construction; invalid values raise
`ValueError` with precise messages — nothing is silently corrected. Highlights:
`embedding_size=None` (full-width), factorized default; `moe_layer_indices` /
`moe_layer_frequency` for MoE placement (dense and MoE layers can coexist);
`capacity_factor=None` (no token dropping); `tie_word_embeddings=False` is an
explicit error (V1 tying is mandatory); resolved defaults are serialized.

## Mixture of Experts (opt-in)

Disabled by default; enabling never changes the dense path.

```python
cfg = NeuratalConfig(
    hidden_size=640, num_hidden_layers=14, num_attention_heads=10,
    num_key_value_heads=2, intermediate_size=1728, embedding_size=64,
    num_experts=8, num_experts_per_token=2, expert_hidden_dim=1728,
)
model = NeuratalModel(cfg)
```

- Each expert is an independent bias-free SwiGLU; the router is an fp32
  top-k softmax with normalized selected gates (bf16 router softmax is not
  trusted).
- Dispatch: assignments are sorted by expert once; tokens are gathered once,
  each expert processes a contiguous slice, and a single scatter-add
  recombines (same byte volume as per-expert gather/scatter, fewer kernel
  launches). One Python loop over experts (never over tokens).
- Empty experts, under-utilized experts, imbalanced routing, top-1/top-2,
  capacity dropping and padding masks are all handled and tested, including
  pathological distributions.
- Performance work (2026-11-17 pass): gather-once/scatter-once dispatch,
  no per-scalar H2D copies, shared `bincount` int64 path; one documented
  host sync (`.tolist()`) for per-expert slices. See
  `benchmarks/benchmark_moe.py` and `benchmarks/results/moe_*.json`.
  Grouped GEMM (`torch._grouped_mm`, private aten op) was measured and
  **rejected** — no meaningful win at these sizes (0.60 vs 0.61 ms) and it is
  a private API. `torch.compile` was measured and rejected (within noise).
- Hard boundaries: no fused all-to-all, no expert parallelism, no global
  router-statistics synchronization across DDP ranks
  (`zero1/zero2/zero3/fsdp` in `estimate()` are accounting models only).

## Presets (measured, not nominal)

Parameter counts are **measured from the real module tree on the meta device**,
not estimated. Numeric suffixes are the rounded measured count, not a marketing
target (e.g. the former `7b` measured 5.57B and is now `5.6b`).

| preset | total params | | preset | total params | active/token |
|---|---|---|---|---|---|
| `50m` | 47,853,056 | | `small-moe` | 402,301,056 | 123,609,216 |
| `75m` | 77,088,896 | | `1b-moe` | 1,074,300,928 | 659,064,832 |
| `100m` | 98,096,896 | | `5.7b-moe` | 5,673,062,400 | 2,351,173,632 |
| `500m` | 496,760,320 | | `30b-moe` | 29,400,112,128 | 8,260,819,968 |
| `1b` | 1,074,104,320 | | `100b-moe` | 99,909,378,048 | 16,157,515,776 |
| `3b` | 2,960,139,008 | | `200b-moe` | 198,474,149,888 | 35,399,610,368 |
| `5.6b` | 5,571,350,528 | | `500b-moe` | 483,293,937,664 | 77,419,528,192 |
| `13b` | 11,915,686,912 | | | | |
| `30b` | 30,235,631,616 | | | | |
| `70b` | 68,488,011,776 | | | | |
| `100b` | 93,843,515,392 | | | | |
| `200b` | 209,420,042,240 | | | | |
| `500b` | 490,777,300,992 | | | | |

A configuration existing does **not** mean it can be trained on available
hardware. Only presets ≤ ~100M fit this machine for a full
forward/backward/optimizer cycle with the 262k vocabulary. See
[Scale readiness](#scale-readiness-honest).

## Scale readiness (honest)

| size | config validated | count measured | init tested | fwd tested | bwd tested | opt step | single-GPU | multi-GPU | distributed | production |
|---|---|---|---|---|---|---|---|---|---|---|
| dev preset (9.8M) | yes | yes | yes (CUDA) | yes | yes | yes | yes | no | no | no |
| `50m`–`100m` | yes | yes | yes (CPU/meta) | yes (75m chain) | yes (75m) | yes (75m) | yes (75m) | no | no | no |
| `500m`–`5.6b` | yes | yes | yes (meta) | no | no | no | no | no | no | no |
| `13b`–`500b` | yes | yes | meta only | no | no | no | no | no | no | no |
| MoE small/75M-class | yes | yes | yes | yes | yes | yes | yes | no | no | no |
| MoE ≥1B | yes | yes | meta only | no | no | no | no | no | no | no |

- "Meta tested" = module tree builds on the meta device and counts are exact.
  No forward pass has ever run for presets above ~100M (they do not fit).
- Multi-GPU: only a single-process gloo DDP smoke test exists. **No
  multi-GPU or multi-node training has been performed or validated anywhere
  in this project.** DDP code paths exist; they are not load-tested, and
  MoE+DDP global router statistics are per-rank.

## Benchmarks

Reproducible, seed-42 benchmark harness in `benchmarks/`; timestamped JSON in
`benchmarks/results/`; methodology in `docs/benchmark_methodology.md`.

Measured on the development machine (RTX 4060 Laptop 8 GB, torch 2.15.0+cu132,
Python 3.14.4; this is a 45 W-capped mobile GPU — inter-run spread of ±15% is
normal and all numbers should be read as such):

| measurement | baseline | this pass | notes |
|---|---|---|---|
| Dense prefill fp32 T=128 | 2.092 ms | **1.854 ms** (-11%) | fused RMSNorm |
| Dense prefill fp32 T=512 | 4.121 ms | **3.926 ms** (-5%) | T=1024: within noise |
| Dense prefill bf16 T=512 | 3.032 ms | **2.725 ms** (-10%) | T=1024: 5.43→5.14 ms (-5%) |
| MoE module fwd (B2, T=128) | 1.927 ms | **1.611 ms** (-16%) | gather/scatter-once |
| MoE module fwd (B2, T=512) | 3.447 ms | **3.353 ms** (-3%) | within noise band |
| MoE module fwd (B4, T=128) | 1.973 ms | **1.946 ms** (-1%) | wash |
| MoE module fwd (B4, T=512) | 5.473 ms | 5.29–5.86 ms (run-dependent) | within noise |
| `import neuratal` | 13.4 s | **1.3–1.4 s** | meta-init skip + lazy presets |
| Chunked CE fwd+bwd (1024 tok) | 88.0 ms | 88.0 ms | unchanged; allocates 1.29 GB vs 4.29 GB standard — the point of the path |

Dense baseline and this pass use identical settings (`benchmark_model.py
--device cuda --dtype float32|bfloat16 --warmup 10 --repeats 25`).

Benchmark suites: `benchmark_environment`, `benchmark_tokenizer`,
`benchmark_model`, `benchmark_kv_cache`, `benchmark_attention`,
`benchmark_end_to_end`, `benchmark_memory`, `benchmark_moe` (new),
`benchmark_chunked_ce` (new).

## Testing

```bash
python -m pytest tests/ -q          # full suite: 413 passed, 3 skipped, 1 warning
python -m pytest tests/ -k cuda     # CUDA-only subset
```

- Skipped: 1 CPU-only device-mismatch probe (needs CUDA device-mismatch),
  1 artifact-dependent integration test (covered in the integration suite),
  1 integration CUDA probe.
- Warning: a Triton FP-fusion notice from a CUDA test (third-party, benign).
- New in this pass: `tests/test_engine_regressions.py` — full-width tied
  head training step (regression: engine used to crash with
  `embedding_size=None`), tokenizer-fingerprint accept/reject on resume,
  GradScaler save/restore, optimizer-state index guard.

## Known limitations

- **V1 is untrained.** No quality claims are made. Dense/MoE training is
  synthetic-data plumbing validation only.
- `enable_gqa=True` is verified on this build (torch 2.15.0+cu132, sm_89);
  SDPA falls back to explicit KV expansion with a one-time warning otherwise.
- MoE is a readable reference implementation: a Python loop over experts, no
  kernel fusion, no capacity-aware CUDA kernel, no all-to-all; dispatch wins
  are constant-factor (see the benchmark table) not asymptotic.
- Local sharded checkpoints are not distributed checkpoints.
- `generate()` is a minimal greedy/sampling helper, not a serving stack.
- `attention_implementation="reference"` is a correctness path, never a
  performance path. `gated_delta` attention is experimental and slower; see
  `docs/rnd_report.md`.
- SDPA backend selection: the default picks FLASH for bf16 on this build;
  forcing cuDNN showed no reproducible win (spread dominated by GPU clocks,
  ±15%). No backend is forced.
- Chunked cross-entropy is ~35% slower than standard CE but allocates ~3.3×
  less memory; it is the enabler for 262k-vocab training, not a speed win.
- `torch.compile` measured on the MoE module: within noise (14.90 vs
  15.07 ms), rejected.
- The 8 GB development GPU cannot run presets above ~100M end-to-end; larger
  presets are validated only as configurations/counts, not as training runs.

## Project boundaries (V1)

Implemented: architecture, inference, training engine (single GPU, DDP smoke),
chunked cross-entropy, dense presets 50M–500B (measured counts), opt-in MoE
with presets, local sharded checkpoints, closed-form memory estimator,
token-level synthetic pipelines. Not implemented: real-corpus pretraining,
production distributed training (ZeRO/FSDP are estimator-only),
expert/tensor/pipeline parallelism, custom fused kernels.

## Documentation map

- `docs/architecture_audit.md` — architecture rationale and correctness rules
- `docs/moe_integration.md` — MoE design, measurements, scale readiness
- `docs/final_report.md` — Phase 5 engineering report
- `docs/ntokenizer_compatibility.md` — tokenizer boundary + fingerprint
- `docs/benchmark_methodology.md`, `docs/benchmark_results.md` — benchmarking
- `docs/optimization_log.md` — what was tried and what stuck
- `docs/release_readiness.md` — this pass: final audit, measurements, verdict
- `docs/training_plan.md` — how real training would be run (not run here)

## License

Dual-licensed `MIT OR Apache-2.0` (as declared in `pyproject.toml`); both
texts are in this repository (`LICENSE-MIT`, `LICENSE-APACHE-2.0`). NTokenizer
is a separate project with its own identity and license; dataset licensing is
deliberately out of scope and remains the responsibility of any real training
run (see `docs/training_plan.md`).
