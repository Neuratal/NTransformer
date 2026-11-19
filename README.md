# NTransformer (Neuratal Transformer)

**A PyTorch implementation of the standard decoder-only LLaMA-style Transformer (GQA, RoPE, SwiGLU, RMSNorm), plus a training engine, written by Neuratal.** Designed for experimentation with decoder-only language models, deterministic training, and full NTokenizer v0.3-polished integration.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-460%20passed%20%C2%B7%203%20skipped-brightgreen.svg)](tests/)

---

## What it is — and what it isn't

**NTransformer is a research/engineering framework, not a product.**

- It is not a pretrained model — no weights are distributed with this release.
- It is not production-ready serving — `generate()` is a minimal greedy/sampling helper, not a serving stack.
- It is not a replacement for dense, tensor-parallel frameworks (Megatron-LM), GPUs-optimized kernels (Transformer Engine), or general frameworks (Hugging Face Transformers). See [Comparison](docs/comparison.md) for an honest side-by-side.

**What it is:**

- A clean, readable, correct implementation of the Transformer (dense GQA + optional MoE), directly verifiable, with everything you need to study/modify/debug.
- A training engine with deterministic resumption (model + optimizer + scheduler + RNG + counters + GradScaler), gradient checkpointing, and chunked linear cross-entropy that never materializes the 262,144-vocab logits.
- A library of architecture traits (factorized tied embeddings, RoPE, RMSNorm, GQA, SwiGLU) ready for experiment.
- **NTokenizer-native by construction**: the entire pipeline is designed around Neuratal NTokenizer v0.3-polished; no re-tokenization, no ID remapping, no silent BOS/EOS injection, fingerprint-verified checkpoints.

---

## Architecture overview

V1 is a decoder-only, pre-norm Transformer with the following decisions:

| Component | Choice |
|---|---|
| **Topology** | Decoder-only, pre-norm residual |
| **Positions** | RoPE (rotate-half, NeoX style), offset-aware for KV-cache decode |
| **Attention** | Grouped-query attention (SDPA), explicit causal regimes; 16k/32k context paths with memory-bounded attention |
| **MLP** | SwiGLU (bias-free); opt-in Mixture-of-Experts |
| **Embeddings** | Factorized tied: one shared `V×F` matrix feeds both input & output; `F` ≈ 1/6 of the hidden width by default |
| **Normalization** | RMSNorm (fp32 stats; fused `F.rms_norm` fast path) |
| **Serialization** | safetensors + `config.json`, atomic writes, strict validation |
| **Precision** | fp32 reference; bf16/fp16 via autocast (+GradScaler for fp16) |
| **Optimizer** | AdamW (decoupled weight decay on ≥2D tensors); warmup → cosine decay |
| **Distributed** | Two-process CPU `gloo` DDP verified (dense and MoE, with exact parity); multi-GPU NCCL **not** verified on dev hardware |
| **Loss** | Chunked linear cross-entropy (recompute-in-backward); never materializes `[B, T, 262144]` logits |
| **Vocab** | 262,144 token IDs (bound to NTokenizer v0.3-polished) — **IDs are consumed verbatim** |

### Scale-ready candidates

Presets (`get_preset("50m")` … `get_preset("500b")`) are **measured from the real module tree**, not marketed. The dense presets go from a ~48M development preset to 500B. Configs are validated/ counted; **none are trained**.

---

## Key features

- **Honest development**: every benchmark and claim is labeled as measured or estimated; rejected experiments are archived.
- **NTokenizer integration**: full fingerprint verification, resume-safety, correct boundary on every layer of the stack.
- **Memory-bounded training**: 16K/32K context samples fit an 8 GB development GPU via:
  - gradient checkpointing
  - chunked cross-entropy
  - last-position-only forward
  - KV-cache decode
- **A complete checkpointing story**: single-file safetensors, atomic writes, strict validation, shards for large models (`save_pretrained_sharded`).
- **Mixture of Experts (opt-in)**: readable reference implementation, tested dispatch paths (counted/ sorted/ scattered/empties).
- **No hidden corpus**: nothing is downloaded, trained, or embedded.
- **NTokenizer compatibility**: it's the one tokenizer that's pre-integrated; everyone else must port this layer.

---

## Why NTransformer?

If you are a researcher, practitioner, or curious developer, Neuratal offers four things that existing frameworks do not combine:

1. **Legibility with correctness.** You can read every line and every benchmark. If a claim isn't measured, we say so. There are no "marketing benchmarks."
2. **Full NTokenizer compatibility.** NTokenizer v0.3-polished (vocab 262,144) is the canonical Neuratal tokenizer; this package is its direct consumer — not a bolt-on integration, not a re-tokenization, not a silent boundary violation. With **full fingerprint verification**, `neuratal.integration` binds a checkpoint to the exact `.ntok` artifact: wrong hashes are rejected at load, at resume, and at inference.
3. **Single-command edge deployment.** A development preset that runs a full forward pass, generation, and a 10-step training micro-cycle on an RTX 4060 Laptop in minutes with a 262,144-token vocabulary — with nothing to download, nothing to configure, no keys.
4. **Honest scope.** If it doesn't work, we say it. This is not a sales page.

---

## Installation

Requires Python ≥ 3.10; verified with Python 3.14.4, torch 2.15.0+cu132, CUDA 13.2.

```bash
pip install -e .             # runtime: torch>=2.4, safetensors>=0.4
pip install -e .[dev]        # + pytest, numpy
```

---

## Quick start

```python
import torch
from neuratal import NeuratalModel, NeuratalConfig, generate

model = NeuratalModel(NeuratalConfig()).eval()
ids = torch.tensor([[168069, 5431, 33]])       # token IDs are consumed verbatim
out = generate(model, ids, max_new_tokens=16)
print(out.shape)  # (1, 19)
```

The development preset is **untrained** — outputs are gibberish. The example validates the pipeline, not quality.

---

## Examples

### Generation with KV cache

```python
import torch
from neuratal import NeuratalModel, NeuratalConfig, generate

model = NeuratalModel(NeuratalConfig()).eval()
ids = torch.tensor([[168069, 5431, 33]])
out = generate(model, ids, max_new_tokens=16)
```

### Forward pass + loss

```bash
python examples/inference.py --prompt "Merhaba, dünya!"
python examples/neuratal_inference.py --prompt "Merhaba, dünya!" --max-new-tokens 16
```

The repository contains complete, runnable examples; see `examples/`. The loss
pipeline is intentionally low-level (chunked cross-entropy against a frozen
factorized tied head) and is wired into the training engine — see
`docs/training.md` for the API and `docs/architecture.md` for the design.

### Save / load checkpoint

```python
from neuratal import NeuratalModel, NeuratalConfig, save_pretrained, load_pretrained

model = NeuratalModel(NeuratalConfig())
save_pretrained(model, "checkpoints/demo")
model2 = load_pretrained("checkpoints/demo")
```

### Training (synthetic data only)

```python
import torch
from neuratal import NeuratalModel, get_preset
from neuratal.training.data import SyntheticTokenDataset
from neuratal.training.engine import TrainConfig, train

print("building...")
model = NeuratalModel(get_preset("75m").config).to("cuda", torch.bfloat16)
ds = SyntheticTokenDataset(16, 32, vocab_size=262_144, seed=1)
cfg = TrainConfig(precision="bfloat16", max_steps=10, warmup_steps=2,
                  out_dir="checkpoints/demo", device="cuda")
model, history = train(model, ds, None, cfg)
```

**Synthetic data only. No real corpus is downloaded, used, or bundled anywhere in this repository.**

### End-to-end with NTokenizer

```bash
python examples/neuratal_inference.py --prompt "Merhaba, dünya!" --max-new-tokens 16
```

(Requires the NTokenizer artifact at `NTOKENIZER_ROOT`; output is gibberish — this validates plumbing only.)

---

## Configuration

`NeuratalConfig` validates everything on construction — nothing is silently corrected.
Highlights:

- `embedding_size=None` → full-width (bypass factorization)
- `moe_layer_indices` / `moe_layer_frequency` → MoE placement (mix dense and MoE layers)
- `capacity_factor=None` → no token dropping
- `tie_word_embeddings=False` is an explicit error — tying is mandatory in V1
- resolved defaults are serialized into `config.json`
- `gradient_checkpointing=True` — required for 16k/32k training
- `last_position_only=True` — memory-bounded forward for inference

Run `python -m pytest tests/ -k config -q` to see the validation rules.

---

## Context length support

| Context | Training | Inference (KV-cache) | Memory fit |
|---|---|---|---|
| 2K   | ✅ | ✅ | fits easily |
| 8K   | ✅ | ✅ | fits with headroom |
| 16K  | ✅ (gradient checkpointing required) | ✅ | 8 GB dev GPU OK |
| 32K  | ✅ (gradient checkpointing required) | ✅ | 8 GB dev GPU OK |

The 16K/32K paths are part of v1.4.0. To run 32K training, set `gradient_checkpointing=True` in your `NeuratalConfig`/`TrainConfig`. Do not expect normal throughput at these lengths on this hardware — it is for correctness, not throughput.

---

## Benchmarks

**Reproducible, seed-42, with `benchmarks/` harness.**

| Measurement | `baseline` | Latest | Notes |
|---|---|---|---|
| Dense prefill fp32 T=512 | 4.121 ms | **3.926 ms** (-5%) | fused RMSNorm |
| Dense prefill bf16 T=512 | 3.032 ms | **2.725 ms** (-10%) | 5.43→5.14 ms @1024 |
| MoE fwd (B2, T=128) | 1.927 ms | **1.611 ms** (-16%) | gather/scatter-once |
| Chunked CE fwd+bwd (1024 tok) | 88.0 ms | 88.0 ms | allocates 1.29 GB vs 4.29 GB standard |

Measured on an RTX 4060 Laptop (45 W capped) with torch 2.15.0+cu132, Python 3.14.4. Inter-run spread of ±15% is normal; read as indicative, not authoritative.

Benchmark suites live in `benchmarks/`; JSON results in `benchmarks/results/`; methodology in `docs/benchmark_methodology.md`.

**40M pilot** (2026-11-17, synthetic validation only — never use as a quality reference):
- 39,997,440 tokens; 10,431 tokens/sec; ~63m55s
- Validation loss 10.84 → 6.24
- Outputs are **not** usable assistant-grade text
- Treat as an infrastructure check, not a release milestone

---

## Testing

```bash
python -m pytest tests/ -q          # 460 passed, 3 skipped
```

All tests run CPU-only (CUDA-specific ones skip). Tests cover every model module, the training engine, the checkpoint lifecycle, MoE dispatch edge cases, NTokenizer fingerprint checks, and resume determinism.

---

## Documentation

- [Architecture](docs/architecture.md) — what's in the model and why
- [Training](docs/training.md) — data/optimizer/checkpointing
- [Inference](docs/inference.md) — generate, KV cache, last-position-only
- [Tokenizer Integration](docs/tokenizer-integration.md) — **full NTokenizer compatibility** (vocab 262,144; fingerprint-verified)
  - GitHub: https://github.com/Neuratal/neuratal-tokenizer
  - Hugging Face: https://huggingface.co/Neuratal/ntokenizer
  - Release: https://github.com/Neuratal/neuratal-tokenizer/releases/tag/v0.3.0
- [Context Length](docs/context-length.md) — 16K/32K support
- [Comparison](docs/comparison.md) — honest side-by-side with HF/TE/Megatron-LM/nanoGPT
- [Limitations](docs/limitations.md) — what isn't done, won't work
- [Benchmarking](docs/benchmarking.md) — methodology and results
- [Architecture Audit](docs/architecture_audit.md) — correctness rules
- [MoE Integration](docs/moe_integration.md) — opt-in experts design
- [NTokenizer Compatibility](docs/ntokenizer_compatibility.md) — fingerprint details
- [Changelog](CHANGELOG.md)

---

## Positioning (not a marketing pitch)

NTransformer bundles a **modern LLaMA-style architecture** (GQA + KV cache + RoPE + RMSNorm + SwiGLU + factorized tied embeddings) with a training engine and an optional MoE reference implementation. Nothing here is architecturally novel.

**When this is a good fit:**
- You want a *small, readable reference implementation* of a modern decoder-only stack — for learning, auditing, or extending.
- You specifically need **full NTokenizer v0.3-polished compatibility** (vocab 262,144; IDs consumed verbatim; SHA-256 fingerprint-verified checkpoints).
- You are content with a **research/engineering core** — not a production serving stack or a flagship model.

**When it is a poor fit:**
- You need a **pretrained model** — there is none here; the development preset outputs gibberish.
- You need **multi-GPU/NCCL throughput** — not verified; CPU-gloo DDP only.
- You need to plug in an arbitrary tokenizer — NTransformer targets Neuratal's tokenizer (vocab 262,144) only.
- You need ecosystem features (Hugging Face Trainer, DeepSpeed, FSDP templates, deployment integrations) — use Hugging Face.
- You want a *novel architecture* — this is not one; it is a solid, honest implementation of the standard recipe.

For the design trade-offs against HF/Megatron-LM/nanoGPT, see [Comparison](docs/comparison.md).

---

## Roadmap

- v1.4.0 — complete NTokenizer integration, long context
- v1.5.0 — real pretraining on a modest corpus (if corpus licensing allows)
- v2.0.0 — distributed training on multi-GPU

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). NTokenizer is a separate project with its own identity and license; do not conflate the two.

---
