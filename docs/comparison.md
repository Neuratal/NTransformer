# NTransformer vs Other Frameworks

This page is an honest side-by-side. Read it alongside the comparison itself:
these projects solve different problems, so **none of them is a drop-in peer for
any of the others**. Use this to choose the right tool for *your* task.

## Comparison at a glance

| Feature | NTransformer | Hugging Face Transformers | NVIDIA Transformer Engine | Megatron-LM | nanoGPT |
|---|---|---|---|---|---|
| **Primary purpose** | Research/engineering core for Neuratal Transformer training + inference | Model/serving library + reference implementations | FP8/FP4 kernel acceleration layer for NVIDIA GPUs | Distributed training at scale | Single-file educational GPT training loop |
| **Decoder-only support** | ✅ | ✅ (many families) | ➖ (bring-your-own) | ✅ | ✅ |
| **Custom tokenizer integration** | ✅ (Neuratal NTokenizer v0.3-polished only) | ✅ (any HF tokenizer) | ➖ (your responsibility) | ✅ (via HF integration / custom) | ✅ (needs your code) |
| **GQA** | ✅ | ✅ (Llama-3/Qwen-2 family) | ✅ | ✅ | ❌ (unless you add it) |
| **RoPE** | ✅ | ✅ | ✅ | ✅ | ❌ (unless you add it) |
| **SwiGLU** | ✅ | ✅ | ✅ | ✅ | ❌ (unless you add it) |
| **RMSNorm** | ✅ | ✅ | ✅ | ✅ | ✅ (via `nn.LayerNorm` swapped for RMSNorm — you do it) |
| **KV cache** | ✅ | ✅ | ➖ (application provides) | ✅ | ✅ |
| **Gradient checkpointing** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Mixed precision (bf16/fp16)** | ✅ (bf16 autocast / fp16 + GradScaler) | ✅ | ✅ (FP8/FP4 emphasis) | ✅ | ✅ |
| **Chunked loss** | ✅ (chunked linear CE) | ➖ (manual / selective API) | ✅ (fused) | ✅ | ❌ |
| **Long-context support** | ✅ (16K/32K, experimental) | ✅ (FlashAttention dependent) | ➖ (depends on recipe) | ✅ | ✅ (block-size limited) |
| **Distributed training** | ⚠️ CPU gloo verified; multi-GPU/NCCL **not verified** | ✅ (Trainer + DDP/DeepSpeed/etc.) | ✅ (TP/EP/CP focus) | ✅ (TP/PP/DP/EP/CP) | ✅ (DDP only) |
| **Safetensors** | ✅ | ✅ | ➖ (checkpoint format varies) | ✅ (Megatron Bridge) | ❌ (pickles/manual) |
| **License** | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT |
| **Pretrained weights** | ❌ none | ✅ extensive | ➖ (none provided; NN training toolkit) | ➖ (none provided; GPT-3-era demos) | ➖ (none; GPT-2-format but you supply weights) |

Legend: ✅ = present, ➖ = varies / hands-on / out of scope, ❌ = absent.

## NTransformer vs Hugging Face Transformers

- **NTransformer** is a focused, single-architecture (decoder-only GQA/MoE)
  framework with a handful of choices made deliberately (factorized tied embeddings,
  262k vocab, NTokenizer-only). **Hugging Face Transformers** is the reference
  library: thousands of model implementations, serving ecosystem, fine-tuning.
- NTransformer is **smaller, more readable, less configurable** — you read the
  code, not a 14-argument factory. If you want a pretrained Llama/Whisper/Whisper-V2,
  HF is the right answer.
- NTransformer's value is in **experimenting with a single, well-documented
  architecture end-to-end**, not in deploying someone else's weights.

## NTransformer vs nanoGPT

- **nanoGPT** is an educational GPT-2 re-implementation, a teaching tool.
- NTransformer is **not** a teaching tool; it is an engineering core: deterministic
  resume, fingerprint-verified checkpoints, long-context paths, 262k-vocab support.
  If you are learning Transformers, nanoGPT. If you are building on a real research
  core, NTransformer.
- NTransformer ships GQA, RoPE, SwiGLU, RMSNorm, factorized tied embeddings, and
  MoE — nanoGPT ships none of those (they were intentional simplifications).

## NTransformer vs Megatron-LM

- **Megatron-LM** is the workhorse of frontier labs: TP/PP/DP/EP/CP, kernel fusion,
  FP8/FP4 emphasis, multi-trillion-parameter training at real scale.
- NTransformer runs **no multi-GPU training** today; MoE is read-optimized, not a
  throughput-optimizing path; 16K/32K is experimental. If your goal is to train
  7B+ on a cluster, Megatron-LM. If you want to understand or modify the architecture,
  NTransformer.

## NTransformer vs NVIDIA Transformer Engine

- **Transformer Engine** is a **kernel/fp8 toolbox**. It accelerates PyTorch/JAX
  models on NVIDIA GPUs; it is not a model framework, not a tokenizer, not a
  training loop.
- NTransformer **could integrate TE's FP8 path** in principle (we explored TE's
  model implementations and rejected it for now — private-API surface and unproven
  on this hardware). If you need FP8 throughput, use Transformer Engine *to build*
  the model; NTransformer remains the reference GQA/SwiGLU core.

## Recommendation

- **Research/education, architecture tinkering, embedded experiments:** NTransformer
- **Deploy pretrained models:** Hugging Face Transformers
- **Maximum throughput on NVIDIA:** Megatron-LM (+ Transformer Engine)
- **Learn how Transformers work, fastest path:** nanoGPT
