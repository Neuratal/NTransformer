# Inference in NTransformer

Status: **matches `neuratal` 1.4.0 (2026-11-19).**

## Quick example

```python
import torch
from neuratal import NeuratalModel, NeuratalConfig, generate

model = NeuratalModel(NeuratalConfig()).eval()
ids = torch.tensor([[168069, 5431, 33]])
out = generate(model, ids, max_new_tokens=16)
print(out.shape)
```

The development preset is **untrained** — outputs are gibberish. The example
validates plumbing, not quality.

## `generate()` signature

```python
generate(
    model, input_ids,
    *, max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    experimental_cuda_graphs: bool = False,
) -> torch.Tensor
```

Generation is greedy by default; `do_sample=True` enables sampling.

**No CUDA graphs by default** — `experimental_cuda_graphs` is opt-in only,
batch=1, CUDA, dropout=0; window size = next power of two ≥ prompt+max_new+1).

## KV cache

KV cache is managed internally by `generate()`. The `KVCache` class stores
preallocated slots (`num_layers × num_key_value_heads × head_dim`); overflow
raises `CacheOverflowError`. For long-context `last_position_only=True` forward
passes can be used to bound memory.

## Last-position-only forward

`model(input_ids, output_hidden_only=True, return_aux=True)` returns
`(hidden, moe_stats)` with the last-position-only property; hidden shape is
`[B, 1, F]`. This is the memory-bounded path used during long-context
training and inference.

## Safetensors (why, how)

Every checkpoint saved by this package uses `safetensors` for weights and
`config.json` for architecture. `pickle` is not used for weights.

```python
from neuratal import save_pretrained, load_pretrained

save_pretrained(model, "checkpoints/demo")
model2 = load_pretrained("checkpoints/demo")
```

Strict validation on load: architecture name, format version, shape, dtype,
missing/unexpected keys, per-tensor shapes, zero-tolerance corruption tests,
all covered by `tests/test_serialization.py`.

## Sharded checkpoints

For large models: `save_pretrained_sharded(model, "out_dir", max_shard_bytes=...)`
and `load_pretrained_sharded(...)`. This is a **local** format — not distributed
checkpointing, no resharding.

## EF examples

See `examples/`:

- `examples/inference.py` — end-to-end with NTokenizer (if available)
- `examples/neuratal_inference.py` — with `NTOKENIZER_ROOT` set
- `examples/benchmark.py` — latency sampling
