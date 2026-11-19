# Context Length Support in NTransformer

Status: **v1.4.0 — supports 4K, 8K, 16K and 32K training with memory-bounded paths; 32K is experimental.**

## How context length is handled

| Path | Context | Feasibility on development hardware |
|---|---|---|
| Short (standard, default) | 2K | fits with headroom |
| Standard | 8K | fits with headroom |
| Long (v1.4+) | 16K | fits but requires `gradient_checkpointing=True` |
| Very long (v1.4+) | 32K | feasible, experimental; `gradient_checkpointing=True` required; slow |

## Why gradient checkpointing is required for 16K/32K

At 16K tokens with a 262k vocabulary, the logits difference per token is
262,144 elements. Computing cross-entropy over the full sequence at once is an
explicit out-of-memory error. The chunked linear cross-entropy used by the
training engine is recompute-in-backward: peak memory is `O(chunk)`, not
`O(vocab × seq_len)`.

**Activation memory** at 16K/32K is also large; gradient checkpointing
discards and recomputes activations, trading compute for memory. Without
checkpointing, 16K does not fit an 8 GB card.

## Settings

```python
from neuratal import NeuratalModel, NeuratalConfig
import torch

model = NeuratalModel(NeuratalConfig(
    gradient_checkpointing=True,
    max_position_embeddings=32768,
))
```

`last_position_only=True` (default in v1.4+) also keeps memory bounded by
computing the loss over the last token position per chunk; pairing it with
gradient checkpointing is the intended configuration.

## What is NOT claimed

- **16K/32K is not pretrained** on this hardware. No weights of any large
  model exist here.
- **32K speed** is unmeasured on the development GPU; do not expect it to be
  fast.
- **32K inference** fits in memory with headroom for prefill; decode still
  needs KV-cache sizing.
- This is not a solved problem on this hardware; the point is that the paths
  exist and are tested.
