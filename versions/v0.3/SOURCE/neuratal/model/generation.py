"""Basic autoregressive generation.

Scope (deliberately small): greedy decoding or temperature/top-k sampling,
with optional EOS early stop. No beam search, no serving, no batching logic
beyond what the model already provides. Decoding/tokenization are the
NTokenizer's job; this module operates purely on token IDs.

The model's ``config.use_cache`` is honored; a fresh :class:`KVCache` is used
for each call, so concurrent calls never share state. Pass an explicit
``eos_token_id`` (falling back to ``config.eos_token_id``) to stop early;
sequences that reach EOS are padded with ``pad_token_id``/eos for the rest of
the horizon. No special token is ever injected into the input.
"""

from __future__ import annotations

from typing import Optional

import torch

from neuratal.config import NeuratalConfig
from neuratal.model.cache import KVCache
from neuratal.model.model import NeuratalModel


@torch.no_grad()
def generate(
    model: NeuratalModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be an integer >= 1, got {max_new_tokens}")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError(f"top_k must be an integer >= 1 when set, got {top_k}")
    if input_ids.dim() != 2 or input_ids.size(1) < 1:
        raise ValueError(f"input_ids must be 2-D [batch, seq] with seq >= 1, got shape {tuple(input_ids.shape)}")

    cfg: NeuratalConfig = model.config
    if eos_token_id is None:
        eos_token_id = cfg.eos_token_id
    if pad_token_id is None:
        pad_token_id = cfg.pad_token_id
    fill_id = pad_token_id if pad_token_id is not None else (eos_token_id if eos_token_id is not None else 0)

    was_training = model.training
    model.eval()
    try:
        cache = KVCache(cfg) if cfg.use_cache else None
        generated = input_ids
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for step in range(max_new_tokens):
            if cache is not None:
                step_ids = generated if step == 0 else generated[:, -1:]
                out = model(step_ids, past_key_values=cache)
            else:
                out = model(generated)
            next_logits = out.logits[:, -1, :].float()  # fp32 for stable sampling

            if do_sample:
                next_tokens = _sample_next(next_logits, temperature, top_k, generator, input_ids.device)
            else:
                next_tokens = next_logits.argmax(dim=-1)

            if eos_token_id is not None:
                next_tokens = torch.where(
                    finished, torch.full_like(next_tokens, fill_id), next_tokens
                )
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
                finished = finished | (next_tokens == eos_token_id)
                if bool(finished.all()):
                    break
            else:
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        return generated
    finally:
        model.train(was_training)


def _sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    generator: Optional[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    logits = logits / float(temperature)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if generator is not None and generator.device != device:
        raise ValueError(
            f"generator is on {generator.device} but input_ids are on {device}; "
            "pass a generator on the same device (e.g. torch.Generator(device).manual_seed(...))"
        )
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
