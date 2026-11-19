# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
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

``experimental_cuda_graphs=False`` (default): opt-in static-shape CUDA-graph
decode (CUDA, batch=1 only). Bitwise-identical logits to the eager path on
the tested build; measured faster for generations of any length here (see
docs/rnd_report.md). Never used unless explicitly requested.
"""

from __future__ import annotations

from typing import Optional

import torch

from neuratal.config import NeuratalConfig
from neuratal.model.cache import KVCache
from neuratal.model.model import NeuratalModel


def _select_next(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    generator: Optional[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Choose the next token ids for a [B, vocab] fp32 logit tensor."""
    if not do_sample:
        return logits.argmax(dim=-1)
    return _sample_next(logits, temperature, top_k, generator, device)


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
    experimental_cuda_graphs: bool = False,
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
        if experimental_cuda_graphs:
            return _generate_cuda_graph(
                model, input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature, top_k=top_k,
                eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                generator=generator, fill_id=fill_id,
            )

        cache = KVCache(cfg) if cfg.use_cache else None
        generated = input_ids
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for step in range(max_new_tokens):
            if cache is not None:
                step_ids = generated if step == 0 else generated[:, -1:]
                out = model(step_ids, past_key_values=cache, last_position_only=True)
            else:
                # No cache: project only the final position — sampling never needs more.
                out = model(generated, last_position_only=True)
            next_logits = out.logits[:, -1, :].float()  # fp32 for stable sampling/argmax

            next_tokens = _select_next(
                next_logits, do_sample=do_sample, temperature=temperature,
                top_k=top_k, generator=generator, device=input_ids.device,
            )

            if eos_token_id is not None:
                next_tokens = torch.where(
                    finished, torch.full_like(next_tokens, fill_id), next_tokens)
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
                finished = finished | (next_tokens == eos_token_id)
                if bool(finished.all()):
                    break
            else:
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        return generated
    finally:
        model.train(was_training)


def _generate_cuda_graph(
    model: NeuratalModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    generator: Optional[torch.Generator],
    fill_id: int,
) -> torch.Tensor:
    from neuratal.model.graph_decode import GraphDecodeSession

    if input_ids.size(0) != 1:
        raise ValueError(f"experimental_cuda_graphs requires batch size 1, got {input_ids.size(0)}")
    if input_ids.device.type != "cuda":
        raise ValueError("experimental_cuda_graphs requires a CUDA device")

    session = GraphDecodeSession(model, input_ids, max_new_tokens)
    try:
        generated = input_ids
        finished = False
        logits = session.prefill_logits  # [1, vocab] fp32

        for step in range(max_new_tokens):
            next_tokens = _select_next(
                logits, do_sample=do_sample, temperature=temperature,
                top_k=top_k, generator=generator, device=input_ids.device,
            )
            token = int(next_tokens.item())
            if eos_token_id is not None and finished:
                token = fill_id
            generated = torch.cat(
                [generated, torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)],
                dim=1,
            )
            if eos_token_id is not None and token == eos_token_id:
                finished = True
                break
            if step + 1 < max_new_tokens:
                logits = session.step(token)
        return generated
    finally:
        session.close()


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
