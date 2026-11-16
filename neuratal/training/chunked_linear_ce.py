"""Chunked cross-entropy with recompute-in-backward.

Why not ``torch.utils.checkpoint``? Per-chunk checkpointing keeps each
chunk's local autograd graph alive for the backward pass (128 chunks × 256
positions × 262k vocab ≈ 4–8 GiB), so peak memory grows with sequence length.
This autograd Function instead:

* forward: computes per-chunk CE under no_grad; memory = one chunk of logits;
* backward: re-computes logits chunk-by-chunk and accumulates grads; memory
  stays O(chunk), independent of B*T.

Mathematically identical to full-batch cross_entropy except reduction order
(documented in tests; fp32 accumulate).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_IGNORE_INDEX = -100


class ChunkedLinearCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, targets, out_proj_weight, emb_weight, chunk_size, ignore_index):
        chunk_size = int(chunk_size) if chunk_size else 256
        flat_h = hidden.reshape(-1, hidden.size(-1)).contiguous()
        flat_t = targets.reshape(-1).to(torch.long).contiguous()
        n = (flat_t != ignore_index).sum().detach()
        total = flat_h.new_zeros((), dtype=torch.float32)
        # Hoist fp32 weight conversion out of the chunk loop. For bf16/fp16
        # models this avoids re-allocating the full [V, F] matrix once per chunk.
        op32 = None if out_proj_weight is None else out_proj_weight.float()
        emb32 = emb_weight.float()
        with torch.no_grad():
            for start in range(0, flat_h.size(0), chunk_size):
                h, t = flat_h[start:start + chunk_size], flat_t[start:start + chunk_size]
                x = h.float() if op32 is None else F.linear(h.float(), op32)
                logits = F.linear(x, emb32)
                total = total + F.cross_entropy(logits, t, ignore_index=ignore_index, reduction="sum")
        ctx.save_for_backward(flat_h, flat_t, out_proj_weight, emb_weight)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = int(ignore_index)
        ctx.n_tokens = n
        ctx.hidden_shape = hidden.shape
        return total

    @staticmethod
    def backward(ctx, grad_output):
        flat_h, flat_t, out_proj_weight, emb_weight = ctx.saved_tensors
        n = ctx.n_tokens
        chunk_size = ctx.chunk_size
        ignore_index = ctx.ignore_index
        go = grad_output.to(torch.float32)

        grad_h = torch.zeros_like(flat_h, dtype=torch.float32)
        grad_op = None if out_proj_weight is None else torch.zeros_like(out_proj_weight, dtype=torch.float32)
        grad_emb = torch.zeros_like(emb_weight, dtype=torch.float32)
        op32 = None if out_proj_weight is None else out_proj_weight.float()
        emb32 = emb_weight.float()

        for start in range(0, flat_h.size(0), chunk_size):
            h, t = flat_h[start:start + chunk_size], flat_t[start:start + chunk_size]
            h32 = h.float()
            x = h32 if op32 is None else F.linear(h32, op32)
            logits = F.linear(x, emb32)

            # softmax cross-entropy grad: (softmax - onehot) * grad_output
            # (the caller passes grad_output=1/n_tokens for mean reduction)
            grad_logits = torch.softmax(logits, dim=-1)
            valid = t != ignore_index
            rows = torch.arange(t.numel(), device=t.device)[valid]
            grad_logits[rows, t[valid]] -= 1.0
            grad_logits = grad_logits * go * valid.unsqueeze(1).float()

            # grads to x, emb, op, h
            grad_x = grad_logits @ emb32            # [c, d_x]
            grad_emb.add_(grad_logits.t() @ x)      # [V, d_x]
            if op32 is not None:
                grad_op.add_(grad_x.t() @ h32)      # [d_x, H]
                gh = grad_x @ op32
            else:
                gh = grad_x
            grad_h[start:start + chunk_size] = gh

        return (
            grad_h.to(flat_h.dtype).reshape(ctx.hidden_shape),
            None,
            (None if out_proj_weight is None else grad_op.to(out_proj_weight.dtype)),
            grad_emb.to(emb_weight.dtype),
            None,
            None,
        )


def chunked_linear_cross_entropy(hidden, targets, out_proj_weight, emb_weight,
                                 chunk_size: int = 256, ignore_index: int = _IGNORE_INDEX):
    """Returns (sum_loss, n_tokens). Divide by n_tokens externally for mean."""
    return ChunkedLinearCEFunction.apply(hidden, targets, out_proj_weight, emb_weight, int(chunk_size), ignore_index), (targets.reshape(-1) != ignore_index).sum()
