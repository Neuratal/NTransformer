"""Training engine — single-GPU and multi-GPU (DDP) training.

Design contract:
- Pure PyTorch, no Accelerate dependency.
- Single-GPU default; DDP via ``torchrun`` / ``init_process_group``.
- Gradient accumulation counted in *optimizer steps*; LR schedule steps on
  optimizer steps only (never micro-steps).
- Gradient clipping applied to the *accumulated* gradient before stepping.
- BF16 mixed precision via ``torch.autocast``; fp32 master weights handled by
  ``torch.optim.AdamW`` (it keeps fp32 state for bf16 params automatically).
- FP16 mixed precision uses ``torch.autocast`` + ``torch.amp.GradScaler``
  (loss scaling, unscale-before-clip, skipped step on non-finite grads); the
  scaler state is checkpointed and restored on resume.
- DDP: every parameter used by the loss is consumed **inside the wrapped
  forward** (see ``_DDPLossModule``), so the reducer never has to reason
  about parameters consumed outside it. ``find_unused_parameters=True`` is
  still set when the model is an MoE (a micro-batch may legitimately route
  zero tokens to some expert, and a parameter that never enters the autograd
  graph would otherwise wedge the reducer).
- Loss normalization: average per-token loss over supervised tokens.
- MoE router auxiliary/z losses are added to the main LM loss exactly once per
  micro-batch (they are already coefficient-weighted, scalar mean terms).
- Resume restores model + optimizer + scheduler + RNG + counters exactly.
- Non-finite gradient norms (inf/nan, from any precision path) cause the
  optimizer step — and the LR schedule — to be skipped loudly; the weights
  are never poisoned. Skipped steps do not advance ``global_step``.
- Training forward enters through ``model(..., output_hidden_only=True)`` on the
  single-process path (or the joint loss module in DDP) so DDP's reducer
  preparation is not bypassed.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from neuratal.model.model import NeuratalModel
from neuratal.training.chunked_linear_ce import chunked_linear_cross_entropy
from neuratal.training.optim import WarmupCosineSchedule, build_optimizer
from neuratal.training.checkpoint import save_training_checkpoint, resume_training_checkpoint


@dataclass
class TrainConfig:
    # data
    batch_size: int = 2
    sequence_length: int = 1024
    num_workers: int = 0
    val_fraction: float = 0.05
    # optimization
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    # precision
    precision: str = "bfloat16"  # "float32" | "bfloat16" | "float16"
    # validation / logging
    eval_every_steps: int = 200
    log_every_steps: int = 20
    checkpoint_every_steps: int = 500
    # wall-clock control for bounded runs (default: unlimited — historical behavior)
    deadline_unix: Optional[float] = None   # stop starting new optimizer steps after this time
    progress_log_seconds: float = 0.0       # print a progress line every N seconds (0 = off)
    save_final_checkpoint: bool = False     # save ckpt-final when the loop exits for ANY reason
    # misc
    seed: int = 1234
    max_steps: int = 10_000
    out_dir: str = "checkpoints/run"
    tokenizer_fingerprint: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def init_distributed() -> tuple[int, int]:
    if not dist.is_available():
        return 0, 1
    if not dist.is_initialized():
        if "RANK" in os.environ:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        else:
            return 0, 1
    return dist.get_rank(), dist.get_world_size()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_of(model: torch.nn.Module) -> torch.device:
    core = model.module if hasattr(model, "module") else model
    return next(core.parameters()).device


def _head_out_proj_weight(core: torch.nn.Module):
    """Output-projection weight for the tied head, or None for full-width embeddings.

    ``OutputProjection.out_proj`` is None when embedding_size == hidden_size;
    the chunked cross-entropy handles that case natively.
    """
    out_proj = core.output_projection.out_proj
    return out_proj.weight if out_proj is not None else None


class _DDPLossModule(torch.nn.Module):
    """Internal DDP-facing joint forward: hidden-only encoding + chunked LM loss.

    DDP's reducer contracts on parameters in the wrapped forward's autograd
    graph. The v1.x engine applied the tied LM head *outside* the wrapped
    forward (inside :func:`chunked_linear_cross_entropy`), so
    ``output_projection.out_proj.weight`` never appeared in the forward graph.
    Combined with ``find_unused_parameters=True`` the reducer marked that
    parameter "ready (unused)" at the end of the forward, and its autograd
    hook firing during backward crashed the reducer with "marked as ready
    twice" (every factorized-tied-head MoE, on the first backward).

    This module performs hidden-only encoding *and* the chunked LM loss inside
    one wrapped forward, so every reducer-relevant parameter (including the
    factorized tied head's ``out_proj.weight``) is consumed inside the forward
    graph. The loss path is exactly the chunked cross-entropy — mathematically
    identical to the external version, at O(chunk) memory (no full logits).
    """

    def __init__(self, model: NeuratalModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, labels):  # -> (sum_loss, n_tokens, MoEStats|None)
        out = self.model(input_ids=input_ids, output_hidden_only=True, return_aux=True)
        hidden, moe_stats = out
        total_loss, n_tok = chunked_linear_cross_entropy(
            hidden, labels, _head_out_proj_weight(self.model),
            self.model.embed_tokens.embedding_weight,
        )
        return total_loss, n_tok, moe_stats


def _moe_log_fields(stats_sum, micro_count: int) -> dict:
    """Convert accumulated MoE metric tensors to JSON-friendly floats/lists."""
    if stats_sum is None or micro_count == 0:
        return {}
    counts = stats_sum["counts"]
    counts_list = [int(x) for x in counts.detach().cpu().tolist()]
    total = counts.sum().clamp_min(1.0)
    util = (counts / total).detach().cpu().tolist()
    active = int((counts > 0).sum().item())
    mean = counts.mean().clamp_min(1e-12)
    imbalance = float((counts.max() / mean).item())
    num_assignments = max(stats_sum["num_assignments"], 1)
    return {
        "active_experts": active,
        "tokens_per_expert": counts_list,
        "expert_utilization": [round(float(x), 6) for x in util],
        "routing_entropy": float((stats_sum["entropy"] / micro_count).item()),
        "expert_load_imbalance": round(imbalance, 6),
        "dropped_tokens": int(stats_sum["dropped"].item()),
        "dropped_fraction": round(float(stats_sum["dropped"].item()) / num_assignments, 6),
        "aux_loss": float((stats_sum["aux"] / micro_count).item()),
        "router_z_loss": float((stats_sum["z"] / micro_count).item()),
    }


def train(
    model: NeuratalModel,
    train_data,
    val_data,
    cfg: TrainConfig,
    *,
    max_steps: Optional[int] = None,
    resume_from: Optional[str] = None,
):
    """Run training. Returns ``(model, history)`` — no side effects on global state.

    The returned model is the caller's (unwrapped, ``NeuratalModel``) in every
    configuration, including under DDP. DDP wraps an internal joint-loss module
    (``_DDPLossModule``) for the lifetime of the call only.

    ``resume_from`` restores optimizer/scheduler/RNG/step counters from a
    training-checkpoint directory. The model weights are expected to be loaded
    by the caller (``load_training_checkpoint``), which keeps model ownership
    explicit. Data-order parity is exact for ``shuffle=False``; with shuffling,
    the stream is re-permuted from the restored RNG state (documented, not
    claimed as bit-identical data replay).

    On resume, the checkpoint's own LR schedule (including its ``total_steps``)
    always wins: if the requested ``max_steps``/``cfg.max_steps`` differ from the
    restored ``total_steps``, the checkpointed schedule is kept and a
    ``UserWarning`` is issued (extra steps run at the ``min_lr`` floor).
    """
    rank, world_size = init_distributed()
    is_main = rank == 0
    set_seed(cfg.seed + rank)
    model = model.to(cfg.device)
    ddp_model: Optional[torch.nn.Module] = None
    if world_size > 1:
        local_rank = rank % max(1, torch.cuda.device_count())
        device_ids = [local_rank] if cfg.device == "cuda" else None
        wrapper = _DDPLossModule(model).to(cfg.device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            wrapper, device_ids=device_ids,
            # An MoE micro-batch can route zero tokens to some expert; a
            # parameter that never enters the autograd graph would otherwise
            # wedge the reducer (the same holds for mixed dense/MoE stacks).
            find_unused_parameters=bool(model.config.is_moe),
        )
    train_module = ddp_model if ddp_model is not None else model

    optimizer = build_optimizer(train_module, name="adamw", learning_rate=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)
    requested_total = max_steps or cfg.max_steps
    if resume_from is not None:
        # The checkpoint's own schedule wins on resume — construct with
        # arguments guaranteed to pass validation, then let resume restore
        # warmup_steps/total_steps from the checkpoint (which revalidates).
        ctor_total = max(1, requested_total)
        ctor_warmup = min(cfg.warmup_steps, ctor_total - 1)
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=ctor_warmup,
                                         total_steps=ctor_total, min_lr_ratio=cfg.min_lr_ratio)
    else:
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_steps,
                                         total_steps=requested_total, min_lr_ratio=cfg.min_lr_ratio)
    use_amp = cfg.precision in ("bfloat16", "float16") and _device_of(model).type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.precision == "bfloat16" else torch.float16
    use_scaler = cfg.precision == "float16" and _device_of(model).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    out_dir = Path(cfg.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    t0 = time.perf_counter()
    global_step = 0
    tokens_seen = 0
    if resume_from is not None:
        meta = resume_training_checkpoint(
            optimizer, scheduler, resume_from, map_location="cpu", model=model,
            expected_tokenizer_fingerprint=cfg.tokenizer_fingerprint,
            grad_scaler=scaler if use_scaler else None,
        )
        global_step = int(meta["global_step"])
        tokens_seen = int(meta["tokens_seen"])
        if requested_total != scheduler.total_steps:
            message = (
                f"resume schedule/loop bound mismatch: max_steps/cfg.max_steps={requested_total} "
                f"but the checkpoint restores scheduler total_steps={scheduler.total_steps}; "
                f"the checkpointed LR schedule wins"
            )
            if requested_total > scheduler.total_steps:
                message += (
                    f"; training {requested_total - scheduler.total_steps} extra step(s) "
                    f"will run at the min_lr floor"
                )
            warnings.warn(message, UserWarning, stacklevel=2)

    total_steps = requested_total

    from torch.utils.data import DataLoader
    from neuratal.training.data import collate

    core = model.module if hasattr(model, "module") else model
    dl = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, collate_fn=collate, drop_last=True,
                    pin_memory=(_device_of(model).type == "cuda"),
                    persistent_workers=(cfg.num_workers > 0))
    data_iter = iter(dl)

    # Best-effort data cursor restoration (exact for shuffle=False).
    if resume_from is not None:
        tokens_per_micro = cfg.batch_size * cfg.sequence_length
        consumed = tokens_seen // max(1, tokens_per_micro)
        for _ in range(consumed):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                break

    def joint_loss(ids, labels):
        # Enters only through the (possibly DDP-wrapped) joint loss module on
        # the multi-process path, or the raw forward on the single-process path.
        if ddp_model is not None:
            return ddp_model(ids, labels)
        hidden, moe_stats = model(input_ids=ids, output_hidden_only=True, return_aux=True)
        total_loss, n_tok = chunked_linear_cross_entropy(
            hidden, labels,
            _head_out_proj_weight(core),
            core.embed_tokens.embedding_weight,
        )
        return total_loss, n_tok, moe_stats

    iteration_cap = max(total_steps + 1000, 4 * total_steps)
    iterations = 0
    last_progress_log = time.perf_counter()
    while global_step < total_steps:
        if cfg.deadline_unix is not None and time.time() > cfg.deadline_unix:
            print(
                f"[engine] deadline reached after {time.time() - t0:.0f}s "
                f"(tokens={tokens_seen}, step={global_step}/{total_steps}); "
                "stopping before starting a new optimizer step", flush=True)
            break
        iterations += 1
        if iterations > iteration_cap:
            raise RuntimeError(
                f"training stopped after {iterations} iterations with only "
                f"{global_step}/{total_steps} optimizer steps completed — too many "
                f"consecutive non-finite gradient steps; aborting rather than hanging forever"
            )
        step = global_step + 1  # optimizer-step index used for logs/cadence
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Accumulators stay as detached tensors; no per-micro-step host sync.
        main_sum = torch.zeros((), device=_device_of(model))
        aux_sum = torch.zeros((), device=_device_of(model))
        z_sum = torch.zeros((), device=_device_of(model))
        stats_sum = None
        micro_done = 0
        for _ in range(cfg.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            ids = batch["input_ids"].to(cfg.device, non_blocking=True)
            labels = batch["labels"].to(cfg.device, non_blocking=True)
            with torch.autocast(device_type=_device_of(model).type, dtype=amp_dtype, enabled=use_amp):
                total_loss, n_tok, moe_stats = joint_loss(ids, labels)
                lm_loss = total_loss / n_tok.clamp_min(1)
                aux_loss = (
                    moe_stats.aux_loss
                    if moe_stats is not None and moe_stats.aux_loss is not None
                    else lm_loss.new_zeros(())
                )
                z_loss = (
                    moe_stats.z_loss
                    if moe_stats is not None and moe_stats.z_loss is not None
                    else lm_loss.new_zeros(())
                )
                loss = lm_loss + aux_loss + z_loss
            if use_scaler:
                scaler.scale(loss / cfg.gradient_accumulation_steps).backward()
            else:
                (loss / cfg.gradient_accumulation_steps).backward()
            tokens_seen += ids.numel()
            micro_done += 1
            main_sum = main_sum + lm_loss.detach()
            aux_sum = aux_sum + aux_loss.detach()
            z_sum = z_sum + z_loss.detach()
            if moe_stats is not None:
                if stats_sum is None:
                    stats_sum = {
                        "counts": moe_stats.counts.detach().clone(),
                        "entropy": moe_stats.entropy.detach().clone(),
                        "dropped": moe_stats.dropped.detach().clone(),
                        "aux": moe_stats.aux_loss.detach().clone() if moe_stats.aux_loss is not None else torch.zeros(()),
                        "z": moe_stats.z_loss.detach().clone() if moe_stats.z_loss is not None else torch.zeros(()),
                        "num_assignments": int(moe_stats.num_assignments),
                    }
                else:
                    stats_sum["counts"] += moe_stats.counts.detach()
                    stats_sum["entropy"] += moe_stats.entropy.detach()
                    stats_sum["dropped"] += moe_stats.dropped.detach()
                    if moe_stats.aux_loss is not None:
                        stats_sum["aux"] += moe_stats.aux_loss.detach()
                    if moe_stats.z_loss is not None:
                        stats_sum["z"] += moe_stats.z_loss.detach()
                    stats_sum["num_assignments"] += int(moe_stats.num_assignments)

        if use_scaler:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(train_module.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)   # no-op when non-finite grads were found
            scaler.update()
            skipped = not bool(torch.isfinite(grad_norm))
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(train_module.parameters(), cfg.grad_clip_norm)
            if bool(torch.isfinite(grad_norm)):
                optimizer.step()
                skipped = False
            else:
                # Non-finite grads (bf16/fp32 path): refuse to poison the weights;
                # treat exactly like a GradScaler skip — LR schedule not advanced.
                skipped = True
        if skipped:
            print(
                f"[step {step}] non-finite gradient norm ({float(grad_norm)}); "
                "optimizer step and LR schedule SKIPPED (weights unchanged)",
                flush=True,
            )
            history.append({
                "step": step,
                "iteration": iterations,
                "tokens_seen": tokens_seen,
                "grad_norm": float(grad_norm) if torch.isfinite(grad_norm) else None,
                "skipped": True,
            })
            continue
        scheduler.step()
        global_step += 1

        if is_main and cfg.progress_log_seconds > 0 and (time.perf_counter() - last_progress_log) >= cfg.progress_log_seconds:
            elapsed = time.perf_counter() - t0
            rate = tokens_seen / max(elapsed, 1e-9)
            remaining_tokens = max(0, (total_steps - global_step) * cfg.batch_size
                                   * cfg.sequence_length * cfg.gradient_accumulation_steps)
            eta = remaining_tokens / rate if rate > 0 else float("inf")
            print(
                f"[progress] {elapsed:.0f}s elapsed | tokens={tokens_seen} | "
                f"step {global_step}/{total_steps} | tokens/s={rate:.0f} | eta_s={eta:.0f}",
                flush=True)
            last_progress_log = time.perf_counter()

        if is_main and step % cfg.log_every_steps == 0:
            elapsed = time.perf_counter() - t0
            lr = scheduler.optimizer.param_groups[0]["lr"]
            main_loss = float((main_sum / max(1, micro_done)).item())
            entry = {
                "step": step,
                "iteration": iterations,
                "tokens_seen": tokens_seen,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "main_loss": round(main_loss, 6),
                "elapsed_s": round(elapsed, 2),
            }
            entry.update(_moe_log_fields(stats_sum, micro_done))
            history.append(entry)
            moe_txt = ""
            if "aux_loss" in entry:
                moe_txt = (
                    f" aux={entry['aux_loss']:.4f} z={entry['router_z_loss']:.4f}"
                    f" active_experts={entry['active_experts']}"
                    f" imbalance={entry['expert_load_imbalance']:.2f}"
                    f" dropped={entry['dropped_tokens']}"
                )
            print(
                f"[step {step}] tokens={tokens_seen} lr={lr:.2e} "
                f"loss={main_loss:.4f} gnorm={float(grad_norm):.2f}{moe_txt}"
            )

        if is_main and step % cfg.checkpoint_every_steps == 0:
            save_training_checkpoint(model, optimizer, scheduler, out_dir / f"ckpt-{step}",
                                     global_step=step, tokens_seen=tokens_seen,
                                     tokenizer_fingerprint=cfg.tokenizer_fingerprint,
                                     precision=cfg.precision,
                                     grad_scaler=scaler if use_scaler else None)

        if is_main and step % cfg.eval_every_steps == 0 and val_data is not None:
            model.eval()
            with torch.no_grad():
                from torch.utils.data import DataLoader as _DL
                vdl = _DL(val_data, batch_size=cfg.batch_size, shuffle=False,
                          collate_fn=collate, drop_last=True)
                tot, cnt = 0.0, 0
                for batch in vdl:
                    ids = batch["input_ids"].to(cfg.device, non_blocking=True)
                    labels = batch["labels"].to(cfg.device, non_blocking=True)
                    with torch.autocast(device_type=_device_of(model).type, dtype=amp_dtype,
                                        enabled=use_amp):
                        hidden = model(input_ids=ids, output_hidden_only=True, return_aux=False)
                        total_loss, n_tok = chunked_linear_cross_entropy(
                            hidden, labels,
                            _head_out_proj_weight(core),
                            core.embed_tokens.embedding_weight,
                        )
                        loss = total_loss / n_tok.clamp_min(1)
                    tot += float(loss) * ids.numel()
                    cnt += ids.numel()
                val_loss = tot / max(cnt, 1)
                history.append({"step": step, "val_loss": val_loss})
                print(f"[step {step}] val_loss={val_loss:.4f}")

    if is_main and cfg.save_final_checkpoint:
        save_training_checkpoint(model, optimizer, scheduler, out_dir / "ckpt-final",
                                 global_step=global_step, tokens_seen=tokens_seen,
                                 tokenizer_fingerprint=cfg.tokenizer_fingerprint,
                                 precision=cfg.precision,
                                 grad_scaler=scaler if use_scaler else None)
        print(f"[engine] final checkpoint saved to {out_dir / 'ckpt-final'}", flush=True)

    if is_main:
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, history
