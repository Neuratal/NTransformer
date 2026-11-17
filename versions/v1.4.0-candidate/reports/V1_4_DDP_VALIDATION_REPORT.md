# V1.4 — DDP Validation Report

**Question:** does distributed training through `neuratal.training.engine` work — now, not hypothetically?

**Answer: yes, on two-process CPU gloo. Not verified on multi-GPU/NCCL (single GPU on this machine).**

## 1. Environment

| Item | Value |
|---|---|
| Backend tested | `gloo` |
| Processes | 2 per run, same node |
| Device | CPU (CUDA hidden from the validation runs via `CUDA_VISIBLE_DEVICES=""`) |
| Framework | torch 2.15.0+cu132, Python 3.14.4 |
| What this is NOT | A multi-GPU/NCCL result; a throughput/scaling result; a multi-node result |

## 2. The v1.3 defect under test (BUG-01)

`train()` wraps the module in `DistributedDataParallel(..., find_unused_parameters=True)` for MoE.
The training forward entered through `output_hidden_only=True` and stopped **before** the LM head;
the head was applied **outside** the DDP-wrapped forward inside `chunked_linear_cross_entropy`,
consuming `output_projection.out_proj.weight` (all seven shipped MoE presets have factorized tied
heads: F ∈ {64, 128} << hidden ∈ {640 … 16384}, verified programmatically). The reducer marked
that parameter "ready (unused)" at the end of the forward; its autograd hook fired during backward
→ `RuntimeError: Expected to mark a variable ready only once … out_proj.weight has been marked as
ready twice` on both ranks at the **first** backward, before any optimizer step.

Reproduced against unmodified v1.3 export of the audited commit before the fix
(`/home/justerdem14/.cache/ntransformer_audit/v1.3_baseline/`), and the same workload runs fine on v1.4.

## 3. Fix

`neuratal/training/engine.py`: an internal `_DDPLossModule` performs hidden-only encoding **and**
the chunked LM loss inside one DDP-wrapped forward (`forward(input_ids, labels)`). Every
reducer-relevant parameter — including factorized `out_proj.weight` and the tied embedding — lives
inside the forward graph; MoE zero-token experts remain covered by
`find_unused_parameters=True`. The loss is exactly the chunked cross-entropy: mathematically
identical (no full `[B, T, vocab]` logits materialized; O(chunk) peak memory), verified by the
single-process reference below (bitwise weights and grads). No math, loss or memory-format change.
Single-process path is byte-for-byte the same computation as v1.3 (the wrapper only exists in
multi-process mode).

## 4. Validation matrix (all on this machine)

Deterministic fixtures: `SyntheticTokenDataset(1, 8, seed=42)` (one sequence, batch 1) so
permutations cannot hide ordering effects; `TrainConfig(seed=1234, float32, warmup_steps=1,
max_steps=2, gradient_accumulation_steps=1|2)`; identical dataset on both ranks per run.

| # | Configuration | Head | Result | Notes |
|---|---|---|---|---|
| 1 | MoE (`num_experts=2, top_k=1`, 1 MoE layer), factored (F=8 < hidden=32) | factorized | **PASS** — completes 2 steps | v1.3: crash at step 1 |
| 2 | MoE, full-width (`embedding_size=None`) | full-width | **PASS** (and in v1.3; only warning was benign) | |
| 3 | Dense, factored | factorized | **PASS** — bitwise-equal to single-process | |
| 4 | Dense, full-width | full-width | **PASS** | |
| 5 | MoE, factored, gradient accumulation ×2 | factorized | **PASS** — bitwise-equal to single-process | |
| 6 | MoE, factored + capacity (`capacity_factor=0.5, min_capacity=1`) | factorized | **PASS** — steps complete, `dropped_tokens` logged (8, 9) | no reducer errors despite token drops |
| 7 | MoE, skewed router (`router_aux_loss_coef=0.1, router_z_loss_coef=0.01`) | factorized | **PASS** | |
| 8 | Checkpointed DDP run (`checkpoint_every_steps=1`) → `resume_training_checkpoint` → fresh `NeuratalModel` | factorized | **PASS** — `global_step==2`; reloaded weights bitwise-equal to rank 0's post-DDP weights | |
| 9 | Bitwise parity (params): DDP vs single-process on identical data | factorized | **max diff 0.0, bitwise equal** (diag + test) | |
| 10 | Rank-to-rank parity (params + grads) post `all-reduce` | factorized | **bitwise equal on both ranks** | |
| 11 | Gradient parity vs single-process | factorized | **bitwise equal** | covers tied matrix from both paths (input + head) |

Config equivalence for the shipped presets (all 7): the fix is topology-dependent, not
size-dependent — every shipped MoE preset has the identical factorized-head + tied-embedding
topology (`embedding_size ∈ {64, 128}`, `hidden ∈ {640…16384}`), and every one of them passes
through the same `_DDPLossModule` path under `train()`. Reproducing 500B-parameter DDP training
is neither feasible nor necessary to validate the mechanism; the exact factorized-head path is
independently verified above.

## 5. What the tests assert (not just "no exception")

- DDP completes ≥ 2 optimizer steps with history recorded (loss finite).
- **Post-all-reduce gradients and parameters are bitwise identical** across ranks.
- **DDP result equals single-process result bitwise** (params **and** grads) on identical data —
  so forward, backward, all-reduce, accumulation, and the optimizer step are all validated
  end-to-end, including through the tied embedding (used at both embedding and head paths) and
  the factorized `out_proj`.
- DDP-saved checkpoints resume into a fresh process with weight equality and counter parity.
- Capacity-dropped micro-batches under DDP do not wedge the reducer.

## 6. Failure-mode evidence (negative control)

- Unmodified v1.3 export: same 2-proc workload → deterministic `RuntimeError` (both ranks, step 1).
- v1.4 code with `find_unused_parameters=True` retained: **passes** (experts with zero tokens
  remain safe) — the flag is NOT blindly disabled.
- Full-width MoE (no factorized head): passes in v1.4, matching v1.3 behavior.
- Dense with `find_unused_parameters=False`: passes.
- A skipped step (forced non-finite grads): optimizer step + LR schedule both skipped; both ranks
  make the identical decision (all-reduce yields identical grads) — verified in
  `TestDDPEndToEnd` + `TestEngineStockFp16`.

## 7. Honest limitations

- **No multi-GPU/NCCL was tested** — only single-node, 2-process, CPU/gloo. NCCL kernel paths,
  inter-GPU ordering, and async reductions are untested; throughput/scaling claims are none.
- **CPU gloo is a correctness validation of the reducer**, not of performance.
- Per-rank router statistics remain per-rank (documented v1.x limitation; not fixed).
- DDP gradient accumulation uses per-micro-batch all-reduce (correct, not optimal); `no_sync`
  was not introduced (behavior preservation chosen over an unproven optimization).
- DDP validation does not exercise `master`/`v1.2` branches; it exercises the v1.4 tree.

## 8. Reproduction

```
pytest tests/test_v14_regressions.py -k DDP -q     # 6 tests, ~20 s, gloo auto-skipped if absent
python /home/justerdem14/.cache/ntransformer_audit/diag_ddp_1sample.py      # bitwise parity probe
python /home/justerdem14/.cache/ntransformer_audit/diag_ddp_capacity.py     # drops under DDP
python /home/justerdem14/.cache/ntransformer_audit/diag_ddp_moe.py          # v1.3 crashes / v1.4 passes
```
