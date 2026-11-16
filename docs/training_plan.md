# AŞAMA 3 — Training Gate: Plan (DO NOT EXECUTE WITHOUT EXPLICIT CONSENT)

Status: **not started. No training of any kind has been run, is running, or is
scheduled.** This document exists so a future training decision can be made
from informed prerequisites — nothing here consumes data, GPU, or money.

## Why a separate gate
Architecture and inference R&D ship no language capability. Frontier models
are capable because of pretraining on trillions of tokens + RLHF-style
post-training + safety + serving stacks. None of those is trainable by an
architectural iteration. The claim "matches frontier AI" cannot be made or
measured from this repository as-is; model capability stays **ölçülemedi**
until a training run exists.

## Prerequisites (all before a single FLOP is spent)

1. **Scope & licensing**
   - Target: a small capable dense/hybrid model for the Turkish + English +
     multilingual mix, sized to this machine class (≤ a few B, or a
     50M-500M pilot).
   - Dataset candidates (public, license-verified): FineWeb-Edu, SmolLM
     corpora, StarCoder2 data, Dolma, Pile subsets, Turkish web (CC-BY),
     Wikipedia/books (open-licensed). Every shard needs a license ledger.
   - No leaked, proprietary, or unverifiable corpora.

2. **Tokenizer compatibility**
   - NTokenizer v0.3 (262,144) is the mandated vocab — IDs unchanged.
   - Verify fertility (bytes/token) per language on chosen corpora; our
     BPE is byte-level so coverage is total; measure compression ratio
     (expected ~4.0 B/T on English, see NTokenizer reports; Turkish will differ).

3. **Compute budget & environment**
   - Single-laptop GPU (RTX 4060 8 GB) cannot pretrain a frontier model;
     a 50M-class pilot is possible in days; larger requires external budget.
   - Checkpointing every N steps; gradient accumulation; grad clipping;
     AdamW; cosine LR with warmup; bf16 with loss scaling.

4. **Architecture choice (from measured candidates, not defaults)**
   - v0.4 baseline (dense GQA, tied factorized embeddings) — safe default.
   - v0.5-candidate GDN — only with quality evidence; untrainable to verify here.
   - v0.6-candidate (chunkwise GDN) — depends on pending v0.6 outcome.
   - Decision rule: whichever has the best measured loss/mbit after a 50M pilot
     (~5-10B tokens), not vibes.

5. **Evaluation (committed before training)**
   - Held-out perplexity (fixed seed, fixed split), lambada/wiki-text style
     probes limited to license-safe corpora, Turkish and English splits.
   - No public test-set contamination; no leaderboard gaming.
   - Baselines: our own 50M-dense vs 50M-GDN; external references only via
     published scores (labeled as external, unverified here).

6. **Post-training (each needs its own consent)**
   - SFT on permissive instruction sets (e.g., open chat datasets).
   - Preference optimization (DPO) on licensed preference data.
   - Safety: abuse-refusal eval, jailbreak probes, PII filtering in data.

7. **Serving & tools** (build only after a checkpoint exists)
   - vLLM/TensorRT-LLM style serving, or our own KV-split serving.
   - Tool use requires the model to actually support tool schemas — training-gated.

## Honest cost estimate (order of magnitude)
- 50M pilot (dense, tied factorized): weeks on a single 8 GB GPU, ~$0 if
  electricity ignored; ~100 MB checkpoint.
- 1-7B model: months / ~100-1000× compute; requires cluster access or capital;
  not realistic on this machine.
- Frontier parity: not this project's feasible goal; goal instead = a
  well-measured open small model + honest eval.

## What I need from you (explicit consent items)
A. Target capability (pilot vs ambitious) & license acceptances per corpus.
B. Budget/hardware (this machine only? cloud budget?).
C. Training objective priority (Turkish-strong? code? general?).
D. Safety requirements (Turkish-only safety data? release as open weights?).

Until A–D land, the training track does not start; architectural R&D continues.
