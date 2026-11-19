# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Neuratal end-to-end integration example: text → model → text.

Technical integration demonstration for the *untrained* development preset —
the generated text is gibberish and does not reflect model quality. No
training happens anywhere in this project.

Pipeline
--------
    NTokenizer v0.3 (.ntok) --encode--> IDs --NeuratalModel--> logits/IDs --decode--> text

Steps performed (each visible):
1. Load + verify the NTokenizer v0.3 artifact (CRC-checked NTOK parse, embedded
   metadata cross-check, SHA-256 manifest).
2. Load a checkpoint (--checkpoint) or build the development preset.
3. Verify tokenizer ↔ checkpoint compatibility (vocab size, EOS/PAD ids, sidecar).
4. Tokenize the prompt; forward; generate with KV cache; stop at the
   artifact's EOS id (never hardcoded).
5. Decode the generated ids (strict UTF-8 with graceful fallback note).

Usage
-----
    python examples/neuratal_inference.py --prompt "Merhaba, dünya!" --max-new-tokens 16
    python examples/neuratal_inference.py --checkpoint path/to/ckpt --save-checkpoint path/to/ckpt2
    python examples/neuratal_inference.py --tokenizer /abs/path/to/model.ntok
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NTOKENIZER_ROOT = Path(os.environ.get("NTOKENIZER_ROOT", PROJECT_ROOT.parent / "NTokenizer"))
DEFAULT_TOKENIZER = NTOKENIZER_ROOT / "output" / "v0.3" / "ntokenizer-0.3-256k.ntok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="Merhaba, dünya!", help="prompt text (tokenizer-native)")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint directory with config.json + model.safetensors "
                             "(default: untrained development preset)")
    parser.add_argument("--save-checkpoint", default=None,
                        help="optionally write the used checkpoint + tokenizer.json sidecar here")
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER), help="path to the .ntok artifact")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--do-sample", action="store_true", help="temperature sampling instead of greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from neuratal import NeuratalConfig, NeuratalModel, generate, save_pretrained
    from neuratal.integration import (
        NTokenizerAdapter,
        TokenizerMismatchError,
        verify_checkpoint_tokenizer,
        write_tokenizer_sidecar,
    )
    from neuratal.serialization.checkpoint import load_pretrained

    device = torch.device(args.device)
    ntok = Path(args.tokenizer)
    if not ntok.is_file():
        raise SystemExit(f"tokenizer artifact not found: {ntok}\n"
                         f"Set NTOKENIZER_ROOT or pass --tokenizer /abs/path/to/model.ntok")

    # --- Step 1+2: load tokenizer + model, keep them pinned to each other ---
    adapter = NTokenizerAdapter(ntok)
    print(f"tokenizer: {adapter.manifest['tokenizer_name']} v{adapter.manifest['model_version']} "
          f"({adapter.manifest['format']} v{adapter.manifest['format_version']}) "
          f"vocab={adapter.vocab_size} eos={adapter.eos_token_id}")

    if args.checkpoint:
        model = load_pretrained(args.checkpoint, map_location=device)
        print(f"checkpoint: loaded {args.checkpoint}")
    else:
        print("checkpoint: UNTRAINED development preset (integration demo only — output is gibberish)")
        model = NeuratalModel(NeuratalConfig(
            eos_token_id=adapter.eos_token_id,
            pad_token_id=adapter.pad_token_id,
            bos_token_id=adapter.bos_token_id,
        ))
    model = model.to(device).eval()

    # --- Step 3: compatibility verification (hard requirement) ---
    try:
        if args.checkpoint:
            report = verify_checkpoint_tokenizer(model.config, args.checkpoint, ntok, strict_sidecar=False)
            print(f"compatibility: {report['checks']}")
        else:
            adapter.assert_model_compatible(model.config)
            print("compatibility: model vocab_size == tokenizer vocab_size (262,144) OK")
    except TokenizerMismatchError as exc:
        raise SystemExit(f"tokenizer/checkpoint mismatch:\n{exc}")

    if args.save_checkpoint:
        save_pretrained(model, args.save_checkpoint)
        write_tokenizer_sidecar(ntok, args.save_checkpoint)
        print(f"checkpoint saved to {args.save_checkpoint} (+ tokenizer.json sidecar)")

    # --- Step 4: tokenize ---
    prompt_ids = adapter.encode(args.prompt)
    print(f"prompt ({len(prompt_ids)} ids): {prompt_ids}")
    input_ids = torch.tensor([prompt_ids], device=device)

    # --- Step 5: inference + generation (KV cache, artifact EOS id) ---
    generated = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=adapter.eos_token_id,
    )
    new_ids = generated[0, len(prompt_ids):].tolist()
    stopped_at_eos = len(new_ids) > 0 and new_ids[-1] == adapter.eos_token_id
    print(f"generated {len(new_ids)} ids ({'stopped at EOS' if stopped_at_eos else 'max_new_tokens reached'}): {new_ids}")

    # --- Step 6: decode ---
    try:
        print(f"decoded: {adapter.decode(new_ids)!r}")
    except Exception as exc:
        print(f"decoded: <not valid UTF-8 ({exc}); raw bytes: {adapter.decode_bytes(new_ids)!r}>")

    print(f"\nmodel parameters: {model.num_parameters():,}  "
          f"(untrained — this demonstrates the pipeline, not language quality)")


if __name__ == "__main__":
    main()
