# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Minimal end-to-end example: NTokenizer -> NeuratalTransformer generation.

What this shows
---------------
1. (Optional) Load the real NTokenizer v0.3 model and encode text to IDs.
2. Build a NeuratalModel (development preset by default, or a checkpoint via
   --checkpoint).
3. Generate token IDs greedily.
4. (Optional) Decode the generated IDs back to text with NTokenizer.

IMPORTANT: the development preset is a fresh, *untrained* network — its outputs
are random gibberish ids. This example validates the plumbing, not model
quality. Point --checkpoint at a trained checkpoint to get meaningful text.
No training happens anywhere in this project.

Usage
-----
    python examples/inference.py --prompt "Hello, world!"
    python examples/inference.py --checkpoint path/to/ckpt --prompt "..." --max-new-tokens 32
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


def load_tokenizer(path: Path):
    sys.path.insert(0, str(NTOKENIZER_ROOT / "python"))
    try:
        from neuratal_tokenizer import Tokenizer
    except Exception as exc:
        raise SystemExit(f"NTokenizer bindings unavailable: {exc}")
    return Tokenizer.from_file(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="Hello, world!", help="prompt text (used only if the tokenizer loads)")
    parser.add_argument("--checkpoint", default=None, help="optional checkpoint directory (config.json + model.safetensors)")
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER), help="path to a .ntok tokenizer model")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--eos-token-id", type=int, default=262_138, help="<|eos|> for NTokenizer v0.3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from neuratal import NeuratalConfig, NeuratalModel, generate, load_pretrained

    device = torch.device(args.device)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = load_pretrained(args.checkpoint, map_location=device)
    else:
        print("No checkpoint given — using the development preset with UNTRAINED weights.")
        print("Output will be gibberish; this validates the pipeline only.")
        model = NeuratalModel(NeuratalConfig())
    model = model.to(device).eval()

    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.is_file():
        tokenizer = load_tokenizer(tokenizer_path)
        prompt_ids = tokenizer.encode(args.prompt)
        print(f"Prompt: {args.prompt!r} -> {len(prompt_ids)} ids: {prompt_ids}")
    else:
        tokenizer = None
        prompt_ids = [168069, 5431, 33]  # ids of "Hello, world!" under NTokenizer v0.3
        print(f"Tokenizer model not found at {tokenizer_path}; using fixed demo ids {prompt_ids}")

    input_ids = torch.tensor([prompt_ids], device=device)
    out_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=args.eos_token_id,
    )
    new_ids = out_ids[0, len(prompt_ids):].tolist()
    print(f"Generated {len(new_ids)} new token ids: {new_ids}")
    if tokenizer is not None:
        try:
            print(f"Decoded: {tokenizer.decode(new_ids)!r}")
        except Exception as exc:
            print(f"Decoded: <not valid UTF-8 ({exc}); raw ids above>")
    print(f"Model parameters: {model.num_parameters():,}")


if __name__ == "__main__":
    main()
