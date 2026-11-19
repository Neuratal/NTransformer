# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""End-to-end pipeline benchmark -> results/end_to_end_results_<ts>.json

Measures: encode -> model prefill -> KV-cached generation -> decode, with
per-stage timings, for a fixed set of prompts and generation lengths.
The model is untrained; texts are gibberish by design.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from benchmark_utils import machine_fingerprint, run_id, save_result, SEED

from neuratal import NeuratalConfig, NeuratalModel, generate
from neuratal.integration import NTokenizerAdapter
import os
NTOKENIZER_ROOT = Path(os.environ.get("NTOKENIZER_ROOT", Path(__file__).resolve().parents[2] / "NTokenizer"))
NTOK = NTOKENIZER_ROOT / "output" / "v0.3" / "ntokenizer-0.3-256k.ntok"

PROMPTS = {
    "turkish": "Bugün hava gerçekten çok güzel, hep birlikte dışarı çıkalım ve manzarayı seyredelim.",
    "english": "The quick brown fox jumps over the lazy dog while the committee of experts watches carefully.",
    "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--gen-lengths", default="16,32,64")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    gen_lengths = [int(x) for x in args.gen_lengths.split(",")]

    if not NTOK.is_file():
        raise SystemExit(f"tokenizer artifact not found: {NTOK}")

    torch.manual_seed(SEED)
    adapter = NTokenizerAdapter(NTOK)
    model = NeuratalModel(NeuratalConfig(
        eos_token_id=adapter.eos_token_id, pad_token_id=adapter.pad_token_id,
    )).to(device, dtype).eval()

    results = {
        "run_id": run_id(),
        "machine": machine_fingerprint(),
        "config": {**model.config.to_dict(), "model_dtype": args.dtype, "device": str(device)},
        "tokenizer_sha256": adapter.manifest["tokenizer_sha256"],
        "prompts": {},
    }

    for name, text in PROMPTS.items():
        prompt_ids = adapter.encode(text)
        input_ids = torch.tensor([prompt_ids], device=device)
        entry = {
            "text": text,
            "prompt_tokens": len(prompt_ids),
            "prompt_bytes": len(text.encode("utf-8")),
            "gen": {},
        }

        # stage timings per generation length
        for gen_len in gen_lengths:
            encode_times, prefill_times, gen_times, decode_times, total_times = [], [], [], [], []
            decode_bytes_list = []
            for _ in range(args.repeats):
                # 1) encode
                t0 = time.perf_counter()
                ids = adapter.encode(text)
                encode_t = time.perf_counter() - t0
                input_ids = torch.tensor([ids], device=device)

                # 2) prefill + 3) generation, timed separately via generate() stages
                torch.cuda.synchronize() if device.type == "cuda" else None
                t0 = time.perf_counter()
                with torch.inference_mode():
                    from neuratal.model.cache import KVCache

                    cache = KVCache(model.config)
                    prefill_out = model(input_ids, past_key_values=cache)
                torch.cuda.synchronize() if device.type == "cuda" else None
                prefill_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                generated = generate(model, input_ids, max_new_tokens=gen_len,
                                     eos_token_id=adapter.eos_token_id)
                torch.cuda.synchronize() if device.type == "cuda" else None
                gen_t = time.perf_counter() - t0

                new_ids = generated[0, len(ids):].tolist()
                decoded = adapter.decode_bytes(new_ids)
                decode_bytes = len(decoded)
                # 4) decode
                t0 = time.perf_counter()
                _ = adapter.decode_bytes(new_ids)
                decode_t = time.perf_counter() - t0

                encode_times.append(encode_t)
                prefill_times.append(prefill_t)
                gen_times.append(gen_t)
                decode_times.append(decode_t)
                total_times.append(encode_t + prefill_t + gen_t + decode_t)
                decode_bytes_list.append(decode_bytes)

            import statistics

            median = statistics.median
            entry["gen"][str(gen_len)] = {
                "repeats": args.repeats,
                "encode_median_s": median(encode_times),
                "prefill_median_s": median(prefill_times),
                "generation_median_s": median(gen_times),
                "decode_median_s": median(decode_times),
                "total_median_s": median(total_times),
                "generation_ms_per_token": median(gen_times) / gen_len * 1e3,
                "encode_bytes_per_second": len(text.encode("utf-8")) / median(encode_times),
                "decode_bytes_per_second": (statistics.median(decode_bytes_list) / median(decode_times)) if decode_times else 0.0,
            }

        results["prompts"][name] = entry
        print(f"prompt={name:12s} tokens={entry['prompt_tokens']}")
        for gen_len in gen_lengths:
            row = entry["gen"][str(gen_len)]
            print(f"  gen {gen_len:>2}: encode {row['encode_median_s']*1e3:.3f} ms  "
                  f"prefill {row['prefill_median_s']*1e3:.2f} ms  "
                  f"gen {row['generation_ms_per_token']:.2f} ms/tok  "
                  f"decode {row['decode_median_s']*1e3:.3f} ms")

    path = save_result("end_to_end", results)
    print(f"end-to-end results -> {path}")


if __name__ == "__main__":
    main()
