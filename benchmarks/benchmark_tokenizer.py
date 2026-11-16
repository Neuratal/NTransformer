"""NTokenizer throughput benchmark -> benchmarks/results/tokenizer_results_<ts>.json

Measured separately: artifact load time vs steady-state encode/decode.
Inputs: short / medium / long, Turkish, English, code, multilingual.
The tokenizer is single-threaded per call; throughput here is per-thread.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from benchmark_utils import run_id, save_result, SEED

from neuratal.integration import NTokenizerAdapter
import os
NTOKENIZER_ROOT = Path(os.environ.get("NTOKENIZER_ROOT", Path(__file__).resolve().parents[2] / "NTokenizer"))
NTOK = NTOKENIZER_ROOT / "output" / "v0.3" / "ntokenizer-0.3-256k.ntok"

CORPUS = {
    "short_en": "Hello, world!",
    "short_tr": "Merhaba, dünya!",
    "medium_en": ("The Transformer architecture relies on scaled dot-product attention. "
                   "This paragraph is a few sentences long and exists to give the tokenizer "
                   "a realistic medium-length input for steady-state measurement purposes.\n") * 4,
    "medium_tr": ("İstanbul, Türkiye'nin en kalabalık şehridir ve tarih boyunca bir ticaret merkezi "
                   "olmuştur. Bu metin, tokenizerin orta uzunluktaki Türkçe metinlerde performansını "
                   "ölçmek için gerçekçi bir örnek oluşturur.\n") * 4,
    "code_python": ("def forward(self, x):\n    h = self.norm(x)\n    attn = self.attn(h)\n"
                     "    h = x + attn\n    return h + self.mlp(self.norm(h))\n\n"
                     "class Block(nn.Module):\n    def __init__(self, dim):\n        super().__init__()\n"
                     "        self.norm = RMSNorm(dim)\n        self.mlp = SwiGLU(dim)\n") * 6,
    "multilingual": ("İstanbul東京القاهرةМоскваSão PauloMünchen — mixed scripts 12345 ✓🚀\n"
                      "हिन्दी · 日本語 · العربية · Türkçe · Français · Español · Deutsch\n") * 30,
    "numbers": "".join(f"{i} " for i in range(2000)),
}


def time_call(fn, reps: int) -> dict:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    import statistics

    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1) + 0.999999))]
    return {"reps": reps, "median_s": statistics.median(samples), "p95_s": p95,
            "mean_s": statistics.fmean(samples), "min_s": min(samples), "max_s": max(samples)}


def main() -> None:
    if not NTOK.is_file():
        raise SystemExit(f"tokenizer artifact not found: {NTOK}")

    torch.manual_seed(SEED)

    t0 = time.perf_counter()
    adapter = NTokenizerAdapter(NTOK)
    load_time = time.perf_counter() - t0
    print(f"artifact load (binding + parse + verify): {load_time:.4f}s")

    results = {
        "run_id": run_id(),
        "artifact": str(NTOK),
        "artifact_sha256": adapter.manifest["tokenizer_sha256"],
        "artifact_load_seconds": load_time,
        "vocab_size": adapter.vocab_size,
        "special_tokens": adapter.special_tokens,
        "corpora": {},
        "batch": {},
    }

    for name, text in CORPUS.items():
        n_bytes = len(text.encode("utf-8"))
        ids = adapter.encode(text)

        # warm-up once, then measured reps
        adapter.encode(text)
        adapter.decode(ids)
        encode_stats = time_call(lambda: adapter.encode(text), reps=_reps_for(n_bytes))
        decode_stats = time_call(lambda: adapter.decode(ids), reps=_reps_for(n_bytes))
        decode_bytes_stats = time_call(lambda: adapter.decode_bytes(ids), reps=_reps_for(n_bytes))
        enc_med = encode_stats["median_s"]
        dec_med = decode_stats["median_s"]
        results["corpora"][name] = {
            "bytes": n_bytes,
            "token_ids": len(ids),
            "bytes_per_token": n_bytes / max(len(ids), 1),
            "encode": encode_stats,
            "encode_bytes_per_second": n_bytes / enc_med,
            "encode_tokens_per_second": len(ids) / enc_med,
            "encode_mib_per_second": n_bytes / enc_med / 2**20,
            "decode": decode_stats,
            "decode_bytes_per_second": n_bytes / dec_med,
            "decode_tokens_per_second": len(ids) / dec_med,
            "decode_mib_per_second": n_bytes / dec_med / 2**20,
            "decode_bytes": decode_bytes_stats,
        }

    # batch encode throughput (32 medium texts)
    batch_texts = [CORPUS["medium_en"]] * 32
    t0 = time.perf_counter()
    batch_out = adapter.encode_batch(batch_texts)
    batch_time = time.perf_counter() - t0
    total_bytes = sum(len(t.encode("utf-8")) for t in batch_texts)
    results["batch"] = {
        "n_texts": len(batch_texts),
        "total_bytes": total_bytes,
        "wall_seconds": batch_time,
        "encode_bytes_per_second": total_bytes / batch_time,
        "encode_mib_per_second": total_bytes / batch_time / 2**20,
        "tokens_per_second": sum(len(x) for x in batch_out) / batch_time,
    }

    path = save_result("tokenizer", results)
    print(f"\ntokenizer results -> {path}")
    for name, row in results["corpora"].items():
        print(f"{name:14s} {row['bytes']:>8}B  {row['token_ids']:>6} ids  "
              f"enc {row['encode_mib_per_second']:.2f} MiB/s  dec {row['decode_mib_per_second']:.2f} MiB/s  "
              f"B/T {row['bytes_per_token']:.2f}")
    print(f"batch(32): {results['batch']['encode_mib_per_second']:.2f} MiB/s")


def _reps_for(n_bytes: int) -> int:
    if n_bytes < 128:
        return 300
    if n_bytes < 2048:
        return 150
    if n_bytes < 16384:
        return 40
    return 8


if __name__ == "__main__":
    main()
