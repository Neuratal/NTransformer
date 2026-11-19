# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Neuratal Transformer Contributors
"""Create an immutable version snapshot under versions/.

Snapshot contents:
    versions/<name>/
        SOURCE/                 copy of neuratal/, tests/, pyproject.toml, README.md
        commit.txt              git commit hash the snapshot was frozen at
        environment.json        host fingerprint at freeze time
        test_results.txt        captured pytest output (full suite)
        file_manifest.json      SHA-256 of every frozen file
        CHANGELOG.md            what this candidate is (to fill in)
        LIMITATIONS.md          known issues at freeze time
        benchmark_results/      copies of benchmarks/results as of the freeze
        registry entry appended/updated in registry/best_version.json

Usage:
    python scripts/make_version_snapshot.py v0.2-baseline --commit <sha> --note "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "versions"
REGISTRY = ROOT / "registry"
FREEZE_PATHS = ["neuratal", "tests", "pyproject.toml", "README.md"]


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd or ROOT, check=True, capture_output=True, text=True).stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--commit", default=None, help="default: HEAD")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    dest = VERSIONS / args.name
    if dest.exists():
        raise SystemExit(f"refusing to overwrite existing snapshot: {dest}")
    dest.mkdir(parents=True)

    commit = args.commit or run(["git", "rev-parse", "HEAD"]).strip()
    (dest / "commit.txt").write_text(commit + "\n")

    src = dest / "SOURCE"
    src.mkdir()
    for rel in FREEZE_PATHS:
        p = ROOT / rel
        if p.is_dir():
            shutil.copytree(p, src / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(p, src / rel)

    env = json.load(open(ROOT / "benchmarks/results/environment.json")) if (ROOT / "benchmarks/results/environment.json").exists() else {}
    (dest / "environment.json").write_text(json.dumps(env, indent=2))

    print(f"freezing at {commit} ... running full test suite")
    test_out = run([sys.executable, "-m", "pytest", "tests/", "-q", "-o", "addopts="])
    (dest / "test_results.txt").write_text(test_out)

    manifest = {}
    for path in sorted(src.rglob("*")):
        if path.is_file():
            manifest[str(path.relative_to(src))] = sha256_file(path)
    (dest / "file_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    bench = dest / "benchmark_results"
    bench.mkdir()
    results_dir = ROOT / "benchmarks/results"
    if results_dir.is_dir():
        for f in results_dir.glob("*.json"):
            if f.name.endswith("_latest.json") or "_" not in f.stem:
                shutil.copy2(f, bench / f.name)
            else:
                shutil.copy2(f, bench / f.name)

    REGISTRY.mkdir(exist_ok=True)
    registry_path = REGISTRY / "best_version.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    else:
        registry = {"best": None, "candidates": []}
    registry["candidates"].append({
        "name": args.name,
        "commit": commit,
        "note": args.note,
        "test_summary": test_out.strip().splitlines()[-1] if test_out.strip() else "unknown",
        "promoted": False,
    }) if not any(c.get("name") == args.name for c in registry["candidates"]) else None
    registry_path.write_text(json.dumps(registry, indent=2))
    print(f"snapshot -> {dest}")
    print(test_out.strip().splitlines()[-1])


if __name__ == "__main__":
    main()
