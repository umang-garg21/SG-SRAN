#!/usr/bin/env python3
"""Evaluate and aggregate completed few-shot 4x4 runs across seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis/out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--force", action="store_true", help="Recompute per-seed metric JSONs.")
    parser.add_argument(
        "--aggregate-prefix",
        default="fewshot_4x4_hardened_metrics_5seed",
        help="Prefix for the aggregate summary files.",
    )
    return parser.parse_args()


def seed_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    seeds = seed_list(args.seeds)
    for seed in seeds:
        manifest = OUT_DIR / f"fewshot_4x4_manifest_s{seed}.json"
        if not manifest.exists():
            print(f"skip seed {seed}: missing {manifest.relative_to(ROOT)}")
            continue
        out_json = OUT_DIR / f"fewshot_4x4_hardened_metrics_s{seed}.json"
        if out_json.exists() and not args.force:
            print(f"skip seed {seed}: existing {out_json.relative_to(ROOT)}")
            continue
        cmd = [
            sys.executable,
            str(ROOT / "analysis/evaluate_fewshot_4x4.py"),
            "--manifest",
            str(manifest),
            "--out-prefix",
            f"fewshot_4x4_hardened_metrics_s{seed}",
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    aggregate_cmd = [
        sys.executable,
        str(ROOT / "analysis/aggregate_fewshot_4x4_multiseed.py"),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--out-prefix",
        args.aggregate_prefix,
    ]
    print(" ".join(aggregate_cmd), flush=True)
    subprocess.run(aggregate_cmd, check=True)


if __name__ == "__main__":
    main()
