#!/usr/bin/env python3
"""Merge OCRP stagewise decoded-error shards into one material payload."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Paper/202608_Umang_EBSD_SR_fwd/EBSD_SR_Nature_NMI"
EVAL_DIR = PAPER_DIR / "evals"

STAGE_ORDER = ["encode_lr", "context_refine", "routed_patch", "hr_refine_1", "hr_refine_2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-glob", required=True, help="Glob, relative to eval dir unless absolute.")
    parser.add_argument("--out-tag", required=True)
    parser.add_argument("--expected-samples", type=int, default=None)
    return parser.parse_args()


def resolve_glob(pattern: str) -> list[Path]:
    root = Path(pattern)
    if root.is_absolute():
        paths = sorted(root.parent.glob(root.name))
    else:
        paths = sorted(EVAL_DIR.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths


def read_rows(csv_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = set()
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["sample_index"]), row["stage"])
                if key in seen:
                    raise ValueError(f"Duplicate sample/stage row {key} in {path}")
                seen.add(key)
                out: dict[str, Any] = dict(row)
                out["sample_index"] = int(row["sample_index"])
                for metric in ("mean_deg", "median_deg", "p90_deg", "p95_deg"):
                    out[metric] = float(row[metric])
                rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        by_stage[str(row["stage"])].append(row)
        labels[str(row["stage"])] = str(row["label"])
    summary: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        vals = np.asarray([float(row["mean_deg"]) for row in by_stage.get(stage, [])], dtype=np.float64)
        if vals.size == 0:
            continue
        summary.append(
            {
                "stage": stage,
                "label": labels.get(stage, stage),
                "mean_deg": float(np.nanmean(vals)),
                "std_patch_mean_deg": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
                "n_samples": int(vals.size),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    csv_paths = resolve_glob(args.shard_glob)
    rows = read_rows(csv_paths)
    rows.sort(key=lambda row: (int(row["sample_index"]), STAGE_ORDER.index(row["stage"])))

    sample_ids = sorted({int(row["sample_index"]) for row in rows})
    if args.expected_samples is not None and len(sample_ids) != int(args.expected_samples):
        missing = sorted(set(range(int(args.expected_samples))).difference(sample_ids))
        raise RuntimeError(
            f"Expected {args.expected_samples} samples, found {len(sample_ids)}; missing {missing[:20]}"
        )
    for stage in STAGE_ORDER:
        count = sum(1 for row in rows if row["stage"] == stage)
        if count != len(sample_ids):
            raise RuntimeError(f"Stage {stage} has {count} rows for {len(sample_ids)} samples")

    shard_jsons = [
        path.with_name(path.name.replace("_per_sample.csv", ".json"))
        for path in csv_paths
    ]
    first_payload = json.loads(shard_jsons[0].read_text(encoding="utf-8")) if shard_jsons[0].exists() else {}
    payload = {
        "provenance": {
            **first_payload.get("provenance", {}),
            "merged_from": [str(path.relative_to(ROOT)) for path in csv_paths],
            "sample_indices": [int(v) for v in sample_ids],
            "num_samples": len(sample_ids),
        },
        "summary": summarize(rows),
    }

    out_json = EVAL_DIR / f"{args.out_tag}.json"
    out_csv = EVAL_DIR / f"{args.out_tag}_per_sample.csv"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(out_csv, rows)
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
