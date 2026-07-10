#!/usr/bin/env python3
"""Aggregate per-seed few-shot 4x4 metric JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis/out"

DEFAULT_METRICS = [
    "mean_deg",
    "median_deg",
    "p90_deg",
    "p95_deg",
    "p99_deg",
    "tol1_mean_deg",
    "interior_mean_deg",
    "boundary_band_mean_deg",
    "boundary_f1",
    "boundary_f1_tol1",
    "grain_count_ratio",
    "grain_log10_size_wasserstein",
    "distance_to_nn_deg",
    "psnr_ipf_xyz_db",
    "ssim_ipf_xyz",
]

HIGHER_IS_BETTER = {"boundary_f1", "boundary_f1_tol1", "psnr_ipf_xyz_db", "ssim_ipf_xyz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--prefix-template",
        default="fewshot_4x4_hardened_metrics_s{seed}.json",
        help="Metric JSON template relative to analysis/out.",
    )
    parser.add_argument(
        "--out-prefix",
        default="fewshot_4x4_hardened_metrics_5seed",
        help="Output prefix in analysis/out.",
    )
    return parser.parse_args()


def _seed_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _metric_path(template: str, seed: int) -> Path:
    path = Path(template.format(seed=seed))
    return path if path.is_absolute() else OUT_DIR / path


def _seed_from_payload(payload: dict[str, Any], fallback: int) -> int:
    value = payload.get("protocol", {}).get("seed", fallback)
    if isinstance(value, list):
        raise ValueError(f"Expected per-seed metric file, got seed list {value}")
    return int(value)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) else None
    return None


def load_rows(seeds: list[int], template: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for seed in seeds:
        path = _metric_path(template, seed)
        if not path.exists():
            missing.append({"seed": seed, "path": str(path), "reason": "missing metrics"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual_seed = _seed_from_payload(payload, seed)
        seed_rows = list(payload.get("rows", []))
        if not seed_rows:
            missing.append({"seed": seed, "path": str(path), "reason": "no rows"})
            continue
        for row in seed_rows:
            out = dict(row)
            out["seed"] = actual_seed
            out["metric_file"] = str(path.relative_to(ROOT))
            rows.append(out)
    return rows, missing


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, method), group in sorted(groups.items()):
        seeds = sorted({int(row["seed"]) for row in group})
        first = group[0]
        out: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": first.get("dataset_label", ""),
            "symmetry": first.get("symmetry", ""),
            "method": method,
            "n_seeds": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric in DEFAULT_METRICS:
            vals = [
                val
                for val in (_finite_float(row.get(metric)) for row in group)
                if val is not None
            ]
            if not vals:
                continue
            out[f"{metric}_mean"] = mean(vals)
            out[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def add_winners(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in summary_rows})
    for dataset in datasets:
        subset = [row for row in summary_rows if str(row["dataset"]) == dataset]
        for metric in DEFAULT_METRICS:
            key = f"{metric}_mean"
            candidates = [row for row in subset if key in row]
            if not candidates:
                continue
            reverse = metric in HIGHER_IS_BETTER
            best = sorted(candidates, key=lambda row: float(row[key]), reverse=reverse)[0]
            best[f"{metric}_winner"] = True
            winners.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "method": best["method"],
                    "value": best[key],
                    "higher_is_better": reverse,
                }
            )
    return winners


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    val = _finite_float(value)
    return "--" if val is None else f"{val:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    lines = [
        "# Few-shot 4x4 multi-seed summary",
        "",
        "| Dataset | Method | Seeds | Mean deg | p90 deg | Boundary F1 | Boundary band deg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {n_seeds} | {mean} +/- {mean_std} | "
            "{p90} +/- {p90_std} | {f1} +/- {f1_std} | {band} +/- {band_std} |".format(
                dataset=row["dataset"],
                method=row["method"],
                n_seeds=row["n_seeds"],
                mean=fmt(row.get("mean_deg_mean")),
                mean_std=fmt(row.get("mean_deg_std")),
                p90=fmt(row.get("p90_deg_mean")),
                p90_std=fmt(row.get("p90_deg_std")),
                f1=fmt(row.get("boundary_f1_tol1_mean")),
                f1_std=fmt(row.get("boundary_f1_tol1_std")),
                band=fmt(row.get("boundary_band_mean_deg_mean")),
                band_std=fmt(row.get("boundary_band_mean_deg_std")),
            )
        )
    if missing:
        lines.extend(["", "## Missing or incomplete inputs", ""])
        for item in missing:
            lines.append(f"- seed {item['seed']}: {item['reason']} ({item['path']})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = _seed_list(args.seeds)
    rows, missing = load_rows(seeds, args.prefix_template)
    summary_rows = aggregate(rows)
    winners = add_winners(summary_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{args.out_prefix}_summary.csv"
    json_path = OUT_DIR / f"{args.out_prefix}_summary.json"
    md_path = OUT_DIR / f"{args.out_prefix}_summary.md"
    write_csv(csv_path, summary_rows)
    json_path.write_text(
        json.dumps(
            {
                "seeds_requested": seeds,
                "n_seed_rows": len(rows),
                "summary_rows": summary_rows,
                "winners": winners,
                "missing": missing,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(md_path, summary_rows, missing)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if missing:
        print(f"Missing/incomplete metric files: {len(missing)}")


if __name__ == "__main__":
    main()
