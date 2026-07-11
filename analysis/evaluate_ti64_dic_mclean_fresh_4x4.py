#!/usr/bin/env python3
"""Evaluate fresh Ti64 DIC McLean 4x4 test-set outputs."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

import metric_panel_hardened as mph


CLASSICAL = OrderedDict(
    [
        ("Nearest", None),
        ("Bicubic", None),
        ("SLERP", None),
        ("Symm-SLERP", None),
    ]
)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def sr_quat_dir(run: dict[str, Any]) -> Path:
    return Path(run["inference_dir"]) / "sr_quaternions"


def summary_path(run: dict[str, Any]) -> Path:
    return Path(run["summary"])


def choose_reference_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    for run in runs:
        if "OCRP" in run["name"] and summary_path(run).exists():
            return run
    for run in runs:
        if summary_path(run).exists():
            return run
    raise FileNotFoundError("No completed inference summary found.")


def write_markdown(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    df = pd.DataFrame(rows)
    lines = [f"# {title}", ""]
    if df.empty:
        lines.append("No rows were evaluated.")
        path.write_text("\n".join(lines) + "\n")
        return

    ordered = [
        "method",
        "n_samples",
        "n_valid_samples",
        "mean_deg",
        "median_deg",
        "p90_deg",
        "p95_deg",
        "p99_deg",
        "tol1_mean_deg",
        "interior_mean_deg",
        "boundary_band_mean_deg",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "boundary_precision_tol1",
        "boundary_recall_tol1",
        "boundary_f1_tol1",
        "grain_count_ratio",
        "grain_log10_size_wasserstein",
        "distance_to_nn_deg",
        "grain_ratio_abs_log10",
        "psnr_ipf_xyz_db",
        "ssim_ipf_xyz",
    ]
    present = [c for c in ordered if c in df.columns]
    df = df[present].copy()
    for col in df.columns:
        if col not in {"method", "n_samples"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(6)
    records = df.astype(object).where(pd.notna(df), "").to_dict(orient="records")
    widths = {
        col: max(len(str(col)), *(len(str(row[col])) for row in records))
        for col in df.columns
    }
    lines.append("| " + " | ".join(str(col).ljust(widths[col]) for col in df.columns) + " |")
    lines.append("| " + " | ".join("-" * widths[col] for col in df.columns) + " |")
    for row in records:
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in df.columns) + " |")
    lines.append("")
    lines.append("Split policy: existing dataset Test split.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--include-classical", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any manifest run lacks a test summary.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    out_dir = Path(manifest["analysis_out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = list(manifest["runs"])
    missing = [run["name"] for run in runs if not summary_path(run).exists()]
    if missing and args.strict:
        raise FileNotFoundError(f"Missing completed inference summaries: {missing}")
    completed = [run for run in runs if summary_path(run).exists()]
    if not completed:
        raise FileNotFoundError("No completed runs to evaluate.")

    ref_run = choose_reference_run(completed)
    ref_dir = sr_quat_dir(ref_run)
    ids = mph.sample_ids(ref_dir)
    ops = mph.conjugated_ops(mph.configure_symmetry(manifest.get("material_symmetry", "D6h")))
    records = mph.preload_dataset_records(ref_dir, ids, ops)

    methods: OrderedDict[str, Path | None] = OrderedDict()
    if args.include_classical:
        methods.update(CLASSICAL)
    for run in completed:
        methods[run["name"]] = sr_quat_dir(run)

    rows: list[dict[str, Any]] = []
    dataset_name = manifest["experiment"]
    dataset_label = "Ti64 DIC McLean fresh 4x4"
    symmetry_group = manifest.get("material_symmetry", "D6h")
    for method_name, sr_dir in methods.items():
        row = mph.summarize_method(dataset_name, dataset_label, symmetry_group, method_name, sr_dir, records, ops)
        row["split_policy"] = manifest["split_policy"]
        row["material_symmetry"] = symmetry_group
        rows.append(row)

    csv_path = out_dir / "metrics.csv"
    json_path = out_dir / "metrics.json"
    md_path = out_dir / "metrics.md"
    best_path = out_dir / "best_by_metric.csv"
    mph.write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    write_markdown(md_path, rows, title="Ti64 DIC McLean Fresh 4x4 Test Metrics")
    best = mph.best_by_dataset(rows)
    mph.write_csv(best_path, best)

    exp_analysis = Path(manifest["experiment_root"]) / "analysis"
    exp_analysis.mkdir(parents=True, exist_ok=True)
    for source in (csv_path, json_path, md_path, best_path):
        (exp_analysis / source.name).write_text(source.read_text())

    print(f"Evaluated {len(methods)} methods on {len(ids)} test samples.")
    print(f"Reference inference directory: {ref_dir}")
    print(f"Metrics CSV: {csv_path}")
    print(f"Metrics Markdown: {md_path}")


if __name__ == "__main__":
    main()
