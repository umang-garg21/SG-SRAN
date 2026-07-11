#!/usr/bin/env python3
"""Boundary-threshold sweep for the Ti64 DIC McLean 4x4 comparison."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis"))

import evaluate_ti64_dic_mclean_fresh_4x4 as eval_ti64  # noqa: E402
import metric_panel_hardened as mph  # noqa: E402


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Boundary Threshold Sweep",
        "",
        "Boundary masks are recomputed with a symmetry-aware neighbor misorientation cutoff of 2, 5, or 10 degrees. "
        "Strict precision/recall/F1 use exact boundary-pixel overlap. Tolerant precision/recall/F1 allow +/-1 px dilation, "
        "matching the existing table convention.",
        "",
    ]
    ordered = [
        "threshold_deg",
        "method",
        "gt_boundary_fraction",
        "sr_boundary_fraction",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "boundary_precision_tol1",
        "boundary_recall_tol1",
        "boundary_f1_tol1",
    ]
    df = df[ordered]
    for col in df.columns:
        if col != "method":
            df[col] = pd.to_numeric(df[col], errors="coerce").round(6)
    records = df.astype(object).where(pd.notna(df), "").to_dict(orient="records")
    widths = {col: max(len(str(col)), *(len(str(row[col])) for row in records)) for col in df.columns}
    lines.append("| " + " | ".join(str(col).ljust(widths[col]) for col in df.columns) + " |")
    lines.append("| " + " | ".join("-" * widths[col] for col in df.columns) + " |")
    for row in records:
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in df.columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "experiments/Ti64_DIC_Mclean_fresh_4x4_full_Train/analysis/current_models_manifest.json",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    args = parser.parse_args()

    manifest = eval_ti64.load_manifest(args.manifest)
    out_dir = Path(manifest["analysis_out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    completed = [run for run in manifest["runs"] if eval_ti64.summary_path(run).exists()]
    ref_run = eval_ti64.choose_reference_run(completed)
    ref_dir = eval_ti64.sr_quat_dir(ref_run)
    ids = mph.sample_ids(ref_dir)
    ops = mph.conjugated_ops(mph.configure_symmetry(manifest.get("material_symmetry", "D6h")))

    records = []
    for sid in ids:
        lr = mph.load_quat(mph.lr_path(ref_dir, sid))
        hr = mph.load_quat(mph.hr_path(ref_dir, sid))
        out_hw = tuple(hr.shape[:2])
        hr_boundaries = {
            threshold: mph.boundary_mask_fast(hr, ops, threshold_deg=threshold)
            for threshold in args.thresholds
        }
        records.append({"sample_id": sid, "lr": lr, "hr": hr, "out_hw": out_hw, "hr_boundaries": hr_boundaries})

    methods: OrderedDict[str, Path | None] = OrderedDict(eval_ti64.CLASSICAL)
    for run in completed:
        methods[run["name"]] = eval_ti64.sr_quat_dir(run)

    rows: list[dict[str, Any]] = []
    for method, method_dir in methods.items():
        print(method, flush=True)
        per_threshold = {
            threshold: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "gt": 0,
                "sr": 0,
                "pixels": 0,
                "tol_precision": [],
                "tol_recall": [],
                "tol_f1": [],
            }
            for threshold in args.thresholds
        }
        for rec in records:
            sid = int(rec["sample_id"])
            sr = mph.method_field(method, method_dir, rec["lr"], rec["out_hw"], sid)
            for threshold in args.thresholds:
                gt_boundary = rec["hr_boundaries"][threshold]
                sr_boundary = mph.boundary_mask_fast(sr, ops, threshold_deg=threshold)
                bucket = per_threshold[threshold]
                tp = int((sr_boundary & gt_boundary).sum())
                fp = int((sr_boundary & ~gt_boundary).sum())
                fn = int((~sr_boundary & gt_boundary).sum())
                bucket["tp"] += tp
                bucket["fp"] += fp
                bucket["fn"] += fn
                bucket["gt"] += int(gt_boundary.sum())
                bucket["sr"] += int(sr_boundary.sum())
                bucket["pixels"] += int(gt_boundary.size)
                precision_t, recall_t, f1_t = mph.tolerant_boundary_f1(sr_boundary, gt_boundary, radius=1)
                bucket["tol_precision"].append(precision_t)
                bucket["tol_recall"].append(recall_t)
                bucket["tol_f1"].append(f1_t)

        for threshold, bucket in per_threshold.items():
            tp = bucket["tp"]
            fp = bucket["fp"]
            fn = bucket["fn"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            rows.append(
                {
                    "threshold_deg": threshold,
                    "method": method,
                    "n_samples": len(records),
                    "gt_boundary_pixels": bucket["gt"],
                    "sr_boundary_pixels": bucket["sr"],
                    "gt_boundary_fraction": bucket["gt"] / bucket["pixels"],
                    "sr_boundary_fraction": bucket["sr"] / bucket["pixels"],
                    "boundary_precision": precision,
                    "boundary_recall": recall,
                    "boundary_f1": f1,
                    "boundary_precision_tol1": sum(bucket["tol_precision"]) / len(bucket["tol_precision"]),
                    "boundary_recall_tol1": sum(bucket["tol_recall"]) / len(bucket["tol_recall"]),
                    "boundary_f1_tol1": sum(bucket["tol_f1"]) / len(bucket["tol_f1"]),
                }
            )

    csv_path = out_dir / "boundary_threshold_sweep.csv"
    json_path = out_dir / "boundary_threshold_sweep.json"
    md_path = out_dir / "boundary_threshold_sweep.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
