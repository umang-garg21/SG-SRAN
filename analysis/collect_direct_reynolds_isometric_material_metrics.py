#!/usr/bin/env python3
"""Collect direct-Reynolds-isometric OCRP metrics for one material seed sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_direct_reynolds_isometric_seed_sweep as seed_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--material",
        required=True,
        choices=sorted(seed_eval.MATERIALS),
        help="Material key from evaluate_direct_reynolds_isometric_seed_sweep.py.",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        type=Path,
        help="Output prefix; .csv and .json are appended.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=seed_eval.SEEDS,
        help="Seed list to aggregate.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
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


def summarize(rows: list[dict]) -> dict:
    metric_keys = [
        "mean_deg",
        "median_deg",
        "p90_deg",
        "p95_deg",
        "p99_deg",
        "boundary_f1",
        "interior_mean_deg",
        "boundary_band_mean_deg",
        "frac_gt1_deg",
        "frac_gt2_deg",
        "frac_gt5_deg",
        "frac_gt10_deg",
        "psnr_ipf_xyz_db",
        "ssim_ipf_xyz",
    ]
    out: dict[str, float | int | str] = {
        "material": rows[0]["material"],
        "method": rows[0]["method"],
        "n_seeds": len(rows),
        "n_samples_per_seed": rows[0]["n_samples"],
    }
    for key in metric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def main() -> None:
    args = parse_args()
    spec = seed_eval.MATERIALS[args.material]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ops = seed_eval.symmetry_ops(spec["symmetry"], device)

    rows: list[dict] = []
    for seed in args.seeds:
        summary_path = seed_eval.summary_for(spec, "direct_reynolds_isometric", seed)
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing inference summary for seed {seed}: {summary_path}")
        metrics = seed_eval.evaluate_summary(summary_path, ops)
        metrics.update(seed_eval.evaluate_ipf_summary(summary_path, spec["symmetry"]))
        row = {
            "material": args.material,
            "method": "direct_reynolds_isometric",
            "seed": seed,
            "summary": str(summary_path.relative_to(seed_eval.ROOT)),
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"{args.material} s{seed}: "
            f"mean={row['mean_deg']:.4f} p95={row['p95_deg']:.4f} "
            f"p99={row['p99_deg']:.4f} bf1={row['boundary_f1']:.4f} "
            f"psnr={row['psnr_ipf_xyz_db']:.3f} ssim={row['ssim_ipf_xyz']:.4f}",
            flush=True,
        )

    aggregate = summarize(rows)
    print(
        f"{args.material} aggregate: "
        f"mean={aggregate['mean_deg_mean']:.4f}+/-{aggregate['mean_deg_std']:.4f}, "
        f"p95={aggregate['p95_deg_mean']:.4f}+/-{aggregate['p95_deg_std']:.4f}, "
        f"p99={aggregate['p99_deg_mean']:.4f}+/-{aggregate['p99_deg_std']:.4f}, "
        f"bf1={aggregate['boundary_f1_mean']:.4f}+/-{aggregate['boundary_f1_std']:.4f}, "
        f"psnr={aggregate['psnr_ipf_xyz_db_mean']:.4f}+/-{aggregate['psnr_ipf_xyz_db_std']:.4f}, "
        f"ssim={aggregate['ssim_ipf_xyz_mean']:.4f}+/-{aggregate['ssim_ipf_xyz_std']:.4f}",
        flush=True,
    )

    csv_path = args.out_prefix.with_suffix(".csv")
    json_path = args.out_prefix.with_suffix(".json")
    write_csv(csv_path, rows + [aggregate])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"per_seed": rows, "aggregate": aggregate}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
