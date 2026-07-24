#!/usr/bin/env python3
"""Aggregate direct-Reynolds-isometric five-seed OCRP test metrics."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np
import torch
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "Paper/EBSD_SR_Nature_v4/evals"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_new_learned_baselines as enb
import export_test_psnr_ssim_ipf as ipf_eval
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry


SEEDS = [42, 43, 44, 45, 46]
OUT_CSV = ROOT / "analysis/direct_reynolds_isometric_seed_sweep_metrics.csv"
OUT_JSON = ROOT / "analysis/direct_reynolds_isometric_seed_sweep_metrics.json"

MATERIALS = {
    "IN718": {
        "symmetry": "Oh",
        "direct_root": ROOT / "experiments/IN718/direct_reynolds_isometric_seed_runs",
        "direct_prefix": "ocrp_direct_reynolds_isometric_l4",
        "cart_root": ROOT / "experiments/IN718/seed_runs",
        "cart_prefix": "ocrp_4x4",
    },
    "Ti_Al_1pct": {
        "symmetry": "D6h",
        "direct_root": ROOT / "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs",
        "direct_prefix": "ocrp_direct_reynolds_isometric_l6",
        "cart_root": ROOT / "experiments/Ti_Al_1pct/seed_runs",
        "cart_prefix": "ocrp_4x4",
    },
}

METHODS = {
    "direct_reynolds_isometric": ("direct_root", "direct_prefix", "inference/test_best/summary.json"),
    "cartesian_tensor_decomp": ("cart_root", "cart_prefix", "inference/test/summary.json"),
}


def symmetry_ops(group: str, device: torch.device) -> torch.Tensor:
    ops = torch.as_tensor(
        proper_symmetry_quaternions(resolve_symmetry(group)),
        dtype=torch.float64,
        device=device,
    ).clone()
    ops[:, 1:] *= -1.0
    return ops


def summary_for(spec: dict, method: str, seed: int) -> Path:
    root_key, prefix_key, rel_summary = METHODS[method]
    return spec[root_key] / f"{spec[prefix_key]}_s{seed}" / rel_summary


def evaluate_summary(summary_path: Path, ops: torch.Tensor) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_mis: list[np.ndarray] = []
    interior: list[np.ndarray] = []
    boundary_band: list[np.ndarray] = []
    tp_total = fp_total = fn_total = nonfinite = 0

    for record in summary["records"]:
        sr_np = enb._load_record_array(record, "sr_npy")
        hr_np = enb._load_record_array(record, "hr_npy")
        sr = torch.from_numpy(np.asarray(sr_np, dtype=np.float32)).to(
            device=ops.device, dtype=torch.float64
        )
        hr = torch.from_numpy(np.asarray(hr_np, dtype=np.float32)).to(
            device=ops.device, dtype=torch.float64
        )
        mis = enb._misorientation_torch(sr, hr, ops).cpu().numpy().astype(np.float32)
        pred_boundary = enb._boundary_mask_torch(sr, ops).cpu().numpy()
        ref_boundary = enb._boundary_mask_torch(hr, ops).cpu().numpy()
        ref_band = binary_dilation(ref_boundary, iterations=5)

        tp_total += int(np.logical_and(pred_boundary, ref_boundary).sum())
        fp_total += int(np.logical_and(pred_boundary, np.logical_not(ref_boundary)).sum())
        fn_total += int(np.logical_and(np.logical_not(pred_boundary), ref_boundary).sum())

        finite = np.isfinite(mis)
        nonfinite += int((~finite).sum())
        all_mis.append(mis[finite])
        interior.append(mis[np.logical_and(~ref_band, finite)])
        boundary_band.append(mis[np.logical_and(ref_band, finite)])

    mis_all = np.concatenate(all_mis)
    mis_interior = np.concatenate(interior)
    mis_boundary_band = np.concatenate(boundary_band)
    denom = 2 * tp_total + fp_total + fn_total
    boundary_f1 = (2.0 * tp_total / denom) if denom else float("nan")

    return {
        "n_samples": int(summary["num_samples"]),
        "mean_deg": float(np.mean(mis_all)),
        "median_deg": float(np.median(mis_all)),
        "p90_deg": float(np.percentile(mis_all, 90)),
        "p95_deg": float(np.percentile(mis_all, 95)),
        "p99_deg": float(np.percentile(mis_all, 99)),
        "boundary_f1": float(boundary_f1),
        "interior_mean_deg": float(np.mean(mis_interior)),
        "boundary_band_mean_deg": float(np.mean(mis_boundary_band)),
        "frac_gt1_deg": float(np.mean(mis_all > 1.0)),
        "frac_gt2_deg": float(np.mean(mis_all > 2.0)),
        "frac_gt5_deg": float(np.mean(mis_all > 5.0)),
        "frac_gt10_deg": float(np.mean(mis_all > 10.0)),
        "nonfinite_pixels": int(nonfinite),
    }


def evaluate_ipf_summary(summary_path: Path, symmetry_name: str) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["task"] = summary.get("task", "test")
    ipf_eval.SYM = resolve_symmetry(symmetry_name)
    row, _ = ipf_eval.evaluate_method(
        summary,
        "OCRP",
        ipf_eval.provider_from_saved_sr,
    )
    return {
        "psnr_ipf_xyz_db": float(row["psnr_mean_xyz"]),
        "ssim_ipf_xyz": float(row["ssim_mean_xyz"]),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    metrics = [
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
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["material"], row["method"])].append(row)

    out = []
    for (material, method), group_rows in sorted(grouped.items()):
        agg = {
            "material": material,
            "method": method,
            "n_seeds": len(group_rows),
            "n_samples_per_seed": group_rows[0]["n_samples"] if group_rows else None,
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in group_rows], dtype=np.float64)
            agg[f"{metric}_mean"] = float(np.mean(values))
            agg[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(agg)
    return out


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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_seed_rows = []
    for material, spec in MATERIALS.items():
        ops = symmetry_ops(spec["symmetry"], device)
        for method in METHODS:
            for seed in SEEDS:
                summary = summary_for(spec, method, seed)
                if not summary.exists():
                    print(f"missing: {summary}")
                    continue
                metrics = evaluate_summary(summary, ops)
                metrics.update(evaluate_ipf_summary(summary, spec["symmetry"]))
                row = {
                    "material": material,
                    "method": method,
                    "seed": seed,
                    "summary": str(summary.relative_to(ROOT)),
                }
                row.update(metrics)
                per_seed_rows.append(row)
                print(
                    f"{material} {method} s{seed}: "
                    f"mean={row['mean_deg']:.4f} p95={row['p95_deg']:.4f} "
                    f"p99={row['p99_deg']:.4f} bf1={row['boundary_f1']:.4f} "
                    f"psnr={row['psnr_ipf_xyz_db']:.3f} ssim={row['ssim_ipf_xyz']:.4f}",
                    flush=True,
                )

    aggregate_rows = aggregate(per_seed_rows)
    write_csv(OUT_CSV, per_seed_rows + aggregate_rows)
    OUT_JSON.write_text(
        json.dumps({"per_seed": per_seed_rows, "aggregate": aggregate_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nAggregate mean +/- std over seeds")
    for row in aggregate_rows:
        print(
            f"{row['material']} {row['method']}: "
            f"mean={row['mean_deg_mean']:.4f}+/-{row['mean_deg_std']:.4f}, "
            f"p95={row['p95_deg_mean']:.4f}+/-{row['p95_deg_std']:.4f}, "
            f"p99={row['p99_deg_mean']:.4f}+/-{row['p99_deg_std']:.4f}, "
            f"bf1={row['boundary_f1_mean']:.4f}+/-{row['boundary_f1_std']:.4f}, "
            f"psnr={row['psnr_ipf_xyz_db_mean']:.4f}+/-{row['psnr_ipf_xyz_db_std']:.4f}, "
            f"ssim={row['ssim_ipf_xyz_mean']:.4f}+/-{row['ssim_ipf_xyz_std']:.4f}"
        )
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
