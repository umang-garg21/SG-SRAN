#!/usr/bin/env python3
"""Evaluate refreshed direct-Reynolds-isometric IN718 diagnostic reruns."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import OrderedDict
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
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry


OUT_CSV = ROOT / "analysis/direct_reynolds_isometric_diagnostics_metrics.csv"
OUT_JSON = ROOT / "analysis/direct_reynolds_isometric_diagnostics_metrics.json"
DIAG_ROOT = ROOT / "experiments/IN718/direct_reynolds_isometric_diagnostics"

SUMMARIES = OrderedDict(
    [
        (
            "OCRP full direct seed42",
            ROOT
            / "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42/inference/test_best/summary.json",
        ),
        (
            "OCRP-no-routing bicubic direct",
            DIAG_ROOT
            / "ocrp_direct_reynolds_isometric_l4_no_routing_bicubic_s42/inference/test_best/summary.json",
        ),
        (
            "OCRP-no-routing nearest direct",
            DIAG_ROOT
            / "ocrp_direct_reynolds_isometric_l4_no_routing_nn_s42/inference/test_best/summary.json",
        ),
        (
            "OCRP-no-residual all direct",
            DIAG_ROOT
            / "ocrp_direct_reynolds_isometric_l4_no_residual_all_s42/inference/test_best/summary.json",
        ),
        (
            "HR residual w0.0 direct",
            DIAG_ROOT
            / "ocrp_direct_reynolds_isometric_l4_hr_residual_w0p0_s42/inference/test_best/summary.json",
        ),
        (
            "HR residual w0.1 direct",
            DIAG_ROOT
            / "ocrp_direct_reynolds_isometric_l4_hr_residual_w0p1_s42/inference/test_best/summary.json",
        ),
        (
            "HR residual w0.5 direct",
            DIAG_ROOT
            / "ocrp_direct_reynolds_isometric_l4_hr_residual_w0p5_s42/inference/test_best/summary.json",
        ),
        (
            "HR residual w1.0 direct",
            DIAG_ROOT
            / "ocrp_direct_reynolds_isometric_l4_hr_residual_w1p0_s42/inference/test_best/summary.json",
        ),
    ]
)


def symmetry_ops(device: torch.device) -> torch.Tensor:
    ops = torch.as_tensor(
        proper_symmetry_quaternions(resolve_symmetry("Oh")),
        dtype=torch.float64,
        device=device,
    ).clone()
    ops[:, 1:] *= -1.0
    return ops


def evaluate_summary(summary_path: Path, label: str, ops: torch.Tensor) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_mis: list[np.ndarray] = []
    interior: list[np.ndarray] = []
    boundary_band: list[np.ndarray] = []
    per_patch_mean: list[float] = []
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
        finite_mis = mis[finite]
        all_mis.append(finite_mis)
        per_patch_mean.append(float(np.mean(finite_mis)))
        interior.append(mis[np.logical_and(~ref_band, finite)])
        boundary_band.append(mis[np.logical_and(ref_band, finite)])

    mis_all = np.concatenate(all_mis)
    mis_interior = np.concatenate(interior)
    mis_boundary_band = np.concatenate(boundary_band)
    patch_means = np.asarray(per_patch_mean, dtype=np.float64)
    denom = 2 * tp_total + fp_total + fn_total
    boundary_f1 = (2.0 * tp_total / denom) if denom else float("nan")

    return {
        "label": label,
        "summary": str(summary_path.relative_to(ROOT)),
        "n_samples": int(summary["num_samples"]),
        "mean_deg": float(np.mean(mis_all)),
        "per_patch_mean_std_deg": float(np.std(patch_means, ddof=0)),
        "median_deg": float(np.median(mis_all)),
        "p90_deg": float(np.percentile(mis_all, 90)),
        "p95_deg": float(np.percentile(mis_all, 95)),
        "p99_deg": float(np.percentile(mis_all, 99)),
        "boundary_f1": float(boundary_f1),
        "boundary_tp": int(tp_total),
        "boundary_fp": int(fp_total),
        "boundary_fn": int(fn_total),
        "interior_mean_deg": float(np.mean(mis_interior)),
        "boundary_band_mean_deg": float(np.mean(mis_boundary_band)),
        "frac_gt1_deg": float(np.mean(mis_all > 1.0)),
        "frac_gt2_deg": float(np.mean(mis_all > 2.0)),
        "frac_gt5_deg": float(np.mean(mis_all > 5.0)),
        "frac_gt10_deg": float(np.mean(mis_all > 10.0)),
        "nonfinite_pixels": int(nonfinite),
    }


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
    ops = symmetry_ops(device)
    rows = []
    missing = []
    for label, summary_path in SUMMARIES.items():
        if not summary_path.exists():
            missing.append(str(summary_path.relative_to(ROOT)))
            continue
        row = evaluate_summary(summary_path, label, ops)
        rows.append(row)
        print(
            f"{label}: mean={row['mean_deg']:.3f} +/- {row['per_patch_mean_std_deg']:.3f}, "
            f"median={row['median_deg']:.3f}, p90={row['p90_deg']:.3f}, "
            f"bf1={row['boundary_f1']:.3f}, bnd={row['boundary_band_mean_deg']:.3f}",
            flush=True,
        )

    payload = {
        "protocol": {
            "material": "IN718",
            "split": "held-out isotropic 4x4 test split",
            "symmetry": "proper cubic Oh rotations",
            "boundary_threshold_deg": 5.0,
            "boundary_band_dilation_px": 5,
        },
        "rows": rows,
        "missing": missing,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(OUT_CSV, rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    if missing:
        print("Missing summaries:")
        for item in missing:
            print(f"  {item}")


if __name__ == "__main__":
    main()
