#!/usr/bin/env python3
"""Focused metric audit for the Open718 4x4 all-method comparison notebook.

This script executes the reusable setup/method cells from
analysis/patch_sample_comparison_all_methods_4x4.ipynb, runs the registered
methods, and writes independent metric checks to a structured audit folder.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation


REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "analysis" / "patch_sample_comparison_all_methods_4x4.ipynb"


def _exec_notebook_cells(cell_ids: list[int]) -> dict:
    nb = json.loads(NB_PATH.read_text())
    ns: dict = {"__name__": "__main__"}
    os.chdir(REPO_ROOT)
    for i in cell_ids:
        print(f"\n===== EXEC notebook cell {i} =====", flush=True)
        t0 = time.time()
        src = "".join(nb["cells"][i].get("source", []))
        try:
            exec(compile(src, f"{NB_PATH}:cell{i}", "exec"), ns)
        except Exception:
            print(f"ERROR in notebook cell {i}", flush=True)
            traceback.print_exc()
            raise
        print(f"===== DONE cell {i} in {time.time() - t0:.1f}s =====", flush=True)
    return ns


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q.reshape(-1, 4), axis=1).reshape(q.shape[:-1] + (1,)) + eps)


def _conj(q: np.ndarray) -> np.ndarray:
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0])


def _mis_passive_left(q_pred_passive: np.ndarray, q_gt_passive: np.ndarray, sym_ops: np.ndarray) -> np.ndarray:
    """Passive convention: crystal symmetry acts by left multiplication."""
    qp = _normalize(q_pred_passive).reshape(-1, 4)
    qg = _normalize(q_gt_passive).reshape(-1, 4)
    ops_inv = np.asarray(sym_ops, dtype=np.float64).copy()
    ops_inv[:, 1:] *= -1.0
    variants = _quat_mul(ops_inv[:, None, :], qg[None, :, :])  # (G,N,4)
    dots = np.abs(np.sum(variants * qp[None, :, :], axis=-1))
    ang = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    return np.degrees(np.min(ang, axis=0)).reshape(q_pred_passive.shape[:-1])


def _mis_active_right_from_passive(q_pred_passive: np.ndarray, q_gt_passive: np.ndarray, sym_ops: np.ndarray) -> np.ndarray:
    """Same physical metric after passive->active conjugation: symmetry acts on the right."""
    qp = _normalize(_conj(q_pred_passive)).reshape(-1, 4)
    qg = _normalize(_conj(q_gt_passive)).reshape(-1, 4)
    ops = np.asarray(sym_ops, dtype=np.float64)
    variants = _quat_mul(qg[:, None, :], ops[None, :, :])  # (N,G,4)
    dots = np.abs(np.sum(variants * qp[:, None, :], axis=-1))
    ang = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    return np.degrees(np.min(ang, axis=1)).reshape(q_pred_passive.shape[:-1])


def _mis_wrong_passive_right(q_pred_passive: np.ndarray, q_gt_passive: np.ndarray, sym_ops: np.ndarray) -> np.ndarray:
    """Diagnostic only: wrong side for passive arrays."""
    qp = _normalize(q_pred_passive).reshape(-1, 4)
    qg = _normalize(q_gt_passive).reshape(-1, 4)
    ops = np.asarray(sym_ops, dtype=np.float64)
    variants = _quat_mul(qg[:, None, :], ops[None, :, :])
    dots = np.abs(np.sum(variants * qp[:, None, :], axis=-1))
    ang = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    return np.degrees(np.min(ang, axis=1)).reshape(q_pred_passive.shape[:-1])


def _stats(prefix: str, arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        f"{prefix} mean": float(np.nanmean(arr)),
        f"{prefix} median": float(np.nanmedian(arr)),
        f"{prefix} p90": float(np.nanpercentile(arr, 90)),
        f"{prefix} p95": float(np.nanpercentile(arr, 95)),
        f"{prefix} p99": float(np.nanpercentile(arr, 99)),
        f"{prefix} max": float(np.nanmax(arr)),
        f"{prefix} frac>5deg": float(np.nanmean(arr > 5.0)),
        f"{prefix} frac>10deg": float(np.nanmean(arr > 10.0)),
        f"{prefix} frac>15deg": float(np.nanmean(arr > 15.0)),
        f"{prefix} frac>30deg": float(np.nanmean(arr > 30.0)),
        f"{prefix} frac>45deg": float(np.nanmean(arr > 45.0)),
    }


def _save_error_png(path: Path, err: np.ndarray, vmax: float = 60.0) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=180)
    im = ax.imshow(err, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_axis_off()
    ax.set_title(path.stem.replace("_", " "))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="misorientation error (°)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ns = _exec_notebook_cells([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    methods: OrderedDict = ns["METHODS"]
    run_all_methods = ns["run_all_methods"]
    lr = ns["lr"]
    hr = ns["hr"]
    sym = ns["sym"]
    sample_id = ns["sample_id"]
    patch_rows = ns["PATCH_ROWS"]
    patch_cols = ns["PATCH_COLS"]
    fig_root = Path(ns["FIG_ROOT"])
    audit_dir = fig_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning registered methods for metric audit:", list(methods.keys()), flush=True)
    results = {}
    for method_name in methods:
        partial = run_all_methods(
            lr,
            hr,
            methods=OrderedDict([(method_name, methods[method_name])]),
        )
        results.update(partial)

    sym_ops = np.asarray(sym.data, dtype=np.float64)
    hr_patch = hr[patch_rows, patch_cols]
    boundary_hr_full = ns["compute_boundary_mask"](hr, sym_ops=sym, threshold_deg=5.0, connectivity=4)
    boundary_hr_patch = boundary_hr_full[patch_rows, patch_cols]
    boundary_dilated_patch = binary_dilation(boundary_hr_patch, iterations=2)

    rows = []
    tail_rows = []
    for method_name in [name for name in methods if name in results]:
        sr_full = results[method_name]["sr_full"]
        sr_patch = results[method_name]["sr_patch"]
        norm_full = np.linalg.norm(sr_full.reshape(-1, 4), axis=1)
        existing_patch = results[method_name]["mis_patch"]
        independent_patch = _mis_passive_left(sr_patch, hr_patch, sym_ops)
        active_right_patch = _mis_active_right_from_passive(sr_patch, hr_patch, sym_ops)
        wrong_side_patch = _mis_wrong_passive_right(sr_patch, hr_patch, sym_ops)
        full_passive_left = _mis_passive_left(sr_full, hr, sym_ops)

        row = {
            "Method": method_name,
            "finite output": bool(np.isfinite(sr_full).all()),
            "quat norm mean": float(np.mean(norm_full)),
            "quat norm max_abs_err_from_1": float(np.max(np.abs(norm_full - 1.0))),
            "patch existing-vs-independent max_abs_diff_deg": float(np.nanmax(np.abs(existing_patch - independent_patch))),
            "patch passive-left-vs-active-right max_abs_diff_deg": float(np.nanmax(np.abs(independent_patch - active_right_patch))),
            "patch passive-left-vs-wrong-passive-right mean_abs_diff_deg": float(np.nanmean(np.abs(independent_patch - wrong_side_patch))),
        }
        row.update(_stats("patch mis", independent_patch))
        row.update(_stats("full mis", full_passive_left))
        rows.append(row)

        for threshold in (5.0, 10.0, 15.0, 30.0):
            high = independent_patch > threshold
            high_count = int(np.count_nonzero(high))
            tail_rows.append(
                {
                    "Method": method_name,
                    "threshold_deg": threshold,
                    "high_error_pixels": high_count,
                    "high_error_fraction": float(np.mean(high)),
                    "fraction_high_error_on_HR_boundary": float(np.mean(boundary_hr_patch[high])) if high_count else np.nan,
                    "fraction_high_error_within_2px_boundary": float(np.mean(boundary_dilated_patch[high])) if high_count else np.nan,
                    "mean_error_on_HR_boundary": float(np.nanmean(independent_patch[boundary_hr_patch])) if np.any(boundary_hr_patch) else np.nan,
                    "mean_error_off_2px_boundary": float(np.nanmean(independent_patch[~boundary_dilated_patch])) if np.any(~boundary_dilated_patch) else np.nan,
                }
            )

        slug = ns["slugify_label"](method_name)
        _save_error_png(audit_dir / f"{slug}_patch_misorientation_error.png", independent_patch)
        _save_error_png(audit_dir / f"{slug}_full_misorientation_error.png", full_passive_left)

    df = pd.DataFrame(rows).set_index("Method")
    tail_df = pd.DataFrame(tail_rows)
    summary_csv = audit_dir / "metric_audit_summary.csv"
    tail_csv = audit_dir / "tail_error_boundary_audit.csv"
    df.to_csv(summary_csv)
    tail_df.to_csv(tail_csv, index=False)

    print("\nSaved audit summary:", summary_csv)
    print("Saved tail/boundary audit:", tail_csv)
    print("\nKey audit columns:")
    key_cols = [
        "patch existing-vs-independent max_abs_diff_deg",
        "patch passive-left-vs-active-right max_abs_diff_deg",
        "patch passive-left-vs-wrong-passive-right mean_abs_diff_deg",
        "patch mis mean",
        "patch mis p90",
        "patch mis p95",
        "patch mis p99",
        "full mis mean",
        "full mis p90",
        "full mis p95",
        "full mis p99",
    ]
    print(df[key_cols].sort_values("patch mis mean").to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\nAudit image directory: {audit_dir}")


if __name__ == "__main__":
    main()
