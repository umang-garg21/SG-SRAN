#!/usr/bin/env python3
"""Diagnose whether OCRP support windows are too large for Ti64 DIC McLean."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import metric_panel_hardened as mph  # noqa: E402


DATASETS = OrderedDict(
    [
        (
            "Ti64_DIC_Mclean",
            Path("/data/home/umang/Materials/Materials_data_mount/datasets/Ti64_DIC_Mclean_QSR_x4"),
        ),
        (
            "Ti_Al_1pct",
            Path("/data/home/umang/Materials/Materials_data_mount/datasets/Ti_Al_1pct_QSR_x4"),
        ),
    ]
)


def load_q(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 quaternion array at {path}, got {arr.shape}")
    if arr.shape[-1] == 4:
        q = arr
    elif arr.shape[0] == 4:
        q = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"No quaternion axis found at {path}: {arr.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(norm, 1e-12, None)
    return np.where(q[..., :1] < 0.0, -q, q).astype(np.float32, copy=False)


def sample_pairs(root: Path, split: str, limit: int | None = None) -> list[tuple[Path, Path]]:
    hr_files = sorted((root / split / "HR_Data").glob("*.npy"))
    lr_files = sorted((root / split / "LR_Data").glob("*.npy"))
    if len(hr_files) != len(lr_files):
        raise RuntimeError(f"{root}/{split}: HR/LR count mismatch {len(hr_files)} vs {len(lr_files)}")
    if limit is not None:
        hr_files = hr_files[:limit]
        lr_files = lr_files[:limit]
    return list(zip(hr_files, lr_files))


def grain_labels(field: np.ndarray, ops: np.ndarray, threshold_deg: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    h, w = field.shape[:2]
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    edges_i = []
    edges_j = []
    if w > 1:
        same = mph.misorientation_fast(field[:, :-1], field[:, 1:], ops) <= threshold_deg
        edges_i.append(idx[:, :-1][same])
        edges_j.append(idx[:, 1:][same])
    if h > 1:
        same = mph.misorientation_fast(field[:-1, :], field[1:, :], ops) <= threshold_deg
        edges_i.append(idx[:-1, :][same])
        edges_j.append(idx[1:, :][same])
    if edges_i and sum(edge.size for edge in edges_i):
        ii = np.concatenate(edges_i)
        jj = np.concatenate(edges_j)
        graph = coo_matrix((np.ones_like(ii, dtype=np.uint8), (ii, jj)), shape=(h * w, h * w))
    else:
        graph = coo_matrix((h * w, h * w), dtype=np.uint8)
    _, labels = connected_components(graph, directed=False)
    labels = labels.reshape(h, w)
    sizes = np.bincount(labels.reshape(-1))
    return labels, sizes


def window_unique_counts(labels: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0 or window < 1:
        raise ValueError(window)
    pad = window // 2
    padded = np.pad(labels, ((pad, pad), (pad, pad)), mode="edge")
    h, w = labels.shape
    counts = np.empty((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            counts[y, x] = len(np.unique(padded[y : y + window, x : x + window]))
    return counts


def boundary_fraction(field: np.ndarray, ops: np.ndarray) -> float:
    return float(mph.boundary_mask_fast(field, ops, threshold_deg=5.0).mean())


def summarize_array(arr: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p10": float(np.percentile(x, 10)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
    }


def diagnose_dataset(
    key: str,
    root: Path,
    splits: list[str],
    ops: np.ndarray,
    limit: int | None,
    windows: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        lr_sizes_all = []
        hr_sizes_all = []
        lr_boundary = []
        hr_boundary = []
        win_counts = {w: [] for w in windows}
        win_dominance = {w: [] for w in windows}
        n_samples = 0
        for hr_path, lr_path in sample_pairs(root, split, limit=limit):
            hr = load_q(hr_path)
            lr = load_q(lr_path)
            scale_y = hr.shape[0] / lr.shape[0]
            scale_x = hr.shape[1] / lr.shape[1]
            lr_lab, lr_sizes = grain_labels(lr, ops)
            _, hr_sizes = grain_labels(hr, ops)
            lr_sizes_all.extend(lr_sizes.tolist())
            hr_sizes_all.extend(hr_sizes.tolist())
            lr_boundary.append(boundary_fraction(lr, ops))
            hr_boundary.append(boundary_fraction(hr, ops))
            padded_cache: dict[int, np.ndarray] = {}
            for w in windows:
                counts = window_unique_counts(lr_lab, w)
                win_counts[w].extend(counts.reshape(-1).tolist())
                pad = w // 2
                padded = padded_cache.setdefault(w, np.pad(lr_lab, ((pad, pad), (pad, pad)), mode="edge"))
                for y in range(lr_lab.shape[0]):
                    for x in range(lr_lab.shape[1]):
                        patch = padded[y : y + w, x : x + w].reshape(-1)
                        _, c = np.unique(patch, return_counts=True)
                        win_dominance[w].append(float(c.max() / patch.size))
            n_samples += 1
        row: dict[str, Any] = {
            "dataset": key,
            "split": split,
            "n_samples": n_samples,
            "scale_y": scale_y,
            "scale_x": scale_x,
            "lr_window_current": 9,
            "lr_window_current_hr_footprint_px": 9 * float(scale_y),
        }
        row.update(summarize_array(lr_sizes_all, "lr_grain_area_px"))
        row.update(summarize_array(np.sqrt(np.asarray(lr_sizes_all) * 4.0 / np.pi), "lr_grain_eq_diam_px"))
        row.update(summarize_array(hr_sizes_all, "hr_grain_area_px"))
        row.update(summarize_array(np.sqrt(np.asarray(hr_sizes_all) * 4.0 / np.pi), "hr_grain_eq_diam_px"))
        row.update(summarize_array(lr_boundary, "lr_boundary_fraction"))
        row.update(summarize_array(hr_boundary, "hr_boundary_fraction"))
        for w in windows:
            row.update(summarize_array(win_counts[w], f"w{w}_unique_lr_grains"))
            row.update(summarize_array(win_dominance[w], f"w{w}_dominant_grain_fraction"))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["Train", "Val", "Test"])
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split sample limit for quick diagnostics.")
    parser.add_argument("--windows", nargs="+", type=int, default=[3, 5, 7, 9])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "out" / "ti64_dic_mclean_fresh_4x4" / "diagnostics",
    )
    args = parser.parse_args()

    ops = mph.conjugated_ops(mph.configure_symmetry("D6h"))
    rows: list[dict[str, Any]] = []
    for key, root in DATASETS.items():
        rows.extend(diagnose_dataset(key, root, args.splits, ops, args.limit, args.windows))

    csv_path = args.out_dir / "ocrp_window_scale_diagnostics.csv"
    json_path = args.out_dir / "ocrp_window_scale_diagnostics.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for row in rows:
        print(
            "{dataset}/{split}: LR grain eq diam median={lr_grain_eq_diam_px_median:.2f}px, "
            "LR boundary={lr_boundary_fraction_mean:.3f}, "
            "w9 unique median={w9_unique_lr_grains_median:.1f}, "
            "w9 dominant median={w9_dominant_grain_fraction_median:.2f}".format(**row)
        )


if __name__ == "__main__":
    main()
