#!/usr/bin/env python3
"""GPU evaluator for boundary-window spurious orientation composition.

For each HR/GT boundary pixel, this metric compares the SR 3x3 window against
the orientations observed in the corresponding HR 3x3 window.  SR window
observations farther than the tolerance from every HR window orientation are
counted as spurious introduced orientations.  Overlapping GT boundary windows
are counted as separate observations by design.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PAPER_EVAL_DIR = (
    ROOT / "Paper" / "202607_Umang_EBSD_SR_paper__NMI" / "EBSD_SR_Nature_v4" / "evals"
)
for path in (ROOT, ROOT / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from metric_panel_hardened import (  # noqa: E402
    CLASSICAL_METHODS,
    DATASETS,
    NonFiniteQuaternionError,
    configure_symmetry,
    conjugated_ops,
    load_quat,
    method_field,
    rel,
)


DEFAULT_DATASETS = ("IN718_indist", "Ti_indist")
DEFAULT_PREFIX = "boundary_composition_prf_8nbr_5deg_20260708"
FORWARD_8N_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))
WINDOW_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def normalize_torch(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def qmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def best_sym_dot_pairwise_same_shape(
    pred: torch.Tensor,
    target: torch.Tensor,
    ops: torch.Tensor,
) -> torch.Tensor:
    """Return max_S |dot(pred_i, S target_i)| for matching pred/target rows."""
    pred_flat = pred.reshape(-1, 4)
    target_flat = target.reshape(-1, 4)
    sym_target = qmul_torch(ops[:, None, :], target_flat[None, :, :])
    dots = torch.abs((sym_target * pred_flat[None, :, :]).sum(dim=-1))
    return dots.max(dim=0).values.reshape(pred.shape[:-1])


def boundary_mask_8_neighbor(hr: torch.Tensor, ops: torch.Tensor, dot_threshold: float) -> torch.Tensor:
    """8-neighbour HR boundary mask, marking both endpoints of each high-angle edge."""
    height, width = hr.shape[:2]
    boundary = torch.zeros((height, width), dtype=torch.bool, device=hr.device)
    for dy, dx in FORWARD_8N_OFFSETS:
        y0_a = max(0, -dy)
        y1_a = min(height, height - dy)
        x0_a = max(0, -dx)
        x1_a = min(width, width - dx)
        y0_b = y0_a + dy
        y1_b = y1_a + dy
        x0_b = x0_a + dx
        x1_b = x1_a + dx
        if y0_a >= y1_a or x0_a >= x1_a:
            continue
        dots = best_sym_dot_pairwise_same_shape(
            hr[y0_a:y1_a, x0_a:x1_a],
            hr[y0_b:y1_b, x0_b:x1_b],
            ops,
        )
        hit = dots < dot_threshold
        boundary[y0_a:y1_a, x0_a:x1_a] |= hit
        boundary[y0_b:y1_b, x0_b:x1_b] |= hit
    return boundary


def gather_windows(field: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather clipped 3x3 windows for centers.

    Returns:
        windows: (K, 9, 4), invalid entries are zero.
        valid: (K, 9), true where the offset is inside the image.
    """
    height, width = field.shape[:2]
    device = field.device
    offsets = torch.tensor(WINDOW_OFFSETS, dtype=torch.long, device=device)
    yy = centers[:, 0:1] + offsets[None, :, 0]
    xx = centers[:, 1:2] + offsets[None, :, 1]
    valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    yy_clamped = yy.clamp(0, height - 1)
    xx_clamped = xx.clamp(0, width - 1)
    windows = field[yy_clamped, xx_clamped]
    windows = torch.where(valid[..., None], windows, torch.zeros_like(windows))
    return windows, valid


def count_spurious_for_sample(
    hr_np: np.ndarray,
    sr_np: np.ndarray,
    ops_np: np.ndarray,
    *,
    device: str,
    threshold_deg: float,
    chunk_centers: int,
) -> dict[str, int | float]:
    dev = torch.device(device)
    dot_threshold = float(np.cos(np.deg2rad(threshold_deg) / 2.0))
    ops = torch.as_tensor(ops_np, dtype=torch.float32, device=dev)
    hr = normalize_torch(torch.as_tensor(hr_np, dtype=torch.float32, device=dev))
    sr = normalize_torch(torch.as_tensor(sr_np, dtype=torch.float32, device=dev))
    if hr.shape != sr.shape:
        raise ValueError(f"SR/HR shape mismatch: {tuple(sr.shape)} != {tuple(hr.shape)}")

    boundary = boundary_mask_8_neighbor(hr, ops, dot_threshold)
    centers = boundary.nonzero(as_tuple=False)
    total_spurious = 0
    total_observed = 0
    total_gt_groups = 0
    total_recovered_groups = 0
    n_centers = int(centers.shape[0])

    for start in range(0, n_centers, chunk_centers):
        c = centers[start : start + chunk_centers]
        hr_win, hr_valid = gather_windows(hr, c)
        sr_win, sr_valid = gather_windows(sr, c)
        k = int(c.shape[0])
        if k == 0:
            continue

        # Compare every SR window observation to every HR observation from the
        # same original GT window, under all crystal symmetry operations.
        sym_hr = qmul_torch(ops[None, :, None, :], hr_win[:, None, :, :])
        dots = torch.abs(
            (
                sr_win[:, None, :, None, :]
                * sym_hr[:, :, None, :, :]
            ).sum(dim=-1)
        )
        dots = dots.masked_fill(~hr_valid[:, None, None, :], -1.0)
        best = dots.amax(dim=(1, 3))
        valid_match = best >= dot_threshold
        observed = sr_valid
        total_observed += int(observed.sum().item())
        total_spurious += int((observed & ~valid_match).sum().item())

        # Recall is computed over GT orientation groups in the HR window.  The
        # groups are connected components of the within-5-degree HR adjacency.
        hr_pair_dots = torch.abs(
            (
                hr_win[:, None, :, None, :]
                * sym_hr[:, :, None, :, :]
            ).sum(dim=-1)
        ).amax(dim=1)
        conn = (
            (hr_pair_dots >= dot_threshold)
            & hr_valid[:, :, None]
            & hr_valid[:, None, :]
        )
        for kk in range(len(WINDOW_OFFSETS)):
            conn = conn | (conn[:, :, kk : kk + 1] & conn[:, kk : kk + 1, :])

        reps = torch.zeros_like(hr_valid)
        for jj in range(len(WINDOW_OFFSETS)):
            if jj == 0:
                has_previous = torch.zeros(k, dtype=torch.bool, device=dev)
            else:
                has_previous = conn[:, jj, :jj].any(dim=-1)
            reps[:, jj] = hr_valid[:, jj] & ~has_previous

        sr_hr_match = (
            (dots.amax(dim=1) >= dot_threshold)
            & sr_valid[:, :, None]
            & hr_valid[:, None, :]
        )
        recovered_reps = (
            (sr_hr_match[:, None, :, :] & conn[:, :, None, :]).any(dim=(2, 3))
            & reps
        )
        total_gt_groups += int(reps.sum().item())
        total_recovered_groups += int(recovered_reps.sum().item())

    rate = float(total_spurious / total_observed) if total_observed else float("nan")
    precision = float(1.0 - rate) if total_observed else float("nan")
    recall = float(total_recovered_groups / total_gt_groups) if total_gt_groups else float("nan")
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else float("nan")
    )
    return {
        "gt_boundary_centers": n_centers,
        "boundary_window_observations": total_observed,
        "spurious_observations": total_spurious,
        "spurious_rate": rate,
        "composition_precision": precision,
        "gt_orientation_groups": total_gt_groups,
        "recovered_gt_orientation_groups": total_recovered_groups,
        "composition_recall": recall,
        "composition_f1": f1,
    }


def sample_ids(ref_dir: Path, limit_samples: int | None = None) -> list[int]:
    ids = sorted(int(path.name.split("_")[1]) for path in ref_dir.glob("sample_*_hr.npy"))
    if not ids:
        ids = sorted(int(path.name.split("_")[1]) for path in ref_dir.glob("sample_*_sr.npy"))
    if not ids:
        raise FileNotFoundError(f"No sample files found in {ref_dir}")
    return ids[:limit_samples] if limit_samples is not None else ids


def build_jobs(dataset_keys: list[str], method_filter: set[str] | None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        spec = DATASETS[dataset_key]
        methods: OrderedDict[str, Path | None] = OrderedDict((m, None) for m in CLASSICAL_METHODS)
        methods.update((name, rel(path)) for name, path in spec["methods"].items())
        methods["OCRP"] = rel(spec["ref"])
        for method, method_dir in methods.items():
            if method_filter is not None and method not in method_filter:
                continue
            if method_dir is not None and not method_dir.exists():
                print(f"skip {dataset_key}/{method}: missing {method_dir}", flush=True)
                continue
            jobs.append(
                {
                    "dataset": dataset_key,
                    "dataset_label": spec["label"],
                    "symmetry": spec["symmetry"],
                    "ref_dir": str(rel(spec["ref"])),
                    "method": method,
                    "method_dir": None if method_dir is None else str(method_dir),
                }
            )
    return jobs


def run_job(job: dict[str, Any], device: str, threshold_deg: float, chunk_centers: int, limit_samples: int | None) -> dict[str, Any]:
    ref_dir = Path(job["ref_dir"])
    method_dir = None if job["method_dir"] is None else Path(job["method_dir"])
    sym_quats = configure_symmetry(str(job["symmetry"]))
    ops_np = conjugated_ops(sym_quats).astype(np.float32)
    ids = sample_ids(ref_dir, limit_samples)

    row = dict(job)
    row.update(
        {
            "device": device,
            "threshold_deg": float(threshold_deg),
            "boundary_connectivity": 8,
            "window_radius_px": 1,
            "counting": "overlapping_gt_boundary_windows_count_separately",
            "n_samples": len(ids),
            "n_valid_samples": 0,
            "n_invalid_samples": 0,
            "invalid_sample_ids": "",
            "gt_boundary_centers": 0,
            "boundary_window_observations": 0,
            "spurious_observations": 0,
            "gt_orientation_groups": 0,
            "recovered_gt_orientation_groups": 0,
        }
    )
    per_sample = []
    invalid_ids = []

    for sid in ids:
        hr = load_quat(ref_dir / f"sample_{sid:06d}_hr.npy")
        lr = load_quat(ref_dir / f"sample_{sid:06d}_lr.npy")
        try:
            sr = method_field(job["method"], method_dir, lr, tuple(hr.shape[:2]), sid)
        except NonFiniteQuaternionError:
            invalid_ids.append(sid)
            continue
        stats = count_spurious_for_sample(
            hr,
            sr,
            ops_np,
            device=device,
            threshold_deg=threshold_deg,
            chunk_centers=chunk_centers,
        )
        row["n_valid_samples"] += 1
        row["gt_boundary_centers"] += int(stats["gt_boundary_centers"])
        row["boundary_window_observations"] += int(stats["boundary_window_observations"])
        row["spurious_observations"] += int(stats["spurious_observations"])
        row["gt_orientation_groups"] += int(stats["gt_orientation_groups"])
        row["recovered_gt_orientation_groups"] += int(stats["recovered_gt_orientation_groups"])
        per_sample.append(
            {
                "dataset": job["dataset"],
                "dataset_label": job["dataset_label"],
                "symmetry": job["symmetry"],
                "method": job["method"],
                "sample_id": sid,
                **stats,
            }
        )

    row["n_invalid_samples"] = len(invalid_ids)
    row["invalid_sample_ids"] = ",".join(f"{sid:06d}" for sid in invalid_ids)
    denom = int(row["boundary_window_observations"])
    row["spurious_rate"] = float(row["spurious_observations"] / denom) if denom else float("nan")
    row["composition_precision"] = float(1.0 - row["spurious_rate"]) if denom else float("nan")
    group_denom = int(row["gt_orientation_groups"])
    row["composition_recall"] = (
        float(row["recovered_gt_orientation_groups"] / group_denom) if group_denom else float("nan")
    )
    row["composition_f1"] = (
        float(
            2.0
            * row["composition_precision"]
            * row["composition_recall"]
            / (row["composition_precision"] + row["composition_recall"])
        )
        if np.isfinite(row["composition_precision"])
        and np.isfinite(row["composition_recall"])
        and row["composition_precision"] + row["composition_recall"]
        else float("nan")
    )
    return {"summary": row, "per_sample": per_sample}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Boundary Spurious Composition",
        "",
        "Protocol: HR boundary centers use an 8-neighbour, symmetry-aware 5 degree contrast rule. "
        "For each center, the clipped HR 3x3 window defines the allowed GT orientation composition; "
        "each SR pixel in the corresponding clipped 3x3 window is spurious if it is farther than "
        "5 degrees from every HR orientation in that original GT window. Recall is group-level: "
        "HR-window orientations are clustered by the transitive closure of the 5 degree adjacency, "
        "and a GT group is recovered if any SR-window pixel matches any member of that group. "
        "Overlapping GT boundary windows are counted as separate observations.",
        "",
    ]
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        if not subset:
            continue
        subset = sorted(subset, key=lambda r: (-r["composition_f1"], r["spurious_rate"]))
        lines.append(f"## {subset[0]['dataset_label']} ({subset[0]['symmetry']})")
        lines.append("")
        lines.append("| method | valid | obs. | spurious | FP rate | precision | GT groups | recovered | recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in subset:
            lines.append(
                "| {method} | {n_valid_samples:d} | {boundary_window_observations:d} | "
                "{spurious_observations:d} | {spurious_rate:.6f} | "
                "{composition_precision:.6f} | {gt_orientation_groups:d} | "
                "{recovered_gt_orientation_groups:d} | {composition_recall:.6f} | "
                "{composition_f1:.6f} |".format(**row)
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_gpus(gpus_arg: str) -> list[int]:
    if gpus_arg == "auto":
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    return [int(item) for item in gpus_arg.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS), choices=list(DATASETS))
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--threshold-deg", type=float, default=5.0)
    parser.add_argument("--chunk-centers", type=int, default=8192)
    parser.add_argument("--gpus", default="auto", help="'auto', comma-separated CUDA ids, or 'cpu'")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=PAPER_EVAL_DIR)
    parser.add_argument("--out-prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    method_filter = set(args.methods) if args.methods else None
    jobs = build_jobs(list(args.datasets), method_filter)
    if not jobs:
        raise RuntimeError("No jobs to run.")

    if args.gpus == "cpu":
        devices = ["cpu"]
    else:
        gpu_ids = parse_gpus(args.gpus)
        if not gpu_ids:
            raise RuntimeError("CUDA is not available. Re-run with GPU access or pass --gpus cpu for a CPU smoke test.")
        devices = [f"cuda:{gid}" for gid in gpu_ids]

    workers = min(len(jobs), len(devices) if args.max_workers is None else args.max_workers)
    worker_devices = [devices[i % len(devices)] for i in range(len(jobs))]
    print(
        f"Running {len(jobs)} jobs with {workers} worker(s) on devices {devices}; "
        f"threshold={args.threshold_deg} deg",
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    if workers == 1:
        for job, device in zip(jobs, worker_devices):
            result = run_job(job, device, args.threshold_deg, args.chunk_centers, args.limit_samples)
            summaries.append(result["summary"])
            per_sample.extend(result["per_sample"])
            print(
                f"done {job['dataset']}/{job['method']}: "
                f"rate={result['summary']['spurious_rate']:.6f}",
                flush=True,
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    run_job,
                    job,
                    device,
                    args.threshold_deg,
                    args.chunk_centers,
                    args.limit_samples,
                ): job
                for job, device in zip(jobs, worker_devices)
            }
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                summaries.append(result["summary"])
                per_sample.extend(result["per_sample"])
                print(
                    f"done {job['dataset']}/{job['method']}: "
                    f"rate={result['summary']['spurious_rate']:.6f}",
                    flush=True,
                )

    summaries.sort(key=lambda r: (r["dataset"], -r["composition_f1"], r["spurious_rate"], r["method"]))
    per_sample.sort(key=lambda r: (r["dataset"], r["method"], r["sample_id"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.out_prefix}.csv"
    sample_csv_path = args.out_dir / f"{args.out_prefix}_per_sample.csv"
    json_path = args.out_dir / f"{args.out_prefix}.json"
    md_path = args.out_dir / f"{args.out_prefix}.md"
    write_csv(csv_path, summaries)
    write_csv(sample_csv_path, per_sample)
    write_markdown(md_path, summaries)
    json_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "protocol": {
                    "boundary_connectivity": 8,
                    "boundary_threshold_deg": args.threshold_deg,
                    "match_threshold_deg": args.threshold_deg,
                    "window_radius_px": 1,
                    "edge_wrapping": "none",
                    "counting": "overlapping GT boundary windows count separately",
                    "spurious_rate_higher_is_worse": True,
                    "precision": "1 - spurious_rate",
                    "recall": "GT orientation groups recovered / GT orientation groups",
                    "f1": "harmonic mean of composition precision and group-level recall",
                },
                "rows": summaries,
                "per_sample": per_sample,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {sample_csv_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
