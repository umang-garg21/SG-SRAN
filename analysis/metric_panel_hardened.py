#!/usr/bin/env python3
"""Hardened topology/registration diagnostics for 4x4 orientation SR.

This script is intentionally separate from the quick metric_panel_*.py scratch
scripts.  It avoids wraparound shifts, records source directories, and reuses
the paper-local symmetry-aware misorientation and classical upsamplers.
"""
from __future__ import annotations

import argparse
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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "Paper" / "EBSD_SR_Nature_v4" / "evals"
OUT_DIR = ROOT / "analysis" / "out"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_anchorless_test_metrics as ev  # noqa: E402
import export_test_psnr_ssim_ipf as ipf_eval  # noqa: E402
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry  # noqa: E402


DATASETS = OrderedDict(
    [
        (
            "IN718_indist",
            {
                "label": "IN718 in-dist",
                "symmetry": "Oh",
                "ref": "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42/inference/test_best/sr_quaternions",
                "methods": OrderedDict(
                    [
                        ("Atindama", "experiments/IN718/atindama_inpainting_4x4_01/inference/test/sr_quaternions"),
                        ("Q-RBSA", "experiments/IN718/qrbsa_4x4_300ep_01/inference/test/sr_quaternions"),
                        ("QEDSR", "experiments/IN718/qedsr_4x4_01/inference/test/sr_quaternions"),
                        ("EDSR", "experiments/IN718/edsr_4x4_01/inference/test/sr_quaternions"),
                        ("RCAN", "experiments/IN718/rcan_4x4_300ep_01/inference/test/sr_quaternions"),
                        ("SAN", "experiments/IN718/san_4x4_300ep_01/inference/test/sr_quaternions"),
                        ("HAN", "experiments/IN718/han_4x4_300ep_01/inference/test/sr_quaternions"),
                    ]
                ),
            },
        ),
        (
            "Ti_indist",
            {
                "label": "Ti-6Al-4V in-dist",
                "symmetry": "D6h",
                "ref": "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions",
                "methods": OrderedDict(
                    [
                        ("Atindama", "experiments/Ti_Al_1pct/atindama_inpainting_4x4_01/inference/test/sr_quaternions"),
                        ("Q-RBSA", "experiments/Ti_Al_1pct/qrbsa_adapted_4x4_300ep_01/inference/test/sr_quaternions"),
                        ("QEDSR", "experiments/Ti_Al_1pct/qedsr_4x4_01/inference/test/sr_quaternions"),
                        ("EDSR", "experiments/Ti_Al_1pct/edsr_4x4_01/inference/test/sr_quaternions"),
                        ("RCAN", "experiments/Ti_Al_1pct/rcan_4x4_300ep_01/inference/test/sr_quaternions"),
                        ("SAN", "experiments/Ti_Al_1pct/san_4x4_300ep_01/inference/test/sr_quaternions"),
                        ("HAN", "experiments/Ti_Al_1pct/han_4x4_300ep_01/inference/test/sr_quaternions"),
                    ]
                ),
            },
        ),
        (
            "CoNi_zeroshot",
            {
                "label": "CoNi zero-shot",
                "symmetry": "Oh",
                "ref": "experiments/Zero_shot_performance_CoNi_x250/ocrp_direct_reynolds_isometric_l4_s42/inference/train_best/sr_quaternions",
                "methods": OrderedDict(
                    [
                        ("Atindama", "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/atindama_inpainting/sr_quaternions"),
                        ("Q-RBSA", "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/qrbsaadapted/sr_quaternions"),
                        ("QEDSR", "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/qedsr/sr_quaternions"),
                        ("RCAN", "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/rcan/sr_quaternions"),
                        ("SAN", "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/san/sr_quaternions"),
                        ("HAN", "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/han/sr_quaternions"),
                    ]
                ),
            },
        ),
        (
            "Ti7_zeroshot",
            {
                "label": "Ti7 zero-shot",
                "symmetry": "D6h",
                "ref": "experiments/Zero_shot_performance_Ti7_deformed/ocrp_direct_reynolds_isometric_l6_s42/inference/train_best/sr_quaternions",
                "methods": OrderedDict(
                    [
                        ("Atindama", "experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/atindama_inpainting/sr_quaternions"),
                        ("Q-RBSA", "experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/qrbsaadapted/sr_quaternions"),
                        ("QEDSR", "experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/qedsr/sr_quaternions"),
                        ("RCAN", "experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/rcan/sr_quaternions"),
                        ("SAN", "experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/san/sr_quaternions"),
                        ("HAN", "experiments/Zero_shot_performance_Ti7_deformed/learned_baselines_4x4/han/sr_quaternions"),
                    ]
                ),
            },
        ),
        (
            "Ti64_zeroshot",
            {
                "label": "Ti64 zero-shot",
                "symmetry": "D6h",
                "ref": "experiments/Zero_shot_performance_Ti64_DIC_Mclean/ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions",
                "methods": OrderedDict(
                    [
                        ("Atindama", "experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/atindama_inpainting/sr_quaternions"),
                        ("Q-RBSA", "experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/qrbsaadapted/sr_quaternions"),
                        ("QEDSR", "experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/qedsr/sr_quaternions"),
                        ("RCAN", "experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/rcan/sr_quaternions"),
                        ("SAN", "experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/san/sr_quaternions"),
                        ("HAN", "experiments/Zero_shot_performance_Ti64_DIC_Mclean/learned_baselines_4x4/han/sr_quaternions"),
                    ]
                ),
            },
        ),
    ]
)

CLASSICAL_METHODS = ("Nearest", "Bicubic", "SLERP", "Symm-SLERP")


class NonFiniteQuaternionError(ValueError):
    def __init__(self, path: Path, n_bad: int):
        self.path = path
        self.n_bad = n_bad
        super().__init__(f"Non-finite quaternion entries in {path}: {n_bad}")


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def configure_symmetry(group: str) -> np.ndarray:
    sym = resolve_symmetry(group)
    sym_quats = proper_symmetry_quaternions(sym)
    ev.SYM = sym
    ipf_eval.SYM = sym
    ev.SYM_QUATS = sym_quats
    ev._SLERP_SYM_OPS_4X4 = ev.make_symmetry_4x4(group, device="cpu", dtype=torch.float32)
    return sym_quats


def conjugated_ops(sym_quats: np.ndarray) -> np.ndarray:
    ops = np.asarray(sym_quats, dtype=np.float32).copy()
    ops[:, 1:] *= -1.0
    return ops


def misorientation_fast(pred: np.ndarray, target: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Fast symmetry-aware misorientation for already-normalized quaternion fields."""
    pred_shape = pred.shape[:-1]
    p = np.asarray(pred, dtype=np.float32).reshape(-1, 4)
    t = np.asarray(target, dtype=np.float32).reshape(-1, 4)
    ow = ops[:, 0:1]
    ox = ops[:, 1:2]
    oy = ops[:, 2:3]
    oz = ops[:, 3:4]
    tw = t[None, :, 0]
    tx = t[None, :, 1]
    ty = t[None, :, 2]
    tz = t[None, :, 3]
    pw = p[None, :, 0]
    px = p[None, :, 1]
    py = p[None, :, 2]
    pz = p[None, :, 3]
    dots = np.abs(
        (ow * tw - ox * tx - oy * ty - oz * tz) * pw
        + (ow * tx + ox * tw + oy * tz - oz * ty) * px
        + (ow * ty - ox * tz + oy * tw + oz * tx) * py
        + (ow * tz + ox * ty - oy * tx + oz * tw) * pz
    )
    best = np.clip(dots.max(axis=0), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(best)).reshape(pred_shape).astype(np.float32)


def boundary_mask_fast(field: np.ndarray, ops: np.ndarray, threshold_deg: float = 5.0) -> np.ndarray:
    height, width = field.shape[:2]
    boundary = np.zeros((height, width), dtype=bool)
    if width > 1:
        hit = misorientation_fast(field[:, :-1], field[:, 1:], ops) > threshold_deg
        boundary[:, :-1] |= hit
        boundary[:, 1:] |= hit
    if height > 1:
        hit = misorientation_fast(field[:-1, :], field[1:, :], ops) > threshold_deg
        boundary[:-1, :] |= hit
        boundary[1:, :] |= hit
    return boundary


def sample_ids(ref_dir: Path) -> list[int]:
    ids = []
    for path in sorted(ref_dir.glob("sample_*_sr.npy")):
        ids.append(int(path.name.split("_")[1].split(".")[0]))
    if not ids:
        raise FileNotFoundError(f"No sample_*_sr.npy files in {ref_dir}")
    return ids


def load_quat(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"Expected HxWx4 quaternion array at {path}, got {arr.shape}")
    if not np.isfinite(arr).all():
        bad = int(arr.size - np.isfinite(arr).sum())
        raise NonFiniteQuaternionError(path, bad)
    return ev.normalize_quat(arr).astype(np.float32)


def sr_path(method_dir: Path, sample_id: int) -> Path:
    return method_dir / f"sample_{sample_id:06d}_sr.npy"


def lr_path(ref_dir: Path, sample_id: int) -> Path:
    return ref_dir / f"sample_{sample_id:06d}_lr.npy"


def hr_path(ref_dir: Path, sample_id: int) -> Path:
    return ref_dir / f"sample_{sample_id:06d}_hr.npy"


def dilate_no_wrap(mask: np.ndarray, radius: int = 1, *, square: bool = True) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    if square:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return binary_dilation(mask, structure=structure, iterations=1, border_value=False)
    return binary_dilation(mask, iterations=radius, border_value=False)


def tolerant_misorientation(pred: np.ndarray, target: np.ndarray, ops: np.ndarray, radius: int = 1) -> np.ndarray:
    height, width = pred.shape[:2]
    best = np.full((height, width), np.inf, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        py0 = max(0, -dy)
        py1 = min(height, height - dy)
        ty0 = py0 + dy
        ty1 = py1 + dy
        if py0 >= py1:
            continue
        for dx in range(-radius, radius + 1):
            px0 = max(0, -dx)
            px1 = min(width, width - dx)
            tx0 = px0 + dx
            tx1 = px1 + dx
            if px0 >= px1:
                continue
            mis = misorientation_fast(pred[py0:py1, px0:px1], target[ty0:ty1, tx0:tx1], ops)
            best[py0:py1, px0:px1] = np.minimum(best[py0:py1, px0:px1], mis)
    if not np.isfinite(best).all():
        raise RuntimeError("Internal tolerance window left invalid pixels.")
    return best


def tolerant_boundary_f1(pred_boundary: np.ndarray, ref_boundary: np.ndarray, radius: int = 1) -> tuple[float, float, float]:
    pred_boundary = np.asarray(pred_boundary, dtype=bool)
    ref_boundary = np.asarray(ref_boundary, dtype=bool)
    ref_d = dilate_no_wrap(ref_boundary, radius=radius, square=True)
    pred_d = dilate_no_wrap(pred_boundary, radius=radius, square=True)
    tp_precision = int(np.logical_and(pred_boundary, ref_d).sum())
    tp_recall = int(np.logical_and(ref_boundary, pred_d).sum())
    precision = tp_precision / int(pred_boundary.sum()) if pred_boundary.any() else 0.0
    recall = tp_recall / int(ref_boundary.sum()) if ref_boundary.any() else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def grain_sizes(field: np.ndarray, ops: np.ndarray, threshold_deg: float = 5.0) -> np.ndarray:
    height, width = field.shape[:2]
    num_nodes = height * width
    index = np.arange(num_nodes, dtype=np.int64).reshape(height, width)
    edges_i = []
    edges_j = []
    if width > 1:
        same = misorientation_fast(field[:, :-1], field[:, 1:], ops) <= threshold_deg
        edges_i.append(index[:, :-1][same])
        edges_j.append(index[:, 1:][same])
    if height > 1:
        same = misorientation_fast(field[:-1, :], field[1:, :], ops) <= threshold_deg
        edges_i.append(index[:-1, :][same])
        edges_j.append(index[1:, :][same])
    if edges_i and sum(edge.size for edge in edges_i):
        ii = np.concatenate(edges_i)
        jj = np.concatenate(edges_j)
        graph = coo_matrix((np.ones_like(ii, dtype=np.uint8), (ii, jj)), shape=(num_nodes, num_nodes))
    else:
        graph = coo_matrix((num_nodes, num_nodes), dtype=np.uint8)
    _, labels = connected_components(graph, directed=False)
    return np.bincount(labels, minlength=int(labels.max()) + 1)


def method_field(method: str, method_dir: Path | None, lr: np.ndarray, out_hw: tuple[int, int], sample_id: int) -> np.ndarray:
    if method == "Nearest":
        return ev.upsample_nn(lr, out_hw)
    if method == "Bicubic":
        return ev.upsample_bicubic(lr, out_hw)
    if method == "SLERP":
        return ev.upsample_slerp(lr, out_hw)
    if method == "Symm-SLERP":
        return ev.upsample_symm_slerp(lr, out_hw)
    if method_dir is None:
        raise ValueError(f"No method directory for {method}")
    return load_quat(sr_path(method_dir, sample_id))


def preload_dataset_records(ref_dir: Path, ids: list[int], ops: np.ndarray) -> list[dict]:
    records = []
    for sid in ids:
        lr = load_quat(lr_path(ref_dir, sid))
        hr = load_quat(hr_path(ref_dir, sid))
        out_hw = tuple(hr.shape[:2])
        nn_ref = ev.upsample_nn(lr, out_hw)
        hr_boundary = boundary_mask_fast(hr, ops)
        records.append(
            {
                "sample_id": sid,
                "lr": lr,
                "hr": hr,
                "out_hw": out_hw,
                "nn_ref": nn_ref,
                "hr_boundary": hr_boundary,
                "boundary_band": dilate_no_wrap(hr_boundary, radius=5, square=False),
                "hr_grains": grain_sizes(hr, ops),
                "hr_ipf_rgbs": ipf_eval.render_three_ipf(hr),
            }
        )
    return records


def ipf_xyz_metrics(sr: np.ndarray, hr_rgbs: dict[str, np.ndarray]) -> tuple[float, float]:
    sr_rgbs = ipf_eval.render_three_ipf(sr)
    psnr_vals = [ipf_eval.psnr_uint8(hr_rgbs[axis], sr_rgbs[axis]) for axis in ("X", "Y", "Z")]
    ssim_vals = [ipf_eval.ssim_uint8(hr_rgbs[axis], sr_rgbs[axis]) for axis in ("X", "Y", "Z")]
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def summarize_method(
    dataset_key: str,
    dataset_label: str,
    symmetry_group: str,
    method: str,
    method_dir: Path | None,
    records: list[dict],
    ops: np.ndarray,
) -> dict:
    pooled_mis = []
    pooled_tol = []
    pooled_interior = []
    pooled_boundary = []
    bf1_values = []
    precision_values = []
    recall_values = []
    strict_tp_total = 0
    strict_fp_total = 0
    strict_fn_total = 0
    grain_ratios = []
    grain_wass = []
    dnn_values = []
    psnr_values = []
    ssim_values = []
    invalid_samples = []
    invalid_entries = 0

    for rec in records:
        sid = int(rec["sample_id"])
        lr = rec["lr"]
        hr = rec["hr"]
        out_hw = rec["out_hw"]
        nn_ref = rec["nn_ref"]
        try:
            sr = method_field(method, method_dir, lr, out_hw, sid)
        except NonFiniteQuaternionError as exc:
            if method == "OCRP":
                raise
            invalid_samples.append(sid)
            invalid_entries += int(exc.n_bad)
            continue
        if sr.shape != hr.shape:
            raise ValueError(f"{dataset_key}/{method}/sample_{sid:06d}: {sr.shape} != {hr.shape}")

        mis = misorientation_fast(sr, hr, ops)
        tol = tolerant_misorientation(sr, hr, ops, radius=1)
        sr_boundary = boundary_mask_fast(sr, ops)
        psnr_ipf, ssim_ipf = ipf_xyz_metrics(sr, rec["hr_ipf_rgbs"])
        strict_tp = int(np.logical_and(sr_boundary, rec["hr_boundary"]).sum())
        strict_fp = int(np.logical_and(sr_boundary, np.logical_not(rec["hr_boundary"])).sum())
        strict_fn = int(np.logical_and(np.logical_not(sr_boundary), rec["hr_boundary"]).sum())
        strict_tp_total += strict_tp
        strict_fp_total += strict_fp
        strict_fn_total += strict_fn
        precision, recall, bf1 = tolerant_boundary_f1(sr_boundary, rec["hr_boundary"], radius=1)
        sr_grains = grain_sizes(sr, ops)

        pooled_mis.append(mis.reshape(-1))
        pooled_tol.append(tol.reshape(-1))
        pooled_interior.append(mis[np.logical_not(rec["boundary_band"])].reshape(-1))
        pooled_boundary.append(mis[rec["boundary_band"]].reshape(-1))
        precision_values.append(precision)
        recall_values.append(recall)
        bf1_values.append(bf1)
        grain_ratios.append(float(len(sr_grains) / max(len(rec["hr_grains"]), 1)))
        grain_wass.append(float(wasserstein_distance(np.log10(sr_grains), np.log10(rec["hr_grains"]))))
        dnn_values.append(float(np.mean(misorientation_fast(sr, nn_ref, ops))))
        psnr_values.append(psnr_ipf)
        ssim_values.append(ssim_ipf)

    n_valid = len(pooled_mis)
    if n_valid:
        mis_all = np.concatenate(pooled_mis)
        tol_all = np.concatenate(pooled_tol)
        interior_all = np.concatenate(pooled_interior)
        boundary_all = np.concatenate(pooled_boundary)
        mean_grain_ratio = float(np.mean(grain_ratios))
    else:
        mis_all = tol_all = interior_all = boundary_all = np.array([np.nan], dtype=np.float32)
        mean_grain_ratio = float("nan")
    strict_precision = strict_tp_total / (strict_tp_total + strict_fp_total) if strict_tp_total + strict_fp_total else 0.0
    strict_recall = strict_tp_total / (strict_tp_total + strict_fn_total) if strict_tp_total + strict_fn_total else 0.0
    strict_f1 = (
        2.0 * strict_tp_total / (2.0 * strict_tp_total + strict_fp_total + strict_fn_total)
        if 2 * strict_tp_total + strict_fp_total + strict_fn_total
        else 0.0
    )
    return {
        "dataset": dataset_key,
        "dataset_label": dataset_label,
        "symmetry": symmetry_group,
        "method": method,
        "n_samples": len(records),
        "n_valid_samples": n_valid,
        "n_invalid_samples": len(invalid_samples),
        "invalid_entries": invalid_entries,
        "invalid_sample_ids": ",".join(f"{sid:06d}" for sid in invalid_samples),
        "source_dir": "" if method_dir is None else str(method_dir),
        "mean_deg": float(np.mean(mis_all)),
        "median_deg": float(np.median(mis_all)),
        "p90_deg": float(np.percentile(mis_all, 90)),
        "p95_deg": float(np.percentile(mis_all, 95)),
        "p99_deg": float(np.percentile(mis_all, 99)),
        "tol1_mean_deg": float(np.mean(tol_all)),
        "interior_mean_deg": float(np.mean(interior_all)),
        "boundary_band_mean_deg": float(np.mean(boundary_all)),
        "boundary_precision": float(strict_precision),
        "boundary_recall": float(strict_recall),
        "boundary_f1": float(strict_f1),
        "boundary_precision_tol1": float(np.mean(precision_values)),
        "boundary_recall_tol1": float(np.mean(recall_values)),
        "boundary_f1_tol1": float(np.mean(bf1_values)),
        "grain_count_ratio": mean_grain_ratio,
        "grain_log10_size_wasserstein": float(np.mean(grain_wass)),
        "distance_to_nn_deg": float(np.mean(dnn_values)),
        "grain_ratio_abs_log10": float(abs(np.log10(max(mean_grain_ratio, 1e-12)))),
        "psnr_ipf_xyz_db": float(np.mean(psnr_values)),
        "ssim_ipf_xyz": float(np.mean(ssim_values)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def best_by_dataset(rows: list[dict]) -> list[dict]:
    out = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    for dataset, group in grouped.items():
        def finite_group(metric: str) -> list[dict]:
            finite = [r for r in group if np.isfinite(r[metric])]
            if not finite:
                raise ValueError(f"No finite values for {dataset}/{metric}")
            return finite

        def min_method(metric: str) -> str:
            return min(finite_group(metric), key=lambda r: r[metric])["method"]

        def max_method(metric: str) -> str:
            return max(finite_group(metric), key=lambda r: r[metric])["method"]

        def closest_one(metric: str) -> str:
            return min(finite_group(metric), key=lambda r: abs(np.log10(max(r[metric], 1e-12))))["method"]

        out.append(
            {
                "dataset": dataset,
                "mean": min_method("mean_deg"),
                "tol1": min_method("tol1_mean_deg"),
                "interior": min_method("interior_mean_deg"),
                "boundary_band": min_method("boundary_band_mean_deg"),
                "boundary_f1_tol1": max_method("boundary_f1_tol1"),
                "grain_ratio_closest": closest_one("grain_count_ratio"),
                "grain_wasserstein": min_method("grain_log10_size_wasserstein"),
                "psnr_ipf_xyz_db": max_method("psnr_ipf_xyz_db"),
                "ssim_ipf_xyz": max_method("ssim_ipf_xyz"),
            }
        )
    return out


def write_markdown(path: Path, rows: list[dict], winners: list[dict]) -> None:
    lines = [
        "# Hardened 4x4 Topology Metrics",
        "",
        "Protocol: strict/tolerant misorientation uses the paper-local symmetry-aware "
        "misorientation; +/-1px tolerance and boundary-F1 dilation do not wrap image "
        "edges; grain counts use 4-connected components with a 5 degree "
        "symmetry-aware edge threshold. Grain-count ratio is SR/HR; closest to 1 is best.",
        "",
    ]
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset]
        if not subset:
            continue
        lines.append(f"## {subset[0]['dataset_label']} ({subset[0]['symmetry']}, n={subset[0]['n_samples']})")
        lines.append("")
        lines.append(
            "| method | valid | invalid | mean | tol1 | interior | boundary | tol-bF1 | grain ratio | grain W1 | dNN |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in subset:
            lines.append(
                "| {method} | {n_valid_samples:d} | {n_invalid_samples:d} | "
                "{mean_deg:.3f} | {tol1_mean_deg:.3f} | {interior_mean_deg:.3f} | "
                "{boundary_band_mean_deg:.3f} | {boundary_f1_tol1:.3f} | "
                "{grain_count_ratio:.3f} | {grain_log10_size_wasserstein:.3f} | "
                "{distance_to_nn_deg:.3f} |".format(**row)
            )
        lines.append("")
    lines.append("## Winners")
    lines.append("")
    lines.append("| dataset | mean | tol1 | interior | boundary | tol-bF1 | grain ratio | grain W1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in winners:
        lines.append(
            "| {dataset} | {mean} | {tol1} | {interior} | {boundary_band} | "
            "{boundary_f1_tol1} | {grain_ratio_closest} | {grain_wasserstein} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--out-prefix", default="hardened_topology_metrics")
    args = parser.parse_args()

    rows: list[dict] = []
    for dataset_key in args.datasets:
        spec = DATASETS[dataset_key]
        ref_dir = rel(spec["ref"])
        ids = sample_ids(ref_dir)
        sym_quats = configure_symmetry(spec["symmetry"])
        ops = conjugated_ops(sym_quats)
        print(f"{dataset_key}: {spec['label']}, {spec['symmetry']}, n={len(ids)}", flush=True)
        records = preload_dataset_records(ref_dir, ids, ops)
        methods: OrderedDict[str, Path | None] = OrderedDict((m, None) for m in CLASSICAL_METHODS)
        methods.update((name, rel(path)) for name, path in spec["methods"].items())
        methods["OCRP"] = ref_dir
        for method, method_dir in methods.items():
            if method_dir is not None and not method_dir.exists():
                print(f"  skip {method}: missing {method_dir}", flush=True)
                continue
            print(f"  {method}", flush=True)
            rows.append(
                summarize_method(
                    dataset_key,
                    spec["label"],
                    spec["symmetry"],
                    method,
                    method_dir,
                    records,
                    ops,
                )
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{args.out_prefix}.csv"
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    winners = best_by_dataset(rows)
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "edge_wrapping": "none",
                    "tolerant_misorientation_radius_px": 1,
                    "tolerant_boundary_f1_radius_px": 1,
                    "boundary_band_radius_px": 5,
                    "grain_connectivity": 4,
                    "grain_edge_threshold_deg": 5.0,
                    "grain_count_ratio": "num_grains_SR / num_grains_HR; closest to 1 is best",
                },
                "rows": rows,
                "winners": winners,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(md_path, rows, winners)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
