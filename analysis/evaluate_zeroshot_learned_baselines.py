#!/usr/bin/env python3
"""Evaluate CoNi zero-shot learned baselines against OCRP and interpolants."""
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
from scipy import ndimage
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "Paper/EBSD_SR_Nature_v4/evals"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_anchorless_test_metrics as anchor_eval
import export_test_psnr_ssim_ipf as ipf_eval
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry

OCRP_SUMMARY = (
    ROOT
    / "experiments/Zero_shot_performance_CoNi_x250/ocrp_direct_reynolds_isometric_l4_s42/"
    / "inference/train_best/summary.json"
)
LEARNED_SUMMARIES = OrderedDict(
    [
        (
            "Atindama inpainting",
            ROOT
            / "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/"
            / "atindama_inpainting/summary.json",
        ),
        (
            "EDSR",
            ROOT
            / "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/"
            / "edsr/summary.json",
        ),
        (
            "QEDSR",
            ROOT
            / "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/"
            / "qedsr/summary.json",
        ),
        (
            "Q-RBSA-adapted",
            ROOT
            / "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/"
            / "qrbsaadapted/summary.json",
        ),
        (
            "RCAN",
            ROOT
            / "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/"
            / "rcan/summary.json",
        ),
        (
            "SAN",
            ROOT
            / "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/"
            / "san/summary.json",
        ),
        (
            "HAN",
            ROOT
            / "experiments/Zero_shot_performance_CoNi_x250/learned_baselines_4x4/"
            / "han/summary.json",
        ),
    ]
)
OUT_ANALYSIS_JSON = ROOT / "analysis/out/zeroshot_coni_4x4_all_baselines_metrics.json"
OUT_ANALYSIS_CSV = ROOT / "analysis/out/zeroshot_coni_4x4_all_baselines_metrics.csv"
OUT_ANALYSIS_SAMPLE_CSV = ROOT / "analysis/out/zeroshot_coni_4x4_all_baselines_persample.csv"
OUT_PAPER_JSON = EVAL_DIR / "zeroshot_coni_4x4_all_baselines_metrics.json"
OUT_PAPER_CSV = EVAL_DIR / "zeroshot_coni_4x4_all_baselines_summary.csv"
OUT_PAPER_SAMPLE_CSV = EVAL_DIR / "zeroshot_coni_4x4_all_baselines_persample.csv"

STALE_OCRP_DIR = "iso_embedding_4x4_ocrp_direct_routing_01"
CURRENT_OCRP_DIR = "ocrp_direct_reynolds_isometric_l4_s42"


def _resolve_path(path_value: str) -> Path:
    path_value = path_value.replace(STALE_OCRP_DIR, CURRENT_OCRP_DIR)
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _load_summary(path: Path, *, task: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text())
    summary["task"] = task
    for record in summary.get("records", []):
        for key in ("sr_npy", "lr_npy", "hr_npy", "ipf_png"):
            if isinstance(record.get(key), str):
                record[key] = record[key].replace(STALE_OCRP_DIR, CURRENT_OCRP_DIR)
    return summary


def _load_array(record: dict, key: str) -> np.ndarray:
    return np.load(_resolve_path(record[key])).astype(np.float32)


def _to_hwc4(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected rank-3 quaternion array, got {array.shape}")
    if array.shape[-1] == 4:
        return array
    if array.shape[0] == 4:
        return np.moveaxis(array, 0, -1)
    raise ValueError(f"No quaternion axis found in {array.shape}")


def _fill_and_normalize_quat(array: np.ndarray, eps: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray]:
    q = _to_hwc4(array).astype(np.float64, copy=True)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    valid = norm[..., 0] > eps
    if not np.all(valid):
        if not np.any(valid):
            q[...] = 0.0
            q[..., 0] = 1.0
            norm = np.ones_like(norm)
        else:
            nearest = ndimage.distance_transform_edt(
                ~valid,
                return_distances=False,
                return_indices=True,
            )
            q = q[tuple(nearest)]
            norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, eps)
    q = np.where(q[..., :1] < 0.0, -q, q)
    return q.astype(np.float32, copy=False), valid


def _normalize_torch(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return torch.where(q[..., :1] < 0.0, -q, q)


def _quat_mul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(dim=-1)
    w2, x2, y2, z2 = b.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _misorientation_torch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    symmetry_ops: torch.Tensor,
) -> torch.Tensor:
    prediction = _normalize_torch(prediction)
    target = _normalize_torch(target)
    shape = prediction.shape[:-1]
    pred_flat = prediction.reshape(-1, 4)
    target_flat = target.reshape(-1, 4)
    equivalent = _quat_mul_torch(symmetry_ops[:, None, :], target_flat[None, :, :])
    dots = torch.abs((equivalent * pred_flat[None, :, :]).sum(dim=-1))
    best = dots.amax(dim=0).clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(best)).reshape(shape)


def _boundary_mask_torch(field: torch.Tensor, symmetry_ops: torch.Tensor) -> torch.Tensor:
    height, width = field.shape[:2]
    boundary = torch.zeros((height, width), dtype=torch.bool, device=field.device)
    if width > 1:
        hit = _misorientation_torch(field[:, :-1], field[:, 1:], symmetry_ops) > 5.0
        boundary[:, :-1] |= hit
        boundary[:, 1:] |= hit
    if height > 1:
        hit = _misorientation_torch(field[:-1], field[1:], symmetry_ops) > 5.0
        boundary[:-1, :] |= hit
        boundary[1:, :] |= hit
    return boundary


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return precision, recall, f1


def _ipf_xyz_metrics(sr: np.ndarray, hr: np.ndarray) -> tuple[float, float]:
    hr_rgbs = ipf_eval.render_three_ipf(hr)
    sr_rgbs = ipf_eval.render_three_ipf(sr)
    psnr_vals = [ipf_eval.psnr_uint8(hr_rgbs[axis], sr_rgbs[axis]) for axis in ("X", "Y", "Z")]
    ssim_vals = [ipf_eval.ssim_uint8(hr_rgbs[axis], sr_rgbs[axis]) for axis in ("X", "Y", "Z")]
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def _evaluate_summary(
    summary: dict,
    method: str,
    device: torch.device,
    symmetry_ops: torch.Tensor,
    sr_provider=None,
) -> tuple[dict, list[dict]]:
    mis_values: list[np.ndarray] = []
    interior_values: list[np.ndarray] = []
    boundary_values: list[np.ndarray] = []
    sample_rows: list[dict] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    tp_total = fp_total = fn_total = 0
    invalid_target_total = 0
    invalid_prediction_total = 0

    for record in summary["records"]:
        lr_raw = _load_array(record, "lr_npy")
        hr_raw = _load_array(record, "hr_npy")
        if sr_provider is None:
            sr_raw = _load_array(record, "sr_npy")
        else:
            sr_raw = sr_provider(record, lr_raw, hr_raw)

        hr_np, valid_hr = _fill_and_normalize_quat(hr_raw)
        sr_np, valid_sr = _fill_and_normalize_quat(sr_raw)
        valid_metric = np.logical_and(valid_hr, valid_sr)
        invalid_target_total += int(np.count_nonzero(~valid_hr))
        invalid_prediction_total += int(np.count_nonzero(~valid_sr))

        sr = torch.from_numpy(sr_np).to(device=device, dtype=torch.float64)
        hr = torch.from_numpy(hr_np).to(device=device, dtype=torch.float64)
        mis_np = _misorientation_torch(sr, hr, symmetry_ops).cpu().numpy().astype(np.float32)
        finite = np.isfinite(mis_np)
        valid_metric = np.logical_and(valid_metric, finite)

        pred_boundary = _boundary_mask_torch(sr, symmetry_ops).cpu().numpy()
        ref_boundary = _boundary_mask_torch(hr, symmetry_ops).cpu().numpy()
        valid_boundary = valid_metric
        pred_eval = np.logical_and(pred_boundary, valid_boundary)
        ref_eval = np.logical_and(ref_boundary, valid_boundary)
        tp = int(np.logical_and(pred_eval, ref_eval).sum())
        fp = int(np.logical_and(pred_eval, np.logical_not(ref_eval)).sum())
        fn = int(np.logical_and(np.logical_not(pred_eval), ref_eval).sum())
        precision, recall, boundary_f1 = _f1(tp, fp, fn)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        ref_band = binary_dilation(ref_eval, iterations=5)
        interior_mask = np.logical_and(valid_metric, np.logical_not(ref_band))
        boundary_mask = np.logical_and(valid_metric, ref_band)
        mis_valid = mis_np[valid_metric]
        interior = mis_np[interior_mask]
        boundary = mis_np[boundary_mask]
        psnr_ipf, ssim_ipf = _ipf_xyz_metrics(sr_np, hr_np)
        psnr_values.append(psnr_ipf)
        ssim_values.append(ssim_ipf)

        sample_rows.append(
            {
                "task": summary["task"],
                "method": method,
                "sample_id": int(record.get("sample_id", len(sample_rows))),
                "valid_pixels": int(valid_metric.sum()),
                "invalid_target_pixels_excluded": int(np.count_nonzero(~valid_hr)),
                "invalid_prediction_pixels_excluded": int(np.count_nonzero(~valid_sr)),
                "mis_mean_deg": float(np.mean(mis_valid)),
                "mis_median_deg": float(np.median(mis_valid)),
                "mis_p90_deg": float(np.percentile(mis_valid, 90)),
                "mis_p95_deg": float(np.percentile(mis_valid, 95)),
                "mis_p99_deg": float(np.percentile(mis_valid, 99)),
                "boundary_precision": precision,
                "boundary_recall": recall,
                "boundary_f1": boundary_f1,
                "interior_mean_deg": float(np.mean(interior)),
                "boundary_band_mean_deg": float(np.mean(boundary)),
                "psnr_ipf_xyz_db": psnr_ipf,
                "ssim_ipf_xyz": ssim_ipf,
            }
        )
        mis_values.append(mis_valid.reshape(-1))
        interior_values.append(interior.reshape(-1))
        boundary_values.append(boundary.reshape(-1))

    all_mis = np.concatenate(mis_values)
    all_interior = np.concatenate(interior_values)
    all_boundary = np.concatenate(boundary_values)
    precision, recall, boundary_f1 = _f1(tp_total, fp_total, fn_total)
    result = {
        "task": summary["task"],
        "method": method,
        "num_samples": int(summary["num_samples"]),
        "valid_pixels": int(all_mis.size),
        "invalid_target_pixels_excluded": invalid_target_total,
        "invalid_prediction_pixels_excluded": invalid_prediction_total,
        "mis_mean_deg": float(np.mean(all_mis)),
        "mis_median_deg": float(np.median(all_mis)),
        "mis_p90_deg": float(np.percentile(all_mis, 90)),
        "mis_p95_deg": float(np.percentile(all_mis, 95)),
        "mis_p99_deg": float(np.percentile(all_mis, 99)),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": boundary_f1,
        "boundary_tp": tp_total,
        "boundary_fp": fp_total,
        "boundary_fn": fn_total,
        "interior_mean_deg": float(np.mean(all_interior)),
        "boundary_band_mean_deg": float(np.mean(all_boundary)),
        "psnr_ipf_xyz_db": float(np.mean(psnr_values)),
        "ssim_ipf_xyz": float(np.mean(ssim_values)),
    }
    return result, sample_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    task = "IN718 -> CoNi 4x4"
    symmetry = resolve_symmetry("Oh")
    anchor_eval.SYM = symmetry
    ipf_eval.SYM = symmetry
    anchor_eval.SYM_QUATS = proper_symmetry_quaternions(symmetry)
    anchor_eval._SLERP_SYM_OPS_4X4 = anchor_eval.make_symmetry_4x4(
        "Oh", device="cpu", dtype=torch.float32
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    symmetry_ops = torch.as_tensor(
        proper_symmetry_quaternions(symmetry),
        dtype=torch.float64,
        device=device,
    ).clone()
    symmetry_ops[:, 1:] *= -1.0

    common = _load_summary(OCRP_SUMMARY, task=task)
    providers = OrderedDict(
        [
            ("Nearest", lambda _rec, lr, hr: anchor_eval.upsample_nn(lr, hr.shape[:2])),
            ("Bicubic", lambda _rec, lr, hr: anchor_eval.upsample_bicubic(lr, hr.shape[:2])),
            ("SLERP", lambda _rec, lr, hr: anchor_eval.upsample_slerp(lr, hr.shape[:2])),
            (
                "Symm-SLERP",
                lambda _rec, lr, hr: anchor_eval.upsample_symm_slerp(lr, hr.shape[:2]),
            ),
        ]
    )

    rows: list[dict] = []
    samples: list[dict] = []
    for method, provider in providers.items():
        print(f"Evaluating {method}", flush=True)
        row, sample_rows = _evaluate_summary(common, method, device, symmetry_ops, sr_provider=provider)
        rows.append(row)
        samples.extend(sample_rows)

    for method, path in LEARNED_SUMMARIES.items():
        print(f"Evaluating {method}", flush=True)
        summary = _load_summary(path, task=task)
        row, sample_rows = _evaluate_summary(summary, method, device, symmetry_ops)
        rows.append(row)
        samples.extend(sample_rows)

    print("Evaluating OCRP", flush=True)
    row, sample_rows = _evaluate_summary(common, "OCRP (ours)", device, symmetry_ops)
    rows.append(row)
    samples.extend(sample_rows)

    payload = {
        "protocol": {
            "target": "CoNi Scan1_x250, FCC Oh, zero-shot Train split",
            "source": "IN718-trained 4x4 checkpoints without target retraining",
            "num_samples": 20,
            "invalid_target_pixels": "Pixels with HR quaternion norm <= 1e-8 are excluded from scalar metrics and boundary counts.",
            "orientation_metric": "Minimum misorientation over the 24 proper cubic rotations.",
            "boundary_metric": "5 degree four-neighbour boundary mask; F1 pooled over valid target pixels.",
        },
        "rows": rows,
    }
    OUT_ANALYSIS_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_ANALYSIS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    OUT_PAPER_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(OUT_ANALYSIS_CSV, rows)
    _write_csv(OUT_ANALYSIS_SAMPLE_CSV, samples)
    _write_csv(OUT_PAPER_CSV, rows)
    _write_csv(OUT_PAPER_SAMPLE_CSV, samples)

    print(f"Wrote {OUT_ANALYSIS_CSV}")
    print(f"Wrote {OUT_PAPER_CSV}")
    for row in rows:
        print(
            f"{row['method']:>20s}  mean={row['mis_mean_deg']:.3f}  "
            f"median={row['mis_median_deg']:.3f}  p90={row['mis_p90_deg']:.3f}  "
            f"F1={row['boundary_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
