from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

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
from inference.infer_iso_embedding_sr_attn import (
    _flatten_quat_chw,
    _load_model_from_checkpoint,
    _to_hwc_quat_single,
    _unpack_batch,
)
from training.data_loading import build_dataloader
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry


def _namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**deepcopy(d))


def _symmetry_ops(group: str, device: torch.device) -> torch.Tensor:
    ops = torch.as_tensor(
        proper_symmetry_quaternions(resolve_symmetry(group)),
        dtype=torch.float64,
        device=device,
    ).clone()
    ops[:, 1:] *= -1.0
    return ops


def _metrics_from_arrays(srs: list[np.ndarray], hrs: list[np.ndarray], ops: torch.Tensor) -> dict[str, float]:
    all_mis: list[np.ndarray] = []
    interior: list[np.ndarray] = []
    boundary_band: list[np.ndarray] = []
    tp_total = fp_total = fn_total = nonfinite = 0
    for sr_np, hr_np in zip(srs, hrs):
        sr = torch.from_numpy(np.asarray(sr_np, dtype=np.float32)).to(device=ops.device, dtype=torch.float64)
        hr = torch.from_numpy(np.asarray(hr_np, dtype=np.float32)).to(device=ops.device, dtype=torch.float64)
        mis = enb._misorientation_torch(sr, hr, ops).detach().cpu().numpy().astype(np.float32)
        pred_boundary = enb._boundary_mask_torch(sr, ops).detach().cpu().numpy()
        ref_boundary = enb._boundary_mask_torch(hr, ops).detach().cpu().numpy()
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
    mis_boundary = np.concatenate(boundary_band)
    denom = 2 * tp_total + fp_total + fn_total
    return {
        "n_samples": len(srs),
        "mean_deg": float(np.mean(mis_all)),
        "median_deg": float(np.median(mis_all)),
        "p90_deg": float(np.percentile(mis_all, 90)),
        "p95_deg": float(np.percentile(mis_all, 95)),
        "p99_deg": float(np.percentile(mis_all, 99)),
        "boundary_f1": float(2.0 * tp_total / denom) if denom else float("nan"),
        "interior_mean_deg": float(np.mean(mis_interior)),
        "boundary_band_mean_deg": float(np.mean(mis_boundary)),
        "nonfinite_pixels": int(nonfinite),
    }


def _sr_diff(a: list[np.ndarray], b: list[np.ndarray], ops: torch.Tensor) -> dict[str, float]:
    vals: list[np.ndarray] = []
    for aa, bb in zip(a, b):
        qa = torch.from_numpy(np.asarray(aa, dtype=np.float32)).to(device=ops.device, dtype=torch.float64)
        qb = torch.from_numpy(np.asarray(bb, dtype=np.float32)).to(device=ops.device, dtype=torch.float64)
        vals.append(enb._misorientation_torch(qa, qb, ops).detach().cpu().numpy().astype(np.float32).reshape(-1))
    x = np.concatenate(vals)
    return {
        "mean_deg": float(np.mean(x)),
        "median_deg": float(np.median(x)),
        "p95_deg": float(np.percentile(x, 95)),
        "p99_deg": float(np.percentile(x, 99)),
        "frac_gt_0p1_deg": float(np.mean(x > 0.1)),
        "frac_gt_1_deg": float(np.mean(x > 1.0)),
    }


def _run_condition(
    *,
    label: str,
    cfg_dict: dict,
    checkpoint: Path,
    cluster_source: str,
    threshold_l2: float | None,
    take_first: int,
    device: torch.device,
) -> tuple[dict[str, float], list[np.ndarray], list[np.ndarray]]:
    cfg_dict = deepcopy(cfg_dict)
    cfg_dict["cluster_source"] = cluster_source
    if threshold_l2 is None:
        cfg_dict.pop("cluster_feature_l2_threshold", None)
    else:
        cfg_dict["cluster_feature_l2_threshold"] = float(threshold_l2)
    cfg = _namespace(cfg_dict)
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.preload = False
    cfg.preload_torch = False
    cfg.pin_memory = False

    model = _load_model_from_checkpoint(cfg, checkpoint, device=device)
    loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split="Test",
        batch_size=1,
        num_workers=0,
        preload=False,
        preload_torch=False,
        pin_memory=False,
        shuffle=False,
        take_first=take_first,
        seed=int(getattr(cfg, "seed", 42)),
        return_lr_boundary_map=False,
    )

    srs: list[np.ndarray] = []
    hrs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            lr_batch, hr_batch, _ = _unpack_batch(batch)
            lr = lr_batch[0].to(device=device, dtype=torch.float32)
            hr = hr_batch[0].to(device=device, dtype=torch.float32)
            lr_flat, lr_shape = _flatten_quat_chw(lr)
            with torch.enable_grad():
                sr_flat = model.forward_sr(lr_flat, lr_shape=lr_shape, normalize_input=True)
            hr_hwc = _to_hwc_quat_single(hr)
            sr_hwc = sr_flat.reshape(int(hr_hwc.shape[0]), int(hr_hwc.shape[1]), 4)
            srs.append(sr_hwc.detach().cpu().numpy().astype(np.float32))
            hrs.append(hr_hwc.detach().cpu().numpy().astype(np.float32))
    ops = _symmetry_ops(getattr(cfg, "symmetry_group", "O"), device)
    metrics = _metrics_from_arrays(srs, hrs, ops)
    metrics["condition"] = label
    metrics["checkpoint"] = str(checkpoint.relative_to(ROOT))
    metrics["cluster_source"] = cluster_source
    metrics["cluster_feature_l2_threshold"] = threshold_l2
    return metrics, srs, hrs


def _material_specs() -> dict[str, dict[str, object]]:
    return {
        "IN718": {
            "run_dir": ROOT / "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42",
            "symmetry": "Oh",
        },
        "Ti_Al_1pct": {
            "run_dir": ROOT / "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l6_s42",
            "symmetry": "D6h",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materials", nargs="+", default=["IN718", "Ti_Al_1pct"])
    parser.add_argument("--take-first", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/out/seed42_cluster_route_ab.json")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    threshold_l2 = float(np.deg2rad(2.0))

    out: dict[str, object] = {
        "take_first": int(args.take_first),
        "device": str(device),
        "materials": {},
    }
    specs = _material_specs()
    for material in args.materials:
        run_dir = Path(specs[material]["run_dir"])
        current_cfg = json.loads((run_dir / "logs/inference_run_config.json").read_text())
        old_cfg = json.loads((run_dir / "pre_feature_cluster_backup_20260705/logs/inference_run_config.json").read_text())
        active_ckpt = run_dir / "checkpoints/best_model.pt"
        old_ckpt = run_dir / "pre_feature_cluster_backup_20260705/checkpoints/best_model.pt"

        conditions = [
            ("old_ckpt_quaternion", old_cfg, old_ckpt, "quaternion", None),
            ("old_ckpt_feature", old_cfg, old_ckpt, "feature", threshold_l2),
            ("new_ckpt_feature", current_cfg, active_ckpt, "feature", threshold_l2),
            ("new_ckpt_quaternion", current_cfg, active_ckpt, "quaternion", None),
        ]
        metrics_rows = []
        sr_by_label: dict[str, list[np.ndarray]] = {}
        hr_ref: list[np.ndarray] | None = None
        for label, cfg, ckpt, source, thr in conditions:
            print(f"[{material}] running {label}", flush=True)
            metrics, srs, hrs = _run_condition(
                label=label,
                cfg_dict=cfg,
                checkpoint=ckpt,
                cluster_source=source,
                threshold_l2=thr,
                take_first=int(args.take_first),
                device=device,
            )
            metrics_rows.append(metrics)
            sr_by_label[label] = srs
            if hr_ref is None:
                hr_ref = hrs

        ops = _symmetry_ops(str(specs[material]["symmetry"]), device)
        diffs = {
            "old_feature_vs_old_quaternion": _sr_diff(
                sr_by_label["old_ckpt_feature"], sr_by_label["old_ckpt_quaternion"], ops
            ),
            "new_feature_vs_new_quaternion": _sr_diff(
                sr_by_label["new_ckpt_feature"], sr_by_label["new_ckpt_quaternion"], ops
            ),
            "new_feature_vs_old_feature": _sr_diff(
                sr_by_label["new_ckpt_feature"], sr_by_label["old_ckpt_feature"], ops
            ),
            "new_quaternion_vs_old_quaternion": _sr_diff(
                sr_by_label["new_ckpt_quaternion"], sr_by_label["old_ckpt_quaternion"], ops
            ),
        }
        out["materials"][material] = {
            "conditions": metrics_rows,
            "sr_diffs": diffs,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
