#!/usr/bin/env python3
"""Unified IN718 learned-baseline evaluation (Table-1 protocol, nan-safe).

Evaluates EDSR / HAN / RCAN / SAN (real & quaternion variants), Q-RBSA-adapted,
and OCRP on the held-out IN718 Test split using the exact pooled orientation +
IPF-X/Y/Z PSNR/SSIM metric functions from the paper evaluator. Pooling is
nan-safe: non-finite predicted pixels (a real failure mode of unconstrained
real-valued nets on the quaternion sphere) are excluded and counted.

Run with the ``material`` conda env.
"""
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

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
EVAL_DIR = ROOT / "Paper/EBSD_SR_Nature_v3/evals"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_new_learned_baselines as enb  # noqa: E402
import export_test_psnr_ssim_ipf as ipf_eval  # noqa: E402
from training.train_jangid_baseline import build_model  # noqa: E402
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry  # noqa: E402

# (task, method) -> (inference summary.json, experiment dir for param count or None)
EXP = "experiments/IN718"
ENTRIES = OrderedDict(
    [
        (("4x1", "EDSR"),           (f"{EXP}/edsr_4x1_01/inference/test/summary.json",            f"{EXP}/edsr_4x1_01")),
        (("4x4", "EDSR"),           (f"{EXP}/edsr_4x4_01/inference/test/summary.json",            f"{EXP}/edsr_4x4_01")),
        (("4x1", "HAN"),            (f"{EXP}/han_4x1_300ep_01/inference/test_ep154/summary.json", f"{EXP}/han_4x1_300ep_01")),
        (("4x4", "HAN"),            (f"{EXP}/han_4x4_300ep_01/inference/test_ep165/summary.json", f"{EXP}/han_4x4_300ep_01")),
        (("4x1", "RCAN"),           (f"{EXP}/rcan_4x1_300ep_01/inference/test_ep171/summary.json",f"{EXP}/rcan_4x1_300ep_01")),
        (("4x4", "RCAN"),           (f"{EXP}/rcan_4x4_300ep_01/inference/test_ep171/summary.json",f"{EXP}/rcan_4x4_300ep_01")),
        (("4x1", "SAN"),            (f"{EXP}/san_4x1_300ep_01/inference/test_ep193/summary.json", f"{EXP}/san_4x1_300ep_01")),
        (("4x4", "SAN"),            (f"{EXP}/san_4x4_300ep_01/inference/test_ep199/summary.json", f"{EXP}/san_4x4_300ep_01")),
        (("4x1", "Q-RBSA-adapted"), (f"{EXP}/qrbsa_4x1_300ep_01/inference/test/summary.json",     f"{EXP}/qrbsa_4x1_300ep_01")),
        (("4x4", "Q-RBSA-adapted"), (f"{EXP}/qrbsa_4x4_300ep_01/inference/test/summary.json",     f"{EXP}/qrbsa_4x4_300ep_01")),
        (("4x1", "OCRP (ours)"),    (None, None)),
        (("4x4", "OCRP (ours)"),    (None, None)),
    ]
)
OCRP_PARAMS = {("4x1", "OCRP (ours)"): 56_641, ("4x4", "OCRP (ours)"): 57_025}

OUT_DIR = SCRIPT_DIR / "out"
OUT_JSON = OUT_DIR / "all_learned_baselines_in718_metrics.json"
OUT_CSV = OUT_DIR / "all_learned_baselines_in718_metrics.csv"


def _params(key, exp_dir) -> int:
    if exp_dir is None:
        return OCRP_PARAMS[key]
    cfg = json.loads((ROOT / exp_dir / "config.json").read_text())
    return int(sum(p.numel() for p in build_model(cfg).parameters() if p.requires_grad))


def _evaluate_orientation_nansafe(summary, method, device, operators):
    all_mis, all_interior, all_boundary = [], [], []
    tp_t = fp_t = fn_t = 0
    nonfinite_pixels = 0
    per_patch_means = []
    for record in summary["records"]:
        hr_np = enb._load_record_array(record, "hr_npy")
        sr_np = enb._load_record_array(record, "sr_npy")
        sr = torch.from_numpy(np.asarray(sr_np, np.float32)).to(device, torch.float64)
        hr = torch.from_numpy(hr_np).to(device, torch.float64)
        mis = enb._misorientation_torch(sr, hr, operators).cpu().numpy().astype(np.float32)
        pred_b = enb._boundary_mask_torch(sr, operators).cpu().numpy()
        ref_b = enb._boundary_mask_torch(hr, operators).cpu().numpy()
        ref_band = binary_dilation(ref_b, iterations=5)

        tp = int(np.logical_and(pred_b, ref_b).sum())
        fp = int(np.logical_and(pred_b, ~ref_b).sum())
        fn = int(np.logical_and(~pred_b, ref_b).sum())
        tp_t += tp; fp_t += fp; fn_t += fn

        finite = np.isfinite(mis)
        nonfinite_pixels += int((~finite).sum())
        mis_f = mis[finite]
        per_patch_means.append(float(np.mean(mis_f)) if mis_f.size else np.nan)
        all_mis.append(mis_f.reshape(-1))
        all_interior.append(mis[np.logical_and(~ref_band, finite)].reshape(-1))
        all_boundary.append(mis[np.logical_and(ref_band, finite)].reshape(-1))

    mis = np.concatenate(all_mis)
    interior = np.concatenate(all_interior)
    boundary = np.concatenate(all_boundary)
    f1 = 2.0 * tp_t / (2.0 * tp_t + fp_t + fn_t) if (2 * tp_t + fp_t + fn_t) else 0.0
    prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0
    rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
    return {
        "mis_mean_deg": float(np.mean(mis)),
        "mis_median_deg": float(np.median(mis)),
        "mis_p90_deg": float(np.percentile(mis, 90)),
        "boundary_precision": prec,
        "boundary_recall": rec,
        "boundary_f1": f1,
        "interior_mean_deg": float(np.mean(interior)),
        "boundary_band_mean_deg": float(np.mean(boundary)),
        "nonfinite_pixels": nonfinite_pixels,
        "per_patch_mean_std_deg": float(np.nanstd(per_patch_means)),
    }


def main() -> None:
    symmetry = resolve_symmetry("Oh")
    ipf_eval.SYM = symmetry
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    operators = torch.as_tensor(
        proper_symmetry_quaternions(symmetry), dtype=torch.float64, device=device
    ).clone()
    operators[:, 1:] *= -1.0

    rows = []
    for (task, method), (summ, exp_dir) in ENTRIES.items():
        summary_path = enb.OCRP_SUMMARIES[task] if summ is None else ROOT / summ
        if not summary_path.exists():
            print(f"  SKIP {task}/{method}: missing {summary_path}")
            continue
        summary = json.loads(summary_path.read_text())
        summary["task"] = f"IN718 {'anisotropic' if task == '4x1' else 'isotropic'} {task}"
        print(f"Evaluating {task} / {method}", flush=True)
        m = _evaluate_orientation_nansafe(summary, method, device, operators)
        ipf_row, _ = ipf_eval.evaluate_method(summary, method, ipf_eval.provider_from_saved_sr)
        m.update({
            "task_key": task, "method": method,
            "num_samples": int(summary["num_samples"]),
            "trainable_parameters": _params((task, method), exp_dir),
            "psnr_ipf_xyz_mean_db": ipf_row["psnr_mean_xyz"],
            "ssim_ipf_xyz_mean": ipf_row["ssim_mean_xyz"],
            "summary_path": str(summary_path),
        })
        rows.append(m)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"protocol": {"dataset": "IN718", "point_group": "Oh",
                                                  "note": "interim best_model.pt for HAN/RCAN/SAN (still training to 300ep)"},
                                    "rows": rows}, indent=2) + "\n")
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    for task in ("4x1", "4x4"):
        print(f"\n===== IN718 {task} =====")
        print(f"{'method':>15} {'params':>11} {'mean':>7} {'median':>7} {'p90':>7} "
              f"{'bF1':>6} {'interior':>8} {'bBand':>7} {'PSNR':>7} {'SSIM':>7} {'nonfin':>6}")
        for r in [x for x in rows if x["task_key"] == task]:
            print(f"{r['method']:>15} {r['trainable_parameters']:>11,} {r['mis_mean_deg']:>7.3f} "
                  f"{r['mis_median_deg']:>7.3f} {r['mis_p90_deg']:>7.3f} {r['boundary_f1']:>6.3f} "
                  f"{r['interior_mean_deg']:>8.3f} {r['boundary_band_mean_deg']:>7.3f} "
                  f"{r['psnr_ipf_xyz_mean_db']:>7.3f} {r['ssim_ipf_xyz_mean']:>7.4f} {r['nonfinite_pixels']:>6}")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
