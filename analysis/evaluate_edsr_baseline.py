#!/usr/bin/env python3
"""Evaluate the original-EDSR IN718 baselines (4x1, 4x4) with the Table-1 protocol.

Reuses the exact pooled orientation + IPF-X/Y/Z PSNR/SSIM metric functions from
``Paper/EBSD_SR_Nature_v3/evals/evaluate_new_learned_baselines.py`` so the EDSR
numbers are directly comparable to Q-RBSA-adapted and OCRP. Prints a side-by-side
table and writes CSV/JSON to ``analysis/out``.

Run with the ``material`` conda env (needs orix / e3nn).
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

# (task, method) -> inference summary.json
SUMMARIES = OrderedDict(
    [
        (("4x1", "EDSR"), ROOT / "experiments/IN718/edsr_4x1_01/inference/test/summary.json"),
        (("4x4", "EDSR"), ROOT / "experiments/IN718/edsr_4x4_01/inference/test/summary.json"),
        (("4x1", "Q-RBSA-adapted"), ROOT / "experiments/IN718/qrbsa_4x1_300ep_01/inference/test/summary.json"),
        (("4x4", "Q-RBSA-adapted"), ROOT / "experiments/IN718/qrbsa_4x4_300ep_01/inference/test/summary.json"),
        (("4x1", "OCRP (ours)"), enb.OCRP_SUMMARIES["4x1"]),
        (("4x4", "OCRP (ours)"), enb.OCRP_SUMMARIES["4x4"]),
    ]
)

# experiment dir for trainable-parameter counts of the EDSR / Q-RBSA configs
PARAM_EXP = {
    ("4x1", "EDSR"): "experiments/IN718/edsr_4x1_01",
    ("4x4", "EDSR"): "experiments/IN718/edsr_4x4_01",
    ("4x1", "Q-RBSA-adapted"): "experiments/IN718/qrbsa_4x1_300ep_01",
    ("4x4", "Q-RBSA-adapted"): "experiments/IN718/qrbsa_4x4_300ep_01",
}
OCRP_PARAMS = {("4x1", "OCRP (ours)"): 56_641, ("4x4", "OCRP (ours)"): 57_025}

OUT_DIR = SCRIPT_DIR / "out"
OUT_JSON = OUT_DIR / "edsr_in718_metrics.json"
OUT_CSV = OUT_DIR / "edsr_in718_metrics.csv"


def _params(key) -> int:
    if key in OCRP_PARAMS:
        return OCRP_PARAMS[key]
    cfg = json.loads((ROOT / PARAM_EXP[key] / "config.json").read_text())
    model = build_model(cfg)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main() -> None:
    symmetry = resolve_symmetry("Oh")
    enb.orientation_eval = getattr(enb, "orientation_eval", None)  # no-op guard
    ipf_eval.SYM = symmetry
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rows: list[dict] = []
    for (task, method), summary_path in SUMMARIES.items():
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        summary["task"] = f"IN718 {'anisotropic' if task == '4x1' else 'isotropic'} {task}"
        print(f"Evaluating {task} / {method}", flush=True)
        metric_row, metric_samples = enb._evaluate_orientation_gpu(summary, method, device)
        ipf_row, ipf_samples = ipf_eval.evaluate_method(
            summary, method, ipf_eval.provider_from_saved_sr
        )
        per_patch = np.asarray([r["mis_mean_deg"] for r in metric_samples], dtype=np.float64)
        metric_row.update(
            {
                "task_key": task,
                "trainable_parameters": _params((task, method)),
                "per_patch_mean_std_deg": float(np.std(per_patch)),
                "psnr_ipf_xyz_mean_db": ipf_row["psnr_mean_xyz"],
                "ssim_ipf_xyz_mean": ipf_row["ssim_mean_xyz"],
                "summary_path": str(summary_path),
            }
        )
        rows.append(metric_row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"protocol": {"dataset": "IN718", "point_group": "Oh"}, "rows": rows}, indent=2) + "\n")
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # pretty side-by-side table
    hdr = ["task", "method", "params", "mean", "median", "p90", "bF1", "interior", "bBand", "PSNR", "SSIM"]
    print("\n" + " | ".join(f"{h:>14}" if i < 2 else f"{h:>8}" for i, h in enumerate(hdr)))
    print("-" * 130)
    for r in rows:
        print(
            f"{r['task_key']:>14} | {r['method']:>14} | "
            f"{r['trainable_parameters']:>8,} | {r['mis_mean_deg']:>8.3f} | {r['mis_median_deg']:>8.3f} | "
            f"{r['mis_p90_deg']:>8.3f} | {r['boundary_f1']:>8.3f} | {r['interior_mean_deg']:>8.3f} | "
            f"{r['boundary_band_mean_deg']:>8.3f} | {r['psnr_ipf_xyz_mean_db']:>8.3f} | {r['ssim_ipf_xyz_mean']:>8.4f}"
        )
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
