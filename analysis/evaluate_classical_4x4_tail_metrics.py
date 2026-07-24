#!/usr/bin/env python3
"""Evaluate pooled tail percentiles for deterministic classical 4x4 baselines."""

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

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "Paper" / "EBSD_SR_Nature_v4"
EVAL_DIR = PAPER_DIR / "evals"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_new_learned_baselines as enb
import export_test_psnr_ssim_ipf as ipf_eval
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry


TASKS = OrderedDict(
    [
        (
            ("IN718", "4x4"),
            {
                "summary": ROOT
                / "experiments/IN718/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/inference/test_epoch_0024/summary.json",
                "symmetry_group": "Oh",
                "task_label": "IN718 isotropic 4x4",
            },
        ),
        (
            ("Ti-6Al-4V", "4x4"),
            {
                "summary": ROOT
                / "experiments/Ti_Al_1pct/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/inference/test_epoch_0044/summary.json",
                "symmetry_group": "D6h",
                "task_label": "Ti-6Al-4V isotropic 4x4",
            },
        ),
    ]
)

METHODS = OrderedDict(
    [
        ("Nearest", ipf_eval.upsample_nn),
        ("Bicubic", ipf_eval.upsample_bicubic),
        ("SLERP", ipf_eval.upsample_slerp),
        ("Symm-SLERP", ipf_eval.upsample_symm_slerp),
    ]
)

OUT_CSV = EVAL_DIR / "classical_4x4_tail_metrics.csv"
OUT_JSON = EVAL_DIR / "classical_4x4_tail_metrics.json"
ANALYSIS_OUT_CSV = ROOT / "analysis/out/classical_4x4_tail_metrics.csv"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _configure_symmetry(group_name: str) -> torch.Tensor:
    symmetry = resolve_symmetry(group_name)
    ipf_eval.SYM = symmetry
    ipf_eval._SLERP_SYM_OPS_4X4 = ipf_eval.make_symmetry_4x4(
        group_name, device="cpu", dtype=torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ops = torch.as_tensor(
        proper_symmetry_quaternions(symmetry), dtype=torch.float64, device=device
    ).clone()
    ops[:, 1:] *= -1.0
    return ops


def _evaluate_method(summary: dict, method: str, upsampler, ops: torch.Tensor) -> dict:
    all_mis: list[np.ndarray] = []
    nonfinite = 0
    for record in summary["records"]:
        lr_np = enb._load_record_array(record, "lr_npy")
        hr_np = enb._load_record_array(record, "hr_npy")
        sr_np = upsampler(lr_np, hr_np.shape[:2])
        sr = torch.from_numpy(np.asarray(sr_np, dtype=np.float32)).to(
            device=ops.device, dtype=torch.float64
        )
        hr = torch.from_numpy(np.asarray(hr_np, dtype=np.float32)).to(
            device=ops.device, dtype=torch.float64
        )
        mis = enb._misorientation_torch(sr, hr, ops).cpu().numpy().astype(np.float32)
        finite = np.isfinite(mis)
        nonfinite += int((~finite).sum())
        all_mis.append(mis[finite])

    mis_all = np.concatenate(all_mis)
    return {
        "method": method,
        "num_samples": int(summary["num_samples"]),
        "num_pixels": int(mis_all.size),
        "mean_deg": float(np.mean(mis_all)),
        "median_deg": float(np.median(mis_all)),
        "p90_deg": float(np.percentile(mis_all, 90)),
        "p95_deg": float(np.percentile(mis_all, 95)),
        "p99_deg": float(np.percentile(mis_all, 99)),
        "nonfinite_pixels": int(nonfinite),
    }


def main() -> None:
    rows: list[dict] = []
    for (material, task_key), cfg in TASKS.items():
        summary_path = Path(cfg["summary"])
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ops = _configure_symmetry(cfg["symmetry_group"])
        for method, upsampler in METHODS.items():
            print(
                f"Evaluating {cfg['task_label']} / {method} "
                f"({cfg['symmetry_group']}, n={summary['num_samples']})",
                flush=True,
            )
            row = {
                "material": material,
                "task": task_key,
                "task_label": cfg["task_label"],
                "symmetry_group": cfg["symmetry_group"],
                "params_k": 0,
                "summary_path": str(summary_path),
            }
            row.update(_evaluate_method(summary, method, upsampler, ops))
            rows.append(row)
            print(
                f"  mean={row['mean_deg']:.4f} p90={row['p90_deg']:.4f} "
                f"p95={row['p95_deg']:.4f} p99={row['p99_deg']:.4f}",
                flush=True,
            )

    payload = {
        "protocol": {
            "split": "held-out test split",
            "methods": list(METHODS),
            "orientation_metric": "pooled per-pixel proper-rotation symmetry-aware misorientation",
            "uncertainty": "not applicable: deterministic non-learnable interpolants",
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(OUT_CSV, rows)
    _write_csv(ANALYSIS_OUT_CSV, rows)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {ANALYSIS_OUT_CSV}")


if __name__ == "__main__":
    main()
