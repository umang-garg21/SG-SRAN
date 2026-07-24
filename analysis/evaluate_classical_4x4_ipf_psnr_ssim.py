#!/usr/bin/env python3
"""Evaluate deterministic classical 4x4 baselines with IPF PSNR/SSIM.

This script fills the reporting gap for the non-learnable baselines in the
paper tables.  It evaluates Nearest, Bicubic, SLERP and Symm-SLERP on the same
held-out test summaries used by the OCRP runs, rendering IPF-X/Y/Z with the
paper evaluator and reporting the average across directions.

The methods are deterministic: the only variability reported here is the
per-patch standard deviation across the fixed held-out test split.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "Paper" / "EBSD_SR_Nature_v3"
EVAL_DIR = PAPER_DIR / "evals"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_test_psnr_ssim_ipf as ipf_eval
from utils.symmetry_utils import resolve_symmetry


TASKS = OrderedDict(
    [
        (
            ("IN718", "4x4"),
            {
                "summary": ROOT
                / "experiments/IN718/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/inference/test_epoch_0024/summary.json",
                "symmetry_group": "Oh",
                "material_label": "IN718",
                "task_label": "IN718 isotropic 4x4",
            },
        ),
        (
            ("Ti-6Al-4V", "4x4"),
            {
                "summary": ROOT
                / "experiments/Ti_Al_1pct/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/inference/test_epoch_0044/summary.json",
                "symmetry_group": "D6h",
                "material_label": "Ti-6Al-4V",
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

OUT_CSV = EVAL_DIR / "classical_4x4_ipf_psnr_ssim.csv"
OUT_JSON = EVAL_DIR / "classical_4x4_ipf_psnr_ssim.json"
OUT_SAMPLE_CSV = EVAL_DIR / "classical_4x4_ipf_psnr_ssim_per_sample.csv"
ANALYSIS_OUT_CSV = ROOT / "analysis/out/classical_4x4_ipf_psnr_ssim.csv"


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


def _configure_symmetry(group_name: str) -> None:
    ipf_eval.SYM = resolve_symmetry(group_name)
    ipf_eval._SLERP_SYM_OPS_4X4 = ipf_eval.make_symmetry_4x4(
        group_name, device="cpu", dtype=torch.float32
    )


def _patch_xyz_mean(samples: list[dict], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            np.mean([row[f"{prefix}_{axis}"] for axis in ("X", "Y", "Z")])
            for row in samples
        ],
        dtype=np.float64,
    )


def evaluate() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    sample_rows: list[dict] = []
    for (material_key, task_key), cfg in TASKS.items():
        summary_path = Path(cfg["summary"])
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        summary["task"] = cfg["task_label"]
        _configure_symmetry(cfg["symmetry_group"])
        for method, upsampler in METHODS.items():
            provider = ipf_eval.provider_from_upsampler(upsampler)
            print(
                f"Evaluating {cfg['task_label']} / {method} "
                f"({cfg['symmetry_group']}, n={summary['num_samples']})",
                flush=True,
            )
            metric_row, samples = ipf_eval.evaluate_method(summary, method, provider)
            patch_psnr = _patch_xyz_mean(samples, "psnr")
            patch_ssim = _patch_xyz_mean(samples, "ssim")
            row = {
                "material": material_key,
                "task": task_key,
                "task_label": cfg["task_label"],
                "method": method,
                "params_k": 0,
                "symmetry_group": cfg["symmetry_group"],
                "num_samples": int(summary["num_samples"]),
                "psnr_ipf_xyz_mean_db": float(metric_row["psnr_mean_xyz"]),
                "psnr_ipf_xyz_patch_std_db": float(np.std(patch_psnr)),
                "ssim_ipf_xyz_mean": float(metric_row["ssim_mean_xyz"]),
                "ssim_ipf_xyz_patch_std": float(np.std(patch_ssim)),
                "psnr_x_mean_db": float(metric_row["psnr_mean"]["X"]),
                "psnr_y_mean_db": float(metric_row["psnr_mean"]["Y"]),
                "psnr_z_mean_db": float(metric_row["psnr_mean"]["Z"]),
                "ssim_x_mean": float(metric_row["ssim_mean"]["X"]),
                "ssim_y_mean": float(metric_row["ssim_mean"]["Y"]),
                "ssim_z_mean": float(metric_row["ssim_mean"]["Z"]),
                "summary_path": str(summary_path),
            }
            rows.append(row)
            for sample in samples:
                sample_rows.append(
                    {
                        "material": material_key,
                        "task": task_key,
                        "method": method,
                        **sample,
                        "psnr_xyz_mean_db": float(
                            np.mean([sample[f"psnr_{axis}"] for axis in ("X", "Y", "Z")])
                        ),
                        "ssim_xyz_mean": float(
                            np.mean([sample[f"ssim_{axis}"] for axis in ("X", "Y", "Z")])
                        ),
                    }
                )
            print(
                f"  PSNR={row['psnr_ipf_xyz_mean_db']:.3f}±{row['psnr_ipf_xyz_patch_std_db']:.3f} dB, "
                f"SSIM={row['ssim_ipf_xyz_mean']:.4f}±{row['ssim_ipf_xyz_patch_std']:.4f}",
                flush=True,
            )
    return rows, sample_rows


def main() -> None:
    rows, sample_rows = evaluate()
    payload = {
        "protocol": {
            "split": "held-out test split",
            "methods": list(METHODS),
            "psnr": "PSNR on IPF-X/Y/Z uint8 RGB renderings, data range 255",
            "ssim": "SSIM on IPF-X/Y/Z uint8 RGB renderings, win_size=7, channel_axis=-1",
            "uncertainty": "per-patch standard deviation across the deterministic held-out test split",
            "seeds": "not applicable: all listed methods are deterministic non-learnable interpolants",
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(OUT_CSV, rows)
    _write_csv(ANALYSIS_OUT_CSV, rows)
    _write_csv(OUT_SAMPLE_CSV, sample_rows)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {ANALYSIS_OUT_CSV}")
    print(f"Wrote {OUT_SAMPLE_CSV}")


if __name__ == "__main__":
    main()
