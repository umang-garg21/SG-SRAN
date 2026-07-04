#!/usr/bin/env python3
"""Compare the only-4e irrep ablation against full (2e+4e) OCRP 4x4 on IN718 Test.

Reuses the exact pooled-orientation + IPF PSNR/SSIM metric functions from
analysis/evaluate_all_learned_baselines_in718.py so numbers are directly
comparable to the paper Table-1 protocol.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
EVAL_DIR = ROOT / "Paper/EBSD_SR_Nature_v3/evals"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_test_psnr_ssim_ipf as ipf_eval  # noqa: E402
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry  # noqa: E402
from analysis.evaluate_all_learned_baselines_in718 import _evaluate_orientation_nansafe  # noqa: E402

EXP = ROOT / "experiments/IN718"
RUNS = [
    ("only-4e (ablation)", EXP / "iso_embedding_4x4_ocrp_anchorless_only4e_01/inference/test/summary.json"),
    ("full 2e+4e (s42)",   EXP / "iso_embedding_4x4_ocrp_anchorless_allepochs_s42/inference/test/summary.json"),
    ("full OCRP (paper)",  EXP / "iso_embedding_4x4_ocrp_anchorless_4x1clone_01/inference/test_epoch_0024/summary.json"),
]


def main() -> None:
    symmetry = resolve_symmetry("Oh")
    ipf_eval.SYM = symmetry
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    operators = torch.as_tensor(
        proper_symmetry_quaternions(symmetry), dtype=torch.float64, device=device
    ).clone()
    operators[:, 1:] *= -1.0

    rows = []
    for label, summ in RUNS:
        if not summ.exists():
            print(f"  SKIP {label}: missing {summ}")
            continue
        summary = json.loads(summ.read_text())
        print(f"Evaluating {label} ...", flush=True)
        m = _evaluate_orientation_nansafe(summary, label, device, operators)
        ipf_row, _ = ipf_eval.evaluate_method(summary, label, ipf_eval.provider_from_saved_sr)
        m.update({
            "label": label,
            "num_samples": int(summary["num_samples"]),
            "psnr_ipf_xyz_mean_db": ipf_row["psnr_mean_xyz"],
            "ssim_ipf_xyz_mean": ipf_row["ssim_mean_xyz"],
        })
        rows.append(m)

    print("\n===== IN718 4x4 Test : only-4e vs full =====")
    hdr = (f"{'run':>20} {'n':>4} {'mean':>7} {'median':>7} {'p90':>7} {'bF1':>6} "
           f"{'interior':>8} {'bBand':>7} {'PSNR':>7} {'SSIM':>7} {'nonfin':>7}")
    print(hdr)
    for r in rows:
        print(f"{r['label']:>20} {r['num_samples']:>4} {r['mis_mean_deg']:>7.3f} "
              f"{r['mis_median_deg']:>7.3f} {r['mis_p90_deg']:>7.3f} {r['boundary_f1']:>6.3f} "
              f"{r['interior_mean_deg']:>8.3f} {r['boundary_band_mean_deg']:>7.3f} "
              f"{r['psnr_ipf_xyz_mean_db']:>7.3f} {r['ssim_ipf_xyz_mean']:>7.4f} {r['nonfinite_pixels']:>7}")

    out = SCRIPT_DIR / "out/only4e_vs_full_in718_4x4.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
