#!/usr/bin/env python3
"""Evaluate HCP 4x4 zero-shot runs against classical interpolants."""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "Paper/EBSD_SR_Nature_v4/evals"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_anchorless_test_metrics as anchor_eval
import export_test_psnr_ssim_ipf as ipf_eval
from analysis.evaluate_zeroshot_learned_baselines import (  # noqa: E402
    _evaluate_summary,
    _load_summary,
    _write_csv,
)
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry

METHODS = OrderedDict(
    [
        ("Atindama inpainting", "atindama_inpainting"),
        ("EDSR", "edsr"),
        ("QEDSR", "qedsr"),
        ("Q-RBSA-adapted", "qrbsaadapted"),
        ("RCAN", "rcan"),
        ("SAN", "san"),
        ("HAN", "han"),
    ]
)

TARGETS = OrderedDict(
    [
        (
            "ti7",
            {
                "task": "Ti-6Al-4V -> Ti7-deformed 4x4",
                "target": "Ti7_deformed_4x4, HCP D6h, zero-shot Train split",
                "source": "Ti-6Al-4V-trained 4x4 checkpoints without target retraining",
                "split": "Train",
                "prefix": "zeroshot_ti7_deformed_4x4_all_baselines",
                "out_root": ROOT / "experiments/Zero_shot_performance_Ti7_deformed",
                "ocrp_summary": ROOT
                / "experiments/Zero_shot_performance_Ti7_deformed/ocrp_direct_reynolds_isometric_l6_s42/"
                / "inference/train_best/summary.json",
            },
        ),
        (
            "ti64_dic_mclean",
            {
                "task": "Ti-6Al-4V -> Ti64 4x4",
                "target": "Ti64, HCP D6h, zero-shot Test split",
                "source": "Ti-6Al-4V-trained 4x4 checkpoints without target retraining",
                "split": "Test",
                "prefix": "zeroshot_ti64_dic_mclean_4x4_all_baselines",
                "out_root": ROOT / "experiments/Zero_shot_performance_Ti64_DIC_Mclean",
                "ocrp_summary": ROOT
                / "experiments/Zero_shot_performance_Ti64_DIC_Mclean/ocrp_direct_reynolds_isometric_l6_s42/"
                / "inference/test_best/summary.json",
            },
        ),
    ]
)


def _learned_summaries(out_root: Path) -> OrderedDict[str, Path]:
    return OrderedDict(
        (method, out_root / "learned_baselines_4x4" / slug / "summary.json")
        for method, slug in METHODS.items()
    )


def _configure_hcp_symmetry(device: torch.device) -> torch.Tensor:
    symmetry = resolve_symmetry("D6h")
    anchor_eval.SYM = symmetry
    ipf_eval.SYM = symmetry
    anchor_eval.SYM_QUATS = proper_symmetry_quaternions(symmetry)
    anchor_eval._SLERP_SYM_OPS_4X4 = anchor_eval.make_symmetry_4x4(
        "D6h", device="cpu", dtype=torch.float32
    )
    symmetry_ops = torch.as_tensor(
        proper_symmetry_quaternions(symmetry),
        dtype=torch.float64,
        device=device,
    ).clone()
    symmetry_ops[:, 1:] *= -1.0
    return symmetry_ops


def evaluate_target(target_key: str, device: torch.device) -> tuple[list[dict], list[dict]]:
    target = TARGETS[target_key]
    symmetry_ops = _configure_hcp_symmetry(device)
    common = _load_summary(Path(target["ocrp_summary"]), task=str(target["task"]))

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
        print(f"[{target_key}] Evaluating {method}", flush=True)
        row, sample_rows = _evaluate_summary(common, method, device, symmetry_ops, sr_provider=provider)
        rows.append(row)
        samples.extend(sample_rows)

    for method, path in _learned_summaries(Path(target["out_root"])).items():
        if not Path(path).exists():
            print(f"[{target_key}] Skipping {method}: missing {path}", flush=True)
            continue
        print(f"[{target_key}] Evaluating {method}", flush=True)
        summary = _load_summary(path, task=str(target["task"]))
        row, sample_rows = _evaluate_summary(summary, method, device, symmetry_ops)
        rows.append(row)
        samples.extend(sample_rows)

    print(f"[{target_key}] Evaluating OCRP", flush=True)
    row, sample_rows = _evaluate_summary(common, "OCRP (ours)", device, symmetry_ops)
    rows.append(row)
    samples.extend(sample_rows)

    payload = {
        "protocol": {
            "target": target["target"],
            "source": target["source"],
            "split": target["split"],
            "num_samples": int(common["num_samples"]),
            "invalid_target_pixels": "Pixels with HR quaternion norm <= 1e-8 are excluded from scalar metrics and boundary counts.",
            "orientation_metric": "Minimum misorientation over the 12 proper hexagonal rotations.",
            "boundary_metric": "5 degree four-neighbour boundary mask; F1 pooled over valid target pixels.",
        },
        "rows": rows,
    }
    prefix = str(target["prefix"])
    analysis_json = ROOT / "analysis/out" / f"{prefix}_metrics.json"
    analysis_csv = ROOT / "analysis/out" / f"{prefix}_metrics.csv"
    analysis_sample_csv = ROOT / "analysis/out" / f"{prefix}_persample.csv"
    paper_json = EVAL_DIR / f"{prefix}_metrics.json"
    paper_csv = EVAL_DIR / f"{prefix}_summary.csv"
    paper_sample_csv = EVAL_DIR / f"{prefix}_persample.csv"
    for path in (analysis_json, paper_json):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(analysis_csv, rows)
    _write_csv(analysis_sample_csv, samples)
    _write_csv(paper_csv, rows)
    _write_csv(paper_sample_csv, samples)

    print(f"[{target_key}] Wrote {analysis_csv}")
    print(f"[{target_key}] Wrote {paper_csv}")
    for row in rows:
        print(
            f"{target_key:>16s} {row['method']:>20s}  mean={row['mis_mean_deg']:.3f}  "
            f"median={row['mis_median_deg']:.3f}  p90={row['mis_p90_deg']:.3f}  "
            f"F1={row['boundary_f1']:.3f}"
        )
    return rows, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=[*TARGETS.keys(), "all"],
        default="all",
        help="Which zero-shot target to evaluate.",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selected = TARGETS if args.target == "all" else OrderedDict([(args.target, TARGETS[args.target])])
    combined_rows: list[dict] = []
    combined_samples: list[dict] = []
    for target_key in selected:
        rows, samples = evaluate_target(target_key, device)
        combined_rows.extend(rows)
        combined_samples.extend(samples)

    if len(selected) > 1:
        _write_csv(EVAL_DIR / "zeroshot_hcp_4x4_all_targets_summary.csv", combined_rows)
        _write_csv(ROOT / "analysis/out/zeroshot_hcp_4x4_all_targets_metrics.csv", combined_rows)
        _write_csv(ROOT / "analysis/out/zeroshot_hcp_4x4_all_targets_persample.csv", combined_samples)


if __name__ == "__main__":
    main()
