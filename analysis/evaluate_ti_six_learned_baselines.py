#!/usr/bin/env python3
"""Evaluate all six Ti_Al_1pct learned-baseline runs with proper D6 symmetry."""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PAPER_DIR = ROOT / "Paper/EBSD_SR_Nature_v3"
PAPER_EVAL_DIR = PAPER_DIR / "evals"
for path in (ROOT, PAPER_EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_new_learned_baselines as common
import export_test_psnr_ssim_ipf as ipf_eval
from training.train_atindama_inpainting import load_authors_model
from training.train_jangid_baseline import build_model
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry


BASELINES = OrderedDict(
    [
        (
            ("4x1", "Atindama inpainting"),
            ROOT
            / "experiments/Ti_Al_1pct/atindama_inpainting_4x1_01/inference/test/summary.json",
        ),
        (
            ("4x1", "QEDSR"),
            ROOT / "experiments/Ti_Al_1pct/qedsr_4x1_01/inference/test/summary.json",
        ),
        (
            ("4x1", "Q-RBSA-adapted"),
            ROOT
            / "experiments/Ti_Al_1pct/qrbsa_adapted_4x1_300ep_01/inference/test/summary.json",
        ),
        (
            ("4x4", "Atindama inpainting"),
            ROOT
            / "experiments/Ti_Al_1pct/atindama_inpainting_4x4_01/inference/test/summary.json",
        ),
        (
            ("4x4", "QEDSR"),
            ROOT / "experiments/Ti_Al_1pct/qedsr_4x4_01/inference/test/summary.json",
        ),
        (
            ("4x4", "Q-RBSA-adapted"),
            ROOT
            / "experiments/Ti_Al_1pct/qrbsa_adapted_4x4_300ep_01/inference/test/summary.json",
        ),
    ]
)

OCRP_SUMMARIES = {
    "4x1": ROOT
    / "experiments/Ti_Al_1pct/iso_embedding_4x1_ocrp_anchorless_01/inference/test_epoch_0010/summary.json",
    "4x4": ROOT
    / "experiments/Ti_Al_1pct/iso_embedding_4x4_ocrp_anchorless_4x1clone_01/inference/test_epoch_0044/summary.json",
}

PAPER_METHODS = (
    "Nearest",
    "Bicubic",
    "SLERP",
    "Symm-SLERP",
    "Atindama inpainting",
    "Q-RBSA-adapted",
    "OCRP (ours)",
)

OUT_DIR = SCRIPT_DIR / "out"
OUT_JSON = OUT_DIR / "ti_all_six_learned_baselines_metrics.json"
OUT_CSV = OUT_DIR / "ti_all_six_learned_baselines_metrics.csv"
OUT_SAMPLE_CSV = OUT_DIR / "ti_all_six_learned_baselines_per_sample.csv"
PAPER_JSON = PAPER_EVAL_DIR / "ti_new_learned_baselines_metrics.json"
PAPER_CSV = PAPER_EVAL_DIR / "ti_new_learned_baselines_metrics.csv"
PAPER_SAMPLE_CSV = PAPER_EVAL_DIR / "ti_new_learned_baselines_per_sample.csv"
OUT_FIGURE = PAPER_DIR / "figs/main_ti_new_learned_baselines_test0.png"


def _parameter_counts() -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    experiments = (
        ("4x1", "QEDSR", "qedsr_4x1_01"),
        ("4x4", "QEDSR", "qedsr_4x4_01"),
        ("4x1", "Q-RBSA-adapted", "qrbsa_adapted_4x1_300ep_01"),
        ("4x4", "Q-RBSA-adapted", "qrbsa_adapted_4x4_300ep_01"),
    )
    for task, method, name in experiments:
        cfg = json.loads(
            (ROOT / "experiments/Ti_Al_1pct" / name / "config.json").read_text()
        )
        model = build_model(cfg)
        counts[(task, method)] = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
    model = load_authors_model(torch.device("cpu"))
    count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    counts[("4x1", "Atindama inpainting")] = count
    counts[("4x4", "Atindama inpainting")] = count
    counts[("4x1", "OCRP (ours)")] = 56_641
    counts[("4x4", "OCRP (ours)")] = 57_025
    for task in ("4x1", "4x4"):
        for method in ("Nearest", "Bicubic", "SLERP", "Symm-SLERP"):
            counts[(task, method)] = 0
    return counts


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_record_array(record: dict, key: str) -> np.ndarray:
    return common._load_record_array(record, key)


def _configure_orientation_eval(symmetry) -> None:
    orientation_eval = common.orientation_eval
    orientation_eval.SYM = symmetry
    orientation_eval.SYM_QUATS = proper_symmetry_quaternions(symmetry).astype(
        np.float64
    )
    orientation_eval._SLERP_SYM_OPS_4X4 = orientation_eval.make_symmetry_4x4(
        "D6h", device="cpu", dtype=torch.float32
    )


def _provider_for_method(method: str):
    orientation_eval = common.orientation_eval
    upsamplers = {
        "Nearest": orientation_eval.upsample_nn,
        "Bicubic": orientation_eval.upsample_bicubic,
        "SLERP": orientation_eval.upsample_slerp,
        "Symm-SLERP": orientation_eval.upsample_symm_slerp,
    }
    if method in upsamplers:
        return orientation_eval.provider_from_upsampler(upsamplers[method])
    return ipf_eval.provider_from_saved_sr


def export_sample0_figure(symmetry) -> None:
    _configure_orientation_eval(symmetry)
    orientation_eval = common.orientation_eval

    task_rows = []
    for task in ("4x1", "4x4"):
        ocrp_summary = json.loads(OCRP_SUMMARIES[task].read_text())
        ocrp_record = ocrp_summary["records"][0]
        lr = _load_record_array(ocrp_record, "lr_npy")
        hr = _load_record_array(ocrp_record, "hr_npy")
        out_hw = tuple(hr.shape[:2])
        panels = OrderedDict(
            [
                ("LR", orientation_eval.upsample_nn(lr, out_hw)),
                ("Nearest", orientation_eval.upsample_nn(lr, out_hw)),
                ("Bicubic", orientation_eval.upsample_bicubic(lr, out_hw)),
                ("SLERP", orientation_eval.upsample_slerp(lr, out_hw)),
                ("Symm-SLERP", orientation_eval.upsample_symm_slerp(lr, out_hw)),
            ]
        )
        for method in ("Atindama inpainting", "Q-RBSA-adapted"):
            summary = json.loads(BASELINES[(task, method)].read_text())
            panels[method] = _load_record_array(summary["records"][0], "sr_npy")
        panels["OCRP (ours)"] = _load_record_array(ocrp_record, "sr_npy")
        panels["HR"] = hr
        task_rows.append((task, panels))

    columns = list(task_rows[0][1].keys())
    fig, axes = plt.subplots(2, len(columns), figsize=(2.05 * len(columns), 4.7))
    for row_index, (task, panels) in enumerate(task_rows):
        for column_index, label in enumerate(columns):
            ax = axes[row_index, column_index]
            ax.imshow(common._render_ipfz(panels[label], symmetry), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(label, fontsize=9)
            if column_index == 0:
                ax.set_ylabel(f"Ti-6Al-4V {task}", fontsize=10)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("0.45")
    fig.subplots_adjust(
        left=0.035,
        right=0.995,
        top=0.89,
        bottom=0.02,
        wspace=0.025,
        hspace=0.04,
    )
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIGURE}")


def main() -> None:
    symmetry = resolve_symmetry("D6h")
    proper_count = int(proper_symmetry_quaternions(symmetry).shape[0])
    if proper_count != 12:
        raise RuntimeError(f"Expected 12 proper D6 rotations, got {proper_count}")
    _configure_orientation_eval(symmetry)
    ipf_eval.SYM = symmetry
    counts = _parameter_counts()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summaries: dict[tuple[str, str], dict] = {}
    orientation_results: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    for (task, method), summary_path in BASELINES.items():
        summary = json.loads(summary_path.read_text())
        if summary.get("symmetry_group") not in {"D6", "D6h", "hcp", "HCP"}:
            raise RuntimeError(
                f"{summary_path} does not record D6/D6h symmetry: "
                f"{summary.get('symmetry_group')!r}"
            )
        if int(summary.get("proper_rotation_count", -1)) != 12:
            raise RuntimeError(
                f"{summary_path} does not record 12 proper D6 rotations"
            )
        summary["task"] = (
            f"Ti_Al_1pct {'anisotropic' if task == '4x1' else 'isotropic'} {task}"
        )
        print(f"Evaluating {task} / {method}", flush=True)
        result = common._evaluate_orientation_gpu(
            summary, method, device, symmetry=symmetry
        )
        summaries[(task, method)] = summary
        orientation_results[(task, method)] = result

    def evaluate_learned_ipf(key):
        task, method = key
        row, samples = ipf_eval.evaluate_method(
            summaries[key], method, ipf_eval.provider_from_saved_sr
        )
        return key, row, samples

    ipf_results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(evaluate_learned_ipf, key) for key in BASELINES]
        for future in as_completed(futures):
            key, row, samples = future.result()
            ipf_results[key] = (row, samples)

    def complete_row(key, summary_path, metric_result, ipf_result):
        task, method = key
        metric_row, metric_samples = metric_result
        ipf_row, ipf_samples = ipf_result
        per_patch = np.asarray(
            [row["mis_mean_deg"] for row in metric_samples], dtype=np.float64
        )
        metric_row.update(
            {
                "task_key": task,
                "symmetry_group": "D6h",
                "proper_rotation_count": proper_count,
                "trainable_parameters": counts[key],
                "per_patch_mean_std_deg": float(np.std(per_patch)),
                "psnr_ipf_xyz_mean_db": ipf_row["psnr_mean_xyz"],
                "psnr_ipf_xyz_patch_std_db": float(
                    np.std(
                        [
                            np.mean([row[f"psnr_{axis}"] for axis in ("X", "Y", "Z")])
                            for row in ipf_samples
                        ]
                    )
                ),
                "ssim_ipf_xyz_mean": ipf_row["ssim_mean_xyz"],
                "ssim_ipf_xyz_patch_std": float(
                    np.std(
                        [
                            np.mean([row[f"ssim_{axis}"] for axis in ("X", "Y", "Z")])
                            for row in ipf_samples
                        ]
                    )
                ),
                "summary_path": str(summary_path),
            }
        )
        for row in metric_samples:
            row["task_key"] = task
            row["symmetry_group"] = "D6h"
            row["proper_rotation_count"] = proper_count
        return metric_row, metric_samples

    rows: list[dict] = []
    sample_rows: list[dict] = []
    completed_learned = {}
    for key, summary_path in BASELINES.items():
        completed = complete_row(
            key,
            summary_path,
            orientation_results[key],
            ipf_results[key],
        )
        completed_learned[key] = completed
        rows.append(completed[0])
        sample_rows.extend(completed[1])

    payload = {
        "protocol": {
            "dataset": "Ti_Al_1pct",
            "split": "held-out test split, n=133",
            "point_group": "D6h",
            "misorientation_group": "D6 proper rotational subgroup",
            "proper_rotation_count": proper_count,
        },
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(OUT_CSV, rows)
    _write_csv(OUT_SAMPLE_CSV, sample_rows)

    paper_orientation_results = {}
    paper_summaries = {}
    for task in ("4x1", "4x4"):
        summary_path = OCRP_SUMMARIES[task]
        summary = json.loads(summary_path.read_text())
        summary["task"] = (
            f"Ti_Al_1pct {'anisotropic' if task == '4x1' else 'isotropic'} {task}"
        )
        paper_summaries[(task, "OCRP (ours)")] = summary
        paper_orientation_results[(task, "OCRP (ours)")] = (
            common._evaluate_orientation_gpu(
                summary,
                "OCRP (ours)",
                device,
                symmetry=symmetry,
            )
        )
        for method in ("Nearest", "Bicubic", "SLERP", "Symm-SLERP"):
            provider = _provider_for_method(method)
            paper_summaries[(task, method)] = summary
            print(f"Evaluating {task} / {method} with proper D6", flush=True)
            paper_orientation_results[(task, method)] = (
                common._evaluate_orientation_gpu(
                    summary,
                    method,
                    device,
                    symmetry=symmetry,
                    sr_provider=provider,
                )
            )
        for method in ("Atindama inpainting", "Q-RBSA-adapted"):
            paper_summaries[(task, method)] = summaries[(task, method)]
            paper_orientation_results[(task, method)] = orientation_results[
                (task, method)
            ]

    def evaluate_paper_ipf(key):
        task, method = key
        row, samples = ipf_eval.evaluate_method(
            paper_summaries[key],
            method,
            _provider_for_method(method),
        )
        return key, row, samples

    paper_ipf_results = {}
    paper_keys = [
        (task, method)
        for task in ("4x1", "4x4")
        for method in PAPER_METHODS
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(evaluate_paper_ipf, key) for key in paper_keys]
        for future in as_completed(futures):
            key, row, samples = future.result()
            paper_ipf_results[key] = (row, samples)

    paper_rows = []
    paper_sample_rows = []
    for key in paper_keys:
        task, method = key
        if method in ("Atindama inpainting", "Q-RBSA-adapted"):
            summary_path = BASELINES[key]
        else:
            summary_path = OCRP_SUMMARIES[task]
        completed = complete_row(
            key,
            summary_path,
            paper_orientation_results[key],
            paper_ipf_results[key],
        )
        paper_rows.append(completed[0])
        paper_sample_rows.extend(completed[1])

    paper_payload = {
        **payload,
        "rows": paper_rows,
    }
    PAPER_JSON.write_text(json.dumps(paper_payload, indent=2) + "\n")
    _write_csv(PAPER_CSV, paper_rows)
    _write_csv(PAPER_SAMPLE_CSV, paper_sample_rows)
    export_sample0_figure(symmetry)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SAMPLE_CSV}")
    print(f"Wrote {PAPER_JSON}")
    print(f"Wrote {PAPER_CSV}")
    print(f"Wrote {PAPER_SAMPLE_CSV}")


if __name__ == "__main__":
    main()
