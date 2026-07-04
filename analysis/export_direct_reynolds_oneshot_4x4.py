#!/usr/bin/env python3
"""Export sample-0 4x4 qualitative comparisons with direct-Reynolds OCRP.

The previous one-shot figures used the older Cartesian/tensor-decomposition OCRP
summary. This exporter keeps that older OCRP as an explicit comparator and adds
the current direct-Reynolds-isometric OCRP checkpoint outputs.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Paper" / "EBSD_SR_Nature_v4"
EVAL_DIR = PAPER_DIR / "evals"
FIG_DIR = PAPER_DIR / "figs"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_anchorless_test_metrics as orientation_eval
import evaluate_new_learned_baselines as learned_eval
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry
from visualization.ipf_render import render_ipf_rgb


MATERIALS = [
    {
        "tag": "in718",
        "label": "IN718",
        "symmetry": "Oh",
        "root": ROOT / "experiments" / "IN718",
        "cartesian_summary": ROOT
        / "experiments/IN718/seed_runs/ocrp_4x4_s42/inference/test/summary.json",
        "direct_summary": ROOT
        / "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42/inference/test_best/summary.json",
        "saved_methods": OrderedDict(
            [
                ("EDSR", ROOT / "experiments/IN718/edsr_4x4_01/inference/test/summary.json"),
                ("QEDSR", ROOT / "experiments/IN718/qedsr_4x4_01/inference/test/summary.json"),
                (
                    "Q-RBSA-adapted",
                    ROOT / "experiments/IN718/qrbsa_4x4_300ep_01/inference/test/summary.json",
                ),
                ("HAN", ROOT / "experiments/IN718/han_4x4_300ep_01/inference/test/summary.json"),
                ("RCAN", ROOT / "experiments/IN718/rcan_4x4_300ep_01/inference/test/summary.json"),
                ("SAN", ROOT / "experiments/IN718/san_4x4_300ep_01/inference/test/summary.json"),
                (
                    "Atindama",
                    ROOT / "experiments/IN718/atindama_inpainting_4x4_01/inference/test/summary.json",
                ),
            ]
        ),
    },
    {
        "tag": "ti",
        "label": "Ti-6Al-4V",
        "symmetry": "D6h",
        "root": ROOT / "experiments" / "Ti_Al_1pct",
        "cartesian_summary": ROOT
        / "experiments/Ti_Al_1pct/seed_runs/ocrp_4x4_s42/inference/test/summary.json",
        "direct_summary": ROOT
        / "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/summary.json",
        "saved_methods": OrderedDict(
            [
                ("EDSR", ROOT / "experiments/Ti_Al_1pct/edsr_4x4_01/inference/test/summary.json"),
                ("QEDSR", ROOT / "experiments/Ti_Al_1pct/qedsr_4x4_01/inference/test/summary.json"),
                (
                    "Q-RBSA-adapted",
                    ROOT
                    / "experiments/Ti_Al_1pct/qrbsa_adapted_4x4_300ep_01/inference/test/summary.json",
                ),
                ("HAN", ROOT / "experiments/Ti_Al_1pct/han_4x4_300ep_01/inference/test/summary.json"),
                ("RCAN", ROOT / "experiments/Ti_Al_1pct/rcan_4x4_300ep_01/inference/test/summary.json"),
                ("SAN", ROOT / "experiments/Ti_Al_1pct/san_4x4_300ep_01/inference/test/summary.json"),
                (
                    "Atindama",
                    ROOT
                    / "experiments/Ti_Al_1pct/atindama_inpainting_4x4_01/inference/test/summary.json",
                ),
            ]
        ),
    },
]


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _load_record_array(record: dict, key: str) -> np.ndarray:
    path = Path(record[key])
    if not path.is_absolute():
        path = ROOT / path
    return np.load(path).astype(np.float32)


def _normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)


def _render_ipfz(q_field: np.ndarray, sym) -> np.ndarray:
    rgb = np.asarray(render_ipf_rgb(_normalize(q_field), sym)[2])
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return rgb


def _configure_symmetry(symmetry_name: str):
    sym = resolve_symmetry(symmetry_name)
    orientation_eval.SYM = sym
    orientation_eval.SYM_QUATS = proper_symmetry_quaternions(sym)
    orientation_eval._SLERP_SYM_OPS_4X4 = orientation_eval.make_symmetry_4x4(
        symmetry_name,
        device="cpu",
        dtype=torch.float32,
    )
    return sym


def _sample0_panels(spec: dict, sym) -> tuple[OrderedDict[str, np.ndarray], dict[str, str]]:
    direct_summary = _load_summary(spec["direct_summary"])
    direct_record = direct_summary["records"][0]
    lr = _load_record_array(direct_record, "lr_npy")
    hr = _load_record_array(direct_record, "hr_npy")
    out_hw = tuple(hr.shape[:2])

    panels = OrderedDict(
        [
            ("LR", orientation_eval.upsample_nn(lr, out_hw)),
            ("Nearest", orientation_eval.upsample_nn(lr, out_hw)),
            ("Bicubic", orientation_eval.upsample_bicubic(lr, out_hw)),
            ("Symm-SLERP", orientation_eval.upsample_symm_slerp(lr, out_hw)),
        ]
    )
    sources = {
        "LR": "direct summary sample 0, nearest display",
        "Nearest": "computed from direct summary LR sample 0",
        "Bicubic": "computed from direct summary LR sample 0",
        "Symm-SLERP": "computed from direct summary LR sample 0",
    }

    cartesian_summary = _load_summary(spec["cartesian_summary"])
    panels["OCRP cartesian"] = _load_record_array(cartesian_summary["records"][0], "sr_npy")
    sources["OCRP cartesian"] = str(spec["cartesian_summary"])

    panels["OCRP direct"] = _load_record_array(direct_record, "sr_npy")
    sources["OCRP direct"] = str(spec["direct_summary"])

    for method, summary_path in spec["saved_methods"].items():
        if not summary_path.exists():
            continue
        summary = _load_summary(summary_path)
        panels[method] = _load_record_array(summary["records"][0], "sr_npy")
        sources[method] = str(summary_path)

    panels["HR"] = hr
    sources["HR"] = "direct summary sample 0"
    return panels, sources


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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


def _misorientation_deg(pred: np.ndarray, ref: np.ndarray, symmetry_ops: torch.Tensor) -> np.ndarray:
    pred_t = torch.as_tensor(pred, dtype=torch.float64, device=symmetry_ops.device)
    ref_t = torch.as_tensor(ref, dtype=torch.float64, device=symmetry_ops.device)
    pred_t = pred_t / pred_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ref_t = ref_t / ref_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    shape = pred_t.shape[:-1]
    pred_flat = pred_t.reshape(-1, 4)
    ref_flat = ref_t.reshape(-1, 4)
    equivalent = _quat_mul(symmetry_ops[:, None, :], ref_flat[None, :, :])
    dots = torch.abs((equivalent * pred_flat[None, :, :]).sum(dim=-1))
    best = dots.amax(dim=0).clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(best)).reshape(shape).cpu().numpy()


def _boundary_mask(field: np.ndarray, symmetry_ops: torch.Tensor, threshold_deg: float = 5.0) -> np.ndarray:
    height, width = field.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    if width > 1:
        hit = _misorientation_deg(field[:, :-1], field[:, 1:], symmetry_ops) > threshold_deg
        mask[:, :-1] |= hit
        mask[:, 1:] |= hit
    if height > 1:
        hit = _misorientation_deg(field[:-1], field[1:], symmetry_ops) > threshold_deg
        mask[:-1, :] |= hit
        mask[1:, :] |= hit
    return mask


def _panel_metrics(panels: OrderedDict[str, np.ndarray], sym) -> list[dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ops = torch.as_tensor(proper_symmetry_quaternions(sym), dtype=torch.float64, device=device).clone()
    ops[:, 1:] *= -1.0
    hr = panels["HR"]
    ref_boundary = _boundary_mask(hr, ops)
    ref_band = binary_dilation(ref_boundary, iterations=5)

    rows = []
    for name, field in panels.items():
        if name == "HR":
            continue
        mis = _misorientation_deg(field, hr, ops).astype(np.float32)
        pred_boundary = _boundary_mask(field, ops)
        tp = int(np.logical_and(pred_boundary, ref_boundary).sum())
        fp = int(np.logical_and(pred_boundary, np.logical_not(ref_boundary)).sum())
        fn = int(np.logical_and(np.logical_not(pred_boundary), ref_boundary).sum())
        rows.append(
            {
                "method": name,
                "mean_deg": float(np.mean(mis)),
                "median_deg": float(np.median(mis)),
                "p90_deg": float(np.percentile(mis, 90)),
                "p95_deg": float(np.percentile(mis, 95)),
                "p99_deg": float(np.percentile(mis, 99)),
                "boundary_f1": float(2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
                "interior_mean_deg": float(np.mean(mis[np.logical_not(ref_band)])),
                "boundary_band_mean_deg": float(np.mean(mis[ref_band])),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_figure(spec: dict, panels: OrderedDict[str, np.ndarray], sym) -> Path:
    out_path = FIG_DIR / f"main_{spec['tag']}_new_learned_baselines_test0_4x4.png"
    method_names = list(panels.keys())
    fig, axes = plt.subplots(1, len(method_names), figsize=(2.05 * len(method_names), 2.35))
    if len(method_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, method_names):
        ax.imshow(_render_ipfz(panels[name], sym), interpolation="nearest")
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("0.45")
    fig.suptitle(f"{spec['label']} 4x4 test sample 0, IPF-Z", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.78, bottom=0.02, wspace=0.035)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for spec in MATERIALS:
        sym = _configure_symmetry(spec["symmetry"])
        panels, sources = _sample0_panels(spec, sym)
        rows = _panel_metrics(panels, sym)
        for row in rows:
            row["material"] = spec["label"]
            row["symmetry"] = spec["symmetry"]
            row["source"] = sources.get(row["method"], "")
        metrics_path = EVAL_DIR / f"direct_reynolds_oneshot_4x4_{spec['tag']}_metrics.csv"
        _write_csv(metrics_path, rows)
        figure_path = _export_figure(spec, panels, sym)
        source_path = EVAL_DIR / f"direct_reynolds_oneshot_4x4_{spec['tag']}_sources.json"
        source_path.write_text(json.dumps(sources, indent=2) + "\n")
        print(f"Wrote {metrics_path}")
        print(f"Wrote {source_path}")
        print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
