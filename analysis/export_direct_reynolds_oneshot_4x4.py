#!/usr/bin/env python3
"""Export selected 4x4 qualitative comparisons with direct-Reynolds OCRP."""
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
FIGURE_FONT = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": FIGURE_FONT,
        "font.size": 10,
        "axes.titlecolor": "#1f2933",
        "text.color": "#1f2933",
        "text.antialiased": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import torch
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(
    os.environ.get(
        "QSR_PAPER_DIR",
        ROOT / "Paper" / "202608_Umang_EBSD_SR_fwd" / "EBSD_SR_Nature_NMI",
    )
)
EVAL_DIR = PAPER_DIR / "evals"
FIG_DIR = PAPER_DIR / "figs"
HELPER_EVAL_DIR = ROOT / "Paper" / "EBSD_SR_Nature_v4" / "evals"
for path in (ROOT, EVAL_DIR, HELPER_EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_anchorless_test_metrics as orientation_eval
import evaluate_new_learned_baselines as learned_eval
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry
from visualization.ipf_render import add_ipf_key_panel, render_ipf_rgb


MATERIALS = [
    {
        "tag": "in718",
        "label": "IN718",
        "symmetry": "Oh",
        "sample_index": 0,
        "root": ROOT / "experiments" / "IN718",
        "direct_summary": [
            ROOT
            / "experiments/IN718/direct_reynolds_isometric_seed_runs/"
            / "ocrp_direct_reynolds_isometric_l4_s42_fresh_allepochs_20260707_2205/"
            / "inference/test_epoch_0012_fig3/summary.json",
            ROOT
            / "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42/inference/test_best/summary.json",
            ROOT
            / "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42/pre_corrected_seed_sweep_backup_corrected_feature_cluster_20260705_202111/inference/test_best/summary.json",
        ],
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
        "sample_index": 7,
        "root": ROOT / "experiments" / "Ti_Al_1pct",
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

FIGURE_LAYOUT = [
    ["LR", "HR", "Nearest", "Bicubic", "SLERP"],
    ["Symm-SLERP", "Atindama", "Q-RBSA-adapted", "QEDSR", "EDSR"],
    ["RCAN", "SAN", "HAN", "OCRP", "IPF key"],
]

PANEL_LETTER_FONTSIZE = 18.0
PANEL_HEADER_FONTSIZE = 15.0
PANEL_TITLE_FONTSIZE = 13.0
IPF_KEY_TITLE_FONTSIZE = 12.5
IPF_KEY_LABEL_FONTSIZE = 9.5
FIGURE_DPI = 900


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _resolve_existing_path(path_or_candidates) -> Path:
    if isinstance(path_or_candidates, (list, tuple)):
        candidates = [Path(path) for path in path_or_candidates]
        for path in candidates:
            if path.exists():
                return path
        joined = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No candidate path exists:\n  {joined}")
    return Path(path_or_candidates)


def _load_record_array(record: dict, key: str, summary_dir: Path | None = None) -> np.ndarray:
    path = Path(record[key])
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists() and summary_dir is not None:
        moved_path = summary_dir / "sr_quaternions" / path.name
        if moved_path.exists():
            path = moved_path
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


def _sample_panels(spec: dict, sym) -> tuple[OrderedDict[str, np.ndarray], dict[str, str]]:
    direct_summary_path = _resolve_existing_path(spec["direct_summary"])
    direct_summary = _load_summary(direct_summary_path)
    sample_index = int(spec.get("sample_index", 0))
    direct_record = direct_summary["records"][sample_index]
    lr = _load_record_array(direct_record, "lr_npy", direct_summary_path.parent)
    hr = _load_record_array(direct_record, "hr_npy", direct_summary_path.parent)
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
    sources = {
        "LR": f"direct summary sample {sample_index}, nearest display",
        "Nearest": f"computed from direct summary LR sample {sample_index}",
        "Bicubic": f"computed from direct summary LR sample {sample_index}",
        "SLERP": f"computed from direct summary LR sample {sample_index}",
        "Symm-SLERP": f"computed from direct summary LR sample {sample_index}",
    }

    panels["OCRP"] = _load_record_array(direct_record, "sr_npy", direct_summary_path.parent)
    sources["OCRP"] = str(direct_summary_path)

    for method, summary_path in spec["saved_methods"].items():
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing {spec['label']} baseline summary for {method}: {summary_path}")
        summary = _load_summary(summary_path)
        panels[method] = _load_record_array(summary["records"][sample_index], "sr_npy", summary_path.parent)
        sources[method] = str(summary_path)

    panels["HR"] = hr
    sources["HR"] = f"direct summary sample {sample_index}"
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


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate_ocrp_row(path: Path) -> dict:
    rows = _read_csv_rows(path)
    for row in rows:
        if row.get("n_seeds") and row.get("mean_deg_mean"):
            return {
                "method": "OCRP",
                "mean_deg": float(row["mean_deg_mean"]),
                "boundary_f1": float(row["boundary_f1_mean"]),
                "metric_source": str(path),
                "metric_scope": "five-seed aggregate",
            }
    raise ValueError(f"Could not find aggregate OCRP row in {path}")


def _table2_metric_rows(spec: dict) -> list[dict]:
    """Load the same aggregate mean/F1 values used in Table 2."""
    material = spec["label"]
    by_method: dict[str, dict] = {}

    classical_path = EVAL_DIR / "classical_4x4_breakdown_verified_20260707.csv"
    for row in _read_csv_rows(classical_path):
        if row.get("material") != material:
            continue
        method = row["method"]
        by_method[method] = {
            "method": method,
            "mean_deg": float(row["mean_deg"]),
            "boundary_f1": float(row["boundary_f1"]),
            "metric_source": str(classical_path),
            "metric_scope": "test-split aggregate",
        }
    if "Nearest" in by_method:
        by_method["LR"] = {**by_method["Nearest"], "method": "LR"}

    learned_path = EVAL_DIR / ("final_4x4_in718_summary.csv" if spec["tag"] == "in718" else "final_4x4_ti_summary.csv")
    display_alias = {"Q-RBSA": "Q-RBSA-adapted"}
    for row in _read_csv_rows(learned_path):
        method = row["model"]
        if method == "OCRP":
            continue
        display_name = display_alias.get(method, method)
        by_method[display_name] = {
            "method": display_name,
            "mean_deg": float(row["mean_mean"]),
            "boundary_f1": float(row["bf1_mean"]),
            "metric_source": str(learned_path),
            "metric_scope": "five-seed aggregate",
        }

    if spec["tag"] == "in718":
        ocrp_path = EVAL_DIR / "IN718_direct_reynolds_isometric_calibrated_inference_20260706_092117.csv"
    else:
        ocrp_path = EVAL_DIR / "Ti_Al_1pct_direct_reynolds_isometric_calibrated_inference_20260706_092117.csv"
    by_method["OCRP"] = _aggregate_ocrp_row(ocrp_path)

    rows = []
    for row in FIGURE_LAYOUT:
        for name in row:
            if name in ("HR", "IPF key"):
                continue
            if name not in by_method:
                raise KeyError(f"Missing Table 2 metric for {material} panel '{name}'")
            rows.append(by_method[name])
    return rows


def _panel_title(name: str, metric_by_method: dict[str, dict]) -> str:
    if name == "HR":
        return "HR target"
    metric = metric_by_method.get(name)
    if metric is None:
        return name
    return f"{name}\nmean {metric['mean_deg']:.2f} deg, F1 {metric['boundary_f1']:.3f}"


def _draw_panel_grid(
    fig,
    subplot_spec,
    spec: dict,
    panels: OrderedDict[str, np.ndarray],
    sym,
    metric_rows: list[dict],
    *,
    panel_label: str | None = None,
) -> None:
    sample_index = int(spec.get("sample_index", 0))
    metric_by_method = {row["method"]: row for row in metric_rows}
    grid = GridSpecFromSubplotSpec(
        len(FIGURE_LAYOUT) + 1,
        len(FIGURE_LAYOUT[0]),
        subplot_spec=subplot_spec,
        height_ratios=[0.18, 1.0, 1.0, 1.0],
        hspace=0.34,
        wspace=0.055,
    )

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_facecolor("white")
    title_ax.axis("off")
    title = f"{spec['label']} test sample, IPF-Z"
    if panel_label:
        title_ax.text(
            0.0,
            0.55,
            panel_label,
            ha="left",
            va="center",
            fontsize=PANEL_LETTER_FONTSIZE,
            fontweight="bold",
            color="#111111",
        )
        title_ax.text(
            0.028,
            0.55,
            title,
            ha="left",
            va="center",
            fontsize=PANEL_HEADER_FONTSIZE,
            fontweight="semibold",
            color="#111111",
        )
    else:
        title_ax.text(
            0.0,
            0.55,
            title,
            ha="left",
            va="center",
            fontsize=PANEL_HEADER_FONTSIZE,
            fontweight="semibold",
            color="#111111",
        )

    for row_index, row in enumerate(FIGURE_LAYOUT):
        for col_index, name in enumerate(row):
            cell = grid[row_index + 1, col_index]
            if name == "IPF key":
                add_ipf_key_panel(
                    fig,
                    cell,
                    sym,
                    title="IPF-Z key",
                    title_fontsize=IPF_KEY_TITLE_FONTSIZE,
                    label_fontsize=IPF_KEY_LABEL_FONTSIZE,
                    title_height_ratio=0.24,
                )
                continue

            ax = fig.add_subplot(cell)
            if name not in panels:
                ax.axis("off")
                continue
            ax.imshow(_render_ipfz(panels[name], sym), interpolation="nearest")
            ax.set_title(_panel_title(name, metric_by_method), fontsize=PANEL_TITLE_FONTSIZE, pad=4.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                if name == "OCRP":
                    spine.set_linewidth(1.6)
                    spine.set_color("#1b7f3a")
                elif name == "HR":
                    spine.set_linewidth(1.0)
                    spine.set_color("#111111")
                else:
                    spine.set_linewidth(0.45)
                    spine.set_color("0.45")


def _export_figure(
    spec: dict,
    panels: OrderedDict[str, np.ndarray],
    sym,
    metric_rows: list[dict],
) -> Path:
    sample_index = int(spec.get("sample_index", 0))
    out_path = FIG_DIR / f"main_{spec['tag']}_new_learned_baselines_test{sample_index}_4x4.png"
    fig = plt.figure(figsize=(13.2, 8.4), dpi=FIGURE_DPI)
    outer = fig.add_gridspec(1, 1)
    _draw_panel_grid(fig, outer[0, 0], spec, panels, sym, metric_rows)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.985, bottom=0.025)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _export_combined_figure(results: list[dict]) -> Path:
    out_path = FIG_DIR / "main_combined_new_learned_baselines_4x4.png"
    fig = plt.figure(figsize=(13.2, 15.7), dpi=FIGURE_DPI)
    outer = fig.add_gridspec(2, 1, hspace=0.06)
    panel_labels = ["a", "b"]
    for index, result in enumerate(results):
        _draw_panel_grid(
            fig,
            outer[index, 0],
            result["spec"],
            result["panels"],
            result["sym"],
            result["rows"],
            panel_label=panel_labels[index],
        )
    fig.subplots_adjust(left=0.025, right=0.995, top=0.995, bottom=0.016)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in MATERIALS:
        sym = _configure_symmetry(spec["symmetry"])
        sample_index = int(spec.get("sample_index", 0))
        suffix = "" if sample_index == 0 else f"_sample{sample_index}"
        panels, sources = _sample_panels(spec, sym)
        rows = _panel_metrics(panels, sym)
        for row in rows:
            row["material"] = spec["label"]
            row["symmetry"] = spec["symmetry"]
            row["sample_index"] = sample_index
            row["source"] = sources.get(row["method"], "")
        display_rows = _table2_metric_rows(spec)
        for row in display_rows:
            row["material"] = spec["label"]
            row["symmetry"] = spec["symmetry"]
            row["sample_index"] = sample_index
        metrics_path = EVAL_DIR / f"direct_reynolds_oneshot_4x4_{spec['tag']}{suffix}_metrics.csv"
        _write_csv(metrics_path, rows)
        display_metrics_path = EVAL_DIR / f"direct_reynolds_oneshot_4x4_{spec['tag']}{suffix}_figure6_table2_metrics.csv"
        _write_csv(display_metrics_path, display_rows)
        figure_path = _export_figure(spec, panels, sym, display_rows)
        source_path = EVAL_DIR / f"direct_reynolds_oneshot_4x4_{spec['tag']}{suffix}_sources.json"
        source_path.write_text(json.dumps(sources, indent=2) + "\n")
        print(f"Wrote {metrics_path}")
        print(f"Wrote {display_metrics_path}")
        print(f"Wrote {source_path}")
        print(f"Wrote {figure_path}")
        results.append({"spec": spec, "panels": panels, "sym": sym, "rows": display_rows})
    combined_path = _export_combined_figure(results)
    print(f"Wrote {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
