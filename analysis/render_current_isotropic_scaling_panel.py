#!/usr/bin/env python3
"""Render the corrected direct-Reynolds OCRP isotropic scaling panel.

The panel is intentionally strict: every listed method must have seed-42
summaries for 2x2, 4x4, and 8x8 on both IN718 and Ti-Al before the figure is
written.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_EVAL_DIR = ROOT / "Paper/EBSD_SR_Nature_v4/evals"
for path in (ROOT, LEGACY_EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_anchorless_test_metrics as ev  # noqa: E402
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry  # noqa: E402


PAPER_DIR = ROOT / "Paper/202608_Umang_EBSD_SR_fwd/EBSD_SR_Nature_NMI"
FIG_DIR = PAPER_DIR / "figs"
EVAL_DIR = PAPER_DIR / "evals"

SUMMARY_PATHS = {
    "IN718": {
        "label": "IN718 (FCC)",
        "symmetry": "Oh",
        "scales": {
            "2x2": ROOT
            / "experiments/direct_reynolds_isometric_scaling/IN718/"
            / "ocrp_direct_reynolds_isometric_l4_x2_s42/inference/test_best/summary.json",
            "4x4": ROOT
            / "experiments/IN718/direct_reynolds_isometric_seed_runs/"
            / "ocrp_direct_reynolds_isometric_l4_s42/inference/test_best/summary.json",
            "8x8": ROOT
            / "experiments/direct_reynolds_isometric_scaling/IN718/"
            / "ocrp_direct_reynolds_isometric_l4_x8_s42/inference/test_best/summary.json",
        },
    },
    "Ti_Al_1pct": {
        "label": "Ti-6Al-4V (HCP)",
        "symmetry": "D6h",
        "scales": {
            "2x2": ROOT
            / "experiments/direct_reynolds_isometric_scaling/Ti_Al_1pct/"
            / "ocrp_direct_reynolds_isometric_l6_x2_s42/inference/test_best/summary.json",
            "4x4": ROOT
            / "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs/"
            / "ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/summary.json",
            "8x8": ROOT
            / "experiments/direct_reynolds_isometric_scaling/Ti_Al_1pct/"
            / "ocrp_direct_reynolds_isometric_l6_x8_s42/inference/test_best/summary.json",
        },
    },
}

LEARNED_METHOD_SLUGS = {
    "Atindama inpainting": "atindama",
    "Q-RBSA-adapted": "qrbsa",
    "QEDSR": "qedsr",
    "EDSR": "edsr",
    "RCAN": "rcan",
    "SAN": "san",
    "HAN": "han",
}


def learned_summary_path(material: str, method: str, scale: str) -> Path:
    slug = LEARNED_METHOD_SLUGS[method]
    if scale == "4x4":
        return ROOT / f"experiments/{material}/seed_runs/{slug}_4x4_s42/inference/test/summary.json"
    scale_num = scale[0]
    return (
        ROOT
        / "experiments/direct_reynolds_isometric_scaling"
        / material
        / f"{slug}_x{scale_num}_s42/inference/test_best/summary.json"
    )

FULL_SCALE_METHODS = ["Nearest", "Bicubic", "SLERP", "Symm-SLERP", "OCRP"]
LEARNED_METHODS = ["Atindama inpainting", "Q-RBSA-adapted", "QEDSR", "EDSR", "RCAN", "SAN", "HAN"]
METHOD_ORDER = FULL_SCALE_METHODS[:-1] + LEARNED_METHODS + ["OCRP"]
METHOD_STYLES = {
    "OCRP": {"label": "OCRP", "color": "#111827", "lw": 2.8, "marker": "o", "ms": 6.0, "zorder": 20, "ls": "-"},
    "Nearest": {"label": "Nearest", "color": "#7a8797", "lw": 1.35, "marker": "s", "ms": 4.4, "zorder": 3, "ls": "--"},
    "Bicubic": {"label": "Bicubic", "color": "#c98230", "lw": 1.35, "marker": "^", "ms": 4.4, "zorder": 3, "ls": "--"},
    "SLERP": {"label": "SLERP", "color": "#5f9f75", "lw": 1.35, "marker": "D", "ms": 4.4, "zorder": 3, "ls": "--"},
    "Symm-SLERP": {"label": "Symm-SLERP", "color": "#b65f5f", "lw": 1.35, "marker": "P", "ms": 4.6, "zorder": 3, "ls": "--"},
    "Atindama inpainting": {"label": "Atindama", "color": "#8b5cf6", "lw": 1.05, "marker": "X", "ms": 5.2, "zorder": 12, "ls": ":"},
    "Q-RBSA-adapted": {"label": "Q-RBSA", "color": "#0891b2", "lw": 1.05, "marker": "v", "ms": 5.2, "zorder": 12, "ls": ":"},
    "QEDSR": {"label": "QEDSR", "color": "#2563eb", "lw": 1.05, "marker": "<", "ms": 5.2, "zorder": 12, "ls": ":"},
    "EDSR": {"label": "EDSR", "color": "#d97706", "lw": 1.05, "marker": ">", "ms": 5.2, "zorder": 12, "ls": ":"},
    "RCAN": {"label": "RCAN", "color": "#db2777", "lw": 1.05, "marker": "h", "ms": 5.2, "zorder": 12, "ls": ":"},
    "SAN": {"label": "SAN", "color": "#16a34a", "lw": 1.05, "marker": "*", "ms": 6.5, "zorder": 12, "ls": ":"},
    "HAN": {"label": "HAN", "color": "#9333ea", "lw": 1.05, "marker": "p", "ms": 5.2, "zorder": 12, "ls": ":"},
}

NMI_FONT = ["Nimbus Sans", "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": NMI_FONT,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d7dbe2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "axes.facecolor": "#fcfcfd",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bind_symmetry(symmetry: str) -> None:
    ev.SYM = resolve_symmetry(symmetry)
    ev.SYM_QUATS = proper_symmetry_quaternions(ev.SYM)
    ev._SLERP_SYM_OPS_4X4 = ev.make_symmetry_4x4(symmetry, device="cpu", dtype=ev.torch.float32)


def provider_for(method: str):
    if method == "OCRP":
        return ev.provider_from_saved_sr
    if method == "Nearest":
        return ev.provider_from_upsampler(ev.upsample_nn)
    if method == "Bicubic":
        return ev.provider_from_upsampler(ev.upsample_bicubic)
    if method == "SLERP":
        return ev.provider_from_upsampler(ev.upsample_slerp)
    if method == "Symm-SLERP":
        return ev.provider_from_upsampler(ev.upsample_symm_slerp)
    if method in LEARNED_METHODS:
        return ev.provider_from_saved_sr
    raise KeyError(method)


def evaluate_all() -> list[dict]:
    missing = []
    for spec in SUMMARY_PATHS.values():
        for path in spec["scales"].values():
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
    for material in SUMMARY_PATHS:
        for scale in ("2x2", "4x4", "8x8"):
            for method in LEARNED_METHODS:
                path = learned_summary_path(material, method, scale)
                if not path.exists():
                    missing.append(str(path.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError("Missing scaling/learned-baseline summaries:\n" + "\n".join(missing))

    jobs = []
    for material, spec in SUMMARY_PATHS.items():
        for scale, summary_path in spec["scales"].items():
            for method in FULL_SCALE_METHODS:
                jobs.append((material, dict(spec), scale, method, summary_path, "full_scale"))
            for method in LEARNED_METHODS:
                jobs.append((material, dict(spec), scale, method, learned_summary_path(material, method, scale), "scale_specific"))
    max_workers = min(len(jobs), int(os.environ.get("QSR_SCALING_WORKERS", "6")))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(evaluate_material_scale_method, *job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (str(row["material"]), str(row["scale"]), METHOD_ORDER.index(str(row["method"]))))
    expected_rows = len(SUMMARY_PATHS) * 3 * len(METHOD_ORDER)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} scaling rows, got {len(rows)}")
    return rows


def evaluate_material_scale_method(
    material: str,
    spec: dict,
    scale: str,
    method: str,
    summary_path: Path,
    protocol: str,
) -> dict:
    bind_symmetry(str(spec["symmetry"]))
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    summary["task"] = f"{material} isotropic {scale}"
    print(f"Evaluating {material} {scale} {method}", flush=True)
    row, _ = ev.evaluate_method(summary, method, provider_for(method), compute_scalars=False)
    row["material"] = material
    row["scale"] = scale
    row["summary"] = str(Path(summary_path).relative_to(ROOT))
    row["protocol"] = protocol
    return row


def plot(rows: list[dict]) -> None:
    setup_style()
    data = {
        (row["material"], row["scale"], row["method"]): row
        for row in rows
    }
    x_labels = ["2x2", "4x4", "8x8"]
    x = np.arange(len(x_labels))
    metrics = [
        ("mis_mean_deg", r"Mean $d_{\mathrm{Stab}}$ ($^\circ$)"),
        ("boundary_f1", "Boundary F1"),
    ]
    materials = [("IN718", "IN718 (FCC)"), ("Ti_Al_1pct", "Ti-6Al-4V (HCP)")]

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.55), sharex="col")
    for col, (material, title) in enumerate(materials):
        axes[0, col].set_title(title)
        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col]
            for method in METHOD_ORDER:
                vals = [
                    float(data[(material, scale, method)][metric])
                    if (material, scale, method) in data
                    else np.nan
                    for scale in x_labels
                ]
                if not np.isfinite(vals).any():
                    continue
                style = METHOD_STYLES[method]
                ax.plot(
                    x,
                    vals,
                    color=style["color"],
                    lw=style["lw"],
                    ls=style["ls"],
                    marker=style["marker"],
                    ms=style["ms"],
                    label=style["label"] if row_idx == 0 and col == 0 else None,
                    zorder=style["zorder"],
                    markeredgecolor="white" if method in LEARNED_METHODS else style["color"],
                    markeredgewidth=0.45 if method in LEARNED_METHODS else 0.0,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("Isotropic upsampling factor" if row_idx == 1 else "")
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.margins(x=0.08)
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylim(0.0, 1.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.03))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(FIG_DIR / "main_isotropic_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "main_isotropic_scaling.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = evaluate_all()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(EVAL_DIR / "current_isotropic_scaling_20260707.csv", rows)
    payload = {
        "provenance": {
            "note": "Corrected seed-42 isotropic scaling panel using IN718 and Ti-Al summaries for 2x2, 4x4, and 8x8.",
            "learned_baseline_note": "All learned baselines use scale-specific seed-42 predictions; OCRP uses the corrected direct-Reynolds-isometric router.",
            "methods": METHOD_ORDER,
        },
        "rows": rows,
    }
    (EVAL_DIR / "current_isotropic_scaling_20260707.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(rows)
    print(f"wrote {FIG_DIR / 'main_isotropic_scaling.pdf'}")
    print(f"wrote {EVAL_DIR / 'current_isotropic_scaling_20260707.csv'}")


if __name__ == "__main__":
    main()
