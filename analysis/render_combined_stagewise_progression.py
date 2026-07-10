#!/usr/bin/env python3
"""Render the combined FCC/HCP OCRP stagewise decoded-error panel."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Paper/202608_Umang_EBSD_SR_fwd/EBSD_SR_Nature_NMI"
FIG_DIR = PAPER_DIR / "figs"
EVAL_DIR = PAPER_DIR / "evals"

INPUTS = [
    {
        "key": "IN718",
        "title": "IN718 (FCC)",
        "path": EVAL_DIR / "current_direct_reynolds_stagewise_notebook_ckpt_20260708.json",
    },
    {
        "key": "Ti_Al_1pct",
        "title": "Ti-6Al-4V (HCP)",
        "path": EVAL_DIR / "current_direct_reynolds_stagewise_ti_20260707.json",
    },
]

LABELS_BY_STAGE = {
    "encode_lr": "Encoded LR\nfeatures",
    "context_refine": "Context\naggregation",
    "routed_patch": "Routed patch\nsynthesis",
    "hr_refine_1": "HR refine 1",
    "hr_refine_2": "HR refine 2",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 9.0,
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


def load_payload(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["material_key"] = spec["key"]
    payload["material_title"] = spec["title"]
    payload["source_json"] = str(path.relative_to(ROOT))
    return payload


def plot(payloads: list[dict[str, Any]]) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.35, 3.95))
    colors = ["#16213e", "#b24725", "#24705a"]
    markers = ["o", "s", "^"]

    all_y: list[float] = []
    all_err: list[float] = []
    drop_annotations: list[dict[str, Any]] = []
    mean_by_material: list[np.ndarray] = []
    for idx, payload in enumerate(payloads):
        rows = payload["summary"]
        labels = [LABELS_BY_STAGE.get(str(row["stage"]), str(row["label"])) for row in rows]
        means = np.asarray([float(row["mean_deg"]) for row in rows], dtype=np.float64)
        stds = np.asarray([float(row["std_patch_mean_deg"]) for row in rows], dtype=np.float64)
        mean_by_material.append(means)
        xs = np.arange(len(labels))
        stage_indices = {str(row["stage"]): i for i, row in enumerate(rows)}
        encode_idx = stage_indices.get("encode_lr", 0)
        context_idx = stage_indices.get("context_refine", 1)
        routed_idx = stage_indices.get("routed_patch", 2)
        final_idx = stage_indices.get("hr_refine_2", len(rows) - 1)
        offset = (idx - (len(payloads) - 1) / 2.0) * 0.07
        total_drop = means[encode_idx] - means[final_idx]
        upsampler_drop = means[context_idx] - means[routed_idx]
        share_pct = 100.0 * upsampler_drop / total_drop if total_drop > 0 else float("nan")
        drop_annotations.append(
            {
                "text": f"{share_pct:.1f}%",
                "color": colors[idx % len(colors)],
                "y": float(means[routed_idx] + 0.55 * (means[context_idx] - means[routed_idx])),
                "material_key": payload["material_key"],
            }
        )
        all_y.extend(means.tolist())
        all_err.extend(stds.tolist())

        ax.errorbar(
            xs + offset,
            means,
            yerr=stds,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            lw=2.5,
            ms=5.8,
            capsize=3.5,
            zorder=5,
            label=str(payload["material_title"]),
        )

    labels = [LABELS_BY_STAGE.get(str(row["stage"]), str(row["label"])) for row in payloads[0]["summary"]]
    xs = np.arange(len(labels))
    if len(labels) >= 3:
        ax.axvspan(1.5, 2.5, color="#f6d365", alpha=0.22, lw=0)
        ymax_data = float(np.nanmax(np.asarray(all_y) + np.asarray(all_err)))
        ymin_data = float(np.nanmin(np.asarray(all_y) - np.asarray(all_err)))
        span = ymax_data - ymin_data
        for ann in drop_annotations:
            ax.text(
                2.05,
                float(ann["y"]),
                str(ann["text"]),
                color=str(ann["color"]),
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold",
            )
    ymax = float(np.nanmax(np.asarray(all_y) + np.asarray(all_err))) + 0.12
    ymin = max(0.0, float(np.nanmin(np.asarray(all_y) - np.asarray(all_err))) - 0.12)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("OCRP pipeline stage")
    ax.set_ylabel(r"Mean decoded $d_{\mathrm{Stab}}$ ($^\circ$)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.96, edgecolor="#ccd3df", fontsize=8.8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "main_stagewise_progression.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "main_stagewise_progression.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    payloads = [load_payload(spec) for spec in INPUTS]
    plot(payloads)
    upsampler_shares = []
    for payload in payloads:
        rows = payload["summary"]
        means = {str(row["stage"]): float(row["mean_deg"]) for row in rows}
        total_drop = means["encode_lr"] - means["hr_refine_2"]
        upsampler_drop = means["context_refine"] - means["routed_patch"]
        upsampler_shares.append(
            {
                "material_key": payload["material_key"],
                "material_title": payload["material_title"],
                "encode_lr_mean_deg": means["encode_lr"],
                "context_refine_mean_deg": means["context_refine"],
                "routed_patch_mean_deg": means["routed_patch"],
                "hr_refine_2_mean_deg": means["hr_refine_2"],
                "total_encode_to_final_drop_deg": total_drop,
                "upsampler_context_to_routed_drop_deg": upsampler_drop,
                "upsampler_share_of_total_drop_percent": 100.0 * upsampler_drop / total_drop,
            }
        )
    combined = {
        "provenance": {
            "note": "Single-axis direct-Reynolds-isometric OCRP stagewise decoded-error panel for FCC and HCP seed-42 checkpoints.",
            "percentage_annotation": "The plotted percentage is 100 * (context aggregation mean - routed patch mean) / (encode LR mean - final HR refine 2 mean).",
            "upsampler_share_of_total_drop_percent": upsampler_shares,
            "sources": [
                {
                    "material_key": payload["material_key"],
                    "material_title": payload["material_title"],
                    "source_json": payload["source_json"],
                    "checkpoint_path": payload.get("provenance", {}).get("checkpoint_path"),
                    "embedding_mode": payload.get("provenance", {}).get("embedding_mode"),
                    "embedding_metric_calibration": payload.get("provenance", {}).get("embedding_metric_calibration"),
                    "max_harmonic_l": payload.get("provenance", {}).get("max_harmonic_l"),
                }
                for payload in payloads
            ],
        },
        "materials": payloads,
    }
    out_path = EVAL_DIR / "current_direct_reynolds_stagewise_combined_20260707.json"
    out_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {FIG_DIR / 'main_stagewise_progression.pdf'}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
