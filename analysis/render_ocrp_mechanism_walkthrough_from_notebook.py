#!/usr/bin/env python3
"""Compose the OCRP notebook walkthrough into a compact mechanism figure.

The source panels are provenance-backed notebook outputs from the current
direct-Reynolds-isometric IN718 seed-42 checkpoint.  This script keeps the
scientific content from the notebook but removes the diagnostic-dashboard feel:
the final figure is a staged visual argument from crop context to support-bank
clustering, routing, patch ownership and decoded branch proposals.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis/out/ocrp_4x4_anchorless_direct_reynolds_isometric_visual_walkthrough"
PAPER_DIR = ROOT / "Paper/202608_Umang_EBSD_SR_fwd/EBSD_SR_Nature_NMI"
FIG_DIR = PAPER_DIR / "figs"
OUT_BASE = FIG_DIR / "ocrp_mechanism_walkthrough"
EXPORT_DPI = 900

NMI_FONT = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]


PANELS = [
    {
        "letter": "a",
        "title": "Boundary-rich crop, output and residual",
        "path": OUT_DIR / "stage7_decoded_crop_boundary_fidelity.png",
        "crop": (0, 88, 3543, 875),
    },
    {
        "letter": "b",
        "title": "Crop-level routing follows boundary neighbourhoods",
        "path": OUT_DIR / "07_router_owner_boundary_context.png",
        "crop": (0, 72, 2720, 708),
    },
    {
        "letter": "c",
        "title": "Support-bank clustering in isometric feature space",
        "path": OUT_DIR / "06_selected_ocrp_support_window.png",
        "crop": (0, 105, 2719, 737),
    },
    {
        "letter": "d",
        "title": "Selected 4x4 patch owner map and router margins",
        "path": OUT_DIR / "08b_upsampler_single_window_decision.png",
        "crop": (0, 94, 3621, 734),
    },
    {
        "letter": "e",
        "title": "Decoded slot proposals form discrete local branches",
        "path": OUT_DIR / "08d_upsampler_decoded_patch_proposals.png",
        "crop": (0, 96, 4799, 782),
    },
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": NMI_FONT,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.6,
            "axes.titlesize": 8.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_panel(row: dict[str, object]) -> np.ndarray:
    path = Path(row["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    img = np.asarray(mpimg.imread(path))
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    x0, y0, x1, y1 = row["crop"]  # type: ignore[misc]
    cropped = img[int(y0) : int(y1), int(x0) : int(x1)]
    if cropped.ndim == 2:
        cropped = np.repeat(cropped[..., None], 3, axis=-1)
    if cropped.shape[-1] == 4:
        alpha = cropped[..., 3:4]
        cropped = cropped[..., :3] * alpha + (1.0 - alpha)
    return np.clip(cropped[..., :3], 0.0, 1.0)


def draw_panel(ax: plt.Axes, letter: str, title: str, panel: np.ndarray) -> None:
    ax.imshow(panel, interpolation="nearest")
    ax.set_axis_off()
    ax.text(
        -0.018,
        1.025,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.025,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.1,
        color="#1f2933",
    )
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            linewidth=0.45,
            edgecolor="#c5ccd6",
            clip_on=False,
        )
    )


def main() -> None:
    setup_style()
    panels = {str(item["letter"]): load_panel(item) for item in PANELS}
    panel_meta = {str(item["letter"]): item for item in PANELS}

    a_ratio = panels["a"].shape[0] / panels["a"].shape[1]
    bc_ratio = max(
        panels["b"].shape[0] / panels["b"].shape[1],
        panels["c"].shape[0] / panels["c"].shape[1],
    )
    de_ratio = max(
        panels["d"].shape[0] / panels["d"].shape[1],
        panels["e"].shape[0] / panels["e"].shape[1],
    )

    width = 7.25
    top_row_h = width * a_ratio
    half_width = 0.5 * width
    mid_row_h = half_width * bc_ratio
    bot_row_h = half_width * de_ratio
    height = top_row_h + mid_row_h + bot_row_h + 1.02

    fig = plt.figure(figsize=(width, height), dpi=EXPORT_DPI)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[top_row_h, mid_row_h, bot_row_h],
        hspace=0.28,
        wspace=0.06,
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.982, bottom=0.03)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    for key, ax in [("a", ax_a), ("b", ax_b), ("c", ax_c), ("d", ax_d), ("e", ax_e)]:
        meta = panel_meta[key]
        draw_panel(ax, str(meta["letter"]), str(meta["title"]), panels[key])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_BASE.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"wrote {OUT_BASE.with_suffix('.pdf')}")
    print(f"wrote {OUT_BASE.with_suffix('.png')}")


if __name__ == "__main__":
    main()
