#!/usr/bin/env python3
"""Re-render the cached OCRP routing gallery with journal-safe figure fonts."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "Paper/202607_Umang_EBSD_SR_paper__NMI/EBSD_SR_Nature_v4/figs"
OUT_BASE = FIG_DIR / "ocrp_routing_gallery"
NMI_FONT = ["Nimbus Sans", "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": NMI_FONT,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )


def render_gallery(
    owner_map: np.ndarray,
    confidence_map: np.ndarray,
    margin_map: np.ndarray,
    active_map: np.ndarray,
    kmax: int,
    out_base: Path,
) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.7), constrained_layout=True)

    owner_colors = plt.get_cmap("tab10", kmax)(np.arange(kmax))
    owner_cmap = ListedColormap(owner_colors)
    owner_norm = BoundaryNorm(np.arange(-0.5, kmax + 0.5, 1.0), kmax)
    active_cmap = ListedColormap(
        ["#2c115f", "#2bb07f", "#fde725", "#fdae61", "#d7191c", "#7f3b08"][:kmax]
    )
    active_norm = BoundaryNorm(np.arange(0.5, kmax + 1.5, 1.0), kmax)

    panels = [
        (owner_map, "Routed owner slot", owner_cmap, owner_norm, list(range(kmax))),
        (confidence_map, "Owner confidence", "viridis", None, None),
        (margin_map, "Top-1 minus top-2 margin", "magma", None, None),
        (active_map, "Active slots per LR site", active_cmap, active_norm, list(range(1, kmax + 1))),
    ]
    for ax, (data, title, cmap, norm, ticks) in zip(axes.ravel(), panels, strict=True):
        image = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(length=2.2, width=0.4, pad=1.5)
        if ticks is not None:
            cbar.set_ticks(ticks)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=600)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    cached = np.load(OUT_BASE.with_suffix(".npz"))
    kmax = max(int(np.max(cached["owner_map"])) + 1, int(np.max(cached["active_map"])))
    render_gallery(
        owner_map=cached["owner_map"],
        confidence_map=cached["confidence_map"],
        margin_map=cached["margin_map"],
        active_map=cached["active_map"],
        kmax=kmax,
        out_base=OUT_BASE,
    )
    print(f"wrote {OUT_BASE.with_suffix('.png')} and {OUT_BASE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
