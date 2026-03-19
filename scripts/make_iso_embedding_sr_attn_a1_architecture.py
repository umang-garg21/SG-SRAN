"""Generate an architecture diagram for the symmetry-active SR model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets"
PNG_PATH = OUTPUT_DIR / "iso_embedding_sr_attn_a1_architecture.png"
SVG_PATH = OUTPUT_DIR / "iso_embedding_sr_attn_a1_architecture.svg"


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = "#f8fbff",
    ec: str = "#1f3a5f",
    fs: float = 9,
    lw: float = 1.4,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        # Keep pad small in axis coordinates to avoid visual overlap.
        boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fs)


def _arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ls: str = "-",
    lw: float = 1.4,
    color: str = "#1f3a5f",
):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, linestyle=ls),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main SR path (symmetry-active features).
    y_top = 0.62
    w = 0.096
    h = 0.145
    x = [0.01, 0.125, 0.24, 0.355, 0.47, 0.585, 0.70, 0.815]

    _box(
        ax,
        x[0],
        y_top,
        w,
        h,
        "Input LR Quaternions\n(B,4,H,W) -> (N,4)",
        fc="#eef6ff",
    )
    _box(
        ax,
        x[1],
        y_top,
        w,
        h,
        "Local-Iso Encoder\nencode_active\n(crystal-aware)",
        fc="#eaf9f1",
    )
    _box(
        ax,
        x[2],
        y_top,
        w,
        h,
        "LR Conv1\nk=3\n(active,active)->active\noptional",
        fc="#fff7e8",
    )
    _box(
        ax,
        x[3],
        y_top,
        w,
        h,
        "LR Conv2\nk=9\n(active,active)->active\noptional",
        fc="#fff7e8",
    )
    _box(
        ax,
        x[4],
        y_top,
        w,
        h,
        "Equivariant\nTransposeConv\nk=3, upsample x r\n(active,active)->active",
        fc="#fff0f0",
    )
    _box(
        ax,
        x[5],
        y_top,
        w,
        h,
        "HR Conv1\nk=3\n(active,active)->active",
        fc="#fff7e8",
    )
    _box(
        ax,
        x[6],
        y_top,
        w,
        h,
        "AttentionBlock x M\n(block-local)\nactive -> active\noptional",
        fc="#f4ecff",
    )
    _box(
        ax,
        x[7],
        y_top,
        w,
        h,
        "SR Feature Field\n(B, rH*rW, d_active)\n(no terminal projection)",
        fc="#f7fbef",
    )

    for i in range(len(x) - 1):
        _arrow(ax, x[i] + w, y_top + h / 2, x[i + 1], y_top + h / 2)

    # Decode path.
    y_mid = 0.34
    w2 = 0.12
    x2 = [0.18, 0.36, 0.54, 0.72]

    _box(
        ax,
        x2[0],
        y_mid,
        w2,
        h,
        "Cubochoric Decoder\n(nearest seeds +\nfeature refinement)\nTarget: active features",
        fc="#f0f7ff",
    )
    _box(
        ax,
        x2[1],
        y_mid,
        w2,
        h,
        "ReduceToFZ\ns^{-1} \u2297 q\nmax |w| selection",
        fc="#f0f7ff",
    )
    _box(
        ax,
        x2[2],
        y_mid,
        w2,
        h,
        "Output SR Quaternions\n(B,4,rH,rW)\nor (N_hr,4)",
        fc="#eef6ff",
    )
    _box(
        ax,
        x2[3],
        y_mid,
        w2,
        h,
        "Symmetry-Canonical\nOrientation Field",
        fc="#eef6ff",
    )

    _arrow(ax, x[7] + w / 2, y_top, x2[0] + w2 / 2, y_mid + h)
    for i in range(len(x2) - 1):
        _arrow(ax, x2[i] + w2, y_mid + h / 2, x2[i + 1], y_mid + h / 2)

    # Training branch (feature-space supervision).
    y_bot = 0.08
    _box(
        ax,
        0.02,
        y_bot,
        0.16,
        0.12,
        "Training branch:\nInput HR Quaternions\n(B,4,rH,rW)",
        fc="#eef6ff",
        fs=8.5,
    )
    _box(
        ax,
        0.22,
        y_bot,
        0.14,
        0.12,
        "encode_active(HR)\nTarget features\n(detached)",
        fc="#eaf9f1",
        fs=8.5,
    )
    _box(
        ax,
        0.50,
        y_bot,
        0.18,
        0.12,
        "MSE Loss in Active Space\nSR features vs\nHR target",
        fc="#fff4cc",
        fs=8.5,
    )

    _arrow(ax, 0.18, y_bot + 0.06, 0.22, y_bot + 0.06)
    _arrow(ax, x[7] + w / 2, y_top, 0.59, y_bot + 0.12, ls="--")
    _arrow(ax, 0.36, y_bot + 0.06, 0.50, y_bot + 0.06)

    ax.text(
        0.5,
        0.94,
        "IsoEmbeddingSRAttn: Symmetry-Active SR Architecture",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#13263f",
    )
    ax.text(
        0.5,
        0.90,
        "Symmetry-active feature pipeline with optional LR conv/attention ablations and geometry-aware decoding",
        ha="center",
        va="center",
        fontsize=10,
        color="#2d4e70",
    )

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {SVG_PATH}")


if __name__ == "__main__":
    main()
