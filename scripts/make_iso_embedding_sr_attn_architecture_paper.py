"""Generate a publication-style architecture diagram for IsoEmbeddingSRAttn."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets"
PNG_PATH = OUTPUT_DIR / "iso_embedding_sr_attn_architecture_paper.png"
SVG_PATH = OUTPUT_DIR / "iso_embedding_sr_attn_architecture_paper.svg"


def _lane(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    face: str = "#f8fafc",
    edge: str = "#d1d5db",
):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
        )
    )
    if title.strip():
        ax.text(
            x + 0.012,
            y + h - 0.026,
            title,
            ha="left",
            va="center",
            fontsize=10,
            color="#111827",
            fontweight="bold",
        )


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = "#ffffff",
    ec: str = "#374151",
    fs: float = 9.0,
    lw: float = 1.1,
):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#111827",
    )


def _arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ls: str = "-",
    lw: float = 1.25,
    color: str = "#1f2937",
):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, linestyle=ls),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(16.6, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Lanes
    _lane(ax, 0.02, 0.56, 0.96, 0.33, "Inference Path")
    _lane(ax, 0.17, 0.31, 0.74, 0.18, "")
    _lane(ax, 0.12, 0.06, 0.76, 0.17, "")

    # Standalone section labels (outside boxes) for cleaner paper layout.
    ax.text(
        0.17,
        0.505,
        "Decoder",
        ha="left",
        va="center",
        fontsize=10,
        color="#111827",
        fontweight="bold",
    )
    ax.text(
        0.12,
        0.245,
        "Training Supervision (Feature Space)",
        ha="left",
        va="center",
        fontsize=10,
        color="#111827",
        fontweight="bold",
    )

    # Inference boxes
    y1, h1, w1 = 0.63, 0.145, 0.102
    x1 = [0.03, 0.146, 0.262, 0.378, 0.494, 0.610, 0.726, 0.842]

    _box(ax, x1[0], y1, w1, h1, "1) LR Quaternion\n(B,4,H,W)\nreshape -> (N,4)", fc="#eef2ff")
    _box(ax, x1[1], y1, w1, h1, "2) Local-Isometric\nInvariant irreps\nEncoder", fc="#ecfeff")
    _box(ax, x1[2], y1, w1, h1, "3) LR Equivariant Conv.\nk=3", fc="#fffbeb")
    _box(ax, x1[3], y1, w1, h1, "4) LR Equivariant Conv.\nk=9", fc="#fffbeb")
    _box(ax, x1[4], y1, w1, h1, "5) Equivariant\nUpsampling\nTransposeConv x r", fc="#fff1f2")
    _box(ax, x1[5], y1, w1, h1, "6) HR Equivariant Conv.\nk=3", fc="#fffbeb")
    _box(ax, x1[6], y1, w1, h1, "7) Block-local\nEquivariant\nAttention", fc="#f5f3ff")
    _box(ax, x1[7], y1, w1, h1, "8) SR Latent\n(B,rH*rW,d)\n(no terminal projection)", fc="#f0fdf4")

    for i in range(len(x1) - 1):
        _arrow(ax, x1[i] + w1, y1 + h1 / 2, x1[i + 1], y1 + h1 / 2)

    # Decoder row
    y2, h2 = 0.345, 0.11
    w2a, w2b = 0.29, 0.23
    x2a, x2b = 0.205, 0.565

    _box(
        ax,
        x2a,
        y2,
        w2a,
        h2,
        "9) Local-Isometric Invariant irreps Decoder\ncubochoric NN seed search + feature refinement\n+ canonicalization",
        fc="#eef2ff",
        fs=8.7,
    )
    _box(
        ax,
        x2b,
        y2,
        w2b,
        h2,
        "10) SR Quaternion Output\n(B,4,rH,rW)\nor (N_hr,4)",
        fc="#ecfeff",
    )

    _arrow(ax, x1[7] + w1 / 2, y1, x2a + w2a / 2, y2 + h2)
    _arrow(ax, x2a + w2a, y2 + h2 / 2, x2b, y2 + h2 / 2)

    # Training supervision row
    y3, h3 = 0.102, 0.092
    _box(ax, 0.15, y3, 0.16, h3, "HR Quaternion\n(B,4,rH,rW)", fc="#eef2ff", fs=8.6)
    _box(ax, 0.35, y3, 0.18, h3, "Local-Isometric\nInvariant irreps Encoder\n(detached target)", fc="#ecfeff", fs=8.6)
    _box(ax, 0.56, y3, 0.17, h3, "MSE Loss in\nlatent feature space", fc="#fef3c7", fs=8.6)

    _arrow(ax, 0.31, y3 + h3 / 2, 0.35, y3 + h3 / 2)
    _arrow(ax, 0.53, y3 + h3 / 2, 0.56, y3 + h3 / 2)
    _arrow(ax, x1[7] + w1 / 2, y1, 0.645, y3 + h3, ls="--")

    # Header
    ax.text(
        0.5,
        0.945,
        "Symmetry-Aware Super-Resolution Network for Orientations",
        ha="center",
        va="center",
        fontsize=15,
        color="#111827",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.912,
        "Local-isometric embedding, equivariant SR backbone, and invariant irreps decoding",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {SVG_PATH}")


if __name__ == "__main__":
    main()
