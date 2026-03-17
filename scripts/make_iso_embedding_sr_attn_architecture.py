"""Generate an architecture diagram for IsoEmbeddingSRAttn."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets"
PNG_PATH = OUTPUT_DIR / "iso_embedding_sr_attn_architecture.png"
SVG_PATH = OUTPUT_DIR / "iso_embedding_sr_attn_architecture.svg"


def _box(ax, x, y, w, h, text, fc="#f8fbff", ec="#1f3a5f", fs=9, lw=1.4):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fs)
    return patch


def _arrow(ax, x0, y0, x1, y1, ls="-", lw=1.4, color="#1f3a5f"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, linestyle=ls),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main SR/inference path.
    y = 0.62
    w = 0.11
    h = 0.14
    x = [0.02, 0.15, 0.28, 0.41, 0.54, 0.67, 0.80]

    _box(
        ax,
        x[0],
        y,
        w,
        h,
        "Input LR Quaternions\n(B,4,H,W)\n-> (N,4)",
        fc="#eef6ff",
    )
    _box(
        ax,
        x[1],
        y,
        w,
        h,
        "encode_a1\nLocalIso encoder\nirreps_a1",
        fc="#eaf9f1",
    )
    _box(
        ax,
        x[2],
        y,
        w,
        h,
        "LR Conv1\nk=3\n(a1,a1)->full",
        fc="#fff7e8",
    )
    _box(
        ax,
        x[3],
        y,
        w,
        h,
        "LR Conv2\nk=9\n(full,full)->full",
        fc="#fff7e8",
    )
    _box(
        ax,
        x[4],
        y,
        w,
        h,
        "Equivariant\nTransposeConv\nk=3, upsample x r\n(full,full)->full",
        fc="#fff0f0",
    )
    _box(
        ax,
        x[5],
        y,
        w,
        h,
        "HR Conv1\nk=3\n(full,full)->full",
        fc="#fff7e8",
    )
    _box(
        ax,
        x[6],
        y,
        w,
        h,
        "AttentionBlock x M\n(block-local)\nfull -> full",
        fc="#f4ecff",
    )

    y2 = 0.34
    x2 = [0.24, 0.41, 0.58, 0.75]
    _box(
        ax,
        x2[0],
        y2,
        w,
        h,
        "Final Linear\nfull -> a1\n(output irreps_a1)",
        fc="#f7fbef",
    )
    _box(
        ax,
        x2[1],
        y2,
        w,
        h,
        "Cubochoric Decoder\n(nearest seeds +\nfeature optimization)\nTarget: a1",
        fc="#f0f7ff",
    )
    _box(
        ax,
        x2[2],
        y2,
        w,
        h,
        "ReduceToFZ\nby crystal\nsymmetry ops",
        fc="#f0f7ff",
    )
    _box(
        ax,
        x2[3],
        y2,
        w,
        h,
        "Output SR Quaternions\n(B,4,rH,rW)\n(or N_hr,4)",
        fc="#eef6ff",
    )

    # Training supervision branch.
    y3 = 0.08
    _box(
        ax,
        0.02,
        y3,
        0.16,
        0.12,
        "Training branch:\nInput HR Quaternions\n(B,4,rH,rW)",
        fc="#eef6ff",
        fs=8.5,
    )
    _box(
        ax,
        0.23,
        y3,
        0.13,
        0.12,
        "encode_a1(HR)\n-> target a1\n(detached)",
        fc="#eaf9f1",
        fs=8.5,
    )
    _box(
        ax,
        0.52,
        y3,
        0.14,
        0.12,
        "MSE loss in a1\nfeature space\npred vs target",
        fc="#fff4cc",
        fs=8.5,
    )

    # Main arrows top row.
    for i in range(len(x) - 1):
        _arrow(ax, x[i] + w, y + h / 2, x[i + 1], y + h / 2)

    # Down to second row.
    _arrow(ax, x[6] + w / 2, y, x2[0] + w / 2, y2 + h)
    for i in range(len(x2) - 1):
        _arrow(ax, x2[i] + w, y2 + h / 2, x2[i + 1], y2 + h / 2)

    # Training branch arrows.
    _arrow(ax, 0.18, y3 + 0.06, 0.23, y3 + 0.06)
    _arrow(ax, x2[0] + w / 2, y2, 0.59, y3 + 0.12, ls="--")
    _arrow(ax, 0.36, y3 + 0.06, 0.52, y3 + 0.06)

    ax.text(
        0.5,
        0.94,
        "IsoEmbeddingSRAttn Architecture (LocalIsoEmbedding-Centered)",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#13263f",
    )
    ax.text(
        0.5,
        0.90,
        "No lift layer, no HR Conv2, final projection to irreps_a1 before decoder",
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

