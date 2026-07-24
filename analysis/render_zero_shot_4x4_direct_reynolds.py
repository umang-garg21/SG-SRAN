#!/usr/bin/env python3
"""Render the main zero-shot 4x4 figure from direct-Reynolds-isometric outputs."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.quat_ops import format_quaternions  # noqa: E402
from utils.symmetry_utils import resolve_symmetry  # noqa: E402
from visualization.ipf_render import add_ipf_key_panel, render_ipf_rgb  # noqa: E402
from visualization.visualize_sr_results import _upsample_lr_rgb_to_hr_nearest  # noqa: E402

FIG_PATH = ROOT / "Paper/EBSD_SR_Nature_v4/figs/figure_zero_shot_4x4.png"
OUT_DIR = ROOT / "Paper/EBSD_SR_Nature_v4/evals/zero_shot_direct_reynolds_4x4_render"


@dataclass(frozen=True)
class ZeroShotRow:
    row_label: str
    sample_index: int
    symmetry: str
    key_title: str
    sr_dir: Path
    source: str


ROWS = [
    ZeroShotRow(
        row_label="CoNi (FCC)\n4x4 zero-shot",
        sample_index=5,
        symmetry="Oh",
        key_title="IPF-Z key\ncubic (m-3m)",
        sr_dir=ROOT
        / "experiments/Zero_shot_performance_CoNi_x250/ocrp_direct_reynolds_isometric_l4_s42"
        / "inference/train_best/sr_quaternions",
        source="IN718 direct-Reynolds-isometric l<=4 checkpoint -> CoNi Scan1_x250",
    ),
    ZeroShotRow(
        row_label="Ti-7 def. (HCP)\n4x4 zero-shot",
        sample_index=8,
        symmetry="D6h",
        key_title="IPF-Z key\nhexagonal (6/mmm)",
        sr_dir=ROOT
        / "experiments/Zero_shot_performance_Ti7_deformed/ocrp_direct_reynolds_isometric_l6_s42"
        / "inference/train_best/sr_quaternions",
        source="Ti-6Al-4V direct-Reynolds-isometric l<=6 checkpoint -> Ti7-deformed",
    ),
]


def _load_quat_triplet(row: ZeroShotRow) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"sample_{row.sample_index:06d}"
    lr = np.load(row.sr_dir / f"{stem}_lr.npy").astype(np.float32, copy=False)
    sr = np.load(row.sr_dir / f"{stem}_sr.npy").astype(np.float32, copy=False)
    hr = np.load(row.sr_dir / f"{stem}_hr.npy").astype(np.float32, copy=False)
    return lr, sr, hr


def _render_triplet(row: ZeroShotRow) -> tuple[list[np.ndarray], dict]:
    lr_q, sr_q, hr_q = _load_quat_triplet(row)
    sym = resolve_symmetry(row.symmetry)

    def fmt(q: np.ndarray) -> np.ndarray:
        return format_quaternions(
            q,
            normalize=True,
            hemisphere=True,
            reduce_fz=True,
            sym=sym,
            to_quat_first=False,
        )

    lr_q = fmt(lr_q)
    sr_q = fmt(sr_q)
    hr_q = fmt(hr_q)

    lr_rgb = render_ipf_rgb(lr_q, sym, ref_dir="Z")
    sr_rgb = render_ipf_rgb(sr_q, sym, ref_dir="Z")
    hr_rgb = render_ipf_rgb(hr_q, sym, ref_dir="Z")
    lr_rgb = _upsample_lr_rgb_to_hr_nearest(lr_rgb, lr_q.shape[:2], hr_q.shape[:2])

    meta = {
        "row_label": row.row_label,
        "sample_index": row.sample_index,
        "symmetry": row.symmetry,
        "source": row.source,
        "sr_dir": str(row.sr_dir),
        "lr_shape": list(lr_q.shape),
        "sr_shape": list(sr_q.shape),
        "hr_shape": list(hr_q.shape),
    }
    return [lr_rgb, sr_rgb, hr_rgb], meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    rendered_rows = []
    metadata = []
    for row in ROWS:
        maps, meta = _render_triplet(row)
        rendered_rows.append((row, maps))
        metadata.append(meta)

    # Pixel-controlled layout: high enough source resolution that IPF keys remain
    # sharp after the wide figure is scaled into the manuscript.
    dpi = 300
    pixels_per_hr_pixel = 5
    panel_h = int(rendered_rows[0][1][0].shape[0]) * pixels_per_hr_pixel
    panel_w = int(rendered_rows[0][1][0].shape[1]) * pixels_per_hr_pixel
    row_label_w = int(round(0.43 * panel_w))
    key_w = int(round(1.12 * panel_w))
    col_gap_w = int(round(0.035 * panel_w))
    header_h = int(round(0.20 * panel_h))
    row_gap_h = int(round(0.055 * panel_h))

    fig_px_w = row_label_w + 3 * panel_w + 2 * col_gap_w + key_w
    fig_px_h = header_h + 2 * panel_h + row_gap_h
    fig = plt.figure(
        figsize=(fig_px_w / dpi, fig_px_h / dpi),
        dpi=dpi,
        facecolor="white",
    )
    gs = fig.add_gridspec(
        4,
        6,
        width_ratios=[row_label_w, panel_w, panel_w, panel_w, col_gap_w, key_w],
        height_ratios=[header_h, panel_h, row_gap_h, panel_h],
        wspace=0.02,
        hspace=0.0,
    )
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.02, top=0.98)

    pt_per_px = 72.0 / dpi
    header_fs = float(0.085 * panel_h * pt_per_px)
    row_fs = float(0.072 * panel_h * pt_per_px)
    key_title_fs = float(0.064 * panel_h * pt_per_px)
    key_label_fs = float(0.045 * panel_h * pt_per_px)

    headers = ["LR (input)", "SR (OCRP)", "HR (ground truth)"]
    for idx, header in enumerate(headers, start=1):
        ax = fig.add_subplot(gs[0, idx])
        ax.axis("off")
        ax.text(
            0.5,
            0.45,
            header,
            ha="center",
            va="center",
            fontsize=header_fs,
            fontweight="bold",
            color="#1f2933",
        )

    for r, (row, maps) in enumerate(rendered_rows):
        gs_row = 1 if r == 0 else 3
        ax_label = fig.add_subplot(gs[gs_row, 0])
        ax_label.axis("off")
        ax_label.text(
            0.52,
            0.5,
            row.row_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=row_fs,
            fontweight="bold",
            color="#1f2933",
            linespacing=1.12,
        )

        for c, img in enumerate(maps, start=1):
            ax = fig.add_subplot(gs[gs_row, c])
            ax.imshow(img, interpolation="nearest", resample=False)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        add_ipf_key_panel(
            fig,
            gs[gs_row, 5],
            resolve_symmetry(row.symmetry),
            title=row.key_title,
            title_fontsize=key_title_fs,
            label_fontsize=key_label_fs,
            title_height_ratio=0.24,
        )

    ax_gap = fig.add_subplot(gs[2, :])
    ax_gap.axis("off")
    ax_gap.axhline(0.5, color="#d8dde3", lw=1.3)

    fig.savefig(FIG_PATH, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "figure": str(FIG_PATH),
        "ref_dir": "Z",
        "rows": metadata,
        "dpi": dpi,
        "pixels_per_hr_pixel": pixels_per_hr_pixel,
        "pixel_shape": [fig_px_h, fig_px_w],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
