#!/usr/bin/env python3
"""Render high-quality Test sample comparisons for Ti64 DIC McLean 4x4 runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.quat_ops import format_quaternions  # noqa: E402
from utils.symmetry_utils import resolve_symmetry  # noqa: E402
from visualization.ipf_render import add_ipf_key_panel, render_ipf_rgb  # noqa: E402


EXP_ROOT = ROOT / "experiments" / "Ti64_DIC_Mclean_fresh_4x4_full_Train"
SAMPLES = [0, 1, 2, 3]
SYM = resolve_symmetry("D6h")

METHODS = [
    ("Atindama", "atindama_inpainting_4x4_fresh_ti64_dic_mclean"),
    ("EDSR", "edsr_4x4_fresh_ti64_dic_mclean"),
    ("HAN", "han_4x4_fresh_ti64_dic_mclean"),
    ("RCAN", "rcan_4x4_fresh_ti64_dic_mclean"),
    ("SAN", "san_4x4_fresh_ti64_dic_mclean"),
    ("Q-RBSA", "qrbsa_adapted_4x4_fresh_ti64_dic_mclean"),
    ("QEDSR", "qedsr_4x4_fresh_ti64_dic_mclean"),
    ("OCRP w5-local", "ocrp_direct_reynolds_isometric_l6_w5local_4x4_fresh_ti64_dic_mclean"),
]


def qdir(slug: str) -> Path:
    return EXP_ROOT / slug / "inference" / "test_best" / "sr_quaternions"


def sample_path(slug: str, sid: int, kind: str) -> Path:
    return qdir(slug) / f"sample_{sid:06d}_{kind}.npy"


def load_q(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise ValueError(f"Expected HxWx4 quaternion array, got {path}: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def ipfz(q: np.ndarray) -> np.ndarray:
    q_fmt = format_quaternions(
        q,
        normalize=True,
        hemisphere=True,
        reduce_fz=True,
        sym=SYM,
        to_quat_first=False,
    )
    rgb = render_ipf_rgb(q_fmt, SYM, ref_dir="Z")
    return np.asarray(rgb, dtype=np.float32)


def nearest_up_to(q_lr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h_lr, w_lr = q_lr.shape[:2]
    h, w = target_hw
    sy = h // h_lr
    sx = w // w_lr
    if h_lr * sy != h or w_lr * sx != w:
        raise ValueError(f"Cannot integer-upsample LR {q_lr.shape[:2]} to HR {target_hw}")
    return np.repeat(np.repeat(q_lr, sy, axis=0), sx, axis=1)


def panels_for_sample(sid: int) -> list[tuple[str, np.ndarray, bool]]:
    ref_slug = METHODS[0][1]
    lr = load_q(sample_path(ref_slug, sid, "lr"))
    hr = load_q(sample_path(ref_slug, sid, "hr"))
    hr_hw = tuple(hr.shape[:2])
    panels: list[tuple[str, np.ndarray, bool]] = [
        ("LR input", ipfz(nearest_up_to(lr, hr_hw)), False),
    ]
    for label, slug in METHODS:
        sr = load_q(sample_path(slug, sid, "sr"))
        panels.append((label, ipfz(sr), label.startswith("OCRP") or label.startswith("Phase")))
    panels.append(("HR target", ipfz(hr), False))
    return panels


def style_axis(ax, title: str | None, *, highlight: bool = False) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(
            title,
            fontsize=8.2,
            fontweight="semibold" if highlight else "regular",
            pad=3.2,
            color="#111111",
        )


def render_combined() -> Path:
    all_panels = [panels_for_sample(sid) for sid in SAMPLES]
    n_cols = len(all_panels[0])
    fig = plt.figure(figsize=(1.25 * n_cols + 1.15, 1.30 * len(SAMPLES) + 0.55))
    gs = fig.add_gridspec(
        len(SAMPLES),
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.88],
        wspace=0.025,
        hspace=0.075,
    )
    for r, sid in enumerate(SAMPLES):
        for c, (title, rgb, highlight) in enumerate(all_panels[r]):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation="nearest")
            style_axis(ax, title if r == 0 else None, highlight=highlight)
            if c == 0:
                ax.text(
                    -0.055,
                    0.5,
                    f"Test {sid}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=9.0,
                    fontweight="semibold",
                    color="#111111",
                )

    add_ipf_key_panel(
        fig,
        gs[:, -1],
        SYM,
        title="IPF-Z key",
        title_fontsize=8.0,
        label_fontsize=6.8,
    )
    fig.suptitle(
        "Ti64 DIC McLean fresh 4x4 | Test samples 0-3 | IPF-Z comparison",
        fontsize=10.5,
        fontweight="semibold",
        y=0.995,
    )
    out_png = EXP_ROOT / "test_samples_0000_0003_ipfz_model_comparison.png"
    out_pdf = EXP_ROOT / "test_samples_0000_0003_ipfz_model_comparison.pdf"
    fig.savefig(out_png, dpi=420, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return out_png


def render_single(sid: int) -> Path:
    panels = panels_for_sample(sid)
    n_cols = len(panels)
    fig = plt.figure(figsize=(1.55 * n_cols + 1.25, 1.80))
    gs = fig.add_gridspec(1, n_cols + 1, width_ratios=[1.0] * n_cols + [0.85], wspace=0.025)
    for c, (title, rgb, highlight) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation="nearest")
        style_axis(ax, title, highlight=highlight)
    add_ipf_key_panel(
        fig,
        gs[0, -1],
        SYM,
        title="IPF-Z key",
        title_fontsize=8.0,
        label_fontsize=6.8,
    )
    fig.suptitle(
        f"Ti64 DIC McLean fresh 4x4 | Test sample {sid} | IPF-Z comparison",
        fontsize=10.0,
        fontweight="semibold",
        y=1.035,
    )
    out_png = EXP_ROOT / f"test_sample_{sid:06d}_ipfz_model_comparison.png"
    out_pdf = EXP_ROOT / f"test_sample_{sid:06d}_ipfz_model_comparison.pdf"
    fig.savefig(out_png, dpi=420, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return out_png


def main() -> None:
    missing = []
    for sid in SAMPLES:
        ref_slug = METHODS[0][1]
        for kind in ("lr", "hr"):
            p = sample_path(ref_slug, sid, kind)
            if not p.exists():
                missing.append(str(p))
        for _, slug in METHODS:
            p = sample_path(slug, sid, "sr")
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing required arrays:\n" + "\n".join(missing[:40]))

    outputs = [render_combined()]
    outputs.extend(render_single(sid) for sid in SAMPLES)
    metadata = {
        "experiment_root": str(EXP_ROOT),
        "samples": SAMPLES,
        "symmetry": "D6h",
        "rendering": "IPF-Z, quaternion normalized, hemisphere aligned, fundamental-zone reduced before coloring",
        "methods": [{"label": label, "slug": slug} for label, slug in METHODS],
        "outputs": [str(p) for p in outputs],
    }
    meta_path = EXP_ROOT / "test_samples_0000_0003_ipfz_model_comparison_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for path in outputs:
        print(path)
    print(meta_path)


if __name__ == "__main__":
    main()
