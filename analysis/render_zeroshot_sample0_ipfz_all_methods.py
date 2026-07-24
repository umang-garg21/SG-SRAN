#!/usr/bin/env python3
"""Render sample-0 IPF-Z sanity panels for all zero-shot 4x4 methods."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "Paper/EBSD_SR_Nature_v4/evals"
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_anchorless_test_metrics as anchor_eval  # noqa: E402
from utils.quat_ops import format_quaternions  # noqa: E402
from utils.symmetry_utils import proper_symmetry_quaternions, resolve_symmetry  # noqa: E402
from visualization.ipf_render import add_ipf_key_panel, render_ipf_rgb  # noqa: E402


@dataclass(frozen=True)
class ZeroShotDataset:
    key: str
    title: str
    split: str
    symmetry: str
    root: Path
    ocrp_sr_dir: Path


DATASETS = [
    ZeroShotDataset(
        key="coni",
        title="CoNi zero-shot",
        split="Train split",
        symmetry="Oh",
        root=ROOT / "experiments/Zero_shot_performance_CoNi_x250",
        ocrp_sr_dir=ROOT
        / "experiments/Zero_shot_performance_CoNi_x250"
        / "ocrp_direct_reynolds_isometric_l4_s42/inference/train_best/sr_quaternions",
    ),
    ZeroShotDataset(
        key="ti7",
        title="Ti7-deformed zero-shot",
        split="Train split",
        symmetry="D6h",
        root=ROOT / "experiments/Zero_shot_performance_Ti7_deformed",
        ocrp_sr_dir=ROOT
        / "experiments/Zero_shot_performance_Ti7_deformed"
        / "ocrp_direct_reynolds_isometric_l6_s42/inference/train_best/sr_quaternions",
    ),
    ZeroShotDataset(
        key="ti64",
        title="Ti64 zero-shot",
        split="Test split",
        symmetry="D6h",
        root=ROOT / "experiments/Zero_shot_performance_Ti64_DIC_Mclean",
        ocrp_sr_dir=ROOT
        / "experiments/Zero_shot_performance_Ti64_DIC_Mclean"
        / "ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions",
    ),
]


LEARNED_METHODS = OrderedDict(
    [
        ("Atindama", "atindama_inpainting"),
        ("EDSR", "edsr"),
        ("QEDSR", "qedsr"),
        ("Q-RBSA", "qrbsaadapted"),
        ("RCAN", "rcan"),
        ("SAN", "san"),
        ("HAN", "han"),
    ]
)

CLASSICAL_METHODS = ("Nearest", "Bicubic", "SLERP", "Symm-SLERP")
PANEL_ORDER = (
    ("LR input", "lr"),
    *((name, "classical") for name in CLASSICAL_METHODS),
    *((name, "learned") for name in LEARNED_METHODS),
    ("OCRP", "ocrp"),
    ("HR target", "hr"),
)


def _hwc4(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 3:
        raise ValueError(f"Expected rank-3 quaternion array, got {q.shape}")
    if q.shape[-1] == 4:
        return q
    if q.shape[0] == 4:
        return np.moveaxis(q, 0, -1).astype(np.float32, copy=False)
    raise ValueError(f"No quaternion axis found in {q.shape}")


def _load_quat(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return _hwc4(np.load(path))


def _format_ipfz(q: np.ndarray, sym_class) -> np.ndarray:
    q = format_quaternions(
        _hwc4(q),
        normalize=True,
        hemisphere=True,
        reduce_fz=True,
        sym=sym_class,
        to_quat_first=False,
    )
    return render_ipf_rgb(q, sym_class, ref_dir="Z")


def _nearest_rgb_to_hw(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    th, tw = int(target_hw[0]), int(target_hw[1])
    if th % h != 0 or tw % w != 0:
        raise ValueError(f"Cannot nearest-expand RGB shape {(h, w)} to {(th, tw)}")
    return np.repeat(np.repeat(rgb, th // h, axis=0), tw // w, axis=1)


def _configure_classical_symmetry(symmetry_name: str) -> None:
    sym = resolve_symmetry(symmetry_name)
    anchor_eval.SYM = sym
    anchor_eval.SYM_QUATS = proper_symmetry_quaternions(sym)
    anchor_eval._SLERP_SYM_OPS_4X4 = anchor_eval.make_symmetry_4x4(
        symmetry_name,
        device="cpu",
        dtype=torch.float32,
    )


def _classical_sr(method: str, lr: np.ndarray, hr_hw: tuple[int, int]) -> np.ndarray:
    if method == "Nearest":
        return anchor_eval.upsample_nn(lr, hr_hw)
    if method == "Bicubic":
        return anchor_eval.upsample_bicubic(lr, hr_hw)
    if method == "SLERP":
        return anchor_eval.upsample_slerp(lr, hr_hw)
    if method == "Symm-SLERP":
        return anchor_eval.upsample_symm_slerp(lr, hr_hw)
    raise KeyError(method)


def _sample_stem(sample_id: int) -> str:
    return f"sample_{int(sample_id):06d}"


def _render_dataset(
    spec: ZeroShotDataset,
    sample_id: int,
    out_dir: Path,
    dpi: int,
) -> tuple[Path, list[tuple[str, np.ndarray]], dict]:
    stem = _sample_stem(sample_id)
    sym = resolve_symmetry(spec.symmetry)
    _configure_classical_symmetry(spec.symmetry)

    lr = _load_quat(spec.ocrp_sr_dir / f"{stem}_lr.npy")
    hr = _load_quat(spec.ocrp_sr_dir / f"{stem}_hr.npy")
    hr_hw = tuple(int(x) for x in hr.shape[:2])
    lr_rgb = _nearest_rgb_to_hw(_format_ipfz(lr, sym), hr_hw)

    panels: list[tuple[str, np.ndarray]] = [("LR input", lr_rgb)]
    sources: dict[str, str] = {
        "LR input": str(spec.ocrp_sr_dir / f"{stem}_lr.npy"),
        "HR target": str(spec.ocrp_sr_dir / f"{stem}_hr.npy"),
    }
    shapes: dict[str, list[int]] = {
        "LR input": list(lr.shape),
        "HR target": list(hr.shape),
    }

    for method in CLASSICAL_METHODS:
        sr = _classical_sr(method, lr, hr_hw)
        panels.append((method, _format_ipfz(sr, sym)))
        sources[method] = "computed from LR sample at render time"
        shapes[method] = list(sr.shape)

    learned_root = spec.root / "learned_baselines_4x4"
    for method, slug in LEARNED_METHODS.items():
        sr_path = learned_root / slug / "sr_quaternions" / f"{stem}_sr.npy"
        sr = _load_quat(sr_path)
        if tuple(sr.shape[:2]) != hr_hw:
            raise ValueError(f"{method} shape {sr.shape} does not match HR {hr.shape}")
        panels.append((method, _format_ipfz(sr, sym)))
        sources[method] = str(sr_path)
        shapes[method] = list(sr.shape)

    ocrp_path = spec.ocrp_sr_dir / f"{stem}_sr.npy"
    ocrp = _load_quat(ocrp_path)
    panels.append(("OCRP", _format_ipfz(ocrp, sym)))
    sources["OCRP"] = str(ocrp_path)
    shapes["OCRP"] = list(ocrp.shape)

    panels.append(("HR target", _format_ipfz(hr, sym)))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{spec.key}_{stem}_ipfz_all_methods.png"
    n_panels = len(panels)
    fig = plt.figure(figsize=(1.75 * n_panels + 2.5, 3.15), dpi=dpi)
    gs = fig.add_gridspec(1, n_panels + 1, width_ratios=[1] * n_panels + [0.9], wspace=0.05)

    for idx, (label, rgb) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(rgb, interpolation="nearest", resample=False)
        ax.set_aspect("equal")
        ax.axis("off")
        weight = "bold" if label in {"OCRP", "HR target"} else "normal"
        ax.set_title(label, fontsize=8.2, fontweight=weight, pad=5)

    add_ipf_key_panel(
        fig,
        gs[0, -1],
        sym,
        title=f"IPF-Z key\n{spec.symmetry}",
        title_fontsize=8.2,
        label_fontsize=7.0,
    )
    fig.suptitle(f"{spec.title} ({spec.split}), {stem}: IPF-Z all methods", fontsize=10.5, y=1.02)
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    meta = {
        "dataset": spec.key,
        "title": spec.title,
        "split": spec.split,
        "sample_id": int(sample_id),
        "symmetry": spec.symmetry,
        "figure": str(out_png),
        "ocrp_sr_dir": str(spec.ocrp_sr_dir),
        "hr_shape": list(hr.shape),
        "lr_shape": list(lr.shape),
        "panel_order": [label for label, _ in panels],
        "sources": sources,
        "shapes": shapes,
    }
    return out_png, panels, meta


def _render_combined(
    rendered: list[tuple[ZeroShotDataset, list[tuple[str, np.ndarray]]]],
    sample_id: int,
    out_dir: Path,
    dpi: int,
) -> Path:
    stem = _sample_stem(sample_id)
    n_rows = len(rendered)
    n_panels = len(rendered[0][1])
    out_png = out_dir / f"combined_{stem}_ipfz_all_zeroshot_datasets_all_methods.png"

    fig = plt.figure(figsize=(1.42 * n_panels + 3.2, 2.55 * n_rows), dpi=dpi)
    gs = fig.add_gridspec(
        n_rows,
        n_panels + 2,
        width_ratios=[0.55] + [1] * n_panels + [1.05],
        wspace=0.05,
        hspace=0.18,
    )

    for row_idx, (spec, panels) in enumerate(rendered):
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.axis("off")
        ax_label.text(
            0.5,
            0.5,
            f"{spec.title}\n{spec.split}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.4,
            fontweight="bold",
        )
        for col_idx, (label, rgb) in enumerate(panels, start=1):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(rgb, interpolation="nearest", resample=False)
            ax.set_aspect("equal")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(label, fontsize=7.5, pad=4)
        add_ipf_key_panel(
            fig,
            gs[row_idx, -1],
            resolve_symmetry(spec.symmetry),
            title=f"IPF-Z key\n{spec.symmetry}",
            title_fontsize=7.5,
            label_fontsize=6.4,
        )

    fig.suptitle(f"Zero-shot sample {sample_id}: IPF-Z sanity check across all methods", fontsize=11.0, y=0.995)
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis/out/zeroshot_sample0_ipfz_sanity",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[spec.key for spec in DATASETS],
        choices=[spec.key for spec in DATASETS],
    )
    args = parser.parse_args()

    selected = {spec.key: spec for spec in DATASETS}
    rendered: list[tuple[ZeroShotDataset, list[tuple[str, np.ndarray]]]] = []
    metadata: list[dict] = []
    figures: list[str] = []
    for key in args.datasets:
        spec = selected[key]
        out_png, panels, meta = _render_dataset(spec, args.sample_id, args.out_dir, args.dpi)
        rendered.append((spec, panels))
        metadata.append(meta)
        figures.append(str(out_png))
        print(f"saved {out_png}", flush=True)

    combined = _render_combined(rendered, args.sample_id, args.out_dir, args.dpi)
    figures.append(str(combined))
    print(f"saved {combined}", flush=True)

    summary = {
        "sample_id": int(args.sample_id),
        "ref_dir": "Z",
        "quaternion_display_convention": "passive scalar-first wxyz arrays; normalized, hemisphere-aligned, symmetry-FZ-reduced before IPF-Z rendering",
        "classical_methods": list(CLASSICAL_METHODS),
        "learned_methods": list(LEARNED_METHODS.keys()),
        "figures": figures,
        "datasets": metadata,
    }
    summary_path = args.out_dir / f"sample_{args.sample_id:06d}_ipfz_all_methods_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
