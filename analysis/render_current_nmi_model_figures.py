#!/usr/bin/env python3
"""Render current direct-Reynolds OCRP figures used by the NMI manuscript."""
from __future__ import annotations

import json
import os
import sys
import inspect
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.infer_iso_embedding_sr_attn import (  # noqa: E402
    _flatten_quat_chw,
    _load_model_from_checkpoint,
    _resolve_checkpoint,
)
from training.config_utils import load_and_prepare_config  # noqa: E402
from utils.quat_ops import format_quaternions  # noqa: E402
from utils.symmetry_utils import resolve_symmetry  # noqa: E402
from visualization.ipf_render import add_ipf_key_panel, render_ipf_rgb  # noqa: E402

NMI_FONT = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]

PAPER_DIR = ROOT / "Paper/202608_Umang_EBSD_SR_fwd/EBSD_SR_Nature_NMI"
FIG_DIR = PAPER_DIR / "figs"
EVAL_DIR = PAPER_DIR / "evals"

OPEN718_FCC_SPEC = {
    "name": "open718_current_direct_reynolds_seed42",
    "symmetry": "Oh",
    "scale": 4,
    "hr_source": ROOT / "Paper/EBSD_SR_Nature_v4/Open_718_Test_hr_x_block_0.npy",
    "exp_dir": ROOT
    / "experiments/IN718/direct_reynolds_isometric_seed_runs/"
    / "ocrp_direct_reynolds_isometric_l4_s42",
    "config": "config_new.json",
    "checkpoint": "best_model.pt",
    "out_png": FIG_DIR / "main_4x4_val_x230_lr_sr_hr_ipf.png",
    "reuse_cached_triplet": True,
}

SAVED_TRIPTYCHS = [
    {
        "name": "ti_current_direct_reynolds_test7",
        "symmetry": "D6h",
        "sample_id": 7,
        "sr_dir": ROOT
        / "experiments/Ti_Al_1pct/direct_reynolds_isometric_seed_runs/"
        / "ocrp_direct_reynolds_isometric_l6_s42/inference/test_best/sr_quaternions",
        "out_png": FIG_DIR / "main_4x4_TiAl_lr_sr_hr_ipf.png",
    },
]


def _load_triplet(sr_dir: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"sample_{sample_id:06d}"
    lr = np.load(sr_dir / f"{stem}_lr.npy").astype(np.float32, copy=False)
    sr = np.load(sr_dir / f"{stem}_sr.npy").astype(np.float32, copy=False)
    hr = np.load(sr_dir / f"{stem}_hr.npy").astype(np.float32, copy=False)
    return lr, sr, hr


def _normalize_quat_hwc(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return (q / np.maximum(norm, eps)).astype(np.float32, copy=False)


def _crop_triplet_to_square(
    lr: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    *,
    scale: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Center-crop HR/SR to the largest scale-aligned square and crop LR consistently."""
    scale = int(scale)
    h_hr, w_hr = hr.shape[:2]
    side = (min(h_hr, w_hr) // scale) * scale
    if side <= 0:
        raise ValueError(f"Cannot square-crop HR shape {(h_hr, w_hr)} with scale {scale}")
    y0 = ((h_hr - side) // 2 // scale) * scale
    x0 = ((w_hr - side) // 2 // scale) * scale
    y1 = y0 + side
    x1 = x0 + side
    ly0, lx0 = y0 // scale, x0 // scale
    lside = side // scale
    return (
        lr[ly0 : ly0 + lside, lx0 : lx0 + lside].copy(),
        sr[y0:y1, x0:x1].copy(),
        hr[y0:y1, x0:x1].copy(),
        {
            "crop_hr_yx": [int(y0), int(x0)],
            "crop_hr_hw": [int(side), int(side)],
            "crop_lr_yx": [int(ly0), int(lx0)],
            "crop_lr_hw": [int(lside), int(lside)],
        },
    )


def _upsample_lr_rgb_to_hr(lr_rgb: np.ndarray, lr_hw: tuple[int, int], hr_hw: tuple[int, int]) -> np.ndarray:
    sy = int(hr_hw[0]) // int(lr_hw[0])
    sx = int(hr_hw[1]) // int(lr_hw[1])
    if int(lr_hw[0]) * sy != int(hr_hw[0]) or int(lr_hw[1]) * sx != int(hr_hw[1]):
        raise ValueError(f"HR shape {hr_hw} is not an integer multiple of LR shape {lr_hw}")
    return np.repeat(np.repeat(lr_rgb, sy, axis=0), sx, axis=1)


def _rgb01(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.size and float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _ipfz_residual_rgb(
    sr_q_arr: np.ndarray,
    hr_q_arr: np.ndarray,
    sym_class,
) -> tuple[np.ndarray, float]:
    sr_z = _rgb01(render_ipf_rgb(sr_q_arr, sym_class, ref_dir="Z"))
    hr_z = _rgb01(render_ipf_rgb(hr_q_arr, sym_class, ref_dir="Z"))
    err = np.linalg.norm(hr_z - sr_z, axis=-1) / np.sqrt(3.0)
    vmax = float(np.quantile(err[np.isfinite(err)], 0.995)) if np.isfinite(err).any() else 1.0
    vmax = float(np.clip(vmax, 0.05, 1.0))
    rgba = plt.get_cmap("magma")(np.clip(err / vmax, 0.0, 1.0))
    return rgba[..., :3].astype(np.float32, copy=False), vmax


def _render_fig7_triplet(
    *,
    lr_q_arr: np.ndarray,
    sr_q_arr: np.ndarray,
    hr_q_arr: np.ndarray,
    sym_class,
    out_png: Path,
    scale: int,
    crop_square: bool = True,
    dpi: int = 600,
    panel_px: int = 900,
) -> dict[str, object]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": NMI_FONT,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.0,
            "axes.titlesize": 6.5,
        }
    )
    if crop_square:
        lr_q_arr, sr_q_arr, hr_q_arr, crop_meta = _crop_triplet_to_square(
            lr_q_arr,
            sr_q_arr,
            hr_q_arr,
            scale=scale,
        )
    else:
        crop_meta = {}

    def _fmt(arr: np.ndarray) -> np.ndarray:
        return format_quaternions(
            arr,
            normalize=True,
            hemisphere=True,
            reduce_fz=True,
            sym=sym_class,
            to_quat_first=False,
        )

    lr_q_arr = _fmt(lr_q_arr)
    sr_q_arr = _fmt(sr_q_arr)
    hr_q_arr = _fmt(hr_q_arr)

    lr_hw = lr_q_arr.shape[:2]
    hr_hw = hr_q_arr.shape[:2]
    lr_rgbs = [_upsample_lr_rgb_to_hr(_rgb01(img), lr_hw, hr_hw) for img in render_ipf_rgb(lr_q_arr, sym_class, ref_dir="ALL")]
    sr_rgbs = [_rgb01(img) for img in render_ipf_rgb(sr_q_arr, sym_class, ref_dir="ALL")]
    hr_rgbs = [_rgb01(img) for img in render_ipf_rgb(hr_q_arr, sym_class, ref_dir="ALL")]
    error_rgb, error_vmax = _ipfz_residual_rgb(sr_q_arr, hr_q_arr, sym_class)

    fig_px_w = int(panel_px * 4 + panel_px * 1.05)
    fig_px_h = int(panel_px * 3)
    fig = plt.figure(figsize=(fig_px_w / dpi, fig_px_h / dpi), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(
        3,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.05],
        height_ratios=[1.0, 1.0, 1.0],
        left=0.006,
        right=0.994,
        bottom=0.010,
        top=0.955,
        wspace=0.018,
        hspace=0.018,
    )

    col_titles = ["IPF-X", "IPF-Y", "IPF-Z"]
    def _imshow(ax: plt.Axes, img: np.ndarray) -> None:
        ax.imshow(img, interpolation="nearest", resample=False)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for c_idx, title in enumerate(col_titles):
        for r_idx, panels in enumerate((lr_rgbs, sr_rgbs, hr_rgbs)):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            _imshow(ax, panels[c_idx])
            if r_idx == 0:
                ax.set_title(title, fontsize=8.0, pad=2.6, fontweight="regular")

    ax_err = fig.add_subplot(gs[1, 3])
    _imshow(ax_err, error_rgb)
    ax_err.set_title("IPF-Z\n|HR-SR|", fontsize=7.6, pad=2.6, fontweight="regular", linespacing=0.95)

    add_ipf_key_panel(
        fig,
        gs[1, 4],
        sym_class,
        title="IPF-Z key",
        title_fontsize=7.6,
        label_fontsize=6.8,
        title_height_ratio=0.20,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved Fig. 7 panel to: {out_png}")
    return {
        **crop_meta,
        "rendered_shape_px": [int(fig_px_h), int(fig_px_w)],
        "panel_px": int(panel_px),
        "dpi": int(dpi),
        "ipfz_residual": "Euclidean RGB residual magnitude |IPF-Z(HR) - IPF-Z(SR)| normalized by sqrt(3).",
        "ipfz_residual_vmax_995": error_vmax,
    }


def _infer_scalar_layout(q_raw: np.ndarray, ratio_threshold: float = 1.5) -> tuple[str, np.ndarray]:
    q = np.asarray(q_raw, dtype=np.float32)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion-last array, got shape {q.shape}")
    abs_mean = np.mean(np.abs(q.reshape(-1, 4)), axis=0)
    first = float(abs_mean[0])
    last = float(abs_mean[-1])
    if first > ratio_threshold * max(last, 1e-12):
        return "wxyz", abs_mean
    if last > ratio_threshold * max(first, 1e-12):
        return "xyzw", abs_mean
    raise ValueError(
        "Could not infer scalar component layout for Open718 quaternions: "
        f"component abs means are {abs_mean.tolist()}"
    )


def _raw_to_passive_wxyz(q_raw: np.ndarray) -> tuple[np.ndarray, str, np.ndarray]:
    layout, abs_mean = _infer_scalar_layout(q_raw)
    q = np.asarray(q_raw, dtype=np.float32)
    if layout == "xyzw":
        q = np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)
    return _normalize_quat_hwc(q), layout, abs_mean


def _infer_open718_triplet(spec: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    hr_source = Path(spec["hr_source"])
    if not hr_source.exists():
        raise FileNotFoundError(hr_source)

    triplet_dir = EVAL_DIR / "fig7_open718_current_direct_reynolds_seed42"
    cached_paths = {
        "lr": triplet_dir / "lr.npy",
        "sr": triplet_dir / "sr.npy",
        "hr": triplet_dir / "hr.npy",
    }
    if bool(spec.get("reuse_cached_triplet", False)) and all(path.exists() for path in cached_paths.values()):
        lr = np.load(cached_paths["lr"]).astype(np.float32, copy=False)
        sr = np.load(cached_paths["sr"]).astype(np.float32, copy=False)
        hr = np.load(cached_paths["hr"]).astype(np.float32, copy=False)
        meta = {
            "name": spec["name"],
            "hr_source": str(hr_source.relative_to(ROOT)),
            "triplet_dir": str(triplet_dir.relative_to(PAPER_DIR)),
            "cached_triplet_reused": True,
            "lr_shape": list(lr.shape),
            "sr_shape": list(sr.shape),
            "hr_shape": list(hr.shape),
            "symmetry": spec["symmetry"],
            "scale": int(spec["scale"]),
        }
        return lr, sr, hr, meta

    sym = resolve_symmetry(str(spec["symmetry"]))
    scale = int(spec["scale"])
    hr_raw_file = np.load(hr_source).astype(np.float32, copy=False)
    hr_passive_raw, raw_scalar_layout, raw_component_abs_mean = _raw_to_passive_wxyz(hr_raw_file)
    hr_full = format_quaternions(
        hr_passive_raw,
        normalize=True,
        hemisphere=True,
        reduce_fz=True,
        sym=sym,
        to_quat_first=False,
    )
    aligned_h = (int(hr_full.shape[0]) // scale) * scale
    aligned_w = (int(hr_full.shape[1]) // scale) * scale
    hr = hr_full[:aligned_h, :aligned_w].copy()
    lr = hr[::scale, ::scale].copy()

    exp_dir = Path(spec["exp_dir"])
    cfg = load_and_prepare_config(exp_dir / str(spec["config"]))
    checkpoint_path = _resolve_checkpoint(cfg, exp_dir, str(spec["checkpoint"]))
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_checkpoint(cfg, checkpoint_path, device=device)
    forward_params = inspect.signature(model.forward_sr).parameters
    forward_supports_lr_boundary = "lr_boundary_map" in forward_params
    forward_requires_lr_boundary = (
        forward_supports_lr_boundary
        and forward_params["lr_boundary_map"].default is inspect._empty
    )
    if forward_requires_lr_boundary:
        raise RuntimeError("Open718 custom Fig. 7 render cannot supply a required LR boundary map.")

    lr_tensor = torch.from_numpy(lr).to(device=device, dtype=torch.float32)
    lr_flat, lr_shape = _flatten_quat_chw(lr_tensor)
    with torch.no_grad():
        with torch.enable_grad():
            sr_flat = model.forward_sr(
                lr_flat,
                lr_shape=lr_shape,
                normalize_input=True,
            )
    expected_pixels = int(hr.shape[0] * hr.shape[1])
    if int(sr_flat.shape[0]) != expected_pixels:
        raise ValueError(
            f"SR size mismatch for Open718: got {int(sr_flat.shape[0])}, expected {expected_pixels}"
        )
    sr = sr_flat.reshape(hr.shape[0], hr.shape[1], 4).detach().cpu().numpy().astype(np.float32)
    sr = format_quaternions(
        sr,
        normalize=True,
        hemisphere=True,
        reduce_fz=True,
        sym=sym,
        to_quat_first=False,
    )

    triplet_dir.mkdir(parents=True, exist_ok=True)
    np.save(triplet_dir / "lr.npy", lr)
    np.save(triplet_dir / "sr.npy", sr)
    np.save(triplet_dir / "hr.npy", hr)

    meta = {
        "name": spec["name"],
        "hr_source": str(hr_source.relative_to(ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "config": str((exp_dir / str(spec["config"])).relative_to(ROOT)),
        "triplet_dir": str(triplet_dir.relative_to(PAPER_DIR)),
        "raw_shape": list(hr_raw_file.shape),
        "aligned_hr_shape": list(hr.shape),
        "derived_lr_shape": list(lr.shape),
        "sr_shape": list(sr.shape),
        "raw_scalar_layout": raw_scalar_layout,
        "raw_component_abs_mean": [float(x) for x in raw_component_abs_mean],
        "symmetry": spec["symmetry"],
        "scale": scale,
        "device": str(device),
    }
    return lr, sr, hr, meta


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, object] = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "note": "Current OCRP qualitative manuscript figures rendered from corrected direct-Reynolds-isometric seed-42 outputs. The Fig. 7 FCC panel is regenerated from the Open718 HR array named in the manuscript-side source file.",
        "figures": [],
    }

    lr, sr, hr, open718_meta = _infer_open718_triplet(OPEN718_FCC_SPEC)
    sym = resolve_symmetry(str(OPEN718_FCC_SPEC["symmetry"]))
    out_png = Path(OPEN718_FCC_SPEC["out_png"])
    render_meta = _render_fig7_triplet(
        lr_q_arr=lr,
        sr_q_arr=sr,
        hr_q_arr=hr,
        sym_class=sym,
        out_png=out_png,
        scale=int(OPEN718_FCC_SPEC["scale"]),
        crop_square=True,
        dpi=600,
        panel_px=900,
    )
    open718_meta["figure"] = str(out_png.relative_to(PAPER_DIR))
    open718_meta["render"] = render_meta
    provenance["figures"].append(open718_meta)
    print(f"Wrote {out_png}")

    for spec in SAVED_TRIPTYCHS:
        sr_dir = Path(spec["sr_dir"])
        sample_id = int(spec["sample_id"])
        if not sr_dir.exists():
            raise FileNotFoundError(sr_dir)
        lr, sr, hr = _load_triplet(sr_dir, sample_id)
        sym = resolve_symmetry(str(spec["symmetry"]))
        out_png = Path(spec["out_png"])
        render_meta = _render_fig7_triplet(
            lr_q_arr=lr,
            sr_q_arr=sr,
            hr_q_arr=hr,
            sym_class=sym,
            out_png=out_png,
            scale=4,
            crop_square=True,
            dpi=600,
            panel_px=900,
        )
        provenance["figures"].append(
            {
                "name": spec["name"],
                "figure": str(out_png.relative_to(PAPER_DIR)),
                "source_dir": str(sr_dir.relative_to(ROOT)),
                "sample_id": sample_id,
                "symmetry": spec["symmetry"],
                "lr_shape": list(lr.shape),
                "sr_shape": list(sr.shape),
                "hr_shape": list(hr.shape),
                "render": render_meta,
            }
        )
        print(f"Wrote {out_png}")

    out_json = EVAL_DIR / "current_model_figure_provenance_20260707.json"
    out_json.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
