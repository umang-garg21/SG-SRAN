#!/usr/bin/env python
"""Batch IPF-only visualization for SDF SR models."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from orix import plot as orix_plot
from orix.quaternion import Orientation
from orix.vector import Vector3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.symmetry_utils import resolve_symmetry
from visualization.model_stage_walkthrough import load_lr_input, load_model_from_experiment
from models.train_irrep_sdf_sr_quatdataset import FullTrainConfig, build_model_from_cfg
from utils.runtime_helpers import load_checkpoint_state_compat
from training.train_irrep_sdf_sr_microstats_prior import (
    _coerce_cfg as _coerce_microstats_cfg,
    build_model as _build_microstats_model,
)
from training.train_irrep_a1_boundary_gated_sr import _coerce_cfg as _coerce_gated_cfg
from models.train_irrep_a1_boundary_gated_sr import build_model as _build_gated_model


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _ensure_hwc_quaternion_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 quaternion image, got {arr.shape}")
    if arr.shape[-1] == 4:
        out = arr
    elif arr.shape[0] == 4:
        out = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"Could not find quaternion axis in shape {arr.shape}")
    return out.astype(np.float32, copy=False)


def _load_lr_pairs(dataset_root: Path, split: str) -> list[tuple[tuple[str, int], Path, Path]]:
    name_re = re.compile(
        r"^(?P<ds>.+)_(?P<split>train|val|test)_(?P<which>hr|lr)_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
        re.IGNORECASE,
    )

    def _pair_key(path: Path) -> tuple[str, int] | None:
        m = name_re.match(path.name)
        if m is None:
            return None
        return m.group("axis").lower(), int(m.group("id"))

    split_dir = dataset_root / str(split)
    lr_dir = split_dir / "LR_Data"
    hr_dir = split_dir / "HR_Data"
    if not lr_dir.exists() or not hr_dir.exists():
        raise FileNotFoundError(f"Missing LR/HR split dirs under: {split_dir}")

    lr_map: dict[tuple[str, int], Path] = {}
    for fp in sorted(lr_dir.glob("*.npy")):
        key = _pair_key(fp)
        if key is not None:
            lr_map[key] = fp

    hr_map: dict[tuple[str, int], Path] = {}
    for fp in sorted(hr_dir.glob("*.npy")):
        key = _pair_key(fp)
        if key is not None:
            hr_map[key] = fp

    common = sorted(set(lr_map).intersection(hr_map))
    return [(key, lr_map[key], hr_map[key]) for key in common]


def _load_hr_input(
    cfg: Any,
    *,
    split: str,
    sample_offset: int,
    dataset_root: str | None,
    crop_hw: tuple[int, int] | None,
    scale: int,
) -> np.ndarray | None:
    root = Path(dataset_root or str(getattr(cfg, "dataset_root", ""))).resolve()
    if not str(root):
        return None

    pairs = _load_lr_pairs(root, split)
    if sample_offset < 0 or sample_offset >= len(pairs):
        return None
    _, _, hr_fp = pairs[sample_offset]
    hr = _ensure_hwc_quaternion_image(np.load(hr_fp))

    if crop_hw is not None:
        ch, cw = int(crop_hw[0]), int(crop_hw[1])
        hr = hr[: ch * int(scale), : cw * int(scale), :]
    return hr


def _render_ipf_rgb_all(q_hwc: np.ndarray, sym_class) -> list[np.ndarray]:
    q = _ensure_hwc_quaternion_image(q_hwc)
    ori = Orientation(q)
    ori.symmetry = sym_class
    ckey = orix_plot.IPFColorKeyTSL(sym_class.laue)

    dirs = [
        ("X", Vector3d((1, 0, 0))),
        ("Y", Vector3d((0, 1, 0))),
        ("Z", Vector3d((0, 0, 1))),
    ]
    out: list[np.ndarray] = []
    for _, ref_dir in dirs:
        ckey.direction = ref_dir
        out.append(ckey.orientation2color(ori))
    return out


def _plot_ipf_rows(
    *,
    lr_q_hwc: np.ndarray,
    sr_q_hwc: np.ndarray,
    hr_q_hwc: np.ndarray | None,
    sym_class,
    out_png: Path,
    title: str,
) -> None:
    rows: list[tuple[str, list[np.ndarray]]] = []
    rows.append(("LR", _render_ipf_rgb_all(lr_q_hwc, sym_class)))
    rows.append(("SR", _render_ipf_rgb_all(sr_q_hwc, sym_class)))
    if hr_q_hwc is not None:
        rows.append(("HR", _render_ipf_rgb_all(hr_q_hwc, sym_class)))

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(12.0, 3.8 * nrows))
    axes = np.asarray(axes).reshape(nrows, 3)

    for r, (row_name, rgbs) in enumerate(rows):
        for c, axis_name in enumerate(("X", "Y", "Z")):
            ax = axes[r, c]
            ax.imshow(rgbs[c])
            ax.set_title(f"{row_name} IPF-{axis_name}")
            ax.axis("off")

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _set_decoder_settings(
    model,
    *,
    num_starts: int,
    steps: int,
    lr: float,
) -> dict[str, tuple[Any, Any]]:
    dec = getattr(model, "decoder", None)
    if dec is None:
        return {}

    changed: dict[str, tuple[Any, Any]] = {}

    if hasattr(dec, "num_starts"):
        old = getattr(dec, "num_starts")
        new = max(1, int(num_starts))
        setattr(dec, "num_starts", new)
        changed["num_starts"] = (old, new)

    if hasattr(dec, "steps"):
        old = getattr(dec, "steps")
        new = max(0, int(steps))
        setattr(dec, "steps", new)
        changed["steps"] = (old, new)

    if hasattr(dec, "lr"):
        old = getattr(dec, "lr")
        new = float(lr)
        setattr(dec, "lr", new)
        changed["lr"] = (old, new)

    return changed


def _coerce_train_cfg(exp_dir: Path, raw: dict[str, Any]) -> FullTrainConfig:
    valid_keys = {f.name for f in fields(FullTrainConfig)}
    aliases = {
        "scale": "upsample_factor",
        "clip": "grad_clip",
        "boundary_loss_weight": "lambda_boundary",
        "boundary_lr_loss_weight": "lambda_lr_boundary",
        "boundary_teacher_thr_deg": "boundary_thr_deg",
        "boundary_teacher_connectivity": "boundary_connectivity",
        "boundary_use_focal": "use_focal_boundary",
        "use_amp": "amp",
    }

    mapped: dict[str, Any] = {}
    for k, v in raw.items():
        kk = aliases.get(k, k)
        if kk in valid_keys:
            mapped[kk] = v

    if "dataset_root" not in mapped:
        raise ValueError("config must define dataset_root")

    mapped.setdefault("train_split", "Train")
    mapped.setdefault("val_split", "Val")
    mapped.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    mapped.setdefault("out_dir", str(exp_dir / "checkpoints"))
    return FullTrainConfig(**mapped)


def _load_model_for_ipf(
    exp_dir: Path,
    *,
    config_name: str,
    checkpoint_path: Path,
    device: str | None,
):
    # Primary path: walkthrough loader for attention-model experiments.
    try:
        return load_model_from_experiment(
            exp_dir,
            config_name=config_name,
            checkpoint_name=str(checkpoint_path),
            device=device,
        )
    except RuntimeError as exc:
        print(f"[viz] default loader failed, trying one-sided fallback: {exc}")

    # Fallback path A: microstats-prior trainer config/model.
    config_path = (exp_dir / config_name).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    try:
        ms_cfg = _coerce_microstats_cfg(exp_dir=exp_dir, raw=raw_cfg)
        if device is not None:
            ms_cfg.device = str(device)

        ms_model = _build_microstats_model(ms_cfg).eval()
        ms_device_obj = torch.device(ms_cfg.device)
        ckpt = torch.load(checkpoint_path, map_location=ms_device_obj)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        load_checkpoint_state_compat(ms_model, state, context=f"checkpoint {checkpoint_path}")
        return ms_model, ms_cfg, checkpoint_path
    except Exception as exc:
        print(f"[viz] microstats-prior fallback failed, trying gated fallback: {exc}")

    # Fallback path B: gated-SR trainer config/model.
    try:
        gated_cfg = _coerce_gated_cfg(exp_dir=exp_dir, raw=raw_cfg)
        if device is not None:
            gated_cfg.device = str(device)

        gated_model = _build_gated_model(gated_cfg).eval()
        gated_device_obj = torch.device(gated_cfg.device)
        ckpt = torch.load(checkpoint_path, map_location=gated_device_obj)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        load_checkpoint_state_compat(gated_model, state, context=f"checkpoint {checkpoint_path}")
        return gated_model, gated_cfg, checkpoint_path
    except Exception as exc:
        print(f"[viz] gated fallback failed, trying one-sided fallback: {exc}")

    # Fallback path: one-sided trainer config/model.
    cfg = _coerce_train_cfg(exp_dir=exp_dir, raw=raw_cfg)
    if device is not None:
        cfg.device = str(device)

    model = build_model_from_cfg(cfg).eval()
    device_obj = torch.device(cfg.device)
    ckpt = torch.load(checkpoint_path, map_location=device_obj)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    load_checkpoint_state_compat(model, state, context=f"checkpoint {checkpoint_path}")
    return model, cfg, checkpoint_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IPF-only LR/SR/HR maps for first N samples.")
    p.add_argument("--exp_dir", required=True, type=str)
    p.add_argument("--config", type=str, default="config.json")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"])
    p.add_argument("--start_offset", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--crop_hw", nargs=2, type=int, default=None)
    p.add_argument("--out_dir", type=str, required=True)

    # Good decode defaults for robust IPF rendering quality.
    p.add_argument("--decoder_num_starts", type=int, default=16)
    p.add_argument("--decoder_steps", type=int, default=20)
    p.add_argument("--decoder_lr", type=float, default=0.03)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (exp_dir / ckpt_path).resolve()

    model, cfg, _ = _load_model_for_ipf(
        exp_dir,
        config_name=args.config,
        checkpoint_path=ckpt_path,
        device=args.device,
    )

    decoder_changes = _set_decoder_settings(
        model,
        num_starts=int(args.decoder_num_starts),
        steps=int(args.decoder_steps),
        lr=float(args.decoder_lr),
    )

    crystal = str(getattr(cfg, "crystal", "fcc")).lower()
    sym_name = "D6h" if crystal == "hcp" else "Oh"
    sym_class = resolve_symmetry(sym_name)

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scale = int(getattr(model, "upsample_factor", int(getattr(cfg, "scale", 4))))
    crop_hw = None if args.crop_hw is None else (int(args.crop_hw[0]), int(args.crop_hw[1]))

    start = int(args.start_offset)
    stop = start + int(args.num_samples)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Decoder changes: {decoder_changes}")
    print(f"Split: {args.split} | offsets: {start}..{stop - 1}")

    for sample_offset in range(start, stop):
        lr_q_hwc, label = load_lr_input(
            cfg,
            split=args.split,
            sample_offset=sample_offset,
            dataset_root=args.dataset_root,
            lr_npy=None,
            crop_hw=crop_hw,
        )

        h, w, _ = lr_q_hwc.shape
        lr_shape = (h, w)
        hr_shape = (h * scale, w * scale)

        hr_q_hwc = _load_hr_input(
            cfg,
            split=args.split,
            sample_offset=sample_offset,
            dataset_root=args.dataset_root,
            crop_hw=crop_hw,
            scale=scale,
        )

        lr_flat = torch.from_numpy(lr_q_hwc.reshape(h * w, 4)).to(model.device, dtype=torch.float32)
        # The optimizing decoder performs gradient-based inner steps, so keep grad enabled.
        with torch.enable_grad():
            q_sr_flat = model.forward_sr(
                lr_quats=lr_flat,
                lr_shape=lr_shape,
                normalize_input=True,
                return_aux=False,
            )

        q_sr_hwc = q_sr_flat.detach().cpu().reshape(hr_shape[0], hr_shape[1], 4).numpy().astype(np.float32, copy=False)

        stem = f"sample_{sample_offset:02d}_{_safe_name(label)}"
        out_png = out_root / f"{stem}__ipf_lr_sr_hr.png"
        _plot_ipf_rows(
            lr_q_hwc=lr_q_hwc,
            sr_q_hwc=q_sr_hwc,
            hr_q_hwc=hr_q_hwc,
            sym_class=sym_class,
            out_png=out_png,
            title=f"{label} | offset={sample_offset}",
        )
        print(f"[saved] {out_png}")

    print(f"Done. IPF maps written to: {out_root}")


if __name__ == "__main__":
    main()
