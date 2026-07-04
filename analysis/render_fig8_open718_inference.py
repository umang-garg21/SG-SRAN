#!/usr/bin/env python3
"""Regenerate Fig. 8 IN718 row from the Open718 HR-only sample."""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.infer_iso_embedding_sr_attn import (  # noqa: E402
    _flatten_quat_chw,
    _load_model_from_checkpoint,
)
from training.config_utils import load_and_prepare_config  # noqa: E402
from utils.quat_ops import enforce_hemisphere, normalize_quaternions  # noqa: E402
from utils.symmetry_utils import resolve_symmetry  # noqa: E402
from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side  # noqa: E402

HR_PATH = ROOT / "Paper/EBSD_SR_Nature_v4/Open_718_Test_hr_x_block_0.npy"
EXP_DIR = ROOT / "experiments/IN718/direct_reynolds_isometric_seed_runs/ocrp_direct_reynolds_isometric_l4_s42"
CONFIG_NAME = "config_new.json"
CHECKPOINT_NAME = "best_model.pt"
OUT_DIR = ROOT / "Paper/EBSD_SR_Nature_v4/evals/fig8_open718_direct_reynolds_isometric_l4_s42"
FIG_PATH = ROOT / "Paper/EBSD_SR_Nature_v4/figs/main_4x4_open718_test0_lr_sr_hr_ipf.png"
HCP_SR_DIR = (
    ROOT
    / "experiments/Ti_Al_1pct/iso_embedding_4x4_ocrp_anchorless_direct_reynolds_isometric_l6_s42"
    / "inference/test_best/sr_quaternions"
)
HCP_FIG_PATH = ROOT / "Paper/EBSD_SR_Nature_v4/figs/main_4x4_TiAl_test_s010_lr_sr_hr_ipf.png"
SCALE = (4, 4)


def _load_open718_hr(path: Path) -> tuple[np.ndarray, dict]:
    raw = np.load(path).astype(np.float32, copy=False)
    if raw.ndim != 3 or raw.shape[-1] != 4:
        raise ValueError(f"Expected HWC quaternion array with 4 channels, got {raw.shape}")

    # This paper-side Open718 file is scalar-last [x, y, z, w]; the model and
    # renderer use passive scalar-first [w, x, y, z].
    hr = raw[..., [3, 0, 1, 2]].astype(np.float32, copy=False)
    hr = normalize_quaternions(hr, axis=-1).astype(np.float32, copy=False)
    hr = enforce_hemisphere(hr, scalar_first=True).astype(np.float32, copy=False)

    side = min((hr.shape[0] // SCALE[0]) * SCALE[0], (hr.shape[1] // SCALE[1]) * SCALE[1])
    y0 = (hr.shape[0] - side) // 2
    x0 = (hr.shape[1] - side) // 2
    aligned = hr[y0 : y0 + side, x0 : x0 + side].copy()
    meta = {
        "source_path": str(path),
        "raw_shape": list(raw.shape),
        "source_quaternion_layout": "scalar-last [x,y,z,w]",
        "model_quaternion_layout": "passive scalar-first [w,x,y,z]",
        "aligned_shape": list(aligned.shape),
        "crop": "centered largest exact square 4x4-aligned region",
        "crop_origin_yx": [int(y0), int(x0)],
    }
    return aligned, meta


def _run_inference(hr_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    config_path = EXP_DIR / CONFIG_NAME
    run_config_path = OUT_DIR / "inference_run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    checkpoint_path = EXP_DIR / "checkpoints" / CHECKPOINT_NAME
    model = _load_model_from_checkpoint(cfg, checkpoint_path, device=device)
    forward_params = inspect.signature(model.forward_sr).parameters
    if "lr_boundary_map" in forward_params and forward_params["lr_boundary_map"].default is inspect._empty:
        raise RuntimeError("This Fig. 8 inference script does not provide LR boundary maps.")

    lr_np = hr_np[:: SCALE[0], :: SCALE[1]].copy()
    lr = torch.from_numpy(lr_np).to(device=device, dtype=torch.float32)
    lr_flat, lr_shape = _flatten_quat_chw(lr)

    with torch.no_grad():
        with torch.enable_grad():
            sr_flat = model.forward_sr(lr_flat, lr_shape=lr_shape, normalize_input=True)

    expected = int(hr_np.shape[0] * hr_np.shape[1])
    if int(sr_flat.shape[0]) != expected:
        raise ValueError(f"SR size mismatch: got {int(sr_flat.shape[0])}, expected {expected}")
    sr_np = sr_flat.reshape(hr_np.shape[0], hr_np.shape[1], 4).detach().cpu().numpy().astype(np.float32)
    sr_np = normalize_quaternions(sr_np, axis=-1).astype(np.float32, copy=False)
    sr_np = enforce_hemisphere(sr_np, scalar_first=True).astype(np.float32, copy=False)
    return lr_np, sr_np


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sr_quaternions").mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    hr_np, meta = _load_open718_hr(HR_PATH)
    lr_np, sr_np = _run_inference(hr_np)

    np.save(OUT_DIR / "sr_quaternions/sample_000000_hr.npy", hr_np)
    np.save(OUT_DIR / "sr_quaternions/sample_000000_lr.npy", lr_np)
    np.save(OUT_DIR / "sr_quaternions/sample_000000_sr.npy", sr_np)

    sym_class = resolve_symmetry("Oh")
    render_sr_hr_lr_side_by_side(
        sr_q_arr=sr_np,
        hr_q_arr=hr_np,
        lr_q_arr=lr_np,
        sym_class=sym_class,
        out_png=str(FIG_PATH),
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=True,
        dpi=300,
        include_row_labels=False,
    )

    summary = {
        **meta,
        "experiment": str(EXP_DIR),
        "config": CONFIG_NAME,
        "checkpoint": CHECKPOINT_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "scale": list(SCALE),
        "lr_shape": list(lr_np.shape),
        "sr_shape": list(sr_np.shape),
        "hr_shape": list(hr_np.shape),
        "figure": str(FIG_PATH),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    hcp_lr = np.load(HCP_SR_DIR / "sample_000010_lr.npy").astype(np.float32, copy=False)
    hcp_sr = np.load(HCP_SR_DIR / "sample_000010_sr.npy").astype(np.float32, copy=False)
    hcp_hr = np.load(HCP_SR_DIR / "sample_000010_hr.npy").astype(np.float32, copy=False)
    render_sr_hr_lr_side_by_side(
        sr_q_arr=hcp_sr,
        hr_q_arr=hcp_hr,
        lr_q_arr=hcp_lr,
        sym_class=resolve_symmetry("D6h"),
        out_png=str(HCP_FIG_PATH),
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=True,
        dpi=300,
        pixels_per_image_pixel=3,
        include_row_labels=False,
    )
    summary["hcp_figure"] = str(HCP_FIG_PATH)
    summary["hcp_sample"] = "sample_000010 from Ti_Al_1pct test_best inference"
    summary["hcp_lr_shape"] = list(hcp_lr.shape)
    summary["hcp_sr_shape"] = list(hcp_sr.shape)
    summary["hcp_hr_shape"] = list(hcp_hr.shape)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
