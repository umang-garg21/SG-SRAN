import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from orix.quaternion import Orientation
from orix.quaternion.orientation_region import OrientationRegion
from scipy.spatial.transform import Rotation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from training.config_utils import load_and_prepare_config
from utils.quat_ops import (
    enforce_hemisphere,
    normalize_quaternions,
    quat_left_multiply_numpy,
    reduce_to_fz_min_angle,
    to_scalar_first,
    to_spatial_quat,
)
from utils.symmetry_utils import resolve_symmetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a .npy quaternion file and diagnose convention")
    parser.add_argument("--npy_path", required=True, type=str)
    parser.add_argument("--symmetry", type=str, default="O")
    parser.add_argument("--out_dir", type=str, default="outputs/npy_convention_debug")

    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _resolve_checkpoint_path(exp_dir: Path, checkpoint_arg: str) -> Path:
    if checkpoint_arg == "last":
        return exp_dir / "checkpoints" / "last_checkpoint.pt"
    if checkpoint_arg == "best":
        return exp_dir / "checkpoints" / "best_model.pt"
    return Path(checkpoint_arg)


def _load_model_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        blob = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(str(ckpt_path), map_location=device)

    if isinstance(blob, dict) and "model_state_dict" in blob:
        state_dict = blob["model_state_dict"]
    elif isinstance(blob, dict):
        state_dict = blob
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        print(f"[debug] Unexpected keys (up to 10): {load_result.unexpected_keys[:10]}")
    if load_result.missing_keys:
        print(f"[debug] Missing keys (up to 10): {load_result.missing_keys[:10]}")


def quat_right_multiply_numpy(q_left_field: np.ndarray, q_right_ops: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q_spatial = np.asarray(to_spatial_quat(q_left_field), dtype=np.float32)
    ops = np.asarray(q_right_ops, dtype=np.float32)

    spatial_shape = q_spatial.shape[:-1]
    n = int(np.prod(spatial_shape))
    m = ops.shape[0]
    flat = q_spatial.reshape(n, 4)

    w0, x0, y0, z0 = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    w1, x1, y1, z1 = ops[:, 0], ops[:, 1], ops[:, 2], ops[:, 3]

    out = np.empty((m, n, 4), dtype=np.float32)
    out[:, :, 0] = w0[None, :] * w1[:, None] - x0[None, :] * x1[:, None] - y0[None, :] * y1[:, None] - z0[None, :] * z1[:, None]
    out[:, :, 1] = w0[None, :] * x1[:, None] + x0[None, :] * w1[:, None] + y0[None, :] * z1[:, None] - z0[None, :] * y1[:, None]
    out[:, :, 2] = w0[None, :] * y1[:, None] - x0[None, :] * z1[:, None] + y0[None, :] * w1[:, None] + z0[None, :] * x1[:, None]
    out[:, :, 3] = w0[None, :] * z1[:, None] + x0[None, :] * y1[:, None] - y0[None, :] * x1[:, None] + z0[None, :] * w1[:, None]

    out /= np.clip(np.linalg.norm(out, axis=2, keepdims=True), eps, None)
    return out.reshape((m, *spatial_shape, 4))


def reduce_to_fz_min_angle_right(
    q: np.ndarray,
    sym,
    normalize: bool = True,
    hemisphere: bool = True,
    return_op_map: bool = False,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    q_spatial = np.ascontiguousarray(to_spatial_quat(q), dtype=np.float32)

    if normalize:
        normalize_quaternions(q_spatial, axis=-1, eps=eps)
    if hemisphere:
        enforce_hemisphere(q_spatial, scalar_first=True)

    if isinstance(sym, str):
        sym = resolve_symmetry(sym)

    flat = q_spatial.reshape(-1, 4)
    n = flat.shape[0]
    region = OrientationRegion.from_symmetry(sym)

    if (Orientation(flat, sym) < region).all():
        if return_op_map:
            return q_spatial, np.zeros(q_spatial.shape[:-1], dtype=np.int32)
        return q_spatial

    ops = sym.data.astype(np.float32, copy=False)
    m = ops.shape[0]
    cand = quat_right_multiply_numpy(q_spatial, ops, eps=eps)
    cand_flat = cand.reshape(m * n, 4)

    inside_mask = (Orientation(cand_flat, sym) < region).reshape(m, n)

    cand_2d = cand.reshape(m, n, 4)
    w_vals = cand_2d[..., 0]
    w_vals[~inside_mask] = -np.inf

    best_idx = np.argmax(w_vals, axis=0)
    pick = np.take_along_axis(cand_2d, best_idx[np.newaxis, :, np.newaxis], axis=0).squeeze(0)
    pick = pick.reshape(q_spatial.shape)
    pick = enforce_hemisphere(pick, scalar_first=True)

    if return_op_map:
        return pick.astype(np.float32, copy=False), best_idx.reshape(q_spatial.shape[:-1]).astype(np.int32, copy=False)
    return pick.astype(np.float32, copy=False)


def _mis_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a = q1.reshape(-1, 4).astype(np.float32)
    b = q2.reshape(-1, 4).astype(np.float32)
    a /= np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b /= np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    dots = np.sum(a * b, axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return (2.0 * np.arccos(dots) * 180.0 / np.pi).astype(np.float32)


def _stats(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
    }


def _analyze_interpretation(q_spatial_scalar_first: np.ndarray, sym) -> dict[str, Any]:
    q0 = np.ascontiguousarray(q_spatial_scalar_first, dtype=np.float32)
    normalize_quaternions(q0, axis=-1, eps=1e-12)

    left_fz, left_idx = reduce_to_fz_min_angle(
        q0,
        sym=sym,
        normalize=True,
        hemisphere=True,
        return_op_map=True,
    )
    left_fz = enforce_hemisphere(left_fz, scalar_first=True)

    right_fz, right_idx = reduce_to_fz_min_angle_right(
        q0,
        sym=sym,
        normalize=True,
        hemisphere=True,
        return_op_map=True,
    )

    mis_left_vs_right = _mis_deg(left_fz, right_fz)

    return {
        "left_vs_right_deg": _stats(mis_left_vs_right),
        "left_idx_hist": np.bincount(left_idx.reshape(-1), minlength=len(sym.data)).tolist(),
        "right_idx_hist": np.bincount(right_idx.reshape(-1), minlength=len(sym.data)).tolist(),
        "left_fz_w_lt_0": int((left_fz[..., 0] < 0).sum()),
        "right_fz_w_lt_0": int((right_fz[..., 0] < 0).sum()),
        "_left_fz": left_fz,
        "_right_fz": right_fz,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.load(args.npy_path)
    sym = resolve_symmetry(args.symmetry)

    # Normalize various common input formats to quaternion spatial-last (..., 4), scalar-first
    if arr.ndim >= 3 and arr.shape[-2:] == (3, 3):
        # rotation matrices -> quaternions
        mats = arr.reshape(-1, 3, 3)
        quat_xyzw = Rotation.from_matrix(mats).as_quat().astype(np.float32)  # (x,y,z,w)
        quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
        q_spatial = quat_wxyz.reshape(arr.shape[:-2] + (4,))
    else:
        quat_axes = [i for i, s in enumerate(arr.shape) if s == 4]
        if len(quat_axes) == 0:
            raise ValueError(f"No quaternion axis (size 4) found in shape {arr.shape}")
        if len(quat_axes) > 1:
            # Prefer last axis if present, else first axis
            chosen_axis = quat_axes[-1]
            arr = np.moveaxis(arr, chosen_axis, -1)
            q_spatial = np.asarray(arr, dtype=np.float32)
        else:
            q_spatial = to_spatial_quat(arr)

    quat_axes_after = [i for i, s in enumerate(q_spatial.shape) if s == 4]
    if len(quat_axes_after) != 1:
        raise ValueError(
            "Input does not resolve to a single quaternion axis. "
            f"Resolved shape={q_spatial.shape}, quaternion-like axes={quat_axes_after}. "
            "This file is likely not an orientation quaternion field (e.g., it may be a symmetry/operator table)."
        )

    # Interpretation A: as loaded is scalar-first
    q_sf = np.asarray(q_spatial, dtype=np.float32)
    # Interpretation B: loaded is scalar-last -> convert to scalar-first
    q_sl_to_sf = to_scalar_first(q_spatial)

    report: dict[str, Any] = {
        "npy_path": args.npy_path,
        "input_shape": list(arr.shape),
        "spatial_shape": list(q_spatial.shape),
        "symmetry": args.symmetry,
        "analysis": {},
    }

    ana_a = _analyze_interpretation(q_sf, sym)
    ana_b = _analyze_interpretation(q_sl_to_sf, sym)

    report["analysis"]["interpret_as_scalar_first"] = {
        k: v for k, v in ana_a.items() if not k.startswith("_")
    }
    report["analysis"]["interpret_as_scalar_last"] = {
        k: v for k, v in ana_b.items() if not k.startswith("_")
    }

    # Optional model alignment (most decisive)
    if args.exp_dir is not None:
        exp_dir = Path(args.exp_dir)
        cfg = load_and_prepare_config(exp_dir / args.config, exp_dir / "logs" / "run_config.json")
        if args.device is not None:
            cfg.device = args.device
        device = torch.device(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))

        model = build_model(cfg).to(device)
        model.eval()
        ckpt_path = _resolve_checkpoint_path(exp_dir, args.checkpoint)
        _load_model_state(model, ckpt_path, device)

        def _model_compare(q_sf_in: np.ndarray, ana: dict[str, Any]) -> dict[str, Any]:
            q_t = torch.from_numpy(q_sf_in.reshape(-1, 4)).to(device=device, dtype=torch.float32)
            q_t = q_t / torch.norm(q_t, dim=1, keepdim=True).clamp_min(1e-12)
            with torch.no_grad():
                q_model_fz, _ = model.reduce_to_fz(q_t, return_op_map=True)
            q_model_np = q_model_fz.detach().cpu().numpy().reshape(q_sf_in.shape)

            mis_left = _mis_deg(ana["_left_fz"], q_model_np)
            mis_right = _mis_deg(ana["_right_fz"], q_model_np)
            return {
                "left_vs_model_deg": _stats(mis_left),
                "right_vs_model_deg": _stats(mis_right),
            }

        report["model_alignment"] = {
            "exp_dir": str(exp_dir),
            "checkpoint": str(ckpt_path),
            "interpret_as_scalar_first": _model_compare(q_sf, ana_a),
            "interpret_as_scalar_last": _model_compare(q_sl_to_sf, ana_b),
        }

        # pick best by minimum mean misorientation to model
        candidates = [
            ("scalar_first", "left", report["model_alignment"]["interpret_as_scalar_first"]["left_vs_model_deg"]["mean"]),
            ("scalar_first", "right", report["model_alignment"]["interpret_as_scalar_first"]["right_vs_model_deg"]["mean"]),
            ("scalar_last", "left", report["model_alignment"]["interpret_as_scalar_last"]["left_vs_model_deg"]["mean"]),
            ("scalar_last", "right", report["model_alignment"]["interpret_as_scalar_last"]["right_vs_model_deg"]["mean"]),
        ]
        best = sorted(candidates, key=lambda x: x[2])[0]
        report["verdict"] = {
            "scalar_interpretation": best[0],
            "symmetry_side": best[1],
            "mean_misorientation_to_model_deg": float(best[2]),
            "criterion": "minimum mean misorientation to model.reduce_to_fz",
        }

    out_json = out_dir / "npy_convention_debug_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved: {out_json}")
    if "verdict" in report:
        v = report["verdict"]
        print(
            f"Verdict -> scalar={v['scalar_interpretation']}, side={v['symmetry_side']}, "
            f"mean_mis={v['mean_misorientation_to_model_deg']:.6f} deg"
        )


if __name__ == "__main__":
    main()
