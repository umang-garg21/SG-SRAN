import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from training.config_utils import load_and_prepare_config
from training.quaternion_dataset import QuaternionDataset
from utils.quat_ops import (
    enforce_hemisphere,
    quat_left_multiply_numpy,
    reduce_to_fz_min_angle,
    to_spatial_quat,
)
from utils.symmetry_utils import resolve_symmetry
from visualization.ipf_render import render_ipf_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug whether symmetry ops should be applied on the left or right"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data/warren/materials/EBSD/IN718_FZ_2D_SR_x4",
    )
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--symmetry", type=str, default="O")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/ipf_fcc_sym_ops_side_debug",
    )
    parser.add_argument("--render_all_ops", action="store_true")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument(
        "--source",
        type=str,
        default="decoded",
        choices=["hr", "decoded"],
        help="Quaternion source to test symmetry-side on",
    )

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


def quat_right_multiply_numpy(
    q_left_field: np.ndarray,
    q_right_ops: np.ndarray,
    eps: float = 1e-12,
    normalize: bool = True,
    layout: str = "quat_last",
) -> np.ndarray:
    q_spatial = to_spatial_quat(np.asarray(q_left_field, dtype=np.float32))
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

    if normalize:
        norms = np.linalg.norm(out, axis=2, keepdims=True)
        norms = np.clip(norms, eps, None)
        out = out / norms

    out = out.reshape(m, *spatial_shape, 4)
    if layout == "quat_last":
        return out
    if layout == "quat_first":
        return np.moveaxis(out, -1, 1)
    raise ValueError(f"Invalid layout: {layout}")


def _select_best_by_max_abs_scalar(q_family: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # q_family: (M, H, W, 4)
    w_abs = np.abs(q_family[..., 0])
    best_idx = np.argmax(w_abs, axis=0)

    m, h, w, _ = q_family.shape
    flat_family = q_family.reshape(m, h * w, 4)
    flat_idx = best_idx.reshape(h * w)
    cols = np.arange(h * w)
    best = flat_family[flat_idx, cols].reshape(h, w, 4)
    best = enforce_hemisphere(best, scalar_first=True)
    return best.astype(np.float32, copy=False), best_idx.astype(np.int32, copy=False)


def _mis_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a = q1.reshape(-1, 4)
    b = q2.reshape(-1, 4)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
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


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = QuaternionDataset(dataset_root=args.dataset_root, split=args.split)
    q_hr = ds[args.sample_idx][1]  # (4,H,W)
    q_hr_np = q_hr.cpu().numpy() if hasattr(q_hr, "cpu") else np.array(q_hr)
    q_hr_np = to_spatial_quat(q_hr_np)
    q_source_np = q_hr_np.copy()

    sym = resolve_symmetry(args.symmetry)
    ops = np.asarray(sym.data, dtype=np.float32)

    model = None
    model_fz_np = None
    model_idx_np = None
    ckpt_path = None
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

        with torch.no_grad():
            q_t = torch.from_numpy(q_hr_np.reshape(-1, 4)).to(device=device, dtype=torch.float32)
            q_t = q_t / torch.norm(q_t, dim=1, keepdim=True).clamp_min(1e-12)
            q_dec = model(q_t, normalize_input=True)
            q_dec = q_dec / torch.norm(q_dec, dim=1, keepdim=True).clamp_min(1e-12)
            q_source_np = q_dec.detach().cpu().numpy().reshape(q_hr_np.shape)

            q_model_fz, model_idx = model.reduce_to_fz(q_dec, return_op_map=True)
            model_fz_np = q_model_fz.detach().cpu().numpy().reshape(q_hr_np.shape)
            model_idx_np = model_idx.detach().cpu().numpy().reshape(q_hr_np.shape[:2])

    if args.source == "hr":
        q_source_np = q_hr_np.copy()
    elif args.source == "decoded" and model is None:
        raise ValueError("--source decoded requires --exp_dir / --config / --checkpoint to run model inference")

    left_family = quat_left_multiply_numpy(q_source_np, ops, layout="quat_last")
    right_family = quat_right_multiply_numpy(q_source_np, ops, layout="quat_last")

    left_best, left_idx = _select_best_by_max_abs_scalar(left_family)
    right_best, right_idx = _select_best_by_max_abs_scalar(right_family)

    min_angle_best, min_angle_idx = reduce_to_fz_min_angle(
        q_source_np,
        sym=sym,
        normalize=True,
        hemisphere=True,
        return_op_map=True,
    )
    min_angle_best = enforce_hemisphere(min_angle_best, scalar_first=True)

    # IPF renders
    render_ipf_image(
        q_hr_np,
        sym,
        out_png=str(out_dir / "ipf_input_hr.png"),
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=False,
    )
    render_ipf_image(
        left_best,
        sym,
        out_png=str(out_dir / "ipf_left_best.png"),
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=False,
    )
    render_ipf_image(
        right_best,
        sym,
        out_png=str(out_dir / "ipf_right_best.png"),
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=False,
    )
    render_ipf_image(
        min_angle_best,
        sym,
        out_png=str(out_dir / "ipf_reduce_to_fz_min_angle.png"),
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=False,
    )

    if args.render_all_ops:
        left_dir = out_dir / "left_ops"
        right_dir = out_dir / "right_ops"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        for i in range(left_family.shape[0]):
            render_ipf_image(
                left_family[i],
                sym,
                out_png=str(left_dir / f"ipf_left_op_{i:02d}.png"),
                ref_dir="ALL",
                include_key=True,
                overwrite=True,
                format_input=False,
            )
            render_ipf_image(
                right_family[i],
                sym,
                out_png=str(right_dir / f"ipf_right_op_{i:02d}.png"),
                ref_dir="ALL",
                include_key=True,
                overwrite=True,
                format_input=False,
            )

    mis_left_vs_min = _mis_deg(left_best, min_angle_best)
    mis_right_vs_min = _mis_deg(right_best, min_angle_best)

    report: dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "split": args.split,
        "sample_idx": int(args.sample_idx),
        "source": args.source,
        "symmetry": args.symmetry,
        "shape_hw": list(q_source_np.shape[:2]),
        "metrics": {
            "left_best_vs_reduce_to_fz_min_angle_deg": _stats(mis_left_vs_min),
            "right_best_vs_reduce_to_fz_min_angle_deg": _stats(mis_right_vs_min),
        },
        "index_hist": {
            "left_best": np.bincount(left_idx.reshape(-1), minlength=ops.shape[0]).tolist(),
            "right_best": np.bincount(right_idx.reshape(-1), minlength=ops.shape[0]).tolist(),
            "reduce_to_fz_min_angle": np.bincount(min_angle_idx.reshape(-1), minlength=ops.shape[0]).tolist(),
        },
    }

    if model_fz_np is not None and model_idx_np is not None:
        mis_left_vs_model = _mis_deg(left_best, model_fz_np)
        mis_right_vs_model = _mis_deg(right_best, model_fz_np)
        mis_min_vs_model = _mis_deg(min_angle_best, model_fz_np)

        report["model_comparison"] = {
            "exp_dir": str(args.exp_dir),
            "checkpoint": str(ckpt_path),
            "left_best_vs_model_reduce_to_fz_deg": _stats(mis_left_vs_model),
            "right_best_vs_model_reduce_to_fz_deg": _stats(mis_right_vs_model),
            "min_angle_vs_model_reduce_to_fz_deg": _stats(mis_min_vs_model),
            "model_reduce_to_fz_idx_hist": np.bincount(model_idx_np.reshape(-1), minlength=ops.shape[0]).tolist(),
        }

        # top mismatches for model alignment
        mis_choice = mis_left_vs_model if float(np.mean(mis_left_vs_model)) < float(np.mean(mis_right_vs_model)) else mis_right_vs_model
        chosen_side = "left" if mis_choice is mis_left_vs_model else "right"
        order = np.argsort(-mis_choice)
        top_k = min(int(args.top_k), order.size)
        top_rows = []
        h, w = q_hr_np.shape[:2]
        for rank, flat in enumerate(order[:top_k], start=1):
            r = int(flat // w)
            c = int(flat % w)
            top_rows.append(
                {
                    "rank": rank,
                    "row": r,
                    "col": c,
                    "mis_deg": float(mis_choice[flat]),
                    "chosen_side": chosen_side,
                    "left_idx": int(left_idx[r, c]),
                    "right_idx": int(right_idx[r, c]),
                    "min_angle_idx": int(min_angle_idx[r, c]),
                    "model_idx": int(model_idx_np[r, c]),
                }
            )
        report["top_model_mismatch_pixels"] = top_rows

    out_json = out_dir / "sym_op_side_debug_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved report: {out_json}")
    print(f"Saved IPF images in: {out_dir}")


if __name__ == "__main__":
    main()
