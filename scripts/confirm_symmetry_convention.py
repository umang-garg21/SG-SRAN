import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from training.config_utils import load_and_prepare_config
from training.quaternion_dataset import QuaternionDataset
from utils.quat_ops import enforce_hemisphere, quat_left_multiply_numpy, reduce_to_fz_min_angle, to_spatial_quat
from utils.symmetry_utils import resolve_symmetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confirm left/right symmetry convention with quantitative checks")
    parser.add_argument("--dataset_root", type=str, default="/data/warren/materials/EBSD/IN718_FZ_2D_SR_x4")
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--symmetry", type=str, default="O")
    parser.add_argument("--source", type=str, default="decoded", choices=["hr", "decoded"])

    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
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
    q_spatial = to_spatial_quat(np.asarray(q_left_field, dtype=np.float32))
    ops = np.asarray(q_right_ops, dtype=np.float32)

    h, w = q_spatial.shape[:2]
    n = h * w
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
    return out.reshape(m, h, w, 4)


def _select_max_abs_scalar(q_family: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # q_family: (M, H, W, 4)
    m, h, w, _ = q_family.shape
    w_abs = np.abs(q_family[..., 0])
    best_idx = np.argmax(w_abs, axis=0)
    flat_family = q_family.reshape(m, h * w, 4)
    cols = np.arange(h * w)
    selected = flat_family[best_idx.reshape(-1), cols].reshape(h, w, 4)
    selected = enforce_hemisphere(selected, scalar_first=True)
    return selected.astype(np.float32, copy=False), best_idx.astype(np.int32)


def _mis_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a = q1.reshape(-1, 4)
    b = q2.reshape(-1, 4)
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


def main() -> None:
    args = parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "visualizations" / "symmetry_convention_confirm"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_and_prepare_config(exp_dir / args.config, exp_dir / "logs" / "run_config.json")
    if args.device is not None:
        cfg.device = args.device
    device = torch.device(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg).to(device)
    model.eval()
    ckpt_path = _resolve_checkpoint_path(exp_dir, args.checkpoint)
    _load_model_state(model, ckpt_path, device)

    ds = QuaternionDataset(dataset_root=args.dataset_root, split=args.split)
    q_hr = ds[args.sample_idx][1]
    q_hr_np = to_spatial_quat(q_hr.cpu().numpy() if hasattr(q_hr, "cpu") else np.array(q_hr))

    with torch.no_grad():
        q_hr_t = torch.from_numpy(q_hr_np.reshape(-1, 4)).to(device=device, dtype=torch.float32)
        q_hr_t = q_hr_t / torch.norm(q_hr_t, dim=1, keepdim=True).clamp_min(1e-12)
        q_dec_t = model(q_hr_t, normalize_input=True)
        q_dec_t = q_dec_t / torch.norm(q_dec_t, dim=1, keepdim=True).clamp_min(1e-12)
        q_model_fz_t, model_idx_t = model.reduce_to_fz(q_dec_t, return_op_map=True)

    h, w = q_hr_np.shape[:2]
    q_dec_np = q_dec_t.detach().cpu().numpy().reshape(h, w, 4)
    q_model_fz_np = q_model_fz_t.detach().cpu().numpy().reshape(h, w, 4)
    model_idx_np = model_idx_t.detach().cpu().numpy().reshape(h, w)

    q_source_np = q_hr_np if args.source == "hr" else q_dec_np

    sym = resolve_symmetry(args.symmetry)
    ops = np.asarray(sym.data, dtype=np.float32)

    left_family = quat_left_multiply_numpy(q_source_np, ops, layout="quat_last")
    right_family = quat_right_multiply_numpy(q_source_np, ops)

    left_best, left_idx = _select_max_abs_scalar(left_family)
    right_best, right_idx = _select_max_abs_scalar(right_family)

    min_angle_best, min_angle_idx = reduce_to_fz_min_angle(
        q_source_np,
        sym=sym,
        normalize=True,
        hemisphere=True,
        return_op_map=True,
    )
    min_angle_best = enforce_hemisphere(min_angle_best, scalar_first=True)

    # Key checks
    mis_left_vs_model = _mis_deg(left_best, q_model_fz_np)
    mis_right_vs_model = _mis_deg(right_best, q_model_fz_np)
    mis_min_vs_model = _mis_deg(min_angle_best, q_model_fz_np)

    # Invariant-feature consistency check
    with torch.no_grad():
        left_t = torch.from_numpy(left_best.reshape(-1, 4)).to(device=device, dtype=torch.float32)
        right_t = torch.from_numpy(right_best.reshape(-1, 4)).to(device=device, dtype=torch.float32)
        src_t = torch.from_numpy(q_source_np.reshape(-1, 4)).to(device=device, dtype=torch.float32)
        src_t = src_t / torch.norm(src_t, dim=1, keepdim=True).clamp_min(1e-12)
        left_t = left_t / torch.norm(left_t, dim=1, keepdim=True).clamp_min(1e-12)
        right_t = right_t / torch.norm(right_t, dim=1, keepdim=True).clamp_min(1e-12)

        f4_src, f6_src = model.encode(src_t)
        f4_left, f6_left = model.encode(left_t)
        f4_right, f6_right = model.encode(right_t)

        feat_err_left = torch.mean((f4_left - f4_src) ** 2, dim=1) + torch.mean((f6_left - f6_src) ** 2, dim=1)
        feat_err_right = torch.mean((f4_right - f4_src) ** 2, dim=1) + torch.mean((f6_right - f6_src) ** 2, dim=1)

    mean_left = float(np.mean(mis_left_vs_model))
    mean_right = float(np.mean(mis_right_vs_model))
    if mean_right < mean_left:
        verdict = "right"
    elif mean_left < mean_right:
        verdict = "left"
    else:
        verdict = "tie"

    report = {
        "source": args.source,
        "checkpoint": str(ckpt_path),
        "shape_hw": [h, w],
        "verdict": {
            "recommended_side": verdict,
            "criterion": "lower mean misorientation to model.reduce_to_fz",
        },
        "model_alignment_deg": {
            "left_best_vs_model_reduce_to_fz": _stats(mis_left_vs_model),
            "right_best_vs_model_reduce_to_fz": _stats(mis_right_vs_model),
            "min_angle_vs_model_reduce_to_fz": _stats(mis_min_vs_model),
        },
        "feature_consistency_mse": {
            "left_vs_source": _stats(feat_err_left.detach().cpu().numpy()),
            "right_vs_source": _stats(feat_err_right.detach().cpu().numpy()),
        },
        "index_hist": {
            "left_best": np.bincount(left_idx.reshape(-1), minlength=ops.shape[0]).tolist(),
            "right_best": np.bincount(right_idx.reshape(-1), minlength=ops.shape[0]).tolist(),
            "reduce_to_fz_min_angle": np.bincount(min_angle_idx.reshape(-1), minlength=ops.shape[0]).tolist(),
            "model_reduce_to_fz": np.bincount(model_idx_np.reshape(-1), minlength=ops.shape[0]).tolist(),
        },
    }

    out_json = out_dir / f"convention_confirm_{args.source}_sample_{args.sample_idx}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved: {out_json}")
    print(f"Recommended side: {verdict}")
    print(f"Mean misalignment left vs model:  {mean_left:.6f} deg")
    print(f"Mean misalignment right vs model: {mean_right:.6f} deg")


if __name__ == "__main__":
    main()
