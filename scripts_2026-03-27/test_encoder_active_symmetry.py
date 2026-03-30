#!/usr/bin/env python
"""Standalone encoder symmetry check for active-convention A1 features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from visualization.model_stage_walkthrough import load_lr_input, load_model_from_experiment


def _normalize_quaternions(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0.0, -q, q)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    out = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    return _normalize_quaternions(out)


def _quat_to_matrix_active(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quaternions(q)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    matrix = torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    )
    return matrix.reshape(*q.shape[:-1], 3, 3)


def _right_action_quaternions(q: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    sym_batch = sym.view(1, 4).expand(q.shape[0], 4)
    return _quat_mul(q, sym_batch)


def _left_action_quaternions(q: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    sym_batch = sym.view(1, 4).expand(q.shape[0], 4)
    return _quat_mul(sym_batch, q)


def _quat_min_dist_to_set(q: torch.Tensor, qset: torch.Tensor) -> float:
    qn = _normalize_quaternions(q.view(1, 4))
    s = _normalize_quaternions(qset).to(device=qn.device, dtype=qn.dtype)
    dot = torch.abs((qn * s).sum(dim=-1))
    d = torch.sqrt(torch.clamp(2.0 - 2.0 * dot, min=0.0))
    return float(torch.min(d).item())


def _choose_so3_probe_quaternion(sym_ops: torch.Tensor) -> torch.Tensor:
    dtype = sym_ops.dtype
    device = sym_ops.device
    candidates = [
        torch.tensor([0.913, 0.143, 0.287, 0.258], dtype=dtype, device=device),
        torch.tensor([0.801, 0.211, -0.413, 0.376], dtype=dtype, device=device),
        torch.tensor([0.692, -0.361, 0.492, -0.382], dtype=dtype, device=device),
    ]
    for cand in candidates:
        q = _normalize_quaternions(cand.view(1, 4))[0]
        if _quat_min_dist_to_set(q, sym_ops) > 1e-2:
            return q
    return _normalize_quaternions(candidates[0].view(1, 4))[0]


def _error_metrics(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    diff = (x - y).detach()
    rms = float(torch.sqrt(torch.mean(diff * diff)).item())
    rel = float(torch.linalg.norm(diff).item() / (torch.linalg.norm(y.detach()).item() + 1e-12))
    return rel, rms


def _pass(rel: float, rms: float, tol_rel: float, tol_rms: float) -> bool:
    return bool(rel <= tol_rel or rms <= tol_rms)


def _aggregate(metrics: list[dict[str, Any]], *, tol_rel: float, tol_rms: float) -> dict[str, Any]:
    rels = [float(m["rel"]) for m in metrics]
    rmss = [float(m["rms"]) for m in metrics]
    passes = [_pass(float(m["rel"]), float(m["rms"]), tol_rel, tol_rms) for m in metrics]
    return {
        "rel_avg": float(sum(rels) / len(rels)),
        "rms_avg": float(sum(rmss) / len(rmss)),
        "rel_max": float(max(rels)),
        "rms_max": float(max(rmss)),
        "passed_all": bool(all(passes)),
        "n": len(metrics),
    }


def _encode_a1_active_compat(model, quats: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_a1_active"):
        return model.encode_a1_active(quats)

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise AttributeError("Model has no encoder for active A1 encoding.")
    if hasattr(encoder, "forward_a1_active"):
        return encoder.forward_a1_active(quats)

    embedding = getattr(encoder, "embedding", None)
    if embedding is None or not hasattr(embedding, "forward_from_quaternions"):
        raise AttributeError(
            "Active A1 encoding requires model.encode_a1_active, "
            "model.encoder.forward_a1_active, or model.encoder.embedding.forward_from_quaternions."
        )

    q = quats.to(device=embedding.group_mats.device, dtype=embedding.group_mats.dtype)
    return embedding.forward_from_quaternions(q, active_only=True)


def _feature_left_expected(base: torch.Tensor, irreps, sym: torch.Tensor) -> torch.Tensor:
    rot = _quat_to_matrix_active(sym.view(1, 4))[0]
    d = irreps.D_from_matrix(rot.detach().cpu()).to(device=base.device, dtype=base.dtype)
    return base @ d.T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the encoder A1 convention directly on active quaternions. "
            "The script converts dataset passive quaternions to active, then tests "
            "RIGHT_INVARIANCE[G] and LEFT_EQUIVARIANCE[SO(3)]."
        )
    )
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    parser.add_argument("--config", type=str, default="config.json", help="Config filename inside exp_dir.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename in exp_dir/checkpoints, or absolute path. Defaults to best_model.pt when present.",
    )
    parser.add_argument(
        "--fresh_init",
        action="store_true",
        help="Use a freshly initialized model and skip checkpoint loading.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--split",
        type=str,
        default="Test",
        choices=["Train", "Val", "Test"],
        help="Dataset split used when resolving an LR sample from dataset_root.",
    )
    parser.add_argument("--sample_offset", type=int, default=0, help="Which LR/HR pair to inspect.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional dataset_root override. Defaults to the resolved config value.",
    )
    parser.add_argument(
        "--lr_npy",
        type=str,
        default=None,
        help="Optional direct path to one LR quaternion .npy file. Bypasses dataset split lookup.",
    )
    parser.add_argument(
        "--crop_hw",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Optional LR crop size to test only the top-left HxW region.",
    )
    parser.add_argument("--tol_rel", type=float, default=5e-3, help="Relative-error tolerance.")
    parser.add_argument("--tol_rms", type=float, default=2e-4, help="RMS-error tolerance.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any reported property fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-symmetry metrics in addition to the summary.",
    )
    parser.add_argument("--out_json", type=str, default=None, help="Optional path to save the JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, cfg, checkpoint_path = load_model_from_experiment(
        Path(args.exp_dir),
        config_name=args.config,
        checkpoint_name=args.checkpoint,
        prefer_best_checkpoint=not bool(args.fresh_init),
        device=args.device,
    )
    lr_arr, label = load_lr_input(
        cfg,
        split=args.split,
        sample_offset=int(args.sample_offset),
        dataset_root=args.dataset_root,
        lr_npy=args.lr_npy,
        crop_hw=None if args.crop_hw is None else (int(args.crop_hw[0]), int(args.crop_hw[1])),
    )

    q_passive = torch.from_numpy(lr_arr.reshape(-1, 4)).to(device=model.device, dtype=torch.float32)
    q_active = _quat_conjugate(q_passive)
    feat_base = _encode_a1_active_compat(model, q_active)
    sym_ops = model.encoder.sym_ops.detach()
    so3_probe = _choose_so3_probe_quaternion(sym_ops)

    right_metrics: list[dict[str, Any]] = []
    left_g_metrics: list[dict[str, Any]] = []

    for sym_idx in range(int(sym_ops.shape[0])):
        sym = sym_ops[sym_idx]

        feat_right = _encode_a1_active_compat(model, _right_action_quaternions(q_active, sym))
        rel_right, rms_right = _error_metrics(feat_base, feat_right)
        right_metrics.append(
            {
                "sym_index": sym_idx,
                "rel": rel_right,
                "rms": rms_right,
            }
        )

        feat_left = _encode_a1_active_compat(model, _left_action_quaternions(q_active, sym))
        feat_left_expected = _feature_left_expected(feat_base, model.irreps_a1, sym)
        rel_left, rms_left = _error_metrics(feat_left_expected, feat_left)
        left_g_metrics.append(
            {
                "sym_index": sym_idx,
                "rel": rel_left,
                "rms": rms_left,
            }
        )

    feat_left_so3 = _encode_a1_active_compat(model, _left_action_quaternions(q_active, so3_probe))
    feat_left_so3_expected = _feature_left_expected(feat_base, model.irreps_a1, so3_probe)
    left_so3_rel, left_so3_rms = _error_metrics(feat_left_so3_expected, feat_left_so3)

    right_summary = _aggregate(right_metrics, tol_rel=float(args.tol_rel), tol_rms=float(args.tol_rms))
    left_g_summary = _aggregate(left_g_metrics, tol_rel=float(args.tol_rel), tol_rms=float(args.tol_rms))
    left_so3_summary = {
        "rel": float(left_so3_rel),
        "rms": float(left_so3_rms),
        "passed": _pass(float(left_so3_rel), float(left_so3_rms), float(args.tol_rel), float(args.tol_rms)),
    }

    summary = {
        "exp_dir": str(Path(args.exp_dir).resolve()),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "sample": label,
        "stage_1_input_convention": "active",
        "dataset_input_convention": "passive",
        "encoder_feature_family": str(model.irreps_a1),
        "tol_rel": float(args.tol_rel),
        "tol_rms": float(args.tol_rms),
        "num_group_symmetries": int(sym_ops.shape[0]),
        "so3_probe_quaternion": so3_probe.detach().cpu().tolist(),
        "right_invariance_G": right_summary,
        "left_equivariance_G": left_g_summary,
        "left_equivariance_SO3": left_so3_summary,
        "per_symmetry": {
            "right_invariance_G": right_metrics,
            "left_equivariance_G": left_g_metrics,
        },
    }

    print(f"Checkpoint: {checkpoint_path if checkpoint_path is not None else '[random init]'}")
    print(f"Sample: {label}")
    print("Dataset quaternions: passive")
    print("Encoder test input: active q_active = conj(q_passive)")
    print(f"A1 irreps: {model.irreps_a1}")
    print(f"SO(3) probe quaternion: {so3_probe.detach().cpu().tolist()}")
    print()
    print(
        "RIGHT_INVARIANCE[G]: "
        f"rel_avg={right_summary['rel_avg']:.3e} rms_avg={right_summary['rms_avg']:.3e} "
        f"rel_max={right_summary['rel_max']:.3e} rms_max={right_summary['rms_max']:.3e} "
        f"passed_all={right_summary['passed_all']}"
    )
    print(
        "LEFT_EQUIVARIANCE[G]: "
        f"rel_avg={left_g_summary['rel_avg']:.3e} rms_avg={left_g_summary['rms_avg']:.3e} "
        f"rel_max={left_g_summary['rel_max']:.3e} rms_max={left_g_summary['rms_max']:.3e} "
        f"passed_all={left_g_summary['passed_all']}"
    )
    print(
        "LEFT_EQUIVARIANCE[SO(3) probe]: "
        f"rel={left_so3_summary['rel']:.3e} rms={left_so3_summary['rms']:.3e} "
        f"passed={left_so3_summary['passed']}"
    )

    if args.verbose:
        print()
        print("Per-symmetry RIGHT_INVARIANCE[G]")
        for row in right_metrics:
            passed = _pass(float(row["rel"]), float(row["rms"]), float(args.tol_rel), float(args.tol_rms))
            print(
                f"  sym[{row['sym_index']:2d}] rel={row['rel']:.3e} "
                f"rms={row['rms']:.3e} pass={passed}"
            )
        print()
        print("Per-symmetry LEFT_EQUIVARIANCE[G]")
        for row in left_g_metrics:
            passed = _pass(float(row["rel"]), float(row["rms"]), float(args.tol_rel), float(args.tol_rms))
            print(
                f"  sym[{row['sym_index']:2d}] rel={row['rel']:.3e} "
                f"rms={row['rms']:.3e} pass={passed}"
            )

    if args.out_json is not None:
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print()
        print(f"Saved report: {out_json}")

    overall_pass = bool(
        right_summary["passed_all"]
        and left_g_summary["passed_all"]
        and left_so3_summary["passed"]
    )
    if args.strict and not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
