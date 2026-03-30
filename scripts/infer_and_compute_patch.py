#!/usr/bin/env python3
"""Infer SR quaternions for a single LR/HR pair and compute GROD/KAM maps.

Usage:
    python scripts/infer_and_compute_patch.py \
        --exp_dir experiments/IN718/iso_embedding_sr_attn_01 \
        --checkpoint best_model.pt \
        --hr_path /path/to/IN718_QSR_4x1_train_hr_x_block_338.npy \
        --lr_path /path/to/IN718_QSR_4x1_train_lr_x_block_338.npy \
        --out_dir out/infer_patch_338

"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.config_utils import load_and_prepare_config
from inference.infer_iso_embedding_sr_attn import (
    _resolve_checkpoint,
    _load_model_from_checkpoint,
    _flatten_quat_chw,
    _to_hwc_quat_single,
)
from utils.quat_ops import (
    to_spatial_quat,
    format_quaternions,
    normalize_quaternions,
    quat_mul_np,
)
from utils.symmetry_utils import resolve_symmetry
from scipy.ndimage import uniform_filter


def crystallographic_misorientation(q_pred, q_gt, sym_quats=None, degrees=True):
    q_pred = np.asarray(q_pred, dtype=np.float64)
    q_gt = np.asarray(q_gt, dtype=np.float64)
    if q_pred.shape[-1] != 4 or q_gt.shape[-1] != 4:
        raise ValueError('Input quaternions must have last axis length 4')
    # normalize
    def _norm(q):
        q = q.reshape(-1, 4)
        n = np.linalg.norm(q, axis=1).reshape(-1, 1)
        return (q / (n + 1e-12)).reshape(q_pred.shape)

    q_pred = _norm(q_pred)
    q_gt = _norm(q_gt)

    # resolve symmetry quaternions
    if sym_quats is None:
        sym_obj = resolve_symmetry('D6h')
        sym_qs = np.asarray(sym_obj.data, dtype=np.float64)
    elif isinstance(sym_quats, str):
        sym_obj = resolve_symmetry(sym_quats)
        sym_qs = np.asarray(sym_obj.data, dtype=np.float64)
    elif hasattr(sym_quats, 'data'):
        sym_qs = np.asarray(sym_quats.data, dtype=np.float64)
    elif isinstance(sym_quats, np.ndarray):
        sym_qs = np.asarray(sym_quats, dtype=np.float64)
        if sym_qs.ndim != 2 or sym_qs.shape[1] != 4:
            raise ValueError('sym_quats numpy array must have shape (G,4)')
    else:
        raise ValueError('Unknown sym_quats type')

    shape = q_pred.shape[:-1]
    qp = q_pred.reshape(-1, 4)
    qg = q_gt.reshape(-1, 4)

    ops = sym_qs.copy()
    ops[:, 1:] *= -1.0  # conjugate -> inverse for unit quaternions

    # cand: (G, N, 4)
    Gg = ops[:, None, :]
    qg_b = qg[None, :, :]
    gq = quat_mul_np(Gg, qg_b)

    dots = np.abs(np.einsum('gni,ni->gn', gq, qp))  # (G,N)
    dots = np.clip(dots, -1.0, 1.0)
    ang = 2.0 * np.arccos(dots)
    min_ang = np.min(ang, axis=0)
    if degrees:
        min_ang = np.degrees(min_ang)
    return min_ang.reshape(shape)


def compute_grod(ori_map, sym_ops=None, window_size=21):
    ori_map = np.asarray(ori_map, dtype=np.float64)
    H, W = ori_map.shape[:2]
    q = ori_map.copy()
    # normalize
    q = (q / (np.linalg.norm(q.reshape(-1, 4), axis=1).reshape(H, W, 1) + 1e-12))

    M_local = np.zeros((H, W, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(i, 4):
            arr = q[..., i] * q[..., j]
            local = uniform_filter(arr, size=window_size, mode='reflect')
            M_local[..., i, j] = local
            M_local[..., j, i] = local

    M_flat = M_local.reshape(-1, 4, 4)
    w, v = np.linalg.eigh(M_flat)
    avg = v[:, :, -1]
    signs = np.where(avg[:, 0] < 0, -1.0, 1.0).reshape(-1, 1)
    avg = (avg * signs)
    avg = avg.reshape(H, W, 4)
    # normalize avg
    avg = (avg / (np.linalg.norm(avg.reshape(-1, 4), axis=1).reshape(H, W, 1) + 1e-12))

    grod = crystallographic_misorientation(ori_map, avg, sym_quats=sym_ops, degrees=True)
    stats = {
        'mean': float(np.nanmean(grod)),
        'max': float(np.nanmax(grod)),
        'std': float(np.nanstd(grod)),
    }
    return grod, stats


def compute_kam_patch(patch_ori_map, radius=1, sym_ops=None, ignore_threshold_deg=15.0):
    patch = np.asarray(patch_ori_map, dtype=np.float64)
    H, W = patch.shape[:2]
    neighbors = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            neighbors.append((dy, dx))
    vals = np.zeros((len(neighbors), H, W), dtype=np.float64)
    for k, (dy, dx) in enumerate(neighbors):
        src_y0 = max(0, -dy); src_y1 = H - max(0, dy)
        src_x0 = max(0, -dx); src_x1 = W - max(0, dx)
        dst_y0 = max(0, dy); dst_y1 = H - max(0, -dy)
        dst_x0 = max(0, dx); dst_x1 = W - max(0, -dx)
        shifted = np.zeros_like(patch)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = patch[src_y0:src_y1, src_x0:src_x1]
        vm = np.zeros((H, W), dtype=bool)
        vm[dst_y0:dst_y1, dst_x0:dst_x1] = True
        mis = np.full((H, W), np.nan, dtype=np.float64)
        if np.any(vm):
            mis_vals = crystallographic_misorientation(patch[vm], shifted[vm], sym_quats=sym_ops, degrees=True)
            mis[vm] = mis_vals
        if ignore_threshold_deg is not None:
            mis[mis > ignore_threshold_deg] = np.nan
        vals[k] = mis
    kam = np.nanmean(vals, axis=0)
    stats = {'mean': float(np.nanmean(kam)), 'max': float(np.nanmax(kam)), 'std': float(np.nanstd(kam))}
    return kam, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--hr_path", required=True)
    parser.add_argument("--lr_path", default=None)
    parser.add_argument("--out_dir", default="out/infer_patch")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    cfg = load_and_prepare_config(exp_dir / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _resolve_checkpoint(cfg, exp_dir, args.checkpoint)
    print(f"Using checkpoint: {ckpt}")
    model = _load_model_from_checkpoint(cfg, ckpt, device=device)

    hr_path = Path(args.hr_path)
    if args.lr_path:
        lr_path = Path(args.lr_path)
    else:
        # infer LR path by replacing 'hr' -> 'lr' in filename
        lr_path = Path(str(hr_path).replace("_hr_", "_lr_"))

    print(f"Loading HR: {hr_path}")
    print(f"Loading LR: {lr_path}")
    hr = np.load(hr_path)
    lr = np.load(lr_path)

    # Ensure quaternion-last layout (H,W,4)
    hr = to_spatial_quat(hr)
    lr = to_spatial_quat(lr)

    # Convert to float32
    hr = hr.astype(np.float32)
    lr = lr.astype(np.float32)

    # Flatten LR for model
    lr_t = torch.from_numpy(lr)
    lr_flat, lr_shape = _flatten_quat_chw(lr_t)
    lr_flat = lr_flat.to(device=device, dtype=torch.float32)

    hr_h, hr_w = int(hr.shape[0]), int(hr.shape[1])

    with torch.enable_grad():
        sr_flat = model.forward_sr(lr_flat, lr_shape=lr_shape, normalize_input=True)

    sr_np = sr_flat.reshape(hr_h, hr_w, 4).detach().cpu().numpy().astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr_out = out_dir / f"sample_sr_block_{hr_path.stem}.npy"
    np.save(sr_out, sr_np)
    print(f"Saved SR quaternions: {sr_out}")

    # Compute GROD and KAM for the HR and SR patches (use same local-window settings)
    grod_sr, grod_stats_sr = compute_grod(sr_np, sym_ops=resolve_symmetry('Oh'), window_size=21)
    grod_hr, grod_stats_hr = compute_grod(hr, sym_ops=resolve_symmetry('Oh'), window_size=21)
    kam_sr, kam_stats_sr = compute_kam_patch(sr_np, radius=1, sym_ops=resolve_symmetry('Oh'))
    kam_hr, kam_stats_hr = compute_kam_patch(hr, radius=1, sym_ops=resolve_symmetry('Oh'))

    np.save(out_dir / "grod_sr.npy", grod_sr)
    np.save(out_dir / "grod_hr.npy", grod_hr)
    np.save(out_dir / "kam_sr.npy", kam_sr)
    np.save(out_dir / "kam_hr.npy", kam_hr)

    print("GROD HR stats:", grod_stats_hr)
    print("GROD SR stats:", grod_stats_sr)
    print("KAM HR stats:", kam_stats_hr)
    print("KAM SR stats:", kam_stats_sr)
    print(f"All outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()
