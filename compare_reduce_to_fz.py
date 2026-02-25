"""Compare several "reduce to fundamental zone" implementations.

This script loads an orientation field (scalar-last XYZW numpy file)
and runs multiple FZ-reduction routines:
 - ORIX: `map_into_symmetry_reduced_zone_with_ops` (via debug_reduce_to_fz)
 - numpy/vectorized: `reduce_to_fz_min_angle` (from `utils.quat_ops`)
 - model: `FCCAutoEncoder.reduce_to_fz` (from `models.autoencoder`)

It prints summary misorientation statistics comparing outputs and
saves per-method reduced quaternion arrays and operator maps to `out_dir`.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import json

try:
    import torch
except Exception:
    torch = None

# Ensure repo root is on sys.path so we can import sibling modules when running
# this script from the `scripts/` directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orix.quaternion import symmetry

from orix.quaternion import Orientation, symmetry
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from utils.quat_ops import reduce_to_fz_min_angle

from models.autoencoder import FCCAutoEncoder


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], axis=-1)


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.stack([q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3], q_wxyz[..., 0]], axis=-1)


def misorientation_deg_between_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return misorientation angle in degrees between two wxyz quaternions arrays.

    Both `a` and `b` must be (...,4) scalar-first, unit quaternions.
    angle = 2*acos(|dot(a,b)|) in radians -> converted to degrees.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # flatten last dim
    flat_a = a.reshape(-1, 4)
    flat_b = b.reshape(-1, 4)
    dots = np.abs(np.sum(flat_a * flat_b, axis=1).clip(-1.0, 1.0))
    ang = 2.0 * np.arccos(dots)
    deg = ang * (180.0 / np.pi)
    return deg.reshape(a.shape[:-1])


def summarize(name: str, ang_map: np.ndarray) -> dict:
    arr = ang_map[np.isfinite(ang_map)]
    out = {
        "method": name,
        "mean_deg": float(np.mean(arr)),
        "p95_deg": float(np.quantile(arr, 0.95)),
        "max_deg": float(np.max(arr)),
    }
    return out

def reduce_to_fz_with_ops_orix(
    q_xyzw: np.ndarray,              # scalar-last input (...,4)
    sym=symmetry.O,                  # use symmetry.O for proper cubic (24); symmetry.Oh for Laue (48)
    verbose: bool = True,
    return_order: str = "xyzw",      # "xyzw" or "wxyz"
):
    """
    Reduce quaternions into the symmetry-reduced zone (fundamental zone) and
    return the symmetry operators used.

    Returns
    -------
    q_reduced : ndarray (...,4)
        Reduced quaternions.
    s_l : orix.quaternion.Rotation (or Symmetry-like)
        Left operators used per quaternion (vectorized container).
    s_r : orix.quaternion.Rotation (or Symmetry-like)
        Right operators used per quaternion.
    """
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    assert q_xyzw.shape[-1] == 4, f"Expected (...,4), got {q_xyzw.shape}"

    orig_shape = q_xyzw.shape[:-1]

    # xyzw -> wxyz for orix
    q_wxyz = np.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], axis=-1)

    # normalize + hemisphere
    n = np.linalg.norm(q_wxyz, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    q_wxyz = q_wxyz / n
    q_wxyz = np.where(q_wxyz[..., :1] < 0, -q_wxyz, q_wxyz)

    # Build Orientation and map into reduced zone (FZ) with ops
    ori = Orientation(q_wxyz.reshape(-1, 4), symmetry=sym)

    # This is the call you referenced:
    q_red, s_l, s_r = ori.map_into_symmetry_reduced_zone_with_ops(verbose=verbose)

    # q_red is an Orientation; get back quaternions
    q_red_wxyz = q_red.data.reshape(orig_shape + (4,))

    # Ensure left-side operations are inverted (s^-1) to match
    # the convention used by the numpy/vectorized implementation
    try:
        s_l_data = np.asarray(getattr(s_l, "data"))
        s_l_data_inv = s_l_data.copy()
        s_l_data_inv[..., 1:] *= -1.0
        # try to assign back into s_l if possible
        try:
            s_l.data[...] = s_l_data_inv
        except Exception:
            s_l = s_l_data_inv
    except Exception:
        # fallback: if s_l is already an array-like
        try:
            s_l = np.asarray(s_l)
            s_l[..., 1:] *= -1.0
        except Exception:
            pass

    if return_order.lower() == "wxyz":
        return q_red_wxyz, s_l, s_r
    elif return_order.lower() == "xyzw":
        q_red_xyzw = np.stack([q_red_wxyz[..., 1], q_red_wxyz[..., 2], q_red_wxyz[..., 3], q_red_wxyz[..., 0]], axis=-1)
        return q_red_xyzw, s_l, s_r
    else:
        raise ValueError("return_order must be 'xyzw' or 'wxyz'")

if __name__ == "__main__":

    # Interactive prompts with sensible defaults
    default_npy = "/data/warren/materials/EBSD/IN718_2D_SR_x4/Test/Original_Data/Open_718_Test_hr_x_block_0.npy"
    default_out = "debug_fz"
    default_max = "200000"
    default_sym = "Oh"

    # Non-interactive defaults (no prompts)
    npy_path = default_npy
    out_dir = default_out
    max_samples = int(default_max) if default_max else 0
    sym = default_sym

    os.makedirs(out_dir, exist_ok=True)

    print("Loading:", npy_path)
    q_xyzw = np.load(npy_path)
    H, W, C = q_xyzw.shape
    assert C == 4, "Expected (H,W,4) XYZW input"

    # Run explicit ORIX full-image reduction calls (as in debug_sym_op.py)
    try:
        q_wxyz_full = xyzw_to_wxyz(q_xyzw)
        sym_obj = getattr(symmetry, sym)
        print("Running ORIX full-image map_into_symmetry_reduced_zone_with_ops(verbose=True) on entire image...")
        ori_full = Orientation(q_wxyz_full.reshape(-1, 4), symmetry=sym_obj)
        # access reduce attribute if available
        try:
            _ = ori_full.reduce
        except Exception:
            pass
        q_reduced_full, s_l_full, s_r_full = ori_full.map_into_symmetry_reduced_zone_with_ops(verbose=False)
        # q_reduced_full is Orientation; extract numpy wxyz array and save
        q_reduced_wxyz = np.asarray(getattr(q_reduced_full, "data"))
        print("ORIX full-image reduction completed (skipping np.save of arrays).")
    except Exception as exc:
        print("ORIX full-image reduction failed:", exc)

    # Flatten for faster processing and optional subsample
    flat = q_xyzw.reshape(-1, 4)
    N = flat.shape[0]
    if max_samples and N > max_samples:
        idx = np.linspace(0, N - 1, max_samples, dtype=np.int64)
        flat = flat[idx]
        sample_shape = (max_samples,)
        print(f"Subsampling {max_samples}/{N} pixels for comparison")
    else:
        idx = None
        sample_shape = (N,)

    # ORIX method (via debug_reduce_to_fz helper) -> returns XYZW by default
    print("Running ORIX reduction (map_into_symmetry_reduced_zone_with_ops)...")
    q_orix_xyzw, s_l, s_r = reduce_to_fz_with_ops_orix(flat.reshape(-1, 4), sym=getattr(symmetry, sym), verbose=False, return_order="xyzw")
    q_orix_xyzw = q_orix_xyzw.reshape(sample_shape + (4,))

    # numpy/vectorized method
    print("Running numpy vectorized reduce_to_fz_min_angle...")
    q_np_xyzw, op_map = reduce_to_fz_min_angle(flat.reshape(-1, 4), getattr(symmetry, sym), return_op_map=True)
    q_np_xyzw = q_np_xyzw.reshape(sample_shape + (4,))

    # model method (requires torch)
    print("Running model.FCCAutoEncoder.reduce_to_fz (CPU)...")
    # model expects scalar-first (wxyz) and returns wxyz
    q_wxyz = xyzw_to_wxyz(flat.reshape(-1, 4))
    if torch is None:
        raise RuntimeError("Torch is required to run the model reduce_to_fz. Activate the environment with torch installed.")

    model = FCCAutoEncoder(device="cpu")
    with torch.no_grad():
        q_t = torch.from_numpy(q_wxyz.astype(np.float32))
        q_model_wxyz = model.reduce_to_fz(q_t)
        if isinstance(q_model_wxyz, tuple):
            q_model_wxyz = q_model_wxyz[0]
        q_model_wxyz = q_model_wxyz.detach().cpu().numpy()

    q_model_xyzw = wxyz_to_xyzw(q_model_wxyz).reshape(sample_shape + (4,))

    # --- Full-image numpy & model reductions + plotting (may be slow) ---
    q_model_xyzw = wxyz_to_xyzw(q_model_wxyz).reshape(sample_shape + (4,))

    # Full-image numpy reduction (vectorized)
    q_np_full_wxyz = None
    try:
        print("Running full-image numpy reduce_to_fz_min_angle (this may take time)...")
        q_np_full_xyzw, _ = reduce_to_fz_min_angle(q_xyzw.reshape(-1, 4), getattr(symmetry, sym), return_op_map=True)
        q_np_full_xyzw = q_np_full_xyzw.reshape(q_xyzw.shape)
        q_np_full_wxyz = xyzw_to_wxyz(q_np_full_xyzw)
    except Exception as exc:
        print("Full-image numpy reduction failed:", exc)

    # Model full reduction in chunks to limit memory
    q_model_full_wxyz = None
    try:
        print("Running full-image model.reduce_to_fz in chunks (CPU)...")
        q_wxyz_full_flat = xyzw_to_wxyz(q_xyzw).reshape(-1, 4)
        M = q_wxyz_full_flat.shape[0]
        chunk = 65536
        q_model_full = []
        for start in range(0, M, chunk):
            end = min(start + chunk, M)
            q_chunk = q_wxyz_full_flat[start:end]
            with torch.no_grad():
                tq = torch.from_numpy(q_chunk.astype(np.float32))
                q_out = model.reduce_to_fz(tq)
                if isinstance(q_out, tuple):
                    q_out = q_out[0]
                q_model_full.append(q_out.detach().cpu().numpy())
        q_model_full_wxyz = np.concatenate(q_model_full, axis=0).reshape(q_xyzw.shape)
    except Exception as exc:
        print("Full-image model reduction failed:", exc)

    # IPF coloring helper
    def ipf_rgb(q_wxyz_hw4: np.ndarray, laue_sym, ref_dir="Z") -> np.ndarray:
        H, W, _ = q_wxyz_hw4.shape
        ori = Orientation(q_wxyz_hw4.reshape(-1, 4)).reshape(H, W)
        ckey = IPFColorKeyTSL(laue_sym)
        ckey.direction = Vector3d((0, 0, 1)) if ref_dir.upper() == "Z" else Vector3d((1, 0, 0))
        return ckey.orientation2color(ori)

    laue = getattr(symmetry, "Oh")
    # Save IPF images for original and reduced (if available)
    try:
        orig_rgb = ipf_rgb(xyzw_to_wxyz(q_xyzw), laue, ref_dir="Z")
        plt.imsave(os.path.join(out_dir, "ipf_original.png"), orig_rgb)
    except Exception as exc:
        print("Failed to save original IPF image:", exc)

    try:
        if 'q_reduced_wxyz' in locals():
            orix_rgb = ipf_rgb(q_reduced_wxyz.reshape(q_xyzw.shape), laue, ref_dir="Z")
            plt.imsave(os.path.join(out_dir, "ipf_orix_full.png"), orix_rgb)
    except Exception as exc:
        print("Failed to save ORIX IPF image:", exc)

    try:
        if q_np_full_wxyz is not None:
            np_rgb = ipf_rgb(q_np_full_wxyz.reshape(q_xyzw.shape), laue, ref_dir="Z")
            plt.imsave(os.path.join(out_dir, "ipf_np_full.png"), np_rgb)
    except Exception as exc:
        print("Failed to save numpy IPF image:", exc)

    try:
        if q_model_full_wxyz is not None:
            model_rgb = ipf_rgb(q_model_full_wxyz.reshape(q_xyzw.shape), laue, ref_dir="Z")
            plt.imsave(os.path.join(out_dir, "ipf_model_full.png"), model_rgb)
    except Exception as exc:
        print("Failed to save model IPF image:", exc)

    # Misorientation heatmaps vs original
    try:
        def save_heatmap(angle_map, fname):
            vmax = np.quantile(angle_map[np.isfinite(angle_map)], 0.99)
            plt.figure(figsize=(6, 6))
            plt.imshow(angle_map, vmin=0, vmax=vmax, cmap="inferno")
            plt.colorbar(label="degrees")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(fname, bbox_inches="tight")
            plt.close()

        if 'q_reduced_wxyz' in locals():
            ang = misorientation_deg_between_wxyz(q_reduced_wxyz.reshape(-1, 4), xyzw_to_wxyz(q_xyzw).reshape(-1, 4)).reshape(q_xyzw.shape[:-1])
            save_heatmap(ang, os.path.join(out_dir, "misori_orix_vs_orig.png"))
        if q_np_full_wxyz is not None:
            ang = misorientation_deg_between_wxyz(q_np_full_wxyz.reshape(-1, 4), xyzw_to_wxyz(q_xyzw).reshape(-1, 4)).reshape(q_xyzw.shape[:-1])
            save_heatmap(ang, os.path.join(out_dir, "misori_np_vs_orig.png"))
        if q_model_full_wxyz is not None:
            ang = misorientation_deg_between_wxyz(q_model_full_wxyz.reshape(-1, 4), xyzw_to_wxyz(q_xyzw).reshape(-1, 4)).reshape(q_xyzw.shape[:-1])
            save_heatmap(ang, os.path.join(out_dir, "misori_model_vs_orig.png"))
    except Exception as exc:
        print("Failed to save misorientation heatmaps:", exc)

    # Evaluate misorientation maps vs ORIX (use scalar-first wxyz internally)
    q_orix_wxyz = xyzw_to_wxyz(q_orix_xyzw.reshape(-1, 4)).reshape(sample_shape + (4,))
    q_np_wxyz = xyzw_to_wxyz(q_np_xyzw.reshape(-1, 4)).reshape(sample_shape + (4,))
    q_model_wxyz = q_model_wxyz.reshape(sample_shape + (4,))

    # original (sampled) in wxyz
    q_orig_wxyz = xyzw_to_wxyz(flat.reshape(-1, 4)).reshape(sample_shape + (4,))

    ang_np_vs_orix = misorientation_deg_between_wxyz(q_np_wxyz, q_orix_wxyz)
    ang_model_vs_orix = misorientation_deg_between_wxyz(q_model_wxyz, q_orix_wxyz)
    ang_model_vs_np = misorientation_deg_between_wxyz(q_model_wxyz, q_np_wxyz)

    # compare each reduced result to the original orientations
    ang_orix_vs_orig = misorientation_deg_between_wxyz(q_orix_wxyz, q_orig_wxyz)
    ang_np_vs_orig = misorientation_deg_between_wxyz(q_np_wxyz, q_orig_wxyz)
    ang_model_vs_orig = misorientation_deg_between_wxyz(q_model_wxyz, q_orig_wxyz)

    stats = []
    stats.append(summarize("np_vs_orix", ang_np_vs_orix))
    stats.append(summarize("model_vs_orix", ang_model_vs_orix))
    stats.append(summarize("model_vs_np", ang_model_vs_np))
    stats.append(summarize("orix_vs_orig", ang_orix_vs_orig))
    stats.append(summarize("np_vs_orig", ang_np_vs_orig))
    stats.append(summarize("model_vs_orig", ang_model_vs_orig))

    report = {
        "input_npy": npy_path,
        "num_compared": int(np.prod(sample_shape)),
        "symmetry": sym,
        "stats": stats,
    }

    report_path = os.path.join(out_dir, "compare_reduce_to_fz_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Skipping np.save of arrays per user request; report JSON still written.
    print("Skipping saving numpy arrays (reduced_orix_xyzw, reduced_np_xyzw, reduced_model_xyzw, op_map_np).")

    print("Comparison complete. Report written to:", report_path)
    for s in stats:
        print(f"{s['method']}: mean={s['mean_deg']:.4g}°, p95={s['p95_deg']:.4g}°, max={s['max_deg']:.4g}°")
