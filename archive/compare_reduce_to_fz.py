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
import torch 
from orix.quaternion import symmetry as SYM


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


def misorientation_deg_between_wxyz_torch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Torch-accelerated misorientation (degrees) between wxyz numpy arrays.

    Falls back to the numpy implementation if `torch` is not available.
    """
    if torch is None:
        return misorientation_deg_between_wxyz(a, b)
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    flat_a = a_np.reshape(-1, 4).astype(np.float32)
    flat_b = b_np.reshape(-1, 4).astype(np.float32)
    ta = torch.from_numpy(flat_a)
    tb = torch.from_numpy(flat_b)
    # normalize
    nta = ta / ta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ntb = tb / tb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos_half = (nta * ntb).sum(dim=-1).abs().clamp(max=1.0)
    ang = 2.0 * torch.acos(cos_half)
    deg = ang * (180.0 / np.pi)
    return deg.cpu().numpy().reshape(a_np.shape[:-1])


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
    q_red = ori.reduce(verbose=verbose)

    # q_red is an Orientation; get back quaternions
    q_red_wxyz = q_red.data.reshape(orig_shape + (4,))


    if return_order.lower() == "wxyz":
        return q_red_wxyz
    elif return_order.lower() == "xyzw":
        q_red_xyzw = np.stack([q_red_wxyz[..., 1], q_red_wxyz[..., 2], q_red_wxyz[..., 3], q_red_wxyz[..., 0]], axis=-1)
        return q_red_xyzw
    else:
        raise ValueError("return_order must be 'xyzw' or 'wxyz'")


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a ⊗ b, both (...,4) [w,x,y,z]."""
    wa, xa, ya, za = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    wb, xb, yb, zb = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        wa*wb - xa*xb - ya*yb - za*zb,
        wa*xb + xa*wb + ya*zb - za*yb,
        wa*yb - xa*zb + ya*wb + za*xb,
        wa*zb + xa*yb - ya*xb + za*wb,
    ], dim=-1)



def build_fcc_syms_inv() -> torch.Tensor:
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    half = 0.5
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [inv_sqrt_2, -inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, -inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, -inv_sqrt_2],
            [inv_sqrt_2, inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, inv_sqrt_2],
            [0, -inv_sqrt_2, -inv_sqrt_2, 0],
            [0, -inv_sqrt_2, 0, -inv_sqrt_2],
            [0, 0, -inv_sqrt_2, -inv_sqrt_2],
            [0, -inv_sqrt_2, inv_sqrt_2, 0],
            [0, 0, -inv_sqrt_2, inv_sqrt_2],
            [0, -inv_sqrt_2, 0, inv_sqrt_2],
            [half, -half, -half, -half],
            [half, half, half, -half],
            [half, half, -half, half],
            [half, -half, half, half],
            [half, -half, -half, half],
            [half, -half, half, -half],
            [half, half, -half, -half],
            [half, half, half, half],
        ],
        dtype=torch.float32,
    )



def reduce_to_fz_oh(q_1x4: torch.Tensor, sym_inv: torch.Tensor) -> torch.Tensor:
    """
    Reduce q to the Oh fundamental zone via left-multiplication: s_inv ⊗ q.
    Returns the equivalent with maximum w (= minimum misorientation from identity).
    """
    G = sym_inv.shape[0]
    cands = quat_mul(sym_inv, q_1x4.expand(G, -1))   # (G, 4): s_inv ⊗ q
    cands = torch.where(cands[..., 0:1] < 0, -cands, cands)   # enforce w >= 0
    return cands[cands[:, 0].argmax()]                        # (4,) best representative


def reduce_to_fz_oh_many(q_Nx4: torch.Tensor, sym_inv_Gx4: torch.Tensor) -> torch.Tensor:
    """Vectorised reduction of many quaternions into the Oh fundamental zone.

    q_Nx4: (N,4) tensor of quaternions in [w,x,y,z]
    sym_inv_Gx4: (G,4) tensor of symmetry inverse quaternions (conjugates)

    Returns: (N,4) tensor of chosen representatives (w,x,y,z).

    Warning: this fully broadcasts over G and N producing an intermediate of size G*N*4
    floats. Ensure you have enough memory for your input size.
    """
    if q_Nx4.dim() == 1:
        q_Nx4 = q_Nx4.unsqueeze(0)
    # ensure inputs are float tensors on the same device
    device = q_Nx4.device
    dtype = q_Nx4.dtype
    sym = sym_inv_Gx4.to(device=device, dtype=dtype)

    # cands shape: (G, N, 4)
    cands = quat_mul(sym[:, None, :], q_Nx4[None, :, :])
    cands = torch.where(cands[..., 0:1] < 0, -cands, cands)

    # select best scalar part per quaternion
    ws = cands[..., 0]  # (G, N)
    best_idx = ws.argmax(dim=0)  # (N,)
    ar = torch.arange(q_Nx4.shape[0], device=device)
    best = cands[best_idx, ar, :]  # (N,4)
    return best


def reduce_to_fz_oh_image(q_hw4, sym_inv_Gx4) -> "np.ndarray | torch.Tensor":
    """Reduce an image-shaped quaternion array (H,W,4) into the Oh FZ.

    Accepts either a numpy array (H,W,4) or a torch tensor (H,W,4).
    Returns the same type as the input: numpy array or torch tensor with shape (H,W,4).
    """
    input_is_numpy = isinstance(q_hw4, np.ndarray)

    # flatten to (N,4)
    if input_is_numpy:
        q_flat = q_hw4.reshape(-1, 4).astype(np.float32)
        tq = torch.from_numpy(q_flat)
    else:
        if not torch.is_tensor(q_hw4):
            raise TypeError("q_hw4 must be a numpy array or torch tensor")
        tq = q_hw4.reshape(-1, 4).contiguous().to(dtype=torch.float32)

    # ensure sym_inv is a torch tensor on CPU
    if isinstance(sym_inv_Gx4, np.ndarray):
        sym_t = torch.from_numpy(np.asarray(sym_inv_Gx4, dtype=np.float32))
    else:
        sym_t = sym_inv_Gx4.to(dtype=torch.float32)

    # run vectorised reduction (may allocate G*N intermediate)
    reps = reduce_to_fz_oh_many(tq, sym_t)

    # reshape back
    reps = reps.reshape(q_hw4.shape)
    if input_is_numpy:
        return reps.cpu().numpy()
    return reps


def reduce_to_fz_oh_batch_images(q_bhw4, sym_inv_Gx4):
    """
    Reduce a batch of images of quaternions.

    Args:
      q_bhw4: numpy array or torch tensor with shape (B,H,W,4) in [w,x,y,z]
      sym_inv_Gx4: numpy array or torch tensor with shape (G,4) (conjugated/inverse sym ops)

    Returns:
      same-type result with shape (B,H,W,4) of chosen FZ representatives.
    """
    import torch
    is_numpy = isinstance(q_bhw4, np.ndarray)

    if is_numpy:
        B, H, W, _ = q_bhw4.shape
        flat = q_bhw4.reshape(-1, 4).astype(np.float32)
        tq = torch.from_numpy(flat)
    else:
        if not torch.is_tensor(q_bhw4):
            raise TypeError("q_bhw4 must be numpy or torch tensor")
        B, H, W, _ = q_bhw4.shape
        tq = q_bhw4.reshape(-1, 4).contiguous().to(dtype=torch.float32)

    # make sure sym is torch tensor
    if isinstance(sym_inv_Gx4, np.ndarray):
        sym_t = torch.from_numpy(np.asarray(sym_inv_Gx4, dtype=np.float32))
    else:
        sym_t = sym_inv_Gx4.to(dtype=torch.float32)

    # call the vectorised reducer (may allocate G*N intermediate)
    reps = reduce_to_fz_oh_many(tq, sym_t)  # returns (N,4) torch tensor

    reps = reps.reshape(B, H, W, 4)
    if is_numpy:
        return reps.cpu().numpy()
    return reps

    # IPF coloring helper
def ipf_rgb(q_wxyz_hw4: np.ndarray, laue_sym, ref_dir="Z") -> np.ndarray:
    H, W, _ = q_wxyz_hw4.shape
    ori = Orientation(q_wxyz_hw4.reshape(-1, 4)).reshape(H, W)
    ckey = IPFColorKeyTSL(laue_sym)
    ckey.direction = Vector3d((0, 0, 1)) if ref_dir.upper() == "Z" else Vector3d((1, 0, 0))
    return ckey.orientation2color(ori)



if __name__ == "__main__":

    # Interactive prompts with sensible defaults
    default_npy = "/data/warren/materials/EBSD/IN718_2D_SR_x4/Test/Original_Data/Open_718_Test_hr_x_block_0.npy"
    default_out = "debug_fz"
    default_sym = "Oh"
    default_max_samples = None

    # Non-interactive defaults (no prompts)
    npy_path = default_npy
    out_dir = default_out
    sym = default_sym
    max_samples = default_max_samples

    os.makedirs(out_dir, exist_ok=True)

    print("Loading:", npy_path)
    q_xyzw = np.load(npy_path)
    H, W, C = q_xyzw.shape
    assert C == 4, "Expected (H,W,4) XYZW input"

    q_wxyz = xyzw_to_wxyz(q_xyzw)
    sym_obj = getattr(symmetry, sym)

    # ---------------------------------------------------------------------------
    # Tested FZ Reduce
    # ---------------------------------------------------------------------------
    print("Running tested_fz reduction (PyTorch left-multiplication)...")

    sym_ops_np = np.asarray(sym_obj.data, dtype=np.float32)  # (G,4) in wxyz
    sym_inv_np = sym_ops_np.copy()
    sym_inv_np[:, 1:] *= -1.0
    sym_inv_t = torch.from_numpy(sym_inv_np)

    tq = torch.from_numpy(q_wxyz.astype(np.float32))

    q_wxyz_fz_tested = reduce_to_fz_oh_image(q_wxyz, sym_inv_t)



    ori = Orientation(q_wxyz_fz_tested.reshape(-1,4), symmetry=sym_obj).reshape(H, W)

    ckey = IPFColorKeyTSL(sym_obj.laue)
    ref_dirs = {'X': Vector3d.xvector(), 'Y': Vector3d.yvector(), 'Z': Vector3d.zvector()}
    colors = {}
    for name, vec in ref_dirs.items():
        ckey.direction = vec
        colors[name] = ckey.orientation2color(ori)

    colors["Z"].shape
    ori.shape
    fig = plt.figure(figsize=(12, 6))
    ori.scatter(projection='axangle', figure=fig, position=(1, 2, 1))
    # Orientation(ori[1].data, symmetry=sym_obj).scatter(projection='axangle', figure=fig, position=(1, 2, 2), c=colors['Z'][1])
    
    plt.suptitle('q1 (●) vs q2 (■) — IPF colors, Oh symmetry', y=1.01)
    plt.tight_layout()
    plt.show()










    # misorientation_deg_between_wxyz(q_wxyz_fz_tested, q_wxyz).mean()



    # # Run explicit ORIX full-image reduction calls (as in debug_sym_op.py)
    # try:
    #     q_wxyz_full = xyzw_to_wxyz(q_xyzw)
    #     sym_obj = getattr(symmetry, sym)
    #     print("Running ORIX full-image map_into_symmetry_reduced_zone_with_ops(verbose=True) on entire image...")
    #     ori_full = Orientation(q_wxyz_full.reshape(-1, 4), symmetry=sym_obj)
    #     # access reduce attribute if available
    #     try:
    #         _ = ori_full.reduce
    #     except Exception:
    #         pass
    #     q_reduced_full = ori_full.reduce(verbose=False)
    #     # q_reduced_full is Orientation; extract numpy wxyz array and save
    #     q_reduced_wxyz = np.asarray(getattr(q_reduced_full, "data"))
    #     print("ORIX full-image reduction completed (skipping np.save of arrays).")
    # except Exception as exc:
    #     print("ORIX full-image reduction failed:", exc)

    # # Flatten for faster processing and optional subsample
    # flat = q_xyzw.reshape(-1, 4)
    # N = flat.shape[0]
    # if max_samples and N > max_samples:
    #     idx = np.linspace(0, N - 1, max_samples, dtype=np.int64)
    #     flat = flat[idx]
    #     sample_shape = (max_samples,)
    #     print(f"Subsampling {max_samples}/{N} pixels for comparison")
    # else:
    #     idx = None
    #     sample_shape = (N,)

    # # ORIX method (via debug_reduce_to_fz helper) -> returns XYZW by default
    # print("Running ORIX reduction (map_into_symmetry_reduced_zone_with_ops)...")
    # q_orix_xyzw = reduce_to_fz_with_ops_orix(flat.reshape(-1, 4), sym=getattr(symmetry, sym), verbose=False, return_order="xyzw")
    # q_orix_xyzw = q_orix_xyzw.reshape(sample_shape + (4,))

    # # numpy/vectorized method
    # print("Running numpy vectorized reduce_to_fz_min_angle...")
    # q_np_xyzw, op_map = reduce_to_fz_min_angle(flat.reshape(-1, 4), getattr(symmetry, sym), return_op_map=True)
    # q_np_xyzw = q_np_xyzw.reshape(sample_shape + (4,))

    # # model method (requires torch)
    # print("Running model.FCCAutoEncoder.reduce_to_fz (CPU)...")
    # # model expects scalar-first (wxyz) and returns wxyz
    # q_wxyz = xyzw_to_wxyz(flat.reshape(-1, 4))
    # if torch is None:
    #     raise RuntimeError("Torch is required to run the model reduce_to_fz. Activate the environment with torch installed.")

    # model = FCCAutoEncoder(device="cpu")
    # with torch.no_grad():
    #     q_t = torch.from_numpy(q_wxyz.astype(np.float32))
    #     q_model_wxyz = model.reduce_to_fz(q_t)
    #     if isinstance(q_model_wxyz, tuple):
    #         q_model_wxyz = q_model_wxyz[0]
    #     q_model_wxyz = q_model_wxyz.detach().cpu().numpy()

    # q_model_xyzw = wxyz_to_xyzw(q_model_wxyz).reshape(sample_shape + (4,))

    # # --- tested_fz method: PyTorch left-multiplication reduction using conjugated symmetry ops ---
    # q_tested_xyzw = None
    # q_tested_full_wxyz = None
    # try:
    #     if torch is None:
    #         raise RuntimeError("Torch is required to run the tested_fz reduction. Activate the environment with torch installed.")
    #     print("Running tested_fz reduction (PyTorch left-multiplication, vectorised)...")
    #     sym_obj = getattr(symmetry, sym)
    #     sym_ops_np = np.asarray(sym_obj.data, dtype=np.float32)  # (G,4) in wxyz
    #     sym_inv_np = sym_ops_np.copy()
    #     sym_inv_np[:, 1:] *= -1.0

    #     # Sampled reduction (uses flattened `flat` computed above)
    #     q_wxyz_sampled = xyzw_to_wxyz(flat.reshape(-1, 4))
    #     tq_sampled = torch.from_numpy(q_wxyz_sampled.astype(np.float32))
    #     sym_inv_t = torch.from_numpy(sym_inv_np)
    #     q_tested_wxyz = reduce_to_fz_oh_many(tq_sampled, sym_inv_t).detach().cpu().numpy()
    #     q_tested_xyzw = wxyz_to_xyzw(q_tested_wxyz.reshape(-1, 4)).reshape(sample_shape + (4,))

    #     # Full-image reduction (may allocate G*H*W intermediate) — try, but catch OOM or other failures
    #     try:
    #         q_tested_full = reduce_to_fz_oh_image(xyzw_to_wxyz(q_xyzw), sym_inv_np)
    #         # ensure wxyz numpy array for downstream usage
    #         if torch.is_tensor(q_tested_full):
    #             q_tested_full_wxyz = q_tested_full.cpu().numpy()
    #         else:
    #             q_tested_full_wxyz = q_tested_full
    #     except Exception as e_full:
    #         print("Full-image tested_fz reduction failed:", e_full)
    # except Exception as exc:
    #     print("tested_fz reduction failed:", exc)

    # # --- Full-image numpy & model reductions + plotting (may be slow) ---
    # q_model_xyzw = wxyz_to_xyzw(q_model_wxyz).reshape(sample_shape + (4,))

    # # Full-image numpy reduction (vectorized)
    # q_np_full_wxyz = None
    # try:
    #     print("Running full-image numpy reduce_to_fz_min_angle (this may take time)...")
    #     q_np_full_xyzw, _ = reduce_to_fz_min_angle(q_xyzw.reshape(-1, 4), getattr(symmetry, sym), return_op_map=True)
    #     q_np_full_xyzw = q_np_full_xyzw.reshape(q_xyzw.shape)
    #     q_np_full_wxyz = xyzw_to_wxyz(q_np_full_xyzw)
    # except Exception as exc:
    #     print("Full-image numpy reduction failed:", exc)

    # # Model full reduction in chunks to limit memory
    # q_model_full_wxyz = None
    # try:
    #     print("Running full-image model.reduce_to_fz in chunks (CPU)...")
    #     q_wxyz_full_flat = xyzw_to_wxyz(q_xyzw).reshape(-1, 4)
    #     M = q_wxyz_full_flat.shape[0]
    #     chunk = 65536
    #     q_model_full = []
    #     for start in range(0, M, chunk):
    #         end = min(start + chunk, M)
    #         q_chunk = q_wxyz_full_flat[start:end]
    #         with torch.no_grad():
    #             tq = torch.from_numpy(q_chunk.astype(np.float32))
    #             q_out = model.reduce_to_fz(tq)
    #             if isinstance(q_out, tuple):
    #                 q_out = q_out[0]
    #             q_model_full.append(q_out.detach().cpu().numpy())
    #     q_model_full_wxyz = np.concatenate(q_model_full, axis=0).reshape(q_xyzw.shape)
    # except Exception as exc:
    #     print("Full-image model reduction failed:", exc)

    # # Full-image tested_fz reduction (chunked)
    # try:
    #     if 'sym_ops_np' not in locals():
    #         sym_obj = getattr(symmetry, sym)
    #         sym_ops_np = np.asarray(sym_obj.data, dtype=np.float32)
    #     sym_inv_np = sym_ops_np.copy()
    #     sym_inv_np[:, 1:] *= -1.0
    #     sym_inv_t = torch.from_numpy(sym_inv_np)
    #     print("Running full-image tested_fz reduction in chunks (CPU)...")
    #     q_wxyz_full_flat = xyzw_to_wxyz(q_xyzw).reshape(-1, 4)
    #     M = q_wxyz_full_flat.shape[0]
    #     chunk = 65536
    #     q_tested_full = []
    #     for start in range(0, M, chunk):
    #         end = min(start + chunk, M)
    #         q_chunk = torch.from_numpy(q_wxyz_full_flat[start:end].astype(np.float32))
    #         for i in range(q_chunk.shape[0]):
    #             rep = reduce_to_fz_oh(q_chunk[i], sym_inv_t)
    #             q_tested_full.append(rep.detach().cpu().numpy())
    #     q_tested_full_wxyz = np.stack(q_tested_full, axis=0).reshape(q_xyzw.shape)
    # except Exception as exc:
    #     print("Full-image tested_fz reduction failed:", exc)

    # # IPF coloring helper
    # def ipf_rgb(q_wxyz_hw4: np.ndarray, laue_sym, ref_dir="Z") -> np.ndarray:
    #     H, W, _ = q_wxyz_hw4.shape
    #     ori = Orientation(q_wxyz_hw4.reshape(-1, 4)).reshape(H, W)
    #     ckey = IPFColorKeyTSL(laue_sym)
    #     ckey.direction = Vector3d((0, 0, 1)) if ref_dir.upper() == "Z" else Vector3d((1, 0, 0))
    #     return ckey.orientation2color(ori)

    # laue = getattr(symmetry, "Oh")
    # # Save IPF images for original and reduced (if available)
    # try:
    #     orig_rgb = ipf_rgb(xyzw_to_wxyz(q_xyzw), laue, ref_dir="Z")
    #     plt.imsave(os.path.join(out_dir, "ipf_original.png"), orig_rgb)
    # except Exception as exc:
    #     print("Failed to save original IPF image:", exc)

    # try:
    #     if 'q_reduced_wxyz' in locals():
    #         orix_rgb = ipf_rgb(q_reduced_wxyz.reshape(q_xyzw.shape), laue, ref_dir="Z")
    #         plt.imsave(os.path.join(out_dir, "ipf_orix_full.png"), orix_rgb)
    # except Exception as exc:
    #     print("Failed to save ORIX IPF image:", exc)

    # try:
    #     if q_np_full_wxyz is not None:
    #         np_rgb = ipf_rgb(q_np_full_wxyz.reshape(q_xyzw.shape), laue, ref_dir="Z")
    #         plt.imsave(os.path.join(out_dir, "ipf_np_full.png"), np_rgb)
    # except Exception as exc:
    #     print("Failed to save numpy IPF image:", exc)

    # try:
    #     if q_model_full_wxyz is not None:
    #         model_rgb = ipf_rgb(q_model_full_wxyz.reshape(q_xyzw.shape), laue, ref_dir="Z")
    #         plt.imsave(os.path.join(out_dir, "ipf_model_full.png"), model_rgb)
    # except Exception as exc:
    #     print("Failed to save model IPF image:", exc)

    # try:
    #     if q_tested_full_wxyz is not None:
    #         tested_rgb = ipf_rgb(q_tested_full_wxyz.reshape(q_xyzw.shape), laue, ref_dir="Z")
    #         plt.imsave(os.path.join(out_dir, "ipf_tested_full.png"), tested_rgb)
    # except Exception as exc:
    #     print("Failed to save tested_fz IPF image:", exc)

    # # Misorientation heatmaps vs original
    # try:
    #     def save_heatmap(angle_map, fname):
    #         vmax = np.quantile(angle_map[np.isfinite(angle_map)], 0.99)
    #         plt.figure(figsize=(6, 6))
    #         plt.imshow(angle_map, vmin=0, vmax=vmax, cmap="inferno")
    #         plt.colorbar(label="degrees")
    #         plt.axis("off")
    #         plt.tight_layout()
    #         plt.savefig(fname, bbox_inches="tight")
    #         plt.close()

    #     if 'q_reduced_wxyz' in locals():
    #         ang = misorientation_deg_between_wxyz_torch(q_reduced_wxyz.reshape(-1, 4), xyzw_to_wxyz(q_xyzw).reshape(-1, 4)).reshape(q_xyzw.shape[:-1])
    #         save_heatmap(ang, os.path.join(out_dir, "misori_orix_vs_orig.png"))
    #     if q_np_full_wxyz is not None:
    #         ang = misorientation_deg_between_wxyz_torch(q_np_full_wxyz.reshape(-1, 4), xyzw_to_wxyz(q_xyzw).reshape(-1, 4)).reshape(q_xyzw.shape[:-1])
    #         save_heatmap(ang, os.path.join(out_dir, "misori_np_vs_orig.png"))
    #     if q_model_full_wxyz is not None:
    #         ang = misorientation_deg_between_wxyz_torch(q_model_full_wxyz.reshape(-1, 4), xyzw_to_wxyz(q_xyzw).reshape(-1, 4)).reshape(q_xyzw.shape[:-1])
    #         save_heatmap(ang, os.path.join(out_dir, "misori_model_vs_orig.png"))
    #     if q_tested_full_wxyz is not None:
    #         ang = misorientation_deg_between_wxyz(q_tested_full_wxyz.reshape(-1, 4), xyzw_to_wxyz(q_xyzw).reshape(-1, 4)).reshape(q_xyzw.shape[:-1])
    #         save_heatmap(ang, os.path.join(out_dir, "misori_tested_vs_orig.png"))
    # except Exception as exc:
    #     print("Failed to save misorientation heatmaps:", exc)

    # # Evaluate misorientation maps vs ORIX (use scalar-first wxyz internally)
    # q_orix_wxyz = xyzw_to_wxyz(q_orix_xyzw.reshape(-1, 4)).reshape(sample_shape + (4,))
    # q_np_wxyz = xyzw_to_wxyz(q_np_xyzw.reshape(-1, 4)).reshape(sample_shape + (4,))
    # q_model_wxyz = q_model_wxyz.reshape(sample_shape + (4,))

    # q_tested_wxyz = None
    # if q_tested_xyzw is not None:
    #     q_tested_wxyz = xyzw_to_wxyz(q_tested_xyzw.reshape(-1, 4)).reshape(sample_shape + (4,))

    # # original (sampled) in wxyz
    # q_orig_wxyz = xyzw_to_wxyz(flat.reshape(-1, 4)).reshape(sample_shape + (4,))

    # # use torch-accelerated misorientation where available
    # ang_np_vs_orix = misorientation_deg_between_wxyz_torch(q_np_wxyz, q_orix_wxyz)
    # ang_model_vs_orix = misorientation_deg_between_wxyz_torch(q_model_wxyz, q_orix_wxyz)
    # ang_model_vs_np = misorientation_deg_between_wxyz_torch(q_model_wxyz, q_np_wxyz)
    # ang_tested_vs_orix = None
    # ang_tested_vs_orig = None
    # if q_tested_wxyz is not None:
    #     ang_tested_vs_orix = misorientation_deg_between_wxyz(q_tested_wxyz, q_orix_wxyz)
    #     ang_tested_vs_orig = misorientation_deg_between_wxyz(q_tested_wxyz, q_orig_wxyz)

    # # compare each reduced result to the original orientations
    # ang_orix_vs_orig = misorientation_deg_between_wxyz_torch(q_orix_wxyz, q_orig_wxyz)
    # ang_np_vs_orig = misorientation_deg_between_wxyz_torch(q_np_wxyz, q_orig_wxyz)
    # ang_model_vs_orig = misorientation_deg_between_wxyz_torch(q_model_wxyz, q_orig_wxyz)

    # stats = []
    # stats.append(summarize("np_vs_orix", ang_np_vs_orix))
    # stats.append(summarize("model_vs_orix", ang_model_vs_orix))
    # stats.append(summarize("model_vs_np", ang_model_vs_np))
    # if ang_tested_vs_orix is not None:
    #     stats.append(summarize("tested_vs_orix", ang_tested_vs_orix))
    # stats.append(summarize("orix_vs_orig", ang_orix_vs_orig))
    # stats.append(summarize("np_vs_orig", ang_np_vs_orig))
    # stats.append(summarize("model_vs_orig", ang_model_vs_orig))
    # if ang_tested_vs_orig is not None:
    #     stats.append(summarize("tested_vs_orig", ang_tested_vs_orig))

    # report = {
    #     "input_npy": npy_path,
    #     "num_compared": int(np.prod(sample_shape)),
    #     "symmetry": sym,
    #     "stats": stats,
    # }

    # report_path = os.path.join(out_dir, "compare_reduce_to_fz_report.json")
    # with open(report_path, "w") as f:
    #     json.dump(report, f, indent=2)

    # # Skipping np.save of arrays per user request; report JSON still written.
    # print("Skipping saving numpy arrays (reduced_orix_xyzw, reduced_np_xyzw, reduced_model_xyzw, op_map_np).")

    # print("Comparison complete. Report written to:", report_path)
    # for s in stats:
    #     print(f"{s['method']}: mean={s['mean_deg']:.4g}°, p95={s['p95_deg']:.4g}°, max={s['max_deg']:.4g}°")
