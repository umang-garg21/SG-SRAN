# -*- coding:utf-8 -*-
"""
File:        test_slerp_interp_ipf.py
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description:
    Symmetry-aware bilinear SLERP-based quaternion upsampling test
    with IPF visualization and seam-crossing diagnostics, using
    4×4 symmetry matrices loaded from O_group.npy and O_group_inv.npy.
"""

import os
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from orix.quaternion import symmetry as SYM

from visualization.ipf_render import render_ipf_image


# =============================================================================
# Quaternion helpers
# =============================================================================

import torch
import math


def make_fcc_symmetry_4x4(device="cpu", dtype=torch.float32):
    """
    Returns:
        sym_ops_4x4 : (24, 4, 4) symmetry operator matrices
    """
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    half = 0.5

    # Your original 24 cubic symmetry quaternions
    ops = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [inv_sqrt_2, inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, inv_sqrt_2],
            [inv_sqrt_2, -inv_sqrt_2, 0, 0],
            [inv_sqrt_2, 0, -inv_sqrt_2, 0],
            [inv_sqrt_2, 0, 0, -inv_sqrt_2],
            [0, inv_sqrt_2, inv_sqrt_2, 0],
            [0, inv_sqrt_2, 0, inv_sqrt_2],
            [0, 0, inv_sqrt_2, inv_sqrt_2],
            [0, inv_sqrt_2, -inv_sqrt_2, 0],
            [0, 0, inv_sqrt_2, -inv_sqrt_2],
            [0, inv_sqrt_2, 0, -inv_sqrt_2],
            [half, half, half, half],
            [half, -half, -half, half],
            [half, -half, half, -half],
            [half, half, -half, -half],
            [half, half, half, -half],
            [half, half, -half, half],
            [half, -half, half, half],
            [half, -half, -half, -half],
        ],
        dtype=dtype,
        device=device,
    )

    # Normalize the quaternions
    ops = ops / ops.norm(dim=1, keepdim=True)

    # Allocate matrix tensor (24, 4, 4)
    mats = torch.empty((ops.shape[0], 4, 4), dtype=dtype, device=device)

    # Build the 4x4 left-multiply matrices
    for i, (w, x, y, z) in enumerate(ops):
        mats[i] = torch.tensor(
            [
                [w, -x, -y, -z],
                [x, w, -z, y],
                [y, z, w, -x],
                [z, -y, x, w],
            ],
            dtype=dtype,
            device=device,
        )

    return mats


def qnorm(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize quaternion magnitude along the quaternion axis.

    Supports:
      - (B,4,H,W)  -> normalize over dim=1
      - (...,4)    -> normalize over last dim
    """
    if q.ndim >= 4 and q.shape[1] == 4:
        # (B,4,H,W) or similar
        return F.normalize(q, dim=1)
    elif q.shape[-1] == 4:
        # (...,4)
        return F.normalize(q, dim=-1)
    else:
        raise ValueError(
            f"qnorm: expected quaternion axis of size 4, got shape {q.shape}"
        )


def qdot(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion dot product along last dimension."""
    return (q1 * q2).sum(dim=-1)


def angle_between(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Angular distance between unit quaternions in radians:
        d(q1, q2) = 2 * acos(|<q1,q2>|)
    Works with broadcasting.
    """
    q1 = qnorm(q1)
    q2 = qnorm(q2)
    dot = qdot(q1, q2).clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot.abs())


def slerp(q0: torch.Tensor, q1: torch.Tensor, t, eps: float = 1e-8) -> torch.Tensor:
    """
    Vectorized SLERP between q0 and q1.

    q0, q1 : (N,4)  or (B,4)
    t      : scalar, (N,), or (N,1)

    Returns:
        (N,4)   (or (B,4)) normalized quaternions
    """
    q0 = qnorm(q0)
    q1 = qnorm(q1)

    # hemisphere correction – take shortest arc
    dot = qdot(q0, q1).unsqueeze(-1)  # (...,1)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.clamp(-1 + 1e-7, 1 - 1e-7)

    theta = torch.acos(dot)  # (...,1)
    sin_theta = torch.sin(theta)  # (...,1)

    # Broadcast t to (...,1)
    if not torch.is_tensor(t):
        t = torch.full(
            (q0.shape[0], 1),
            float(t),
            dtype=q0.dtype,
            device=q0.device,
        )
    else:
        t = t.to(device=q0.device, dtype=q0.dtype)
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # (N,) -> (N,1)
        else:
            t = t.view(-1, 1)  # flatten everything to (N,1)

    w0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w1 = torch.sin(t * theta) / (sin_theta + eps)

    return qnorm(w0 * q0 + w1 * q1)


# =============================================================================
# Symmetry application (4×4 matrices)
# =============================================================================


def apply_sym_ops(sym_ops: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Apply all symmetry operators to quaternions.

    sym_ops: (G,4,4)     4x4 linear ops in quaternion space
    q:       (B,4)       quaternion batch
    returns: (G,B,4)     all symmetry images of each q[b]
    """
    # (G,4,4) x (B,4) -> (G,B,4)
    return torch.einsum("gij,bj->gbi", sym_ops, q)


# =============================================================================
# Symmetry-aware pairing + bilinear SLERP
# =============================================================================


def symmetrize_pair(
    q0: torch.Tensor, q1: torch.Tensor, sym_ops: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Choose g ∈ G such that q0 and (g ⊗ q1) are as close as possible on S^3.

    q0:      (B,4)
    q1:      (B,4)
    sym_ops: (G,4,4)  4×4 symmetry matrices

    returns: q0, q1_best  both (B,4)
    """
    # All symmetry variants of q1: (G,B,4)
    q1_all = apply_sym_ops(sym_ops, q1)

    # Angular distances: q0.unsqueeze(0) -> (1,B,4) broadcasts vs (G,B,4)
    angles = angle_between(q0.unsqueeze(0), q1_all)  # (G,B)
    idx = angles.argmin(dim=0)  # (B,)

    # Gather best symmetry representative for each batch element
    # q1_all: (G,B,4) -> index over first dim with idx (B,)
    q1_best = q1_all[idx, torch.arange(q1.shape[0])]  # (B,4)
    return q0, q1_best


def bilinear_slerp_sym(
    q00: torch.Tensor,
    q01: torch.Tensor,
    q10: torch.Tensor,
    q11: torch.Tensor,
    u: float,
    v: float,
    sym_ops: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetry-aware bilinear SLERP on a single cell.

    q00,q01,q10,q11 : (B,4)
    u,v             : float in [0,1]
    sym_ops         : (G,4,4)
    returns         : (B,4)
    """
    # Horizontal edge at top
    q00m, q01m = symmetrize_pair(q00, q01, sym_ops)
    q0u = slerp(q00m, q01m, u)

    # Horizontal edge at bottom
    q10m, q11m = symmetrize_pair(q10, q11, sym_ops)
    q1u = slerp(q10m, q11m, u)

    # Vertical interpolation between these two
    q0u_m, q1u_m = symmetrize_pair(q0u, q1u, sym_ops)
    q_uv = slerp(q0u_m, q1u_m, v)
    return q_uv


# =============================================================================
# Symmetry-aware bilinear SLERP upsampler (using 4×4 sym matrices)
# =============================================================================


def seam_crossing(
    q0: torch.Tensor,
    q1: torch.Tensor,
    sym_ops: torch.Tensor,
    threshold_deg: float = 5.0,
) -> torch.Tensor:
    """
    Detect if (q0, q1) crosses a symmetry seam.

    q0, q1    : (N,4)
    sym_ops   : (G,4,4)
    returns   : (N,) bool mask
    """
    q0 = qnorm(q0)
    q1 = qnorm(q1)

    q0_sym, q1_sym = symmetrize_pair(q0, q1, sym_ops)
    ang_orig = angle_between(q0, q1)  # (N,)
    ang_sym = angle_between(q0_sym, q1_sym)  # (N,)

    diff_deg = (ang_orig - ang_sym).abs() * 180.0 / math.pi
    return diff_deg > threshold_deg


@torch.no_grad()
def seam_crossing_heatmap(
    q: torch.Tensor, sym_ops: torch.Tensor, threshold_deg: float = 5.0
) -> torch.Tensor:
    """
    Compute a seam-crossing heatmap for a quaternion field.

    Args:
        q : (B,4,H,W) quaternion field (normalized)
        sym_ops : (G,4,4) symmetry operators
        threshold_deg : threshold to classify seam crossing

    Returns:
        heatmap : (H,W) uint8 heatmap (0=no seam, 255=seam)
    """
    B, C, H, W = q.shape
    assert B == 1, "Use B=1 for visualization"
    assert C == 4

    q = qnorm(q)

    # ------------------------------------------------------------
    # Flatten all 4 neighbor edges into 4 big (N,4) lists
    # (H-1)*W vertical edges: q[y,x] -- q[y+1,x]
    # H*(W-1) horizontal edges: q[y,x] -- q[y,x+1]
    # ------------------------------------------------------------
    # --- Vertical edges
    q0_v = q[:, :, 0 : H - 1, :].permute(0, 2, 3, 1).reshape(-1, 4)  # (Nv,4)
    q1_v = q[:, :, 1:H, :].permute(0, 2, 3, 1).reshape(-1, 4)  # (Nv,4)

    # --- Horizontal edges
    q0_h = q[:, :, :, 0 : W - 1].permute(0, 2, 3, 1).reshape(-1, 4)  # (Nh,4)
    q1_h = q[:, :, :, 1:W].permute(0, 2, 3, 1).reshape(-1, 4)  # (Nh,4)

    # ------------------------------------------------------------
    # Seam detection for vertical & horizontal edges
    # ------------------------------------------------------------
    mask_v = seam_crossing(q0_v, q1_v, sym_ops, threshold_deg)  # (Nv,)
    mask_h = seam_crossing(q0_h, q1_h, sym_ops, threshold_deg)  # (Nh,)

    # ------------------------------------------------------------
    # Convert masks back into a logical (H,W) heatmap
    # Each pixel participates in 2 edges:
    #
    #    vertical edges:   heat[y,x] |= mask_v[y,x] or mask_v[y-1,x]
    #    horizontal edges: heat[y,x] |= mask_h[y,x] or mask_h[y,x-1]
    # ------------------------------------------------------------

    heat = torch.zeros((H, W), dtype=torch.uint8, device=q.device)

    # ---- vertical edges contributions ----
    # mask_v is shaped into (H-1, W)
    mv = mask_v.reshape(H - 1, W)
    heat[:-1, :] |= mv.to(torch.uint8) * 255  # seam on top edge
    heat[1:, :] |= mv.to(torch.uint8) * 255  # seam on bottom edge

    # ---- horizontal edges contributions ----
    # mask_h is shaped into (H, W-1)
    mh = mask_h.reshape(H, W - 1)
    heat[:, :-1] |= mh.to(torch.uint8) * 255  # seam on left edge
    heat[:, 1:] |= mh.to(torch.uint8) * 255  # seam on right edge

    return heat


class SymBilinearSlerpUpsample(nn.Module):
    def __init__(
        self,
        scale_factor: int,
        seam_threshold_deg: float = 0.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Symmetry-aware bilinear SLERP upsampler using 4×4 symmetry matrices.

        Args:
            scale_factor       : integer upsampling factor
            seam_threshold_deg : >0 to print seam stats, 0 to disable
        """
        super().__init__()
        self.scale = int(scale_factor)

        # 24 left-multiply 4×4 operators in quaternion space
        sym_ops = make_fcc_symmetry_4x4(device=device, dtype=dtype)
        self.register_buffer("sym_ops", sym_ops)  # (G,4,4)

        self.seam_threshold_deg = float(seam_threshold_deg)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B,4,H,W) scalar-first quaternion field
        return: (B,4,H*s,W*s)
        """
        B, C, H, W = x.shape
        assert C == 4, "Expected 4 quaternion channels (B,4,H,W)"

        s = self.scale
        x = qnorm(x)

        H_out, W_out = H * s, W * s
        device, dtype = x.device, x.dtype

        # ------------------------------------------------------------
        # Build continuous coordinates in LR space for every HR pixel
        # ------------------------------------------------------------
        iy = torch.arange(H_out, device=device, dtype=dtype)  # (H_out,)
        ix = torch.arange(W_out, device=device, dtype=dtype)  # (W_out,)

        y = (iy + 0.5) / s - 0.5  # (H_out,)
        xcoord = (ix + 0.5) / s - 0.5  # (W_out,)

        y0 = torch.floor(y).clamp(0, H - 1).long()  # (H_out,)
        x0 = torch.floor(xcoord).clamp(0, W - 1).long()
        y1 = (y0 + 1).clamp(0, H - 1)
        x1 = (x0 + 1).clamp(0, W - 1)

        v = (y - y0.to(dtype)).clamp(0.0, 1.0)  # (H_out,)
        u = (xcoord - x0.to(dtype)).clamp(0.0, 1.0)  # (W_out,)

        # Make full HR (v,u) grid
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")  # (H_out,W_out)

        # Expand indices to (H_out,W_out) for advanced indexing
        y0g = y0.view(H_out, 1).expand(H_out, W_out)
        y1g = y1.view(H_out, 1).expand(H_out, W_out)
        x0g = x0.view(1, W_out).expand(H_out, W_out)
        x1g = x1.view(1, W_out).expand(H_out, W_out)

        # ------------------------------------------------------------
        # Gather q00, q01, q10, q11 for all HR pixels at once
        # ------------------------------------------------------------
        q00 = x[:, :, y0g, x0g]  # (B,4,H_out,W_out)
        q01 = x[:, :, y0g, x1g]
        q10 = x[:, :, y1g, x0g]
        q11 = x[:, :, y1g, x1g]

        def flat(q: torch.Tensor) -> torch.Tensor:
            # (B,4,H_out,W_out) -> (N,4) where N = B * H_out * W_out
            return q.permute(0, 2, 3, 1).reshape(-1, 4)

        q00_f = flat(q00)
        q01_f = flat(q01)
        q10_f = flat(q10)
        q11_f = flat(q11)

        # Flatten u,v for each (b,iy,ix)
        u_full = u_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1)  # (N,)
        v_full = v_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1)  # (N,)

        sym_ops = self.sym_ops

        # ------------------------------------------------------------
        # Symmetry-aware bilinear SLERP in one big batch
        # ------------------------------------------------------------

        # Top edge: q00 --(u)--> q0u
        q00m, q01m = symmetrize_pair(q00_f, q01_f, sym_ops)
        q0u = slerp(q00m, q01m, u_full)

        # Bottom edge: q10 --(u)--> q1u
        q10m, q11m = symmetrize_pair(q10_f, q11_f, sym_ops)
        q1u = slerp(q10m, q11m, u_full)

        # Vertical interpolation: q0u --(v)--> q_uv
        q0u_m, q1u_m = symmetrize_pair(q0u, q1u, sym_ops)
        q_uv = slerp(q0u_m, q1u_m, v_full)  # (N,4)

        # ------------------------------------------------------------
        # Optional seam-crossing diagnostics (also vectorized)
        # ------------------------------------------------------------
        if self.seam_threshold_deg > 0.0:
            mask_top = seam_crossing(q00_f, q01_f, sym_ops, self.seam_threshold_deg)
            mask_bot = seam_crossing(q10_f, q11_f, sym_ops, self.seam_threshold_deg)

            seam_count = int(mask_top.sum().item() + mask_bot.sum().item())
            seam_pairs = mask_top.numel() + mask_bot.numel()

            print(
                f"[SymBilinearSlerpUpsample] seam crossings: "
                f"{seam_count}/{seam_pairs} "
                f"({100.0 * seam_count / max(seam_pairs, 1):.3f}%)"
            )

        # ------------------------------------------------------------
        # Reshape back to (B,4,H_out,W_out)
        # ------------------------------------------------------------
        out = q_uv.view(B, H_out, W_out, 4).permute(0, 3, 1, 2).contiguous()
        return qnorm(out)


# =============================================================================
# Test harness
# =============================================================================


def test_slerp_on_ipf():
    from training.data_loading import QuaternionDataset

    # from models.final import soft_gb_map_from_quats
    # from utils.quat_ops import normalize_quaternions, is_in_fz, reduce_to_fz_min_angle

    dataset_out_root = "/data/warren/materials/EBSD"
    dataset_name = "IN718_FZ_2D_SR_x4"
    dataset_dir = os.path.join(dataset_out_root, dataset_name)
    train_ds = QuaternionDataset(dataset_root=dataset_dir, split="Test")

    # ---- Load LR quaternion field (4,H,W), scalar-first ----
    q_lr = train_ds[0][0].unsqueeze(0)  # (1,4,H,W)
    q_lr = qnorm(q_lr)

    # ---- Load symmetry 4×4 matrices ----
    sym_np_path = "/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group.npy"
    sym_inv_np_path = (
        "/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group_inv.npy"
    )

    device = q_lr.device
    dtype = q_lr.dtype

    gt = torch.tensor(np.load(sym_np_path), dtype=dtype, device=device)  # (G,4,4)
    gti = torch.tensor(np.load(sym_inv_np_path), dtype=dtype, device=device)  # (G,4,4)
    # NOTE: gti is not strictly needed in this bilinear SLERP, but is available
    # if you later want to map back to a specific FZ via the selected group element.

    sym_ops = gt  # use forward operators for branch tracking

    # For IPF rendering (orix symmetry class)
    sym_class = SYM.O

    scale = 4
    ups = SymBilinearSlerpUpsample(
        scale_factor=scale,
        seam_threshold_deg=5.0,
    ).to(device)

    # ---- Run SLERP interpolation ----
    with torch.inference_mode():
        q_hr = ups(q_lr)

    print(f"Upsampled quaternion tensor: {tuple(q_hr.shape)}")

    # ============================================================
    # Convert our SLERP output to numpy for IPF rendering
    # ============================================================
    q_hr = qnorm(q_hr)

    q_hr_np = q_hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    q_lr_np = q_lr.squeeze(0).permute(1, 2, 0).cpu().numpy()

    out_dir = "outputs/ipf_sym_aware_slerp"
    os.makedirs(out_dir, exist_ok=True)

    render_ipf_image(
        q_lr_np,
        sym_class,
        out_png=f"{out_dir}/ipf_lr.png",
        ref_dir="Z",
        include_key=True,
        overwrite=True,
    )
    render_ipf_image(
        q_hr_np,
        sym_class,
        out_png=f"{out_dir}/ipf_slerp_x{scale}.png",
        ref_dir="Z",
        include_key=True,
        overwrite=True,
    )
    heat = seam_crossing_heatmap(q_hr, sym_ops, threshold_deg=5.0)
    heat_np = heat.cpu().numpy()

    import matplotlib.pyplot as plt

    plt.imshow(heat_np, cmap="hot")
    plt.colorbar()
    plt.savefig(f"{out_dir}/seam_heatmap.png")

    print(f"✅ Saved IPF visualizations to {out_dir}")
    return q_lr_np, q_hr_np


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    test_slerp_on_ipf()
