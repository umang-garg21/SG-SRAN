from __future__ import annotations

import itertools
import math
import os
from typing import Union

import torch
import torch.nn as nn


# =============================================================================
# Quaternion + symmetry helpers
# =============================================================================


def _proper_cubic_group_mats(
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the 24 proper cubic rotations as 3x3 matrices."""
    mats = []
    eye = torch.eye(3, device=device, dtype=dtype)

    for perm in itertools.permutations(range(3)):
        P = eye[list(perm), :]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            S = torch.diag(torch.tensor(signs, device=device, dtype=dtype))
            R = S @ P
            if float(torch.det(R).item()) > 0.0:
                mats.append(R)

    G = torch.stack(mats, dim=0)
    keep = []
    for i in range(G.shape[0]):
        if not any(torch.allclose(G[i], G[j], atol=1e-6, rtol=0.0) for j in keep):
            keep.append(i)
    return G[keep]


def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to scalar-first quaternion [w,x,y,z]."""
    r00, r01, r02 = float(R[0, 0].item()), float(R[0, 1].item()), float(R[0, 2].item())
    r10, r11, r12 = float(R[1, 0].item()), float(R[1, 1].item()), float(R[1, 2].item())
    r20, r21, r22 = float(R[2, 0].item()), float(R[2, 1].item()), float(R[2, 2].item())

    tr = r00 + r11 + r22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s

    q = torch.tensor([w, x, y, z], device=R.device, dtype=R.dtype)
    return q / q.norm().clamp_min(1e-12)


def _left_mult_matrix_wxyz(q: torch.Tensor) -> torch.Tensor:
    """4x4 matrix for left quaternion multiplication: (q ⊗ p) = L(q) p."""
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([w, -x, -y, -z]),
            torch.stack([x, w, -z, y]),
            torch.stack([y, z, w, -x]),
            torch.stack([z, -y, x, w]),
        ],
        dim=0,
    )


def make_fcc_symmetry_4x4(
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return cubic/FCC rotational symmetry ops as quaternion 4x4 linear maps.

    Output shape: (G, 4, 4), with G=24 for proper cubic rotations.
    """
    R = _proper_cubic_group_mats(device=device, dtype=dtype)  # (24,3,3)
    q_syms = torch.stack([_rotmat_to_quat_wxyz(r) for r in R], dim=0)  # (24,4)
    mats = torch.stack([_left_mult_matrix_wxyz(q) for q in q_syms], dim=0)  # (24,4,4)
    return mats


def qnorm(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize quaternion tensors.

    Supports either:
      * (..., 4)  -> normalize along last dim
      * (B, 4, H, W) -> normalize along channel dim
    """
    if q.shape[-1] == 4:
        dim = -1
    elif q.ndim >= 2 and q.shape[1] == 4:
        dim = 1
    else:
        raise ValueError(f"qnorm: expected quaternion axis of size 4, got {q.shape}")
    return q / q.norm(dim=dim, keepdim=True).clamp_min(eps)


def qdot(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion dot product along last dimension."""
    return (q1 * q2).sum(dim=-1)


def angle_between(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic quaternion angle (radians), sign-invariant via abs(dot)."""
    q1 = qnorm(q1)
    q2 = qnorm(q2)
    dot = qdot(q1, q2).clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot.abs())


def slerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    t: Union[float, torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Spherical linear interpolation between quaternions q0 and q1.

    q0, q1: (..., 4)
    t: scalar or tensor broadcastable to q0.shape[:-1]
    """
    if q0.shape[-1] != 4 or q1.shape[-1] != 4:
        raise ValueError(f"slerp: expected (...,4) inputs, got {q0.shape} and {q1.shape}")

    q0 = qnorm(q0)
    q1 = qnorm(q1)

    dot = qdot(q0, q1)
    flip = dot < 0.0
    q1 = torch.where(flip.unsqueeze(-1), -q1, q1)
    dot = torch.where(flip, -dot, dot).clamp(-1.0, 1.0)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    t = torch.as_tensor(t, dtype=q0.dtype, device=q0.device)
    t = t + torch.zeros_like(dot)

    w0 = torch.sin((1.0 - t) * omega) / sin_omega.clamp_min(eps)
    w1 = torch.sin(t * omega) / sin_omega.clamp_min(eps)

    small = sin_omega.abs() < eps
    w0 = torch.where(small, 1.0 - t, w0)
    w1 = torch.where(small, t, w1)

    return qnorm(w0.unsqueeze(-1) * q0 + w1.unsqueeze(-1) * q1)


def apply_sym_ops(sym_ops: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Apply all symmetry ops to a batch of quaternions."""
    if sym_ops.ndim != 3 or sym_ops.shape[-2:] != (4, 4):
        raise ValueError(f"apply_sym_ops: expected (G,4,4), got {tuple(sym_ops.shape)}")
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError(f"apply_sym_ops: expected (B,4), got {tuple(q.shape)}")
    q = qnorm(q)
    out = torch.einsum("gij,bj->gbi", sym_ops, q)
    return qnorm(out)


def symmetrize_pair(
    q0: torch.Tensor,
    q1: torch.Tensor,
    sym_ops: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each pair (q0_i, q1_i), choose symmetry-equivalent q1_i' minimizing angle to q0_i.
    """
    if q0.ndim != 2 or q0.shape[-1] != 4:
        raise ValueError(f"symmetrize_pair: expected q0 (B,4), got {tuple(q0.shape)}")
    if q1.ndim != 2 or q1.shape[-1] != 4:
        raise ValueError(f"symmetrize_pair: expected q1 (B,4), got {tuple(q1.shape)}")
    if q0.shape[0] != q1.shape[0]:
        raise ValueError(f"symmetrize_pair: batch mismatch {q0.shape[0]} vs {q1.shape[0]}")

    q0 = qnorm(q0)
    q1 = qnorm(q1)
    q1_all = apply_sym_ops(sym_ops, q1)  # (G,B,4)

    dots = torch.einsum("gbi,bi->gb", q1_all, q0)  # (G,B)
    best = dots.abs().argmax(dim=0)  # (B,)
    bidx = torch.arange(q0.shape[0], device=q0.device)
    q1_best = q1_all[best, bidx]

    # Resolve antipodal sign to be consistent with q0.
    sign_fix = qdot(q0, q1_best) < 0.0
    q1_best = torch.where(sign_fix.unsqueeze(-1), -q1_best, q1_best)

    return q0, qnorm(q1_best)


def misorientation_deg(
    q0: torch.Tensor,
    q1: torch.Tensor,
    sym_ops: torch.Tensor,
) -> torch.Tensor:
    q0, q1_best = symmetrize_pair(q0, q1, sym_ops)
    ang = angle_between(q0, q1_best)
    return ang * 180.0 / math.pi


# =============================================================================
# Seam diagnostics (unchanged)
# =============================================================================


def seam_crossing(
    q0: torch.Tensor,
    q1: torch.Tensor,
    sym_ops: torch.Tensor,
    threshold_deg: float = 5.0,
) -> torch.Tensor:
    diff_deg = misorientation_deg(q0, q1, sym_ops)
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

    # --- Vertical edges
    q0_v = q[:, :, 0 : H - 1, :].permute(0, 2, 3, 1).reshape(-1, 4)  # (Nv,4)
    q1_v = q[:, :, 1:H, :].permute(0, 2, 3, 1).reshape(-1, 4)  # (Nv,4)

    # --- Horizontal edges
    q0_h = q[:, :, :, 0 : W - 1].permute(0, 2, 3, 1).reshape(-1, 4)  # (Nh,4)
    q1_h = q[:, :, :, 1:W].permute(0, 2, 3, 1).reshape(-1, 4)  # (Nh,4)

    mask_v = seam_crossing(q0_v, q1_v, sym_ops, threshold_deg)  # (Nv,)
    mask_h = seam_crossing(q0_h, q1_h, sym_ops, threshold_deg)  # (Nh,)

    heat = torch.zeros((H, W), dtype=torch.uint8, device=q.device)

    mv = mask_v.reshape(H - 1, W)
    heat[:-1, :] |= mv.to(torch.uint8) * 255
    heat[1:, :] |= mv.to(torch.uint8) * 255

    mh = mask_h.reshape(H, W - 1)
    heat[:, :-1] |= mh.to(torch.uint8) * 255
    heat[:, 1:] |= mh.to(torch.uint8) * 255

    return heat


# =============================================================================
# Symmetry-aware, boundary-aware bilinear upsampler
# =============================================================================


class SymBilinearSlerpUpsample(nn.Module):
    """
    Boundary-aware, symmetry-aware quaternion upsampler.

    Idea:
      * Start from LR quaternion field x (B,4,H,W).
      * For each HR pixel, gather its 4 LR neighbors (q00,q01,q10,q11).
      * Find the nearest LR pixel (q_center).
      * For each corner, compute symmetry-reduced misorientation
        to q_center. If angle <= gb_threshold_deg, that corner
        is considered "same grain" and participates in the blend.
      * Weights are bilinear (u,v), renormalized over valid corners.
      * If <2 valid corners, fallback to nearest neighbor (q_center).

    This is fully vectorized and runs efficiently on CUDA.
    """

    def __init__(
        self,
        scale_factor: int,
        seam_threshold_deg: float = 0.0,
        gb_threshold_deg: float = 5.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        align_corners: bool = True,
    ):
        super().__init__()
        self.scale = int(scale_factor)
        self.align_corners = align_corners
        sym_ops = make_fcc_symmetry_4x4(device=device, dtype=dtype)
        self.register_buffer("sym_ops", sym_ops)
        self.seam_threshold_deg = float(seam_threshold_deg)
        self.gb_threshold_deg = float(gb_threshold_deg)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 4
        s = self.scale
        x = qnorm(x)  # normalize input

        H_out, W_out = H * s, W * s
        device, dtype = x.device, x.dtype

        # --------------------------------------------------------
        # HR coordinate grid
        # --------------------------------------------------------
        if self.align_corners:
            # Match F.interpolate(..., align_corners=True)
            ys = torch.linspace(0, H - 1, H_out, device=device, dtype=dtype)
            xs = torch.linspace(0, W - 1, W_out, device=device, dtype=dtype)
        else:
            # align_corners=False behavior
            iy = torch.arange(H_out, device=device, dtype=dtype)
            ix = torch.arange(W_out, device=device, dtype=dtype)
            ys = (iy + 0.5) / s - 0.5
            xs = (ix + 0.5) / s - 0.5

        y0 = torch.floor(ys).clamp(0, H - 1).long()
        x0 = torch.floor(xs).clamp(0, W - 1).long()
        y1 = (y0 + 1).clamp(0, H - 1)
        x1 = (x0 + 1).clamp(0, W - 1)

        v = (ys - y0.to(dtype)).clamp(0.0, 1.0)
        u = (xs - x0.to(dtype)).clamp(0.0, 1.0)

        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

        y0g = y0.view(H_out, 1).expand(H_out, W_out)
        y1g = y1.view(H_out, 1).expand(H_out, W_out)
        x0g = x0.view(1, W_out).expand(H_out, W_out)
        x1g = x1.view(1, W_out).expand(H_out, W_out)

        # --------------------------------------------------------
        # Gather q00, q01, q10, q11 for all HR pixels
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Nearest LR pixel per HR pixel (for "center" orientation)
        # --------------------------------------------------------
        y_nn = ys.round().clamp(0, H - 1).long()
        x_nn = xs.round().clamp(0, W - 1).long()
        y_nng = y_nn.view(H_out, 1).expand(H_out, W_out)
        x_nng = x_nn.view(1, W_out).expand(H_out, W_out)

        q_center = x[:, :, y_nng, x_nng]  # (B,4,H_out,W_out)
        q_center_f = flat(q_center)  # (N,4)

        # --------------------------------------------------------
        # Bilinear weights for 4 corners
        # --------------------------------------------------------
        u_full = u_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1)  # (N,)
        v_full = v_grid.unsqueeze(0).expand(B, -1, -1).reshape(-1)  # (N,)

        w00 = (1.0 - u_full) * (1.0 - v_full)
        w01 = u_full * (1.0 - v_full)
        w10 = (1.0 - u_full) * v_full
        w11 = u_full * v_full

        w = torch.stack([w00, w01, w10, w11], dim=1)  # (N,4)

        # --------------------------------------------------------
        # Boundary awareness: only corners within misorientation
        # threshold to q_center participate in the blend.
        # --------------------------------------------------------
        sym_ops = self.sym_ops.to(device=device, dtype=dtype)
        thr = self.gb_threshold_deg

        # symmetry-reduced misorientations
        ang00 = misorientation_deg(q_center_f, q00_f, sym_ops)
        ang01 = misorientation_deg(q_center_f, q01_f, sym_ops)
        ang10 = misorientation_deg(q_center_f, q10_f, sym_ops)
        ang11 = misorientation_deg(q_center_f, q11_f, sym_ops)

        valid = torch.stack(
            [
                ang00 <= thr,
                ang01 <= thr,
                ang10 <= thr,
                ang11 <= thr,
            ],
            dim=1,
        )  # (N,4) bool

        # stack corner quats: (N,4_corners,4_components)
        q_corners = torch.stack([q00_f, q01_f, q10_f, q11_f], dim=1)  # (N,4,4)

        # zero out invalid weights
        w_valid = w * valid.to(dtype=w.dtype)  # (N,4)
        sum_w = w_valid.sum(dim=1, keepdim=True)  # (N,1)
        num_valid = valid.sum(dim=1)  # (N,)

        # avoid cross-grain smoothing:
        # if fewer than 2 neighbors are valid, fallback to nearest neighbor
        eps = 1e-8
        use_blend = (num_valid >= 2) & (sum_w.squeeze(1) > eps)

        # normalized weights over valid corners
        w_norm = torch.where(
            sum_w > eps,
            w_valid / (sum_w + eps),
            torch.zeros_like(w_valid),
        )  # (N,4)

        # weighted Euclidean average in quaternion space, then renormalize.
        # Because we only blend orientations within `thr` degrees,
        # this is extremely close to the true Riemannian mean.
        q_blend = (w_norm.unsqueeze(-1) * q_corners).sum(dim=1)  # (N,4)
        q_blend = qnorm(q_blend)

        # choose between blended and nearest neighbor
        out_flat = torch.where(
            use_blend.unsqueeze(-1),
            q_blend,
            q_center_f,
        )  # (N,4)

        # --------------------------------------------------------
        # Optional seam-crossing diagnostics (using LR field)
        # --------------------------------------------------------
        if self.seam_threshold_deg > 0.0:
            # Note: diagnostic still uses LR neighbors, same as before
            mask_top = seam_crossing(q00_f, q01_f, sym_ops, self.seam_threshold_deg)
            mask_bot = seam_crossing(q10_f, q11_f, sym_ops, self.seam_threshold_deg)
            seam_count = int(mask_top.sum().item() + mask_bot.sum().item())
            seam_pairs = int(mask_top.numel() + mask_bot.numel())
            print(
                f"[SymBilinearSlerpUpsample] seam crossings (LR): "
                f"{seam_count}/{seam_pairs} "
                f"({100.0 * seam_count / max(seam_pairs, 1):.3f}%)"
            )

        # --------------------------------------------------------
        # Reshape back to (B,4,H_out,W_out)
        # --------------------------------------------------------
        out = out_flat.view(B, H_out, W_out, 4).permute(0, 3, 1, 2).contiguous()
        return qnorm(out)


# =============================================================================
# Test harness
# =============================================================================


def test_slerp_on_ipf():
    import numpy as np
    import matplotlib.pyplot as plt
    from orix.quaternion import symmetry as SYM

    from training.data_loading import QuaternionDataset
    from visualization.ipf_render import render_ipf_image

    dataset_out_root = "/data/warren/materials/EBSD"
    dataset_name = "IN718_FZ_2D_SR_x4"
    dataset_dir = os.path.join(dataset_out_root, dataset_name)
    train_ds = QuaternionDataset(dataset_root=dataset_dir, split="Test")

    # ---- Load LR quaternion field (4,H,W), scalar-first ----
    q_lr = train_ds[6][0].unsqueeze(0)  # (1,4,H,W)
    q_lr = qnorm(q_lr)

    # ---- Load symmetry 4x4 matrices (fallback to generated ops) ----
    sym_np_path = "/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group.npy"
    device = q_lr.device
    dtype = q_lr.dtype

    if os.path.exists(sym_np_path):
        sym_ops = torch.tensor(np.load(sym_np_path), dtype=dtype, device=device)
    else:
        sym_ops = make_fcc_symmetry_4x4(device=device, dtype=dtype)

    # For IPF rendering (orix symmetry class)
    sym_class = SYM.O

    scale = 4
    ups = SymBilinearSlerpUpsample(
        scale_factor=scale,
        seam_threshold_deg=0.5,  # seam diagnostics
        gb_threshold_deg=1.0,  # grain-boundary cutoff (deg); tweak 3-7
    ).to(device)

    # ---- Run boundary-aware interpolation ----
    with torch.inference_mode():
        q_hr = ups(q_lr)

    print(f"Upsampled quaternion tensor: {tuple(q_hr.shape)}")

    # ============================================================
    # Convert for IPF rendering
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

    # seam heatmap on HR field (optional)
    heat = seam_crossing_heatmap(q_hr, sym_ops, threshold_deg=5.0)
    heat_np = heat.cpu().numpy()

    plt.imshow(heat_np, cmap="hot")
    plt.colorbar()
    plt.savefig(f"{out_dir}/seam_heatmap.png")

    print(f"Saved IPF visualizations to {out_dir}")
    return q_lr_np, q_hr_np


if __name__ == "__main__":
    test_slerp_on_ipf()
