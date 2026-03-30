from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from boundary_aware_slerp import (
    apply_sym_ops,
    make_fcc_symmetry_4x4,
    qnorm,
    seam_crossing,
    slerp,
    symmetrize_pair,
)


def compute_thin_gb_mask(labels_np: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """
    Compute a 1-pixel-thin grain-boundary mask from integer grain IDs.

    Args:
        labels_np: (H,W) integer label map
        connectivity: 4 or 8 neighborhood for boundary detection

    Returns:
        gb: (H,W) bool mask, True where a boundary exists
    """
    if labels_np.ndim != 2:
        raise ValueError(f"labels_np must be rank-2, got shape {labels_np.shape}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    labels_np = np.asarray(labels_np)
    H, W = labels_np.shape
    gb = np.zeros((H, W), dtype=bool)

    gb[:, 1:] |= labels_np[:, 1:] != labels_np[:, :-1]
    gb[1:, :] |= labels_np[1:, :] != labels_np[:-1, :]

    if connectivity == 8:
        gb[1:, 1:] |= labels_np[1:, 1:] != labels_np[:-1, :-1]
        gb[1:, :-1] |= labels_np[1:, :-1] != labels_np[:-1, 1:]

    return gb


def _compact_labels(region: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    uniq, inv = torch.unique(region, sorted=True, return_inverse=True)
    compact = inv.view_as(region)
    return compact, uniq


@torch.no_grad()
def curvature_smooth_labels_cuda(
    region: torch.Tensor,
    iterations: int = 40,
    lam: float = 0.15,
) -> torch.Tensor:
    """
    Curvature-like smoothing for discrete labels using iterative diffusion
    in one-hot space + argmax projection.

    Args:
        region: (H,W) or (1,H,W) integer label map
        iterations: number of smoothing steps
        lam: diffusion blend weight in [0,1]

    Returns:
        smoothed label map with the same shape semantics as input
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0,1], got {lam}")

    squeeze_back = False
    if region.ndim == 3:
        if region.shape[0] != 1:
            raise ValueError(f"Expected (1,H,W) or (H,W), got {tuple(region.shape)}")
        region = region[0]
        squeeze_back = True
    elif region.ndim != 2:
        raise ValueError(f"Expected rank-2 or rank-3 labels, got {tuple(region.shape)}")

    region = region.long()
    compact, uniq = _compact_labels(region)  # compact in [0, K-1]
    K = int(uniq.numel())
    if K <= 1 or iterations <= 0:
        out = region.long()
        return out.unsqueeze(0) if squeeze_back else out

    # (H,W,K) -> (1,K,H,W)
    probs = F.one_hot(compact, num_classes=K).permute(2, 0, 1).unsqueeze(0).float()
    probs = probs.to(device=region.device)

    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=region.device,
        dtype=probs.dtype,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(K, 1, 1, 1)

    for _ in range(int(iterations)):
        blur = F.conv2d(probs, kernel, padding=1, groups=K)
        probs = (1.0 - lam) * probs + lam * blur
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

    compact_out = probs.argmax(dim=1)[0]  # (H,W)
    out = uniq[compact_out].long()
    return out.unsqueeze(0) if squeeze_back else out


@torch.no_grad()
def upsample_and_smooth_labels(
    labels_lr: torch.Tensor,
    scale: int,
    iterations: int = 40,
    lam: float = 0.15,
) -> torch.Tensor:
    """
    Nearest-neighbor label upsample followed by curvature-like smoothing.

    Args:
        labels_lr: (H,W) or (1,H,W) long tensor
        scale: integer upsampling factor
        iterations: smoothing iterations
        lam: smoothing blend

    Returns:
        lab_hr_smooth: (H*scale, W*scale) long tensor
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    if labels_lr.ndim == 3:
        if labels_lr.shape[0] != 1:
            raise ValueError(f"Expected (1,H,W) or (H,W), got {tuple(labels_lr.shape)}")
        labels_lr_2d = labels_lr[0]
    elif labels_lr.ndim == 2:
        labels_lr_2d = labels_lr
    else:
        raise ValueError(f"Expected (H,W) or (1,H,W), got {tuple(labels_lr.shape)}")

    labels_lr_2d = labels_lr_2d.long()
    lab_hr = F.interpolate(
        labels_lr_2d.float().unsqueeze(0).unsqueeze(0),
        scale_factor=int(scale),
        mode="nearest",
    )[0, 0].long()

    lab_hr_smooth = curvature_smooth_labels_cuda(
        lab_hr,
        iterations=int(iterations),
        lam=float(lam),
    )
    return lab_hr_smooth.long()


def _build_soft_sdf_from_labels(
    labels_lr: torch.Tensor,
    scale: int,
    sigma: float = 2.0,
    iters: int = 8,
) -> torch.Tensor:
    """
    Build a soft boundary field on HR grid from LR labels.
    Higher values are near boundaries.
    """
    if labels_lr.ndim != 2:
        raise ValueError(f"Expected labels_lr (H,W), got {tuple(labels_lr.shape)}")
    device = labels_lr.device
    H, W = labels_lr.shape
    Hh, Wh = H * scale, W * scale

    gb = torch.zeros((H, W), dtype=torch.float32, device=device)
    gb[:, 1:] = torch.maximum(gb[:, 1:], (labels_lr[:, 1:] != labels_lr[:, :-1]).float())
    gb[1:, :] = torch.maximum(gb[1:, :], (labels_lr[1:, :] != labels_lr[:-1, :]).float())

    gb_hr = F.interpolate(gb[None, None], size=(Hh, Wh), mode="nearest")[0, 0]

    box = (torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32) / 9.0)
    dist = gb_hr.clone()
    for _ in range(int(iters)):
        dist = F.conv2d(dist[None, None], box, padding=1)[0, 0]

    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    half = size // 2

    coords = torch.arange(size, device=device, dtype=torch.float32) - half
    g1 = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum().clamp_min(1e-8)
    g2 = (g1[:, None] * g1[None, :])[None, None]

    sdf = F.conv2d(dist[None, None], g2, padding=half)[0, 0]
    sdf = sdf / sdf.max().clamp_min(1e-8)
    return sdf


class SymBilinearSlerpUpsampleV2(nn.Module):
    """
    Symmetry-aware bilinear SLERP upsampler with optional label and SDF guidance.

    Behavior:
      - `labels is None`: full bilinear SLERP
      - `labels provided`: do not interpolate across LR grain boundaries
      - `sdf provided`: nudge sampling coordinates along SDF normals to reduce staircases
    """

    def __init__(
        self,
        scale_factor: int,
        seam_threshold_deg: float = 0.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.scale = int(scale_factor)
        self.register_buffer("sym_ops", make_fcc_symmetry_4x4(device=device, dtype=dtype))
        self.seam_threshold_deg = float(seam_threshold_deg)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sdf: Optional[torch.Tensor] = None,
        sdf_shift: float = 0.0,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != 4:
            raise ValueError(f"Expected x shape (B,4,H,W), got {tuple(x.shape)}")
        if B != 1:
            raise ValueError("SymBilinearSlerpUpsampleV2 currently expects B=1.")

        x = qnorm(x)
        s = self.scale
        H_out, W_out = H * s, W * s
        device, dtype = x.device, x.dtype

        iy = torch.arange(H_out, device=device, dtype=dtype)
        ix = torch.arange(W_out, device=device, dtype=dtype)
        y_base = (iy + 0.5) / s - 0.5
        x_base = (ix + 0.5) / s - 0.5
        y_grid, x_grid = torch.meshgrid(y_base, x_base, indexing="ij")

        y = y_grid.clone()
        xcoord = x_grid.clone()

        if sdf is not None and float(sdf_shift) != 0.0:
            if sdf.ndim == 3:
                if sdf.shape[0] != 1:
                    raise ValueError(f"Expected sdf (H,W) or (1,H,W), got {tuple(sdf.shape)}")
                sdf_2d = sdf[0]
            elif sdf.ndim == 2:
                sdf_2d = sdf
            else:
                raise ValueError(f"Expected sdf (H,W) or (1,H,W), got {tuple(sdf.shape)}")

            sdf_2d = sdf_2d.to(device=device, dtype=dtype)
            if tuple(sdf_2d.shape) != (H_out, W_out):
                sdf_2d = F.interpolate(
                    sdf_2d[None, None],
                    size=(H_out, W_out),
                    mode="bilinear",
                    align_corners=True,
                )[0, 0]

            sdf_pad = F.pad(sdf_2d[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
            dy = (sdf_pad[2:, 1:-1] - sdf_pad[:-2, 1:-1]) * 0.5
            dx = (sdf_pad[1:-1, 2:] - sdf_pad[1:-1, :-2]) * 0.5

            nrm = torch.sqrt(dx * dx + dy * dy + 1e-12)
            nx = dx / nrm
            ny = dy / nrm

            y = y + float(sdf_shift) * ny
            xcoord = xcoord + float(sdf_shift) * nx

        y0g = torch.floor(y).clamp(0, H - 1).long()
        x0g = torch.floor(xcoord).clamp(0, W - 1).long()
        y1g = (y0g + 1).clamp(0, H - 1)
        x1g = (x0g + 1).clamp(0, W - 1)

        v = (y - y0g.to(dtype)).clamp(0.0, 1.0)
        u = (xcoord - x0g.to(dtype)).clamp(0.0, 1.0)
        u_full = u.reshape(-1)
        v_full = v.reshape(-1)

        q00 = x[:, :, y0g, x0g]
        q01 = x[:, :, y0g, x1g]
        q10 = x[:, :, y1g, x0g]
        q11 = x[:, :, y1g, x1g]

        def flat(q: torch.Tensor) -> torch.Tensor:
            return q.permute(0, 2, 3, 1).reshape(-1, 4)

        q00_f = flat(q00)
        q01_f = flat(q01)
        q10_f = flat(q10)
        q11_f = flat(q11)

        sym_ops = self.sym_ops.to(device=device, dtype=dtype)

        # Full SLERP path when labels are not provided.
        if labels is None:
            q00m, q01m = symmetrize_pair(q00_f, q01_f, sym_ops)
            q0u = slerp(q00m, q01m, u_full)
            q10m, q11m = symmetrize_pair(q10_f, q11_f, sym_ops)
            q1u = slerp(q10m, q11m, u_full)
            q0u_m, q1u_m = symmetrize_pair(q0u, q1u, sym_ops)
            q_uv = slerp(q0u_m, q1u_m, v_full)
            out = q_uv.view(B, H_out, W_out, 4).permute(0, 3, 1, 2).contiguous()
            return qnorm(out)

        # Label-aware path: use LR labels only for corner classification.
        if labels.ndim == 3:
            if labels.shape[0] != 1:
                raise ValueError(f"Expected labels (H,W) or (1,H,W), got {tuple(labels.shape)}")
            lab = labels[0]
        elif labels.ndim == 2:
            lab = labels
        else:
            raise ValueError(f"Expected labels (H,W) or (1,H,W), got {tuple(labels.shape)}")
        lab = lab.to(device=device, dtype=torch.long)

        if tuple(lab.shape) == (H_out, W_out):
            lab = F.interpolate(lab.float()[None, None], size=(H, W), mode="nearest")[0, 0].long()
        elif tuple(lab.shape) != (H, W):
            raise ValueError(
                f"labels must be LR (H,W)={(H,W)} or HR (H_out,W_out)={(H_out,W_out)}, got {tuple(lab.shape)}"
            )

        L00 = lab[y0g, x0g].reshape(-1)
        L01 = lab[y0g, x1g].reshape(-1)
        L10 = lab[y1g, x0g].reshape(-1)
        L11 = lab[y1g, x1g].reshape(-1)

        same_all = (L00 == L01) & (L00 == L10) & (L00 == L11)
        vert_boundary = (L00 == L10) & (L01 == L11) & (L00 != L01)
        horiz_boundary = (L00 == L01) & (L10 == L11) & (L00 != L10)
        complex_boundary = ~(same_all | vert_boundary | horiz_boundary)

        out_flat = torch.empty_like(q00_f)

        idx_same = same_all.nonzero(as_tuple=False).squeeze(1)
        if idx_same.numel() > 0:
            q00_s = q00_f[idx_same]
            q01_s = q01_f[idx_same]
            q10_s = q10_f[idx_same]
            q11_s = q11_f[idx_same]
            u_s = u_full[idx_same]
            v_s = v_full[idx_same]

            q00m, q01m = symmetrize_pair(q00_s, q01_s, sym_ops)
            q0u = slerp(q00m, q01m, u_s)
            q10m, q11m = symmetrize_pair(q10_s, q11_s, sym_ops)
            q1u = slerp(q10m, q11m, u_s)
            q0u_m, q1u_m = symmetrize_pair(q0u, q1u, sym_ops)
            out_flat[idx_same] = slerp(q0u_m, q1u_m, v_s)

        idx_vert = vert_boundary.nonzero(as_tuple=False).squeeze(1)
        if idx_vert.numel() > 0:
            u_v = u_full[idx_vert]
            left_mask = u_v < 0.5
            right_mask = ~left_mask

            if left_mask.any():
                idx_left = idx_vert[left_mask]
                q0m, q1m = symmetrize_pair(q00_f[idx_left], q10_f[idx_left], sym_ops)
                out_flat[idx_left] = slerp(q0m, q1m, v_full[idx_left])
            if right_mask.any():
                idx_right = idx_vert[right_mask]
                q0m, q1m = symmetrize_pair(q01_f[idx_right], q11_f[idx_right], sym_ops)
                out_flat[idx_right] = slerp(q0m, q1m, v_full[idx_right])

        idx_h = horiz_boundary.nonzero(as_tuple=False).squeeze(1)
        if idx_h.numel() > 0:
            v_h = v_full[idx_h]
            top_mask = v_h < 0.5
            bot_mask = ~top_mask

            if top_mask.any():
                idx_top = idx_h[top_mask]
                q0m, q1m = symmetrize_pair(q00_f[idx_top], q01_f[idx_top], sym_ops)
                out_flat[idx_top] = slerp(q0m, q1m, u_full[idx_top])
            if bot_mask.any():
                idx_bot = idx_h[bot_mask]
                q0m, q1m = symmetrize_pair(q10_f[idx_bot], q11_f[idx_bot], sym_ops)
                out_flat[idx_bot] = slerp(q0m, q1m, u_full[idx_bot])

        idx_c = complex_boundary.nonzero(as_tuple=False).squeeze(1)
        if idx_c.numel() > 0:
            y_nn = y.round().clamp(0, H - 1).long()
            x_nn = xcoord.round().clamp(0, W - 1).long()
            q_nn = x[:, :, y_nn, x_nn]
            out_flat[idx_c] = flat(q_nn)[idx_c]

        if self.seam_threshold_deg > 0.0:
            m_top = seam_crossing(q00_f, q01_f, sym_ops, self.seam_threshold_deg)
            m_bot = seam_crossing(q10_f, q11_f, sym_ops, self.seam_threshold_deg)
            seam_count = int(m_top.sum().item() + m_bot.sum().item())
            seam_pairs = int(m_top.numel() + m_bot.numel())
            print(
                f"[SymBilinearSlerpUpsampleV2] seam crossings (LR): "
                f"{seam_count}/{seam_pairs} "
                f"({100.0 * seam_count / max(seam_pairs, 1):.3f}%)"
            )

        out = out_flat.view(B, H_out, W_out, 4).permute(0, 3, 1, 2).contiguous()
        return qnorm(out)


def run_boundary_smoothed_slerp(
    q_lr: torch.Tensor,  # (1,4,H,W)
    labels_lr: torch.Tensor,  # (H,W) long
    scale: int,
    upsampler: nn.Module,
    *,
    smooth_iterations: int = 40,
    smooth_lam: float = 0.15,
    use_sdf: bool = True,
    sdf_shift: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One-call utility for:
      1) upsampling/smoothing LR labels to HR for diagnostics/overlay
      2) running label-aware quaternion upsampling

    Returns:
        q_hr: (1,4,H*scale,W*scale)
        labels_hr_smooth: (H*scale,W*scale) long
    """
    if q_lr.ndim != 4 or q_lr.shape[1] != 4 or q_lr.shape[0] != 1:
        raise ValueError(f"Expected q_lr shape (1,4,H,W), got {tuple(q_lr.shape)}")
    if labels_lr.ndim != 2:
        raise ValueError(f"Expected labels_lr shape (H,W), got {tuple(labels_lr.shape)}")

    device = q_lr.device
    labels_lr = labels_lr.to(device=device, dtype=torch.long)

    labels_hr_smooth = upsample_and_smooth_labels(
        labels_lr,
        scale=scale,
        iterations=smooth_iterations,
        lam=smooth_lam,
    ).to(device)

    sdf_hr = None
    if use_sdf:
        sdf_hr = _build_soft_sdf_from_labels(labels_lr, scale=scale)

    if sdf_shift is None:
        sdf_shift = 1.2 / float(scale) if use_sdf else 0.0

    with torch.inference_mode():
        try:
            q_hr = upsampler(
                q_lr,
                labels=labels_lr,  # LR labels are used for corner grain checks
                sdf=sdf_hr,
                sdf_shift=float(sdf_shift),
            )
        except TypeError:
            # Backward-compatible fallback for older upsamplers.
            q_hr = upsampler(q_lr, labels=labels_lr)

    return q_hr, labels_hr_smooth


__all__ = [
    "SymBilinearSlerpUpsampleV2",
    "compute_thin_gb_mask",
    "curvature_smooth_labels_cuda",
    "upsample_and_smooth_labels",
    "run_boundary_smoothed_slerp",
]

