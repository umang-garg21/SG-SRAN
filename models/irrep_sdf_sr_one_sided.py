# ---------------------------------------------------------------------------
# Irrep-respecting boundary-aware SDF SR
# Standalone module version.
#
# Overview
# --------
# This model keeps representation learning in A1 irreps feature space while
# boundary reasoning is done in scalar channels derived from pairwise irrep
# comparisons. The scalar branch predicts boundary/SDF geometry that drives a
# shifted upsampler, then equivariant HR cleanup/refinement is applied before
# optional decoding back to quaternions.
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps
from models.SR_double_conv_SRattn_a1_boundary_refine import (
    BoundaryRefinementHead,
    CubochoricOptimizingLocalIsoDecoder,
    EquivariantSpatialConv,
    LocalIsoCrystalEncoder,
)

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ============================================================
# Small helpers
# ============================================================

def _flat_to_image(x: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
    """
    (B, H*W, C) or (H*W, C) -> (B, C, H, W) or (C, H, W)
    """
    H, W = img_shape
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)

    B, N, C = x.shape
    if N != H * W:
        raise ValueError(f"Expected N={H*W}, got {N}")

    y = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    if not batched:
        y = y.squeeze(0)
    return y


def _image_to_flat(x: torch.Tensor) -> torch.Tensor:
    """
    (B, C, H, W) or (C, H, W) -> (B, H*W, C) or (H*W, C)
    """
    batched = x.dim() == 4
    if not batched:
        x = x.unsqueeze(0)

    B, C, H, W = x.shape
    y = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
    if not batched:
        y = y.squeeze(0)
    return y


def _normalize_quaternions(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternion vectors with a numerically-stable denominator."""
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

def _safe_mean_std(x: torch.Tensor, dim=(-2, -1), eps: float = 1e-8):
    """Compute robust mean/std over spatial dimensions for learned stats coding."""
    mean = x.mean(dim=dim)
    var = ((x - mean.unsqueeze(-1).unsqueeze(-1)) ** 2).mean(dim=dim)
    std = torch.sqrt(var + eps)
    return mean, std


def _broadcast_code(code: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Tile per-sample latent code vectors over spatial dimensions."""
    return code[:, :, None, None].expand(-1, -1, H, W).contiguous()


def _as_scale_tuple(scale: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize scalar or anisotropic scale config into (scale_y, scale_x)."""
    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError(f"Expected 2D scale, got {scale!r}")
        return int(scale[0]), int(scale[1])
    scale_i = int(scale)
    return scale_i, scale_i


def _build_hr_base_coords(
    H_lr: int,
    W_lr: int,
    scale: int | tuple[int, int] | list[int],
    device: torch.device,
    dtype: torch.dtype,
):
    """Create HR pixel centers expressed in LR pixel coordinates."""
    scale_y, scale_x = _as_scale_tuple(scale)
    H_hr, W_hr = H_lr * scale_y, W_lr * scale_x
    iy = torch.arange(H_hr, device=device, dtype=dtype)
    ix = torch.arange(W_hr, device=device, dtype=dtype)
    y_base = (iy + 0.5) / scale_y - 0.5
    x_base = (ix + 0.5) / scale_x - 0.5
    y_grid, x_grid = torch.meshgrid(y_base, x_base, indexing="ij")
    return y_grid, x_grid


def _lr_coords_to_grid(
    y_lr: torch.Tensor,
    x_lr: torch.Tensor,
    H_lr: int,
    W_lr: int,
):
    """Convert LR pixel coordinates to [-1,1] grid_sample coordinates."""
    if H_lr > 1:
        y_norm = 2.0 * (y_lr / (H_lr - 1.0)) - 1.0
    else:
        y_norm = torch.zeros_like(y_lr)

    if W_lr > 1:
        x_norm = 2.0 * (x_lr / (W_lr - 1.0)) - 1.0
    else:
        x_norm = torch.zeros_like(x_lr)

    return torch.stack([x_norm, y_norm], dim=-1)


def _sample_feat_lr(
    feat_lr_img: torch.Tensor,  # (B,C,H_lr,W_lr)
    y_lr: torch.Tensor,         # (B,H_hr,W_hr)
    x_lr: torch.Tensor,         # (B,H_hr,W_hr)
):
    """Bilinearly sample LR feature field at arbitrary LR-coordinate queries."""
    B, C, H_lr, W_lr = feat_lr_img.shape
    grid = _lr_coords_to_grid(
        y_lr.clamp(0.0, H_lr - 1.0),
        x_lr.clamp(0.0, W_lr - 1.0),
        H_lr,
        W_lr,
    )
    return F.grid_sample(
        feat_lr_img,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _shift_with_valid_mask(
    x: torch.Tensor,  # (B,H,W,C)
    dy: int,
    dx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift by roll and output mask to zero wrapped pixels."""
    B, H, W, C = x.shape
    y = torch.roll(x, shifts=(dy, dx), dims=(1, 2))

    valid = torch.ones((H, W), device=x.device, dtype=x.dtype)
    if dy > 0:
        valid[:dy, :] = 0.0
    elif dy < 0:
        valid[dy:, :] = 0.0
    if dx > 0:
        valid[:, :dx] = 0.0
    elif dx < 0:
        valid[:, dx:] = 0.0

    return y, valid.view(1, H, W)


# ============================================================
# 1) Pairwise irrep boundary evidence
#    - one scalar map per irrep block L and offset delta
#    - sums over all m and all copies inside each block
# ============================================================

def build_irrep_boundary_evidence(
    feat_lr: torch.Tensor,
    lr_shape: tuple[int, int],
    irreps_feat: Irreps | str,
    offsets: list[tuple[int, int]] | None = None,
    radius: int = 1,
    eps: float = 1e-8,
) -> dict[str, object]:
    """
    Build scalar evidence maps:
      E_{L,δ}(i) = mean_{copy,m} |F_L(i) - F_L(i+δ)|^2
      C_{L,δ}(i) = 1 - cos(F_L(i), F_L(i+δ))

    where F_L(i) is the full irrep block for that L, flattened over all copies
    and all m values.

    Returns:
      {
        "maps": dict[name -> tensor(B,H,W)],
        "tensor": tensor(B,S,H,W),
        "channel_names": list[str],
        "offsets": offsets,
      }
    """
    irreps_feat = Irreps(irreps_feat)
    H, W = lr_shape

    batched = feat_lr.dim() == 3
    if not batched:
        feat_lr = feat_lr.unsqueeze(0)

    B, N, C = feat_lr.shape
    if N != H * W:
        raise ValueError(f"Expected N={H * W}, got {N}")
    if C != irreps_feat.dim:
        raise ValueError(f"Expected feature dim {irreps_feat.dim}, got {C}")

    if offsets is None:
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                offsets.append((dy, dx))

    feat_img = feat_lr.view(B, H, W, C)

    block_labels = []
    seen = {}
    for mul_ir in irreps_feat:
        ir = mul_ir.ir
        label = f"{ir.l}{'e' if ir.p == 1 else 'o'}"
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}_blk{seen[label]}"
        block_labels.append(label)

    maps: dict[str, torch.Tensor] = {}
    stacked: list[torch.Tensor] = []
    channel_names: list[str] = []

    for dy, dx in offsets:
        neigh, valid = _shift_with_valid_mask(feat_img, dy=dy, dx=dx)

        for blk_idx, (sl, mul_ir) in enumerate(zip(irreps_feat.slices(), irreps_feat)):
            mul = int(mul_ir.mul)
            d = int(mul_ir.ir.dim)
            label = block_labels[blk_idx]

            fc = feat_img[..., sl].reshape(B, H, W, mul * d)
            fn = neigh[..., sl].reshape(B, H, W, mul * d)

            E = ((fc - fn) ** 2).mean(dim=-1) * valid
            dot = (fc * fn).sum(dim=-1)
            nfc = (fc * fc).sum(dim=-1).clamp_min(eps)
            nfn = (fn * fn).sum(dim=-1).clamp_min(eps)
            Cmis = (1.0 - dot / torch.sqrt(nfc * nfn)) * valid

            name_E = f"E_{label}_dy{dy:+d}_dx{dx:+d}"
            name_C = f"C_{label}_dy{dy:+d}_dx{dx:+d}"

            maps[name_E] = E if batched else E.squeeze(0)
            maps[name_C] = Cmis if batched else Cmis.squeeze(0)

            stacked.append(E.unsqueeze(1))
            stacked.append(Cmis.unsqueeze(1))
            channel_names.extend([name_E, name_C])

    tensor = torch.cat(stacked, dim=1)
    if not batched:
        tensor = tensor.squeeze(0)

    return {
        "maps": maps,
        "tensor": tensor,
        "channel_names": channel_names,
        "offsets": offsets,
    }


# ============================================================
# 2) Scalar boundary/SDF head
#    - consumes only scalar evidence tensor
# ============================================================

class IrrepEvidenceBoundarySDFHead(nn.Module):
    """Scalar head that predicts boundary maps, SDF-like fields, and shift controls."""

    def __init__(
        self,
        in_channels: int,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        hidden_dim: int = 64,
        guidance_dim: int = 16,
        stats_code_dim: int = 16,
        extra_stats_dim: int = 0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.hidden_dim = int(hidden_dim)
        self.guidance_dim = int(guidance_dim)
        self.stats_code_dim = int(stats_code_dim)
        self.extra_stats_dim = int(extra_stats_dim)

        self.lr_stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.lr_guidance = nn.Conv2d(self.hidden_dim, self.guidance_dim, kernel_size=1)
        self.lr_boundary = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

        self.hr_refine = nn.Sequential(
            nn.Conv2d(self.guidance_dim + 1, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.hr_guidance = nn.Conv2d(self.hidden_dim, self.guidance_dim, kernel_size=1)
        self.hr_boundary = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

        stats_in_dim = 2 * self.in_channels + 2 * self.hidden_dim + self.extra_stats_dim
        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 5 + self.stats_code_dim),
        )

        nn.init.zeros_(self.lr_boundary.weight)
        nn.init.zeros_(self.lr_boundary.bias)
        nn.init.zeros_(self.hr_boundary.weight)
        nn.init.zeros_(self.hr_boundary.bias)

    def _gaussian_blur_per_batch(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Apply per-sample Gaussian smoothing with sigma predicted from evidence stats."""
        B, _, _, _ = x.shape
        out = []
        for b in range(B):
            s = float(sigma[b, 0].item())
            k = max(3, int(math.ceil(6.0 * s)) | 1)
            half = k // 2
            coords = torch.arange(k, device=x.device, dtype=x.dtype) - half
            g = torch.exp(-(coords**2) / (2.0 * s * s + 1e-8))
            g = g / g.sum()
            g2 = torch.outer(g, g)[None, None]
            out.append(F.conv2d(x[b : b + 1], g2, padding=half))
        return torch.cat(out, dim=0)

    def _spatial_grad(self, x: torch.Tensor):
        """Finite-difference spatial gradients for boundary normal estimation."""
        dy = torch.zeros_like(x)
        dx = torch.zeros_like(x)

        dy[:, :, 1:-1, :] = 0.5 * (x[:, :, 2:, :] - x[:, :, :-2, :])
        dy[:, :, 0, :] = x[:, :, 1, :] - x[:, :, 0, :]
        dy[:, :, -1, :] = x[:, :, -1, :] - x[:, :, -2, :]

        dx[:, :, :, 1:-1] = 0.5 * (x[:, :, :, 2:] - x[:, :, :, :-2])
        dx[:, :, :, 0] = x[:, :, :, 1] - x[:, :, :, 0]
        dx[:, :, :, -1] = x[:, :, :, -1] - x[:, :, :, -2]
        return dy, dx

    def forward(
        self,
        evidence_tensor_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        extra_stats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Produce guidance, boundary logits, SDF field, normals, and shift controls."""
        H, W = lr_shape
        scale_y, scale_x = self.upsample_factor
        Hr, Wr = H * scale_y, W * scale_x

        if evidence_tensor_lr.dim() == 3:
            evidence_tensor_lr = evidence_tensor_lr.unsqueeze(0)

        stem = self.lr_stem(evidence_tensor_lr)
        guidance_lr = self.lr_guidance(stem)
        boundary_logits_lr = self.lr_boundary(stem)

        ev_mean, ev_std = _safe_mean_std(evidence_tensor_lr)
        st_mean, st_std = _safe_mean_std(stem)
        pooled = torch.cat([ev_mean, ev_std, st_mean, st_std], dim=1)
        if extra_stats is not None:
            pooled = torch.cat([pooled, extra_stats], dim=1)

        # Global learned controls inferred from evidence statistics.
        raw = self.stats_mlp(pooled)
        shift_px = 0.35 + 1.15 * torch.sigmoid(raw[:, 0:1])       # LR pixel units
        sigma_hr = 0.8 + 2.2 * torch.sigmoid(raw[:, 1:2])         # boundary-field smoothing
        band_center = 0.10 + 0.45 * torch.sigmoid(raw[:, 2:3])    # boundary-band threshold
        band_sharpness = 8.0 + 24.0 * torch.sigmoid(raw[:, 3:4])  # boundary-band steepness
        side_temp = 0.20 + 0.80 * torch.sigmoid(raw[:, 4:5])      # side-selector temperature
        stats_code = raw[:, 5:]

        guidance_up = F.interpolate(guidance_lr, size=(Hr, Wr), mode="bilinear", align_corners=False)
        boundary_up = F.interpolate(boundary_logits_lr, size=(Hr, Wr), mode="bilinear", align_corners=False)

        hr_hidden = self.hr_refine(torch.cat([guidance_up, torch.sigmoid(boundary_up)], dim=1))
        guidance_hr = guidance_up + self.hr_guidance(hr_hidden)
        boundary_logits_hr = boundary_up + self.hr_boundary(hr_hidden)

        boundary_prob_hr = torch.sigmoid(boundary_logits_hr)
        sdf_hr = self._gaussian_blur_per_batch(boundary_prob_hr, sigma_hr)
        sdf_hr = sdf_hr / (sdf_hr.amax(dim=(-2, -1), keepdim=True) + 1e-8)

        dy, dx = self._spatial_grad(sdf_hr)
        norm = torch.sqrt(dx * dx + dy * dy + 1e-8)
        nx_hr = dx / norm
        ny_hr = dy / norm

        return {
            "guidance_lr": guidance_lr,
            "boundary_logits_lr": boundary_logits_lr,
            "guidance_hr": guidance_hr,
            "boundary_logits_hr": boundary_logits_hr,
            "boundary_prob_hr": boundary_prob_hr,
            "sdf_hr": sdf_hr,
            "nx_hr": nx_hr,
            "ny_hr": ny_hr,
            "shift_px": shift_px,
            "band_center": band_center,
            "band_sharpness": band_sharpness,
            "side_temp": side_temp,
            "stats_code": stats_code,
            "hr_shape": (Hr, Wr),
        }


# ============================================================
# 3) Irrep-respecting shifted upsampler
#    - scalar logic decides where to sample
#    - irrep field itself only gets bilinear sampled + scalar weighted
# ============================================================

class IrrepShiftedSDFUpsample(nn.Module):
    """Use scalar SDF/normal fields to decide where to sample irreps features."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        guidance_dim: int = 16,
        stats_code_dim: int = 16,
        hidden_dim: int = 64,
        hard_one_sided: bool = True,
        hard_boundary_band: bool = False,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feature_dim = int(self.irreps_feat.dim)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.guidance_dim = int(guidance_dim)
        self.stats_code_dim = int(stats_code_dim)
        self.hidden_dim = int(hidden_dim)
        self.hard_one_sided = bool(hard_one_sided)
        self.hard_boundary_band = bool(hard_boundary_band)

        # Side selector is scalar-only
        cls_in = self.guidance_dim + 1 + 1 + self.stats_code_dim
        self.side_head = nn.Sequential(
            nn.Conv2d(cls_in, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 2, kernel_size=1),
        )

    @staticmethod
    def _straight_through_argmax_probs(logits: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        soft = torch.softmax(logits / temp.clamp_min(1e-4), dim=1)
        idx = soft.argmax(dim=1, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(1, idx, 1.0)
        return hard + soft - soft.detach()

    @staticmethod
    def _straight_through_threshold(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        hard = (x >= threshold).to(x.dtype)
        return hard + x - x.detach()

    def forward(
        self,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        sdf_out: dict[str, torch.Tensor],
        return_aux: bool = False,
    ):
        """Shifted LR sampling and side-selection to produce HR irreps features."""
        H_lr, W_lr = lr_shape
        Hr, Wr = sdf_out["hr_shape"]

        batched = feat_lr.dim() == 3
        if not batched:
            feat_lr = feat_lr.unsqueeze(0)

        B = feat_lr.shape[0]
        feat_lr_img = _flat_to_image(feat_lr, lr_shape)  # (B,C,H_lr,W_lr)

        guidance_hr = sdf_out["guidance_hr"]
        boundary_prob_hr = sdf_out["boundary_prob_hr"]
        sdf_hr = sdf_out["sdf_hr"]
        nx_hr = sdf_out["nx_hr"]
        ny_hr = sdf_out["ny_hr"]
        shift_px = sdf_out["shift_px"][:, :, None, None]
        band_center = sdf_out["band_center"][:, :, None, None]
        band_sharpness = sdf_out["band_sharpness"][:, :, None, None]
        side_temp = sdf_out["side_temp"][:, :, None, None]
        stats_code = _broadcast_code(sdf_out["stats_code"], Hr, Wr)

        y_base, x_base = _build_hr_base_coords(
            H_lr,
            W_lr,
            self.upsample_factor,
            feat_lr.device,
            feat_lr.dtype,
        )
        y_base = y_base[None].expand(B, -1, -1)
        x_base = x_base[None].expand(B, -1, -1)

        y_plus = y_base + shift_px[:, 0] * ny_hr[:, 0]
        x_plus = x_base + shift_px[:, 0] * nx_hr[:, 0]
        y_minus = y_base - shift_px[:, 0] * ny_hr[:, 0]
        x_minus = x_base - shift_px[:, 0] * nx_hr[:, 0]

        # Three candidate samples per HR location: center, +normal, -normal.
        feat_center = _sample_feat_lr(feat_lr_img, y_base, x_base)
        feat_plus = _sample_feat_lr(feat_lr_img, y_plus, x_plus)
        feat_minus = _sample_feat_lr(feat_lr_img, y_minus, x_minus)

        side_in = torch.cat([guidance_hr, boundary_prob_hr, sdf_hr, stats_code], dim=1)
        side_logits = self.side_head(side_in)
        if self.hard_one_sided:
            side_probs = self._straight_through_argmax_probs(side_logits, side_temp)
        else:
            side_probs = torch.softmax(side_logits / side_temp.clamp_min(1e-4), dim=1)

        feat_side = side_probs[:, 0:1] * feat_plus + side_probs[:, 1:2] * feat_minus

        # Interior uses center sample; boundary band blends toward one-sided sample.
        band_soft = torch.sigmoid(band_sharpness * (sdf_hr - band_center))
        band = self._straight_through_threshold(band_soft) if self.hard_boundary_band else band_soft
        feat_hr = (1.0 - band) * feat_center + band * feat_side

        feat_hr_flat = _image_to_flat(feat_hr)

        aux = {
            "feat_center_hr": feat_center,
            "feat_plus_hr": feat_plus,
            "feat_minus_hr": feat_minus,
            "side_logits_hr": side_logits,
            "side_probs_hr": side_probs,
            "boundary_band_hr": band,
        }

        if not batched:
            feat_hr_flat = feat_hr_flat.squeeze(0)
            aux = {
                k: (v.squeeze(0) if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v)
                for k, v in aux.items()
            }

        if return_aux:
            return feat_hr_flat, aux
        return feat_hr_flat


# ============================================================
# 4) Optional teacher target from quaternions
#    mirrors your uploaded graph-misorientation boundary code
# ============================================================

def _left_mult_matrix_wxyz_batch(q_syms: torch.Tensor) -> torch.Tensor:
    """Convert symmetry quaternions to left-multiplication matrices."""
    w, x, y, z = q_syms.unbind(dim=-1)
    r0 = torch.stack([w, -x, -y, -z], dim=-1)
    r1 = torch.stack([x,  w, -z,  y], dim=-1)
    r2 = torch.stack([y,  z,  w, -x], dim=-1)
    r3 = torch.stack([z, -y,  x,  w], dim=-1)
    return torch.stack([r0, r1, r2, r3], dim=1)


def _misorientation_angle_sym(q1: torch.Tensor, q2: torch.Tensor, sym_ops: torch.Tensor) -> torch.Tensor:
    q2var = torch.einsum("gij,...j->...gi", sym_ops, q2)
    dots = (q1.unsqueeze(-2) * q2var).sum(dim=-1).abs().clamp(0.0, 1.0)
    best = dots.max(dim=-1).values
    return 2.0 * torch.acos(best)


def _neighbor_misorientation(q_bchw: torch.Tensor, sym_ops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, H, W = q_bchw.shape
    q_hw = q_bchw.squeeze(0).permute(1, 2, 0)
    qL = q_hw[:, :-1]
    qR = q_hw[:, 1:]
    qU = q_hw[:-1, :]
    qD = q_hw[1:, :]
    mis_x = _misorientation_angle_sym(qL, qR, sym_ops)
    mis_y = _misorientation_angle_sym(qU, qD, sym_ops)
    return mis_x, mis_y


def graph_boundary_teacher_from_quats(
    q_bchw: torch.Tensor,
    sym_ops_quat: torch.Tensor,
    thr_deg: float = 3.0,
    connectivity: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q_bchw: (1,4,H,W)
    sym_ops_quat: (G,4) symmetry quaternions from encoder.sym_ops
    Returns:
      labels (H,W) long
      gb_mask (H,W) float in {0,1}
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for graph_boundary_teacher_from_quats")

    q_bchw = q_bchw.detach().cpu()
    sym_ops = _left_mult_matrix_wxyz_batch(sym_ops_quat.detach().cpu())

    mis_x, mis_y = _neighbor_misorientation(q_bchw, sym_ops)
    thr = np.deg2rad(float(thr_deg))

    mis_x_np = mis_x.numpy()
    mis_y_np = mis_y.numpy()

    _, _, H, W = q_bchw.shape
    N = H * W

    rows = []
    cols = []

    def idx(i, j):
        return i * W + j

    mask_x = mis_x_np <= thr
    ii, jj = np.where(mask_x)
    p1 = idx(ii, jj)
    p2 = idx(ii, jj + 1)
    rows += list(p1) + list(p2)
    cols += list(p2) + list(p1)

    mask_y = mis_y_np <= thr
    ii, jj = np.where(mask_y)
    p1 = idx(ii, jj)
    p2 = idx(ii + 1, jj)
    rows += list(p1) + list(p2)
    cols += list(p2) + list(p1)

    data = np.ones(len(rows), np.uint8)
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    _, labels = connected_components(A, directed=False)
    labels = labels.reshape(H, W).astype(np.int64)

    gb = np.zeros((H, W), dtype=bool)
    gb[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    gb[1:, :] |= labels[1:, :] != labels[:-1, :]
    if connectivity == 8:
        gb[1:, 1:] |= labels[1:, 1:] != labels[:-1, :-1]
        gb[1:, :-1] |= labels[1:, :-1] != labels[:-1, 1:]

    labels_t = torch.from_numpy(labels)
    gb_t = torch.from_numpy(gb.astype(np.float32))
    return labels_t, gb_t


# ============================================================
# 5) Full model
# ============================================================

class IsoEmbeddingSRSDF(nn.Module):
    """
    Full boundary-aware SR model:

      LR quats
        -> encoder.forward_a1
        -> 1-2 equivariant LR blocks
        -> pairwise irrep evidence tensor
        -> scalar boundary/SDF head
        -> shifted irrep upsampler
        -> equivariant HR cleanup
        -> boundary-aware refinement
        -> optimizing decoder back to quats

    Boundary supervision target can be generated with graph_boundary_teacher_from_quats().
    """

    def __init__(
        self,
        crystal: str = "fcc",
        d6_convention: str = "z_axis",
        device: str | torch.device | None = None,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        evidence_offsets: list[tuple[int, int]] | None = None,
        evidence_radius: int = 1,
        sdf_hidden_dim: int = 64,
        guidance_dim: int = 16,
        stats_code_dim: int = 16,
        extra_stats_dim: int = 0,
        num_lr_blocks: int = 1,
        num_hr_blocks: int = 1,
        use_pre_lr: Optional[bool] = None,
        use_post_hr: Optional[bool] = None,
        use_refinement: bool = True,
        refinement_num_steps: int = 2,
        refinement_hidden_dim: int = 32,
        refinement_kernel_size: int = 3,
        decoder_cubochoric_resolution: int = 1,
        decoder_num_starts: int = 6,
        decoder_steps: int = 25,
        decoder_lr: float = 0.05,
        decoder_method: str = "cubochoric",
        decoder_max_table_rows: int | None = None,
        decoder_table_cache_dir: str | Path | None = "out/decoder_lookup_tables",
        hard_one_sided: bool = True,
        hard_boundary_band: bool = False,
    ):
        super().__init__()

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.encoder = LocalIsoCrystalEncoder(
            crystal=crystal,
            d6_convention=d6_convention,
            dtype=torch.float32,
            device=self.device,
        )
        self.irreps_a1 = self.encoder.irreps_a1
        self.feature_dim_a1 = int(self.encoder.out_dim_a1)

        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.evidence_radius = int(evidence_radius)

        if evidence_offsets is None:
            evidence_offsets = []
            for dy in range(-self.evidence_radius, self.evidence_radius + 1):
                for dx in range(-self.evidence_radius, self.evidence_radius + 1):
                    if dy == 0 and dx == 0:
                        continue
                    evidence_offsets.append((dy, dx))
        self.evidence_offsets = evidence_offsets

        n_irrep_blocks = len(list(self.irreps_a1))
        evidence_channels = len(self.evidence_offsets) * n_irrep_blocks * 2

        n_lr_blocks = max(0, int(num_lr_blocks))
        if use_pre_lr is not None:
            n_lr_blocks = 1 if bool(use_pre_lr) else 0
        self.lr_blocks = nn.ModuleList(
            [
                EquivariantSpatialConv(
                    kernel_size=3,
                    irreps_in=self.irreps_a1,
                    irreps_out=self.irreps_a1,
                    use_residual=True,
                )
                for _ in range(n_lr_blocks)
            ]
        )

        self.boundary_sdf_head = IrrepEvidenceBoundarySDFHead(
            in_channels=evidence_channels,
            upsample_factor=self.upsample_factor,
            hidden_dim=sdf_hidden_dim,
            guidance_dim=guidance_dim,
            stats_code_dim=stats_code_dim,
            extra_stats_dim=extra_stats_dim,
        )

        self.sdf_upsample = IrrepShiftedSDFUpsample(
            irreps_feat=self.irreps_a1,
            upsample_factor=self.upsample_factor,
            guidance_dim=guidance_dim,
            stats_code_dim=stats_code_dim,
            hidden_dim=sdf_hidden_dim,
            hard_one_sided=hard_one_sided,
            hard_boundary_band=hard_boundary_band,
        )

        n_hr_blocks = max(0, int(num_hr_blocks))
        if use_post_hr is not None:
            n_hr_blocks = 1 if bool(use_post_hr) else 0
        self.hr_blocks = nn.ModuleList(
            [
                EquivariantSpatialConv(
                    kernel_size=3,
                    irreps_in=self.irreps_a1,
                    irreps_out=self.irreps_a1,
                    use_residual=True,
                )
                for _ in range(n_hr_blocks)
            ]
        )

        self.use_refinement = bool(use_refinement)
        self.refinement_head = (
            BoundaryRefinementHead(
                irreps_feat=self.irreps_a1,
                guidance_dim=guidance_dim,
                kernel_size=int(refinement_kernel_size),
                num_steps=int(refinement_num_steps),
                hidden_dim=int(refinement_hidden_dim),
                refine_boundary_logits=True,
            )
            if self.use_refinement
            else None
        )

        self.decoder = CubochoricOptimizingLocalIsoDecoder(
            encoder=self.encoder,
            cubochoric_resolution=int(decoder_cubochoric_resolution),
            method=str(decoder_method),
            num_starts=int(decoder_num_starts),
            steps=int(decoder_steps),
            lr=float(decoder_lr),
            target_irreps="a1",
            max_table_rows=decoder_max_table_rows,
            table_cache_dir=decoder_table_cache_dir,
        )

    def encode_a1(self, quats: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_a1(quats)

    def decode(self, features_a1: torch.Tensor) -> torch.Tensor:
        """Decode A1 feature vectors back to normalized quaternions."""
        batched = features_a1.dim() == 3
        if batched:
            B, N, C = features_a1.shape
            q = self.decoder(features_a1.reshape(B * N, C))
            q = _normalize_quaternions(q).reshape(B, N, 4)
            return q
        q = self.decoder(features_a1)
        return _normalize_quaternions(q)

    def _forward_sr_features(
        self,
        feat_lr_a1: torch.Tensor,
        lr_shape: tuple[int, int],
        extra_material_stats: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Core SR forward pass in A1 space; decode is intentionally separate."""
        feat = feat_lr_a1.to(self.device)
        for blk in self.lr_blocks:
            feat = blk(feat, lr_shape)

        evidence = build_irrep_boundary_evidence(
            feat_lr=feat,
            lr_shape=lr_shape,
            irreps_feat=self.irreps_a1,
            offsets=self.evidence_offsets,
            radius=self.evidence_radius,
        )
        evidence_tensor = evidence["tensor"]

        # Scalar branch infers boundary/SDF geometry that drives the sampler.
        sdf_out = self.boundary_sdf_head(
            evidence_tensor_lr=evidence_tensor,
            lr_shape=lr_shape,
            extra_stats=extra_material_stats,
        )

        feat_hr, up_aux = self.sdf_upsample(
            feat_lr=feat,
            lr_shape=lr_shape,
            sdf_out=sdf_out,
            return_aux=True,
        )

        for blk in self.hr_blocks:
            feat_hr = blk(
                feat_hr,
                sdf_out["hr_shape"],
                guidance=sdf_out["guidance_hr"],
                boundary_logits=sdf_out["boundary_logits_hr"],
            )

        boundary_logits_hr_refined = None
        if self.refinement_head is not None:
            feat_hr, boundary_logits_hr_refined = self.refinement_head(
                feat_hr,
                sdf_out["hr_shape"],
                guidance=sdf_out["guidance_hr"],
                boundary_logits=sdf_out["boundary_logits_hr"],
            )

        if return_aux:
            aux = {}
            aux.update(evidence)
            aux.update(sdf_out)
            aux.update(up_aux)
            aux["feat_hr_a1"] = feat_hr
            aux["boundary_logits_hr_refined"] = boundary_logits_hr_refined
            return feat_hr, sdf_out["hr_shape"], aux

        return feat_hr, sdf_out["hr_shape"]

    def forward_sr(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = True,
        extra_material_stats: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """End-to-end SR wrapper from LR quaternions (returns quats and optional aux)."""
        lr_quats = lr_quats.to(self.device)
        if normalize_input:
            lr_quats = _normalize_quaternions(lr_quats)

        feat_lr_a1 = self.encode_a1(lr_quats)

        if return_aux:
            feat_hr_a1, _, aux = self._forward_sr_features(
                feat_lr_a1=feat_lr_a1,
                lr_shape=lr_shape,
                extra_material_stats=extra_material_stats,
                return_aux=True,
            )
            q_hr = self.decode(feat_hr_a1)
            return q_hr, aux

        feat_hr_a1, _ = self._forward_sr_features(
            feat_lr_a1=feat_lr_a1,
            lr_shape=lr_shape,
            extra_material_stats=extra_material_stats,
            return_aux=False,
        )
        return self.decode(feat_hr_a1)

    def feature_loss_sr(
        self,
        lr_quats: torch.Tensor,
        hr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = False,
        extra_material_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lr_quats = lr_quats.to(self.device)
        hr_quats = hr_quats.to(self.device)

        batched = lr_quats.dim() == 3
        if batched:
            B = lr_quats.shape[0]
            lr_flat = lr_quats.reshape(-1, 4)
            hr_flat = hr_quats.reshape(-1, 4)
        else:
            B = 1
            lr_flat = lr_quats
            hr_flat = hr_quats

        if normalize_input:
            lr_flat = _normalize_quaternions(lr_flat)
            hr_flat = _normalize_quaternions(hr_flat)

        with torch.no_grad():
            feat_lr_a1_flat = self.encode_a1(lr_flat).detach()
            feat_hr_tgt_flat = self.encode_a1(hr_flat).detach()

        if batched:
            feat_lr_a1 = feat_lr_a1_flat.reshape(B, -1, feat_lr_a1_flat.shape[-1])
            feat_hr_tgt = feat_hr_tgt_flat.reshape(B, -1, feat_hr_tgt_flat.shape[-1])
        else:
            feat_lr_a1 = feat_lr_a1_flat
            feat_hr_tgt = feat_hr_tgt_flat

        feat_hr, _ = self._forward_sr_features(
            feat_lr_a1=feat_lr_a1,
            lr_shape=lr_shape,
            extra_material_stats=extra_material_stats,
            return_aux=False,
        )
        return F.mse_loss(feat_hr, feat_hr_tgt)

    @torch.no_grad()
    def boundary_teacher_from_quats(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int],
        thr_deg: float = 3.0,
        connectivity: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generic teacher target from quaternions at any resolution.

        Args
        ----
        quats:
            (H*W,4) or (1,H*W,4)
        img_shape:
            (H, W)

        Returns
        -------
        labels: (H,W) long
        gb:     (H,W) float in {0,1}
        """
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for boundary_teacher_from_quats")

        H, W = img_shape
        if quats.dim() == 2:
            q = quats.view(H, W, 4).permute(2, 0, 1).unsqueeze(0).contiguous()
        elif quats.dim() == 3 and quats.shape[0] == 1:
            q = quats.view(1, H, W, 4).permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("boundary_teacher_from_quats expects one image: (H*W,4) or (1,H*W,4)")

        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
        labels, gb = graph_boundary_teacher_from_quats(
            q_bchw=q.detach().cpu(),
            sym_ops_quat=self.encoder.sym_ops.detach().cpu(),
            thr_deg=float(thr_deg),
            connectivity=int(connectivity),
        )
        return labels, gb

    @torch.no_grad()
    def batch_boundary_teacher_from_quats(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int],
        thr_deg: float = 3.0,
        connectivity: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched boundary-teacher wrapper.

        Args
        ----
        quats:
            (B,H*W,4) or (H*W,4)

        Returns
        -------
        labels: (B,H,W) long
        gb:     (B,1,H,W) float
        """
        if quats.dim() == 2:
            labels, gb = self.boundary_teacher_from_quats(
                quats=quats,
                img_shape=img_shape,
                thr_deg=thr_deg,
                connectivity=connectivity,
            )
            return labels.unsqueeze(0), gb.unsqueeze(0).unsqueeze(0)

        if quats.dim() != 3:
            raise ValueError("Expected quats shape (B,H*W,4) or (H*W,4)")

        labels_list: list[torch.Tensor] = []
        gb_list: list[torch.Tensor] = []
        for b in range(int(quats.shape[0])):
            labels_b, gb_b = self.boundary_teacher_from_quats(
                quats=quats[b],
                img_shape=img_shape,
                thr_deg=thr_deg,
                connectivity=connectivity,
            )
            labels_list.append(labels_b)
            gb_list.append(gb_b)

        labels = torch.stack(labels_list, dim=0)
        gb = torch.stack(gb_list, dim=0).unsqueeze(1)
        return labels, gb

    def boundary_supervision_loss(
        self,
        aux: dict[str, torch.Tensor],
        hr_quats: torch.Tensor,
        hr_shape: tuple[int, int],
        lr_quats: torch.Tensor | None = None,
        lr_shape: tuple[int, int] | None = None,
        thr_deg: float = 3.0,
        connectivity: int = 4,
        lambda_hr: float = 1.0,
        lambda_lr: float = 0.25,
        use_refined_hr: bool = True,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Boundary supervision using teacher boundaries from quaternions.

        This is the training bridge you asked for:
        LR -> irreps -> boundary approximation at HR -> compare with HR teacher.

        Main path:
          - HR predicted boundary vs known HR boundary from HR quats

        Optional:
          - LR auxiliary boundary supervision if LR inputs are provided and lambda_lr > 0
        """
        _, gb_hr = self.batch_boundary_teacher_from_quats(
            quats=hr_quats,
            img_shape=hr_shape,
            thr_deg=thr_deg,
            connectivity=connectivity,
        )

        pred_hr_key = (
            "boundary_logits_hr_refined"
            if (use_refined_hr and aux.get("boundary_logits_hr_refined") is not None)
            else "boundary_logits_hr"
        )
        pred_hr = aux[pred_hr_key]
        gb_hr = gb_hr.to(device=pred_hr.device, dtype=pred_hr.dtype)

        if tuple(gb_hr.shape[-2:]) != tuple(pred_hr.shape[-2:]):
            gb_hr = F.interpolate(gb_hr, size=pred_hr.shape[-2:], mode="nearest")

        if use_focal:
            p = torch.sigmoid(pred_hr)
            ce = F.binary_cross_entropy_with_logits(pred_hr, gb_hr, reduction="none")
            pt = p * gb_hr + (1.0 - p) * (1.0 - gb_hr)
            loss_hr = ((1.0 - pt) ** float(focal_gamma) * ce).mean()
        else:
            loss_hr = F.binary_cross_entropy_with_logits(pred_hr, gb_hr)

        total = float(lambda_hr) * loss_hr
        info: dict[str, torch.Tensor] = {
            "gb_target_hr": gb_hr.detach(),
            "loss_boundary_hr": loss_hr.detach(),
        }

        if (
            lr_quats is not None
            and lr_shape is not None
            and ("boundary_logits_lr" in aux)
            and float(lambda_lr) > 0.0
        ):
            _, gb_lr = self.batch_boundary_teacher_from_quats(
                quats=lr_quats,
                img_shape=lr_shape,
                thr_deg=thr_deg,
                connectivity=connectivity,
            )

            pred_lr = aux["boundary_logits_lr"]
            gb_lr = gb_lr.to(device=pred_lr.device, dtype=pred_lr.dtype)
            if tuple(gb_lr.shape[-2:]) != tuple(pred_lr.shape[-2:]):
                gb_lr = F.interpolate(gb_lr, size=pred_lr.shape[-2:], mode="nearest")

            if use_focal:
                p = torch.sigmoid(pred_lr)
                ce = F.binary_cross_entropy_with_logits(pred_lr, gb_lr, reduction="none")
                pt = p * gb_lr + (1.0 - p) * (1.0 - gb_lr)
                loss_lr = ((1.0 - pt) ** float(focal_gamma) * ce).mean()
            else:
                loss_lr = F.binary_cross_entropy_with_logits(pred_lr, gb_lr)

            total = total + float(lambda_lr) * loss_lr
            info["gb_target_lr"] = gb_lr.detach()
            info["loss_boundary_lr"] = loss_lr.detach()

        info["loss_boundary_total"] = total.detach()
        return total, info

    def boundary_teacher_from_lr_quats(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        thr_deg: float = 3.0,
        connectivity: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible LR-only convenience wrapper."""
        return self.boundary_teacher_from_quats(
            quats=lr_quats,
            img_shape=lr_shape,
            thr_deg=thr_deg,
            connectivity=connectivity,
        )

    def forward(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        quats = quats.to(self.device)
        if quats.dim() != 2 or quats.shape[-1] != 4:
            raise ValueError(f"IsoEmbeddingSRSDF expects (N,4), got {tuple(quats.shape)}")
        if normalize_input:
            quats = _normalize_quaternions(quats)

        if img_shape is not None:
            return self.forward_sr(quats, lr_shape=img_shape, normalize_input=False)

        feat_a1 = self.encode_a1(quats)
        return self.decode(feat_a1)


# Backward-compatible alias expected by runtime helpers.
IsoEmbeddingSRAttn = IsoEmbeddingSRSDF

__all__ = [
    "build_irrep_boundary_evidence",
    "IrrepEvidenceBoundarySDFHead",
    "IrrepShiftedSDFUpsample",
    "graph_boundary_teacher_from_quats",
    "IsoEmbeddingSRSDF",
    "IsoEmbeddingSRAttn",
]

# ============================================================
# 7) Minimal feature-space training helpers
# ============================================================

@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lambda_feat: float = 1.0
    lambda_boundary: float = 0.75
    lambda_lr_boundary: float = 0.20
    lambda_side_entropy: float = 0.01
    boundary_thr_deg: float = 3.0
    boundary_connectivity: int = 4
    use_focal_boundary: bool = True
    focal_gamma: float = 2.0


def side_entropy_loss(side_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    side_probs: (B,2,H,W)
    Encourage decisive side assignment near boundaries (low entropy selector).
    """
    p = side_probs.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=1, keepdim=True)
    return ent.mean()


def train_step_feature_space(
    model: IsoEmbeddingSRSDF,
    batch: dict[str, torch.Tensor | tuple[int, int]],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    """
    Expected batch keys:
      - lr_quats: (B,N_lr,4) or (N_lr,4)
      - hr_quats: (B,N_hr,4) or (N_hr,4)
      - lr_shape: (H_lr, W_lr)
      - hr_shape: (H_hr, W_hr)
      - optional stats: (B,D) or (D,)
    Trains in A1 feature space and uses HR boundary supervision.
    """
    model.train()

    lr_quats = batch["lr_quats"].to(model.device)
    hr_quats = batch["hr_quats"].to(model.device)
    lr_shape = tuple(batch["lr_shape"])
    hr_shape = tuple(batch["hr_shape"])
    extra_stats = batch.get("stats", None)
    if extra_stats is not None:
        extra_stats = extra_stats.to(model.device)

    with torch.no_grad():
        # Train in feature space: target is encoded HR A1 features.
        feat_lr = model.encode_a1(lr_quats)
        feat_hr_tgt = model.encode_a1(hr_quats)

    feat_hr_pred, _, aux = model._forward_sr_features(
        feat_lr_a1=feat_lr,
        lr_shape=lr_shape,
        extra_material_stats=extra_stats,
        return_aux=True,
    )

    loss_feat = F.mse_loss(feat_hr_pred, feat_hr_tgt)

    loss_boundary, _ = model.boundary_supervision_loss(
        aux=aux,
        hr_quats=hr_quats,
        hr_shape=hr_shape,
        lr_quats=lr_quats,
        lr_shape=lr_shape,
        thr_deg=cfg.boundary_thr_deg,
        connectivity=cfg.boundary_connectivity,
        lambda_hr=1.0,
        lambda_lr=cfg.lambda_lr_boundary,
        use_refined_hr=True,
        use_focal=cfg.use_focal_boundary,
        focal_gamma=cfg.focal_gamma,
    )

    loss_side = side_entropy_loss(aux["side_probs_hr"])
    loss = (
        cfg.lambda_feat * loss_feat
        + cfg.lambda_boundary * loss_boundary
        + cfg.lambda_side_entropy * loss_side
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    return {
        "loss_total": float(loss.item()),
        "loss_feat": float(loss_feat.item()),
        "loss_boundary": float(loss_boundary.item()),
        "loss_side_entropy": float(loss_side.item()),
    }


@torch.no_grad()
def validate_batch_feature_space(
    model: IsoEmbeddingSRSDF,
    batch: dict[str, torch.Tensor | tuple[int, int]],
    cfg: TrainConfig,
) -> dict[str, float]:
    model.eval()

    lr_quats = batch["lr_quats"].to(model.device)
    hr_quats = batch["hr_quats"].to(model.device)
    lr_shape = tuple(batch["lr_shape"])
    hr_shape = tuple(batch["hr_shape"])
    extra_stats = batch.get("stats", None)
    if extra_stats is not None:
        extra_stats = extra_stats.to(model.device)

    feat_loss = model.feature_loss_sr(
        lr_quats=lr_quats,
        hr_quats=hr_quats,
        lr_shape=lr_shape,
        normalize_input=False,
        extra_material_stats=extra_stats,
    )

    # Validation avoids decode: boundary loss depends only on SR aux tensors.
    # This keeps eval faster and deterministic (no decoder local optimization).
    feat_lr = model.encode_a1(lr_quats)
    _, _, aux = model._forward_sr_features(
        feat_lr_a1=feat_lr,
        lr_shape=lr_shape,
        extra_material_stats=extra_stats,
        return_aux=True,
    )

    boundary_loss, _ = model.boundary_supervision_loss(
        aux=aux,
        hr_quats=hr_quats,
        hr_shape=hr_shape,
        lr_quats=lr_quats,
        lr_shape=lr_shape,
        thr_deg=cfg.boundary_thr_deg,
        connectivity=cfg.boundary_connectivity,
        lambda_hr=1.0,
        lambda_lr=cfg.lambda_lr_boundary,
        use_refined_hr=True,
        use_focal=cfg.use_focal_boundary,
        focal_gamma=cfg.focal_gamma,
    )

    return {
        "loss_feat": float(feat_loss.item()),
        "loss_boundary": float(boundary_loss.item()),
    }


__all__ = [
    "IrrepEvidenceBoundarySDFHead",
    "IrrepShiftedSDFUpsample",
    "IsoEmbeddingSRSDF",
    "graph_boundary_teacher_from_quats",
    "TrainConfig",
    "side_entropy_loss",
    "train_step_feature_space",
    "validate_batch_feature_space",
]
