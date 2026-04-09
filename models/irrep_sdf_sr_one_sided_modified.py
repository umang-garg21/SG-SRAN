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
from models.SR_double_conv_SRattn_a1_Boundary_aware import BoundaryAwareAttentionUpsampler
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


def _randomize_parameter_(param: torch.Tensor, std: float = 0.02) -> None:
    """Small random init used for the lightweight grain-attention helper."""
    if param is None:
        return
    with torch.no_grad():
        nn.init.normal_(param, mean=0.0, std=float(std))


def _as_scale_tuple(scale: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize scalar or anisotropic scale config into (scale_y, scale_x)."""
    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError(f"Expected 2D scale, got {scale!r}")
        return int(scale[0]), int(scale[1])
    scale_i = int(scale)
    return scale_i, scale_i


def _as_kernel_tuple(kernel: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize a scalar or pair into an odd 2D kernel size."""
    ky, kx = _as_scale_tuple(kernel)
    if ky <= 0 or kx <= 0:
        raise ValueError(f"Kernel sizes must be positive, got {(ky, kx)!r}")
    if (ky % 2) == 0 or (kx % 2) == 0:
        raise ValueError(f"Kernel sizes must be odd to preserve shape, got {(ky, kx)!r}")
    return ky, kx


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, eps: float = 1.0) -> torch.Tensor:
    """Average a tensor with an optional broadcastable spatial mask."""
    if mask is None:
        return x.mean()
    w = mask.to(dtype=x.dtype)
    return (x * w).sum() / w.sum().clamp_min(eps)


def _dilate_binary_mask(mask: torch.Tensor, kernel_size: int | tuple[int, int] | list[int]) -> torch.Tensor:
    """Dilate a binary mask with max pooling while preserving spatial size."""
    ky, kx = _as_kernel_tuple(kernel_size)
    pooled = F.max_pool2d(mask.float(), kernel_size=(ky, kx), stride=1, padding=(ky // 2, kx // 2))
    return (pooled > 0.0).to(dtype=mask.dtype)


def _append_probe_stage(
    probe_stages: list[dict[str, object]] | None,
    name: str,
    feat: torch.Tensor,
    img_shape: tuple[int, int],
) -> None:
    """Record an A1 feature tensor for later decode/visualization."""
    if probe_stages is None:
        return

    feat_store = feat
    if feat.dim() == 4:
        feat_store = _image_to_flat(feat)
    elif feat.dim() == 3 and tuple(feat.shape[-2:]) == tuple(img_shape):
        feat_store = _image_to_flat(feat)
    probe_stages.append(
        {
            "name": str(name),
            "feat": feat_store.detach().clone().contiguous(),
            "shape": (int(img_shape[0]), int(img_shape[1])),
        }
    )


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
        hidden_dim: int = 32,
        guidance_dim: int = 16,
        stats_code_dim: int = 0,
        stats_hidden_dim: int = 32,
        extra_stats_dim: int = 0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.hidden_dim = int(hidden_dim)
        self.guidance_dim = int(guidance_dim)
        self.stats_code_dim = max(0, int(stats_code_dim))
        self.stats_hidden_dim = int(stats_hidden_dim)
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
            nn.Linear(stats_in_dim, self.stats_hidden_dim),
            nn.GELU(),
            nn.Linear(self.stats_hidden_dim, 5 + self.stats_code_dim),
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
        if self.stats_code_dim > 0:
            stats_code = raw[:, 5:]
        else:
            stats_code = raw.new_zeros((raw.shape[0], 0))

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
        stats_code_dim: int = 0,
        hidden_dim: int = 32,
        hard_one_sided: bool = True,
        hard_boundary_band: bool = False,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feature_dim = int(self.irreps_feat.dim)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.guidance_dim = int(guidance_dim)
        self.stats_code_dim = max(0, int(stats_code_dim))
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
        center_seed_hr: torch.Tensor | None = None,
        center_valid_hr: torch.Tensor | None = None,
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
        feat_center_bilinear = _sample_feat_lr(feat_lr_img, y_base, x_base)
        if center_seed_hr is not None:
            if center_valid_hr is not None:
                center_mix = center_valid_hr.to(
                    device=feat_center_bilinear.device,
                    dtype=feat_center_bilinear.dtype,
                )
                feat_center = (
                    center_mix * center_seed_hr
                    + (1.0 - center_mix) * feat_center_bilinear
                )
            else:
                feat_center = center_seed_hr
        else:
            feat_center = feat_center_bilinear
        feat_plus = _sample_feat_lr(feat_lr_img, y_plus, x_plus)
        feat_minus = _sample_feat_lr(feat_lr_img, y_minus, x_minus)

        side_inputs = [guidance_hr, boundary_prob_hr, sdf_hr]
        if self.stats_code_dim > 0:
            side_inputs.append(stats_code)
        side_in = torch.cat(side_inputs, dim=1)
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
            "feat_shifted_mix_hr": feat_hr,
            "side_logits_hr": side_logits,
            "side_probs_hr": side_probs,
            "boundary_band_hr": band,
        }
        if center_valid_hr is not None:
            aux["center_valid_hr"] = center_valid_hr

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

    if connectivity == 8:
        q_hw = q_bchw.squeeze(0).permute(1, 2, 0)
        qUL = q_hw[:-1, :-1]
        qDR = q_hw[1:, 1:]
        mis_d1_np = _misorientation_angle_sym(qUL, qDR, sym_ops).numpy()
        mask_d1 = mis_d1_np <= thr
        ii, jj = np.where(mask_d1)
        p1 = idx(ii, jj)
        p2 = idx(ii + 1, jj + 1)
        rows += list(p1) + list(p2)
        cols += list(p2) + list(p1)

        qUR = q_hw[:-1, 1:]
        qDL = q_hw[1:, :-1]
        mis_d2_np = _misorientation_angle_sym(qUR, qDL, sym_ops).numpy()
        mask_d2 = mis_d2_np <= thr
        ii, jj = np.where(mask_d2)
        p1 = idx(ii, jj + 1)
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
        sdf_hidden_dim: int = 32,
        guidance_dim: int = 16,
        stats_code_dim: int = 0,
        stats_hidden_dim: int = 32,
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
        feature_upsampler_type: str = "shifted_bilinear",
        upsample_context_kernel_size: int = 3,
        upsample_boundary_threshold: float = 0.5,
        upsample_boundary_smooth_sigma: float = 2.0,
        upsample_boundary_smooth_iters: int = 12,
        upsample_boundary_sdf_shift: float = 0.7,
        use_boundary_gate: bool = False,
        hard_one_sided: bool = True,
        hard_boundary_band: bool = False,
        lambda_feat: float = 1.0,
        lambda_boundary: float = 0.5,
        lambda_lr_boundary: float = 0.10,
        lambda_side_correct: float = 0.10,
        lambda_side_entropy: float = 0.002,
        boundary_thr_deg: float = 3.0,
        boundary_connectivity: int = 4,
        use_focal_boundary: bool = True,
        focal_gamma: float = 2.0,
        side_correct_band_kernel: int | tuple[int, int] | list[int] = (3, 3),
        side_correct_rel_gap: float = 0.05,
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
        self.lambda_feat = float(lambda_feat)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_lr_boundary = float(lambda_lr_boundary)
        self.lambda_side_correct = float(lambda_side_correct)
        self.lambda_side_entropy = float(lambda_side_entropy)
        self.boundary_thr_deg = float(boundary_thr_deg)
        self.boundary_connectivity = int(boundary_connectivity)
        self.use_focal_boundary = bool(use_focal_boundary)
        self.focal_gamma = float(focal_gamma)
        self.side_correct_band_kernel = _as_kernel_tuple(side_correct_band_kernel)
        self.side_correct_rel_gap = float(side_correct_rel_gap)
        self.feature_upsampler_type = str(feature_upsampler_type).strip().lower()
        if self.feature_upsampler_type not in {"shifted_bilinear", "grain_attention"}:
            raise ValueError(
                "feature_upsampler_type must be 'shifted_bilinear' or 'grain_attention', "
                f"got {feature_upsampler_type!r}"
            )
        self.upsample_context_kernel_size = int(upsample_context_kernel_size)
        self.upsample_boundary_threshold = float(upsample_boundary_threshold)
        self.upsample_boundary_smooth_sigma = float(upsample_boundary_smooth_sigma)
        self.upsample_boundary_smooth_iters = int(upsample_boundary_smooth_iters)
        self.upsample_boundary_sdf_shift = float(upsample_boundary_sdf_shift)
        self.use_boundary_gate = bool(use_boundary_gate)

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
            stats_hidden_dim=stats_hidden_dim,
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
        self.grain_attention_helper = None
        if self.feature_upsampler_type == "grain_attention":
            self.grain_attention_helper = BoundaryAwareAttentionUpsampler(
                kernel_size=self.upsample_context_kernel_size,
                upsample_factor=self.upsample_factor,
                use_residual=False,
                use_boundary_gate=self.use_boundary_gate,
                irreps_in=self.irreps_a1,
                irreps_out=self.irreps_a1,
                boundary_threshold=self.upsample_boundary_threshold,
                boundary_smooth_sigma=self.upsample_boundary_smooth_sigma,
                boundary_smooth_iters=self.upsample_boundary_smooth_iters,
                boundary_sdf_shift=self.upsample_boundary_sdf_shift,
            )
            self._random_initialize_grain_attention_helper(self.grain_attention_helper)

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

    def _make_boundary_prep_helper(self) -> BoundaryAwareAttentionUpsampler:
        return BoundaryAwareAttentionUpsampler(
            kernel_size=self.upsample_context_kernel_size,
            upsample_factor=self.upsample_factor,
            use_residual=False,
            use_boundary_gate=self.use_boundary_gate,
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
            boundary_threshold=self.upsample_boundary_threshold,
            boundary_smooth_sigma=self.upsample_boundary_smooth_sigma,
            boundary_smooth_iters=self.upsample_boundary_smooth_iters,
            boundary_sdf_shift=self.upsample_boundary_sdf_shift,
        ).to(self.device)

    def _random_initialize_grain_attention_helper(self, helper: BoundaryAwareAttentionUpsampler) -> None:
        """
        Reinitialize helper parameters so the grain-attention path learns its own
        seed/context behavior instead of inheriting deterministic defaults.
        """
        _randomize_parameter_(helper.log_grain_attn_temp, std=0.02)
        _randomize_parameter_(helper.pos_bias.weight, std=0.02)
        _randomize_parameter_(helper.pos_bias.bias, std=0.02)
        _randomize_parameter_(helper.spatial_weights, std=0.05)
        tp_weight = getattr(helper.tp, "weight", None)
        if isinstance(tp_weight, torch.Tensor):
            _randomize_parameter_(tp_weight, std=0.02)
        residual_proj = getattr(helper, "residual_proj", None)
        if residual_proj is not None:
            reset_parameters = getattr(residual_proj, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

    @staticmethod
    def _thin_boundary_from_labels_2d(
        labels_2d: torch.Tensor,
        connectivity: int = 4,
    ) -> torch.Tensor:
        """One-pixel boundary mask from integer labels with 4- or 8-neighbor connectivity."""
        if labels_2d.ndim != 2:
            raise ValueError(f"Expected (H,W), got {tuple(labels_2d.shape)}")
        if int(connectivity) not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

        b = torch.zeros_like(labels_2d, dtype=torch.bool)
        b[:, 1:] |= labels_2d[:, 1:] != labels_2d[:, :-1]
        b[1:, :] |= labels_2d[1:, :] != labels_2d[:-1, :]
        if int(connectivity) == 8:
            b[1:, 1:] |= labels_2d[1:, 1:] != labels_2d[:-1, :-1]
            b[1:, :-1] |= labels_2d[1:, :-1] != labels_2d[:-1, 1:]
        return b

    @torch.no_grad()
    def _prepare_boundary_context_from_lr_boundary_map(
        self,
        lr_boundary_map: torch.Tensor,
        lr_shape: tuple[int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        learned_sdf_hr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        helper = self.grain_attention_helper
        if helper is None:
            helper = self._make_boundary_prep_helper()

        H, W = lr_shape
        boundary_lr = helper._format_lr_boundary_map(
            lr_boundary_map=lr_boundary_map,
            batch_size=int(batch_size),
            lr_shape=(H, W),
            device=device,
            dtype=dtype,
        )
        boundary_lr_1px, boundary_hr_1px, hr_to_lr_map, lr_labels, lr_labels_dense = (
            helper._build_hr_1px_boundary_and_maps(
                boundary_lr=boundary_lr,
                sdf_hr_override=learned_sdf_hr,
                return_dense_labels=True,
            )
        )
        hr_to_lr_owner = hr_to_lr_map.clone()
        for b in range(int(hr_to_lr_owner.shape[0])):
            if bool((hr_to_lr_owner[b] < 0).any()):
                hr_to_lr_owner[b] = helper._fill_unlabeled_pixels_4n(hr_to_lr_owner[b])
        return {
            "boundary_lr": boundary_lr,
            "boundary_lr_1px": boundary_lr_1px,
            "boundary_hr_1px": boundary_hr_1px,
            "hr_to_lr_map": hr_to_lr_map,
            "hr_to_lr_owner": hr_to_lr_owner,
            "lr_labels": lr_labels,
            "lr_labels_dense": lr_labels_dense,
        }

    @torch.no_grad()
    def _prepare_boundary_context(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        learned_sdf_hr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Build grain-attention boundary context from LR quaternion misorientation:
          - LR grains from misorientation connectivity at boundary_thr_deg
          - HR labels from learned sdf_hr normal-shift remap
        """
        helper = self.grain_attention_helper
        if helper is None:
            helper = self._make_boundary_prep_helper()

        conn = int(self.boundary_connectivity)
        if conn not in (4, 8):
            raise ValueError(f"boundary_connectivity must be 4 or 8, got {conn}")

        H, W = lr_shape
        if lr_quats.dim() == 2:
            lr_quats = lr_quats.unsqueeze(0)
        elif lr_quats.dim() != 3:
            raise ValueError(
                f"lr_quats must have shape (N,4) or (B,N,4), got {tuple(lr_quats.shape)}"
            )
        if int(lr_quats.shape[-1]) != 4:
            raise ValueError(f"Expected quaternion last dim=4, got {tuple(lr_quats.shape)}")

        B = int(lr_quats.shape[0])
        expected_n = int(H * W)
        if int(lr_quats.shape[1]) != expected_n:
            raise ValueError(
                f"lr_quats has N={int(lr_quats.shape[1])}, expected {expected_n} from lr_shape={(H, W)}"
            )

        lr_quats = _normalize_quaternions(lr_quats.to(device=device, dtype=torch.float32))
        lr_labels_dense, boundary_lr_1px = self.batch_boundary_teacher_from_quats(
            quats=lr_quats,
            img_shape=(H, W),
            thr_deg=float(self.boundary_thr_deg),
            connectivity=conn,
        )

        lr_labels_dense = lr_labels_dense.to(device=device, dtype=torch.long)
        boundary_lr_1px = boundary_lr_1px.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        if learned_sdf_hr.dim() == 4:
            if learned_sdf_hr.shape[0] != B or learned_sdf_hr.shape[1] != 1:
                raise ValueError(
                    f"Expected learned_sdf_hr shape ({B},1,Hr,Wr), got {tuple(learned_sdf_hr.shape)}"
                )
            Hr, Wr = int(learned_sdf_hr.shape[-2]), int(learned_sdf_hr.shape[-1])
            sdf_hr_all = learned_sdf_hr[:, 0].to(device=device, dtype=dtype)
        elif learned_sdf_hr.dim() == 3:
            if learned_sdf_hr.shape[0] != B:
                raise ValueError(
                    f"Expected learned_sdf_hr batch {B}, got {tuple(learned_sdf_hr.shape)}"
                )
            Hr, Wr = int(learned_sdf_hr.shape[-2]), int(learned_sdf_hr.shape[-1])
            sdf_hr_all = learned_sdf_hr.to(device=device, dtype=dtype)
        else:
            raise ValueError(
                f"learned_sdf_hr must have shape (B,1,Hr,Wr) or (B,Hr,Wr), got {tuple(learned_sdf_hr.shape)}"
            )

        boundary_lr = boundary_lr_1px.clone()
        boundary_hr_1px = torch.zeros((B, 1, Hr, Wr), device=device, dtype=dtype)
        hr_to_lr_map = torch.full((B, Hr, Wr), -1, device=device, dtype=torch.long)
        hr_to_lr_owner = torch.full((B, Hr, Wr), -1, device=device, dtype=torch.long)
        lr_labels = torch.full((B, H, W), -1, device=device, dtype=torch.long)

        for b in range(B):
            labels_lr_dense_b = lr_labels_dense[b]
            b_lr = self._thin_boundary_from_labels_2d(labels_lr_dense_b, connectivity=conn)
            boundary_lr_1px[b, 0] = b_lr.to(dtype=dtype)
            boundary_lr[b, 0] = b_lr.to(dtype=dtype)

            labels_lr_sparse_b = labels_lr_dense_b.clone()
            labels_lr_sparse_b[b_lr] = -1
            lr_labels[b] = labels_lr_sparse_b

            valid_ids = labels_lr_dense_b[labels_lr_dense_b >= 0]
            if valid_ids.numel() > 0:
                num_components = int(valid_ids.max().item()) + 1
                tiny_ids, narrow_ids = helper._component_ids_by_size_and_width(
                    labels_2d=labels_lr_dense_b,
                    num_components=num_components,
                    tiny_size_max=int(helper.tiny_component_max_pixels),
                    narrow_width_max=int(helper.narrow_component_max_width),
                )
            else:
                tiny_ids = torch.empty((0,), device=device, dtype=torch.long)
                narrow_ids = torch.empty((0,), device=device, dtype=torch.long)

            sdf_hr = sdf_hr_all[b]
            shift_scale_hr = None
            if narrow_ids.numel() > 0:
                narrow_lr_mask = helper._labels_mask_from_ids(labels_lr_dense_b, narrow_ids)
                if bool(narrow_lr_mask.any()):
                    narrow_hr_mask = F.interpolate(
                        narrow_lr_mask.to(dtype=sdf_hr.dtype).unsqueeze(0).unsqueeze(0),
                        size=(Hr, Wr),
                        mode="nearest",
                    )[0, 0] > 0.5
                    shift_scale_hr = torch.ones((Hr, Wr), device=device, dtype=sdf_hr.dtype)
                    shift_scale_hr[narrow_hr_mask] = float(helper.narrow_region_shift_scale)

            labels_hr = helper._remap_lr_labels_to_hr_via_sdf(
                labels_lr_dense_b,
                sdf_hr,
                shift_scale_hr=shift_scale_hr,
            )
            labels_hr = helper._reinject_missing_components_from_lr(labels_hr, labels_lr_sparse_b, tiny_ids)

            # Dense owner map for grain-attention seeding: every HR pixel keeps a grain id
            # from the learned-SDF remap, while hr_to_lr_map stays sparse for diagnostics.
            hr_to_lr_owner[b] = labels_hr

            b_hr = self._thin_boundary_from_labels_2d(labels_hr, connectivity=conn)
            boundary_hr_1px[b, 0] = b_hr.to(dtype=dtype)

            hr_map = labels_hr.clone()
            hr_map[b_hr] = -1
            hr_to_lr_map[b] = hr_map
        return {
            "boundary_lr": boundary_lr,
            "boundary_lr_1px": boundary_lr_1px,
            "boundary_hr_1px": boundary_hr_1px,
            "hr_to_lr_map": hr_to_lr_map,
            "hr_to_lr_owner": hr_to_lr_owner,
            "lr_labels": lr_labels,
            "lr_labels_dense": lr_labels_dense,
        }

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
        lr_boundary_map: torch.Tensor | None = None,
        lr_quats: torch.Tensor | None = None,
        extra_material_stats: torch.Tensor | None = None,
        return_aux: bool = False,
        return_probe: bool = False,
    ):
        """Core SR forward pass in A1 space; decode is intentionally separate."""
        feat = feat_lr_a1.to(self.device)
        probe_stages: list[dict[str, object]] | None = [] if return_probe else None
        _append_probe_stage(probe_stages, "encode_a1_lr", feat, lr_shape)

        for bi, blk in enumerate(self.lr_blocks, start=1):
            feat = blk(feat, lr_shape)
            _append_probe_stage(probe_stages, f"lr_block_{bi}", feat, lr_shape)

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

        boundary_ctx = None
        center_seed_hr = None
        center_valid_hr = None
        if self.feature_upsampler_type == "grain_attention":
            if lr_quats is not None:
                boundary_ctx = self._prepare_boundary_context(
                    lr_quats=lr_quats,
                    lr_shape=lr_shape,
                    device=feat.device,
                    dtype=feat.dtype,
                    learned_sdf_hr=sdf_out["sdf_hr"],
                )
            elif lr_boundary_map is not None:
                batch_size = int(feat.shape[0]) if feat.dim() == 3 else 1
                boundary_ctx = self._prepare_boundary_context_from_lr_boundary_map(
                    lr_boundary_map=lr_boundary_map,
                    lr_shape=lr_shape,
                    batch_size=batch_size,
                    device=feat.device,
                    dtype=feat.dtype,
                    learned_sdf_hr=sdf_out["sdf_hr"],
                )
            else:
                raise ValueError(
                    "grain_attention requires lr_quats for 3-degree misorientation LR grains."
                )
            helper = self.grain_attention_helper
            if helper is None:
                raise RuntimeError("grain_attention_helper is not initialized.")
            center_seed_hr_flat, center_shape = helper(
                features=feat,
                img_shape=lr_shape,
                hr_to_lr_map=boundary_ctx["hr_to_lr_owner"],
                lr_labels=boundary_ctx["lr_labels"],
            )
            if tuple(center_shape) != tuple(sdf_out["hr_shape"]):
                raise RuntimeError(
                    f"grain attention produced HR shape {tuple(center_shape)}, "
                    f"expected {tuple(sdf_out['hr_shape'])}"
                )
            center_seed_hr = _flat_to_image(center_seed_hr_flat, center_shape)
            if center_seed_hr.dim() == 3:
                center_seed_hr = center_seed_hr.unsqueeze(0)
            _append_probe_stage(probe_stages, "grain_attention_out", center_seed_hr_flat, center_shape)
            center_valid_hr = (
                (boundary_ctx["hr_to_lr_owner"] >= 0)
                .to(device=feat.device, dtype=feat.dtype)
                .unsqueeze(1)
            )

        feat_hr, up_aux = self.sdf_upsample(
            feat_lr=feat,
            lr_shape=lr_shape,
            sdf_out=sdf_out,
            center_seed_hr=center_seed_hr,
            center_valid_hr=center_valid_hr,
            return_aux=True,
        )

        _append_probe_stage(probe_stages, "upsample_center_hr", up_aux["feat_center_hr"], sdf_out["hr_shape"])
        _append_probe_stage(probe_stages, "upsample_plus_hr", up_aux["feat_plus_hr"], sdf_out["hr_shape"])
        _append_probe_stage(probe_stages, "upsample_minus_hr", up_aux["feat_minus_hr"], sdf_out["hr_shape"])
        _append_probe_stage(probe_stages, "upsample_shifted_mix_hr", up_aux["feat_shifted_mix_hr"], sdf_out["hr_shape"])

        for bi, blk in enumerate(self.hr_blocks, start=1):
            feat_hr = blk(
                feat_hr,
                sdf_out["hr_shape"],
                guidance=sdf_out["guidance_hr"],
                boundary_logits=sdf_out["boundary_logits_hr"],
            )
            _append_probe_stage(probe_stages, f"hr_block_{bi}", feat_hr, sdf_out["hr_shape"])

        boundary_logits_hr_refined = None
        if self.refinement_head is not None:
            if return_probe:
                feat_hr, boundary_logits_hr_refined, refinement_trace = self.refinement_head(
                    feat_hr,
                    sdf_out["hr_shape"],
                    guidance=sdf_out["guidance_hr"],
                    boundary_logits=sdf_out["boundary_logits_hr"],
                    return_trace=True,
                )
                for trace_item in refinement_trace:
                    _append_probe_stage(
                        probe_stages,
                        f"refine_step_{int(trace_item['step'])}",
                        trace_item["feat"],
                        sdf_out["hr_shape"],
                    )
            else:
                feat_hr, boundary_logits_hr_refined = self.refinement_head(
                    feat_hr,
                    sdf_out["hr_shape"],
                    guidance=sdf_out["guidance_hr"],
                    boundary_logits=sdf_out["boundary_logits_hr"],
                )

        _append_probe_stage(probe_stages, "final_feat_before_decode", feat_hr, sdf_out["hr_shape"])

        if return_aux:
            aux = {}
            aux.update(evidence)
            aux.update(sdf_out)
            aux.update(up_aux)
            aux["feature_upsampler_type"] = self.feature_upsampler_type
            if boundary_ctx is not None:
                aux.update(boundary_ctx)
            aux["feat_lr_a1_post_lr"] = feat
            aux["feat_hr_a1"] = feat_hr
            aux["boundary_logits_hr_refined"] = boundary_logits_hr_refined
            if probe_stages is not None:
                aux["probe_stages"] = probe_stages
            return feat_hr, sdf_out["hr_shape"], aux

        return feat_hr, sdf_out["hr_shape"]

    def forward_sr(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        lr_boundary_map: torch.Tensor | None = None,
        normalize_input: bool = True,
        extra_material_stats: torch.Tensor | None = None,
        return_aux: bool = False,
        return_probe: bool = False,
    ):
        """End-to-end SR wrapper from LR quaternions (returns quats and optional aux)."""
        if return_probe and not return_aux:
            raise ValueError("return_probe=True requires return_aux=True.")
        lr_quats = lr_quats.to(self.device)
        if normalize_input:
            lr_quats = _normalize_quaternions(lr_quats)

        feat_lr_a1 = self.encode_a1(lr_quats)

        if return_aux:
            feat_hr_a1, _, aux = self._forward_sr_features(
                feat_lr_a1=feat_lr_a1,
                lr_shape=lr_shape,
                lr_boundary_map=lr_boundary_map,
                lr_quats=lr_quats,
                extra_material_stats=extra_material_stats,
                return_aux=True,
                return_probe=return_probe,
            )
            q_hr = self.decode(feat_hr_a1)
            return q_hr, aux

        feat_hr_a1, _ = self._forward_sr_features(
            feat_lr_a1=feat_lr_a1,
            lr_shape=lr_shape,
            lr_boundary_map=lr_boundary_map,
            lr_quats=lr_quats,
            extra_material_stats=extra_material_stats,
            return_aux=False,
            return_probe=False,
        )
        return self.decode(feat_hr_a1)

    def feature_loss_sr(
        self,
        lr_quats: torch.Tensor,
        hr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        lr_boundary_map: torch.Tensor | None = None,
        normalize_input: bool = False,
        extra_material_stats: torch.Tensor | None = None,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        hr_shape = (
            int(lr_shape[0]) * int(self.upsample_factor[0]),
            int(lr_shape[1]) * int(self.upsample_factor[1]),
        )
        expected_hr_n = int(hr_shape[0] * hr_shape[1])
        actual_hr_n = int(hr_quats.shape[1] if batched else hr_quats.shape[0])
        if actual_hr_n != expected_hr_n:
            raise ValueError(
                f"Expected HR quaternions with N={expected_hr_n} from lr_shape={lr_shape} "
                f"and scale={self.upsample_factor}, got N={actual_hr_n}"
            )

        feat_hr, _, aux = self._forward_sr_features(
            feat_lr_a1=feat_lr_a1,
            lr_shape=lr_shape,
            lr_boundary_map=lr_boundary_map,
            lr_quats=lr_quats,
            extra_material_stats=extra_material_stats,
            return_aux=True,
        )
        loss_feat = F.mse_loss(feat_hr, feat_hr_tgt)
        total = self.lambda_feat * loss_feat
        info: dict[str, torch.Tensor] = {
            "loss_feat": loss_feat.detach(),
        }

        needs_boundary_teacher = any(
            weight > 0.0
            for weight in (
                self.lambda_boundary,
                self.lambda_side_correct,
                self.lambda_side_entropy,
            )
        )

        teacher_band = None
        if needs_boundary_teacher:
            loss_boundary_raw, boundary_info = self.boundary_supervision_loss(
                aux=aux,
                hr_quats=hr_quats,
                hr_shape=hr_shape,
                lr_quats=lr_quats,
                lr_shape=lr_shape,
                thr_deg=self.boundary_thr_deg,
                connectivity=self.boundary_connectivity,
                lambda_hr=1.0,
                lambda_lr=self.lambda_lr_boundary,
                use_refined_hr=True,
                use_focal=self.use_focal_boundary,
                focal_gamma=self.focal_gamma,
            )
            total = total + self.lambda_boundary * loss_boundary_raw
            info["loss_boundary"] = loss_boundary_raw.detach()
            teacher_band = _dilate_binary_mask(
                boundary_info["gb_target_hr"],
                self.side_correct_band_kernel,
            )
        else:
            info["loss_boundary"] = total.new_zeros(())

        if (self.lambda_side_correct > 0.0 or self.lambda_side_entropy > 0.0) and teacher_band is not None:
            feat_hr_tgt_img = _flat_to_image(feat_hr_tgt, hr_shape)
            if feat_hr_tgt_img.dim() == 3:
                feat_hr_tgt_img = feat_hr_tgt_img.unsqueeze(0)

            feat_plus = aux["feat_plus_hr"]
            feat_minus = aux["feat_minus_hr"]
            side_logits = aux["side_logits_hr"]
            side_probs = aux["side_probs_hr"]

            if feat_plus.dim() == 3:
                feat_plus = feat_plus.unsqueeze(0)
            if feat_minus.dim() == 3:
                feat_minus = feat_minus.unsqueeze(0)
            if side_logits.dim() == 3:
                side_logits = side_logits.unsqueeze(0)
            if side_probs.dim() == 3:
                side_probs = side_probs.unsqueeze(0)

            if self.lambda_side_correct > 0.0:
                loss_side_correct = side_correctness_loss(
                    side_logits=side_logits,
                    feat_plus=feat_plus,
                    feat_minus=feat_minus,
                    feat_target=feat_hr_tgt_img,
                    teacher_band=teacher_band,
                    rel_gap_threshold=self.side_correct_rel_gap,
                )
                total = total + self.lambda_side_correct * loss_side_correct
                info["loss_side_correct"] = loss_side_correct.detach()
            else:
                info["loss_side_correct"] = total.new_zeros(())

            if self.lambda_side_entropy > 0.0:
                loss_side_entropy = side_entropy_loss(
                    side_probs=side_probs,
                    mask=teacher_band,
                )
                total = total + self.lambda_side_entropy * loss_side_entropy
                info["loss_side_entropy"] = loss_side_entropy.detach()
            else:
                info["loss_side_entropy"] = total.new_zeros(())
        else:
            info["loss_side_correct"] = total.new_zeros(())
            info["loss_side_entropy"] = total.new_zeros(())

        info["loss_total"] = total.detach()
        if return_info:
            return total, info
        return total

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
        lr_boundary_map: torch.Tensor | None = None,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        quats = quats.to(self.device)
        if quats.dim() != 2 or quats.shape[-1] != 4:
            raise ValueError(f"IsoEmbeddingSRSDF expects (N,4), got {tuple(quats.shape)}")
        if normalize_input:
            quats = _normalize_quaternions(quats)

        if img_shape is not None:
            return self.forward_sr(
                quats,
                lr_shape=img_shape,
                lr_boundary_map=lr_boundary_map,
                normalize_input=False,
            )

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
    lambda_boundary: float = 0.5
    lambda_lr_boundary: float = 0.10
    lambda_side_correct: float = 0.10
    lambda_side_entropy: float = 0.002
    boundary_thr_deg: float = 3.0
    boundary_connectivity: int = 4
    use_focal_boundary: bool = True
    focal_gamma: float = 2.0
    side_correct_band_kernel: tuple[int, int] = (3, 3)
    side_correct_rel_gap: float = 0.05


def side_entropy_loss(
    side_probs: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    side_probs: (B,2,H,W)
    Encourage decisive side assignment near boundaries (low entropy selector).
    """
    p = side_probs.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=1, keepdim=True)
    return _masked_mean(ent, mask)


def side_correctness_loss(
    side_logits: torch.Tensor,
    feat_plus: torch.Tensor,
    feat_minus: torch.Tensor,
    feat_target: torch.Tensor,
    teacher_band: torch.Tensor,
    rel_gap_threshold: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Supervise plus/minus side selection only near teacher boundaries.

    The target side is whichever candidate feature is closer to the encoded HR
    target at that HR pixel. Ambiguous pixels are ignored with a relative-gap
    threshold so smooth intragranular variation does not dominate the loss.
    """
    d_plus = ((feat_plus - feat_target) ** 2).mean(dim=1, keepdim=True)
    d_minus = ((feat_minus - feat_target) ** 2).mean(dim=1, keepdim=True)
    rel_gap = (d_plus - d_minus).abs() / (d_plus + d_minus).clamp_min(eps)
    conf_mask = (rel_gap > float(rel_gap_threshold)).to(dtype=d_plus.dtype)
    mask = teacher_band.to(dtype=d_plus.dtype) * conf_mask

    target_side = (d_minus < d_plus).long().squeeze(1)
    ce = F.cross_entropy(side_logits, target_side, reduction="none").unsqueeze(1)
    return _masked_mean(ce, mask)


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
    lr_boundary_map = batch.get("lr_boundary_map", None)
    if lr_boundary_map is not None:
        lr_boundary_map = lr_boundary_map.to(model.device)
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
        lr_boundary_map=lr_boundary_map,
        lr_quats=lr_quats,
        extra_material_stats=extra_stats,
        return_aux=True,
    )

    loss_feat = F.mse_loss(feat_hr_pred, feat_hr_tgt)

    loss_boundary, boundary_info = model.boundary_supervision_loss(
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

    teacher_band = _dilate_binary_mask(boundary_info["gb_target_hr"], cfg.side_correct_band_kernel)
    feat_hr_tgt_img = _flat_to_image(feat_hr_tgt, hr_shape)
    loss_side_correct = side_correctness_loss(
        side_logits=aux["side_logits_hr"],
        feat_plus=aux["feat_plus_hr"],
        feat_minus=aux["feat_minus_hr"],
        feat_target=feat_hr_tgt_img,
        teacher_band=teacher_band,
        rel_gap_threshold=cfg.side_correct_rel_gap,
    )
    loss_side = side_entropy_loss(aux["side_probs_hr"], mask=teacher_band)
    loss = (
        cfg.lambda_feat * loss_feat
        + cfg.lambda_boundary * loss_boundary
        + cfg.lambda_side_correct * loss_side_correct
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
        "loss_side_correct": float(loss_side_correct.item()),
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
    lr_boundary_map = batch.get("lr_boundary_map", None)
    if lr_boundary_map is not None:
        lr_boundary_map = lr_boundary_map.to(model.device)
    extra_stats = batch.get("stats", None)
    if extra_stats is not None:
        extra_stats = extra_stats.to(model.device)

    feat_lr = model.encode_a1(lr_quats)
    feat_hr_tgt = model.encode_a1(hr_quats)
    feat_hr_pred, _, aux = model._forward_sr_features(
        feat_lr_a1=feat_lr,
        lr_shape=lr_shape,
        lr_boundary_map=lr_boundary_map,
        lr_quats=lr_quats,
        extra_material_stats=extra_stats,
        return_aux=True,
    )
    loss_feat = F.mse_loss(feat_hr_pred, feat_hr_tgt)

    boundary_loss, boundary_info = model.boundary_supervision_loss(
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
    teacher_band = _dilate_binary_mask(boundary_info["gb_target_hr"], cfg.side_correct_band_kernel)
    feat_hr_tgt_img = _flat_to_image(feat_hr_tgt, hr_shape)
    loss_side_correct = side_correctness_loss(
        side_logits=aux["side_logits_hr"],
        feat_plus=aux["feat_plus_hr"],
        feat_minus=aux["feat_minus_hr"],
        feat_target=feat_hr_tgt_img,
        teacher_band=teacher_band,
        rel_gap_threshold=cfg.side_correct_rel_gap,
    )
    loss_side_entropy = side_entropy_loss(aux["side_probs_hr"], mask=teacher_band)
    total_loss = (
        cfg.lambda_feat * loss_feat
        + cfg.lambda_boundary * boundary_loss
        + cfg.lambda_side_correct * loss_side_correct
        + cfg.lambda_side_entropy * loss_side_entropy
    )

    return {
        "loss_total": float(total_loss.item()),
        "loss_feat": float(loss_feat.item()),
        "loss_boundary": float(boundary_loss.item()),
        "loss_side_correct": float(loss_side_correct.item()),
        "loss_side_entropy": float(loss_side_entropy.item()),
    }


__all__ = [
    "IrrepEvidenceBoundarySDFHead",
    "IrrepShiftedSDFUpsample",
    "IsoEmbeddingSRSDF",
    "graph_boundary_teacher_from_quats",
    "TrainConfig",
    "side_entropy_loss",
    "side_correctness_loss",
    "train_step_feature_space",
    "validate_batch_feature_space",
]
