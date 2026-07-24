from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear as IrrepsLinear

from models.local_iso_embedding import (
    build_fcc_syms_mtex,
    build_hcp_syms_mtex,
    build_local_iso_fcc_embedding,
    build_local_iso_hcp_embedding,
)


def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
    q = quats / norm
    return torch.where(q[..., :1] < 0.0, -q, q)


def _quat_conjugate(quats: torch.Tensor) -> torch.Tensor:
    return torch.cat([quats[..., :1], -quats[..., 1:]], dim=-1)


def _sample_fz_quaternions_passive(
    group_name: str,
    resolution: int,
    method: str,
    dtype: torch.dtype,
    device: torch.device,
    max_rows: int | None = None,
) -> torch.Tensor:
    from orix.quaternion import symmetry
    from orix.sampling import get_sample_fundamental

    g = str(group_name).upper()
    if g == "O":
        point_group = symmetry.Oh
    elif g == "D6":
        point_group = symmetry.D6h
    else:
        raise ValueError(f"group_name must be 'O' or 'D6', got {group_name}")

    rot = get_sample_fundamental(
        int(resolution),
        point_group=point_group,
        method=str(method),
    )

    raw = np.asarray(getattr(rot, "data", rot), dtype=np.float32)
    if raw.ndim != 2:
        raw = raw.reshape(-1, 4)
    if raw.shape[-1] != 4 and raw.shape[0] == 4:
        raw = raw.T
    if raw.shape[-1] != 4:
        raise ValueError(f"Unexpected sampled quaternion shape: {tuple(raw.shape)}")

    q = torch.as_tensor(raw, dtype=dtype, device=device)
    q = _normalize_quaternions(q)
    if max_rows is not None:
        q = q[: int(max_rows)]
    return q


def _irrep_block_slices(irreps: Irreps) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    start = 0
    for mul, ir in Irreps(irreps):
        width = int(mul) * int(ir.dim)
        slices.append((start, start + width))
        start += width
    return slices


def resolve_ocrp_upsample_residual_weight(
    cfg,
    *,
    epoch: int | None = None,
    total_epochs: int | None = None,
    for_training: bool = False,
) -> float:
    """Resolve the OCRP LR-anchor residual weight from config and training progress."""
    start = float(getattr(cfg, "ocrp_upsample_residual_weight", 1.0))
    final_cfg = getattr(cfg, "ocrp_upsample_residual_weight_final", None)
    final = start if final_cfg is None else float(final_cfg)
    schedule = str(
        getattr(cfg, "ocrp_upsample_residual_weight_schedule", "constant")
    ).strip().lower()

    if start < 0.0 or final < 0.0:
        raise ValueError(
            "OCRP upsample residual weights must be >= 0, "
            f"got start={start} final={final}"
        )

    if schedule in {"", "constant", "none", "off"}:
        return start

    if schedule not in {"linear", "cosine"}:
        raise ValueError(
            "ocrp_upsample_residual_weight_schedule must be one of "
            f"['constant', 'linear', 'cosine'], got {schedule!r}"
        )

    if not for_training:
        return final

    if epoch is None or total_epochs is None or int(total_epochs) <= 1:
        return start

    denom = max(1, int(total_epochs) - 1)
    progress = min(max(float(epoch) / float(denom), 0.0), 1.0)
    if schedule == "cosine":
        mix = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        mix = progress
    return (1.0 - mix) * start + mix * final


def _approx_l2_threshold_from_legacy_cosine_threshold(cosine_threshold: float) -> float:
    """
    Approximate an embedding-space L2 cutoff from the earlier cosine-based OCRP mask.

    This uses the measured FCC local-iso correspondence between cosine thresholds
    and misorientation angles, then maps angle to radians because the embedding is
    close to isometric in L2.
    """
    cos_points = np.array([0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995], dtype=np.float64)
    deg_points = np.array([15.0, 13.0, 11.0, 8.0, 6.0, 5.0, 4.0, 3.0], dtype=np.float64)
    cos_clamped = float(np.clip(float(cosine_threshold), float(cos_points[0]), float(cos_points[-1])))
    approx_deg = float(np.interp(cos_clamped, cos_points, deg_points))
    return float(math.radians(approx_deg))


def _resolve_feature_l2_threshold(
    *,
    explicit_l2_threshold: float | None,
    legacy_cosine_threshold: float | None,
    default_l2_threshold: float,
) -> float:
    if explicit_l2_threshold is not None:
        return float(explicit_l2_threshold)
    if legacy_cosine_threshold is not None:
        return _approx_l2_threshold_from_legacy_cosine_threshold(float(legacy_cosine_threshold))
    return float(default_l2_threshold)


def _resolve_cluster_feature_l2_threshold(
    *,
    cluster_feature_l2_threshold: float | None,
    legacy_cluster_threshold_deg: float,
) -> float:
    if cluster_feature_l2_threshold is not None:
        return float(cluster_feature_l2_threshold)
    return float(math.radians(float(legacy_cluster_threshold_deg)))


def _as_scale_tuple(scale: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError(f"Expected scale as int or length-2 tuple/list, got {scale}")
        scale_y = int(scale[0])
        scale_x = int(scale[1])
    else:
        scale_y = scale_x = int(scale)
    if scale_y < 1 or scale_x < 1:
        raise ValueError(f"scale values must be >= 1, got {(scale_y, scale_x)}")
    return (scale_y, scale_x)


def _as_patch_shape(
    patch_size: int | tuple[int, int] | list[int],
    *,
    name: str,
) -> tuple[int, int]:
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2:
            raise ValueError(f"Expected {name} as int or length-2 tuple/list, got {patch_size}")
        patch_h = int(patch_size[0])
        patch_w = int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)
    if patch_h < 1 or patch_w < 1:
        raise ValueError(f"{name} must be >= 1 in both dims, got {(patch_h, patch_w)}")
    return (patch_h, patch_w)


def _num_patch_tokens(patch_shape: tuple[int, int]) -> int:
    return int(patch_shape[0]) * int(patch_shape[1])


def _as_isotropic_scale(scale: int | tuple[int, int] | list[int]) -> int:
    scale_y, scale_x = _as_scale_tuple(scale)
    if scale_y != scale_x:
        raise ValueError(f"OCRP currently assumes isotropic SR, got scale={scale}")
    return int(scale_y)


def _left_mult_matrix_wxyz_batch(q_syms: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q_syms.unbind(dim=-1)
    r0 = torch.stack([w, -x, -y, -z], dim=-1)
    r1 = torch.stack([x, w, -z, y], dim=-1)
    r2 = torch.stack([y, z, w, -x], dim=-1)
    r3 = torch.stack([z, -y, x, w], dim=-1)
    return torch.stack([r0, r1, r2, r3], dim=1)


def _misorientation_angle_sym(
    q1: torch.Tensor,
    q2: torch.Tensor,
    sym_ops: torch.Tensor,
) -> torch.Tensor:
    q2var = torch.einsum("gij,...j->...gi", sym_ops, q2)
    dots = (q1.unsqueeze(-2) * q2var).sum(dim=-1).abs().clamp(0.0, 1.0)
    best = dots.max(dim=-1).values
    return 2.0 * torch.acos(best)


def _flat_to_image(features: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
    h, w = int(img_shape[0]), int(img_shape[1])
    batched = features.dim() == 3
    if not batched:
        features = features.unsqueeze(0)
    bsz, n, cdim = features.shape
    if n != h * w:
        raise ValueError(f"Expected N={h*w}, got {n}")
    img = features.view(bsz, h, w, cdim).permute(0, 3, 1, 2).contiguous()
    if not batched:
        img = img.squeeze(0)
    return img


def _build_local_patch_bank(
    features: torch.Tensor,
    img_shape: tuple[int, int],
    window_size: int,
) -> torch.Tensor:
    h, w = int(img_shape[0]), int(img_shape[1])
    batched = features.dim() == 3
    if not batched:
        features = features.unsqueeze(0)
    bsz, n, cdim = features.shape
    if n != h * w:
        raise ValueError(f"Expected N={h*w}, got {n}")

    feat_img = features.view(bsz, h, w, cdim).permute(0, 3, 1, 2).contiguous()
    pad = int(window_size // 2)
    feat_pad = F.pad(feat_img, (pad, pad, pad, pad), mode="replicate")
    patches = feat_pad.unfold(2, window_size, 1).unfold(3, window_size, 1)
    bank = (
        patches.permute(0, 2, 3, 4, 5, 1)
        .contiguous()
        .reshape(bsz, h * w, window_size * window_size, cdim)
    )
    if not batched:
        bank = bank.squeeze(0)
    return bank


def _resolve_ocrp_mode(ocrp_mode: str) -> str:
    mode = str(ocrp_mode).strip().lower()
    if mode not in {"pixel_patch", "macro_tile"}:
        raise ValueError(
            f"ocrp_mode must be 'pixel_patch' or 'macro_tile', got {ocrp_mode}"
        )
    return mode


def _validate_odd_positive_int(name: str, value: int) -> int:
    value_int = int(value)
    if value_int < 1 or value_int % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}")
    return value_int


def _build_macro_tile_patch_bank(
    features: torch.Tensor,
    img_shape: tuple[int, int],
    window_size: int,
    tile_size: int,
) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    h, w = int(img_shape[0]), int(img_shape[1])
    tile_size = _validate_odd_positive_int("macro_lr_tile_size", tile_size)
    batched = features.dim() == 3
    if not batched:
        features = features.unsqueeze(0)
    bsz, n, cdim = features.shape
    if n != h * w:
        raise ValueError(f"Expected N={h*w}, got {n}")

    tile_h = (h + tile_size - 1) // tile_size
    tile_w = (w + tile_size - 1) // tile_size
    h_pad = tile_h * tile_size
    w_pad = tile_w * tile_size
    center = tile_size // 2

    feat_img = features.view(bsz, h, w, cdim).permute(0, 3, 1, 2).contiguous()
    if h_pad != h or w_pad != w:
        feat_img = F.pad(
            feat_img,
            (0, w_pad - w, 0, h_pad - h),
            mode="replicate",
        )
    pad = int(window_size // 2)
    feat_pad = F.pad(feat_img, (pad, pad, pad, pad), mode="replicate")
    sampled = feat_pad[:, :, center:, center:]
    patches = sampled.unfold(2, window_size, tile_size).unfold(3, window_size, tile_size)
    if int(patches.shape[2]) != tile_h or int(patches.shape[3]) != tile_w:
        raise ValueError(
            "Macro-tile support extraction produced an unexpected grid shape: "
            f"got {(int(patches.shape[2]), int(patches.shape[3]))}, "
            f"expected {(tile_h, tile_w)}"
        )

    bank = (
        patches.permute(0, 2, 3, 4, 5, 1)
        .contiguous()
        .reshape(bsz, tile_h * tile_w, window_size * window_size, cdim)
    )
    if not batched:
        bank = bank.squeeze(0)
    return bank, {"tile_shape": (tile_h, tile_w), "padded_shape": (h_pad, w_pad)}


def _build_patch_token_coords(
    patch_shape: int | tuple[int, int] | list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(patch_shape, int):
        patch_tokens = int(patch_shape)
        if patch_tokens < 1:
            raise ValueError(f"patch_tokens must be >= 1, got {patch_tokens}")
        patch_edge = int(round(float(patch_tokens) ** 0.5))
        if patch_edge * patch_edge != patch_tokens:
            raise ValueError(f"patch_tokens must be a perfect square when patch shape is implicit, got {patch_tokens}")
        patch_h, patch_w = patch_edge, patch_edge
    else:
        patch_h, patch_w = _as_patch_shape(patch_shape, name="patch_shape")
        patch_tokens = _num_patch_tokens((patch_h, patch_w))
    if patch_h == 1 and patch_w == 1:
        return torch.zeros((1, 2), device=device, dtype=dtype)
    coord_y = (
        torch.zeros((1,), device=device, dtype=dtype)
        if patch_h == 1
        else torch.linspace(-1.0, 1.0, steps=patch_h, device=device, dtype=dtype)
    )
    coord_x = (
        torch.zeros((1,), device=device, dtype=dtype)
        if patch_w == 1
        else torch.linspace(-1.0, 1.0, steps=patch_w, device=device, dtype=dtype)
    )
    yy, xx = torch.meshgrid(coord_y, coord_x, indexing="ij")
    return torch.stack([yy, xx], dim=-1).reshape(patch_tokens, 2)


class FeatureDistanceMaskedEquivariantSpatialConv(nn.Module):
    """
    Equivariant local convolution with a feature-space Euclidean-distance mask.

    The local context stays a weighted average over existing nearby orientation
    features, but neighbors whose embedding L2 distance to the center rises
    above the configured threshold are excluded before renormalization.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        irreps_in: Irreps | str = "1x4e",
        irreps_out: Irreps | str | None = None,
        use_residual: bool = False,
        residual_weight: float = 1.0,
        dilation: int = 1,
        feature_mask_l2_threshold: float | None = None,
        feature_mask_cosine_threshold: float | None = None,
        feature_mask_soft: bool = False,
        feature_mask_temperature: float = 32.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding = (self.kernel_size // 2) * self.dilation
        self.feature_mask_l2_threshold = _resolve_feature_l2_threshold(
            explicit_l2_threshold=feature_mask_l2_threshold,
            legacy_cosine_threshold=feature_mask_cosine_threshold,
            default_l2_threshold=math.radians(5.0),
        )
        self.feature_mask_cosine_threshold = (
            None if feature_mask_cosine_threshold is None else float(feature_mask_cosine_threshold)
        )
        self.feature_mask_soft = bool(feature_mask_soft)
        self.feature_mask_temperature = float(feature_mask_temperature)
        self.eps = float(eps)

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out) if irreps_out is not None else self.irreps_in
        self.in_dim = int(self.irreps_in.dim)
        self.out_dim = int(self.irreps_out.dim)
        self.use_residual = bool(use_residual)
        self.residual_weight = float(residual_weight)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_out,
            shared_weights=True,
        )
        self.residual_proj: o3.Linear | None = None
        if self.use_residual and (self.irreps_in != self.irreps_out):
            self.residual_proj = o3.Linear(self.irreps_in, self.irreps_out)

        self.spatial_logits = nn.Parameter(torch.zeros(self.kernel_size, self.kernel_size))
        center_mask = torch.zeros(self.kernel_size, self.kernel_size, dtype=torch.bool)
        center_mask[self.kernel_size // 2, self.kernel_size // 2] = True
        self.register_buffer("center_mask", center_mask.view(1, 1, 1, self.kernel_size, self.kernel_size), persistent=False)

    def _extract_patches(self, feat_img: torch.Tensor) -> torch.Tensor:
        bsz, cdim, h, w = feat_img.shape
        feat_padded = F.pad(
            feat_img,
            (self.padding, self.padding, self.padding, self.padding),
            mode="replicate",
        )
        if self.dilation == 1:
            return feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        patches = F.unfold(
            feat_padded,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            stride=1,
        )
        return (
            patches.view(bsz, cdim, self.kernel_size, self.kernel_size, h * w)
            .permute(0, 1, 4, 2, 3)
            .reshape(bsz, cdim, h, w, self.kernel_size, self.kernel_size)
        )

    def _masked_spatial_weights(self, l2_dist: torch.Tensor) -> torch.Tensor:
        base_w = F.softmax(self.spatial_logits.reshape(-1), dim=0).view(
            1, 1, 1, self.kernel_size, self.kernel_size
        )
        if self.feature_mask_soft:
            mask = torch.sigmoid(
                self.feature_mask_temperature * (self.feature_mask_l2_threshold - l2_dist)
            )
            mask = torch.where(self.center_mask, torch.ones_like(mask), mask)
        else:
            mask = (l2_dist <= self.feature_mask_l2_threshold)
            mask = torch.logical_or(mask, self.center_mask).to(base_w.dtype)
        return base_w * mask

    def forward(self, features: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
        h, w = int(img_shape[0]), int(img_shape[1])
        batched = features.dim() == 3
        if not batched:
            features = features.unsqueeze(0)
        bsz, n, cdim = features.shape
        if cdim != self.in_dim:
            raise ValueError(f"Expected feature dim {self.in_dim}, got {cdim}")
        if n != h * w:
            raise ValueError(f"Expected N={h*w}, got {n}")

        feat_img = features.view(bsz, h, w, cdim).permute(0, 3, 1, 2).contiguous()
        patches = self._extract_patches(feat_img)
        center = feat_img.unsqueeze(-1).unsqueeze(-1)
        l2_dist = (patches - center).pow(2).sum(dim=1).clamp_min(self.eps).sqrt()
        masked_w = self._masked_spatial_weights(l2_dist)
        denom = masked_w.sum(dim=(-1, -2)).clamp_min(self.eps)
        neigh = (patches * masked_w.unsqueeze(1)).sum(dim=(-1, -2)) / denom.unsqueeze(1)

        feat_flat = features.reshape(bsz * n, cdim)
        neigh_flat = neigh.permute(0, 2, 3, 1).reshape(bsz * n, cdim)
        out = self.tp(feat_flat, neigh_flat)

        if self.use_residual:
            if self.residual_proj is None:
                out = out + self.residual_weight * feat_flat
            else:
                out = out + self.residual_weight * self.residual_proj(feat_flat)

        out = out.reshape(bsz, n, self.out_dim)
        if not batched:
            out = out.squeeze(0)
        return out


CosineMaskedEquivariantSpatialConv = FeatureDistanceMaskedEquivariantSpatialConv


class LocalIsoCrystalEncoder(nn.Module):
    """Local-iso encoder wrapper with crystal-family switch."""

    def __init__(
        self,
        crystal: str = "fcc",
        d6_convention: str = "z_axis",
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        crystal_key = str(crystal).lower()
        if crystal_key in {"fcc", "o", "cubic"}:
            self.group_name = "O"
            self.embedding = build_local_iso_fcc_embedding(dtype=dtype, device=device).eval()
            sym = build_fcc_syms_mtex(dtype=dtype, device=device)
        elif crystal_key in {"hcp", "d6"}:
            self.group_name = "D6"
            self.embedding = build_local_iso_hcp_embedding(
                d6_convention=d6_convention,
                dtype=dtype,
                device=device,
            ).eval()
            sym = build_hcp_syms_mtex(dtype=dtype, device=device)
        else:
            raise ValueError(f"crystal must be 'fcc' or 'hcp', got: {crystal}")

        self.irreps_a1 = self.embedding.irreps_a1
        self.irreps_full = self.embedding.irreps_full
        self.out_dim_a1 = int(self.irreps_a1.dim)
        self.out_dim_full = int(self.irreps_full.dim)

        sym = _normalize_quaternions(sym)
        self.register_buffer("sym_ops", sym, persistent=False)
        self.register_buffer("sym_ops_inv", _normalize_quaternions(_quat_conjugate(sym)), persistent=False)

    def _to_embedding_device(self, quats_passive: torch.Tensor) -> torch.Tensor:
        return quats_passive.to(
            device=self.embedding.group_mats.device,
            dtype=self.embedding.group_mats.dtype,
        )

    def forward_a1(self, quats_passive: torch.Tensor) -> torch.Tensor:
        q = self._to_embedding_device(quats_passive)
        return self.embedding.forward_irreps_passive(q, active_only=True)

    def forward_full(self, quats_passive: torch.Tensor) -> torch.Tensor:
        q = self._to_embedding_device(quats_passive)
        return self.embedding.forward_irreps_passive(q, active_only=False)


class CubochoricOptimizingLocalIsoDecoder(nn.Module):
    """Decode local-iso irreps features to passive quaternions by nearest-table seeding followed by optimization."""

    def __init__(
        self,
        encoder: LocalIsoCrystalEncoder,
        cubochoric_resolution: int = 1,
        method: str = "cubochoric",
        num_starts: int = 6,
        steps: int = 25,
        lr: float = 0.05,
        target_irreps: str = "full",
        max_table_rows: int | None = None,
        table_encode_chunk_size: int = 256,
        seed_search_query_chunk_size: int = 1024,
        seed_search_table_chunk_size: int = 8192,
        table_cache_dir: str | Path | None = "out/decoder_lookup_tables",
    ):
        super().__init__()
        self.encoder = encoder
        self.cubochoric_resolution = int(cubochoric_resolution)
        self.method = str(method)
        self.num_starts = max(1, int(num_starts))
        self.steps = max(0, int(steps))
        self.lr = float(lr)
        self.target_irreps = str(target_irreps).lower()
        self.max_table_rows = max_table_rows
        self.table_encode_chunk_size = max(1, int(table_encode_chunk_size))
        self.seed_search_query_chunk_size = max(1, int(seed_search_query_chunk_size))
        self.seed_search_table_chunk_size = max(1, int(seed_search_table_chunk_size))
        self.table_cache_dir = (
            Path(table_cache_dir).expanduser().resolve()
            if table_cache_dir is not None
            else None
        )

        if self.cubochoric_resolution < 1:
            raise ValueError(
                f"cubochoric_resolution must be >= 1, got {cubochoric_resolution}"
            )
        if self.target_irreps not in {"a1", "full"}:
            raise ValueError(f"target_irreps must be 'a1' or 'full', got {target_irreps}")
        if self.max_table_rows is not None:
            warnings.warn(
                "decoder_max_table_rows is set, so the decoder is not using the full lookup table.",
                stacklevel=2,
            )

        self.target_dim = int(
            self.encoder.out_dim_a1 if self.target_irreps == "a1" else self.encoder.out_dim_full
        )

        with torch.no_grad():
            cached = self._try_load_cached_table()
            if cached is not None:
                table_quats, table_feat = cached
            else:
                table_quats = _sample_fz_quaternions_passive(
                    group_name=self.encoder.group_name,
                    resolution=self.cubochoric_resolution,
                    method=self.method,
                    dtype=torch.float32,
                    device=self.encoder.embedding.group_mats.device,
                    max_rows=self.max_table_rows,
                )
                feat_chunks = []
                for start in range(0, int(table_quats.shape[0]), self.table_encode_chunk_size):
                    end = min(start + self.table_encode_chunk_size, int(table_quats.shape[0]))
                    feat_chunks.append(
                        self._encode_target_features(table_quats[start:end]).to(torch.float32)
                    )
                table_feat = torch.cat(feat_chunks, dim=0)
                self._save_cached_table(table_quats, table_feat)
            table_feat_norm = (table_feat * table_feat).sum(dim=-1)

        self.register_buffer("table_quats", table_quats, persistent=False)
        self.register_buffer("table_feat", table_feat, persistent=False)
        self.register_buffer("table_feat_norm", table_feat_norm, persistent=False)

    def _cache_metadata(self) -> dict[str, object]:
        return {
            "cache_version": 1,
            "group_name": str(self.encoder.group_name),
            "d6_convention": str(getattr(self.encoder.embedding, "d6_convention", "na")),
            "target_irreps": str(self.target_irreps),
            "target_dim": int(self.target_dim),
            "irreps_a1": str(self.encoder.irreps_a1),
            "irreps_full": str(self.encoder.irreps_full),
            "cubochoric_resolution": int(self.cubochoric_resolution),
            "method": str(self.method),
            "max_table_rows": None if self.max_table_rows is None else int(self.max_table_rows),
        }

    def _cache_paths(self) -> tuple[Path, Path, Path] | None:
        if self.table_cache_dir is None:
            return None
        meta = self._cache_metadata()
        meta_str = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(meta_str.encode("utf-8")).hexdigest()[:16]
        stem = (
            f"{self.encoder.group_name.lower()}_"
            f"{self.target_irreps}_{self.method}_r{self.cubochoric_resolution}_{digest}"
        )
        q_path = self.table_cache_dir / f"{stem}_quats.npy"
        f_path = self.table_cache_dir / f"{stem}_feat.npy"
        m_path = self.table_cache_dir / f"{stem}_meta.json"
        return q_path, f_path, m_path

    def _try_load_cached_table(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        paths = self._cache_paths()
        if paths is None:
            return None
        q_path, f_path, _ = paths
        if not q_path.exists() or not f_path.exists():
            return None

        try:
            quats_np = np.load(q_path)
            feat_np = np.load(f_path)
        except Exception:
            return None

        if quats_np.ndim != 2 or quats_np.shape[1] != 4:
            return None
        if feat_np.ndim != 2:
            return None
        if feat_np.shape[0] != quats_np.shape[0]:
            return None
        if feat_np.shape[1] != self.target_dim:
            return None

        device = self.encoder.embedding.group_mats.device
        table_quats = torch.from_numpy(quats_np.astype(np.float32, copy=False)).to(device=device)
        table_feat = torch.from_numpy(feat_np.astype(np.float32, copy=False)).to(device=device)
        table_quats = _normalize_quaternions(table_quats)
        return table_quats, table_feat

    def _save_cached_table(self, table_quats: torch.Tensor, table_feat: torch.Tensor) -> None:
        paths = self._cache_paths()
        if paths is None:
            return
        q_path, f_path, m_path = paths
        q_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(q_path, table_quats.detach().cpu().numpy().astype(np.float32, copy=False))
        np.save(f_path, table_feat.detach().cpu().numpy().astype(np.float32, copy=False))
        with open(m_path, "w") as f:
            json.dump(self._cache_metadata(), f, indent=2)

    def _nearest_seed_indices(self, feat_target: torch.Tensor) -> torch.Tensor:
        k = min(self.num_starts, int(self.table_feat.shape[0]))
        n = int(feat_target.shape[0])
        t = int(self.table_feat.shape[0])
        q_chunk = int(self.seed_search_query_chunk_size)
        t_chunk = int(self.seed_search_table_chunk_size)
        out_idx: list[torch.Tensor] = []

        for q_start in range(0, n, q_chunk):
            q_end = min(q_start + q_chunk, n)
            q = feat_target[q_start:q_end]
            qb = int(q.shape[0])
            qn = (q * q).sum(dim=-1, keepdim=True)
            best_dist = torch.full((qb, k), float("inf"), device=q.device, dtype=q.dtype)
            best_idx = torch.zeros((qb, k), device=q.device, dtype=torch.long)

            for t_start in range(0, t, t_chunk):
                t_end = min(t_start + t_chunk, t)
                tf = self.table_feat[t_start:t_end]
                tfn = self.table_feat_norm[t_start:t_end]
                dist = qn + tfn.unsqueeze(0) - 2.0 * (q @ tf.T)

                cand_k = min(k, int(dist.shape[1]))
                d_chunk, i_chunk = torch.topk(dist, k=cand_k, largest=False, dim=1)
                i_chunk = i_chunk + int(t_start)

                if cand_k < k:
                    pad_d = torch.full(
                        (qb, k - cand_k), float("inf"), device=q.device, dtype=q.dtype
                    )
                    pad_i = torch.zeros((qb, k - cand_k), device=q.device, dtype=torch.long)
                    d_chunk = torch.cat([d_chunk, pad_d], dim=1)
                    i_chunk = torch.cat([i_chunk, pad_i], dim=1)

                merged_d = torch.cat([best_dist, d_chunk], dim=1)
                merged_i = torch.cat([best_idx, i_chunk], dim=1)
                keep = torch.topk(merged_d, k=k, largest=False, dim=1).indices
                best_dist = torch.gather(merged_d, 1, keep)
                best_idx = torch.gather(merged_i, 1, keep)

            out_idx.append(best_idx)

        return torch.cat(out_idx, dim=0)

    def _encode_target_features(self, quats_passive: torch.Tensor) -> torch.Tensor:
        if self.target_irreps == "a1":
            return self.encoder.forward_a1(quats_passive)
        return self.encoder.forward_full(quats_passive)

    def forward(self, feat_target: torch.Tensor) -> torch.Tensor:
        feat_target = feat_target.detach().to(self.table_feat.device, dtype=torch.float32)
        bsz, cdim = feat_target.shape
        if cdim != self.target_dim:
            raise ValueError(f"Expected target dim {self.target_dim} for {self.target_irreps}, got {cdim}")

        with torch.no_grad():
            idx = self._nearest_seed_indices(feat_target)
            q0 = self.table_quats[idx]

        if self.steps == 0:
            return _normalize_quaternions(q0[:, 0, :])

        k = q0.shape[1]
        u = nn.Parameter(q0.clone())
        opt = torch.optim.Adam([u], lr=self.lr)

        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            q = _normalize_quaternions(u)
            q_flat = q.reshape(bsz * k, 4)
            feat_pred = self._encode_target_features(q_flat).reshape(bsz, k, cdim)
            loss_per = (feat_pred - feat_target.unsqueeze(1)).pow(2).mean(dim=-1)
            loss = loss_per.mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            q = _normalize_quaternions(u)
            q_flat = q.reshape(bsz * k, 4)
            feat_pred = self._encode_target_features(q_flat).reshape(bsz, k, cdim)
            loss_per = (feat_pred - feat_target.unsqueeze(1)).pow(2).mean(dim=-1)
            best_k = torch.argmin(loss_per, dim=1)
            batch_idx = torch.arange(bsz, device=feat_target.device)
            q_best = q[batch_idx, best_k]
        return _normalize_quaternions(q_best)


class PhaseEmbeddingGrid(nn.Module):
    """Learned phase embedding for positions inside one emitted HR patch."""

    def __init__(
        self,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        emb_dim: int = 32,
        patch_size: int | tuple[int, int] | list[int] | None = None,
    ):
        super().__init__()
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.patch_shape = _as_patch_shape(
            patch_size if patch_size is not None else self.upsample_factor,
            name="patch_size",
        )
        self.patch_size = (
            int(self.patch_shape[0])
            if self.patch_shape[0] == self.patch_shape[1]
            else self.patch_shape
        )
        self.num_phases = _num_patch_tokens(self.patch_shape)
        self.emb = nn.Embedding(self.num_phases, int(emb_dim))
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, phase_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(phase_ids)


class QuaternionBankClusterer(nn.Module):
    """Cluster a local odd-sized LR quaternion bank into orientation-connected components."""

    def __init__(
        self,
        sym_ops_quat: torch.Tensor,
        threshold_deg: float = 2.0,
        connectivity: int = 8,
        window_size: int = 5,
    ):
        super().__init__()
        if int(window_size) < 3 or int(window_size) % 2 == 0:
            raise ValueError(f"OCRP expects an odd window_size >= 3, got {window_size}")
        if int(connectivity) not in (4, 8):
            raise ValueError(f"OCRP currently expects 4- or 8-neighbor clustering, got {connectivity}")
        self.threshold_rad = float(np.deg2rad(float(threshold_deg)))
        self.connectivity = int(connectivity)
        self.window_size = int(window_size)
        self.num_nodes = int(self.window_size * self.window_size)

        sym_ops_mat = _left_mult_matrix_wxyz_batch(_normalize_quaternions(sym_ops_quat.detach().cpu()))
        self.register_buffer("sym_ops_mat", sym_ops_mat, persistent=False)

        edges: list[tuple[int, int]] = []
        for y in range(self.window_size):
            for x in range(self.window_size):
                idx = y * self.window_size + x
                if x + 1 < self.window_size:
                    edges.append((idx, idx + 1))
                if y + 1 < self.window_size:
                    edges.append((idx, idx + self.window_size))
                if self.connectivity == 8 and y + 1 < self.window_size:
                    if x + 1 < self.window_size:
                        edges.append((idx, idx + self.window_size + 1))
                    if x - 1 >= 0:
                        edges.append((idx, idx + self.window_size - 1))
        edge_idx_a = torch.tensor([a for a, _ in edges], dtype=torch.long)
        edge_idx_b = torch.tensor([b for _, b in edges], dtype=torch.long)
        self.register_buffer("edge_idx_a", edge_idx_a, persistent=False)
        self.register_buffer("edge_idx_b", edge_idx_b, persistent=False)

    def forward(self, bank_q: torch.Tensor) -> torch.Tensor:
        batched = bank_q.dim() == 4
        if not batched:
            bank_q = bank_q.unsqueeze(0)
        bsz, nwin, nnode, qdim = bank_q.shape
        if nnode != self.num_nodes or qdim != 4:
            raise ValueError(
                f"Expected bank_q shape (B,N,{self.num_nodes},4), got {tuple(bank_q.shape)}"
            )

        q = _normalize_quaternions(bank_q.to(dtype=torch.float32))
        flat = q.reshape(bsz * nwin, nnode, 4)

        idx_a = self.edge_idx_a.to(device=flat.device)
        idx_b = self.edge_idx_b.to(device=flat.device)
        q1 = flat.index_select(1, idx_a)
        q2 = flat.index_select(1, idx_b)
        sym_ops = self.sym_ops_mat.to(device=flat.device, dtype=flat.dtype)
        mis = _misorientation_angle_sym(q1, q2, sym_ops)
        keep_edges = mis <= self.threshold_rad

        num_windows = int(flat.shape[0])
        labels_flat = (
            torch.arange(nnode, device=flat.device, dtype=torch.long)
            .view(1, nnode)
            .expand(num_windows, -1)
            .clone()
        )
        edge_a = idx_a.view(1, -1).expand(num_windows, -1)
        edge_b = idx_b.view(1, -1).expand(num_windows, -1)
        inactive_label = torch.full_like(edge_a, fill_value=nnode)

        # Vectorized connected-components by repeated minimum-label propagation.
        for _ in range(nnode - 1):
            new_labels = labels_flat.clone()
            
            # First propagation: edge_a <- edge_b
            la = labels_flat.gather(1, edge_a)
            lb = labels_flat.gather(1, edge_b)
            edge_min = torch.minimum(la, lb)
            masked_min_a = torch.where(keep_edges, edge_min, inactive_label)
            new_labels.scatter_reduce_(1, edge_a, masked_min_a, reduce="amin", include_self=True)
            
            # Second propagation: edge_b <- updated edge_a
            # Recompute masked_min using UPDATED labels from first propagation
            la_updated = new_labels.gather(1, edge_a)  # UPDATED labels
            lb = labels_flat.gather(1, edge_b)
            edge_min_b = torch.minimum(la_updated, lb)
            masked_min_b = torch.where(keep_edges, edge_min_b, inactive_label)
            new_labels.scatter_reduce_(1, edge_b, masked_min_b, reduce="amin", include_self=True)
            
            if torch.equal(new_labels, labels_flat):
                break
            labels_flat = new_labels

        labels = labels_flat.view(bsz, nwin, nnode)
        if not batched:
            labels = labels.squeeze(0)
        return labels


class FeatureBankClusterer(nn.Module):
    """Cluster a local LR feature bank into connected components using pairwise L2 edges."""

    def __init__(
        self,
        threshold_l2: float = 0.035,
        connectivity: int = 8,
        window_size: int = 5,
        eps: float = 1e-8,
    ):
        super().__init__()
        if int(window_size) < 3 or int(window_size) % 2 == 0:
            raise ValueError(f"OCRP expects an odd window_size >= 3, got {window_size}")
        if int(connectivity) not in (4, 8):
            raise ValueError(f"OCRP currently expects 4- or 8-neighbor clustering, got {connectivity}")
        self.threshold_l2 = float(threshold_l2)
        self.connectivity = int(connectivity)
        self.window_size = int(window_size)
        self.num_nodes = int(self.window_size * self.window_size)
        self.eps = float(eps)

        edges: list[tuple[int, int]] = []
        for y in range(self.window_size):
            for x in range(self.window_size):
                idx = y * self.window_size + x
                if x + 1 < self.window_size:
                    edges.append((idx, idx + 1))
                if y + 1 < self.window_size:
                    edges.append((idx, idx + self.window_size))
                if self.connectivity == 8 and y + 1 < self.window_size:
                    if x + 1 < self.window_size:
                        edges.append((idx, idx + self.window_size + 1))
                    if x - 1 >= 0:
                        edges.append((idx, idx + self.window_size - 1))
        edge_idx_a = torch.tensor([a for a, _ in edges], dtype=torch.long)
        edge_idx_b = torch.tensor([b for _, b in edges], dtype=torch.long)
        self.register_buffer("edge_idx_a", edge_idx_a, persistent=False)
        self.register_buffer("edge_idx_b", edge_idx_b, persistent=False)

    def forward(self, bank_f: torch.Tensor) -> torch.Tensor:
        batched = bank_f.dim() == 4
        if not batched:
            bank_f = bank_f.unsqueeze(0)
        bsz, nwin, nnode, cdim = bank_f.shape
        if nnode != self.num_nodes:
            raise ValueError(
                f"Expected bank_f shape (B,N,{self.num_nodes},C), got {tuple(bank_f.shape)}"
            )
        if cdim < 1:
            raise ValueError(f"Expected bank_f feature dim >= 1, got {cdim}")

        flat = bank_f.to(dtype=torch.float32).reshape(bsz * nwin, nnode, cdim)
        idx_a = self.edge_idx_a.to(device=flat.device)
        idx_b = self.edge_idx_b.to(device=flat.device)
        f1 = flat.index_select(1, idx_a)
        f2 = flat.index_select(1, idx_b)
        l2 = (f1 - f2).pow(2).sum(dim=-1).clamp_min(self.eps).sqrt()
        keep_edges = l2 <= self.threshold_l2

        num_windows = int(flat.shape[0])
        labels_flat = (
            torch.arange(nnode, device=flat.device, dtype=torch.long)
            .view(1, nnode)
            .expand(num_windows, -1)
            .clone()
        )
        edge_a = idx_a.view(1, -1).expand(num_windows, -1)
        edge_b = idx_b.view(1, -1).expand(num_windows, -1)
        inactive_label = torch.full_like(edge_a, fill_value=nnode)

        for _ in range(nnode - 1):
            new_labels = labels_flat.clone()

            la = labels_flat.gather(1, edge_a)
            lb = labels_flat.gather(1, edge_b)
            edge_min = torch.minimum(la, lb)
            masked_min_a = torch.where(keep_edges, edge_min, inactive_label)
            new_labels.scatter_reduce_(1, edge_a, masked_min_a, reduce="amin", include_self=True)

            la_updated = new_labels.gather(1, edge_a)
            lb = labels_flat.gather(1, edge_b)
            edge_min_b = torch.minimum(la_updated, lb)
            masked_min_b = torch.where(keep_edges, edge_min_b, inactive_label)
            new_labels.scatter_reduce_(1, edge_b, masked_min_b, reduce="amin", include_self=True)

            if torch.equal(new_labels, labels_flat):
                break
            labels_flat = new_labels

        labels = labels_flat.view(bsz, nwin, nnode)
        if not batched:
            labels = labels.squeeze(0)
        return labels


class ClusterSlotBuilder(nn.Module):
    """Pack top-ranked clusters into deterministic slots and emit cheap metadata."""

    SLOT_TYPE_DIM = 10
    MAX_SLOTS = 10
    META_DIM = 25
    META_VALID = 0
    META_SLOT_TYPE_START = 1
    META_MASS = 7
    META_CENTROID_Y = 8
    META_CENTROID_X = 9
    META_SPATIAL_DISP = 10

    def __init__(self, kmax_slots: int = 10, window_size: int = 5):
        super().__init__()
        if int(kmax_slots) < 1 or int(kmax_slots) > self.MAX_SLOTS:
            raise ValueError(
                f"OCRP expects 1 <= kmax_slots <= {self.MAX_SLOTS}, got {kmax_slots}"
            )
        if int(window_size) < 3 or int(window_size) % 2 == 0:
            raise ValueError(f"OCRP expects an odd window_size >= 3, got {window_size}")
        self.kmax_slots = int(kmax_slots)
        self.window_size = int(window_size)
        self.num_nodes = int(self.window_size * self.window_size)

        coords = []
        den = float(max(1, self.window_size // 2))
        for y in range(self.window_size):
            for x in range(self.window_size):
                coords.append(((float(y) - den) / den, (float(x) - den) / den))
        coord_t = torch.tensor(coords, dtype=torch.float32)
        self.register_buffer("coords", coord_t, persistent=False)

    def forward(self, cluster_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        batched = cluster_ids.dim() == 3
        if not batched:
            cluster_ids = cluster_ids.unsqueeze(0)
        bsz, nwin, nnode = cluster_ids.shape
        if nnode != self.num_nodes:
            raise ValueError(f"Expected cluster_ids last dim {self.num_nodes}, got {nnode}")

        labels = cluster_ids.to(dtype=torch.long)
        label_onehot = F.one_hot(labels.clamp(min=0, max=self.num_nodes - 1), num_classes=self.num_nodes).to(torch.float32)
        label_mass = label_onehot.mean(dim=2)
        label_mask = label_onehot.permute(0, 1, 3, 2).contiguous()
        label_count = label_mask.sum(dim=-1)

        coords = self.coords.to(device=cluster_ids.device, dtype=torch.float32)
        coords_exp = coords.view(1, 1, 1, nnode, 2)
        label_centroid = (
            label_mask.unsqueeze(-1) * coords_exp
        ).sum(dim=-2) / label_count.clamp_min(1.0).unsqueeze(-1)
        # Prefer larger clusters first; use centrality only as a light tie-break.
        centrality = -(label_centroid.pow(2).sum(dim=-1))
        label_idx = torch.arange(self.num_nodes, device=cluster_ids.device, dtype=torch.float32).view(1, 1, -1)
        rank_score = label_mass + 1e-4 * centrality - 1e-6 * label_idx
        rank_score = rank_score.masked_fill(label_mass <= 0.0, float("-inf"))
        topk = self.kmax_slots
        top_score, top_label = torch.topk(rank_score, k=topk, dim=2)

        slot_cluster_label = torch.full(
            (bsz, nwin, self.kmax_slots),
            -1,
            device=cluster_ids.device,
            dtype=torch.long,
        )
        slot_cluster_label[..., :topk] = torch.where(
            torch.isfinite(top_score),
            top_label,
            torch.full_like(top_label, -1),
        )

        slot_valid = (slot_cluster_label >= 0).to(torch.float32)
        slot_mask = labels.unsqueeze(2) == slot_cluster_label.clamp_min(0).unsqueeze(-1)
        slot_mask = slot_mask & slot_valid.bool().unsqueeze(-1)

        coords_exp = coords.view(1, 1, 1, nnode, 2)
        mask_f = slot_mask.to(torch.float32)
        count = mask_f.sum(dim=-1)
        centroid = (mask_f.unsqueeze(-1) * coords_exp).sum(dim=-2) / count.clamp_min(1.0).unsqueeze(-1)
        disp = (
            (((coords_exp - centroid.unsqueeze(-2)) ** 2).sum(dim=-1) * mask_f).sum(dim=-1)
            / count.clamp_min(1.0)
        )
        mass = count / float(self.num_nodes)

        slot_meta = torch.zeros(
            (bsz, nwin, self.kmax_slots, self.META_DIM),
            device=cluster_ids.device,
            dtype=torch.float32,
        )
        slot_type_eye = torch.eye(self.SLOT_TYPE_DIM, device=cluster_ids.device, dtype=torch.float32)
        for slot_idx in range(self.kmax_slots):
            valid = slot_valid[..., slot_idx].bool().unsqueeze(-1)
            slot_meta[..., slot_idx, self.META_SLOT_TYPE_START : self.META_SLOT_TYPE_START + self.SLOT_TYPE_DIM] = torch.where(
                valid,
                slot_type_eye[slot_idx].view(1, 1, -1),
                torch.zeros((1, 1, self.SLOT_TYPE_DIM), device=cluster_ids.device, dtype=torch.float32),
            )
        slot_meta[..., self.META_VALID] = slot_valid
        slot_meta[..., self.META_MASS] = mass * slot_valid
        slot_meta[..., self.META_CENTROID_Y] = centroid[..., 0] * slot_valid
        slot_meta[..., self.META_CENTROID_X] = centroid[..., 1] * slot_valid
        slot_meta[..., self.META_SPATIAL_DISP] = disp * slot_valid

        out = {
            "slot_mask": slot_mask,
            "slot_valid": slot_valid,
            "slot_meta": slot_meta,
            "slot_cluster_label": slot_cluster_label,
        }
        if not batched:
            out = {
                key: (val.squeeze(0) if isinstance(val, torch.Tensor) and val.shape[0] == 1 else val)
                for key, val in out.items()
            }
        return out


class InvariantSlotSummary(nn.Module):
    """Compute cheap invariant summaries from irrep-valued features."""

    def __init__(self, irreps_feat: Irreps | str):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.block_slices = _irrep_block_slices(self.irreps_feat)
        self.out_dim = int(len(self.block_slices))

    def summarize(self, feat: torch.Tensor) -> torch.Tensor:
        outs: list[torch.Tensor] = []
        for start, end in self.block_slices:
            outs.append(feat[..., start:end].norm(dim=-1, keepdim=True))
        return torch.cat(outs, dim=-1)

    def pair_stats(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        outs: list[torch.Tensor] = []
        for start, end in self.block_slices:
            aa = a[..., start:end]
            bb = b[..., start:end]
            outs.append(aa.norm(dim=-1, keepdim=True))
            outs.append(bb.norm(dim=-1, keepdim=True))
            outs.append((aa * bb).sum(dim=-1, keepdim=True))
        return torch.cat(outs, dim=-1)


class LearnedWeightedSlotContextBuilder(nn.Module):
    """Learn scalar mixing weights over masked slot members to build the anchor."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        meta_dim: int,
        window_size: int = 5,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.summary = InvariantSlotSummary(self.irreps_feat)
        self.meta_dim = int(meta_dim)
        self.window_size = int(window_size)
        self.num_nodes = int(self.window_size * self.window_size)
        den = float(max(1, self.window_size // 2))
        coords = []
        for y in range(self.window_size):
            for x in range(self.window_size):
                coords.append(((float(y) - den) / den, (float(x) - den) / den))
        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32), persistent=False)
        member_in_dim = int(self.summary.out_dim) + self.meta_dim + 2
        self.anchor_mixer = nn.Linear(member_in_dim, 1)
        # Start near uniform, but with a tiny symmetry break so the mixer can learn away from mean.
        nn.init.normal_(self.anchor_mixer.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.anchor_mixer.bias)

    def forward(
        self,
        bank_f: torch.Tensor,
        slot_mask: torch.Tensor,
        slot_meta: torch.Tensor,
        return_alpha: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batched = bank_f.dim() == 4
        if not batched:
            bank_f = bank_f.unsqueeze(0)
            slot_mask = slot_mask.unsqueeze(0)
            slot_meta = slot_meta.unsqueeze(0)
        _, _, nnode, _ = bank_f.shape
        if slot_mask.shape[-1] != nnode:
            raise ValueError("slot_mask and bank_f disagree on bank size")
        if slot_meta.shape[:3] != slot_mask.shape[:3]:
            raise ValueError("slot_meta and slot_mask must agree on (B, N, K)")

        bsz, nwin, kmax, _ = slot_meta.shape
        member_summary = self.summary.summarize(bank_f).unsqueeze(2).expand(-1, -1, kmax, -1, -1)
        coord_feat = self.coords.to(device=bank_f.device, dtype=slot_meta.dtype)
        coord_feat = coord_feat.view(1, 1, 1, self.num_nodes, 2).expand(bsz, nwin, kmax, -1, -1)
        member_meta = slot_meta.unsqueeze(-2).expand(-1, -1, -1, self.num_nodes, -1)
        logits = self.anchor_mixer(torch.cat([member_summary, member_meta, coord_feat], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~slot_mask.bool(), -1e4)

        alpha = torch.softmax(logits, dim=-1)
        alpha = alpha * slot_mask.to(dtype=alpha.dtype)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_ctx = torch.einsum("bnkm,bnmc->bnkc", alpha.to(dtype=bank_f.dtype), bank_f)
        has_member = slot_mask.any(dim=-1, keepdim=True)
        slot_ctx = torch.where(has_member, slot_ctx, torch.zeros_like(slot_ctx))

        if not batched:
            slot_ctx = slot_ctx.squeeze(0)
            alpha = alpha.squeeze(0)
        return slot_ctx, (alpha.to(dtype=bank_f.dtype) if return_alpha else None)


MeanSlotContextBuilder = LearnedWeightedSlotContextBuilder
MedoidSlotContextBuilder = LearnedWeightedSlotContextBuilder


class WithinSlotInvariantPool(nn.Module):
    """Token-conditioned equivariant within-slot pooling using scalar invariant weights only."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        meta_dim: int,
        phase_dim: int,
        window_size: int = 5,
        patch_shape: int | tuple[int, int] | list[int] | None = None,
        hidden_dim: int = 96,
        chunk_size: int = 512,
        token_conditioned_member_bias: bool = True,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.summary = InvariantSlotSummary(self.irreps_feat)
        self.phase_dim = int(phase_dim)
        self.window_size = int(window_size)
        self.num_nodes = int(self.window_size * self.window_size)
        self.chunk_size = int(chunk_size)
        self.patch_shape = (
            _as_patch_shape(patch_shape, name="patch_shape")
            if patch_shape is not None
            else None
        )
        self.token_conditioned_member_bias = bool(token_conditioned_member_bias)
        den = float(max(1, self.window_size // 2))
        coords = []
        for y in range(self.window_size):
            for x in range(self.window_size):
                coords.append(((float(y) - den) / den, (float(x) - den) / den))
        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32), persistent=False)
        pair_dim = int(self.summary.out_dim * 3)
        member_in_dim = pair_dim + int(meta_dim) + 2
        self.token_geom_dim = 5
        phase_in_dim = int(meta_dim) + self.phase_dim
        if self.token_conditioned_member_bias:
            phase_in_dim += self.token_geom_dim
        self.member_key = nn.Sequential(
            nn.Linear(member_in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.member_bias = (
            None
            if self.token_conditioned_member_bias
            else nn.Linear(int(hidden_dim), 1)
        )
        self.phase_query = nn.Sequential(
            nn.Linear(phase_in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.token_bias_ctrl = (
            nn.Sequential(
                nn.Linear(phase_in_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 6),
            )
            if self.token_conditioned_member_bias
            else None
        )
        self.logit_scale = float(max(1, hidden_dim)) ** -0.5

    @staticmethod
    def _token_geom_features(token_coords: torch.Tensor) -> torch.Tensor:
        if token_coords.dim() != 2 or int(token_coords.shape[-1]) != 2:
            raise ValueError(
                "token_coords must have shape (T,2), "
                f"got {tuple(token_coords.shape)}"
            )
        ty = token_coords[:, :1]
        tx = token_coords[:, 1:2]
        return torch.cat([ty, tx, ty.square(), tx.square(), ty * tx], dim=-1)

    def forward(
        self,
        slot_anchor_ctx: torch.Tensor,
        slot_meta: torch.Tensor,
        bank_f: torch.Tensor,
        slot_mask: torch.Tensor,
        phase_grid: torch.Tensor,
        return_alpha: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, nwin, kmax, cdim = slot_anchor_ctx.shape
        if bank_f.shape[:2] != (bsz, nwin):
            raise ValueError("bank_f and slot_anchor_ctx must agree on (B, N)")
        if slot_mask.shape[:3] != (bsz, nwin, kmax):
            raise ValueError("slot_mask and slot_anchor_ctx must agree on (B, N, K)")
        if phase_grid.dim() == 2:
            phase = phase_grid.view(1, 1, phase_grid.shape[0], phase_grid.shape[1]).expand(bsz, nwin, -1, -1)
        elif phase_grid.dim() == 4:
            phase = phase_grid
        else:
            raise ValueError(
                "phase_grid must have shape (T,D_phase) or (B,N,T,D_phase), "
                f"got {tuple(phase_grid.shape)}"
            )
        patch_tokens = int(phase.shape[2])

        coords = self.coords.to(device=bank_f.device, dtype=slot_meta.dtype)
        token_coords = None
        token_geom = None
        if self.token_conditioned_member_bias:
            patch_shape = self.patch_shape if self.patch_shape is not None else patch_tokens
            if self.patch_shape is not None and patch_tokens != _num_patch_tokens(self.patch_shape):
                raise ValueError(
                    f"Expected phase patch tokens {_num_patch_tokens(self.patch_shape)}, got {patch_tokens}"
                )
            token_coords = _build_patch_token_coords(
                patch_shape,
                device=bank_f.device,
                dtype=slot_meta.dtype,
            )
            token_geom = self._token_geom_features(token_coords)
        anchor_flat = slot_anchor_ctx.reshape(bsz * nwin, kmax, cdim)
        meta_flat = slot_meta.reshape(bsz * nwin, kmax, slot_meta.shape[-1])
        bank_flat = bank_f.reshape(bsz * nwin, self.num_nodes, cdim)
        mask_flat = slot_mask.reshape(bsz * nwin, kmax, self.num_nodes)
        phase_flat = phase.reshape(bsz * nwin, patch_tokens, self.phase_dim)

        pooled_flat = torch.zeros(
            (bsz * nwin, kmax, patch_tokens, cdim),
            device=bank_f.device,
            dtype=bank_f.dtype,
        )
        alpha_flat = (
            torch.zeros(
                (bsz * nwin, kmax, patch_tokens, self.num_nodes),
                device=bank_f.device,
                dtype=bank_f.dtype,
            )
            if return_alpha
            else None
        )

        for start in range(0, anchor_flat.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, anchor_flat.shape[0])
            anchor_chunk = anchor_flat[start:end]
            meta_chunk = meta_flat[start:end]
            bank_chunk = bank_flat[start:end]
            mask_chunk = mask_flat[start:end]
            phase_chunk = phase_flat[start:end]

            bank_exp = bank_chunk.unsqueeze(1).expand(-1, kmax, -1, -1)
            anchor_exp = anchor_chunk.unsqueeze(2).expand(-1, kmax, self.num_nodes, -1)
            pair = self.summary.pair_stats(anchor_exp, bank_exp)
            coord_feat = coords.view(1, 1, self.num_nodes, 2).expand(end - start, kmax, -1, -1)
            member_meta = meta_chunk.unsqueeze(2).expand(-1, kmax, self.num_nodes, -1)
            member_key = self.member_key(torch.cat([pair, member_meta, coord_feat], dim=-1))

            phase_meta = meta_chunk.unsqueeze(2).expand(-1, kmax, patch_tokens, -1)
            phase_input_parts = [
                phase_meta,
                phase_chunk.unsqueeze(1).expand(-1, kmax, -1, -1),
            ]
            if self.token_conditioned_member_bias:
                assert token_geom is not None and token_coords is not None
                token_geom_chunk = token_geom.view(1, 1, patch_tokens, self.token_geom_dim).expand(
                    end - start,
                    kmax,
                    -1,
                    -1,
                )
                phase_input_parts.append(token_geom_chunk)
            phase_input = torch.cat(phase_input_parts, dim=-1)
            phase_query = self.phase_query(phase_input)

            if self.token_conditioned_member_bias:
                assert self.token_bias_ctrl is not None
                bias_ctrl = self.token_bias_ctrl(phase_input)
                member_coord = coords.view(1, 1, 1, self.num_nodes, 2)
                token_coord = token_coords.view(1, 1, patch_tokens, 1, 2)
                rel_y = member_coord[..., 0] - token_coord[..., 0]
                rel_x = member_coord[..., 1] - token_coord[..., 1]
                rel_yy = rel_y.square()
                rel_xx = rel_x.square()
                rel_yx = rel_y * rel_x
                bias_chunk = (
                    bias_ctrl[..., 0].unsqueeze(-1)
                    + bias_ctrl[..., 1].unsqueeze(-1) * rel_y
                    + bias_ctrl[..., 2].unsqueeze(-1) * rel_x
                    + bias_ctrl[..., 3].unsqueeze(-1) * rel_yy
                    + bias_ctrl[..., 4].unsqueeze(-1) * rel_xx
                    + bias_ctrl[..., 5].unsqueeze(-1) * rel_yx
                )
            else:
                assert self.member_bias is not None
                bias_chunk = self.member_bias(member_key).squeeze(-1).unsqueeze(2)

            logits = bias_chunk + self.logit_scale * torch.einsum(
                "bkth,bkmh->bktm",
                phase_query,
                member_key,
            )
            logits = logits.masked_fill(~mask_chunk.unsqueeze(2), -1e4)
            alpha_chunk = torch.softmax(logits, dim=-1)
            alpha_chunk = alpha_chunk * mask_chunk.unsqueeze(2).to(dtype=alpha_chunk.dtype)
            alpha_chunk = alpha_chunk / alpha_chunk.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            pooled_chunk = torch.einsum("bktm,bmc->bktc", alpha_chunk, bank_chunk)
            has_member = mask_chunk.any(dim=-1, keepdim=True).unsqueeze(-1)
            pooled_chunk = torch.where(
                has_member,
                pooled_chunk,
                anchor_chunk.unsqueeze(2).expand(-1, -1, patch_tokens, -1),
            )
            pooled_flat[start:end] = pooled_chunk
            if alpha_flat is not None:
                alpha_flat[start:end] = alpha_chunk.to(dtype=bank_f.dtype)

        pooled = pooled_flat.view(bsz, nwin, kmax, patch_tokens, cdim)
        alpha = (
            alpha_flat.view(bsz, nwin, kmax, patch_tokens, self.num_nodes)
            if alpha_flat is not None
            else None
        )
        return pooled, alpha


class PatchSlotRouter(nn.Module):
    """Predict joint patchwise slot ownership logits for the full HR patch."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        meta_dim: int,
        kmax_slots: int = 10,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        patch_size: int | tuple[int, int] | list[int] | None = None,
        phase_dim: int = 32,
        hidden_dim: int = 128,
        conv_hidden_dim: int = 64,
        chunk_size: int = 512,
        slot_mass_power: float = 0.25,
        uniform_slot_mix: float = 0.75,
        use_slot_type_meta: bool = True,
        use_raw_token_ctx: bool = False,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.kmax_slots = int(kmax_slots)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.patch_shape = _as_patch_shape(
            patch_size if patch_size is not None else self.upsample_factor,
            name="patch_size",
        )
        self.patch_size = (
            int(self.patch_shape[0])
            if self.patch_shape[0] == self.patch_shape[1]
            else self.patch_shape
        )
        self.patch_tokens = _num_patch_tokens(self.patch_shape)
        self.phase_dim = int(phase_dim)
        self.summary = InvariantSlotSummary(self.irreps_feat)
        self.slot_hidden_dim = int(hidden_dim)
        self.chunk_size = max(1, int(chunk_size))
        self.slot_mass_power = float(slot_mass_power)
        self.uniform_slot_mix = float(uniform_slot_mix)
        self.use_slot_type_meta = bool(use_slot_type_meta)
        self.use_raw_token_ctx = bool(use_raw_token_ctx)
        if self.slot_mass_power < 0.0:
            raise ValueError(f"slot_mass_power must be >= 0, got {slot_mass_power}")
        if not (0.0 <= self.uniform_slot_mix <= 1.0):
            raise ValueError(f"uniform_slot_mix must be in [0,1], got {uniform_slot_mix}")

        in_slot = int(self.summary.out_dim) + int(meta_dim)
        self.slot_proj = nn.Sequential(
            nn.Linear(in_slot, self.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.slot_hidden_dim, self.slot_hidden_dim),
        )
        raw_in_slot = int(self.irreps_feat.dim) + int(meta_dim)
        self.slot_proj_raw = (
            nn.Sequential(
                nn.Linear(raw_in_slot, self.slot_hidden_dim),
                nn.GELU(),
                nn.Linear(self.slot_hidden_dim, self.slot_hidden_dim),
            )
            if self.use_raw_token_ctx
            else None
        )
        self.phase_proj = nn.Sequential(
            nn.Linear(self.phase_dim, self.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.slot_hidden_dim, self.slot_hidden_dim),
        )
        self.base_logit = nn.Sequential(
            nn.Linear(2 * self.slot_hidden_dim, self.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.slot_hidden_dim, 1),
        )

    def _router_weight_context(
        self,
        slot_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_valid = slot_meta[
            ..., ClusterSlotBuilder.META_VALID : ClusterSlotBuilder.META_VALID + 1
        ]
        raw_mass = slot_meta[
            ..., ClusterSlotBuilder.META_MASS : ClusterSlotBuilder.META_MASS + 1
        ].clamp_min(0.0)
        tempered_mass = torch.where(
            raw_mass > 0.0,
            raw_mass.clamp_min(1e-6).pow(self.slot_mass_power),
            torch.zeros_like(raw_mass),
        )
        slot_meta_router = slot_meta.clone()
        slot_meta_router[
            ..., ClusterSlotBuilder.META_MASS : ClusterSlotBuilder.META_MASS + 1
        ] = tempered_mass * slot_valid
        if not self.use_slot_type_meta:
            slot_meta_router[
                ...,
                ClusterSlotBuilder.META_SLOT_TYPE_START : (
                    ClusterSlotBuilder.META_SLOT_TYPE_START + ClusterSlotBuilder.SLOT_TYPE_DIM
                ),
            ] = 0.0
        return slot_valid, slot_meta_router

    def forward(
        self,
        slot_ctx: torch.Tensor,
        slot_meta: torch.Tensor,
        phase_grid: torch.Tensor,
        weak_parent_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if slot_ctx.dim() == 4:
            bsz, nwin, kmax, cdim = slot_ctx.shape
            token_ctx = None
        elif slot_ctx.dim() == 5:
            bsz, nwin, kmax, patch_tokens_ctx, cdim = slot_ctx.shape
            token_ctx = slot_ctx
        else:
            raise ValueError(
                "slot_ctx must have shape (B,N,K,C) or (B,N,K,T,C), "
                f"got {tuple(slot_ctx.shape)}"
            )
        if kmax != self.kmax_slots:
            raise ValueError(f"Expected kmax_slots={self.kmax_slots}, got {kmax}")

        slot_valid, slot_meta_router = self._router_weight_context(slot_meta)
        nflat = int(bsz * nwin)
        slot_meta_flat = slot_meta_router.reshape(nflat, kmax, slot_meta_router.shape[-1])

        phase_flat: torch.Tensor | None = None
        phase_shared: torch.Tensor | None = None
        if phase_grid.dim() == 2:
            if int(phase_grid.shape[0]) != self.patch_tokens:
                raise ValueError(
                    f"Expected phase patch tokens {self.patch_tokens}, got {int(phase_grid.shape[0])}"
                )
            phase_shared = self.phase_proj(phase_grid)
        elif phase_grid.dim() == 4:
            if int(phase_grid.shape[2]) != self.patch_tokens:
                raise ValueError(
                    f"Expected phase patch tokens {self.patch_tokens}, got {int(phase_grid.shape[2])}"
                )
            phase_flat = self.phase_proj(phase_grid).reshape(nflat, self.patch_tokens, self.slot_hidden_dim)
        else:
            raise ValueError(
                "phase_grid must have shape (T,D_phase) or (B,N,T,D_phase), "
                f"got {tuple(phase_grid.shape)}"
            )

        logits_flat = torch.empty(
            (nflat, self.patch_tokens, self.kmax_slots),
            device=slot_meta.device,
            dtype=slot_meta.dtype,
        )

        if token_ctx is None:
            slot_inv_flat = self.summary.summarize(slot_ctx).reshape(nflat, kmax, -1)
            for start in range(0, nflat, self.chunk_size):
                end = min(start + self.chunk_size, nflat)
                slot_inv_chunk = slot_inv_flat[start:end]
                slot_meta_chunk = slot_meta_flat[start:end]
                slot_desc_chunk = self.slot_proj(torch.cat([slot_inv_chunk, slot_meta_chunk], dim=-1))
                slot_desc_exp = slot_desc_chunk.unsqueeze(2).expand(-1, -1, self.patch_tokens, -1)
                phase_chunk = phase_shared.unsqueeze(0).expand(end - start, -1, -1) if phase_shared is not None else phase_flat[start:end]
                phase_exp = phase_chunk.unsqueeze(1).expand(-1, self.kmax_slots, -1, -1)
                base_chunk = self.base_logit(torch.cat([slot_desc_exp, phase_exp], dim=-1)).squeeze(-1)
                logits_flat[start:end] = base_chunk.permute(0, 2, 1)
        else:
            if patch_tokens_ctx != self.patch_tokens:
                raise ValueError(
                    f"Expected slot_ctx token dim {self.patch_tokens}, got {patch_tokens_ctx}"
                )
            if self.use_raw_token_ctx:
                slot_inv_flat = token_ctx.reshape(nflat, kmax, self.patch_tokens, -1)
                proj = self.slot_proj_raw
                if proj is None:
                    raise ValueError("slot_proj_raw is required when use_raw_token_ctx=True")
            else:
                slot_inv_flat = self.summary.summarize(token_ctx).reshape(nflat, kmax, self.patch_tokens, -1)
                proj = self.slot_proj
            for start in range(0, nflat, self.chunk_size):
                end = min(start + self.chunk_size, nflat)
                slot_inv_chunk = slot_inv_flat[start:end]
                slot_meta_chunk = slot_meta_flat[start:end]
                meta_exp = slot_meta_chunk.unsqueeze(2).expand(-1, -1, self.patch_tokens, -1)
                slot_desc_chunk = proj(torch.cat([slot_inv_chunk, meta_exp], dim=-1))
                phase_chunk = phase_shared.unsqueeze(0).expand(end - start, -1, -1) if phase_shared is not None else phase_flat[start:end]
                phase_exp = phase_chunk.unsqueeze(1).expand(-1, self.kmax_slots, -1, -1)
                base_chunk = self.base_logit(torch.cat([slot_desc_chunk, phase_exp], dim=-1)).squeeze(-1)
                logits_flat[start:end] = base_chunk.permute(0, 2, 1)

        logits = logits_flat.view(bsz, nwin, self.patch_tokens, self.kmax_slots)

        invalid = slot_valid.squeeze(-1) <= 0.0
        logits = logits.masked_fill(invalid.unsqueeze(2), -1e4)
        return logits


class EquivariantSlotPatchQueryAnchor(nn.Module):
    """Build a weak equivariant patch query from slot context plus token metadata."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        meta_dim: int,
        phase_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.block_slices = _irrep_block_slices(self.irreps_feat)
        self.num_blocks = len(self.block_slices)
        in_dim = int(meta_dim) + int(phase_dim)
        self.slot_scale_net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.num_blocks),
        )

    def forward(
        self,
        slot_ctx: torch.Tensor,
        slot_meta: torch.Tensor,
        phase_feat: torch.Tensor,
        weak_parent_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctrl = torch.cat([slot_meta, phase_feat], dim=-1)
        slot_scale = 1.0 + 0.5 * torch.tanh(self.slot_scale_net(ctrl))

        out = torch.zeros_like(slot_ctx)
        for j, (start, end) in enumerate(self.block_slices):
            out[..., start:end] = slot_scale[..., j : j + 1] * slot_ctx[..., start:end]
        return out


class SharedTPPatchProposalHead(nn.Module):
    """Produce one HR patch proposal per slot using TP-based equivariant mixing."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        meta_dim: int,
        phase_dim: int,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        patch_size: int | tuple[int, int] | list[int] | None = None,
        hidden_dim: int = 128,
        chunk_size: int = 128,
        token_chunk_size: int | None = None,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feature_dim = int(self.irreps_feat.dim)
        self.block_slices = _irrep_block_slices(self.irreps_feat)
        self.num_blocks = len(self.block_slices)
        self.phase_dim = int(phase_dim)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.patch_shape = _as_patch_shape(
            patch_size if patch_size is not None else self.upsample_factor,
            name="patch_size",
        )
        self.patch_size = (
            int(self.patch_shape[0])
            if self.patch_shape[0] == self.patch_shape[1]
            else self.patch_shape
        )
        self.patch_tokens = _num_patch_tokens(self.patch_shape)
        self.chunk_size = max(1, int(chunk_size))
        self.token_chunk_size = (
            self.patch_tokens
            if token_chunk_size is None
            else max(1, min(int(token_chunk_size), self.patch_tokens))
        )

        self.query_anchor = EquivariantSlotPatchQueryAnchor(
            self.irreps_feat,
            meta_dim=meta_dim,
            phase_dim=self.phase_dim,
            hidden_dim=max(32, hidden_dim // 2),
        )
        self.lin_query = IrrepsLinear(self.irreps_feat, self.irreps_feat)
        self.lin_ctx = IrrepsLinear(self.irreps_feat, self.irreps_feat)
        self.tp = FullyConnectedTensorProduct(
            self.irreps_feat,
            self.irreps_feat,
            self.irreps_feat,
            shared_weights=True,
        )
        self.ctrl_net = nn.Sequential(
            nn.Linear(int(meta_dim) + self.phase_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 2 * self.num_blocks),
        )

    def forward(
        self,
        slot_ctx: torch.Tensor,
        slot_meta: torch.Tensor,
        phase_grid: torch.Tensor,
        weak_parent_prior: torch.Tensor | None = None,
        slot_anchor_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if slot_ctx.dim() == 4:
            bsz, nwin, kmax, cdim = slot_ctx.shape
            token_ctx = None
        elif slot_ctx.dim() == 5:
            bsz, nwin, kmax, patch_tokens_ctx, cdim = slot_ctx.shape
            token_ctx = slot_ctx
        else:
            raise ValueError(
                "slot_ctx must have shape (B,N,K,C) or (B,N,K,T,C), "
                f"got {tuple(slot_ctx.shape)}"
            )
        if cdim != self.feature_dim:
            raise ValueError(f"Expected slot_ctx dim {self.feature_dim}, got {cdim}")
        phase_flat: torch.Tensor | None = None
        phase_shared: torch.Tensor | None = None
        if phase_grid.dim() == 2:
            if int(phase_grid.shape[0]) != self.patch_tokens:
                raise ValueError(
                    f"Expected phase patch tokens {self.patch_tokens}, got {int(phase_grid.shape[0])}"
                )
            phase_shared = phase_grid
        elif phase_grid.dim() == 4:
            if int(phase_grid.shape[2]) != self.patch_tokens:
                raise ValueError(
                    f"Expected phase patch tokens {self.patch_tokens}, got {int(phase_grid.shape[2])}"
                )
            phase_flat = phase_grid.reshape(bsz * nwin, self.patch_tokens, self.phase_dim)
        else:
            raise ValueError(
                "phase_grid must have shape (T,D_phase) or (B,N,T,D_phase), "
                f"got {tuple(phase_grid.shape)}"
            )

        if slot_anchor_ctx is None:
            slot_anchor_ctx = slot_ctx if token_ctx is None else token_ctx.mean(dim=3)
        if slot_anchor_ctx.dim() != 4:
            raise ValueError(
                "slot_anchor_ctx must have shape (B,N,K,C), "
                f"got {tuple(slot_anchor_ctx.shape)}"
            )
        if slot_anchor_ctx.shape != (bsz, nwin, kmax, cdim):
            raise ValueError(
                f"slot_anchor_ctx must have shape {(bsz, nwin, kmax, cdim)}, "
                f"got {tuple(slot_anchor_ctx.shape)}"
            )

        nflat = int(bsz * nwin)
        slot_meta_flat = slot_meta.reshape(nflat, kmax, slot_meta.shape[-1])
        slot_anchor_flat = slot_anchor_ctx.reshape(nflat, kmax, cdim)
        if token_ctx is None:
            ctx_flat = slot_ctx.reshape(nflat, kmax, cdim)
        else:
            if patch_tokens_ctx != self.patch_tokens:
                raise ValueError(
                    f"Expected slot_ctx token dim {self.patch_tokens}, got {patch_tokens_ctx}"
                )
            ctx_flat = token_ctx.reshape(nflat, kmax, self.patch_tokens, cdim)

        out_flat = torch.empty(
            (nflat, kmax, self.patch_tokens, self.feature_dim),
            device=slot_ctx.device,
            dtype=slot_ctx.dtype,
        )

        for start in range(0, nflat, self.chunk_size):
            end = min(start + self.chunk_size, nflat)
            chunk = end - start
            slot_meta_chunk = slot_meta_flat[start:end]
            if phase_shared is not None:
                phase_chunk = phase_shared.unsqueeze(0).expand(chunk, -1, -1)
            else:
                phase_chunk = phase_flat[start:end]
            anchor_base = slot_anchor_flat[start:end]
            meta_base = slot_meta_chunk
            if token_ctx is None:
                ctx_base = ctx_flat[start:end]
            else:
                ctx_base = ctx_flat[start:end]

            out_chunk = torch.empty(
                (chunk, kmax, self.patch_tokens, self.feature_dim),
                device=slot_ctx.device,
                dtype=slot_ctx.dtype,
            )
            for token_start in range(0, self.patch_tokens, self.token_chunk_size):
                token_end = min(token_start + self.token_chunk_size, self.patch_tokens)
                token_count = token_end - token_start
                anchor_chunk = anchor_base.unsqueeze(2).expand(-1, -1, token_count, -1)
                meta_chunk = meta_base.unsqueeze(2).expand(-1, -1, token_count, -1)
                phase_chunk_slice = phase_chunk[:, token_start:token_end, :]
                phase_exp = phase_chunk_slice.unsqueeze(1).expand(-1, kmax, -1, -1)
                if token_ctx is None:
                    ctx_chunk = ctx_base.unsqueeze(2).expand(-1, -1, token_count, -1)
                else:
                    ctx_chunk = ctx_base[:, :, token_start:token_end, :]

                query_chunk = self.query_anchor(
                    slot_ctx=anchor_chunk,
                    slot_meta=meta_chunk,
                    phase_feat=phase_exp,
                )

                q_flat = self.lin_query(query_chunk.reshape(-1, self.feature_dim))
                c_flat = self.lin_ctx(ctx_chunk.reshape(-1, self.feature_dim))
                tp_flat = self.tp(q_flat, c_flat)
                tp_out = tp_flat.reshape(chunk, kmax, token_count, self.feature_dim)

                coeffs = self.ctrl_net(torch.cat([meta_chunk, phase_exp], dim=-1))
                alpha, beta = coeffs.chunk(2, dim=-1)
                alpha = 1.0 + 0.5 * torch.tanh(alpha)
                beta = 0.5 * torch.tanh(beta)

                out_token = torch.empty_like(tp_out)
                for j, (block_start, block_end) in enumerate(self.block_slices):
                    out_token[..., block_start:block_end] = (
                        alpha[..., j : j + 1] * tp_out[..., block_start:block_end]
                    )
                    out_token[..., block_start:block_end] = (
                        out_token[..., block_start:block_end]
                        + beta[..., j : j + 1] * query_chunk[..., block_start:block_end]
                    )
                out_chunk[:, :, token_start:token_end, :] = out_token
            out_flat[start:end] = out_chunk

        return out_flat.view(bsz, nwin, kmax, self.patch_tokens, self.feature_dim)


class OCRPPatchUpsampler(nn.Module):
    """Orientation-Cluster Routed Patch (OCRP) upsampler."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        sym_ops_quat: torch.Tensor,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        window_size: int = 5,
        kmax_slots: int = 10,
        cluster_threshold_deg: float = 2.0,
        cluster_feature_l2_threshold: float | None = None,
        cluster_connectivity: int = 8,
        phase_dim: int = 32,
        router_hidden_dim: int = 128,
        router_conv_hidden_dim: int = 64,
        router_slot_mass_power: float = 0.25,
        router_uniform_slot_mix: float = 0.75,
        router_use_slot_type_meta: bool = True,
        router_geom_logit_bias: float = 0.0,
        proposal_hidden_dim: int = 128,
        straight_through: bool = True,
        ocrp_mode: str = "pixel_patch",
        macro_lr_tile_size: int = 3,
        token_conditioned_member_bias: bool | None = None,
        pool_chunk_size: int = 512,
        router_chunk_size: int = 512,
        proposal_chunk_size: int = 128,
        proposal_token_chunk_size: int | None = None,
        use_upsample_residual: bool = False,
        upsample_residual_weight: float = 1.0,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feature_dim = int(self.irreps_feat.dim)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.window_size = int(window_size)
        self.kmax_slots = int(kmax_slots)
        self.cluster_feature_l2_threshold = _resolve_cluster_feature_l2_threshold(
            cluster_feature_l2_threshold=cluster_feature_l2_threshold,
            legacy_cluster_threshold_deg=cluster_threshold_deg,
        )
        self.straight_through = bool(straight_through)
        self.ocrp_mode = _resolve_ocrp_mode(ocrp_mode)
        self.macro_lr_tile_size = _validate_odd_positive_int(
            "macro_lr_tile_size",
            macro_lr_tile_size,
        )
        self.hr_patch_shape = (
            self.upsample_factor
            if self.ocrp_mode == "pixel_patch"
            else (
                int(self.macro_lr_tile_size * self.upsample_factor[0]),
                int(self.macro_lr_tile_size * self.upsample_factor[1]),
            )
        )
        self.hr_patch_size = (
            int(self.hr_patch_shape[0])
            if self.hr_patch_shape[0] == self.hr_patch_shape[1]
            else self.hr_patch_shape
        )
        self.hr_patch_tokens = _num_patch_tokens(self.hr_patch_shape)
        self.token_conditioned_member_bias = (
            True
            if token_conditioned_member_bias is None
            else bool(token_conditioned_member_bias)
        )
        self.router_geom_logit_bias = float(router_geom_logit_bias)
        if self.router_geom_logit_bias < 0.0:
            raise ValueError(
                f"router_geom_logit_bias must be >= 0, got {router_geom_logit_bias}"
            )
        self.use_upsample_residual = bool(use_upsample_residual)
        self.set_upsample_residual_weight(float(upsample_residual_weight))

        self.phase_embed = PhaseEmbeddingGrid(
            upsample_factor=self.upsample_factor,
            emb_dim=int(phase_dim),
            patch_size=self.hr_patch_shape,
        )
        self.quat_clusterer = QuaternionBankClusterer(
            sym_ops_quat=sym_ops_quat,
            threshold_deg=float(cluster_threshold_deg),
            connectivity=int(cluster_connectivity),
            window_size=int(window_size),
        )
        self.feature_clusterer = FeatureBankClusterer(
            threshold_l2=self.cluster_feature_l2_threshold,
            connectivity=int(cluster_connectivity),
            window_size=int(window_size),
        )
        # Backward-compatible alias used by notebooks and analysis helpers.
        self.clusterer = self.quat_clusterer
        del sym_ops_quat
        self.slot_builder = ClusterSlotBuilder(
            kmax_slots=int(kmax_slots),
            window_size=int(window_size),
        )
        self.context_builder = LearnedWeightedSlotContextBuilder(
            irreps_feat=self.irreps_feat,
            meta_dim=ClusterSlotBuilder.META_DIM,
            window_size=int(window_size),
        )
        self.slot_pool = WithinSlotInvariantPool(
            irreps_feat=self.irreps_feat,
            meta_dim=ClusterSlotBuilder.META_DIM,
            phase_dim=int(phase_dim),
            window_size=int(window_size),
            patch_shape=self.hr_patch_shape,
            hidden_dim=max(64, int(proposal_hidden_dim)),
            chunk_size=int(pool_chunk_size),
            token_conditioned_member_bias=self.token_conditioned_member_bias,
        )
        self.router = PatchSlotRouter(
            irreps_feat=self.irreps_feat,
            meta_dim=ClusterSlotBuilder.META_DIM,
            kmax_slots=int(kmax_slots),
            upsample_factor=self.upsample_factor,
            patch_size=self.hr_patch_shape,
            phase_dim=int(phase_dim),
            hidden_dim=int(router_hidden_dim),
            conv_hidden_dim=int(router_conv_hidden_dim),
            chunk_size=int(router_chunk_size),
            slot_mass_power=float(router_slot_mass_power),
            uniform_slot_mix=float(router_uniform_slot_mix),
            use_slot_type_meta=bool(router_use_slot_type_meta),
            use_raw_token_ctx=True,
        )
        self.proposal_head = SharedTPPatchProposalHead(
            irreps_feat=self.irreps_feat,
            meta_dim=ClusterSlotBuilder.META_DIM,
            phase_dim=int(phase_dim),
            upsample_factor=self.upsample_factor,
            patch_size=self.hr_patch_shape,
            hidden_dim=int(proposal_hidden_dim),
            chunk_size=int(proposal_chunk_size),
            token_chunk_size=proposal_token_chunk_size,
        )
        token_support_index = self._build_router_token_support_index()
        self.register_buffer(
            "router_token_support_index",
            token_support_index,
            persistent=False,
        )

    def set_upsample_residual_weight(self, weight: float) -> None:
        weight = float(weight)
        if weight < 0.0:
            raise ValueError(f"upsample_residual_weight must be >= 0, got {weight}")
        self.upsample_residual_weight = weight

    def _phase_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        phase_ids = torch.arange(
            self.hr_patch_tokens,
            device=device,
            dtype=torch.long,
        )
        return self.phase_embed(phase_ids).to(dtype=dtype)

    def _build_router_token_support_index(self) -> torch.Tensor:
        patch_h, patch_w = self.hr_patch_shape
        up_h, up_w = self.upsample_factor
        center = self.window_size // 2
        if self.ocrp_mode == "pixel_patch":
            center_idx = center * self.window_size + center
            return torch.full((self.hr_patch_tokens,), center_idx, dtype=torch.long)

        tile_half = self.macro_lr_tile_size // 2
        yy = torch.arange(patch_h, dtype=torch.long)
        xx = torch.arange(patch_w, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        lr_local_y = (grid_y // up_h).clamp(min=0, max=self.macro_lr_tile_size - 1)
        lr_local_x = (grid_x // up_w).clamp(min=0, max=self.macro_lr_tile_size - 1)
        support_y = (center - tile_half) + lr_local_y
        support_x = (center - tile_half) + lr_local_x
        return (support_y * self.window_size + support_x).reshape(-1)

    def _router_targets_from_slot_mask(
        self,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_index = self.router_token_support_index.to(device=slot_mask.device)
        target_mask = slot_mask.index_select(dim=-1, index=node_index).permute(0, 1, 3, 2)
        target_valid = target_mask.any(dim=-1)
        target_slot = target_mask.to(dtype=torch.long).argmax(dim=-1)
        return target_slot, target_valid

    def _build_support_banks(
        self,
        lr_quats: torch.Tensor,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        need_quat_bank: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, int]]]:
        if self.ocrp_mode == "pixel_patch":
            bank_q = (
                _build_local_patch_bank(lr_quats, img_shape=lr_shape, window_size=self.window_size)
                if need_quat_bank
                else None
            )
            bank_f = _build_local_patch_bank(feat_lr, img_shape=lr_shape, window_size=self.window_size)
            support_meta = {
                "grid_shape": (int(lr_shape[0]), int(lr_shape[1])),
                "padded_shape": (int(lr_shape[0]), int(lr_shape[1])),
            }
            return bank_q, bank_f, support_meta

        bank_q = None
        bank_f, meta_f = _build_macro_tile_patch_bank(
            feat_lr,
            img_shape=lr_shape,
            window_size=self.window_size,
            tile_size=self.macro_lr_tile_size,
        )
        if need_quat_bank:
            bank_q, meta_q = _build_macro_tile_patch_bank(
                lr_quats,
                img_shape=lr_shape,
                window_size=self.window_size,
                tile_size=self.macro_lr_tile_size,
            )
            if meta_q != meta_f:
                raise ValueError(
                    "Quaternion and feature macro-tile banks disagree on support metadata: "
                    f"{meta_q} vs {meta_f}"
                )
        return bank_q, bank_f, {
            "grid_shape": meta_f["tile_shape"],
            "padded_shape": meta_f["padded_shape"],
        }

    def _assemble_patch_tokens(
        self,
        patch_out: torch.Tensor,
        lr_shape: tuple[int, int],
    ) -> torch.Tensor:
        bsz, nwin, patch_tokens, cdim = patch_out.shape
        h_lr, w_lr = int(lr_shape[0]), int(lr_shape[1])
        if nwin != h_lr * w_lr:
            raise ValueError(f"Expected {h_lr*w_lr} patches, got {nwin}")
        r_h, r_w = self.upsample_factor
        if patch_tokens != r_h * r_w:
            raise ValueError(f"Expected patch tokens {r_h * r_w}, got {patch_tokens}")
        img = patch_out.view(bsz, h_lr, w_lr, r_h, r_w, cdim).permute(0, 1, 3, 2, 4, 5)
        img = img.reshape(bsz, h_lr * r_h, w_lr * r_w, cdim)
        return img.reshape(bsz, h_lr * r_h * w_lr * r_w, cdim)

    def _assemble_macro_patch_tokens(
        self,
        patch_out: torch.Tensor,
        grid_shape: tuple[int, int],
        hr_crop_shape: tuple[int, int],
    ) -> torch.Tensor:
        bsz, nwin, patch_tokens, cdim = patch_out.shape
        grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
        if nwin != grid_h * grid_w:
            raise ValueError(f"Expected {grid_h*grid_w} macro tiles, got {nwin}")
        patch_h, patch_w = self.hr_patch_shape
        if patch_tokens != patch_h * patch_w:
            raise ValueError(
                f"Expected patch tokens {patch_h * patch_w}, got {patch_tokens}"
            )

        img = (
            patch_out.view(bsz, grid_h, grid_w, patch_h, patch_w, cdim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(bsz, grid_h * patch_h, grid_w * patch_w, cdim)
        )
        crop_h, crop_w = int(hr_crop_shape[0]), int(hr_crop_shape[1])
        if crop_h > int(img.shape[1]) or crop_w > int(img.shape[2]):
            raise ValueError(
                f"Cannot crop macro-tile assembly of shape {tuple(img.shape[1:3])} "
                f"to requested HR shape {(crop_h, crop_w)}"
            )
        img = img[:, :crop_h, :crop_w, :]
        return img.reshape(bsz, crop_h * crop_w, cdim)

    
    @staticmethod
    def _hard_owner_from_logits(logits: torch.Tensor, straight_through: bool) -> tuple[torch.Tensor, torch.Tensor]:
        owner_idx = torch.argmax(logits, dim=-1)
        onehot = F.one_hot(owner_idx, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
        if straight_through and logits.requires_grad:
            soft = torch.softmax(logits, dim=-1)
            onehot = onehot + soft - soft.detach()
        return owner_idx, onehot

    def forward(
        self,
        lr_quats: torch.Tensor,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        return_aux: bool = False,
    ):
        batched = feat_lr.dim() == 3
        if not batched:
            feat_lr = feat_lr.unsqueeze(0)
            lr_quats = lr_quats.unsqueeze(0)
        if feat_lr.shape[:2] != lr_quats.shape[:2]:
            raise ValueError("lr_quats and feat_lr must agree on batch and LR token count")

        bank_q, bank_f, support_meta = self._build_support_banks(
            lr_quats=lr_quats,
            feat_lr=feat_lr,
            lr_shape=lr_shape,
            need_quat_bank=True,
        )

        if bank_q is not None:
            cluster_ids = self.quat_clusterer(bank_q)
        else:
            cluster_ids = self.feature_clusterer(bank_f)
        slot_info = self.slot_builder(cluster_ids)
        slot_mask = slot_info["slot_mask"]
        slot_meta = slot_info["slot_meta"]
        slot_valid = slot_info["slot_valid"]
        router_target_slot, router_target_valid = self._router_targets_from_slot_mask(
            slot_mask
        )

        slot_ctx, slot_anchor_alpha = self.context_builder(
            bank_f=bank_f,
            slot_mask=slot_mask,
            slot_meta=slot_meta,
            return_alpha=return_aux,
        )
        phase_grid = self._phase_grid(device=feat_lr.device, dtype=feat_lr.dtype)
        slot_pooled_ctx, slot_pool_alpha = self.slot_pool(
            slot_anchor_ctx=slot_ctx,
            slot_meta=slot_meta,
            bank_f=bank_f,
            slot_mask=slot_mask,
            phase_grid=phase_grid,
            return_alpha=return_aux,
        )

        router_logits = self.router(
            slot_ctx=slot_pooled_ctx,
            slot_meta=slot_meta,
            phase_grid=phase_grid,
        )
        if self.router_geom_logit_bias > 0.0:
            router_prior = F.one_hot(
                router_target_slot,
                num_classes=self.kmax_slots,
            ).to(dtype=router_logits.dtype)
            router_prior = (
                router_prior
                * router_target_valid.unsqueeze(-1).to(dtype=router_logits.dtype)
            )
            router_logits = router_logits + self.router_geom_logit_bias * router_prior
        owner_idx, owner_onehot = self._hard_owner_from_logits(
            router_logits,
            straight_through=self.straight_through and self.training,
        )

        patch_prop = self.proposal_head(
            slot_ctx=slot_pooled_ctx,
            slot_meta=slot_meta,
            phase_grid=phase_grid,
            slot_anchor_ctx=slot_ctx,
        )
        owner_mask = owner_onehot.permute(0, 1, 3, 2).unsqueeze(-1)
        patch_out = (owner_mask * patch_prop).sum(dim=2)
        r_h, r_w = self.upsample_factor
        hr_shape = (
            int(lr_shape[0]) * r_h,
            int(lr_shape[1]) * r_w,
        )
        if self.ocrp_mode == "pixel_patch":
            feat_hr = self._assemble_patch_tokens(patch_out, lr_shape=lr_shape)
        else:
            feat_hr = self._assemble_macro_patch_tokens(
                patch_out,
                grid_shape=support_meta["grid_shape"],
                hr_crop_shape=hr_shape,
            )

        if self.use_upsample_residual:
            bsz = feat_lr.shape[0]
            h_lr, w_lr = int(lr_shape[0]), int(lr_shape[1])
            cdim = feat_hr.shape[-1]
            feat_hr_img = feat_hr.view(bsz, int(hr_shape[0]), int(hr_shape[1]), cdim)
            lr_img = feat_lr.view(bsz, h_lr, w_lr, cdim)
            feat_hr_img[:, 0::r_h, 0::r_w, :] += self.upsample_residual_weight * lr_img
            feat_hr = feat_hr_img.reshape(bsz, int(hr_shape[0]) * int(hr_shape[1]), cdim)

        if not batched:
            feat_hr = feat_hr.squeeze(0)

        if not return_aux:
            return feat_hr, hr_shape

        aux = {
            "bank_q": bank_q,
            "bank_f": bank_f,
            "cluster_ids": cluster_ids,
            "slot_mask": slot_mask,
            "slot_valid": slot_valid,
            "slot_meta": slot_meta,
            "slot_ctx": slot_ctx,
            "slot_anchor_alpha": slot_anchor_alpha,
            "slot_pooled_ctx": slot_pooled_ctx,
            "slot_pool_alpha": slot_pool_alpha,
            "router_logits": router_logits,
            "router_target_slot": router_target_slot,
            "router_target_valid": router_target_valid,
            "owner_idx": owner_idx,
            "patch_prop": patch_prop,
            "patch_out": patch_out,
            "ocrp_mode": self.ocrp_mode,
            "macro_lr_tile_size": self.macro_lr_tile_size,
            "hr_patch_size": self.hr_patch_size,
            "hr_patch_shape": self.hr_patch_shape,
            "hr_patch_tokens": self.hr_patch_tokens,
            "pool_chunk_size": self.slot_pool.chunk_size,
            "router_chunk_size": self.router.chunk_size,
            "proposal_chunk_size": self.proposal_head.chunk_size,
            "proposal_token_chunk_size": self.proposal_head.token_chunk_size,
            "token_conditioned_member_bias": self.token_conditioned_member_bias,
            "support_grid_shape": support_meta["grid_shape"],
            "lr_padded_shape": support_meta["padded_shape"],
        }
        if not batched:
            aux = {
                key: (val.squeeze(0) if isinstance(val, torch.Tensor) and val.shape[0] == 1 else val)
                for key, val in aux.items()
            }
        return feat_hr, hr_shape, aux


class IsoEmbeddingSROCRP(nn.Module):
    """OCRP SR model: encoder -> OCRP upsampler -> decoder."""

    def __init__(
        self,
        crystal: str = "fcc",
        d6_convention: str = "z_axis",
        device: str | torch.device | None = None,
        feature_irreps: str = "full",
        use_lr_conv1: bool = True,
        lr_conv1_kernel_size: int = 5,
        use_residual_lr1: bool = True,
        lr_conv1_residual_weight: float = 1.0,
        conv_feature_mask_cosine_threshold: float | None = 0.98,
        conv_feature_mask_l2_threshold: float | None = None,
        conv_feature_mask_soft: bool = False,
        conv_feature_mask_temperature: float = 32.0,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        window_size: int = 5,
        kmax_slots: int = 10,
        cluster_threshold_deg: float = 2.0,
        cluster_feature_l2_threshold: float | None = None,
        cluster_connectivity: int = 8,
        phase_dim: int = 32,
        ocrp_router_hidden_dim: int = 128,
        ocrp_router_conv_hidden_dim: int = 64,
        ocrp_router_slot_mass_power: float = 0.25,
        ocrp_router_uniform_slot_mix: float = 0.75,
        ocrp_router_use_slot_type_meta: bool = True,
        ocrp_router_geom_logit_bias: float = 0.0,
        ocrp_proposal_hidden_dim: int = 128,
        ocrp_slot_ratio_loss_weight: float = 0.0,
        ocrp_router_geom_loss_weight: float = 0.0,
        ocrp_router_geom_boundary_only: bool = False,
        ocrp_slot_ratio_temperature: float = 1.0,
        ocrp_straight_through: bool = True,
        ocrp_mode: str = "pixel_patch",
        macro_lr_tile_size: int = 3,
        ocrp_token_conditioned_member_bias: bool | None = None,
        ocrp_upsample_residual: bool = False,
        ocrp_upsample_residual_weight: float = 1.0,
        ocrp_pool_chunk_size: int = 512,
        ocrp_router_chunk_size: int = 512,
        ocrp_proposal_chunk_size: int = 128,
        ocrp_proposal_token_chunk_size: int | None = None,
        use_hr_conv1: bool = True,
        hr_conv1_kernel_size: int = 7,
        use_residual_hr1: bool = True,
        hr_conv1_residual_weight: float = 1.0,
        hr_conv_feature_mask_l2_threshold: float | None = None,
        hr_conv_feature_mask_cosine_threshold: float | None = None,
        hr_conv_feature_mask_soft: bool | None = None,
        hr_conv_feature_mask_temperature: float | None = None,
        use_hr_conv2: bool = False,
        hr_conv2_kernel_size: int | None = None,
        use_residual_hr2: bool = True,
        hr_conv2_residual_weight: float = 1.0,
        hr_conv2_feature_mask_l2_threshold: float | None = None,
        hr_conv2_feature_mask_cosine_threshold: float | None = None,
        hr_conv2_feature_mask_soft: bool | None = None,
        hr_conv2_feature_mask_temperature: float | None = None,
        use_hr_conv3: bool = False,
        hr_conv3_kernel_size: int | None = None,
        use_residual_hr3: bool = True,
        hr_conv3_residual_weight: float = 1.0,
        hr_conv3_feature_mask_l2_threshold: float | None = None,
        hr_conv3_feature_mask_cosine_threshold: float | None = None,
        hr_conv3_feature_mask_soft: bool | None = None,
        hr_conv3_feature_mask_temperature: float | None = None,
        decoder_cubochoric_resolution: int = 1,
        decoder_num_starts: int = 6,
        decoder_steps: int = 25,
        decoder_lr: float = 0.05,
        decoder_method: str = "cubochoric",
        decoder_max_table_rows: int | None = None,
        decoder_table_cache_dir: str | Path | None = "out/decoder_lookup_tables",
        decoder_eager_init: bool = False,
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

        feature_key = str(feature_irreps).lower()
        if feature_key not in {"a1", "full"}:
            raise ValueError(f"feature_irreps must be 'a1' or 'full', got {feature_irreps}")
        self.feature_irreps = feature_key
        if self.feature_irreps == "a1":
            self.irreps_feat = self.encoder.irreps_a1
            self.feature_dim = int(self.encoder.out_dim_a1)
        else:
            self.irreps_feat = self.encoder.irreps_full
        self.feature_dim = int(self.encoder.out_dim_full)

        self.use_lr_conv1 = bool(use_lr_conv1)
        self.use_hr_conv1 = bool(use_hr_conv1)
        self.use_hr_conv2 = bool(use_hr_conv2)
        self.use_hr_conv3 = bool(use_hr_conv3)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.use_residual_lr1 = bool(use_residual_lr1)
        self.use_residual_hr1 = bool(use_residual_hr1)
        self.use_residual_hr2 = bool(use_residual_hr2)
        self.use_residual_hr3 = bool(use_residual_hr3)
        self.lr_conv1_residual_weight = float(lr_conv1_residual_weight)
        self.hr_conv1_residual_weight = float(hr_conv1_residual_weight)
        self.hr_conv2_residual_weight = float(hr_conv2_residual_weight)
        self.hr_conv3_residual_weight = float(hr_conv3_residual_weight)
        self.ocrp_slot_ratio_loss_weight = float(ocrp_slot_ratio_loss_weight)
        self.ocrp_router_geom_loss_weight = float(ocrp_router_geom_loss_weight)
        self.ocrp_router_geom_boundary_only = bool(ocrp_router_geom_boundary_only)
        self.ocrp_slot_ratio_temperature = float(ocrp_slot_ratio_temperature)
        if self.ocrp_slot_ratio_loss_weight < 0.0:
            raise ValueError(
                "ocrp_slot_ratio_loss_weight must be >= 0, "
                f"got {ocrp_slot_ratio_loss_weight}"
            )
        if self.ocrp_router_geom_loss_weight < 0.0:
            raise ValueError(
                "ocrp_router_geom_loss_weight must be >= 0, "
                f"got {ocrp_router_geom_loss_weight}"
            )
        if self.ocrp_slot_ratio_temperature <= 0.0:
            raise ValueError(
                "ocrp_slot_ratio_temperature must be > 0, "
                f"got {ocrp_slot_ratio_temperature}"
            )
        self.lr_conv_feature_mask_cosine_threshold = (
            None if conv_feature_mask_cosine_threshold is None else float(conv_feature_mask_cosine_threshold)
        )
        self.lr_conv_feature_mask_l2_threshold = _resolve_feature_l2_threshold(
            explicit_l2_threshold=conv_feature_mask_l2_threshold,
            legacy_cosine_threshold=self.lr_conv_feature_mask_cosine_threshold,
            default_l2_threshold=math.radians(5.0),
        )
        self.lr_conv_feature_mask_soft = bool(conv_feature_mask_soft)
        self.lr_conv_feature_mask_temperature = float(conv_feature_mask_temperature)
        self.hr_conv_feature_mask_cosine_threshold = (
            self.lr_conv_feature_mask_cosine_threshold
            if hr_conv_feature_mask_cosine_threshold is None
            else float(hr_conv_feature_mask_cosine_threshold)
        )
        self.hr_conv_feature_mask_l2_threshold = _resolve_feature_l2_threshold(
            explicit_l2_threshold=(
                self.lr_conv_feature_mask_l2_threshold
                if hr_conv_feature_mask_l2_threshold is None
                else hr_conv_feature_mask_l2_threshold
            ),
            legacy_cosine_threshold=self.hr_conv_feature_mask_cosine_threshold,
            default_l2_threshold=self.lr_conv_feature_mask_l2_threshold,
        )
        self.hr_conv_feature_mask_soft = bool(
            self.lr_conv_feature_mask_soft
            if hr_conv_feature_mask_soft is None
            else hr_conv_feature_mask_soft
        )
        self.hr_conv_feature_mask_temperature = float(
            self.lr_conv_feature_mask_temperature
            if hr_conv_feature_mask_temperature is None
            else hr_conv_feature_mask_temperature
        )
        self.hr_conv2_kernel_size = int(
            hr_conv1_kernel_size if hr_conv2_kernel_size is None else hr_conv2_kernel_size
        )
        self.hr_conv2_feature_mask_cosine_threshold = (
            self.hr_conv_feature_mask_cosine_threshold
            if hr_conv2_feature_mask_cosine_threshold is None
            else float(hr_conv2_feature_mask_cosine_threshold)
        )
        self.hr_conv2_feature_mask_l2_threshold = _resolve_feature_l2_threshold(
            explicit_l2_threshold=(
                self.hr_conv_feature_mask_l2_threshold
                if hr_conv2_feature_mask_l2_threshold is None
                else hr_conv2_feature_mask_l2_threshold
            ),
            legacy_cosine_threshold=self.hr_conv2_feature_mask_cosine_threshold,
            default_l2_threshold=self.hr_conv_feature_mask_l2_threshold,
        )
        self.hr_conv2_feature_mask_soft = bool(
            self.hr_conv_feature_mask_soft
            if hr_conv2_feature_mask_soft is None
            else hr_conv2_feature_mask_soft
        )
        self.hr_conv2_feature_mask_temperature = float(
            self.hr_conv_feature_mask_temperature
            if hr_conv2_feature_mask_temperature is None
            else hr_conv2_feature_mask_temperature
        )
        self.hr_conv3_kernel_size = int(
            hr_conv1_kernel_size if hr_conv3_kernel_size is None else hr_conv3_kernel_size
        )
        self.hr_conv3_feature_mask_cosine_threshold = (
            self.hr_conv_feature_mask_cosine_threshold
            if hr_conv3_feature_mask_cosine_threshold is None
            else float(hr_conv3_feature_mask_cosine_threshold)
        )
        self.hr_conv3_feature_mask_l2_threshold = _resolve_feature_l2_threshold(
            explicit_l2_threshold=(
                self.hr_conv_feature_mask_l2_threshold
                if hr_conv3_feature_mask_l2_threshold is None
                else hr_conv3_feature_mask_l2_threshold
            ),
            legacy_cosine_threshold=self.hr_conv3_feature_mask_cosine_threshold,
            default_l2_threshold=self.hr_conv_feature_mask_l2_threshold,
        )
        self.hr_conv3_feature_mask_soft = bool(
            self.hr_conv_feature_mask_soft
            if hr_conv3_feature_mask_soft is None
            else hr_conv3_feature_mask_soft
        )
        self.hr_conv3_feature_mask_temperature = float(
            self.hr_conv_feature_mask_temperature
            if hr_conv3_feature_mask_temperature is None
            else hr_conv3_feature_mask_temperature
        )
        self.conv_lr1 = FeatureDistanceMaskedEquivariantSpatialConv(
            kernel_size=int(lr_conv1_kernel_size),
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=self.use_residual_lr1,
            residual_weight=self.lr_conv1_residual_weight,
            feature_mask_l2_threshold=self.lr_conv_feature_mask_l2_threshold,
            feature_mask_cosine_threshold=self.lr_conv_feature_mask_cosine_threshold,
            feature_mask_soft=self.lr_conv_feature_mask_soft,
            feature_mask_temperature=self.lr_conv_feature_mask_temperature,
        )
        self.ocrp = OCRPPatchUpsampler(
            irreps_feat=self.irreps_feat,
            sym_ops_quat=self.encoder.sym_ops,
            upsample_factor=self.upsample_factor,
            window_size=int(window_size),
            kmax_slots=int(kmax_slots),
            cluster_threshold_deg=float(cluster_threshold_deg),
            cluster_feature_l2_threshold=cluster_feature_l2_threshold,
            cluster_connectivity=int(cluster_connectivity),
            phase_dim=int(phase_dim),
            router_hidden_dim=int(ocrp_router_hidden_dim),
            router_conv_hidden_dim=int(ocrp_router_conv_hidden_dim),
            router_slot_mass_power=float(ocrp_router_slot_mass_power),
            router_uniform_slot_mix=float(ocrp_router_uniform_slot_mix),
            router_use_slot_type_meta=bool(ocrp_router_use_slot_type_meta),
            router_geom_logit_bias=float(ocrp_router_geom_logit_bias),
            proposal_hidden_dim=int(ocrp_proposal_hidden_dim),
            straight_through=bool(ocrp_straight_through),
            ocrp_mode=str(ocrp_mode),
            macro_lr_tile_size=int(macro_lr_tile_size),
            token_conditioned_member_bias=ocrp_token_conditioned_member_bias,
            use_upsample_residual=bool(ocrp_upsample_residual),
            upsample_residual_weight=float(ocrp_upsample_residual_weight),
            pool_chunk_size=int(ocrp_pool_chunk_size),
            router_chunk_size=int(ocrp_router_chunk_size),
            proposal_chunk_size=int(ocrp_proposal_chunk_size),
            proposal_token_chunk_size=ocrp_proposal_token_chunk_size,
        )
        self.conv_hr1 = FeatureDistanceMaskedEquivariantSpatialConv(
            kernel_size=int(hr_conv1_kernel_size),
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=self.use_residual_hr1,
            residual_weight=self.hr_conv1_residual_weight,
            feature_mask_l2_threshold=self.hr_conv_feature_mask_l2_threshold,
            feature_mask_cosine_threshold=self.hr_conv_feature_mask_cosine_threshold,
            feature_mask_soft=self.hr_conv_feature_mask_soft,
            feature_mask_temperature=self.hr_conv_feature_mask_temperature,
        )
        self.conv_hr2 = FeatureDistanceMaskedEquivariantSpatialConv(
            kernel_size=self.hr_conv2_kernel_size,
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=self.use_residual_hr2,
            residual_weight=self.hr_conv2_residual_weight,
            feature_mask_l2_threshold=self.hr_conv2_feature_mask_l2_threshold,
            feature_mask_cosine_threshold=self.hr_conv2_feature_mask_cosine_threshold,
            feature_mask_soft=self.hr_conv2_feature_mask_soft,
            feature_mask_temperature=self.hr_conv2_feature_mask_temperature,
        )
        self.conv_hr3: FeatureDistanceMaskedEquivariantSpatialConv | None = None
        if self.use_hr_conv3:
            self.conv_hr3 = FeatureDistanceMaskedEquivariantSpatialConv(
                kernel_size=self.hr_conv3_kernel_size,
                irreps_in=self.irreps_feat,
                irreps_out=self.irreps_feat,
                use_residual=self.use_residual_hr3,
                residual_weight=self.hr_conv3_residual_weight,
                feature_mask_l2_threshold=self.hr_conv3_feature_mask_l2_threshold,
                feature_mask_cosine_threshold=self.hr_conv3_feature_mask_cosine_threshold,
                feature_mask_soft=self.hr_conv3_feature_mask_soft,
                feature_mask_temperature=self.hr_conv3_feature_mask_temperature,
            )

        self.decoder: CubochoricOptimizingLocalIsoDecoder | None = None
        self._decoder_eager_init = bool(decoder_eager_init)
        self._decoder_kwargs = {
            "encoder": self.encoder,
            "cubochoric_resolution": int(decoder_cubochoric_resolution),
            "method": str(decoder_method),
            "num_starts": int(decoder_num_starts),
            "steps": int(decoder_steps),
            "lr": float(decoder_lr),
            "target_irreps": self.feature_irreps,
            "max_table_rows": decoder_max_table_rows,
            "table_cache_dir": decoder_table_cache_dir,
        }
        if self._decoder_eager_init:
            self.decoder = self._build_decoder()

    def _build_decoder(self) -> CubochoricOptimizingLocalIsoDecoder:
        return CubochoricOptimizingLocalIsoDecoder(**self._decoder_kwargs)

    def _ensure_decoder(self) -> CubochoricOptimizingLocalIsoDecoder:
        if self.decoder is None:
            self.decoder = self._build_decoder()
        return self.decoder

    def _filter_incompatible_state_dict(self, state_dict):
        filtered_state_dict = dict(state_dict)
        if self.conv_hr3 is None:
            legacy_hr3_keys = [key for key in filtered_state_dict.keys() if key.startswith("conv_hr3.")]
            if legacy_hr3_keys:
                for key in legacy_hr3_keys:
                    filtered_state_dict.pop(key, None)
                warnings.warn(
                    "Ignoring legacy conv_hr3 weights because use_hr_conv3=False for this 4x4 OCRP model.",
                    RuntimeWarning,
                )
        return filtered_state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        filtered_state_dict = self._filter_incompatible_state_dict(state_dict)
        return super().load_state_dict(filtered_state_dict, strict=strict, assign=assign)

    def set_ocrp_upsample_residual_weight(self, weight: float) -> None:
        self.ocrp.set_upsample_residual_weight(weight)

    def get_ocrp_upsample_residual_weight(self) -> float:
        return float(self.ocrp.upsample_residual_weight)

    @staticmethod
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=1,
        )

    def reduce_to_fz(self, quats: torch.Tensor) -> torch.Tensor:
        quats = _normalize_quaternions(quats)
        bsz = quats.shape[0]
        q_exp = quats.unsqueeze(1).expand(-1, self.encoder.sym_ops_inv.shape[0], -1)
        syms = self.encoder.sym_ops_inv.unsqueeze(0).expand(bsz, -1, -1)
        fam = self.quat_mul(syms.reshape(-1, 4), q_exp.reshape(-1, 4)).view(bsz, syms.shape[1], 4)
        fam = _normalize_quaternions(fam.reshape(-1, 4)).view(bsz, syms.shape[1], 4)
        idx = torch.argmax(fam[..., 0].abs(), dim=1)
        batch_idx = torch.arange(bsz, device=quats.device)
        return _normalize_quaternions(fam[batch_idx, idx])

    def encode(self, quats: torch.Tensor) -> torch.Tensor:
        if self.feature_irreps == "a1":
            return self.encoder.forward_a1(quats)
        return self.encoder.forward_full(quats)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        decoder = self._ensure_decoder()
        batched = features.dim() == 3
        if batched:
            bsz, n, cdim = features.shape
            q = decoder(features.reshape(bsz * n, cdim))
            return self.reduce_to_fz(q).reshape(bsz, n, 4)
        q = decoder(features)
        return self.reduce_to_fz(q)

    def _forward_sr_features(
        self,
        lr_quats: torch.Tensor,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        return_aux: bool = False,
    ):
        feat_pre = feat_lr
        if self.use_lr_conv1:
            feat_pre = self.conv_lr1(feat_pre, lr_shape)

        out = self.ocrp(
            lr_quats=lr_quats,
            feat_lr=feat_pre,
            lr_shape=lr_shape,
            return_aux=return_aux,
        )
        if return_aux:
            feat_hr, hr_shape, aux = out
            feat_hr_raw_ocrp = feat_hr
        else:
            feat_hr, hr_shape = out

        feat_hr_after_conv1 = feat_hr
        if self.use_hr_conv1:
            feat_hr_after_conv1 = self.conv_hr1(feat_hr_after_conv1, hr_shape)
        feat_hr_after_conv2 = feat_hr_after_conv1
        if self.use_hr_conv2:
            feat_hr_after_conv2 = self.conv_hr2(feat_hr_after_conv2, hr_shape)
        feat_hr = feat_hr_after_conv2
        if self.conv_hr3 is not None:
            feat_hr = self.conv_hr3(feat_hr, hr_shape)

        if not return_aux:
            return feat_hr, hr_shape

        aux["feat_lr_pre_ocrp"] = feat_pre
        aux["feat_hr_raw_ocrp"] = feat_hr_raw_ocrp
        aux["feat_hr_post_hr_conv1"] = feat_hr_after_conv1
        aux["feat_hr_post_hr_conv2"] = feat_hr_after_conv2
        aux["feat_hr_post_hr_conv"] = feat_hr
        return feat_hr, hr_shape, aux

    def forward_sr(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = True,
        return_aux: bool = False,
    ):
        lr_quats = lr_quats.to(self.device)
        if normalize_input:
            lr_quats = _normalize_quaternions(lr_quats)

        feat_lr = self.encode(lr_quats)
        if return_aux:
            feat_hr, _hr_shape, aux = self._forward_sr_features(
                lr_quats=lr_quats.view(feat_lr.shape[0], feat_lr.shape[1], 4) if feat_lr.dim() == 3 else lr_quats,
                feat_lr=feat_lr,
                lr_shape=lr_shape,
                return_aux=True,
            )
            return self.decode(feat_hr), aux

        feat_hr, _hr_shape = self._forward_sr_features(
            lr_quats=lr_quats.view(feat_lr.shape[0], feat_lr.shape[1], 4) if feat_lr.dim() == 3 else lr_quats,
            feat_lr=feat_lr,
            lr_shape=lr_shape,
            return_aux=False,
        )
        return self.decode(feat_hr)

    def forward(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = True,
        return_aux: bool = False,
    ):
        return self.forward_sr(
            lr_quats,
            lr_shape=lr_shape,
            normalize_input=normalize_input,
            return_aux=return_aux,
        )

    def _slot_ratio_loss_from_aux(
        self,
        aux: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        router_logits = aux["router_logits"]
        slot_meta = aux["slot_meta"]
        slot_valid = aux["slot_valid"]
        if router_logits.dim() == 3:
            router_logits = router_logits.unsqueeze(0)
            slot_meta = slot_meta.unsqueeze(0)
            slot_valid = slot_valid.unsqueeze(0)

        eps = 1e-6
        valid = slot_valid.to(dtype=router_logits.dtype)
        probs = torch.softmax(
            router_logits / self.ocrp_slot_ratio_temperature,
            dim=-1,
        )
        probs = probs * valid.unsqueeze(2)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

        pred_ratio = probs.mean(dim=2) * valid
        pred_ratio = pred_ratio / pred_ratio.sum(dim=-1, keepdim=True).clamp_min(eps)

        target_ratio = slot_meta[..., ClusterSlotBuilder.META_MASS].to(dtype=router_logits.dtype)
        target_ratio = target_ratio * valid
        target_ratio = target_ratio / target_ratio.sum(dim=-1, keepdim=True).clamp_min(eps)

        per_slot = F.smooth_l1_loss(pred_ratio, target_ratio, reduction="none") * valid
        valid_count = valid.sum(dim=-1)
        per_window = per_slot.sum(dim=-1) / valid_count.clamp_min(1.0)
        has_valid = valid_count > 0.0
        if has_valid.any():
            ratio_loss = per_window[has_valid].mean()
            ratio_l1 = (
                ((pred_ratio - target_ratio).abs() * valid).sum(dim=-1)
                / valid_count.clamp_min(1.0)
            )[has_valid].mean()
        else:
            ratio_loss = router_logits.new_zeros(())
            ratio_l1 = router_logits.new_zeros(())
        return ratio_loss, {
            "slot_ratio_unweighted": ratio_loss,
            "slot_ratio_l1": ratio_l1,
        }

    def _router_geom_loss_from_aux(
        self,
        aux: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        router_logits = aux["router_logits"]
        router_target_slot = aux["router_target_slot"]
        router_target_valid = aux["router_target_valid"]
        if router_logits.dim() == 3:
            router_logits = router_logits.unsqueeze(0)
            router_target_slot = router_target_slot.unsqueeze(0)
            router_target_valid = router_target_valid.unsqueeze(0)

        valid = router_target_valid.to(dtype=torch.bool)
        selected = valid
        boundary_frac = router_logits.new_ones(())
        if self.ocrp_router_geom_boundary_only:
            selected = self._router_geom_boundary_mask(
                router_target_slot=router_target_slot,
                router_target_valid=router_target_valid,
            )
            valid_count = valid.to(dtype=router_logits.dtype).sum()
            if valid_count > 0:
                boundary_frac = selected.to(dtype=router_logits.dtype).sum() / valid_count
            else:
                boundary_frac = router_logits.new_zeros(())

        if not selected.any():
            zero = router_logits.new_zeros(())
            info = {
                "router_geom_unweighted": zero,
                "router_geom_acc": zero,
                "router_geom_supervised_frac": zero,
            }
            if self.ocrp_router_geom_boundary_only:
                info["router_geom_boundary_frac"] = boundary_frac
            return zero, info

        logits_valid = router_logits[selected]
        target_valid = router_target_slot[selected].to(dtype=torch.long)
        geom_loss = F.cross_entropy(logits_valid, target_valid)
        geom_acc = (logits_valid.argmax(dim=-1) == target_valid).to(dtype=router_logits.dtype).mean()
        info = {
            "router_geom_unweighted": geom_loss,
            "router_geom_acc": geom_acc,
            "router_geom_supervised_frac": selected.to(dtype=router_logits.dtype).mean(),
        }
        if self.ocrp_router_geom_boundary_only:
            info["router_geom_boundary_frac"] = boundary_frac
        return geom_loss, info

    def _router_geom_boundary_mask(
        self,
        router_target_slot: torch.Tensor,
        router_target_valid: torch.Tensor,
    ) -> torch.Tensor:
        patch_h, patch_w = self.ocrp.hr_patch_shape
        patch_tokens = patch_h * patch_w
        if router_target_slot.shape[-1] != patch_tokens:
            return router_target_valid.to(dtype=torch.bool)

        slot = router_target_slot.reshape(*router_target_slot.shape[:-1], patch_h, patch_w)
        valid = router_target_valid.reshape(*router_target_valid.shape[:-1], patch_h, patch_w).to(dtype=torch.bool)
        boundary = torch.zeros_like(valid)

        horiz = valid[..., :, 1:] & valid[..., :, :-1] & (slot[..., :, 1:] != slot[..., :, :-1])
        boundary[..., :, 1:] |= horiz
        boundary[..., :, :-1] |= horiz

        vert = valid[..., 1:, :] & valid[..., :-1, :] & (slot[..., 1:, :] != slot[..., :-1, :])
        boundary[..., 1:, :] |= vert
        boundary[..., :-1, :] |= vert

        diag = valid[..., 1:, 1:] & valid[..., :-1, :-1] & (slot[..., 1:, 1:] != slot[..., :-1, :-1])
        boundary[..., 1:, 1:] |= diag
        boundary[..., :-1, :-1] |= diag

        anti = valid[..., 1:, :-1] & valid[..., :-1, 1:] & (slot[..., 1:, :-1] != slot[..., :-1, 1:])
        boundary[..., 1:, :-1] |= anti
        boundary[..., :-1, 1:] |= anti

        return boundary.reshape_as(router_target_valid) & router_target_valid.to(dtype=torch.bool)

    def feature_loss_sr(
        self,
        lr_quats: torch.Tensor,
        hr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = True,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        lr_quats = lr_quats.to(self.device)
        hr_quats = hr_quats.to(self.device)

        batched = lr_quats.dim() == 3
        if batched:
            bsz = lr_quats.shape[0]
            lr_flat = lr_quats.reshape(-1, 4)
            hr_flat = hr_quats.reshape(-1, 4)
        else:
            bsz = 1
            lr_flat = lr_quats
            hr_flat = hr_quats

        if normalize_input:
            lr_flat = _normalize_quaternions(lr_flat)
            hr_flat = _normalize_quaternions(hr_flat)

        with torch.no_grad():
            feat_lr = self.encode(lr_flat).detach()
            feat_hr_tgt = self.encode(hr_flat).detach()

        if batched:
            feat_lr = feat_lr.reshape(bsz, -1, feat_lr.shape[-1])
            feat_hr_tgt = feat_hr_tgt.reshape(bsz, -1, feat_hr_tgt.shape[-1])
            lr_q_batched = lr_flat.reshape(bsz, -1, 4)
        else:
            lr_q_batched = lr_flat

        need_ratio_loss = self.ocrp_slot_ratio_loss_weight > 0.0
        need_router_geom_loss = self.ocrp_router_geom_loss_weight > 0.0
        need_aux = need_ratio_loss or need_router_geom_loss
        if need_aux:
            feat_hr_pred, _, aux = self._forward_sr_features(
                lr_quats=lr_q_batched,
                feat_lr=feat_lr,
                lr_shape=lr_shape,
                return_aux=True,
            )
        else:
            feat_hr_pred, _ = self._forward_sr_features(
                lr_quats=lr_q_batched,
                feat_lr=feat_lr,
                lr_shape=lr_shape,
                return_aux=False,
            )

        feature_loss = F.mse_loss(feat_hr_pred, feat_hr_tgt)
        total_loss = feature_loss
        info: dict[str, torch.Tensor] = {
            "loss_feature": feature_loss,
        }

        if need_ratio_loss:
            ratio_loss, ratio_info = self._slot_ratio_loss_from_aux(aux)
            weighted_ratio_loss = self.ocrp_slot_ratio_loss_weight * ratio_loss
            total_loss = total_loss + weighted_ratio_loss
            info["loss_slot_ratio"] = weighted_ratio_loss
            info.update(ratio_info)

        if need_router_geom_loss:
            router_geom_loss, router_geom_info = self._router_geom_loss_from_aux(aux)
            weighted_router_geom_loss = self.ocrp_router_geom_loss_weight * router_geom_loss
            total_loss = total_loss + weighted_router_geom_loss
            info["loss_router_geom"] = weighted_router_geom_loss
            info.update(router_geom_info)

        info["loss_total"] = total_loss
        if return_info:
            return total_loss, info
        return total_loss


__all__ = [
    "LocalIsoCrystalEncoder",
    "CubochoricOptimizingLocalIsoDecoder",
    "CosineMaskedEquivariantSpatialConv",
    "PhaseEmbeddingGrid",
    "QuaternionBankClusterer",
    "ClusterSlotBuilder",
    "LearnedWeightedSlotContextBuilder",
    "MeanSlotContextBuilder",
    "MedoidSlotContextBuilder",
    "InvariantSlotSummary",
    "WithinSlotInvariantPool",
    "PatchSlotRouter",
    "EquivariantSlotPatchQueryAnchor",
    "SharedTPPatchProposalHead",
    "OCRPPatchUpsampler",
    "IsoEmbeddingSROCRP",
]
