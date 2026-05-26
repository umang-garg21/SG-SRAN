from __future__ import annotations

import hashlib
import inspect
import json
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
    lead_shape = q1.shape[:-1]
    if q2.shape[:-1] != lead_shape or q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError(
            f"Expected q1 and q2 with matching leading dims and trailing quaternion dim 4, "
            f"got {tuple(q1.shape)} and {tuple(q2.shape)}"
        )

    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    max_pairs_per_chunk = 262144
    out_chunks: list[torch.Tensor] = []
    for start in range(0, q1_flat.shape[0], max_pairs_per_chunk):
        end = min(start + max_pairs_per_chunk, q1_flat.shape[0])
        q1_chunk = q1_flat[start:end]
        q2_chunk = q2_flat[start:end]
        dots = torch.einsum("bi,gij,bj->bg", q1_chunk, sym_ops, q2_chunk).abs().clamp(0.0, 1.0)
        best = dots.max(dim=-1).values
        out_chunks.append(2.0 * torch.acos(best))
    return torch.cat(out_chunks, dim=0).view(*lead_shape)


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


def _validate_positive_shape(
    name: str,
    value: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    shape = _as_patch_shape(value, name=name)
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError(f"{name} must be positive in both dims, got {shape}")
    return shape


def _build_macro_stride_patch_bank(
    features: torch.Tensor,
    img_shape: tuple[int, int],
    window_size: int,
    stride_shape: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    h, w = int(img_shape[0]), int(img_shape[1])
    stride_h, stride_w = _validate_positive_shape("macro_lr_stride_shape", stride_shape)
    batched = features.dim() == 3
    if not batched:
        features = features.unsqueeze(0)
    bsz, n, cdim = features.shape
    if n != h * w:
        raise ValueError(f"Expected N={h*w}, got {n}")

    tile_h = (h + stride_h - 1) // stride_h
    tile_w = (w + stride_w - 1) // stride_w
    h_pad = tile_h * stride_h
    w_pad = tile_w * stride_w
    center_y = stride_h // 2
    center_x = stride_w // 2

    feat_img = features.view(bsz, h, w, cdim).permute(0, 3, 1, 2).contiguous()
    if h_pad != h or w_pad != w:
        feat_img = F.pad(
            feat_img,
            (0, w_pad - w, 0, h_pad - h),
            mode="replicate",
        )
    pad = int(window_size // 2)
    feat_pad = F.pad(feat_img, (pad, pad, pad, pad), mode="replicate")
    sampled = feat_pad[:, :, center_y:, center_x:]
    patches = sampled.unfold(2, window_size, stride_h).unfold(3, window_size, stride_w)
    if int(patches.shape[2]) != tile_h or int(patches.shape[3]) != tile_w:
        raise ValueError(
            "Macro-stride support extraction produced an unexpected grid shape: "
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
    return bank, {
        "tile_shape": (tile_h, tile_w),
        "padded_shape": (h_pad, w_pad),
        "stride_shape": (stride_h, stride_w),
        "sample_center_offset": (center_y, center_x),
    }


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


class CosineMaskedEquivariantSpatialConv(nn.Module):
    """
    Equivariant local convolution with a cosine-similarity neighbor mask.

    The local context stays a weighted average over existing nearby orientation
    features, but neighbors whose embedding cosine to the center falls below
    the configured threshold are excluded before renormalization.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        irreps_in: Irreps | str = "1x4e",
        irreps_out: Irreps | str | None = None,
        use_residual: bool = False,
        residual_weight: float = 1.0,
        dilation: int = 1,
        feature_mask_cosine_threshold: float = 0.99,
        feature_mask_soft: bool = False,
        feature_mask_temperature: float = 32.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding = (self.kernel_size // 2) * self.dilation
        self.feature_mask_cosine_threshold = float(feature_mask_cosine_threshold)
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

    def _masked_spatial_weights(self, cosine: torch.Tensor) -> torch.Tensor:
        base_w = F.softmax(self.spatial_logits.reshape(-1), dim=0).view(
            1, 1, 1, self.kernel_size, self.kernel_size
        )
        if self.feature_mask_soft:
            mask = torch.sigmoid(
                self.feature_mask_temperature * (cosine - self.feature_mask_cosine_threshold)
            )
            mask = torch.where(self.center_mask, torch.ones_like(mask), mask)
        else:
            mask = (cosine >= self.feature_mask_cosine_threshold)
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

        dot = (patches * center).sum(dim=1)
        patch_norm = patches.pow(2).sum(dim=1).clamp_min(self.eps).sqrt()
        center_norm = feat_img.pow(2).sum(dim=1).clamp_min(self.eps).sqrt().unsqueeze(-1).unsqueeze(-1)
        cosine = (dot / (patch_norm * center_norm).clamp_min(self.eps)).clamp(-1.0, 1.0)

        masked_w = self._masked_spatial_weights(cosine)
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
            la = labels_flat.gather(1, edge_a)
            lb = labels_flat.gather(1, edge_b)
            edge_min = torch.minimum(la, lb)
            masked_min = torch.where(keep_edges, edge_min, inactive_label)

            new_labels = labels_flat.clone()
            new_labels.scatter_reduce_(1, edge_a, masked_min, reduce="amin", include_self=True)
            new_labels.scatter_reduce_(1, edge_b, masked_min, reduce="amin", include_self=True)
            if torch.equal(new_labels, labels_flat):
                break
            labels_flat = new_labels

        labels = labels_flat.view(bsz, nwin, nnode)
        if not batched:
            labels = labels.squeeze(0)
        return labels


class ClusterSlotBuilder(nn.Module):
    """Pack top-ranked clusters into deterministic slots and emit cheap metadata."""

    SLOT_TYPE_DIM = 6
    META_DIM = 11
    META_VALID = 0
    META_SLOT_TYPE_START = 1
    META_MASS = 7
    META_CENTROID_Y = 8
    META_CENTROID_X = 9
    META_SPATIAL_DISP = 10

    def __init__(self, kmax_slots: int = 6, window_size: int = 5):
        super().__init__()
        if int(kmax_slots) < 1 or int(kmax_slots) > self.SLOT_TYPE_DIM:
            raise ValueError(
                f"OCRP expects 1 <= kmax_slots <= {self.SLOT_TYPE_DIM}, got {kmax_slots}"
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

    def forward(
        self,
        cluster_ids: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batched = cluster_ids.dim() == 3
        if not batched:
            cluster_ids = cluster_ids.unsqueeze(0)
        bsz, nwin, nnode = cluster_ids.shape
        if nnode != self.num_nodes:
            raise ValueError(f"Expected cluster_ids last dim {self.num_nodes}, got {nnode}")

        coords = self.coords.to(device=cluster_ids.device, dtype=torch.float32)
        labels = cluster_ids.to(dtype=torch.long)
        labels_clamped = labels.clamp(min=0, max=self.num_nodes - 1)
        num_windows = int(bsz * nwin)
        labels_flat = labels_clamped.view(num_windows, nnode)

        # Optional valid_mask restricts which LR pixels can join a cluster
        # (used for subwindow-only clustering).
        valid_f_full: torch.Tensor | None = None
        if valid_mask is not None:
            if valid_mask.shape[-1] != nnode:
                raise ValueError(
                    f"valid_mask last dim {valid_mask.shape[-1]} disagrees with cluster_ids ({nnode})"
                )
            valid_f_full = valid_mask.to(dtype=torch.float32)
            if valid_f_full.dim() == 1:
                valid_f_full = valid_f_full.view(1, nnode).expand(num_windows, nnode)
            elif valid_f_full.dim() == 2 and valid_f_full.shape[0] == nwin:
                valid_f_full = valid_f_full.unsqueeze(0).expand(bsz, nwin, nnode).reshape(num_windows, nnode)
            elif valid_f_full.dim() == 3:
                valid_f_full = valid_f_full.reshape(num_windows, nnode)
            ones = valid_f_full.clone()
        else:
            ones = torch.ones((num_windows, nnode), device=cluster_ids.device, dtype=torch.float32)
        label_count_flat = torch.zeros(
            (num_windows, self.num_nodes),
            device=cluster_ids.device,
            dtype=torch.float32,
        )
        label_count_flat.scatter_add_(1, labels_flat, ones)
        label_count = label_count_flat.view(bsz, nwin, self.num_nodes)
        label_mass = label_count / float(self.num_nodes)

        label_coord_sum_flat = torch.zeros(
            (num_windows, self.num_nodes, 2),
            device=cluster_ids.device,
            dtype=torch.float32,
        )
        scatter_idx = labels_flat.unsqueeze(-1).expand(-1, -1, 2)
        coord_src = coords.view(1, nnode, 2).expand(num_windows, -1, -1)
        if valid_f_full is not None:
            coord_src = coord_src * valid_f_full.unsqueeze(-1)
        label_coord_sum_flat.scatter_add_(1, scatter_idx, coord_src)
        label_centroid = label_coord_sum_flat.view(bsz, nwin, self.num_nodes, 2) / label_count.clamp_min(1.0).unsqueeze(-1)
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
        # Pixels outside valid_mask must never be slot members.
        if valid_f_full is not None:
            valid_b = valid_f_full.to(torch.bool).view(bsz, nwin, 1, nnode)
            slot_mask = slot_mask & valid_b

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


class PatchSlotRouter(nn.Module):
    """Geometric router: predicts per-HR-pixel slot ownership logits from the LR
    slot-composition image (K binary slot masks over the window) plus phase."""

    def __init__(
        self,
        kmax_slots: int,
        window_size: int,
        patch_size: int | tuple[int, int] | list[int],
        phase_dim: int = 32,
        hidden_dim: int = 128,
        chunk_size: int = 512,
        use_mlp_encoder: bool = False,
        center_prior_weight: float = 0.0,
    ):
        super().__init__()
        self.kmax_slots = int(kmax_slots)
        self.window_size = int(window_size)
        self.num_nodes = self.window_size * self.window_size
        self.patch_shape = _as_patch_shape(patch_size, name="patch_size")
        self.patch_size = (
            int(self.patch_shape[0])
            if self.patch_shape[0] == self.patch_shape[1]
            else self.patch_shape
        )
        self.patch_tokens = _num_patch_tokens(self.patch_shape)
        self.phase_dim = int(phase_dim)
        self.hidden_dim = int(hidden_dim)
        self.chunk_size = max(1, int(chunk_size))
        self.use_mlp_encoder = bool(use_mlp_encoder)
        self.center_prior_weight = float(center_prior_weight)
        self.center_idx = (self.window_size // 2) * self.window_size + (self.window_size // 2)

        if self.use_mlp_encoder:
            self.encoder = None
            feat_dim = self.kmax_slots * self.num_nodes
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(self.kmax_slots, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            feat_dim = self.hidden_dim * self.num_nodes
        self.phase_proj = nn.Sequential(
            nn.Linear(self.phase_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.logit_head = nn.Sequential(
            nn.Linear(feat_dim + self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.kmax_slots),
        )

    def forward(
        self,
        slot_mask: torch.Tensor,
        phase_grid: torch.Tensor,
    ) -> torch.Tensor:
        if slot_mask.dim() != 4:
            raise ValueError(
                "slot_mask must have shape (B, N, K, num_nodes), "
                f"got {tuple(slot_mask.shape)}"
            )
        bsz, nwin, kmax, num_nodes = slot_mask.shape
        if kmax != self.kmax_slots:
            raise ValueError(f"Expected kmax_slots={self.kmax_slots}, got {kmax}")
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {num_nodes}")

        nflat = int(bsz * nwin)
        W = self.window_size
        mask_f32 = slot_mask.to(dtype=torch.float32)
        if self.use_mlp_encoder:
            x = mask_f32.reshape(nflat, self.kmax_slots * self.num_nodes)
        else:
            x = mask_f32.reshape(nflat, kmax, W, W)

        phase_emb_shared: torch.Tensor | None = None
        phase_emb_flat: torch.Tensor | None = None
        if phase_grid.dim() == 2:
            if int(phase_grid.shape[0]) != self.patch_tokens:
                raise ValueError(
                    f"Expected phase patch tokens {self.patch_tokens}, got {int(phase_grid.shape[0])}"
                )
            phase_emb_shared = self.phase_proj(phase_grid.to(dtype=torch.float32))
        elif phase_grid.dim() == 4:
            if int(phase_grid.shape[2]) != self.patch_tokens:
                raise ValueError(
                    f"Expected phase patch tokens {self.patch_tokens}, got {int(phase_grid.shape[2])}"
                )
            phase_emb_flat = self.phase_proj(phase_grid.to(dtype=torch.float32)).reshape(
                nflat, self.patch_tokens, self.hidden_dim
            )
        else:
            raise ValueError(
                "phase_grid must have shape (T,D_phase) or (B,N,T,D_phase), "
                f"got {tuple(phase_grid.shape)}"
            )

        logits_flat = torch.empty(
            (nflat, self.patch_tokens, self.kmax_slots),
            device=slot_mask.device,
            dtype=x.dtype,
        )
        for start in range(0, nflat, self.chunk_size):
            end = min(start + self.chunk_size, nflat)
            if self.use_mlp_encoder:
                feat_pool = x[start:end]
            else:
                feat = self.encoder(x[start:end])
                feat_pool = feat.reshape(end - start, -1)
            if phase_emb_shared is not None:
                phase_chunk = phase_emb_shared.unsqueeze(0).expand(end - start, -1, -1)
            else:
                phase_chunk = phase_emb_flat[start:end]
            feat_exp = feat_pool.unsqueeze(1).expand(-1, self.patch_tokens, -1)
            combined = torch.cat([feat_exp, phase_chunk], dim=-1)
            logits_flat[start:end] = self.logit_head(combined)

        logits = logits_flat.view(bsz, nwin, self.patch_tokens, self.kmax_slots)
        if self.center_prior_weight != 0.0:
            center_slot = mask_f32.view(bsz, nwin, kmax, self.num_nodes)[..., self.center_idx]
            logits = logits + self.center_prior_weight * center_slot.unsqueeze(2)
        slot_present = slot_mask.to(dtype=torch.float32).sum(dim=-1) > 0
        logits = logits.masked_fill(~slot_present.unsqueeze(2), -1e4)
        return logits


class DirectHRMemberCrossAttn(nn.Module):
    """Anchorless cross-attention from HR tokens (queries) to LR slot members
    (keys/values). For each (slot k, HR token t), attention weights over slot
    members are computed from invariants (LR position, member feature norms,
    HR phase, slot meta); the HR-token feature is the attention-weighted sum
    of member irrep features. No slot anchor is computed."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        meta_dim: int,
        phase_dim: int,
        window_size: int,
        patch_size: int | tuple[int, int] | list[int],
        hidden_dim: int = 128,
        attn_dim: int = 64,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feature_dim = int(self.irreps_feat.dim)
        self.summary = InvariantSlotSummary(self.irreps_feat)
        self.meta_dim = int(meta_dim)
        self.phase_dim = int(phase_dim)
        self.window_size = int(window_size)
        self.num_nodes = self.window_size * self.window_size
        self.patch_shape = _as_patch_shape(patch_size, name="patch_size")
        self.patch_tokens = _num_patch_tokens(self.patch_shape)
        self.hidden_dim = int(hidden_dim)
        self.attn_dim = int(attn_dim)
        self.chunk_size = max(1, int(chunk_size))

        den = float(max(1, self.window_size // 2))
        coords = []
        for y in range(self.window_size):
            for x in range(self.window_size):
                coords.append(((float(y) - den) / den, (float(x) - den) / den))
        self.register_buffer(
            "coords",
            torch.tensor(coords, dtype=torch.float32),
            persistent=False,
        )

        key_in = 2 + int(self.summary.out_dim)
        self.key_mlp = nn.Sequential(
            nn.Linear(key_in, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.attn_dim),
        )
        q_in = self.phase_dim + self.meta_dim
        self.query_mlp = nn.Sequential(
            nn.Linear(q_in, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.attn_dim),
        )

    def forward(
        self,
        bank_f: torch.Tensor,
        slot_mask: torch.Tensor,
        slot_meta: torch.Tensor,
        phase_grid: torch.Tensor,
        return_alpha: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, nwin, nnode, cdim = bank_f.shape
        kmax = slot_meta.shape[2]
        T = self.patch_tokens
        if nnode != self.num_nodes:
            raise ValueError(
                f"bank_f num_nodes mismatch: {nnode} vs {self.num_nodes}"
            )
        if slot_mask.shape[:3] != slot_meta.shape[:3]:
            raise ValueError("slot_mask and slot_meta must agree on (B, N, K)")
        if slot_mask.shape[-1] != nnode:
            raise ValueError("slot_mask and bank_f disagree on bank size")

        member_summary = self.summary.summarize(bank_f)
        coord_feat = self.coords.to(device=bank_f.device, dtype=bank_f.dtype)
        coord_feat = coord_feat.view(1, 1, self.num_nodes, 2).expand(
            bsz, nwin, -1, -1
        )
        key_in = torch.cat([coord_feat, member_summary], dim=-1)
        keys = self.key_mlp(key_in)

        if phase_grid.dim() == 2:
            if int(phase_grid.shape[0]) != T:
                raise ValueError(
                    f"Expected phase patch tokens {T}, got {int(phase_grid.shape[0])}"
                )
            phase = (
                phase_grid.to(dtype=bank_f.dtype)
                .view(1, 1, 1, T, self.phase_dim)
                .expand(bsz, nwin, kmax, -1, -1)
            )
        elif phase_grid.dim() == 4:
            if int(phase_grid.shape[2]) != T:
                raise ValueError(
                    f"Expected phase patch tokens {T}, got {int(phase_grid.shape[2])}"
                )
            phase = (
                phase_grid.to(dtype=bank_f.dtype)
                .unsqueeze(2)
                .expand(-1, -1, kmax, -1, -1)
            )
        else:
            raise ValueError(
                "phase_grid must have shape (T,D_phase) or (B,N,T,D_phase), "
                f"got {tuple(phase_grid.shape)}"
            )
        meta_exp = slot_meta.to(dtype=bank_f.dtype).unsqueeze(3).expand(
            -1, -1, -1, T, -1
        )
        q_in = torch.cat([phase, meta_exp], dim=-1)
        queries = self.query_mlp(q_in)

        nflat = int(bsz * nwin)
        keys_flat = keys.reshape(nflat, nnode, self.attn_dim)
        queries_flat = queries.reshape(nflat, kmax, T, self.attn_dim)
        mask_flat = slot_mask.reshape(nflat, kmax, nnode).bool()
        bank_flat = bank_f.reshape(nflat, nnode, cdim)

        out_flat = torch.empty(
            (nflat, kmax, T, cdim),
            device=bank_f.device,
            dtype=bank_f.dtype,
        )
        alpha_flat: torch.Tensor | None = None
        if return_alpha:
            alpha_flat = torch.zeros(
                (nflat, kmax, T, nnode),
                device=bank_f.device,
                dtype=bank_f.dtype,
            )

        scale = float(self.attn_dim) ** -0.5
        for start in range(0, nflat, self.chunk_size):
            end = min(start + self.chunk_size, nflat)
            q_chunk = queries_flat[start:end]
            k_chunk = keys_flat[start:end]
            mask_chunk = mask_flat[start:end]
            bank_chunk = bank_flat[start:end]
            scores = torch.einsum("nktd,nmd->nktm", q_chunk, k_chunk) * scale
            mask_exp = mask_chunk.unsqueeze(2).expand(-1, -1, T, -1)
            slot_has = mask_chunk.any(dim=-1)
            scores = scores.masked_fill(~mask_exp, float("-inf"))
            scores = torch.where(
                slot_has.unsqueeze(-1).unsqueeze(-1),
                scores,
                torch.zeros_like(scores),
            )
            alpha = F.softmax(scores, dim=-1)
            alpha = alpha * slot_has.unsqueeze(-1).unsqueeze(-1).to(dtype=alpha.dtype)
            out_flat[start:end] = torch.einsum(
                "nktm,nmc->nktc", alpha.to(dtype=bank_f.dtype), bank_chunk
            )
            if alpha_flat is not None:
                alpha_flat[start:end] = alpha.to(dtype=bank_f.dtype)

        out = out_flat.view(bsz, nwin, kmax, T, cdim)
        alpha_out = (
            alpha_flat.view(bsz, nwin, kmax, T, nnode)
            if alpha_flat is not None
            else None
        )
        return out, alpha_out


class OCRP4x1PatchUpsampler(nn.Module):
    """4x1-specialized OCRP upsampler with decoupled macro support and stride."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        sym_ops_quat: torch.Tensor,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        window_size: int = 5,
        kmax_slots: int = 6,
        cluster_threshold_deg: float = 2.0,
        cluster_connectivity: int = 8,
        cluster_window_size: int | None = None,
        phase_dim: int = 32,
        router_hidden_dim: int = 128,
        router_conv_hidden_dim: int = 64,
        router_slot_mass_power: float = 0.25,
        router_uniform_slot_mix: float = 0.75,
        proposal_hidden_dim: int = 128,
        proposal_query_residual_scale: float = 0.5,
        proposal_query_per_token_weight: float = 0.0,
        proposal_mode: str = "tp",
        proposal_tconv_transpose_overlap: int = 2,
        proposal_tconv_mass_threshold: float = 1e-3,
        proposal_tconv_chunk_size: int = 1024,
        proposal_tconv_trainable_kernel: bool = True,
        straight_through: bool = True,
        ocrp_mode: str = "pixel_patch",
        macro_lr_tile_size: int = 3,
        macro_lr_stride_shape: int | tuple[int, int] | list[int] | None = (1, 4),
        token_conditioned_member_bias: bool | None = None,
        pool_center_bias_init: float | None = None,
        pool_chunk_size: int = 512,
        router_chunk_size: int = 512,
        router_use_mlp_encoder: bool = False,
        router_center_prior_weight: float = 0.0,
        router_mode: str = "geometric",
        router_use_raw_token_ctx: bool = False,
        proposal_chunk_size: int = 128,
        proposal_token_chunk_size: int | None = None,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feature_dim = int(self.irreps_feat.dim)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.window_size = int(window_size)
        self.kmax_slots = int(kmax_slots)
        # Cluster (sub)window: clustering only over a central cw x cw subwindow.
        cw = int(cluster_window_size) if cluster_window_size is not None else self.window_size
        if cw < 1 or cw > self.window_size or (cw % 2 == 0 and cw != self.window_size):
            raise ValueError(
                f"cluster_window_size must be an odd integer in [1, {self.window_size}] "
                f"(or None / window_size), got {cluster_window_size}"
            )
        self.cluster_window_size = cw
        cluster_valid_mask = torch.zeros(self.window_size * self.window_size, dtype=torch.bool)
        if cw == self.window_size:
            cluster_valid_mask[:] = True
        else:
            half_full = self.window_size // 2
            half_sub = cw // 2
            for y in range(self.window_size):
                for x in range(self.window_size):
                    if abs(y - half_full) <= half_sub and abs(x - half_full) <= half_sub:
                        cluster_valid_mask[y * self.window_size + x] = True
        self.register_buffer("cluster_valid_mask", cluster_valid_mask, persistent=False)
        self.straight_through = bool(straight_through)
        self.ocrp_mode = _resolve_ocrp_mode(ocrp_mode)
        self.macro_lr_stride_shape = _validate_positive_shape(
            "macro_lr_stride_shape",
            macro_lr_stride_shape if macro_lr_stride_shape is not None else macro_lr_tile_size,
        )
        self.macro_lr_tile_size = (
            int(self.macro_lr_stride_shape[0])
            if self.macro_lr_stride_shape[0] == self.macro_lr_stride_shape[1]
            else self.macro_lr_stride_shape
        )
        self.hr_patch_shape = (
            self.upsample_factor
            if self.ocrp_mode == "pixel_patch"
            else (
                int(self.macro_lr_stride_shape[0] * self.upsample_factor[0]),
                int(self.macro_lr_stride_shape[1] * self.upsample_factor[1]),
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

        self.phase_embed = PhaseEmbeddingGrid(
            upsample_factor=self.upsample_factor,
            emb_dim=int(phase_dim),
            patch_size=self.hr_patch_shape,
        )
        self.clusterer = QuaternionBankClusterer(
            sym_ops_quat=sym_ops_quat,
            threshold_deg=float(cluster_threshold_deg),
            connectivity=int(cluster_connectivity),
            window_size=int(window_size),
        )
        self.slot_builder = ClusterSlotBuilder(
            kmax_slots=int(kmax_slots),
            window_size=int(window_size),
        )
        # Anchorless: no context_builder, no slot_pool, no proposal_head.
        # A single DirectHRMemberCrossAttn computes HR-token features by
        # attending each HR token to its slot's LR members directly.
        self.context_builder = None
        self.slot_pool = None
        self.proposal_head = None
        self.proposal_module = None
        self.proposal_mode = "xattn"
        self.router_mode = "geometric"
        if router_mode != "geometric":
            raise ValueError(
                f"SR_4x1_ocrp_anchorless only supports router_mode='geometric', got {router_mode!r}"
            )

        self.router = PatchSlotRouter(
            kmax_slots=int(kmax_slots),
            window_size=int(window_size),
            patch_size=self.hr_patch_shape,
            phase_dim=int(phase_dim),
            hidden_dim=int(router_hidden_dim),
            chunk_size=int(router_chunk_size),
            use_mlp_encoder=bool(router_use_mlp_encoder),
            center_prior_weight=float(router_center_prior_weight),
        )

        self.xattn = DirectHRMemberCrossAttn(
            irreps_feat=self.irreps_feat,
            meta_dim=ClusterSlotBuilder.META_DIM,
            phase_dim=int(phase_dim),
            window_size=int(window_size),
            patch_size=self.hr_patch_shape,
            hidden_dim=int(proposal_hidden_dim),
            attn_dim=max(32, int(proposal_hidden_dim) // 2),
            chunk_size=int(pool_chunk_size),
        )

    def _phase_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        phase_ids = torch.arange(
            self.hr_patch_tokens,
            device=device,
            dtype=torch.long,
        )
        return self.phase_embed(phase_ids).to(dtype=dtype)

    def _build_support_banks(
        self,
        lr_quats: torch.Tensor,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, int]]]:
        if self.ocrp_mode == "pixel_patch":
            bank_q = _build_local_patch_bank(lr_quats, img_shape=lr_shape, window_size=self.window_size)
            bank_f = _build_local_patch_bank(feat_lr, img_shape=lr_shape, window_size=self.window_size)
            support_meta = {
                "grid_shape": (int(lr_shape[0]), int(lr_shape[1])),
                "padded_shape": (int(lr_shape[0]), int(lr_shape[1])),
            }
            return bank_q, bank_f, support_meta

        bank_q, meta_q = _build_macro_stride_patch_bank(
            lr_quats,
            img_shape=lr_shape,
            window_size=self.window_size,
            stride_shape=self.macro_lr_stride_shape,
        )
        bank_f, meta_f = _build_macro_stride_patch_bank(
            feat_lr,
            img_shape=lr_shape,
            window_size=self.window_size,
            stride_shape=self.macro_lr_stride_shape,
        )
        if meta_q != meta_f:
            raise ValueError(
                "Quaternion and feature macro-tile banks disagree on support metadata: "
                f"{meta_q} vs {meta_f}"
            )
        return bank_q, bank_f, {
            "grid_shape": meta_q["tile_shape"],
            "padded_shape": meta_q["padded_shape"],
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
        )

        cluster_ids = self.clusterer(bank_q)
        slot_info = self.slot_builder(
            cluster_ids,
            valid_mask=(
                self.cluster_valid_mask
                if self.cluster_window_size != self.window_size
                else None
            ),
        )
        slot_mask = slot_info["slot_mask"]
        slot_meta = slot_info["slot_meta"]
        slot_valid = slot_info["slot_valid"]

        phase_grid = self._phase_grid(device=feat_lr.device, dtype=feat_lr.dtype)

        router_logits = self.router(
            slot_mask=slot_mask,
            phase_grid=phase_grid,
        )
        owner_idx, owner_onehot = self._hard_owner_from_logits(
            router_logits,
            straight_through=self.straight_through and self.training,
        )

        patch_prop, xattn_alpha = self.xattn(
            bank_f=bank_f,
            slot_mask=slot_mask,
            slot_meta=slot_meta,
            phase_grid=phase_grid,
            return_alpha=return_aux,
        )
        slot_ctx = None
        slot_anchor_alpha = None
        slot_pooled_ctx = None
        slot_pool_alpha = None
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
            "owner_idx": owner_idx,
            "patch_prop": patch_prop,
            "patch_out": patch_out,
            "ocrp_mode": self.ocrp_mode,
            "macro_lr_tile_size": self.macro_lr_tile_size,
            "macro_lr_stride_shape": self.macro_lr_stride_shape,
            "hr_patch_size": self.hr_patch_size,
            "hr_patch_shape": self.hr_patch_shape,
            "hr_patch_tokens": self.hr_patch_tokens,
            "pool_chunk_size": self.xattn.chunk_size,
            "router_chunk_size": self.router.chunk_size,
            "proposal_chunk_size": self.xattn.chunk_size,
            "proposal_token_chunk_size": None,
            "proposal_mode": self.proposal_mode,
            "proposal_tconv_chunk_size": None,
            "xattn_alpha": xattn_alpha,
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


class IsoEmbedding4x1SROCRP(nn.Module):
    """4x1-specialized OCRP SR model: encoder -> OCRP upsampler -> decoder."""

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
        conv_feature_mask_cosine_threshold: float = 0.98,
        conv_feature_mask_soft: bool = False,
        conv_feature_mask_temperature: float = 32.0,
        upsample_factor: int | tuple[int, int] | list[int] = (4, 1),
        window_size: int = 5,
        kmax_slots: int = 6,
        cluster_threshold_deg: float = 2.0,
        cluster_connectivity: int = 8,
        cluster_window_size: int | None = None,
        phase_dim: int = 32,
        ocrp_router_hidden_dim: int = 128,
        ocrp_router_conv_hidden_dim: int = 64,
        ocrp_router_slot_mass_power: float = 0.25,
        ocrp_router_uniform_slot_mix: float = 0.75,
        ocrp_proposal_hidden_dim: int = 128,
        ocrp_proposal_query_residual_scale: float = 0.5,
        ocrp_proposal_query_per_token_weight: float = 0.0,
        ocrp_proposal_mode: str = "tp",
        ocrp_proposal_tconv_transpose_overlap: int = 2,
        ocrp_proposal_tconv_mass_threshold: float = 1e-3,
        ocrp_proposal_tconv_chunk_size: int = 1024,
        ocrp_proposal_tconv_trainable_kernel: bool = True,
        ocrp_straight_through: bool = True,
        ocrp_mode: str = "pixel_patch",
        macro_lr_tile_size: int = 4,
        macro_lr_stride_shape: int | tuple[int, int] | list[int] | None = (1, 4),
        ocrp_token_conditioned_member_bias: bool | None = None,
        ocrp_pool_center_bias_init: float | None = None,
        ocrp_pool_chunk_size: int = 512,
        ocrp_router_chunk_size: int = 512,
        ocrp_router_use_mlp_encoder: bool = False,
        ocrp_router_center_prior_weight: float = 0.0,
        ocrp_router_mode: str = "geometric",
        ocrp_router_use_raw_token_ctx: bool = False,
        ocrp_proposal_chunk_size: int = 128,
        ocrp_proposal_token_chunk_size: int | None = None,
        use_hr_conv1: bool = True,
        hr_conv1_kernel_size: int = 7,
        use_residual_hr1: bool = True,
        hr_conv1_residual_weight: float = 1.0,
        hr_conv_feature_mask_cosine_threshold: float | None = None,
        hr_conv_feature_mask_soft: bool | None = None,
        hr_conv_feature_mask_temperature: float | None = None,
        use_hr_conv2: bool = False,
        hr_conv2_kernel_size: int | None = None,
        use_residual_hr2: bool = True,
        hr_conv2_residual_weight: float = 1.0,
        hr_conv2_feature_mask_cosine_threshold: float | None = None,
        hr_conv2_feature_mask_soft: bool | None = None,
        hr_conv2_feature_mask_temperature: float | None = None,
        use_hr_conv3: bool = False,
        hr_conv3_kernel_size: int | None = None,
        use_residual_hr3: bool = True,
        hr_conv3_residual_weight: float = 1.0,
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
        self.lr_conv_feature_mask_cosine_threshold = float(conv_feature_mask_cosine_threshold)
        self.lr_conv_feature_mask_soft = bool(conv_feature_mask_soft)
        self.lr_conv_feature_mask_temperature = float(conv_feature_mask_temperature)
        self.hr_conv_feature_mask_cosine_threshold = float(
            self.lr_conv_feature_mask_cosine_threshold
            if hr_conv_feature_mask_cosine_threshold is None
            else hr_conv_feature_mask_cosine_threshold
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
        self.hr_conv2_feature_mask_cosine_threshold = float(
            self.hr_conv_feature_mask_cosine_threshold
            if hr_conv2_feature_mask_cosine_threshold is None
            else hr_conv2_feature_mask_cosine_threshold
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
        self.hr_conv3_feature_mask_cosine_threshold = float(
            self.hr_conv_feature_mask_cosine_threshold
            if hr_conv3_feature_mask_cosine_threshold is None
            else hr_conv3_feature_mask_cosine_threshold
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
        self.conv_lr1 = CosineMaskedEquivariantSpatialConv(
            kernel_size=int(lr_conv1_kernel_size),
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=self.use_residual_lr1,
            residual_weight=self.lr_conv1_residual_weight,
            feature_mask_cosine_threshold=self.lr_conv_feature_mask_cosine_threshold,
            feature_mask_soft=self.lr_conv_feature_mask_soft,
            feature_mask_temperature=self.lr_conv_feature_mask_temperature,
        )
        self.ocrp = OCRP4x1PatchUpsampler(
            irreps_feat=self.irreps_feat,
            sym_ops_quat=self.encoder.sym_ops,
            upsample_factor=self.upsample_factor,
            window_size=int(window_size),
            kmax_slots=int(kmax_slots),
            cluster_threshold_deg=float(cluster_threshold_deg),
            cluster_connectivity=int(cluster_connectivity),
            cluster_window_size=cluster_window_size,
            phase_dim=int(phase_dim),
            router_hidden_dim=int(ocrp_router_hidden_dim),
            router_conv_hidden_dim=int(ocrp_router_conv_hidden_dim),
            router_slot_mass_power=float(ocrp_router_slot_mass_power),
            router_uniform_slot_mix=float(ocrp_router_uniform_slot_mix),
            proposal_hidden_dim=int(ocrp_proposal_hidden_dim),
            proposal_query_residual_scale=float(ocrp_proposal_query_residual_scale),
            proposal_query_per_token_weight=float(ocrp_proposal_query_per_token_weight),
            proposal_mode=str(ocrp_proposal_mode),
            proposal_tconv_transpose_overlap=int(ocrp_proposal_tconv_transpose_overlap),
            proposal_tconv_mass_threshold=float(ocrp_proposal_tconv_mass_threshold),
            proposal_tconv_chunk_size=int(ocrp_proposal_tconv_chunk_size),
            proposal_tconv_trainable_kernel=bool(ocrp_proposal_tconv_trainable_kernel),
            straight_through=bool(ocrp_straight_through),
            ocrp_mode=str(ocrp_mode),
            macro_lr_tile_size=int(macro_lr_tile_size),
            macro_lr_stride_shape=macro_lr_stride_shape,
            token_conditioned_member_bias=ocrp_token_conditioned_member_bias,
            pool_center_bias_init=ocrp_pool_center_bias_init,
            pool_chunk_size=int(ocrp_pool_chunk_size),
            router_chunk_size=int(ocrp_router_chunk_size),
            router_use_mlp_encoder=bool(ocrp_router_use_mlp_encoder),
            router_center_prior_weight=float(ocrp_router_center_prior_weight),
            router_mode=str(ocrp_router_mode),
            router_use_raw_token_ctx=bool(ocrp_router_use_raw_token_ctx),
            proposal_chunk_size=int(ocrp_proposal_chunk_size),
            proposal_token_chunk_size=ocrp_proposal_token_chunk_size,
        )
        self.conv_hr1 = CosineMaskedEquivariantSpatialConv(
            kernel_size=int(hr_conv1_kernel_size),
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=self.use_residual_hr1,
            residual_weight=self.hr_conv1_residual_weight,
            feature_mask_cosine_threshold=self.hr_conv_feature_mask_cosine_threshold,
            feature_mask_soft=self.hr_conv_feature_mask_soft,
            feature_mask_temperature=self.hr_conv_feature_mask_temperature,
        )
        self.conv_hr2 = CosineMaskedEquivariantSpatialConv(
            kernel_size=self.hr_conv2_kernel_size,
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=self.use_residual_hr2,
            residual_weight=self.hr_conv2_residual_weight,
            feature_mask_cosine_threshold=self.hr_conv2_feature_mask_cosine_threshold,
            feature_mask_soft=self.hr_conv2_feature_mask_soft,
            feature_mask_temperature=self.hr_conv2_feature_mask_temperature,
        )
        self.conv_hr3: CosineMaskedEquivariantSpatialConv | None = None
        if self.use_hr_conv3:
            self.conv_hr3 = CosineMaskedEquivariantSpatialConv(
                kernel_size=self.hr_conv3_kernel_size,
                irreps_in=self.irreps_feat,
                irreps_out=self.irreps_feat,
                use_residual=self.use_residual_hr3,
                residual_weight=self.hr_conv3_residual_weight,
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
                    "Ignoring legacy conv_hr3 weights because use_hr_conv3=False for this 4x1 OCRP model.",
                    RuntimeWarning,
                )
        return filtered_state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        filtered_state_dict = self._filter_incompatible_state_dict(state_dict)
        return super().load_state_dict(filtered_state_dict, strict=strict, assign=assign)

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

        aux["feat_lr_encode"] = feat_lr
        aux["feat_lr_pre_ocrp"] = feat_pre
        aux["feat_hr_raw_ocrp"] = feat_hr_raw_ocrp
        aux["feat_hr_post_hr_conv1"] = feat_hr_after_conv1
        aux["feat_hr_post_hr_conv2"] = feat_hr_after_conv2
        aux["feat_hr_post_hr_conv"] = feat_hr

        probe_stages: list[dict[str, object]] = []
        probe_stages.append({"name": "encode_lr", "feat": feat_lr.detach(), "shape": tuple(lr_shape)})
        if self.use_lr_conv1:
            probe_stages.append({"name": "lr_conv1_post", "feat": feat_pre.detach(), "shape": tuple(lr_shape)})
        probe_stages.append({"name": "ocrp_hr_raw", "feat": feat_hr_raw_ocrp.detach(), "shape": tuple(hr_shape)})
        if self.use_hr_conv1:
            probe_stages.append({"name": "hr_conv1_post", "feat": feat_hr_after_conv1.detach(), "shape": tuple(hr_shape)})
        if self.use_hr_conv2:
            probe_stages.append({"name": "hr_conv2_post", "feat": feat_hr_after_conv2.detach(), "shape": tuple(hr_shape)})
        if self.conv_hr3 is not None:
            probe_stages.append({"name": "hr_conv3_post", "feat": feat_hr.detach(), "shape": tuple(hr_shape)})
        aux["probe_stages"] = probe_stages
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

    def feature_loss_sr(
        self,
        lr_quats: torch.Tensor,
        hr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = True,
    ) -> torch.Tensor:
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

        feat_hr_pred, _ = self._forward_sr_features(
            lr_quats=lr_q_batched,
            feat_lr=feat_lr,
            lr_shape=lr_shape,
            return_aux=False,
        )
        return F.mse_loss(feat_hr_pred, feat_hr_tgt)


class OCRPFromSROCRPPatchUpsampler(OCRP4x1PatchUpsampler):
    """Scale-parametrized wrapper around the anchorless OCRP implementation."""

    def __init__(self, *args, upsample_factor=(4, 4), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


class IsoEmbeddingFromSROCRP(IsoEmbedding4x1SROCRP):
    """Scale-parametrized anchorless OCRP model.

    The architecture and hyperparameters are inherited from the validated
    anchorless OCRP implementation; the spatial scale is supplied by
    ``upsample_factor`` in the experiment config.
    """

    def __init__(self, *args, upsample_factor=(4, 4), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


class OCRP2x2FromSROCRPPatchUpsampler(OCRPFromSROCRPPatchUpsampler):
    """Trivial 2x2 default wrapper for readable experiment configs."""

    def __init__(self, *args, upsample_factor=(2, 2), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


class IsoEmbedding2x2FromSROCRP(IsoEmbeddingFromSROCRP):
    """Trivial 2x2 default wrapper for readable experiment configs."""

    def __init__(self, *args, upsample_factor=(2, 2), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


class OCRP4x4FromSROCRPPatchUpsampler(OCRPFromSROCRPPatchUpsampler):
    """Trivial 4x4 default wrapper for readable experiment configs."""

    def __init__(self, *args, upsample_factor=(4, 4), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


class IsoEmbedding4x4FromSROCRP(IsoEmbeddingFromSROCRP):
    """Trivial 4x4 default wrapper for readable experiment configs."""

    def __init__(self, *args, upsample_factor=(4, 4), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


class OCRP8x8FromSROCRPPatchUpsampler(OCRPFromSROCRPPatchUpsampler):
    """Trivial 8x8 default wrapper for readable experiment configs."""

    def __init__(self, *args, upsample_factor=(8, 8), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


class IsoEmbedding8x8FromSROCRP(IsoEmbeddingFromSROCRP):
    """Trivial 8x8 default wrapper for readable experiment configs."""

    def __init__(self, *args, upsample_factor=(8, 8), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


def _with_upsample_default(fn, default):
    sig = inspect.signature(fn)
    params = [
        param.replace(default=default) if name == "upsample_factor" else param
        for name, param in sig.parameters.items()
    ]
    return sig.replace(parameters=params)


OCRPFromSROCRPPatchUpsampler.__init__.__signature__ = _with_upsample_default(
    OCRP4x1PatchUpsampler.__init__, (4, 4)
)
IsoEmbeddingFromSROCRP.__init__.__signature__ = _with_upsample_default(
    IsoEmbedding4x1SROCRP.__init__, (4, 4)
)
OCRP2x2FromSROCRPPatchUpsampler.__init__.__signature__ = _with_upsample_default(
    OCRP4x1PatchUpsampler.__init__, (2, 2)
)
IsoEmbedding2x2FromSROCRP.__init__.__signature__ = _with_upsample_default(
    IsoEmbedding4x1SROCRP.__init__, (2, 2)
)
OCRP4x4FromSROCRPPatchUpsampler.__init__.__signature__ = _with_upsample_default(
    OCRP4x1PatchUpsampler.__init__, (4, 4)
)
IsoEmbedding4x4FromSROCRP.__init__.__signature__ = _with_upsample_default(
    IsoEmbedding4x1SROCRP.__init__, (4, 4)
)
OCRP8x8FromSROCRPPatchUpsampler.__init__.__signature__ = _with_upsample_default(
    OCRP4x1PatchUpsampler.__init__, (8, 8)
)
IsoEmbedding8x8FromSROCRP.__init__.__signature__ = _with_upsample_default(
    IsoEmbedding4x1SROCRP.__init__, (8, 8)
)

# Backwards-compatible names used by existing 4x4 configs and logs.
OCRP4x4From4x1PatchUpsampler = OCRP4x4FromSROCRPPatchUpsampler
IsoEmbedding4x4From4x1SROCRP = IsoEmbedding4x4FromSROCRP

OCRPPatchUpsampler = OCRPFromSROCRPPatchUpsampler
IsoEmbeddingSROCRP = IsoEmbeddingFromSROCRP


__all__ = [
    "LocalIsoCrystalEncoder",
    "CubochoricOptimizingLocalIsoDecoder",
    "CosineMaskedEquivariantSpatialConv",
    "PhaseEmbeddingGrid",
    "QuaternionBankClusterer",
    "ClusterSlotBuilder",
    "InvariantSlotSummary",
    "PatchSlotRouter",
    "DirectHRMemberCrossAttn",
    "OCRP4x1PatchUpsampler",
    "IsoEmbedding4x1SROCRP",
    "OCRPFromSROCRPPatchUpsampler",
    "IsoEmbeddingFromSROCRP",
    "OCRP2x2FromSROCRPPatchUpsampler",
    "IsoEmbedding2x2FromSROCRP",
    "OCRP4x4FromSROCRPPatchUpsampler",
    "IsoEmbedding4x4FromSROCRP",
    "OCRP8x8FromSROCRPPatchUpsampler",
    "IsoEmbedding8x8FromSROCRP",
    "OCRP4x4From4x1PatchUpsampler",
    "IsoEmbedding4x4From4x1SROCRP",
    "OCRPPatchUpsampler",
    "IsoEmbeddingSROCRP",
]
