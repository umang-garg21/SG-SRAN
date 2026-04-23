from __future__ import annotations

import hashlib
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


def _as_isotropic_scale(scale: int | tuple[int, int] | list[int]) -> int:
    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError(f"Expected isotropic scale as int or length-2 tuple/list, got {scale}")
        scale_y = int(scale[0])
        scale_x = int(scale[1])
        if scale_y != scale_x:
            raise ValueError(f"OCRP currently assumes isotropic SR, got scale={scale}")
        scale = scale_y
    scale_int = int(scale)
    if scale_int < 1:
        raise ValueError(f"scale must be >= 1, got {scale_int}")
    return scale_int


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
                out = out + feat_flat
            else:
                out = out + self.residual_proj(feat_flat)

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
    """Learned phase embedding for subpixel positions inside an LR cell."""

    def __init__(self, upsample_factor: int | tuple[int, int] | list[int] = 4, emb_dim: int = 32):
        super().__init__()
        self.upsample_factor = _as_isotropic_scale(upsample_factor)
        self.num_phases = int(self.upsample_factor * self.upsample_factor)
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
        connectivity: int = 4,
        window_size: int = 5,
    ):
        super().__init__()
        if int(window_size) < 3 or int(window_size) % 2 == 0:
            raise ValueError(f"OCRP expects an odd window_size >= 3, got {window_size}")
        if int(connectivity) != 4:
            raise ValueError(f"OCRP currently expects 4-neighbor clustering, got {connectivity}")
        self.threshold_rad = float(np.deg2rad(float(threshold_deg)))
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
    """Pack clusters into parent / alt1 / alt2 / null slots and emit cheap metadata."""

    SLOT_TYPE_DIM = 4
    META_DIM = 9
    META_VALID = 0
    META_SLOT_TYPE_START = 1
    META_MASS = 5
    META_CENTROID_Y = 6
    META_CENTROID_X = 7
    META_SPATIAL_DISP = 8

    def __init__(self, kmax_slots: int = 4, window_size: int = 5):
        super().__init__()
        if int(kmax_slots) != 4:
            raise ValueError(f"OCRP currently expects kmax_slots=4, got {kmax_slots}")
        if int(window_size) < 3 or int(window_size) % 2 == 0:
            raise ValueError(f"OCRP expects an odd window_size >= 3, got {window_size}")
        self.kmax_slots = int(kmax_slots)
        self.window_size = int(window_size)
        self.num_nodes = int(self.window_size * self.window_size)
        self.parent_bank_index = int(self.num_nodes // 2)

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

        parent_label = labels[..., self.parent_bank_index]
        parent_mass = torch.gather(label_mass, 2, parent_label.unsqueeze(-1))

        nonparent_mass = label_mass.clone()
        nonparent_mass.scatter_(2, parent_label.unsqueeze(-1), -1.0)
        top2_mass, top2_label = torch.topk(nonparent_mass, k=2, dim=2)

        slot_cluster_label = torch.full(
            (bsz, nwin, self.kmax_slots),
            -1,
            device=cluster_ids.device,
            dtype=torch.long,
        )
        slot_cluster_label[..., 0] = parent_label
        slot_cluster_label[..., 1:3] = torch.where(
            top2_mass > 0.0,
            top2_label,
            torch.full_like(top2_label, -1),
        )

        slot_valid = (slot_cluster_label >= 0).to(torch.float32)
        slot_mask = labels.unsqueeze(2) == slot_cluster_label.clamp_min(0).unsqueeze(-1)
        slot_mask = slot_mask & slot_valid.bool().unsqueeze(-1)

        coords = self.coords.to(device=cluster_ids.device, dtype=torch.float32)
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
        null_type = torch.tensor([0.0, 0.0, 0.0, 1.0], device=cluster_ids.device, dtype=torch.float32)
        slot_meta[..., self.META_SLOT_TYPE_START : self.META_SLOT_TYPE_START + self.SLOT_TYPE_DIM] = null_type
        slot_type_eye = torch.eye(self.SLOT_TYPE_DIM, device=cluster_ids.device, dtype=torch.float32)
        for slot_idx in range(self.kmax_slots - 1):
            valid = slot_valid[..., slot_idx].bool().unsqueeze(-1)
            slot_meta[..., slot_idx, self.META_SLOT_TYPE_START : self.META_SLOT_TYPE_START + self.SLOT_TYPE_DIM] = torch.where(
                valid,
                slot_type_eye[slot_idx].view(1, 1, -1),
                null_type.view(1, 1, -1),
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


class MedoidSlotContextBuilder(nn.Module):
    """Build one representative equivariant slot context per slot using medoid selection."""

    def __init__(self, sym_ops_quat: torch.Tensor, chunk_size: int = 1024):
        super().__init__()
        sym_ops_mat = _left_mult_matrix_wxyz_batch(_normalize_quaternions(sym_ops_quat.detach().cpu()))
        self.register_buffer("sym_ops_mat", sym_ops_mat, persistent=False)
        self.chunk_size = int(chunk_size)

    def forward(
        self,
        bank_q: torch.Tensor,
        bank_f: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batched = bank_q.dim() == 4
        if not batched:
            bank_q = bank_q.unsqueeze(0)
            bank_f = bank_f.unsqueeze(0)
            slot_mask = slot_mask.unsqueeze(0)
        bsz, nwin, nnode, qdim = bank_q.shape
        _, _, _, fdim = bank_f.shape
        kmax = int(slot_mask.shape[2])
        if qdim != 4:
            raise ValueError(f"Expected bank_q last dim 4, got {qdim}")
        if slot_mask.shape[-1] != nnode:
            raise ValueError("slot_mask and bank_q disagree on bank size")

        q = _normalize_quaternions(bank_q.to(dtype=torch.float32))
        sym_ops = self.sym_ops_mat.to(device=bank_f.device, dtype=torch.float32)
        flat_q = q.reshape(bsz * nwin, nnode, qdim)
        flat_f = bank_f.reshape(bsz * nwin, nnode, fdim)
        flat_mask = slot_mask.reshape(bsz * nwin, kmax, nnode)

        medoid_flat = torch.full((bsz * nwin, kmax), -1, device=bank_f.device, dtype=torch.long)
        slot_ctx_flat = torch.zeros((bsz * nwin, kmax, fdim), device=bank_f.device, dtype=bank_f.dtype)

        for start in range(0, flat_q.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, flat_q.shape[0])
            q_chunk = flat_q[start:end]
            f_chunk = flat_f[start:end]
            mask_chunk = flat_mask[start:end]
            active_chunk = mask_chunk.any(dim=-1)
            if not bool(active_chunk.any().item()):
                continue

            q_sym = torch.einsum("gij,cmj->cgmi", sym_ops, q_chunk)
            dots = torch.einsum("cni,cgmi->cgnm", q_chunk, q_sym).abs()
            best = dots.amax(dim=1).clamp(0.0, 1.0)
            mis = 2.0 * torch.acos(best)

            mask_f_chunk = mask_chunk.to(dtype=mis.dtype)
            pair_mask = mask_f_chunk.unsqueeze(-1) * mask_f_chunk.unsqueeze(-2)
            sum_mis = (mis.unsqueeze(1) * pair_mask).sum(dim=-1)
            inf = torch.full_like(sum_mis, float("inf"))
            sum_mis = torch.where(mask_chunk, sum_mis, inf)

            medoid_idx = torch.argmin(sum_mis, dim=-1)
            medoid_idx = torch.where(
                active_chunk,
                medoid_idx,
                torch.full_like(medoid_idx, -1),
            )
            medoid_flat[start:end] = medoid_idx

            gather_idx = medoid_idx.clamp_min(0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, fdim)
            gathered = torch.gather(
                f_chunk.unsqueeze(1).expand(-1, kmax, -1, -1),
                2,
                gather_idx,
            ).squeeze(2)
            slot_ctx_flat[start:end] = torch.where(
                active_chunk.unsqueeze(-1),
                gathered,
                torch.zeros_like(gathered),
            )

        slot_ctx = slot_ctx_flat.view(bsz, nwin, kmax, fdim)
        medoid_bank_idx = medoid_flat.view(bsz, nwin, kmax)

        if not batched:
            slot_ctx = slot_ctx.squeeze(0)
            medoid_bank_idx = medoid_bank_idx.squeeze(0)
        return slot_ctx, medoid_bank_idx


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


class WithinSlotInvariantPool(nn.Module):
    """Token-conditioned equivariant within-slot pooling using scalar invariant weights only."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        meta_dim: int,
        phase_dim: int,
        window_size: int = 5,
        hidden_dim: int = 96,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.summary = InvariantSlotSummary(self.irreps_feat)
        self.phase_dim = int(phase_dim)
        self.window_size = int(window_size)
        self.num_nodes = int(self.window_size * self.window_size)
        self.chunk_size = int(chunk_size)
        den = float(max(1, self.window_size // 2))
        coords = []
        for y in range(self.window_size):
            for x in range(self.window_size):
                coords.append(((float(y) - den) / den, (float(x) - den) / den))
        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32), persistent=False)
        pair_dim = int(self.summary.out_dim * 3)
        member_in_dim = pair_dim + int(meta_dim) + 2
        phase_in_dim = int(meta_dim) + self.phase_dim
        self.member_key = nn.Sequential(
            nn.Linear(member_in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.member_bias = nn.Linear(int(hidden_dim), 1)
        self.phase_query = nn.Sequential(
            nn.Linear(phase_in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.logit_scale = float(max(1, hidden_dim)) ** -0.5

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
            member_bias = self.member_bias(member_key).squeeze(-1)

            phase_meta = meta_chunk.unsqueeze(2).expand(-1, kmax, patch_tokens, -1)
            phase_query = self.phase_query(
                torch.cat([phase_meta, phase_chunk.unsqueeze(1).expand(-1, kmax, -1, -1)], dim=-1)
            )

            logits = member_bias.unsqueeze(2) + self.logit_scale * torch.einsum(
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
        kmax_slots: int = 4,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        phase_dim: int = 32,
        hidden_dim: int = 128,
        conv_hidden_dim: int = 64,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.kmax_slots = int(kmax_slots)
        self.upsample_factor = _as_isotropic_scale(upsample_factor)
        self.patch_tokens = int(self.upsample_factor * self.upsample_factor)
        self.phase_dim = int(phase_dim)
        self.summary = InvariantSlotSummary(self.irreps_feat)
        self.slot_hidden_dim = int(hidden_dim)

        in_slot = int(self.summary.out_dim) + int(meta_dim)
        self.slot_proj = nn.Sequential(
            nn.Linear(in_slot, self.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.slot_hidden_dim, self.slot_hidden_dim),
        )
        self.phase_proj = nn.Sequential(
            nn.Linear(self.phase_dim, self.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.slot_hidden_dim, self.slot_hidden_dim),
        )
        self.base_logit = nn.Sequential(
            nn.Linear(3 * self.slot_hidden_dim, self.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.slot_hidden_dim, 1),
        )
        self.patch_refine = nn.Sequential(
            nn.Conv2d(self.kmax_slots, int(conv_hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(conv_hidden_dim), self.kmax_slots, kernel_size=3, padding=1),
        )
        self.parent_prior_scale = nn.Parameter(torch.tensor(0.10))
        self.parent_prior_proj = nn.Sequential(
            nn.Linear(int(self.summary.out_dim) + self.phase_dim, self.slot_hidden_dim),
            nn.GELU(),
            nn.Linear(self.slot_hidden_dim, 1),
        )

    def forward(
        self,
        slot_ctx: torch.Tensor,
        slot_meta: torch.Tensor,
        phase_grid: torch.Tensor,
        weak_parent_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, nwin, kmax, cdim = slot_ctx.shape
        if kmax != self.kmax_slots:
            raise ValueError(f"Expected kmax_slots={self.kmax_slots}, got {kmax}")

        if phase_grid.dim() == 2:
            phase = phase_grid.view(1, 1, self.patch_tokens, self.phase_dim).expand(bsz, nwin, -1, -1)
        elif phase_grid.dim() == 4:
            phase = phase_grid
        else:
            raise ValueError(
                "phase_grid must have shape (T,D_phase) or (B,N,T,D_phase), "
                f"got {tuple(phase_grid.shape)}"
            )
        if int(phase.shape[2]) != self.patch_tokens:
            raise ValueError(
                f"Expected phase patch tokens {self.patch_tokens}, got {int(phase.shape[2])}"
            )

        slot_inv = self.summary.summarize(slot_ctx)
        slot_desc = self.slot_proj(torch.cat([slot_inv, slot_meta], dim=-1))

        slot_valid = slot_meta[..., ClusterSlotBuilder.META_VALID : ClusterSlotBuilder.META_VALID + 1]
        slot_mass = slot_meta[..., ClusterSlotBuilder.META_MASS : ClusterSlotBuilder.META_MASS + 1]
        weights = slot_valid * slot_mass
        global_desc = (weights * slot_desc).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-6)
        phase_desc = self.phase_proj(phase)

        slot_desc_exp = slot_desc.unsqueeze(3).expand(-1, -1, -1, self.patch_tokens, -1)
        global_exp = global_desc.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.kmax_slots, self.patch_tokens, -1)
        phase_exp = phase_desc.unsqueeze(2).expand(-1, -1, self.kmax_slots, -1, -1)

        base = self.base_logit(torch.cat([slot_desc_exp, global_exp, phase_exp], dim=-1)).squeeze(-1)

        if weak_parent_prior is not None:
            if weak_parent_prior.dim() != 3 or int(weak_parent_prior.shape[-1]) != cdim:
                raise ValueError(
                    f"weak_parent_prior must have shape (B,N,{cdim}), got {tuple(weak_parent_prior.shape)}"
                )
            weak_inv = self.summary.summarize(weak_parent_prior)
            weak_inv_exp = weak_inv.unsqueeze(2).expand(-1, -1, self.patch_tokens, -1)
            parent_bias = self.parent_prior_proj(torch.cat([weak_inv_exp, phase], dim=-1)).squeeze(-1)
            parent_slot_flag = slot_meta[
                ...,
                ClusterSlotBuilder.META_SLOT_TYPE_START : ClusterSlotBuilder.META_SLOT_TYPE_START + 1,
            ]
            base = base + torch.tanh(self.parent_prior_scale) * parent_slot_flag * parent_bias.unsqueeze(2)

        base_map = base.view(bsz * nwin, self.kmax_slots, self.upsample_factor, self.upsample_factor)
        refined = self.patch_refine(base_map)
        logits_map = base_map + refined
        logits = logits_map.view(bsz, nwin, self.kmax_slots, self.patch_tokens).permute(0, 1, 3, 2).contiguous()

        invalid = slot_valid.squeeze(-1) <= 0.0
        logits = logits.masked_fill(invalid.unsqueeze(2), -1e4)
        return logits


class EquivariantSlotPatchQueryAnchor(nn.Module):
    """Build a weak equivariant patch query from the slot context and an optional weak parent prior."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        meta_dim: int,
        phase_dim: int,
        hidden_dim: int = 64,
        max_parent_scale: float = 0.25,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.block_slices = _irrep_block_slices(self.irreps_feat)
        self.num_blocks = len(self.block_slices)
        self.max_parent_scale = float(max_parent_scale)
        in_dim = int(meta_dim) + int(phase_dim)
        self.slot_scale_net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.num_blocks),
        )
        self.parent_scale_net = nn.Sequential(
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
        if weak_parent_prior is None:
            parent_scale = torch.zeros_like(slot_scale)
            parent = torch.zeros_like(slot_ctx)
        else:
            parent_scale = self.max_parent_scale * torch.tanh(self.parent_scale_net(ctrl))
            parent = weak_parent_prior

        out = torch.zeros_like(slot_ctx)
        for j, (start, end) in enumerate(self.block_slices):
            out[..., start:end] = slot_scale[..., j : j + 1] * slot_ctx[..., start:end]
            out[..., start:end] = out[..., start:end] + parent_scale[..., j : j + 1] * parent[..., start:end]
        return out


class SharedTPPatchProposalHead(nn.Module):
    """Produce one HR patch proposal per slot using TP-based equivariant mixing."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        meta_dim: int,
        phase_dim: int,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feature_dim = int(self.irreps_feat.dim)
        self.block_slices = _irrep_block_slices(self.irreps_feat)
        self.num_blocks = len(self.block_slices)
        self.phase_dim = int(phase_dim)
        self.patch_tokens = int(_as_isotropic_scale(upsample_factor) ** 2)

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
        if phase_grid.dim() == 2:
            phase = phase_grid.view(1, 1, self.patch_tokens, self.phase_dim).expand(bsz, nwin, -1, -1)
        elif phase_grid.dim() == 4:
            phase = phase_grid
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

        if token_ctx is None:
            ctx = slot_ctx.unsqueeze(3).expand(-1, -1, -1, self.patch_tokens, -1)
        else:
            if patch_tokens_ctx != self.patch_tokens:
                raise ValueError(
                    f"Expected slot_ctx token dim {self.patch_tokens}, got {patch_tokens_ctx}"
                )
            ctx = token_ctx
        anchor_ctx = slot_anchor_ctx.unsqueeze(3).expand(-1, -1, -1, self.patch_tokens, -1)
        meta = slot_meta.unsqueeze(3).expand(-1, -1, -1, self.patch_tokens, -1)
        phase_exp = phase.unsqueeze(2).expand(-1, -1, kmax, -1, -1)

        weak_parent_exp = None
        if weak_parent_prior is not None:
            if weak_parent_prior.dim() != 3 or int(weak_parent_prior.shape[-1]) != self.feature_dim:
                raise ValueError(
                    f"weak_parent_prior must have shape (B,N,{self.feature_dim}), got {tuple(weak_parent_prior.shape)}"
                )
            weak_parent_exp = weak_parent_prior.unsqueeze(2).unsqueeze(3).expand(
                -1, -1, kmax, self.patch_tokens, -1
            )

        query = self.query_anchor(
            slot_ctx=anchor_ctx,
            slot_meta=meta,
            phase_feat=phase_exp,
            weak_parent_prior=weak_parent_exp,
        )

        q_flat = self.lin_query(query.reshape(-1, self.feature_dim))
        c_flat = self.lin_ctx(ctx.reshape(-1, self.feature_dim))
        tp_flat = self.tp(q_flat, c_flat)
        tp_out = tp_flat.reshape(bsz, nwin, kmax, self.patch_tokens, self.feature_dim)

        coeffs = self.ctrl_net(torch.cat([meta, phase_exp], dim=-1))
        alpha, beta = coeffs.chunk(2, dim=-1)
        alpha = 1.0 + 0.5 * torch.tanh(alpha)
        beta = 0.5 * torch.tanh(beta)

        out = torch.zeros_like(tp_out)
        for j, (start, end) in enumerate(self.block_slices):
            out[..., start:end] = alpha[..., j : j + 1] * tp_out[..., start:end]
            out[..., start:end] = out[..., start:end] + beta[..., j : j + 1] * query[..., start:end]
        return out


class OCRPPatchUpsampler(nn.Module):
    """Orientation-Cluster Routed Patch (OCRP) upsampler."""

    def __init__(
        self,
        irreps_feat: Irreps | str,
        sym_ops_quat: torch.Tensor,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        window_size: int = 5,
        kmax_slots: int = 4,
        cluster_threshold_deg: float = 2.0,
        cluster_connectivity: int = 4,
        phase_dim: int = 32,
        router_hidden_dim: int = 128,
        router_conv_hidden_dim: int = 64,
        proposal_hidden_dim: int = 128,
        straight_through: bool = True,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feature_dim = int(self.irreps_feat.dim)
        self.upsample_factor = _as_isotropic_scale(upsample_factor)
        self.window_size = int(window_size)
        self.kmax_slots = int(kmax_slots)
        self.straight_through = bool(straight_through)

        self.phase_embed = PhaseEmbeddingGrid(
            upsample_factor=self.upsample_factor,
            emb_dim=int(phase_dim),
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
        self.context_builder = MedoidSlotContextBuilder(sym_ops_quat=sym_ops_quat)
        self.slot_pool = WithinSlotInvariantPool(
            irreps_feat=self.irreps_feat,
            meta_dim=ClusterSlotBuilder.META_DIM,
            phase_dim=int(phase_dim),
            window_size=int(window_size),
            hidden_dim=max(64, int(proposal_hidden_dim)),
        )
        self.router = PatchSlotRouter(
            irreps_feat=self.irreps_feat,
            meta_dim=ClusterSlotBuilder.META_DIM,
            kmax_slots=int(kmax_slots),
            upsample_factor=self.upsample_factor,
            phase_dim=int(phase_dim),
            hidden_dim=int(router_hidden_dim),
            conv_hidden_dim=int(router_conv_hidden_dim),
        )
        self.proposal_head = SharedTPPatchProposalHead(
            irreps_feat=self.irreps_feat,
            meta_dim=ClusterSlotBuilder.META_DIM,
            phase_dim=int(phase_dim),
            upsample_factor=self.upsample_factor,
            hidden_dim=int(proposal_hidden_dim),
        )

    def _phase_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        phase_ids = torch.arange(
            self.upsample_factor * self.upsample_factor,
            device=device,
            dtype=torch.long,
        )
        return self.phase_embed(phase_ids).to(dtype=dtype)

    def _assemble_patch_tokens(
        self,
        patch_out: torch.Tensor,
        lr_shape: tuple[int, int],
    ) -> torch.Tensor:
        bsz, nwin, patch_tokens, cdim = patch_out.shape
        h_lr, w_lr = int(lr_shape[0]), int(lr_shape[1])
        if nwin != h_lr * w_lr:
            raise ValueError(f"Expected {h_lr*w_lr} patches, got {nwin}")
        r = self.upsample_factor
        if patch_tokens != r * r:
            raise ValueError(f"Expected patch tokens {r*r}, got {patch_tokens}")
        img = patch_out.view(bsz, h_lr, w_lr, r, r, cdim).permute(0, 1, 3, 2, 4, 5)
        img = img.reshape(bsz, h_lr * r, w_lr * r, cdim)
        return img.reshape(bsz, h_lr * r * w_lr * r, cdim)

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

        bank_q = _build_local_patch_bank(lr_quats, img_shape=lr_shape, window_size=self.window_size)
        bank_f = _build_local_patch_bank(feat_lr, img_shape=lr_shape, window_size=self.window_size)

        cluster_ids = self.clusterer(bank_q)
        slot_info = self.slot_builder(cluster_ids)
        slot_mask = slot_info["slot_mask"]
        slot_meta = slot_info["slot_meta"]
        slot_valid = slot_info["slot_valid"]

        slot_ctx, medoid_bank_idx = self.context_builder(
            bank_q=bank_q,
            bank_f=bank_f,
            slot_mask=slot_mask,
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

        weak_parent_prior = feat_lr
        router_logits = self.router(
            slot_ctx=slot_ctx,
            slot_meta=slot_meta,
            phase_grid=phase_grid,
            weak_parent_prior=weak_parent_prior,
        )
        owner_idx, owner_onehot = self._hard_owner_from_logits(
            router_logits,
            straight_through=self.straight_through and self.training,
        )

        patch_prop = self.proposal_head(
            slot_ctx=slot_pooled_ctx,
            slot_meta=slot_meta,
            phase_grid=phase_grid,
            weak_parent_prior=weak_parent_prior,
            slot_anchor_ctx=slot_ctx,
        )
        owner_mask = owner_onehot.permute(0, 1, 3, 2).unsqueeze(-1)
        patch_out = (owner_mask * patch_prop).sum(dim=2)
        feat_hr = self._assemble_patch_tokens(patch_out, lr_shape=lr_shape)

        if not batched:
            feat_hr = feat_hr.squeeze(0)

        if not return_aux:
            return feat_hr, (int(lr_shape[0]) * self.upsample_factor, int(lr_shape[1]) * self.upsample_factor)

        aux = {
            "bank_q": bank_q,
            "bank_f": bank_f,
            "cluster_ids": cluster_ids,
            "slot_mask": slot_mask,
            "slot_valid": slot_valid,
            "slot_meta": slot_meta,
            "slot_ctx": slot_ctx,
            "slot_pooled_ctx": slot_pooled_ctx,
            "slot_pool_alpha": slot_pool_alpha,
            "medoid_bank_idx": medoid_bank_idx,
            "router_logits": router_logits,
            "owner_idx": owner_idx,
            "patch_prop": patch_prop,
            "patch_out": patch_out,
        }
        if not batched:
            aux = {
                key: (val.squeeze(0) if isinstance(val, torch.Tensor) and val.shape[0] == 1 else val)
                for key, val in aux.items()
            }
        return feat_hr, (int(lr_shape[0]) * self.upsample_factor, int(lr_shape[1]) * self.upsample_factor), aux


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
        conv_feature_mask_cosine_threshold: float = 0.99,
        conv_feature_mask_soft: bool = False,
        conv_feature_mask_temperature: float = 32.0,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        window_size: int = 5,
        kmax_slots: int = 4,
        cluster_threshold_deg: float = 2.0,
        cluster_connectivity: int = 4,
        phase_dim: int = 32,
        ocrp_router_hidden_dim: int = 128,
        ocrp_router_conv_hidden_dim: int = 64,
        ocrp_proposal_hidden_dim: int = 128,
        ocrp_straight_through: bool = True,
        use_hr_conv1: bool = True,
        hr_conv1_kernel_size: int = 7,
        use_residual_hr1: bool = True,
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
        self.upsample_factor = _as_isotropic_scale(upsample_factor)
        self.conv_lr1 = CosineMaskedEquivariantSpatialConv(
            kernel_size=int(lr_conv1_kernel_size),
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=bool(use_residual_lr1),
            feature_mask_cosine_threshold=float(conv_feature_mask_cosine_threshold),
            feature_mask_soft=bool(conv_feature_mask_soft),
            feature_mask_temperature=float(conv_feature_mask_temperature),
        )
        self.ocrp = OCRPPatchUpsampler(
            irreps_feat=self.irreps_feat,
            sym_ops_quat=self.encoder.sym_ops,
            upsample_factor=self.upsample_factor,
            window_size=int(window_size),
            kmax_slots=int(kmax_slots),
            cluster_threshold_deg=float(cluster_threshold_deg),
            cluster_connectivity=int(cluster_connectivity),
            phase_dim=int(phase_dim),
            router_hidden_dim=int(ocrp_router_hidden_dim),
            router_conv_hidden_dim=int(ocrp_router_conv_hidden_dim),
            proposal_hidden_dim=int(ocrp_proposal_hidden_dim),
            straight_through=bool(ocrp_straight_through),
        )
        self.conv_hr1 = CosineMaskedEquivariantSpatialConv(
            kernel_size=int(hr_conv1_kernel_size),
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=bool(use_residual_hr1),
            feature_mask_cosine_threshold=float(conv_feature_mask_cosine_threshold),
            feature_mask_soft=bool(conv_feature_mask_soft),
            feature_mask_temperature=float(conv_feature_mask_temperature),
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

        if self.use_hr_conv1:
            feat_hr = self.conv_hr1(feat_hr, hr_shape)

        if not return_aux:
            return feat_hr, hr_shape

        aux["feat_lr_pre_ocrp"] = feat_pre
        aux["feat_hr_raw_ocrp"] = feat_hr_raw_ocrp
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


__all__ = [
    "LocalIsoCrystalEncoder",
    "CubochoricOptimizingLocalIsoDecoder",
    "CosineMaskedEquivariantSpatialConv",
    "PhaseEmbeddingGrid",
    "QuaternionBankClusterer",
    "ClusterSlotBuilder",
    "MedoidSlotContextBuilder",
    "InvariantSlotSummary",
    "WithinSlotInvariantPool",
    "PatchSlotRouter",
    "EquivariantSlotPatchQueryAnchor",
    "SharedTPPatchProposalHead",
    "OCRPPatchUpsampler",
    "IsoEmbeddingSROCRP",
]
