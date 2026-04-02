
"""
SR_double_conv_SRattn_a1.py
====================================
Standalone local-iso SR model with double low-resolution (LR) convolution and high-resolution (HR) attention.

Key requirements addressed:
- Single model class: `IsoEmbeddingSRAttn`
- Uses local-iso embedding encoder for crystal orientation representation
- Uses cubochoric-sampled optimizing decoder (feature-space optimization)
- Crystal family selected at top level (`crystal='fcc'` or `crystal='hcp'`)

This file contains:
- Quaternion normalization and manipulation utilities
- LocalIsoCrystalEncoder: wraps local-iso embedding for FCC/HCP
- CubochoricOptimizingLocalIsoDecoder: feature-to-quaternion decoder using optimization
- LearnableA1QuaternionDecoder: MLP-based decoder (not used in main pipeline)
- EquivariantSpatialConv: e3nn-based equivariant spatial convolution
- EquivariantTransposeConv: e3nn-based equivariant upsampling
- AttentionBlock: block-local equivariant self-attention
- IsoEmbeddingSRAttn: the main model class
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear as IrrepsLinear
import numpy as np

from models.local_iso_embedding import (
    build_fcc_syms_mtex,
    build_hcp_syms_mtex,
    build_local_iso_fcc_embedding,
    build_local_iso_hcp_embedding,
)



def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a batch of quaternions to unit norm, ensuring the scalar part is non-negative.
    Args:
        quats: (..., 4) tensor of quaternions
        eps: minimum norm for numerical stability
    Returns:
        (..., 4) tensor of normalized quaternions
    """
    norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
    q = quats / norm
    # Ensure scalar part (w) is non-negative for unique representation
    return torch.where(q[..., :1] < 0.0, -q, q)



def _quat_conjugate(quats: torch.Tensor) -> torch.Tensor:
    """
    Compute the conjugate of a batch of quaternions.
    Args:
        quats: (..., 4) tensor of quaternions
    Returns:
        (..., 4) tensor of conjugated quaternions
    """
    return torch.cat([quats[..., :1], -quats[..., 1:]], dim=-1)



def _sample_fz_quaternions_passive(
    group_name: str,
    resolution: int,
    method: str,
    dtype: torch.dtype,
    device: torch.device,
    max_rows: int | None = None,
) -> torch.Tensor:
    """
    Sample quaternions in the fundamental zone (FZ) for a given symmetry group.
    Args:
        group_name: 'O' (cubic) or 'D6' (hexagonal)
        resolution: sampling resolution
        method: sampling method (e.g., 'cubochoric')
        dtype: output tensor dtype
        device: output tensor device
        max_rows: optional maximum number of quaternions to return
    Returns:
        (N, 4) tensor of unit quaternions
    """
    from orix.quaternion import symmetry
    from orix.sampling import get_sample_fundamental
    import numpy as np

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



class LocalIsoCrystalEncoder(nn.Module):
    """
    Local-iso encoder wrapper with crystal-family switch.
    Selects FCC or HCP embedding and symmetry operators based on `crystal` argument.
    Provides methods to encode quaternions into irreducible representations (irreps).
    """

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

        self.register_buffer("sym_ops", sym, persistent=False)
        self.register_buffer("sym_ops_inv", _quat_conjugate(sym), persistent=False)


    def _to_embedding_device(self, quats_passive: torch.Tensor) -> torch.Tensor:
        """
        Move input quaternions to the device/dtype of the embedding for correct computation.
        """
        return quats_passive.to(
            device=self.embedding.group_mats.device,
            dtype=self.embedding.group_mats.dtype,
        )


    def forward_a1(self, quats_passive: torch.Tensor) -> torch.Tensor:
        """
        Encode quaternions to A1 irreps (lowest-order invariant features).
        """
        q = self._to_embedding_device(quats_passive)
        return self.embedding.forward_irreps_passive(q, active_only=True)


    def forward_a1_active(self, quats_active: torch.Tensor) -> torch.Tensor:
        """
        Encode active-convention quaternions directly to A1-supported irreps.
        """
        q = self._to_embedding_device(quats_active)
        return self.embedding.forward_from_quaternions(q, active_only=True)


    def forward_full(self, quats_passive: torch.Tensor) -> torch.Tensor:
        """
        Encode quaternions to full irreps (all invariant and equivariant features).
        """
        q = self._to_embedding_device(quats_passive)
        return self.embedding.forward_irreps_passive(q, active_only=False)


    def forward_full_active(self, quats_active: torch.Tensor) -> torch.Tensor:
        """
        Encode active-convention quaternions directly to full irreps.
        """
        q = self._to_embedding_device(quats_active)
        return self.embedding.forward_from_quaternions(q, active_only=False)





def _irrep_block_norm_invariants(
    x: torch.Tensor,
    irreps: Irreps | str,
    *,
    normalize: bool = True,
    squared: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Convert mixed irrep features into scalar invariant summaries by taking one norm
    per irrep copy. This does not change the reconstruction representation; it only
    provides scalar guidance features for boundary / affinity prediction.
    """
    irreps = Irreps(irreps)
    out = []
    for sl, (mul, ir) in zip(irreps.slices(), irreps):
        xi = x[..., sl].reshape(*x.shape[:-1], int(mul), int(ir.dim))
        sq = (xi * xi).sum(dim=-1)
        if normalize:
            sq = sq / float(ir.dim)
        val = sq if squared else torch.sqrt(sq.clamp_min(eps))
        out.append(val)
    if len(out) == 0:
        return x[..., :0]
    return torch.cat(out, dim=-1)


def _flat_to_image(feat: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
    """
    (B, H*W, C) or (H*W, C) -> (B, C, H, W) or (1, C, H, W)
    """
    H, W = img_shape
    batched = feat.dim() == 3
    if not batched:
        feat = feat.unsqueeze(0)
    B, N, C = feat.shape
    if N != H * W:
        raise ValueError(f"Expected N={H*W}, got {N}")
    return feat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def _image_to_flat(feat_img: torch.Tensor, batched: bool = True) -> torch.Tensor:
    """
    (B, C, H, W) -> (B, H*W, C) or (H*W, C)
    """
    B, C, H, W = feat_img.shape
    out = feat_img.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
    return out if batched else out.squeeze(0)



class CubochoricOptimizingLocalIsoDecoder(nn.Module):
    """
    Decoder that maps local-iso irreps features (A1 or full) to passive quaternions.
    Uses a two-stage process:
      1. Nearest-neighbor search in a cubochoric-sampled lookup table (fundamental zone)
      2. Local optimization (Adam) to minimize feature-space MSE to the target features

    This enables differentiable, symmetry-aware decoding from feature space to orientation.
    """

    def __init__(
        self,
        encoder: LocalIsoCrystalEncoder,
        cubochoric_resolution: int = 1,
        method: str = "cubochoric",
        num_starts: int = 6,
        steps: int = 25,
        lr: float = 0.05,
        target_irreps: str = "a1",
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
        """
        Return a dictionary of metadata describing the current decoder configuration.
        Used for cache key generation.
        """
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
        """
        Return paths for cached quaternion/feature tables and metadata, or None if caching is disabled.
        """
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
        """
        Attempt to load cached quaternion and feature tables from disk.
        Returns None if not found or invalid.
        """
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
        """
        Save quaternion and feature tables to disk for future reuse.
        """
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
        """
        For each target feature, find the indices of the k nearest seeds in the lookup table.
        Uses chunked distance computation for memory efficiency.
        """
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
        """
        Encode quaternions to the target irreps (A1 or full) using the encoder.
        """
        if self.target_irreps == "a1":
            return self.encoder.forward_a1(quats_passive)
        return self.encoder.forward_full(quats_passive)

    def forward(self, feat_target: torch.Tensor) -> torch.Tensor:
        """
        Decode features to quaternions by nearest-neighbor search and local optimization.
        Args:
            feat_target: (B, C) tensor of target features
        Returns:
            (B, 4) tensor of decoded unit quaternions
        """
        # Decoder is an optimization module; do not backprop to upstream SR features.
        feat_target = feat_target.detach().to(self.table_feat.device, dtype=torch.float32)
        B, C = feat_target.shape
        if C != self.target_dim:
            raise ValueError(f"Expected target dim {self.target_dim} for {self.target_irreps}, got {C}")

        with torch.no_grad():
            idx = self._nearest_seed_indices(feat_target)
            q0 = self.table_quats[idx]

        if self.steps == 0:
            return _normalize_quaternions(q0[:, 0, :])

        # Local optimization (Adam) to refine quaternions.
        # This must run with grads enabled even if caller wrapped execution in
        # torch.no_grad() (common in validation/inference code paths).
        k = q0.shape[1]
        u = nn.Parameter(q0.clone())
        opt = torch.optim.Adam([u], lr=self.lr)

        with torch.enable_grad():
            for _ in range(self.steps):
                opt.zero_grad(set_to_none=True)
                q = _normalize_quaternions(u)
                q_flat = q.reshape(B * k, 4)
                feat_pred = self._encode_target_features(q_flat).reshape(B, k, C)
                loss_per = (feat_pred - feat_target.unsqueeze(1)).pow(2).mean(dim=-1)
                loss = loss_per.mean()
                loss.backward()
                opt.step()

        with torch.no_grad():
            q = _normalize_quaternions(u)
            q_flat = q.reshape(B * k, 4)
            feat_pred = self._encode_target_features(q_flat).reshape(B, k, C)
            loss_per = (feat_pred - feat_target.unsqueeze(1)).pow(2).mean(dim=-1)
            best_k = torch.argmin(loss_per, dim=1)
            batch_idx = torch.arange(B, device=feat_target.device)
            q_best = q[batch_idx, best_k]
        return _normalize_quaternions(q_best)



class LearnableA1QuaternionDecoder(nn.Module):
    """
    Learnable MLP decoder from A1 features to passive unit quaternions.
    Not used in the main pipeline, but useful for ablation or comparison.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dim, 4))
        else:
            for _ in range(self.num_layers - 1):
                layers.append(nn.Linear(in_dim, self.hidden_dim))
                layers.append(nn.GELU())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
                in_dim = self.hidden_dim
            layers.append(nn.Linear(in_dim, 4))
        self.net = nn.Sequential(*layers)

        # Start near identity quaternions.
        last_linear = None
        for mod in reversed(self.net):
            if isinstance(mod, nn.Linear):
                last_linear = mod
                break
        if last_linear is not None:
            with torch.no_grad():
                last_linear.bias.zero_()
                last_linear.bias[0] = 1.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"Expected features of shape (N,C), got {tuple(features.shape)}")
        if int(features.shape[-1]) != self.input_dim:
            raise ValueError(
                f"Expected feature dim {self.input_dim}, got {int(features.shape[-1])}"
            )
        q = self.net(features)
        return _normalize_quaternions(q)



class BoundaryAffinityBranch(nn.Module):
    """
    Predict scalar guidance fields and boundary logits from irrep-valued features.

    The input representation stays exactly the same as the paper-derived A1 features.
    We only derive scalar summaries from those features to drive boundary-aware
    spatial mixing.
    """

    def __init__(
        self,
        irreps_feat: Irreps | str,
        upsample_factor: int = 4,
        hidden_dim: int = 48,
        guidance_dim: int = 16,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.upsample_factor = int(upsample_factor)
        self.scalar_dim = sum(int(mul) for mul, _ in self.irreps_feat)
        self.hidden_dim = int(hidden_dim)
        self.guidance_dim = int(guidance_dim)

        self.lr_stem = nn.Sequential(
            nn.Conv2d(self.scalar_dim, self.hidden_dim, kernel_size=3, padding=1),
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

        nn.init.zeros_(self.lr_boundary.weight)
        nn.init.zeros_(self.lr_boundary.bias)
        nn.init.zeros_(self.hr_boundary.weight)
        nn.init.zeros_(self.hr_boundary.bias)

    def forward(
        self,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        H, W = lr_shape
        inv_lr = _irrep_block_norm_invariants(feat_lr, self.irreps_feat)
        inv_img = _flat_to_image(inv_lr, lr_shape)
        stem = self.lr_stem(inv_img)

        guidance_lr = self.lr_guidance(stem)
        boundary_lr = self.lr_boundary(stem)

        Hr = H * self.upsample_factor
        Wr = W * self.upsample_factor

        guidance_up = F.interpolate(
            guidance_lr,
            size=(Hr, Wr),
            mode="bilinear",
            align_corners=False,
        )
        boundary_up = F.interpolate(
            boundary_lr,
            size=(Hr, Wr),
            mode="bilinear",
            align_corners=False,
        )

        hr_in = torch.cat([guidance_up, torch.sigmoid(boundary_up)], dim=1)
        hr_hidden = self.hr_refine(hr_in)

        guidance_hr = guidance_up + self.hr_guidance(hr_hidden)
        boundary_hr = boundary_up + self.hr_boundary(hr_hidden)

        return {
            "inv_lr": inv_img,
            "guidance_lr": guidance_lr,
            "boundary_logits_lr": boundary_lr,
            "guidance_hr": guidance_hr,
            "boundary_logits_hr": boundary_hr,
            "hr_shape": (Hr, Wr),
        }





class EquivariantSpatialConv(nn.Module):
    """
    Equivariant spatial convolution for irrep-valued feature maps with boundary-aware
    affinity gating. Spatial transport is modulated by scalar guidance and predicted
    boundary logits so neighbors across likely boundaries contribute less.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        irreps_in: Irreps | str = "1x4e",
        irreps_out: Irreps | str | None = None,
        use_residual: bool = False,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2

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
        self.log_tau = nn.Parameter(torch.tensor(math.log(8.0)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))

    def _effective_patch_weights(
        self,
        guidance: torch.Tensor | None,
        boundary_logits: torch.Tensor | None,
        H: int,
        W: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = torch.softmax(self.spatial_logits.reshape(-1), dim=0).reshape(
            1, 1, 1, 1, self.kernel_size, self.kernel_size
        ).to(dtype)

        if guidance is None and boundary_logits is None:
            return base

        tau = F.softplus(self.log_tau).to(dtype)
        beta = F.softplus(self.log_beta).to(dtype)

        if guidance is not None:
            gp = F.pad(guidance, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
            gpatch = gp.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            gcenter = guidance.unsqueeze(-1).unsqueeze(-1)
            d2 = (gpatch - gcenter).pow(2).mean(dim=1, keepdim=True)
            gate = torch.exp(-tau * d2)
        else:
            gate = 1.0

        if boundary_logits is not None:
            b = torch.sigmoid(boundary_logits)
            bp = F.pad(b, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
            bpatch = bp.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            bcenter = b.unsqueeze(-1).unsqueeze(-1)
            bpen = 0.5 * (bpatch + bcenter)
            gate = gate * torch.exp(-beta * bpen)

        eff = base * gate
        eff = eff / eff.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        return eff

    def forward(
        self,
        features: torch.Tensor,
        img_shape: tuple[int, int],
        guidance: torch.Tensor | None = None,
        boundary_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        H, W = img_shape
        batched = features.dim() == 3
        if not batched:
            features = features.unsqueeze(0)
        B, N, C = features.shape
        if C != self.in_dim:
            raise ValueError(f"Expected feature dim {self.in_dim}, got {C}")
        if N != H * W:
            raise ValueError(f"Expected N={H*W}, got N={N}")

        feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)
        feat_padded = F.pad(
            feat_img,
            (self.padding, self.padding, self.padding, self.padding),
            mode="replicate",
        )
        patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        eff = self._effective_patch_weights(guidance, boundary_logits, H, W, feat_img.dtype)
        neigh = (patches * eff).sum(dim=(-1, -2))

        feat_flat = features.reshape(B * N, C)
        neigh_flat = neigh.permute(0, 2, 3, 1).reshape(B * N, C)
        out = self.tp(feat_flat, neigh_flat)

        if self.use_residual:
            if self.residual_proj is None:
                out = out + feat_flat
            else:
                out = out + self.residual_proj(feat_flat)

        out = out.reshape(B, N, self.out_dim)
        if not batched:
            out = out.squeeze(0)
        return out



class EquivariantTransposeConv(nn.Module):
    """
    Equivariant upsampler with boundary-aware HR context aggregation.

    Spatial upsampling itself remains tied across irrep m-components, but the HR
    context used for mixing is affinity-gated by the scalar guidance / boundary branch.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        upsample_factor: int = 4,
        use_residual: bool = True,
        irreps_in: Irreps | str = "1x4e",
        irreps_out: Irreps | str | None = None,
    ):
        super().__init__()
        self.upsample_factor = int(upsample_factor)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.use_residual = bool(use_residual)

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out) if irreps_out is not None else self.irreps_in
        self.in_dim = int(self.irreps_in.dim)
        self.out_dim = int(self.irreps_out.dim)
        C = self.in_dim

        self.transpose_kernel_size = int(self.upsample_factor + 2)
        self.transpose_padding = int((self.transpose_kernel_size - self.upsample_factor) // 2)

        copy_ids: list[int] = []
        copy_count = 0
        for mul, ir in self.irreps_in:
            ir_dim = int(ir.dim)
            for _ in range(int(mul)):
                copy_ids.extend([copy_count] * ir_dim)
                copy_count += 1
        if len(copy_ids) != C:
            raise RuntimeError(
                f"Internal channel mapping error: got {len(copy_ids)} channels, expected {C}"
            )
        self.num_irrep_copies = int(copy_count)
        self.register_buffer(
            "channel_to_copy_idx",
            torch.tensor(copy_ids, dtype=torch.long),
            persistent=False,
        )
        self.transpose_kernels = nn.Parameter(
            torch.empty(self.num_irrep_copies, 1, self.transpose_kernel_size, self.transpose_kernel_size)
        )
        with torch.no_grad():
            self._init_bilinear()

        self.spatial_logits = nn.Parameter(torch.zeros(self.kernel_size, self.kernel_size))
        self.log_tau = nn.Parameter(torch.tensor(math.log(8.0)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_out,
            shared_weights=True,
        )
        self.residual_proj: o3.Linear | None = None
        if self.use_residual and (self.irreps_in != self.irreps_out):
            self.residual_proj = o3.Linear(self.irreps_in, self.irreps_out)

    def _init_bilinear(self) -> None:
        r = self.upsample_factor
        k = self.transpose_kernel_size
        bilinear_1d = torch.zeros(k)
        center = (k - 1) / 2.0
        for i in range(k):
            bilinear_1d[i] = max(0.0, 1.0 - abs(i - center) / r)
        bilinear_2d = bilinear_1d.unsqueeze(1) * bilinear_1d.unsqueeze(0)
        bilinear_2d = bilinear_2d / bilinear_2d.sum()
        self.transpose_kernels.data[:] = bilinear_2d.unsqueeze(0).unsqueeze(0)

    def _expanded_transpose_weight(self) -> torch.Tensor:
        return torch.index_select(self.transpose_kernels, 0, self.channel_to_copy_idx)

    def _effective_patch_weights(
        self,
        guidance: torch.Tensor | None,
        boundary_logits: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = torch.softmax(self.spatial_logits.reshape(-1), dim=0).reshape(
            1, 1, 1, 1, self.kernel_size, self.kernel_size
        ).to(dtype)

        if guidance is None and boundary_logits is None:
            return base

        tau = F.softplus(self.log_tau).to(dtype)
        beta = F.softplus(self.log_beta).to(dtype)

        if guidance is not None:
            gp = F.pad(guidance, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
            gpatch = gp.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            gcenter = guidance.unsqueeze(-1).unsqueeze(-1)
            d2 = (gpatch - gcenter).pow(2).mean(dim=1, keepdim=True)
            gate = torch.exp(-tau * d2)
        else:
            gate = 1.0

        if boundary_logits is not None:
            b = torch.sigmoid(boundary_logits)
            bp = F.pad(b, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
            bpatch = bp.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            bcenter = b.unsqueeze(-1).unsqueeze(-1)
            bpen = 0.5 * (bpatch + bcenter)
            gate = gate * torch.exp(-beta * bpen)

        eff = base * gate
        eff = eff / eff.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        return eff

    def forward(
        self,
        features: torch.Tensor,
        img_shape: tuple[int, int],
        hr_guidance: torch.Tensor | None = None,
        hr_boundary_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        H, W = img_shape
        r = self.upsample_factor
        Hr, Wr = H * r, W * r

        batched = features.dim() == 3
        if not batched:
            features = features.unsqueeze(0)
        B = features.shape[0]
        C = features.shape[-1]
        if C != self.in_dim:
            raise ValueError(f"Expected feature dim {self.in_dim}, got {C}")
        if features.shape[1] != H * W:
            raise ValueError(f"Expected N={H*W}, got N={features.shape[1]}")

        feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)
        up_weight = self._expanded_transpose_weight()
        feat_hr = torch.nn.functional.conv_transpose2d(
            feat_img,
            up_weight,
            bias=None,
            stride=self.upsample_factor,
            padding=self.transpose_padding,
            output_padding=0,
            groups=C,
        )[:, :, :Hr, :Wr]

        feat_padded = F.pad(feat_hr, [self.padding] * 4, mode="replicate")
        patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        eff = self._effective_patch_weights(hr_guidance, hr_boundary_logits, feat_hr.dtype)
        context = (patches * eff).sum(dim=(-1, -2))

        N = Hr * Wr
        feat_flat = feat_hr.permute(0, 2, 3, 1).reshape(B * N, C)
        context_flat = context.permute(0, 2, 3, 1).reshape(B * N, C)
        out = self.tp(feat_flat, context_flat)
        if self.use_residual:
            if self.residual_proj is None:
                out = out + feat_flat
            else:
                out = out + self.residual_proj(feat_flat)

        out = out.reshape(B, N, self.out_dim)
        if not batched:
            out = out.squeeze(0)
        return out, (Hr, Wr)



class AttentionBlock(nn.Module):
    """
    Block-local equivariant self-attention with optional boundary-aware masking.
    The value path is still equivariant; only scalar attention scores are modulated.
    """

    def __init__(
        self,
        irreps_feat: Irreps | str = "1x4e + 1x6e",
        num_channels: int = 8,
        tp_out_chunk_size: int | None = None,
    ):
        super().__init__()
        del tp_out_chunk_size

        c = int(num_channels)
        self.irreps_feat = Irreps(irreps_feat)
        self.irreps_h = Irreps([(int(mul) * c, ir) for mul, ir in self.irreps_feat])

        self.log_s = nn.Parameter(torch.tensor(0.0))
        self.log_tau = nn.Parameter(torch.tensor(math.log(8.0)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))

        self.pos_bias = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.pos_bias.weight)
        nn.init.zeros_(self.pos_bias.bias)

        self.lin_in = IrrepsLinear(self.irreps_feat, self.irreps_h)
        self.tp_out = FullyConnectedTensorProduct(
            self.irreps_h, self.irreps_h, self.irreps_h, shared_weights=True
        )

        self.lin_out = IrrepsLinear(self.irreps_h, self.irreps_feat)
        with torch.no_grad():
            self.lin_out.weight.data.zero_()

    def _invariant_scores(self, feat_blocks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        Bb, Nb, _ = feat_blocks.shape
        scores = feat_blocks.new_zeros(Bb, Nb, Nb)

        start = 0
        for mul, ir in self.irreps_feat:
            mul = int(mul)
            d = ir.dim
            n = mul * d
            x = feat_blocks[..., start:start + n].reshape(Bb, Nb, mul, d)
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
            sim = torch.einsum("bimd,bjmd->bijm", x, x).mean(dim=-1)
            scores = scores + sim
            start += n
        return scores

    def forward(
        self,
        feat: torch.Tensor,
        d_block: torch.Tensor,
        H: int,
        W: int,
        block_h: int,
        block_w: int,
        guidance: torch.Tensor | None = None,
        boundary_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, HW, C_feat = feat.shape
        if HW != H * W:
            raise ValueError(f"Expected feat.shape[1] == H*W = {H*W}, got {HW}")

        num_bh = H // block_h
        num_bw = W // block_w
        Nb = block_h * block_w
        Bb = B * num_bh * num_bw
        dtype = feat.dtype

        feat_blocks = (
            feat.reshape(B, num_bh, block_h, num_bw, block_w, C_feat)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bb, Nb, C_feat)
        )

        scores = torch.exp(self.log_s) * self._invariant_scores(feat_blocks)

        pb = self.pos_bias(d_block.unsqueeze(-1)).squeeze(-1)
        scores = scores + pb.unsqueeze(0)

        if guidance is not None:
            G = guidance.shape[1]
            g_blocks = (
                guidance.permute(0, 2, 3, 1)
                .reshape(B, num_bh, block_h, num_bw, block_w, G)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(Bb, Nb, G)
            )
            gdiff2 = (g_blocks.unsqueeze(2) - g_blocks.unsqueeze(1)).pow(2).mean(dim=-1)
            scores = scores - F.softplus(self.log_tau).to(dtype) * gdiff2

        if boundary_logits is not None:
            b = torch.sigmoid(boundary_logits)
            b_blocks = (
                b.permute(0, 2, 3, 1)
                .reshape(B, num_bh, block_h, num_bw, block_w, 1)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(Bb, Nb, 1)
            )
            bpen = 0.5 * (b_blocks.unsqueeze(2) + b_blocks.unsqueeze(1))
            scores = scores - F.softplus(self.log_beta).to(dtype) * bpen.squeeze(-1)

        attn = torch.softmax(scores.float(), dim=-1).to(dtype)

        feat_flat = feat_blocks.reshape(Bb * Nb, C_feat)
        h = self.lin_in(feat_flat).reshape(Bb, Nb, -1)
        ctx = torch.bmm(attn, h)

        h_out = self.tp_out(
            h.reshape(Bb * Nb, -1),
            ctx.reshape(Bb * Nb, -1),
        ).reshape(Bb, Nb, -1)

        delta_blocks = self.lin_out(h_out.reshape(Bb * Nb, -1)).reshape(Bb, Nb, C_feat)

        delta = (
            delta_blocks.reshape(B, num_bh, num_bw, block_h, block_w, C_feat)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, H * W, C_feat)
        )
        return delta


class BoundaryRefinementHead(nn.Module):
    """
    Final boundary-aware refinement head operating in irrep feature space.

    This stage performs a small number of anisotropic diffusion / snapping steps
    using scalar guidance and boundary logits. Because all transport weights are
    scalar, the refinement preserves equivariance of the irrep-valued features.
    The goal is to clean up boundary-adjacent outliers after the main SR trunk
    has already produced an HR prediction.
    """

    def __init__(
        self,
        irreps_feat: Irreps | str,
        guidance_dim: int,
        kernel_size: int = 3,
        num_steps: int = 3,
        hidden_dim: int = 32,
        refine_boundary_logits: bool = True,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feature_dim = int(self.irreps_feat.dim)
        self.scalar_dim = sum(int(mul) for mul, _ in self.irreps_feat)
        self.guidance_dim = int(guidance_dim)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.num_steps = max(1, int(num_steps))
        self.hidden_dim = int(hidden_dim)
        self.refine_boundary_logits = bool(refine_boundary_logits)

        self.spatial_logits = nn.Parameter(torch.zeros(self.num_steps, self.kernel_size, self.kernel_size))
        self.log_tau = nn.Parameter(torch.full((self.num_steps,), math.log(8.0)))
        self.log_beta = nn.Parameter(torch.full((self.num_steps,), math.log(4.0)))
        self.log_alpha = nn.Parameter(torch.full((self.num_steps,), math.log(0.75)))

        gate_in_ch = self.guidance_dim + 1 + 1 + self.scalar_dim
        self.update_gate = nn.Sequential(
            nn.Conv2d(gate_in_ch, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
        )

        if self.refine_boundary_logits:
            self.boundary_delta = nn.Sequential(
                nn.Conv2d(gate_in_ch, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
            )
            nn.init.zeros_(self.boundary_delta[-1].weight)
            nn.init.zeros_(self.boundary_delta[-1].bias)
        else:
            self.boundary_delta = None

        nn.init.zeros_(self.update_gate[-1].weight)
        nn.init.constant_(self.update_gate[-1].bias, -1.0)

    def _effective_patch_weights(
        self,
        guidance: torch.Tensor | None,
        boundary_logits: torch.Tensor | None,
        step: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = torch.softmax(self.spatial_logits[step].reshape(-1), dim=0).reshape(
            1, 1, 1, 1, self.kernel_size, self.kernel_size
        ).to(dtype)

        if guidance is None and boundary_logits is None:
            return base

        tau = F.softplus(self.log_tau[step]).to(dtype)
        beta = F.softplus(self.log_beta[step]).to(dtype)

        if guidance is not None:
            gp = F.pad(guidance, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
            gpatch = gp.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            gcenter = guidance.unsqueeze(-1).unsqueeze(-1)
            d2 = (gpatch - gcenter).pow(2).mean(dim=1, keepdim=True)
            gate = torch.exp(-tau * d2)
        else:
            gate = 1.0

        if boundary_logits is not None:
            b = torch.sigmoid(boundary_logits)
            bp = F.pad(b, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
            bpatch = bp.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            bcenter = b.unsqueeze(-1).unsqueeze(-1)
            bpen = 0.5 * (bpatch + bcenter)
            gate = gate * torch.exp(-beta * bpen)

        eff = base * gate
        eff = eff / eff.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        return eff

    def forward(
        self,
        feat: torch.Tensor,
        hr_shape: tuple[int, int],
        guidance: torch.Tensor | None = None,
        boundary_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        Hr, Wr = hr_shape
        batched = feat.dim() == 3
        if not batched:
            feat = feat.unsqueeze(0)

        B, N, C = feat.shape
        if N != Hr * Wr:
            raise ValueError(f"Expected N={Hr*Wr}, got {N}")
        if C != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {C}")

        feat_img = feat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2).contiguous()
        b_logits = boundary_logits

        for step in range(self.num_steps):
            feat_padded = F.pad(feat_img, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
            patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            eff = self._effective_patch_weights(guidance, b_logits, step, feat_img.dtype)
            context = (patches * eff).sum(dim=(-1, -2))

            diff_flat = (context - feat_img).permute(0, 2, 3, 1).reshape(B, Hr * Wr, C)
            diff_inv = _irrep_block_norm_invariants(diff_flat, self.irreps_feat)
            diff_inv_img = _flat_to_image(diff_inv, (Hr, Wr))
            discrepancy = diff_inv_img.mean(dim=1, keepdim=True)

            boundary_prob = (
                torch.sigmoid(b_logits) if b_logits is not None
                else feat_img.new_zeros((B, 1, Hr, Wr))
            )
            gate_in = torch.cat([
                guidance if guidance is not None else feat_img.new_zeros((B, self.guidance_dim, Hr, Wr)),
                boundary_prob,
                discrepancy,
                diff_inv_img,
            ], dim=1)

            update_gate = torch.sigmoid(self.update_gate(gate_in))
            alpha = torch.sigmoid(self.log_alpha[step]).to(feat_img.dtype)
            feat_img = feat_img + alpha * update_gate * (context - feat_img)

            if self.boundary_delta is not None and b_logits is not None:
                b_logits = b_logits + self.boundary_delta(gate_in)

        out = feat_img.permute(0, 2, 3, 1).reshape(B, Hr * Wr, C).contiguous()
        if not batched:
            out = out.squeeze(0)
            if b_logits is not None:
                b_logits = b_logits.squeeze(0)
        return out, b_logits


class IsoEmbeddingSRAttn(nn.Module):
    """
    Local-iso SR model.

    - Crystal family is selected at top-level with `crystal`:
      - `fcc` (group O)
      - `hcp` (group D6)
    - Optional ablations:
      - disable LR conv1 (`use_lr_conv1=False`)
      - disable LR conv2 (`use_lr_conv2=False`)
      - disable attention (`use_attention=False`)
    - All SR stages run in `irreps_a1` and there is no terminal projection layer.

    Main pipeline:
      1. Encode LR quaternions to A1 features
      2. Apply two LR equivariant convolutions (optional)
      3. Upsample to HR grid
      4. Apply HR equivariant convolution
      5. Apply block-local equivariant attention (optional)
      6. Decode HR features to quaternions using optimizing decoder
    """

    _SH_IRREPS = Irreps("1x0e + 1x2e")

    def __init__(
        self,
        crystal: str = "fcc",
        d6_convention: str = "z_axis",
        device: str | torch.device | None = None,
        upsample_factor: int = 4,
        upsample_residual: bool = True,
        use_boundary_branch: bool = True,
        boundary_hidden_dim: int = 48,
        boundary_guidance_dim: int = 16,
        use_refinement: bool = True,
        refinement_num_steps: int = 3,
        refinement_hidden_dim: int = 32,
        refinement_kernel_size: int = 3,
        use_lr_conv1: bool = True,
        use_lr_conv2: bool = True,
        use_attention: bool = True,
        num_hr_attn_blocks: int = 1,
        hr_attn_num_channels: int = 8,
        hr_attn_block_size: int = 16,
        hr_attn_tp_out_chunk_size: int | None = 2048,
        hr_attn_checkpoint: bool = False,
        decoder_cubochoric_resolution: int = 1,
        decoder_num_starts: int = 6,
        decoder_steps: int = 25,
        decoder_lr: float = 0.05,
        decoder_method: str = "cubochoric",
        decoder_max_table_rows: int | None = None,
        decoder_table_cache_dir: str | Path | None = "out/decoder_lookup_tables",
        decoder_backend: str = "optimizing",
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
        self.irreps_full = self.encoder.irreps_full
        self.feature_dim_a1 = int(self.encoder.out_dim_a1)
        self.feature_dim = self.feature_dim_a1
        self.output_irreps = self.irreps_a1
        self.output_dim = self.feature_dim_a1
        self.upsample_factor = int(upsample_factor)
        self.use_lr_conv1 = bool(use_lr_conv1)
        self.use_lr_conv2 = bool(use_lr_conv2)
        self.use_attention = bool(use_attention)
        self.use_boundary_branch = bool(use_boundary_branch)
        self.use_refinement = bool(use_refinement)
        self.hr_attn_block_size = int(hr_attn_block_size)
        self.hr_attn_checkpoint = bool(hr_attn_checkpoint)

        self.decoder_backend = str(decoder_backend).lower()
        if self.decoder_backend != "optimizing":
            raise ValueError(
                f"Only decoder_backend='optimizing' is supported, got {decoder_backend}"
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

        self.boundary_branch = BoundaryAffinityBranch(
            irreps_feat=self.irreps_a1,
            upsample_factor=self.upsample_factor,
            hidden_dim=int(boundary_hidden_dim),
            guidance_dim=int(boundary_guidance_dim),
        ) if self.use_boundary_branch else None

        self.refinement_head = BoundaryRefinementHead(
            irreps_feat=self.irreps_a1,
            guidance_dim=int(boundary_guidance_dim),
            kernel_size=int(refinement_kernel_size),
            num_steps=int(refinement_num_steps),
            hidden_dim=int(refinement_hidden_dim),
            refine_boundary_logits=True,
        ) if self.use_refinement else None

        # A1-only SR architecture:
        # LR conv1 k=3: a1 -> a1
        self.conv_lr1 = EquivariantSpatialConv(
            kernel_size=3,
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
            use_residual=True,
        )
        # LR conv2 k=9: a1 -> a1
        self.conv_lr2 = EquivariantSpatialConv(
            kernel_size=9,
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
        )
        # Upsample k=3: a1 -> a1
        self.upsample_conv = EquivariantTransposeConv(
            kernel_size=3,
            upsample_factor=self.upsample_factor,
            use_residual=bool(upsample_residual),
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
        )
        # HR conv1: a1 -> a1
        self.conv_hr1 = EquivariantSpatialConv(
            kernel_size=3,
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
        )
        # Attention block(s)
        if self.use_attention:
            self.attention_blocks = nn.ModuleList(
                [
                    AttentionBlock(
                        self.irreps_a1,
                        num_channels=int(hr_attn_num_channels),
                        tp_out_chunk_size=hr_attn_tp_out_chunk_size,
                    )
                    for _ in range(max(1, int(num_hr_attn_blocks)))
                ]
            )
        else:
            self.attention_blocks = nn.ModuleList([])

        self._cached_hr_block_shape: tuple[int, int] | None = None
        self._cached_hr_sh_block: torch.Tensor | None = None

    @staticmethod
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Hamilton product (multiplication) of two batches of quaternions.
        Args:
            q1, q2: (N, 4) tensors of quaternions
        Returns:
            (N, 4) tensor of quaternion products
        """
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


    def encode_a1(self, quats: torch.Tensor) -> torch.Tensor:
        """
        Encode quaternions to A1 irreps using the encoder.
        """
        return self.encoder.forward_a1(quats)


    def encode_a1_active(self, quats: torch.Tensor) -> torch.Tensor:
        """
        Encode active-convention quaternions to A1-supported irreps.
        """
        return self.encoder.forward_a1_active(quats)


    def encode_full_target(self, quats: torch.Tensor) -> torch.Tensor:
        """
        Encode quaternions to full irreps using the encoder.
        """
        return self.encoder.forward_full(quats)

    def reduce_to_fz(
        self,
        quats: torch.Tensor,
        return_op_map: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Reduce quaternions to the fundamental zone (FZ) using symmetry operations.
        For each input quaternion, applies all symmetry operators and selects the representative
        with the largest absolute scalar part (w), ensuring unique mapping to FZ.
        Args:
            quats: (N, 4) tensor of quaternions
            return_op_map: if True, also return the index of the symmetry op used
        Returns:
            (N, 4) tensor of FZ-reduced quaternions (and optionally op indices)
        """
        quats = _normalize_quaternions(quats)
        batch_size = quats.shape[0]

        q_expanded = quats.unsqueeze(1).expand(-1, self.encoder.sym_ops_inv.shape[0], -1)
        syms = self.encoder.sym_ops_inv.unsqueeze(0).expand(batch_size, -1, -1)
        q_flat = q_expanded.reshape(-1, 4)
        s_flat = syms.reshape(-1, 4)
        fam = self.quat_mul(s_flat, q_flat).view(batch_size, syms.shape[1], 4)
        fam = _normalize_quaternions(fam.reshape(-1, 4)).view(batch_size, syms.shape[1], 4)

        w_abs = fam[..., 0].abs()
        best_idx = torch.argmax(w_abs, dim=1)
        batch_idx = torch.arange(batch_size, device=quats.device)
        q_fz = fam[batch_idx, best_idx]
        q_fz = _normalize_quaternions(q_fz)
        if return_op_map:
            return q_fz, best_idx
        return q_fz


    def decode(self, features_a1: torch.Tensor) -> torch.Tensor:
        """
        Decode A1 features to quaternions using the decoder and reduce to FZ.
        Handles both batched and unbatched input.
        """
        batched = features_a1.dim() == 3
        if batched:
            bsz, n, c = features_a1.shape
            q = self.decoder(features_a1.reshape(bsz * n, c))
            q = self.reduce_to_fz(q).reshape(bsz, n, 4)
            return q
        q = self.decoder(features_a1)
        return self.reduce_to_fz(q)


    @torch.no_grad()
    def _get_hr_sh_block(
        self,
        block_h: int,
        block_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute and cache the pairwise L2 distance matrix for a spatial block.
        Used for position bias in attention blocks.
        """
        if (
            self._cached_hr_block_shape == (block_h, block_w)
            and self._cached_hr_sh_block is not None
            and self._cached_hr_sh_block.device == device
        ):
            return self._cached_hr_sh_block.to(dtype)

        ys = torch.linspace(-1.0, 1.0, block_h, device=device)
        xs = torch.linspace(-1.0, 1.0, block_w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        sh = torch.cdist(coords, coords, p=2)
        self._cached_hr_block_shape = (block_h, block_w)
        self._cached_hr_sh_block = sh
        return sh.to(dtype)




    def _apply_attention(
        self,
        features: torch.Tensor,
        hr_shape: tuple[int, int],
        guidance: torch.Tensor | None = None,
        boundary_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply block-local equivariant attention to HR features.
        Handles padding so that all blocks are full-sized.
        """
        if not self.use_attention or len(self.attention_blocks) == 0:
            return features

        Hr, Wr = hr_shape
        batched = features.dim() == 3
        if not batched:
            features = features.unsqueeze(0)
        B, N, C = features.shape
        if N != Hr * Wr:
            raise ValueError(f"Expected N={Hr*Wr} from hr_shape, got {N}")

        block_h = min(self.hr_attn_block_size, Hr)
        block_w = min(self.hr_attn_block_size, Wr)
        pad_h = (-Hr) % block_h
        pad_w = (-Wr) % block_w
        Hr_pad, Wr_pad = Hr + pad_h, Wr + pad_w

        feat = features
        guide = guidance
        bounds = boundary_logits

        if pad_h > 0 or pad_w > 0:
            feat_2d = feat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
            feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
            feat = feat_2d.permute(0, 2, 3, 1).reshape(B, Hr_pad * Wr_pad, C)

            if guide is not None:
                guide = F.pad(guide, (0, pad_w, 0, pad_h), mode="reflect")
            if bounds is not None:
                bounds = F.pad(bounds, (0, pad_w, 0, pad_h), mode="reflect")

        sh_block = self._get_hr_sh_block(block_h, block_w, feat.device, feat.dtype)
        for block in self.attention_blocks:
            if self.hr_attn_checkpoint and self.training and feat.requires_grad:
                def _run_block(x: torch.Tensor, _block=block, _guide=guide, _bounds=bounds) -> torch.Tensor:
                    return _block(
                        x,
                        sh_block,
                        Hr_pad,
                        Wr_pad,
                        block_h,
                        block_w,
                        guidance=_guide,
                        boundary_logits=_bounds,
                    )

                delta = checkpoint(_run_block, feat, use_reentrant=True)
            else:
                delta = block(
                    feat,
                    sh_block,
                    Hr_pad,
                    Wr_pad,
                    block_h,
                    block_w,
                    guidance=guide,
                    boundary_logits=bounds,
                )
            feat = feat + delta

        if pad_h > 0 or pad_w > 0:
            feat = feat.reshape(B, Hr_pad, Wr_pad, C)[:, :Hr, :Wr, :].reshape(B, Hr * Wr, C)

        if not batched:
            feat = feat.squeeze(0)
        return feat


    def _forward_sr_features(
        self,
        feat_lr_a1: torch.Tensor,
        lr_shape: tuple[int, int],
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int]] | tuple[torch.Tensor, tuple[int, int], dict[str, torch.Tensor]]:
        """
        Main SR feature pipeline: boundary guidance, LR convs, upsampling, HR conv, attention.
        Returns HR features and HR shape, and optionally auxiliary boundary outputs.
        """
        aux: dict[str, torch.Tensor] = {}
        guidance_lr = None
        boundary_logits_lr = None
        guidance_hr = None
        boundary_logits_hr = None

        if self.boundary_branch is not None:
            aux = self.boundary_branch(feat_lr_a1, lr_shape)
            guidance_lr = aux["guidance_lr"]
            boundary_logits_lr = aux["boundary_logits_lr"]
            guidance_hr = aux["guidance_hr"]
            boundary_logits_hr = aux["boundary_logits_hr"]

        if self.use_lr_conv1:
            feat = self.conv_lr1(
                feat_lr_a1,
                lr_shape,
                guidance=guidance_lr,
                boundary_logits=boundary_logits_lr,
            )
        else:
            feat = feat_lr_a1

        if self.use_lr_conv2:
            feat = self.conv_lr2(
                feat,
                lr_shape,
                guidance=guidance_lr,
                boundary_logits=boundary_logits_lr,
            )

        feat, hr_shape = self.upsample_conv(
            feat,
            lr_shape,
            hr_guidance=guidance_hr,
            hr_boundary_logits=boundary_logits_hr,
        )
        feat = self.conv_hr1(
            feat,
            hr_shape,
            guidance=guidance_hr,
            boundary_logits=boundary_logits_hr,
        )
        feat_a1 = self._apply_attention(
            feat,
            hr_shape,
            guidance=guidance_hr,
            boundary_logits=boundary_logits_hr,
        )

        if self.refinement_head is not None:
            feat_a1, boundary_logits_hr = self.refinement_head(
                feat_a1,
                hr_shape,
                guidance=guidance_hr,
                boundary_logits=boundary_logits_hr,
            )
            if len(aux) > 0:
                aux["boundary_logits_hr_refined"] = boundary_logits_hr

        if return_aux:
            return feat_a1, hr_shape, aux
        return feat_a1, hr_shape


    def feature_loss_sr(
        self,
        lr_quats: torch.Tensor,
        hr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = False,
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target HR features (A1 irreps).
        Used for feature-space supervision.
        """
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

        feat_lr_dim = int(feat_lr_a1_flat.shape[-1])
        feat_hr_dim = int(feat_hr_tgt_flat.shape[-1])
        if batched:
            feat_lr_a1 = feat_lr_a1_flat.reshape(B, -1, feat_lr_dim)
            feat_hr_tgt = feat_hr_tgt_flat.reshape(B, -1, feat_hr_dim)
        else:
            feat_lr_a1 = feat_lr_a1_flat
            feat_hr_tgt = feat_hr_tgt_flat

        feat_hr, _ = self._forward_sr_features(feat_lr_a1, lr_shape)
        return F.mse_loss(feat_hr, feat_hr_tgt)



    def forward_sr(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = True,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for SR: input LR quaternions, output SR quaternions.
        """
        lr_quats = lr_quats.to(self.device)
        if normalize_input:
            lr_quats = _normalize_quaternions(lr_quats)
        feat_lr_a1 = self.encode_a1(lr_quats)
        if return_aux:
            feat_hr_a1, _, aux = self._forward_sr_features(feat_lr_a1, lr_shape, return_aux=True)
            return self.decode(feat_hr_a1), aux
        feat_hr_a1, _ = self._forward_sr_features(feat_lr_a1, lr_shape)
        return self.decode(feat_hr_a1)


    def forward(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass: input quaternions (optionally with image shape), output HR quaternions.
        If img_shape is provided, runs SR pipeline; otherwise, decodes input features directly.
        """
        quats = quats.to(self.device)
        if quats.dim() != 2 or quats.shape[-1] != 4:
            raise ValueError(f"IsoEmbeddingSRAttn expects (N,4), got {tuple(quats.shape)}")
        if normalize_input:
            quats = _normalize_quaternions(quats)

        if img_shape is not None:
            return self.forward_sr(quats, lr_shape=img_shape, normalize_input=False)

        feat_a1 = self.encode_a1(quats)
        return self.decode(feat_a1)


__all__ = [
    "AttentionBlock",
    "BoundaryAffinityBranch",
    "BoundaryRefinementHead",
    "CubochoricOptimizingLocalIsoDecoder",
    "EquivariantSpatialConv",
    "EquivariantTransposeConv",
    "IsoEmbeddingSRAttn",
    "LearnableA1QuaternionDecoder",
    "LocalIsoCrystalEncoder",
]

# ---------------------------------------------------------------------------
# Debug Attention
# ---------------------------------------------------------------------------
# import torch
# import matplotlib.pyplot as plt

# def corner_pixel_distance_map(block_h=7, block_w=7, corner="top_left", device="cpu", dtype=torch.float32):
#     device = torch.device(device)

#     # Same coordinate construction as your _get_hr_sh_block
#     ys = torch.linspace(-1.0, 1.0, block_h, device=device)
#     xs = torch.linspace(-1.0, 1.0, block_w, device=device)
#     grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
#     coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (Nb, 2)

#     # Pairwise L2 distance matrix
#     sh = torch.cdist(coords, coords, p=2).to(dtype)

#     # Choose corner index
#     corner_to_rc = {
#         "top_left": (0, 0),
#         "top_right": (0, block_w - 1),
#         "bottom_left": (block_h - 1, 0),
#         "bottom_right": (block_h - 1, block_w - 1),
#     }
#     r, c = corner_to_rc[corner]
#     corner_idx = r * block_w + c

#     # Distances from that corner to all pixels
#     corner_map = sh[corner_idx].reshape(block_h, block_w).cpu()

#     print(f"corner={corner}, idx={corner_idx}, (row,col)=({r},{c})")
#     print(f"distance at selected corner = {corner_map[r, c].item():.4f}")  # should be 0

#     plt.figure(figsize=(5, 4))
#     plt.imshow(corner_map, cmap="magma")
#     plt.colorbar(label=f"L2 distance from {corner}")
#     plt.title(f"{corner.replace('_',' ').title()} Distance Map ({block_h}x{block_w})")
#     plt.xlabel("col")
#     plt.ylabel("row")
#     plt.tight_layout()
#     plt.show()

#     return sh, corner_map

# # Example:
# if __name__ == "__main__":
#     sh, corner_map = corner_pixel_distance_map(7, 7, corner="top_left")


# # attention_trace_with_plots.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from torch.utils.checkpoint import checkpoint


# class SimpleAttentionBlock(nn.Module):
#     """
#     Minimal block-local attention block with the same call signature as your block.
#     """
#     def __init__(self, channels: int, zero_init_out: bool = True):
#         super().__init__()
#         self.log_s = nn.Parameter(torch.tensor(0.0))
#         self.pos_bias = nn.Linear(1, 1, bias=True)
#         nn.init.zeros_(self.pos_bias.weight)
#         nn.init.zeros_(self.pos_bias.bias)

#         self.lin_v = nn.Linear(channels, channels)
#         self.lin_out = nn.Linear(channels, channels)
#         if zero_init_out:
#             nn.init.zeros_(self.lin_out.weight)
#             nn.init.zeros_(self.lin_out.bias)

#     def _run_block_attention(
#         self,
#         feat_blocks: torch.Tensor,   # (Bb, Nb, C)
#         d_block: torch.Tensor,       # (Nb, Nb)
#     ):
#         dtype = feat_blocks.dtype
#         f_n = F.normalize(feat_blocks, dim=-1, eps=1e-6)
#         scores = torch.exp(self.log_s) * torch.bmm(f_n, f_n.transpose(-2, -1))
#         pb = self.pos_bias(d_block.unsqueeze(-1)).squeeze(-1)  # (Nb, Nb)
#         scores = scores + pb.unsqueeze(0)
#         attn = torch.softmax(scores.float(), dim=-1).to(dtype)

#         v = self.lin_v(feat_blocks)
#         ctx = torch.bmm(attn, v)
#         delta_blocks = self.lin_out(ctx)
#         return delta_blocks, scores, attn

#     def forward(
#         self,
#         feat: torch.Tensor,          # (B, H*W, C)
#         d_block: torch.Tensor,       # (Nb, Nb)
#         H: int,
#         W: int,
#         block_h: int,
#         block_w: int,
#     ) -> torch.Tensor:
#         B, _, C = feat.shape
#         num_bh = H // block_h
#         num_bw = W // block_w
#         Nb = block_h * block_w
#         Bb = B * num_bh * num_bw

#         feat_blocks = (
#             feat.reshape(B, num_bh, block_h, num_bw, block_w, C)
#             .permute(0, 1, 3, 2, 4, 5)
#             .reshape(Bb, Nb, C)
#         )

#         delta_blocks, _, _ = self._run_block_attention(feat_blocks, d_block)

#         delta = (
#             delta_blocks.reshape(B, num_bh, num_bw, block_h, block_w, C)
#             .permute(0, 1, 3, 2, 4, 5)
#             .reshape(B, H * W, C)
#         )
#         return delta

#     def forward_debug(
#         self,
#         feat: torch.Tensor,          # (B, H*W, C)
#         d_block: torch.Tensor,       # (Nb, Nb)
#         H: int,
#         W: int,
#         block_h: int,
#         block_w: int,
#         select_bb_idx: int,
#     ):
#         B, _, C = feat.shape
#         num_bh = H // block_h
#         num_bw = W // block_w
#         Nb = block_h * block_w
#         Bb = B * num_bh * num_bw

#         feat_blocks = (
#             feat.reshape(B, num_bh, block_h, num_bw, block_w, C)
#             .permute(0, 1, 3, 2, 4, 5)
#             .reshape(Bb, Nb, C)
#         )

#         delta_blocks, scores, attn = self._run_block_attention(feat_blocks, d_block)

#         delta = (
#             delta_blocks.reshape(B, num_bh, num_bw, block_h, block_w, C)
#             .permute(0, 1, 3, 2, 4, 5)
#             .reshape(B, H * W, C)
#         )

#         dbg = {
#             "scores_sel": scores[select_bb_idx].detach().cpu(),  # (Nb, Nb)
#             "attn_sel": attn[select_bb_idx].detach().cpu(),      # (Nb, Nb)
#         }
#         return delta, dbg


# class DemoHRModule(nn.Module):
#     def __init__(
#         self,
#         channels: int = 16,
#         num_blocks: int = 2,
#         block_size: int = 4,
#         use_checkpoint: bool = False,
#         zero_init_out: bool = False,   # False so you see non-zero deltas in plots
#     ):
#         super().__init__()
#         self.use_attention = True
#         self.hr_attn_block_size = block_size
#         self.hr_attn_checkpoint = use_checkpoint
#         self.attention_blocks = nn.ModuleList(
#             [SimpleAttentionBlock(channels, zero_init_out=zero_init_out) for _ in range(num_blocks)]
#         )

#         self._cached_hr_block_shape = None
#         self._cached_hr_sh_block = None

#     def _get_hr_sh_block(
#         self,
#         block_h: int,
#         block_w: int,
#         device: torch.device,
#         dtype: torch.dtype,
#     ) -> torch.Tensor:
#         if (
#             self._cached_hr_block_shape == (block_h, block_w)
#             and self._cached_hr_sh_block is not None
#             and self._cached_hr_sh_block.device == device
#         ):
#             return self._cached_hr_sh_block.to(dtype)

#         ys = torch.linspace(-1.0, 1.0, block_h, device=device)
#         xs = torch.linspace(-1.0, 1.0, block_w, device=device)
#         grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
#         coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
#         sh = torch.cdist(coords, coords, p=2)

#         self._cached_hr_block_shape = (block_h, block_w)
#         self._cached_hr_sh_block = sh
#         return sh.to(dtype)

#     def _apply_attention(
#         self,
#         features: torch.Tensor,        # (N,C) or (B,N,C)
#         hr_shape: tuple[int, int],
#     ) -> torch.Tensor:
#         if not self.use_attention or len(self.attention_blocks) == 0:
#             return features

#         Hr, Wr = hr_shape
#         batched = features.dim() == 3
#         if not batched:
#             features = features.unsqueeze(0)

#         B, N, C = features.shape
#         if N != Hr * Wr:
#             raise ValueError(f"Expected N={Hr*Wr} from hr_shape, got {N}")

#         block_h = min(self.hr_attn_block_size, Hr)
#         block_w = min(self.hr_attn_block_size, Wr)
#         pad_h = (-Hr) % block_h
#         pad_w = (-Wr) % block_w
#         Hr_pad, Wr_pad = Hr + pad_h, Wr + pad_w

#         feat = features
#         if pad_h > 0 or pad_w > 0:
#             feat_2d = feat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
#             feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
#             feat = feat_2d.permute(0, 2, 3, 1).reshape(B, Hr_pad * Wr_pad, C)

#         sh_block = self._get_hr_sh_block(block_h, block_w, feat.device, feat.dtype)
#         for block in self.attention_blocks:
#             if self.hr_attn_checkpoint and self.training and feat.requires_grad:
#                 def _run_block(x: torch.Tensor, _block=block) -> torch.Tensor:
#                     return _block(x, sh_block, Hr_pad, Wr_pad, block_h, block_w)
#                 delta = checkpoint(_run_block, feat, use_reentrant=True)
#             else:
#                 delta = block(feat, sh_block, Hr_pad, Wr_pad, block_h, block_w)
#             feat = feat + delta

#         if pad_h > 0 or pad_w > 0:
#             feat = feat.reshape(B, Hr_pad, Wr_pad, C)[:, :Hr, :Wr, :].reshape(B, Hr * Wr, C)

#         if not batched:
#             feat = feat.squeeze(0)
#         return feat


# def _imshow(ax, t: torch.Tensor, title: str, cmap: str = "viridis"):
#     im = ax.imshow(t.numpy(), cmap=cmap)
#     ax.set_title(title)
#     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# def step_and_plot_apply_attention(
#     module: DemoHRModule,
#     features: torch.Tensor,           # (N,C) or (B,N,C)
#     hr_shape: tuple[int, int],
#     sample_idx: int = 0,
#     block_row: int = 0,
#     block_col: int = 0,
#     layer_to_plot: int = 0,
# ):
#     """
#     Step through _apply_attention and show plots:
#     1) distance matrix + center/corner distance maps
#     2) attention matrix + center/corner query attention maps for selected layer/block
#     3) feature-norm maps before/after
#     """
#     print("\n=== _apply_attention TRACE START ===")
#     Hr, Wr = hr_shape
#     batched = features.dim() == 3
#     x_in = features if batched else features.unsqueeze(0)
#     B, N, C = x_in.shape
#     print(f"input shape={tuple(features.shape)}  batched={batched}")
#     print(f"Hr={Hr}, Wr={Wr}, N={N}, expected N={Hr*Wr}")
#     if N != Hr * Wr:
#         raise ValueError(f"Expected N={Hr*Wr}, got {N}")

#     block_h = min(module.hr_attn_block_size, Hr)
#     block_w = min(module.hr_attn_block_size, Wr)
#     pad_h = (-Hr) % block_h
#     pad_w = (-Wr) % block_w
#     Hr_pad, Wr_pad = Hr + pad_h, Wr + pad_w
#     print(f"block_h={block_h}, block_w={block_w}, pad_h={pad_h}, pad_w={pad_w}, Hr_pad={Hr_pad}, Wr_pad={Wr_pad}")

#     feat = x_in
#     if pad_h > 0 or pad_w > 0:
#         feat_2d = feat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
#         feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
#         feat = feat_2d.permute(0, 2, 3, 1).reshape(B, Hr_pad * Wr_pad, C)
#         print(f"padded feat shape={tuple(feat.shape)}")
#     else:
#         print("no padding needed")

#     sh_block = module._get_hr_sh_block(block_h, block_w, feat.device, feat.dtype).detach().cpu()
#     print(f"sh_block shape={tuple(sh_block.shape)} min={sh_block.min():.4f} max={sh_block.max():.4f}")

#     num_bh = Hr_pad // block_h
#     num_bw = Wr_pad // block_w
#     block_row = min(block_row, num_bh - 1)
#     block_col = min(block_col, num_bw - 1)
#     sample_idx = min(sample_idx, B - 1)
#     bb_idx = sample_idx * (num_bh * num_bw) + block_row * num_bw + block_col
#     print(f"selected sample={sample_idx}, block_row={block_row}, block_col={block_col}, bb_idx={bb_idx}")

#     dbg_layer = None
#     for i, block in enumerate(module.attention_blocks):
#         delta, dbg = block.forward_debug(feat, sh_block.to(feat.device), Hr_pad, Wr_pad, block_h, block_w, bb_idx)
#         print(
#             f"block {i}: delta_norm={delta.norm().item():.6f}, "
#             f"feat_norm_before={feat.norm().item():.6f}"
#         )
#         feat = feat + delta
#         print(f"block {i}: feat_norm_after={feat.norm().item():.6f}")
#         if i == layer_to_plot:
#             dbg_layer = dbg

#     feat_out = feat
#     if pad_h > 0 or pad_w > 0:
#         feat_out = feat_out.reshape(B, Hr_pad, Wr_pad, C)[:, :Hr, :Wr, :].reshape(B, Hr * Wr, C)

#     out = feat_out if batched else feat_out.squeeze(0)
#     print(f"output shape={tuple(out.shape)}")
#     print("=== _apply_attention TRACE END ===\n")

#     # ---------- Plots ----------
#     Nb = block_h * block_w
#     center_idx = (block_h // 2) * block_w + (block_w // 2)
#     corner_idx = 0

#     # Figure 1: distance and attention
#     fig1, axes = plt.subplots(2, 3, figsize=(14, 9))

#     _imshow(axes[0, 0], sh_block, "Block Pairwise Distance Matrix `sh`", cmap="viridis")
#     _imshow(
#         axes[0, 1],
#         sh_block[center_idx].reshape(block_h, block_w),
#         f"Distance from Center Query (idx={center_idx})",
#         cmap="magma",
#     )
#     _imshow(
#         axes[0, 2],
#         sh_block[corner_idx].reshape(block_h, block_w),
#         f"Distance from Corner Query (idx={corner_idx})",
#         cmap="magma",
#     )

#     if dbg_layer is None:
#         raise RuntimeError(f"layer_to_plot={layer_to_plot} out of range")

#     attn_sel = dbg_layer["attn_sel"]
#     _imshow(axes[1, 0], attn_sel, f"Attention Matrix (Layer {layer_to_plot})", cmap="viridis")
#     _imshow(
#         axes[1, 1],
#         attn_sel[center_idx].reshape(block_h, block_w),
#         f"Attention from Center Query (idx={center_idx})",
#         cmap="plasma",
#     )
#     _imshow(
#         axes[1, 2],
#         attn_sel[corner_idx].reshape(block_h, block_w),
#         f"Attention from Corner Query (idx={corner_idx})",
#         cmap="plasma",
#     )

#     for ax in axes.reshape(-1):
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")

#     plt.tight_layout()

#     # Figure 2: feature norm before vs after
#     before_map = x_in[sample_idx].reshape(Hr, Wr, C).norm(dim=-1).detach().cpu()
#     after_map = feat_out[sample_idx].reshape(Hr, Wr, C).norm(dim=-1).detach().cpu()
#     diff_map = after_map - before_map

#     fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
#     _imshow(axes2[0], before_map, "Feature Norm Before Attention", cmap="cividis")
#     _imshow(axes2[1], after_map, "Feature Norm After Attention", cmap="cividis")
#     _imshow(axes2[2], diff_map, "Norm Difference (After - Before)", cmap="coolwarm")
#     plt.tight_layout()

#     plt.show()
#     return out


# def main():
#     torch.manual_seed(7)

#     # Use a size that forces padding to visualize that behavior
#     B, Hr, Wr, C = 1, 5, 7, 16
#     x = torch.randn(B, Hr * Wr, C)

#     model = DemoHRModule(
#         channels=C,
#         num_blocks=2,
#         block_size=4,
#         use_checkpoint=False,
#         zero_init_out=False,   # set True to mimic near-identity init
#     ).eval()

#     y = step_and_plot_apply_attention(
#         model,
#         x,
#         (Hr, Wr),
#         sample_idx=0,
#         block_row=0,
#         block_col=0,
#         layer_to_plot=0,
#     )

#     y_direct = model._apply_attention(x, (Hr, Wr))
#     print("max |trace - direct| =", (y - y_direct).abs().max().item())


# if __name__ == "__main__":
#     main()
