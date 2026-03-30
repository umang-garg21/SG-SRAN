
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
- BoundaryAwareAttentionUpsampler: boundary-guided grain-attention upsampling
- AttentionBlock: block-local equivariant self-attention
- GrainAttention: grain-local equivariant self-attention (HR-boundary aware)
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


    def forward_full(self, quats_passive: torch.Tensor) -> torch.Tensor:
        """
        Encode quaternions to full irreps (all invariant and equivariant features).
        """
        q = self._to_embedding_device(quats_passive)
        return self.embedding.forward_irreps_passive(q, active_only=False)

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

        # Local optimization (Adam) to refine quaternions
        k = q0.shape[1]
        u = nn.Parameter(q0.clone())
        opt = torch.optim.Adam([u], lr=self.lr)

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

class EquivariantSpatialConv(nn.Module):
    """
    Equivariant spatial convolution for irrep-valued feature maps.
    Uses e3nn's FullyConnectedTensorProduct for channel mixing.
    Optionally includes a residual connection.
    """
    def __init__(
        self,
        kernel_size: int = 3,
        irreps_in: Irreps | str = "1x4e",
        irreps_out: Irreps | str | None = None,
        use_residual: bool = False,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding = (self.kernel_size // 2) * self.dilation

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out) if irreps_out is not None else self.irreps_in
        self.in_dim = int(self.irreps_in.dim)
        self.out_dim = int(self.irreps_out.dim)
        self.use_residual = bool(use_residual)

        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_out,
            shared_weights=True,  # Share weights across all positions for parameter efficiency and regularization
        )
        self.residual_proj: o3.Linear | None = None
        if self.use_residual and (self.irreps_in != self.irreps_out):
            self.residual_proj = o3.Linear(self.irreps_in, self.irreps_out)
        self.spatial_weights = nn.Parameter(
            torch.ones(self.kernel_size, self.kernel_size) / (self.kernel_size * self.kernel_size)
        )

    def forward(self, features: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
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
        # Unfolding extracts local spatial patches for convolution.
        # Each patch is weighted and aggregated to form the neighborhood context.
        if self.dilation == 1:
            patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1) # Shape: (B, C, H, W, k, k)
        else:
            # F.unfold supports dilation; reshape output back to (B, C, H, W, k, k)
            patches = F.unfold(feat_padded, kernel_size=self.kernel_size, dilation=self.dilation, padding=0, stride=1)
            patches = patches.view(B, C, self.kernel_size, self.kernel_size, H * W).permute(0, 1, 4, 2, 3).reshape(B, C, H, W, self.kernel_size, self.kernel_size)
        w = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
        neigh = (patches * w).sum(dim=(-1, -2))

        feat_flat = features.reshape(B * N, C) # Shape: (B*N, C)
        neigh_flat = neigh.permute(0, 2, 3, 1).reshape(B * N, C) # Shape: (B*N, C)
        out = self.tp(feat_flat, neigh_flat) # Take a tensor product of the central feature and the neighborhood context to produce the output feature.
 
        if self.use_residual:
            if self.residual_proj is None:
                out = out + feat_flat
                # If the input and output irreps are the same, we can directly add the input features as a residual connection.
            else:
                out = out + self.residual_proj(feat_flat) 
                # If the input and output irreps differ, we project the input features to the output space before adding them as a residual connection.

        out = out.reshape(B, N, self.out_dim)
        if not batched:
            out = out.squeeze(0)
        return out

class EquivariantTransposeConv(nn.Module):
    """
    Equivariant upsampler for irrep-valued feature maps.

    Design:
    1) Spatial upsample with depthwise transpose-conv (per irrep copy)
    2) Build local spatial context (weighted neighborhood average)
    3) Mix (feature, context) via an e3nn tensor product

    Important constraint:
    Transpose-conv kernels are tied across all m-channels inside each irrep copy
    so the upsample operator does not treat m components differently.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        upsample_factor: int | tuple[int, int] = 4,
        transpose_overlap: int = 2,
        use_residual: bool = True,
        irreps_in: Irreps | str = "1x4e",
        irreps_out: Irreps | str | None = None,
    ):
        super().__init__()
        if isinstance(upsample_factor, (list, tuple)):
            self.upsample_factor = (int(upsample_factor[0]), int(upsample_factor[1]))
        else:
            self.upsample_factor = (int(upsample_factor), int(upsample_factor))
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.use_residual = bool(use_residual)

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out) if irreps_out is not None else self.irreps_in
        self.in_dim = int(self.irreps_in.dim)
        self.out_dim = int(self.irreps_out.dim)
        C = self.in_dim

        r_h, r_w = self.upsample_factor
        self.transpose_kernel_size = (
            int(r_h + int(transpose_overlap)),
            int(r_w + int(transpose_overlap)),
        )
        self.transpose_padding = (
            int((self.transpose_kernel_size[0] - r_h) // 2),
            int((self.transpose_kernel_size[1] - r_w) // 2),
        )

        # To preserve irrep structure, each irrep copy uses one shared spatial kernel
        # across all of its m channels (2l+1 entries). This avoids treating m channels
        # differently during upsampling.
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
        # Learn one kernel per irrep copy (not per channel).
        self.transpose_kernels = nn.Parameter(
            torch.empty(self.num_irrep_copies, 1, self.transpose_kernel_size[0], self.transpose_kernel_size[1])
        )
        with torch.no_grad():
            self._init_bilinear()

        self.spatial_weights = nn.Parameter(
            torch.ones(self.kernel_size, self.kernel_size) / (self.kernel_size * self.kernel_size)
        )
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
        # Bilinear-style initialization gives stable interpolation behavior at start.
        # For anisotropic factors (r_h, r_w), each axis gets its own 1D bilinear kernel;
        # the 2D kernel is their outer product.
        r_h, r_w = self.upsample_factor
        k_h, k_w = self.transpose_kernel_size

        def _make_1d(k: int, r: int) -> torch.Tensor:
            v = torch.zeros(k)
            center = (k - 1) / 2.0
            for i in range(k):
                v[i] = max(0.0, 1.0 - abs(i - center) / r)
            return v

        bilinear_h = _make_1d(k_h, r_h)
        bilinear_w = _make_1d(k_w, r_w)
        bilinear_2d = bilinear_h.unsqueeze(1) * bilinear_w.unsqueeze(0)
        s = bilinear_2d.sum()
        if s > 0:
            bilinear_2d = bilinear_2d / s
        self.transpose_kernels.data[:] = bilinear_2d.unsqueeze(0).unsqueeze(0)

    def _expanded_transpose_weight(self) -> torch.Tensor:
        # Expand copy-wise kernels to channel-wise kernels by indexing with
        # channel_to_copy_idx; all m channels of one copy receive the same kernel.
        return torch.index_select(self.transpose_kernels, 0, self.channel_to_copy_idx)

    def forward(
        self,
        features: torch.Tensor,
        img_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        H, W = img_shape
        r_h, r_w = self.upsample_factor
        Hr, Wr = H * r_h, W * r_w

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
        # Depthwise transpose-conv (groups=C): each channel is upsampled by its
        # assigned kernel; tying across m channels is handled by up_weight.
        feat_hr = torch.nn.functional.conv_transpose2d(
            feat_img,
            up_weight,
            bias=None,
            stride=(r_h, r_w),
            padding=self.transpose_padding,
            output_padding=0,
            groups=C,
        )[:, :, :Hr, :Wr]

        feat_padded = F.pad(feat_hr, [self.padding] * 4, mode="replicate")
        patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        w = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
        # Context tensor: local weighted neighborhood summary at HR resolution.
        context = (patches * w).sum(dim=(-1, -2))

        N = Hr * Wr
        feat_flat = feat_hr.permute(0, 2, 3, 1).reshape(B * N, C)
        context_flat = context.permute(0, 2, 3, 1).reshape(B * N, C)
        # Equivariant channel mixing between upsampled features and local context.
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

class BoundaryAwareAttentionUpsampler(nn.Module):
    """
    Boundary-aware equivariant upsampler for irrep-valued feature maps.

    Design:
    1) Receive precomputed HR/LR grain correspondence maps
    2) For each HR pixel, attend directly to LR features in its mapped LR grain
       (with invariant feature similarity + spatial distance bias)
    3) Build local HR context from the attended HR seed features
    4) Mix (seed, local-context) with equivariant tensor product
       (boundary preprocessing is handled by the parent model forward path)
    """

    def __init__(
        self,
        kernel_size: int = 3,
        upsample_factor: int | tuple[int, int] = 4,
        use_residual: bool = True,
        use_boundary_gate: bool = False,
        irreps_in: Irreps | str = "1x4e",
        irreps_out: Irreps | str | None = None,
        boundary_threshold: float = 0.5,
        boundary_smooth_sigma: float = 0.8,
        boundary_smooth_iters: int = 3,
        boundary_sdf_shift: float = 0.2,
    ):
        super().__init__()
        if isinstance(upsample_factor, (list, tuple)):
            self.upsample_factor = (int(upsample_factor[0]), int(upsample_factor[1]))
        else:
            self.upsample_factor = (int(upsample_factor), int(upsample_factor))
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")
        self.padding = self.kernel_size // 2
        self.use_residual = bool(use_residual)
        self.use_boundary_gate = bool(use_boundary_gate)
        self.boundary_threshold = float(boundary_threshold)
        self.boundary_smooth_sigma = float(boundary_smooth_sigma)
        self.boundary_smooth_iters = int(boundary_smooth_iters)
        self.boundary_sdf_shift = float(boundary_sdf_shift)
        # Thin-structure protection defaults (pipeline-only, deterministic).
        self.tiny_component_max_pixels = 2
        self.narrow_component_max_width = 2
        self.narrow_region_shift_scale = 0.05
        # Learnable temperature for invariant grain-attention logits.
        self.log_grain_attn_temp = nn.Parameter(torch.tensor(0.0))
        # Scalar positional bias on LR-space distance, same idea as AttentionBlock.
        self.pos_bias = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.pos_bias.weight)
        nn.init.zeros_(self.pos_bias.bias)

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out) if irreps_out is not None else self.irreps_in
        self.in_dim = int(self.irreps_in.dim)
        self.out_dim = int(self.irreps_out.dim)

        self.spatial_weights = nn.Parameter(
            torch.ones(self.kernel_size, self.kernel_size) / (self.kernel_size * self.kernel_size)
        )
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_out,
            shared_weights=True,
        )
        self.residual_proj: o3.Linear | None = None
        if self.use_residual and (self.irreps_in != self.irreps_out):
            self.residual_proj = o3.Linear(self.irreps_in, self.irreps_out)

    @staticmethod
    def _same_pad_tuple(kernel_size: int) -> tuple[int, int, int, int]:
        """
        Return (left, right, top, bottom) padding for stride-1 'same' output size.
        Works for both odd and even kernel sizes.
        """
        total = int(kernel_size) - 1
        if total < 0:
            raise ValueError(f"Invalid kernel_size={kernel_size}")
        left = total // 2
        right = total - left
        top = left
        bottom = right
        return (left, right, top, bottom)

    def _smooth_boundary_to_sdf_like_boundary_prep(
        self,
        boundary_lr: torch.Tensor,
        hr_shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Match boundary_prep.ipynb LR->HR boundary smoothing:
          1) nearest upsample to HR
          2) iterative 3x3 box smoothing
          3) Gaussian smoothing
          4) normalize to [0,1]
        Expects boundary_lr shape (B,1,H,W).
        """
        Hr, Wr = hr_shape
        x = F.interpolate(boundary_lr, size=(Hr, Wr), mode="nearest")

        box = torch.ones((1, 1, 3, 3), device=x.device, dtype=x.dtype) / 9.0
        for _ in range(max(0, int(self.boundary_smooth_iters))):
            x = F.conv2d(x, box, padding=1)

        sigma = max(1e-6, float(self.boundary_smooth_sigma))
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
        half = size // 2
        coords = torch.arange(size, device=x.device, dtype=x.dtype) - float(half)
        g1 = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        g1 = g1 / g1.sum().clamp_min(1e-8)
        g2 = torch.outer(g1, g1)[None, None]
        x = F.conv2d(x, g2, padding=half)

        x = x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return x.clamp(0.0, 1.0)

    def _format_lr_boundary_map(
        self,
        lr_boundary_map: torch.Tensor,
        batch_size: int,
        lr_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Normalize LR boundary layout to (B,1,H,W) and clamp to [0,1].
        Enforces strict batch/spatial consistency (no resizing/broadcasting).
        """
        H, W = lr_shape
        bmap = lr_boundary_map
        if bmap.dim() == 2:
            bmap = bmap.unsqueeze(0).unsqueeze(0)
        elif bmap.dim() == 3:
            bmap = bmap.unsqueeze(1)
        elif bmap.dim() == 4:
            if bmap.shape[1] != 1 and bmap.shape[-1] == 1:
                bmap = bmap.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"lr_boundary_map must be 2D/3D/4D, got shape {tuple(lr_boundary_map.shape)}"
            )

        assert bmap.shape[1] == 1, (
            f"Expected one LR boundary channel, got shape {tuple(bmap.shape)}"
        )
        assert bmap.shape[0] == batch_size, (
            f"LR boundary batch mismatch: got {bmap.shape[0]}, expected {batch_size}"
        )
        assert bmap.shape[-2:] == (H, W), (
            f"LR boundary spatial mismatch: got {tuple(bmap.shape[-2:])}, expected {(H, W)}"
        )

        return bmap.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    @staticmethod
    def _thin_boundary_from_labels_2d(labels_2d: torch.Tensor) -> torch.Tensor:
        """
        One-sided 1-pixel boundary mask from integer labels.
        """
        if labels_2d.ndim != 2:
            raise ValueError(f"Expected (H,W), got {tuple(labels_2d.shape)}")
        b = torch.zeros_like(labels_2d, dtype=torch.bool)
        b[:, 1:] |= labels_2d[:, 1:] != labels_2d[:, :-1]
        b[1:, :] |= labels_2d[1:, :] != labels_2d[:-1, :]
        return b

    @staticmethod
    def _fill_unlabeled_pixels_4n(labels_2d: torch.Tensor) -> torch.Tensor:
        """
        Fill unlabeled pixels (-1) by iterative 4-neighbor propagation.
        """
        if labels_2d.ndim != 2:
            raise ValueError(f"Expected (H,W), got {tuple(labels_2d.shape)}")
        out = labels_2d.clone().long()
        H, W = out.shape
        max_iter = int(H + W + 4)
        for _ in range(max_iter):
            unknown = out < 0
            if not bool(unknown.any()):
                break
            new = out.clone()

            cand = torch.full_like(out, -1)
            cand[1:, :] = out[:-1, :]      # up
            fill = (new < 0) & (cand >= 0)
            new = torch.where(fill, cand, new)

            cand = torch.full_like(out, -1)
            cand[:-1, :] = out[1:, :]      # down
            fill = (new < 0) & (cand >= 0)
            new = torch.where(fill, cand, new)

            cand = torch.full_like(out, -1)
            cand[:, 1:] = out[:, :-1]      # left
            fill = (new < 0) & (cand >= 0)
            new = torch.where(fill, cand, new)

            cand = torch.full_like(out, -1)
            cand[:, :-1] = out[:, 1:]      # right
            fill = (new < 0) & (cand >= 0)
            new = torch.where(fill, cand, new)

            if torch.equal(new, out):
                break
            out = new

        if bool((out < 0).any()):
            out = torch.where(out < 0, torch.zeros_like(out), out)
        return out

    @staticmethod
    def _component_ids_by_size_and_width(
        labels_2d: torch.Tensor,
        num_components: int,
        *,
        tiny_size_max: int,
        narrow_width_max: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Classify LR interior components for thin-structure protection.
        Returns:
            tiny_ids: component ids with area <= tiny_size_max
            narrow_ids: component ids whose bbox min(H,W) <= narrow_width_max
        """
        if labels_2d.ndim != 2:
            raise ValueError(f"Expected (H,W), got {tuple(labels_2d.shape)}")
        if num_components <= 0:
            empty = torch.empty((0,), device=labels_2d.device, dtype=torch.long)
            return empty, empty

        valid = labels_2d[labels_2d >= 0].reshape(-1)
        counts = torch.zeros((num_components,), device=labels_2d.device, dtype=torch.long)
        if valid.numel() > 0:
            counts = torch.bincount(valid, minlength=int(num_components)).to(device=labels_2d.device, dtype=torch.long)

        tiny_ids = torch.nonzero(
            (counts > 0) & (counts <= int(tiny_size_max)),
            as_tuple=False,
        ).flatten().to(dtype=torch.long)

        narrow_list: list[int] = []
        for gid in range(int(num_components)):
            if int(counts[gid].item()) <= 0:
                continue
            ys, xs = (labels_2d == gid).nonzero(as_tuple=True)
            if ys.numel() == 0:
                continue
            h_span = int((ys.max() - ys.min() + 1).item())
            w_span = int((xs.max() - xs.min() + 1).item())
            if min(h_span, w_span) <= int(narrow_width_max):
                narrow_list.append(int(gid))

        if len(narrow_list) == 0:
            narrow_ids = torch.empty((0,), device=labels_2d.device, dtype=torch.long)
        else:
            narrow_ids = torch.as_tensor(narrow_list, device=labels_2d.device, dtype=torch.long)
        return tiny_ids, narrow_ids

    @staticmethod
    def _labels_mask_from_ids(labels_2d: torch.Tensor, component_ids: torch.Tensor) -> torch.Tensor:
        """
        Build a boolean mask for pixels whose label is in `component_ids`.
        """
        if labels_2d.ndim != 2:
            raise ValueError(f"Expected (H,W), got {tuple(labels_2d.shape)}")
        if component_ids.numel() == 0:
            return torch.zeros_like(labels_2d, dtype=torch.bool)
        mask = torch.zeros_like(labels_2d, dtype=torch.bool)
        for gid in component_ids.tolist():
            mask |= labels_2d == int(gid)
        return mask

    def _reinject_missing_components_from_lr(
        self,
        labels_hr: torch.Tensor,
        labels_lr_sparse: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reinject tiny LR components when remap drops them entirely in HR labels.
        """
        if labels_hr.ndim != 2 or labels_lr_sparse.ndim != 2:
            raise ValueError(
                f"Expected rank-2 labels, got {tuple(labels_hr.shape)} and {tuple(labels_lr_sparse.shape)}"
            )
        if component_ids.numel() == 0:
            return labels_hr

        Hr, Wr = labels_hr.shape
        repaired = labels_hr.clone()
        for gid in component_ids.tolist():
            gid_i = int(gid)
            if bool((repaired == gid_i).any()):
                continue
            seed_lr = labels_lr_sparse == gid_i
            if not bool(seed_lr.any()):
                continue
            seed_hr = F.interpolate(
                seed_lr.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                size=(Hr, Wr),
                mode="nearest",
            )[0, 0] > 0.5
            repaired[seed_hr] = gid_i
        return repaired

    def _remap_lr_labels_to_hr_via_sdf(
        self,
        labels_lr: torch.Tensor,
        sdf_hr: torch.Tensor,
        shift_scale_hr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate HR labels by normal-shift remap of LR labels using an HR field.
        """
        if labels_lr.ndim != 2 or sdf_hr.ndim != 2:
            raise ValueError(
                f"Expected labels_lr/sdf_hr rank-2, got {tuple(labels_lr.shape)} and {tuple(sdf_hr.shape)}"
            )
        H, W = labels_lr.shape
        Hr, Wr = sdf_hr.shape
        r_h, r_w = self.upsample_factor
        if Hr != H * r_h or Wr != W * r_w:
            raise ValueError(
                f"sdf_hr shape {(Hr, Wr)} incompatible with labels_lr {(H, W)} and upsample_factor {(r_h, r_w)}"
            )

        dy = torch.zeros_like(sdf_hr)
        dx = torch.zeros_like(sdf_hr)
        dy[1:-1, :] = 0.5 * (sdf_hr[2:, :] - sdf_hr[:-2, :])
        dy[0, :] = sdf_hr[1, :] - sdf_hr[0, :]
        dy[-1, :] = sdf_hr[-1, :] - sdf_hr[-2, :]
        dx[:, 1:-1] = 0.5 * (sdf_hr[:, 2:] - sdf_hr[:, :-2])
        dx[:, 0] = sdf_hr[:, 1] - sdf_hr[:, 0]
        dx[:, -1] = sdf_hr[:, -1] - sdf_hr[:, -2]

        norm = torch.sqrt(dx * dx + dy * dy + 1e-12)
        nx = dx / norm
        ny = dy / norm

        yy = torch.arange(Hr, device=sdf_hr.device, dtype=sdf_hr.dtype)
        xx = torch.arange(Wr, device=sdf_hr.device, dtype=sdf_hr.dtype)
        y_base = (yy + 0.5) / float(r_h) - 0.5
        x_base = (xx + 0.5) / float(r_w) - 0.5
        y_grid, x_grid = torch.meshgrid(y_base, x_base, indexing="ij")

        if shift_scale_hr is None:
            shift_scale_hr = torch.ones_like(sdf_hr)
        else:
            if shift_scale_hr.ndim != 2 or shift_scale_hr.shape != sdf_hr.shape:
                raise ValueError(
                    f"shift_scale_hr shape {tuple(shift_scale_hr.shape)} does not match sdf_hr {tuple(sdf_hr.shape)}"
                )
            shift_scale_hr = shift_scale_hr.to(device=sdf_hr.device, dtype=sdf_hr.dtype).clamp(0.0, 1.0)

        y_shift = y_grid - float(self.boundary_sdf_shift) * shift_scale_hr * ny
        x_shift = x_grid - float(self.boundary_sdf_shift) * shift_scale_hr * nx
        y_nn = torch.round(y_shift).clamp(0, H - 1).long()
        x_nn = torch.round(x_shift).clamp(0, W - 1).long()
        return labels_lr[y_nn, x_nn].long()

    @torch.no_grad()
    def _build_hr_1px_boundary_and_maps(
        self,
        boundary_lr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Boundary-prep-aligned pipeline:
          1) LR interior components from LR boundary map
          2) Dense LR labels (fill unlabeled boundary pixels)
          3) LR 1px boundary from dense LR labels
          4) HR soft field from LR 1px boundary (box+gaussian)
          5) HR labels by normal-shift remap
          6) Final HR 1px boundary from HR labels

        Returns:
            boundary_lr_1px: (B,1,H,W) float in {0,1}
            boundary_hr_1px: (B,1,Hr,Wr) float in {0,1}
            hr_to_lr_map: (B,Hr,Wr) long, -1 on HR boundary pixels
            lr_labels_masked: (B,H,W) long, -1 on LR boundary pixels
        """
        B, _, H, W = boundary_lr.shape
        r_h, r_w = self.upsample_factor
        Hr, Wr = H * r_h, W * r_w
        device = boundary_lr.device
        dtype = boundary_lr.dtype

        boundary_lr_1px = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
        boundary_hr_1px = torch.zeros((B, 1, Hr, Wr), device=device, dtype=dtype)
        hr_to_lr_map = torch.full((B, Hr, Wr), -1, device=device, dtype=torch.long)
        lr_labels_masked = torch.full((B, H, W), -1, device=device, dtype=torch.long)

        interior_lr = boundary_lr[:, 0] <= self.boundary_threshold

        for b in range(B):
            labels_lr_cpu, n_lr = self._connected_components_4(interior_lr[b])
            if n_lr <= 0:
                continue
            labels_lr = labels_lr_cpu.to(device=device).long()
            tiny_ids, narrow_ids = self._component_ids_by_size_and_width(
                labels_2d=labels_lr,
                num_components=n_lr,
                tiny_size_max=int(self.tiny_component_max_pixels),
                narrow_width_max=int(self.narrow_component_max_width),
            )
            labels_lr_dense = self._fill_unlabeled_pixels_4n(labels_lr)

            b_lr = self._thin_boundary_from_labels_2d(labels_lr_dense)
            boundary_lr_1px[b, 0] = b_lr.to(dtype=dtype)

            labels_lr_valid = labels_lr_dense.clone()
            labels_lr_valid[b_lr] = -1
            lr_labels_masked[b] = labels_lr_valid

            sdf_hr = self._smooth_boundary_to_sdf_like_boundary_prep(
                boundary_lr=boundary_lr_1px[b:b + 1],
                hr_shape=(Hr, Wr),
            )[0, 0]
            shift_scale_hr = None
            if narrow_ids.numel() > 0:
                narrow_lr_mask = self._labels_mask_from_ids(labels_lr_dense, narrow_ids)
                if bool(narrow_lr_mask.any()):
                    narrow_hr_mask = F.interpolate(
                        narrow_lr_mask.to(dtype=sdf_hr.dtype).unsqueeze(0).unsqueeze(0),
                        size=(Hr, Wr),
                        mode="nearest",
                    )[0, 0] > 0.5
                    shift_scale_hr = torch.ones((Hr, Wr), device=device, dtype=sdf_hr.dtype)
                    shift_scale_hr[narrow_hr_mask] = float(self.narrow_region_shift_scale)

            labels_hr = self._remap_lr_labels_to_hr_via_sdf(
                labels_lr_dense,
                sdf_hr,
                shift_scale_hr=shift_scale_hr,
            )
            labels_hr = self._reinject_missing_components_from_lr(labels_hr, labels_lr, tiny_ids)
            b_hr = self._thin_boundary_from_labels_2d(labels_hr)
            boundary_hr_1px[b, 0] = b_hr.to(dtype=dtype)

            hr_map = labels_hr.clone()
            hr_map[b_hr] = -1
            hr_to_lr_map[b] = hr_map

        return boundary_lr_1px, boundary_hr_1px, hr_to_lr_map, lr_labels_masked

    @staticmethod
    def _connected_components_4(mask_2d: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        4-connected components on a binary mask.
        Returns labels in {-1, 0, ..., K-1} and component count K.
        """
        m = mask_2d.detach().cpu().numpy().astype(np.bool_)
        H, W = m.shape
        labels = np.full((H, W), -1, dtype=np.int64)
        gid = 0

        for y in range(H):
            for x in range(W):
                if (not m[y, x]) or labels[y, x] >= 0:
                    continue
                stack = [(y, x)]
                labels[y, x] = gid
                while stack:
                    cy, cx = stack.pop()
                    if cy > 0 and m[cy - 1, cx] and labels[cy - 1, cx] < 0:
                        labels[cy - 1, cx] = gid
                        stack.append((cy - 1, cx))
                    if cy + 1 < H and m[cy + 1, cx] and labels[cy + 1, cx] < 0:
                        labels[cy + 1, cx] = gid
                        stack.append((cy + 1, cx))
                    if cx > 0 and m[cy, cx - 1] and labels[cy, cx - 1] < 0:
                        labels[cy, cx - 1] = gid
                        stack.append((cy, cx - 1))
                    if cx + 1 < W and m[cy, cx + 1] and labels[cy, cx + 1] < 0:
                        labels[cy, cx + 1] = gid
                        stack.append((cy, cx + 1))
                gid += 1

        return torch.from_numpy(labels), int(gid)

    def _compute_hr_to_lr_grain_maps(
        self,
        boundary_hr: torch.Tensor,
        boundary_lr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build grain-index maps used by both pooled and attention context methods.

        Returns:
            hr_to_lr_map: (B,Hr,Wr) long; each HR interior pixel stores mapped LR grain id, -1 if none.
            lr_labels: (B,H,W) long; LR grain labels from cleaned LR boundaries.
        """
        B, _, Hr, Wr = boundary_hr.shape
        r_h, r_w = self.upsample_factor
        H, W = Hr // r_h, Wr // r_w
        device = boundary_hr.device

        interior_hr = boundary_hr[:, 0] <= self.boundary_threshold
        if boundary_lr.shape[-2:] != (H, W):
            raise ValueError(
                f"boundary_lr shape {tuple(boundary_lr.shape[-2:])} incompatible with LR shape {(H, W)}"
            )
        interior_lr = boundary_lr[:, 0] <= self.boundary_threshold

        hr_to_lr_map = torch.full((B, Hr, Wr), -1, device=device, dtype=torch.long)
        lr_labels_all = torch.full((B, H, W), -1, device=device, dtype=torch.long)

        for b in range(B):
            labels_hr_cpu, n_hr = self._connected_components_4(interior_hr[b])
            labels_lr_cpu, n_lr = self._connected_components_4(interior_lr[b])
            labels_hr = labels_hr_cpu.to(device=device).long()
            labels_lr = labels_lr_cpu.to(device=device).long()
            lr_labels_all[b] = labels_lr

            if n_hr <= 0 or n_lr <= 0:
                continue

            for g in range(n_hr):
                hr_mask = labels_hr == g
                ys, xs = hr_mask.nonzero(as_tuple=True)
                if ys.numel() == 0:
                    continue
                y_lr = torch.div(ys, r_h, rounding_mode="floor").clamp(0, H - 1)
                x_lr = torch.div(xs, r_w, rounding_mode="floor").clamp(0, W - 1)
                cand = labels_lr[y_lr, x_lr]
                cand = cand[cand >= 0]
                if cand.numel() == 0:
                    continue
                ids, cnt = torch.unique(cand, return_counts=True)
                lr_gid = ids[torch.argmax(cnt)]
                hr_to_lr_map[b][hr_mask] = lr_gid

        return hr_to_lr_map, lr_labels_all

    def _invariant_grain_scores(
        self,
        q_feat: torch.Tensor,
        k_feat: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        O(3)-invariant similarity between query (HR) and key (LR) irrep features.
        Score is built from per-irrep-copy normalized dot products.
        """
        nq = int(q_feat.shape[0])
        nk = int(k_feat.shape[0])
        if nq == 0 or nk == 0:
            return q_feat.new_zeros((nq, nk))

        scores = q_feat.new_zeros((nq, nk))
        start = 0
        for mul, ir in self.irreps_in:
            mul = int(mul)
            d = int(ir.dim)
            n = mul * d
            q_blk = q_feat[:, start:start + n].reshape(nq, mul, d)
            k_blk = k_feat[:, start:start + n].reshape(nk, mul, d)
            q_blk = q_blk / q_blk.norm(dim=-1, keepdim=True).clamp_min(eps)
            k_blk = k_blk / k_blk.norm(dim=-1, keepdim=True).clamp_min(eps)
            sim = torch.einsum("qmd,kmd->qk", q_blk, k_blk) / float(max(1, mul))
            scores = scores + sim
            start += n

        return scores

    def _hr_seed_from_lr_grain_attention(
        self,
        feat_lr_img: torch.Tensor,
        hr_to_lr_map: torch.Tensor,
        lr_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build HR seed features directly from LR grain attention.
          - Query for each HR pixel: parent LR feature (floor(y/r), floor(x/r))
          - Keys/values: LR pixels inside mapped LR grain
          - Scores: invariant feature similarity + positional distance bias
        """
        B, C, H, W = feat_lr_img.shape
        Hr, Wr = hr_to_lr_map.shape[-2], hr_to_lr_map.shape[-1]
        attn_temp = torch.exp(self.log_grain_attn_temp)
        r_h, r_w = self.upsample_factor

        y_hr = torch.arange(Hr, device=feat_lr_img.device, dtype=torch.long)
        x_hr = torch.arange(Wr, device=feat_lr_img.device, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(y_hr, x_hr, indexing="ij")
        parent_y = torch.div(grid_y, r_h, rounding_mode="floor").clamp(0, H - 1)
        parent_x = torch.div(grid_x, r_w, rounding_mode="floor").clamp(0, W - 1)
        parent_idx = (parent_y * W + parent_x).reshape(-1)

        # HR pixel locations represented in LR-coordinate space (float).
        qy_lr = (grid_y.reshape(-1).to(feat_lr_img.dtype) + 0.5) / float(r_h) - 0.5
        qx_lr = (grid_x.reshape(-1).to(feat_lr_img.dtype) + 0.5) / float(r_w) - 0.5
        q_coords_all = torch.stack([qx_lr, qy_lr], dim=-1)

        seed_batches: list[torch.Tensor] = []
        valid_batches: list[torch.Tensor] = []

        for b in range(B):
            k_all = feat_lr_img[b].permute(1, 2, 0).reshape(H * W, C)
            map_flat = hr_to_lr_map[b].reshape(-1)
            lr_flat = lr_labels[b].reshape(-1)

            # Default fallback: parent LR feature.
            seed_base = k_all[parent_idx]  # (Hr*Wr, C)
            # Accumulate attended overrides and override mask out-of-place.
            seed_ctx = torch.zeros_like(seed_base)
            mask_flat = torch.zeros((Hr * Wr, 1), device=seed_base.device, dtype=seed_base.dtype)

            grain_ids = torch.unique(map_flat[map_flat >= 0])
            for gid in grain_ids.tolist():
                q_idx = (map_flat == int(gid)).nonzero(as_tuple=False).squeeze(1)
                k_idx = (lr_flat == int(gid)).nonzero(as_tuple=False).squeeze(1)
                if q_idx.numel() == 0 or k_idx.numel() == 0:
                    continue

                parent_idx_q = parent_idx[q_idx]
                q = k_all[parent_idx_q]
                k = k_all[k_idx]

                # Query hardening clause:
                # If the HR query's parent LR pixel is boundary/mismatch, replace the
                # query feature with the nearest interior LR feature from the same grain.
                parent_gid_q = lr_flat[parent_idx_q]
                bad_parent = parent_gid_q != int(gid)  # covers boundary (-1) and mismatch
                k_y = torch.div(k_idx, W, rounding_mode="floor").to(feat_lr_img.dtype)
                k_x = torch.remainder(k_idx, W).to(feat_lr_img.dtype)
                k_coords = torch.stack([k_x, k_y], dim=-1)
                if bool(bad_parent.any()):
                    bad_local = bad_parent.nonzero(as_tuple=False).squeeze(1)
                    q_bad_coords = q_coords_all[q_idx[bad_local]]
                    d_bad = torch.cdist(q_bad_coords, k_coords, p=2)
                    nn_bad = torch.argmin(d_bad, dim=-1)
                    q = q.clone()
                    q[bad_local] = k[nn_bad]

                scores = attn_temp * self._invariant_grain_scores(q, k)
                d = torch.cdist(q_coords_all[q_idx], k_coords, p=2)
                scores = scores + self.pos_bias(d.unsqueeze(-1)).squeeze(-1)
                attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
                ctx = attn @ k

                # Avoid in-place indexed assignment on autograd leaf tensors.
                seed_ctx = torch.index_copy(seed_ctx, 0, q_idx, ctx)
                ones = torch.ones((q_idx.numel(), 1), device=seed_base.device, dtype=seed_base.dtype)
                mask_flat = torch.index_copy(mask_flat, 0, q_idx, ones)

            seed_flat = seed_base * (1.0 - mask_flat) + seed_ctx
            valid_flat = mask_flat.squeeze(1) > 0.5

            seed_batches.append(seed_flat.reshape(Hr, Wr, C).permute(2, 0, 1))
            valid_batches.append(valid_flat.reshape(Hr, Wr))

        seed_hr = torch.stack(seed_batches, dim=0)
        valid_hr = torch.stack(valid_batches, dim=0).unsqueeze(1)
        return seed_hr, valid_hr

    def _hr_seed_from_lr_grain_attention_vectorized(
        self,
        feat_lr_img: torch.Tensor,
        hr_to_lr_map: torch.Tensor,
        lr_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build HR seed features directly from LR grain attention.
          - Query for each HR pixel: parent LR feature (floor(y/r), floor(x/r))
          - Keys/values: LR pixels inside mapped LR grain
          - Scores: invariant feature similarity + positional distance bias
        """
        B, C, H, W = feat_lr_img.shape
        Hr, Wr = hr_to_lr_map.shape[-2], hr_to_lr_map.shape[-1]
        r_h, r_w = self.upsample_factor
        device = feat_lr_img.device
        dtype = feat_lr_img.dtype

        # Flatten LR features across batch for global token indexing.
        lr_hw = H * W
        hr_hw = Hr * Wr
        feat_lr_flat = feat_lr_img.permute(0, 2, 3, 1).reshape(B * lr_hw, C)

        y_hr = torch.arange(Hr, device=feat_lr_img.device, dtype=torch.long)
        x_hr = torch.arange(Wr, device=feat_lr_img.device, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(y_hr, x_hr, indexing="ij")
        parent_y = torch.div(grid_y, r_h, rounding_mode="floor").clamp(0, H - 1)
        parent_x = torch.div(grid_x, r_w, rounding_mode="floor").clamp(0, W - 1)
        parent_idx = (parent_y * W + parent_x).reshape(-1)  # (Hr*Wr,)

        # HR pixel locations represented in LR-coordinate space (float).
        qy_lr = (grid_y.reshape(-1).to(feat_lr_img.dtype) + 0.5) / float(r_h) - 0.5
        qx_lr = (grid_x.reshape(-1).to(feat_lr_img.dtype) + 0.5) / float(r_w) - 0.5
        q_coords_base = torch.stack([qx_lr, qy_lr], dim=-1)  # (Hr*Wr, 2)

        y_lr = torch.arange(H, device=device, dtype=dtype)
        x_lr = torch.arange(W, device=device, dtype=dtype)
        grid_y_lr, grid_x_lr = torch.meshgrid(y_lr, x_lr, indexing="ij")
        k_coords_base = torch.stack([grid_x_lr.reshape(-1), grid_y_lr.reshape(-1)], dim=-1)  # (H*W, 2)

        batch_ids_hr = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(hr_hw)
        parent_global_idx = parent_idx.repeat(B) + batch_ids_hr * lr_hw

        # Default seed for every HR pixel: parent LR feature.
        seed_flat = feat_lr_flat[parent_global_idx].clone()  # (B*Hr*Wr, C)
        valid_flat = torch.zeros(B * hr_hw, device=device, dtype=torch.bool)

        # Batch-disjoint global grain ids.
        lr_gid_local = lr_labels.reshape(B, lr_hw)
        hr_gid_local = hr_to_lr_map.reshape(B, hr_hw)
        if not bool((lr_gid_local >= 0).any()):
            seed_hr = seed_flat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
            valid_hr = valid_flat.reshape(B, 1, Hr, Wr)
            return seed_hr, valid_hr

        max_gid = int(lr_gid_local[lr_gid_local >= 0].max().item()) + 1
        offset = (torch.arange(B, device=device, dtype=torch.long) * max_gid).unsqueeze(1)
        lr_gid = torch.where(
            lr_gid_local >= 0,
            lr_gid_local + offset,
            torch.full_like(lr_gid_local, -1),
        ).reshape(-1)
        hr_gid = torch.where(
            hr_gid_local >= 0,
            hr_gid_local + offset,
            torch.full_like(hr_gid_local, -1),
        ).reshape(-1)

        q_valid = hr_gid >= 0
        k_valid = lr_gid >= 0
        if not bool(q_valid.any()) or not bool(k_valid.any()):
            seed_hr = seed_flat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
            valid_hr = valid_flat.reshape(B, 1, Hr, Wr)
            return seed_hr, valid_hr

        q_gid = hr_gid[q_valid]
        k_gid = lr_gid[k_valid]
        q_out_idx = q_valid.nonzero(as_tuple=False).squeeze(1)

        q_feat = seed_flat[q_out_idx]
        k_feat = feat_lr_flat[k_valid]
        q_coords = q_coords_base.repeat(B, 1)[q_valid]
        k_coords = k_coords_base.repeat(B, 1)[k_valid]

        # Group tokens by grain id and pack to padded tensors.
        all_gid = torch.unique(torch.cat([q_gid, k_gid], dim=0))
        G = int(all_gid.numel())
        q_group = torch.searchsorted(all_gid, q_gid)
        k_group = torch.searchsorted(all_gid, k_gid)

        q_counts = torch.bincount(q_group, minlength=G)
        k_counts = torch.bincount(k_group, minlength=G)
        valid_group = (q_counts > 0) & (k_counts > 0)
        if not bool(valid_group.any()):
            seed_hr = seed_flat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
            valid_hr = valid_flat.reshape(B, 1, Hr, Wr)
            return seed_hr, valid_hr

        q_order = torch.argsort(q_group)
        k_order = torch.argsort(k_group)
        q_group_sorted = q_group[q_order]
        k_group_sorted = k_group[k_order]

        q_starts = torch.cumsum(q_counts, dim=0) - q_counts
        k_starts = torch.cumsum(k_counts, dim=0) - k_counts
        q_pos_sorted = torch.arange(q_group.numel(), device=device) - q_starts[q_group_sorted]
        k_pos_sorted = torch.arange(k_group.numel(), device=device) - k_starts[k_group_sorted]

        q_pos = torch.empty_like(q_group)
        k_pos = torch.empty_like(k_group)
        q_pos[q_order] = q_pos_sorted
        k_pos[k_order] = k_pos_sorted

        Qmax = int(q_counts.max().item())
        Kmax = int(k_counts.max().item())
        Q_pad = q_feat.new_zeros((G, Qmax, C))
        K_pad = k_feat.new_zeros((G, Kmax, C))
        Qc_pad = q_coords.new_zeros((G, Qmax, 2))
        Kc_pad = k_coords.new_zeros((G, Kmax, 2))
        Q_pad[q_group, q_pos] = q_feat
        K_pad[k_group, k_pos] = k_feat
        Qc_pad[q_group, q_pos] = q_coords
        Kc_pad[k_group, k_pos] = k_coords

        q_mask = torch.arange(Qmax, device=device)[None, :] < q_counts[:, None]
        k_mask = torch.arange(Kmax, device=device)[None, :] < k_counts[:, None]

        g_idx = valid_group.nonzero(as_tuple=False).squeeze(1)
        Qv = Q_pad[g_idx]
        Kv = K_pad[g_idx]
        Qcv = Qc_pad[g_idx]
        Kcv = Kc_pad[g_idx]
        q_mask_v = q_mask[g_idx]
        k_mask_v = k_mask[g_idx]

        attn_temp = torch.exp(self.log_grain_attn_temp)
        scores = attn_temp * self._invariant_grain_scores_grouped(Qv, Kv)
        dist = torch.cdist(Qcv, Kcv, p=2)
        scores = scores + self.pos_bias(dist.unsqueeze(-1)).squeeze(-1)

        invalid = (~q_mask_v[:, :, None]) | (~k_mask_v[:, None, :])
        scores = scores.masked_fill(invalid, -1e9)
        attn = torch.softmax(scores.float(), dim=-1).to(dtype)
        Ctx_v = attn @ Kv

        # Scatter attended outputs back to HR token order.
        group_to_compact = torch.full((G,), -1, device=device, dtype=torch.long)
        group_to_compact[g_idx] = torch.arange(g_idx.numel(), device=device, dtype=torch.long)
        q_has_ctx = valid_group[q_group]
        if bool(q_has_ctx.any()):
            q_out = q_out_idx[q_has_ctx]
            q_grp_comp = group_to_compact[q_group[q_has_ctx]]
            q_pos_sel = q_pos[q_has_ctx]
            seed_flat[q_out] = Ctx_v[q_grp_comp, q_pos_sel]
            valid_flat[q_out] = True

        seed_hr = seed_flat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
        valid_hr = valid_flat.reshape(B, 1, Hr, Wr)
        return seed_hr, valid_hr

    def forward(
        self,
        features: torch.Tensor,
        img_shape: tuple[int, int],
        hr_to_lr_map: torch.Tensor,
        lr_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Args:
            features: (B,N,C) or (N,C) LR irrep features, N=H*W.
            img_shape: LR spatial shape (H,W).
            hr_to_lr_map: (B,Hr,Wr) long map from HR interior pixels to LR grain ids.
            lr_labels: (B,H,W) long LR labels, with -1 on LR boundary pixels.
        """
        H, W = img_shape
        r_h, r_w = self.upsample_factor
        Hr, Wr = H * r_h, W * r_w

        batched = features.dim() == 3
        if not batched:
            features = features.unsqueeze(0)
        B = features.shape[0]
        C = features.shape[-1]
        if C != self.in_dim:
            raise ValueError(f"Expected feature dim {self.in_dim}, got {C}")
        if features.shape[1] != H * W:
            raise ValueError(f"Expected N={H*W}, got N={features.shape[1]}")
        if hr_to_lr_map.shape != (B, Hr, Wr):
            raise ValueError(
                f"Expected hr_to_lr_map shape {(B, Hr, Wr)}, got {tuple(hr_to_lr_map.shape)}"
            )
        if lr_labels.shape != (B, H, W):
            raise ValueError(
                f"Expected lr_labels shape {(B, H, W)}, got {tuple(lr_labels.shape)}"
            )

        feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)
        feat_hr, _ = self._hr_seed_from_lr_grain_attention(
            feat_lr_img=feat_img,
            hr_to_lr_map=hr_to_lr_map,
            lr_labels=lr_labels,
        )

        # Grain-aware local context:
        # only neighbors with the same HR grain id as the center pixel
        # are allowed to contribute to the context aggregation.
        pad = self._same_pad_tuple(self.kernel_size)
        feat_padded = F.pad(feat_hr, pad, mode="replicate")
        patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        grain_img = hr_to_lr_map.unsqueeze(1)
        grain_padded = F.pad(
            grain_img,
            pad,
            mode="constant",
            value=-1,
        )
        grain_patches = grain_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        center_gid = grain_img.unsqueeze(-1).unsqueeze(-1)
        same_grain = (grain_patches == center_gid) & (center_gid >= 0)

        # Keep neighborhood weights positive/normalized for stable large-kernel aggregation.
        w = torch.softmax(self.spatial_weights.reshape(-1), dim=0).view(
            1, 1, 1, 1, self.kernel_size, self.kernel_size
        )
        w_masked = w * same_grain.to(feat_hr.dtype)
        norm = w_masked.sum(dim=(-1, -2), keepdim=False)
        context = (patches * w_masked).sum(dim=(-1, -2)) / norm.clamp_min(1e-8)
        # For HR boundary/unmapped centers (no valid same-grain neighbors),
        # fallback to center seed feature to avoid cross-grain leakage.
        context = torch.where(norm > 0.0, context, feat_hr)

        N = Hr * Wr
        feat_flat = feat_hr.permute(0, 2, 3, 1).reshape(B * N, C)
        context_flat = context.permute(0, 2, 3, 1).reshape(B * N, C)
        # Equivariant channel mixing between upsampled features and local context.
        out = self.tp(feat_flat, context_flat)
        if self.use_residual:
            if self.residual_proj is None:
                out = out + feat_flat
            else:
                out = out + self.residual_proj(feat_flat)

        # Optional boundary-safe gating:
        # keep boundary/unmapped pixels on seed fallback when enabled.
        if self.use_boundary_gate:
            interior_flat = (hr_to_lr_map.reshape(B * N) >= 0).to(out.dtype).unsqueeze(1)
            if out.shape[-1] == feat_flat.shape[-1]:
                fallback_flat = feat_flat
            elif self.residual_proj is not None:
                fallback_flat = self.residual_proj(feat_flat)
            else:
                fallback_flat = out
            out = out * interior_flat + fallback_flat * (1.0 - interior_flat)

        out = out.reshape(B, N, self.out_dim)
        if not batched:
            out = out.squeeze(0)
        return out, (Hr, Wr)

class AttentionBlock(nn.Module):
    """
    Block-local equivariant self-attention for a configurable feature irreps.
    Applies dot-product attention within spatial blocks, with O(3)-invariant scores.
    """

    def __init__(
        self,
        irreps_feat: Irreps | str = "1x4e + 1x6e",
        num_channels: int = 8,
        tp_out_chunk_size: int | None = None,
    ):
        super().__init__()
        del tp_out_chunk_size  # Kept for config compatibility.

        self.num_channels= int(num_channels)
        self.irreps_feat = Irreps(irreps_feat)
        self.irreps_h = Irreps([(int(mul) * self.num_channels, ir) for mul, ir in self.irreps_feat])

        # Global attention temperature.
        self.log_s = nn.Parameter(torch.tensor(0.0))

        # Pairwise position bias: d_ij = ||pos_i - pos_j|| -> scalar add to score(i,j).
        self.pos_bias = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.pos_bias.weight)
        nn.init.zeros_(self.pos_bias.bias)

        # Channel expansion and equivariant hidden mixing.
        self.lin_in = IrrepsLinear(self.irreps_feat, self.irreps_h)
        self.tp_out = FullyConnectedTensorProduct(
            self.irreps_h, self.irreps_h, self.irreps_h, shared_weights=True
        )

        # Zero-init so block starts as near-identity residual.
        self.lin_out = IrrepsLinear(self.irreps_h, self.irreps_feat)
        with torch.no_grad():
            self.lin_out.weight.data.zero_()

        # Per-irrep-family slice boundaries for equivariant normalization.
        # Each entry covers the full (mul * ir_dim) block for one irrep type.
        # Normalizing the whole family together preserves the relative scale
        # between the mul copies, which encodes meaningful orientation information.
        self._irreps_norm_slices: list[tuple[int, int]] = [
            (s.start, s.stop)
            for s in self.irreps_feat.slices()
        ]

    def forward_old(
        self,
        feat: torch.Tensor,
        d_block: torch.Tensor,
        H: int,
        W: int,
        block_h: int,
        block_w: int,
    ) -> torch.Tensor:
        """
        Args:
            feat: (B, H*W, C_feat) features for padded spatial grid.
            d_block: (Nb, Nb) pairwise L2 distances within one block.
            H, W: padded spatial dimensions, multiples of block_h and block_w.
            block_h, block_w: block size in pixels.
        Returns:
            (B, H*W, C_feat) attention residual delta.
        """
        B, _, C_feat = feat.shape
        num_bh = H // block_h
        num_bw = W // block_w
        Nb = block_h * block_w
        Bb = B * num_bh * num_bw
        dtype = feat.dtype

        # Partition: (B, H*W, C_feat) -> (Bb, Nb, C_feat)
        feat_blocks = (
            feat.reshape(B, num_bh, block_h, num_bw, block_w, C_feat)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bb, Nb, C_feat)
        )

        # O(3)-invariant dot-product attention with per-irrep-family normalization.
        # Each irrep type (e.g. all mul copies of l=4e) is normalized as one block:
        # the full (mul * ir_dim) vector becomes unit-norm. This avoids mixing
        # scales across different irrep orders AND preserves the relative magnitudes
        # among the mul copies within each family (unlike per-copy normalization,
        # which would discard that intra-family scale information).
        f_n = torch.cat(
            [
                F.normalize(feat_blocks[:, :, start:end], dim=-1)
                for start, end in self._irreps_norm_slices
            ],
            dim=-1,
        )
        scores = torch.exp(self.log_s) * torch.bmm(f_n, f_n.transpose(-2, -1))

        pb = self.pos_bias(d_block.unsqueeze(-1)).squeeze(-1)
        scores = scores + pb.unsqueeze(0)

        attn = torch.softmax(scores.float(), dim=-1).to(dtype)

        feat_flat = feat_blocks.reshape(Bb * Nb, C_feat)
        h = self.lin_in(feat_flat).reshape(Bb, Nb, -1)
        ctx = torch.bmm(attn, h)
        h_out = self.tp_out(h.reshape(Bb * Nb, -1), ctx.reshape(Bb * Nb, -1)).reshape(Bb, Nb, -1)
        delta_blocks = self.lin_out(h_out.reshape(Bb * Nb, -1)).reshape(Bb, Nb, C_feat)

        # Reassemble: (Bb, Nb, C_feat) -> (B, H*W, C_feat)
        delta = (
            delta_blocks.reshape(B, num_bh, num_bw, block_h, block_w, C_feat)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, H * W, C_feat)
        )
        return delta

    def _invariant_scores(self, feat_blocks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Build O(3)-invariant attention scores from mixed-irrep features.

        Args:
            feat_blocks: (Bb, Nb, C_feat)
        Returns:
            scores: (Bb, Nb, Nb)
        """
        Bb, Nb, _ = feat_blocks.shape
        scores = feat_blocks.new_zeros(Bb, Nb, Nb)

        start = 0
        for mul, ir in self.irreps_feat:
            mul = int(mul)
            d = ir.dim
            n = mul * d

            # (Bb, Nb, mul*d) -> (Bb, Nb, mul, d)
            x = feat_blocks[..., start:start + n].reshape(Bb, Nb, mul, d)

            # Normalize each irrep copy independently
            x = x / x.norm(dim=-1,keepdim=True).clamp_min(eps)

            # Copywise invariant dot products:
            # (Bb, Nb, mul, d) x (Bb, Nb, mul, d) -> (Bb, Nb, Nb, mul)
            sim = torch.einsum("bimd,bjmd->bijm", x, x)

            # Average over multiplicity C so score scale does not grow with width
            sim = sim.mean(dim=-1)  # (Bb, Nb, Nb)

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
    ) -> torch.Tensor:
        """
        Args:
            feat: (B, H*W, C_feat) features for padded spatial grid
            d_block: (Nb, Nb) pairwise L2 distances within one block
            H, W: padded spatial dimensions, multiples of block_h and block_w
            block_h, block_w: block size in pixels

        Returns:
            delta: (B, H*W, C_feat) attention residual
        """
        B, HW, C_feat = feat.shape
        if HW != H * W:
            raise ValueError(f"Expected feat.shape[1] == H*W = {H*W}, got {HW}")

        num_bh = H // block_h
        num_bw = W // block_w
        Nb = block_h * block_w
        Bb = B * num_bh * num_bw
        dtype = feat.dtype

        # Partition: (B, H*W, C_feat) -> (Bb, Nb, C_feat)
        feat_blocks = (
            feat.reshape(B, num_bh, block_h, num_bw, block_w, C_feat)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bb, Nb, C_feat)
        )

        # O(3)-invariant attention logits from per-irrep-copy normalized similarities
        scores = torch.exp(self.log_s) * self._invariant_scores(feat_blocks)

        # Add scalar positional bias
        pb = self.pos_bias(d_block.unsqueeze(-1)).squeeze(-1)  # (Nb, Nb)
        scores = scores + pb.unsqueeze(0)  # (Bb, Nb, Nb)

        # Softmax in fp32 for stability, then cast back
        attn = torch.softmax(scores.float(), dim=-1).to(dtype)

        # Equivariant value path
        feat_flat = feat_blocks.reshape(Bb * Nb, C_feat)            # (Bb*Nb, C_feat)
        h = self.lin_in(feat_flat).reshape(Bb, Nb, -1)              # (Bb, Nb, C_h)

        # Context aggregation with invariant weights preserves equivariance
        ctx = torch.bmm(attn, h)                                    # (Bb, Nb, C_h)

        # Equivariant mixing
        h_out = self.tp_out(
            h.reshape(Bb * Nb, -1),
            ctx.reshape(Bb * Nb, -1),
        ).reshape(Bb, Nb, -1)

        delta_blocks = self.lin_out(h_out.reshape(Bb * Nb, -1)).reshape(Bb, Nb, C_feat)

        # Reassemble: (Bb, Nb, C_feat) -> (B, H*W, C_feat)
        delta = (
            delta_blocks.reshape(B, num_bh, num_bw, block_h, block_w, C_feat)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, H * W, C_feat)
        )

        return delta

class GrainAttention(nn.Module):
    """
    Grain-local equivariant self-attention on HR features.
    Grain membership is defined from the HR boundary map (1px boundary):
      - interior pixels attend only within their grain component
      - boundary pixels do not receive a grain-attention update
    """

    def __init__(
        self,
        irreps_feat: Irreps | str = "1x4e + 1x6e",
        num_channels: int = 8,
        tp_out_chunk_size: int | None = None,
        boundary_threshold: float = 0.5,
        upsample_factor: int | tuple[int, int] = 4,
    ):
        super().__init__()
        del tp_out_chunk_size  # Kept for config compatibility.

        self.num_channels = int(num_channels)
        self.irreps_feat = Irreps(irreps_feat)
        self.irreps_h = Irreps([(int(mul) * self.num_channels, ir) for mul, ir in self.irreps_feat])
        self.boundary_threshold = float(boundary_threshold)
        if isinstance(upsample_factor, (list, tuple)):
            self.upsample_factor = (int(upsample_factor[0]), int(upsample_factor[1]))
        else:
            self.upsample_factor = (int(upsample_factor), int(upsample_factor))

        # Global attention temperature.
        self.log_s = nn.Parameter(torch.tensor(0.0))

        # Scalar positional bias on pairwise in-grain distances.
        self.pos_bias = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.pos_bias.weight)
        nn.init.zeros_(self.pos_bias.bias)

        # Equivariant value path and output projection.
        self.lin_in = IrrepsLinear(self.irreps_feat, self.irreps_h)
        self.tp_out = FullyConnectedTensorProduct(
            self.irreps_h, self.irreps_h, self.irreps_h, shared_weights=True
        )
        self.lin_out = IrrepsLinear(self.irreps_h, self.irreps_feat)
        with torch.no_grad():
            self.lin_out.weight.data.zero_()

    @staticmethod
    def _connected_components_4(mask_2d: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        4-connected components on a binary mask.
        Returns labels in {-1, 0, ..., K-1} and component count K.
        """
        m = mask_2d.detach().cpu().numpy().astype(np.bool_)
        H, W = m.shape
        labels = np.full((H, W), -1, dtype=np.int64)
        gid = 0

        for y in range(H):
            for x in range(W):
                if (not m[y, x]) or labels[y, x] >= 0:
                    continue
                stack = [(y, x)]
                labels[y, x] = gid
                while stack:
                    cy, cx = stack.pop()
                    if cy > 0 and m[cy - 1, cx] and labels[cy - 1, cx] < 0:
                        labels[cy - 1, cx] = gid
                        stack.append((cy - 1, cx))
                    if cy + 1 < H and m[cy + 1, cx] and labels[cy + 1, cx] < 0:
                        labels[cy + 1, cx] = gid
                        stack.append((cy + 1, cx))
                    if cx > 0 and m[cy, cx - 1] and labels[cy, cx - 1] < 0:
                        labels[cy, cx - 1] = gid
                        stack.append((cy, cx - 1))
                    if cx + 1 < W and m[cy, cx + 1] and labels[cy, cx + 1] < 0:
                        labels[cy, cx + 1] = gid
                        stack.append((cy, cx + 1))
                gid += 1

        return torch.from_numpy(labels), int(gid)

    def _invariant_scores_pair(
        self,
        q_feat: torch.Tensor,
        k_feat: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        O(3)-invariant pairwise scores between query and key irrep features.
        """
        nq = int(q_feat.shape[0])
        nk = int(k_feat.shape[0])
        if nq == 0 or nk == 0:
            return q_feat.new_zeros((nq, nk))

        scores = q_feat.new_zeros((nq, nk))
        start = 0
        for mul, ir in self.irreps_feat:
            mul = int(mul)
            d = int(ir.dim)
            n = mul * d
            q_blk = q_feat[:, start:start + n].reshape(nq, mul, d)
            k_blk = k_feat[:, start:start + n].reshape(nk, mul, d)
            q_blk = q_blk / q_blk.norm(dim=-1, keepdim=True).clamp_min(eps)
            k_blk = k_blk / k_blk.norm(dim=-1, keepdim=True).clamp_min(eps)
            sim = torch.einsum("qmd,kmd->qk", q_blk, k_blk) / float(max(1, mul))
            scores = scores + sim
            start += n

        return scores

    def _invariant_scores_grouped(
        self,
        feat_grouped: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        O(3)-invariant self-attention scores for grouped features.
        Args:
            feat_grouped: (G, N, C_feat)
        Returns:
            scores: (G, N, N)
        """
        G, N, _ = feat_grouped.shape
        scores = feat_grouped.new_zeros((G, N, N))

        start = 0
        for mul, ir in self.irreps_feat:
            mul = int(mul)
            d = int(ir.dim)
            n = mul * d
            x = feat_grouped[..., start:start + n].reshape(G, N, mul, d)
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
            sim = torch.einsum("gimd,gjmd->gijm", x, x).mean(dim=-1)
            scores = scores + sim
            start += n

        return scores

    def _format_hr_boundary_map(
        self,
        hr_boundary_map: torch.Tensor,
        batch_size: int,
        hr_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Normalize HR boundary layout to (B,1,Hr,Wr) and clamp to [0,1].
        Enforces strict batch/spatial consistency.
        """
        Hr, Wr = hr_shape
        bmap = hr_boundary_map
        if bmap.dim() == 2:
            bmap = bmap.unsqueeze(0).unsqueeze(0)
        elif bmap.dim() == 3:
            bmap = bmap.unsqueeze(1)
        elif bmap.dim() == 4:
            if bmap.shape[1] != 1 and bmap.shape[-1] == 1:
                bmap = bmap.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"hr_boundary_map must be 2D/3D/4D, got shape {tuple(hr_boundary_map.shape)}"
            )

        assert bmap.shape[1] == 1, (
            f"Expected one HR boundary channel, got shape {tuple(bmap.shape)}"
        )
        assert bmap.shape[0] == batch_size, (
            f"HR boundary batch mismatch: got {bmap.shape[0]}, expected {batch_size}"
        )
        assert bmap.shape[-2:] == (Hr, Wr), (
            f"HR boundary spatial mismatch: got {tuple(bmap.shape[-2:])}, expected {(Hr, Wr)}"
        )
        return bmap.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    def _format_lr_boundary_map(
        self,
        lr_boundary_map: torch.Tensor,
        batch_size: int,
        lr_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Normalize LR boundary layout to (B,1,H,W) and clamp to [0,1].
        Enforces strict batch/spatial consistency.
        """
        H, W = lr_shape
        bmap = lr_boundary_map
        if bmap.dim() == 2:
            bmap = bmap.unsqueeze(0).unsqueeze(0)
        elif bmap.dim() == 3:
            bmap = bmap.unsqueeze(1)
        elif bmap.dim() == 4:
            if bmap.shape[1] != 1 and bmap.shape[-1] == 1:
                bmap = bmap.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"lr_boundary_map must be 2D/3D/4D, got shape {tuple(lr_boundary_map.shape)}"
            )

        assert bmap.shape[1] == 1, (
            f"Expected one LR boundary channel, got shape {tuple(bmap.shape)}"
        )
        assert bmap.shape[0] == batch_size, (
            f"LR boundary batch mismatch: got {bmap.shape[0]}, expected {batch_size}"
        )
        assert bmap.shape[-2:] == (H, W), (
            f"LR boundary spatial mismatch: got {tuple(bmap.shape[-2:])}, expected {(H, W)}"
        )
        return bmap.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    def _project_lr_labels_to_hr(
        self,
        lr_labels: torch.Tensor,
        hr_shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Project LR grain labels to HR by nearest parent mapping (floor(y/r), floor(x/r)).
        """
        if lr_labels.ndim != 3:
            raise ValueError(f"Expected lr_labels rank-3 (B,H,W), got {tuple(lr_labels.shape)}")
        B, H, W = lr_labels.shape
        Hr, Wr = hr_shape
        r_h, r_w = self.upsample_factor
        if Hr != H * r_h or Wr != W * r_w:
            raise ValueError(
                f"hr_shape {(Hr, Wr)} incompatible with lr_labels {(H, W)} and upsample_factor {(r_h, r_w)}"
            )

        y_hr = torch.arange(Hr, device=lr_labels.device, dtype=torch.long)
        x_hr = torch.arange(Wr, device=lr_labels.device, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(y_hr, x_hr, indexing="ij")
        y_lr = torch.div(grid_y, r_h, rounding_mode="floor").clamp(0, H - 1)
        x_lr = torch.div(grid_x, r_w, rounding_mode="floor").clamp(0, W - 1)
        return lr_labels[:, y_lr, x_lr]

    def forward(
        self,
        feat: torch.Tensor,
        hr_shape: tuple[int, int],
        hr_boundary_map: torch.Tensor | None = None,
        lr_boundary_map: torch.Tensor | None = None,
        hr_labels: torch.Tensor | None = None,
        lr_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            feat: (B, Hr*Wr, C_feat) HR features.
            hr_shape: (Hr, Wr).
            hr_boundary_map: optional (B,1,Hr,Wr) or compatible shape map.
            lr_boundary_map: optional (B,1,H,W) or compatible shape map.
            hr_labels: optional (B,Hr,Wr) labels. If provided, takes precedence.
            lr_labels: optional (B,H,W) labels used when hr_labels is not provided.
        Returns:
            delta: (B, Hr*Wr, C_feat) grain-attention residual.
        """
        B, HW, C_feat = feat.shape
        Hr, Wr = hr_shape
        if HW != Hr * Wr:
            raise ValueError(f"Expected feat.shape[1] == Hr*Wr = {Hr*Wr}, got {HW}")

        labels_all: torch.Tensor
        if hr_labels is not None:
            if hr_labels.shape != (B, Hr, Wr):
                raise ValueError(f"Expected hr_labels shape {(B, Hr, Wr)}, got {tuple(hr_labels.shape)}")
            labels_all = hr_labels.to(device=feat.device, dtype=torch.long)
        elif lr_labels is not None:
            if lr_labels.ndim != 3 or lr_labels.shape[0] != B:
                raise ValueError(
                    f"Expected lr_labels shape (B,H,W) with B={B}, got {tuple(lr_labels.shape)}"
                )
            labels_all = self._project_lr_labels_to_hr(
                lr_labels=lr_labels.to(device=feat.device, dtype=torch.long),
                hr_shape=(Hr, Wr),
            )
        elif hr_boundary_map is not None:
            bmap_hr = self._format_hr_boundary_map(
                hr_boundary_map=hr_boundary_map,
                batch_size=B,
                hr_shape=(Hr, Wr),
                device=feat.device,
                dtype=feat.dtype,
            )
            interior_hr = bmap_hr[:, 0] <= self.boundary_threshold
            labels_all = torch.full((B, Hr, Wr), -1, device=feat.device, dtype=torch.long)
            for b in range(B):
                labels_b_cpu, _ = self._connected_components_4(interior_hr[b])
                labels_all[b] = labels_b_cpu.to(device=feat.device, dtype=torch.long)
        elif lr_boundary_map is not None:
            r_h, r_w = self.upsample_factor
            H = Hr // r_h
            W = Wr // r_w
            if Hr != H * r_h or Wr != W * r_w:
                raise ValueError(
                    f"hr_shape {(Hr, Wr)} is not divisible by upsample_factor {(r_h, r_w)}"
                )
            bmap_lr = self._format_lr_boundary_map(
                lr_boundary_map=lr_boundary_map,
                batch_size=B,
                lr_shape=(H, W),
                device=feat.device,
                dtype=feat.dtype,
            )
            interior_lr = bmap_lr[:, 0] <= self.boundary_threshold
            labels_lr = torch.full((B, H, W), -1, device=feat.device, dtype=torch.long)
            for b in range(B):
                labels_b_cpu, _ = self._connected_components_4(interior_lr[b])
                labels_lr[b] = labels_b_cpu.to(device=feat.device, dtype=torch.long)
            labels_all = self._project_lr_labels_to_hr(labels_lr=labels_lr, hr_shape=(Hr, Wr))
        else:
            raise ValueError(
                "GrainAttention requires one of {hr_labels, lr_labels, hr_boundary_map, lr_boundary_map}."
            )

        y_hr = torch.arange(Hr, device=feat.device, dtype=feat.dtype)
        x_hr = torch.arange(Wr, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(y_hr, x_hr, indexing="ij")
        coords_flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (HW,2)

        # Flatten all batch tokens and build batch-disjoint grain ids.
        feat_flat = feat.reshape(B * HW, C_feat)
        labels_local = labels_all.reshape(B, HW)
        valid_local = labels_local >= 0
        if not bool(valid_local.any()):
            return torch.zeros_like(feat)

        max_gid = int(labels_local[valid_local].max().item()) + 1
        offsets = (torch.arange(B, device=feat.device, dtype=torch.long) * max_gid).unsqueeze(1)
        gid_flat = torch.where(
            valid_local,
            labels_local + offsets,
            torch.full_like(labels_local, -1),
        ).reshape(-1)
        tok_valid = gid_flat >= 0
        if not bool(tok_valid.any()):
            return torch.zeros_like(feat)

        tok_idx = tok_valid.nonzero(as_tuple=False).squeeze(1)          # (Nv,)
        tok_gid = gid_flat[tok_idx]                                     # (Nv,)
        tok_feat = feat_flat[tok_idx]                                   # (Nv,C_feat)
        tok_coords = coords_flat.repeat(B, 1)[tok_idx]                  # (Nv,2)

        # Group valid tokens by grain id and pack to padded (G,Nmax,*) tensors.
        all_gid = torch.unique(tok_gid)
        G = int(all_gid.numel())
        tok_group = torch.searchsorted(all_gid, tok_gid)                # (Nv,)
        counts = torch.bincount(tok_group, minlength=G)                 # (G,)

        order = torch.argsort(tok_group)
        group_sorted = tok_group[order]
        starts = torch.cumsum(counts, dim=0) - counts
        pos_sorted = torch.arange(tok_group.numel(), device=feat.device) - starts[group_sorted]
        tok_pos = torch.empty_like(tok_group)
        tok_pos[order] = pos_sorted                                     # (Nv,)

        Nmax = int(counts.max().item())
        feat_pad = tok_feat.new_zeros((G, Nmax, C_feat))
        coord_pad = tok_coords.new_zeros((G, Nmax, 2))
        feat_pad[tok_group, tok_pos] = tok_feat
        coord_pad[tok_group, tok_pos] = tok_coords

        mask = torch.arange(Nmax, device=feat.device)[None, :] < counts[:, None]  # (G,Nmax)

        attn_temp = torch.exp(self.log_s)
        scores = attn_temp * self._invariant_scores_grouped(feat_pad)   # (G,Nmax,Nmax)
        dist = torch.cdist(coord_pad, coord_pad, p=2)                   # (G,Nmax,Nmax)
        scores = scores + self.pos_bias(dist.unsqueeze(-1)).squeeze(-1)

        invalid = (~mask[:, :, None]) | (~mask[:, None, :])
        scores = scores.masked_fill(invalid, -1e9)
        attn = torch.softmax(scores.float(), dim=-1).to(feat.dtype)

        h = self.lin_in(feat_pad.reshape(G * Nmax, C_feat)).reshape(G, Nmax, -1)
        h = h * mask.unsqueeze(-1).to(h.dtype)
        ctx = (attn @ h) * mask.unsqueeze(-1).to(h.dtype)
        h_out = self.tp_out(
            h.reshape(G * Nmax, -1),
            ctx.reshape(G * Nmax, -1),
        ).reshape(G, Nmax, -1)
        delta_pad = self.lin_out(h_out.reshape(G * Nmax, -1)).reshape(G, Nmax, C_feat)
        delta_tok = delta_pad[tok_group, tok_pos]                       # (Nv,C_feat)

        delta_flat = feat.new_zeros((B * HW, C_feat))
        delta_flat = torch.index_copy(delta_flat, 0, tok_idx, delta_tok)
        return delta_flat.reshape(B, HW, C_feat)

class IsoEmbeddingSRAttn(nn.Module):
    """
    Local-iso SR model.

    - Crystal family is selected at top-level with `crystal`:
      - `fcc` (group O)
      - `hcp` (group D6)
    - Optional ablations:
      - disable LR conv1 (`use_lr_conv1=False`)
      - disable LR conv2 (`use_lr_conv2=False`)
      - disable HR conv1 (`use_hr_conv1=False`)
      - disable attention (`use_attention=False`)
    - All SR stages run in `irreps_a1` and there is no terminal projection layer.

    Main pipeline:
      1. Encode LR quaternions to A1 features
      2. Apply two LR equivariant convolutions (optional)
      3. Apply LR grain attention using LR 1px boundary map (optional)
      4. Upsample to HR grid (always grain-attention seeded from LR context)
      5. Apply HR equivariant convolution (optional)
      6. Apply post-upsample attention (optional)
      7. Decode HR features to quaternions using optimizing decoder
    """

    _SH_IRREPS = Irreps("1x0e + 1x2e")

    def __init__(
        self,
        crystal: str = "fcc",
        d6_convention: str = "z_axis",
        device: str | torch.device | None = None,
        upsample_factor: int | tuple[int, int] = 4,
        upsample_context_kernel_size: int = 3,
        upsample_residual: bool = True,
        use_boundary_gate: bool = False,
        upsample_boundary_threshold: float = 0.5,
        upsample_boundary_smooth_sigma: float = 0.8,
        upsample_boundary_smooth_iters: int = 3,
        upsample_boundary_sdf_shift: float = 0.2,
        use_lr_conv1: bool = True,
        use_lr_conv2: bool = True,
        use_lr_conv3: bool = False,
        lr_conv1_kernel_size: int = 3,
        lr_conv2_kernel_size: int = 9,
        lr_conv3_kernel_size: int = 9,
        lr_conv3_dilation: int = 1,
        use_residual_lr1: bool = False,
        use_residual_lr2: bool = False,
        use_residual_lr3: bool = False,
        use_residual_hr1: bool = False,
        use_hr_conv1: bool = True,
        hr_conv1_kernel_size: int = 3,
        use_attention: bool = True,
        use_grain_attention: bool = True,
        grain_attention_boundary_source: str = "hr",
        grain_attn_boundary_threshold: float = 0.5,
        enable_lr_grain_attention_layer: bool | None = None,
        enable_hr_grain_attention_layer: bool | None = None,
        enable_hr_block_attention_layer: bool | None = None,
        hr_grain_attention_boundary_source: str | None = None,
        use_lr_grain_attention: bool = False,
        num_lr_grain_attn_blocks: int = 1,
        lr_grain_attn_num_channels: int = 8,
        lr_grain_attn_checkpoint: bool = False,
        lr_grain_attn_boundary_threshold: float = 0.5,
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
        if isinstance(upsample_factor, (list, tuple)):
            self.upsample_factor = (int(upsample_factor[0]), int(upsample_factor[1]))
        else:
            self.upsample_factor = (int(upsample_factor), int(upsample_factor))
        self.use_lr_conv1 = bool(use_lr_conv1)
        self.use_lr_conv2 = bool(use_lr_conv2)
        self.use_lr_conv3 = bool(use_lr_conv3)
        self.use_hr_conv1 = bool(use_hr_conv1)
        # Descriptive layer knobs (new API), with backward-compatible fallbacks.
        self.enable_lr_grain_attention_layer = (
            bool(enable_lr_grain_attention_layer)
            if enable_lr_grain_attention_layer is not None
            else bool(use_lr_grain_attention)
        )
        self.use_lr_grain_attention = self.enable_lr_grain_attention_layer

        if enable_hr_grain_attention_layer is None and enable_hr_block_attention_layer is None:
            if bool(use_attention):
                self.enable_hr_grain_attention_layer = bool(use_grain_attention)
                self.enable_hr_block_attention_layer = not bool(use_grain_attention)
            else:
                self.enable_hr_grain_attention_layer = False
                self.enable_hr_block_attention_layer = False
        else:
            self.enable_hr_grain_attention_layer = bool(enable_hr_grain_attention_layer)
            self.enable_hr_block_attention_layer = bool(enable_hr_block_attention_layer)

        if self.enable_hr_grain_attention_layer and self.enable_hr_block_attention_layer:
            raise ValueError(
                "Choose only one post-upsample attention layer: "
                "`enable_hr_grain_attention_layer` or `enable_hr_block_attention_layer`."
            )

        self.use_attention = self.enable_hr_grain_attention_layer or self.enable_hr_block_attention_layer
        self.use_grain_attention = self.enable_hr_grain_attention_layer

        boundary_src = (
            hr_grain_attention_boundary_source
            if hr_grain_attention_boundary_source is not None
            else grain_attention_boundary_source
        )
        self.grain_attention_boundary_source = str(boundary_src).lower().strip()
        if self.grain_attention_boundary_source not in {"hr", "lr"}:
            raise ValueError(
                "hr_grain_attention_boundary_source must be 'hr' or 'lr', "
                f"got {boundary_src!r}"
            )
        self.lr_grain_attn_checkpoint = bool(lr_grain_attn_checkpoint)
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

        # A1-only SR architecture:
        # LR conv1: a1 -> a1
        self.conv_lr1 = EquivariantSpatialConv(
            kernel_size=int(lr_conv1_kernel_size),
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
            use_residual=bool(use_residual_lr1),
        )
        # LR conv2: a1 -> a1
        self.conv_lr2 = EquivariantSpatialConv(
            kernel_size=int(lr_conv2_kernel_size),
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
            use_residual=bool(use_residual_lr2),
        )
        # LR conv3: a1 -> a1
        self.conv_lr3 = EquivariantSpatialConv(
            kernel_size=int(lr_conv3_kernel_size),
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
            use_residual=bool(use_residual_lr3),
            dilation=int(lr_conv3_dilation),
        )
        # Optional LR grain-attention stack driven by LR 1px boundary map.
        if self.enable_lr_grain_attention_layer:
            self.lr_grain_attention_blocks = nn.ModuleList(
                [
                    GrainAttention(
                        self.irreps_a1,
                        num_channels=int(lr_grain_attn_num_channels),
                        boundary_threshold=float(lr_grain_attn_boundary_threshold),
                        upsample_factor=(1, 1),
                    )
                    for _ in range(max(1, int(num_lr_grain_attn_blocks)))
                ]
            )
        else:
            self.lr_grain_attention_blocks = nn.ModuleList([])
        # Upsample k=3: a1 -> a1
        self.upsample_conv = BoundaryAwareAttentionUpsampler(
            kernel_size=int(upsample_context_kernel_size),
            upsample_factor=self.upsample_factor,
            use_residual=bool(upsample_residual),
            use_boundary_gate=bool(use_boundary_gate),
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
            boundary_threshold=float(upsample_boundary_threshold),
            boundary_smooth_sigma=float(upsample_boundary_smooth_sigma),
            boundary_smooth_iters=int(upsample_boundary_smooth_iters),
            boundary_sdf_shift=float(upsample_boundary_sdf_shift),
        )
        # HR conv1: a1 -> a1
        self.conv_hr1 = EquivariantSpatialConv(
            kernel_size=int(hr_conv1_kernel_size),
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_a1,
            use_residual=bool(use_residual_hr1),
        )
        # Attention block(s)
        if self.use_attention:
            if self.enable_hr_grain_attention_layer:
                self.attention_blocks = nn.ModuleList(
                    [
                        GrainAttention(
                            self.irreps_a1,
                            num_channels=int(hr_attn_num_channels),
                            tp_out_chunk_size=hr_attn_tp_out_chunk_size,
                            boundary_threshold=float(grain_attn_boundary_threshold),
                            upsample_factor=self.upsample_factor,
                        )
                        for _ in range(max(1, int(num_hr_attn_blocks)))
                    ]
                )
            elif self.enable_hr_block_attention_layer:
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
        else:
            self.attention_blocks = nn.ModuleList([])

        self._cached_hr_block_shape: tuple[int, int] | None = None
        self._cached_hr_sh_block: torch.Tensor | None = None
        self._last_boundary_context: dict[str, torch.Tensor | None] | None = None

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
        boundary_ctx: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """
        Apply post-upsample attention to HR features.
          - GrainAttention (default): uses HR boundary map and grain labels.
          - AttentionBlock fallback: original block-local attention.
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

        if self.use_grain_attention:
            if boundary_ctx is None:
                raise ValueError("boundary_ctx is required for GrainAttention.")
            hr_boundary = boundary_ctx.get("boundary_hr_1px", None)
            lr_boundary = boundary_ctx.get("boundary_lr_1px", None)
            hr_labels = boundary_ctx.get("hr_labels", None)
            lr_labels = boundary_ctx.get("lr_labels", None)
            if hr_boundary is None:
                raise ValueError("boundary_ctx['boundary_hr_1px'] is required for GrainAttention.")

            use_lr_source = self.grain_attention_boundary_source == "lr"
            feat = features
            for block in self.attention_blocks:
                if self.hr_attn_checkpoint and self.training and feat.requires_grad:
                    def _run_block(x: torch.Tensor, _block=block) -> torch.Tensor:
                        return _block(
                            x,
                            hr_shape=hr_shape,
                            hr_boundary_map=None if use_lr_source else hr_boundary,
                            lr_boundary_map=lr_boundary if use_lr_source else None,
                            hr_labels=None if use_lr_source else hr_labels,
                            lr_labels=lr_labels if use_lr_source else None,
                        )

                    delta = checkpoint(_run_block, feat, use_reentrant=True)
                else:
                    delta = block(
                        feat,
                        hr_shape=hr_shape,
                        hr_boundary_map=None if use_lr_source else hr_boundary,
                        lr_boundary_map=lr_boundary if use_lr_source else None,
                        hr_labels=None if use_lr_source else hr_labels,
                        lr_labels=lr_labels if use_lr_source else None,
                    )
                feat = feat + delta

            if not batched:
                feat = feat.squeeze(0)
            return feat

        block_h = min(self.hr_attn_block_size, Hr)
        block_w = min(self.hr_attn_block_size, Wr)
        pad_h = (-Hr) % block_h
        pad_w = (-Wr) % block_w
        Hr_pad, Wr_pad = Hr + pad_h, Wr + pad_w

        feat = features
        if pad_h > 0 or pad_w > 0:
            feat_2d = feat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
            feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
            feat = feat_2d.permute(0, 2, 3, 1).reshape(B, Hr_pad * Wr_pad, C)

        sh_block = self._get_hr_sh_block(block_h, block_w, feat.device, feat.dtype)
        for block in self.attention_blocks:
            if self.hr_attn_checkpoint and self.training and feat.requires_grad:
                def _run_block(x: torch.Tensor, _block=block) -> torch.Tensor:
                    return _block(x, sh_block, Hr_pad, Wr_pad, block_h, block_w)

                delta = checkpoint(_run_block, feat, use_reentrant=True)
            else:
                delta = block(feat, sh_block, Hr_pad, Wr_pad, block_h, block_w)
            
            feat = feat + delta

        if pad_h > 0 or pad_w > 0:
            feat = feat.reshape(B, Hr_pad, Wr_pad, C)[:, :Hr, :Wr, :].reshape(B, Hr * Wr, C)

        if not batched:
            feat = feat.squeeze(0)
        return feat

    def _apply_lr_grain_attention(
        self,
        features: torch.Tensor,
        lr_shape: tuple[int, int],
        boundary_ctx: dict[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """
        Apply grain-local attention on LR features using LR 1px boundary map.
        """
        if (not self.enable_lr_grain_attention_layer) or (len(self.lr_grain_attention_blocks) == 0):
            return features
        if boundary_ctx is None:
            raise ValueError("boundary_ctx is required for LR grain attention.")

        H, W = lr_shape
        batched = features.dim() == 3
        if not batched:
            features = features.unsqueeze(0)
        B, N, _ = features.shape
        if N != H * W:
            raise ValueError(f"Expected N={H*W} from lr_shape, got {N}")

        boundary_lr = boundary_ctx.get("boundary_lr_1px", None)
        lr_labels = boundary_ctx.get("lr_labels", None)
        if boundary_lr is None:
            raise ValueError("boundary_ctx['boundary_lr_1px'] is required for LR grain attention.")
        if lr_labels is None:
            raise ValueError("boundary_ctx['lr_labels'] is required for LR grain attention.")

        feat = features
        for block in self.lr_grain_attention_blocks:
            if self.lr_grain_attn_checkpoint and self.training and feat.requires_grad:
                def _run_block(x: torch.Tensor, _block=block) -> torch.Tensor:
                    return _block(
                        x,
                        hr_shape=lr_shape,
                        hr_boundary_map=boundary_lr,
                        hr_labels=lr_labels,
                    )

                delta = checkpoint(_run_block, feat, use_reentrant=True)
            else:
                delta = block(
                    feat,
                    hr_shape=lr_shape,
                    hr_boundary_map=boundary_lr,
                    hr_labels=lr_labels,
                )
            feat = feat + delta

        if not batched:
            feat = feat.squeeze(0)
        return feat

    @torch.no_grad()
    def _prepare_boundary_context(
        self,
        lr_boundary_map: torch.Tensor,
        lr_shape: tuple[int, int],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | None]:
        """
        Standard boundary-prep pipeline run at model level (outside upsampler):
          LR boundary map -> LR 1px boundary -> HR 1px boundary -> grain maps.
        """
        if lr_boundary_map is None:
            raise ValueError("lr_boundary_map is required for boundary-aware SR forward.")

        H, W = lr_shape
        boundary_lr = self.upsample_conv._format_lr_boundary_map(
            lr_boundary_map=lr_boundary_map,
            batch_size=int(batch_size),
            lr_shape=(H, W),
            device=device,
            dtype=dtype,
        )
        boundary_lr_1px, boundary_hr_1px, hr_to_lr_map, lr_labels = (
            self.upsample_conv._build_hr_1px_boundary_and_maps(boundary_lr=boundary_lr)
        )
        hr_labels = None
        if self.enable_hr_grain_attention_layer:
            B, _, Hr, Wr = boundary_hr_1px.shape
            hr_labels = torch.full((B, Hr, Wr), -1, device=device, dtype=torch.long)
            interior_hr = boundary_hr_1px[:, 0] <= self.upsample_conv.boundary_threshold
            for b in range(B):
                labels_hr_cpu, _ = self.upsample_conv._connected_components_4(interior_hr[b])
                hr_labels[b] = labels_hr_cpu.to(device=device, dtype=torch.long)
        return {
            "boundary_lr": boundary_lr,
            "boundary_lr_1px": boundary_lr_1px,
            "boundary_hr_1px": boundary_hr_1px,
            "hr_to_lr_map": hr_to_lr_map,
            "lr_labels": lr_labels,
            "hr_labels": hr_labels,
        }

    def _forward_sr_features(
        self,
        feat_lr_a1: torch.Tensor,
        lr_shape: tuple[int, int],
        lr_boundary_map: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Main SR feature pipeline: LR convs, upsampling, HR conv, attention.
        Returns HR features and HR shape.
        """
        if self.use_lr_conv1:
            feat = self.conv_lr1(feat_lr_a1, lr_shape)
        else:
            feat = feat_lr_a1

        if self.use_lr_conv2:
            feat = self.conv_lr2(feat, lr_shape)

        if self.use_lr_conv3:
            feat = self.conv_lr3(feat, lr_shape)

        batch_size = int(feat.shape[0]) if feat.dim() == 3 else 1
        boundary_ctx = self._prepare_boundary_context(
            lr_boundary_map=lr_boundary_map,
            lr_shape=lr_shape,
            batch_size=batch_size,
            device=feat.device,
            dtype=feat.dtype,
        )
        self._last_boundary_context = boundary_ctx

        feat = self._apply_lr_grain_attention(feat, lr_shape, boundary_ctx=boundary_ctx)

        feat, hr_shape = self.upsample_conv(
            feat,
            lr_shape,
            hr_to_lr_map=boundary_ctx["hr_to_lr_map"],
            lr_labels=boundary_ctx["lr_labels"],
        )
        if self.use_hr_conv1:
            feat = self.conv_hr1(feat, hr_shape)
        feat_a1 = self._apply_attention(feat, hr_shape, boundary_ctx=boundary_ctx)
        return feat_a1, hr_shape


    def feature_loss_sr(
        self,
        lr_quats: torch.Tensor,
        hr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        lr_boundary_map: torch.Tensor,
        normalize_input: bool = False,
        tv_loss_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target HR features (A1 irreps).
        Used for feature-space supervision.
        lr_boundary_map is required.
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

        feat_hr, _ = self._forward_sr_features(
            feat_lr_a1,
            lr_shape,
            lr_boundary_map=lr_boundary_map,
        )
        mse = F.mse_loss(feat_hr, feat_hr_tgt)
        if tv_loss_weight > 0.0:
            H_hr = lr_shape[0] * self.upsample_factor[0]
            W_hr = lr_shape[1] * self.upsample_factor[1]
            f = feat_hr.reshape(B, H_hr, W_hr, -1)
            tv = torch.mean(torch.abs(f[:, 1:, :, :] - f[:, :-1, :, :])) + \
                 torch.mean(torch.abs(f[:, :, 1:, :] - f[:, :, :-1, :]))
            return mse + tv_loss_weight * tv
        return mse


    def forward_sr(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        lr_boundary_map: torch.Tensor,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for SR: input LR quaternions, output SR quaternions.
        lr_boundary_map is required.
        """
        lr_quats = lr_quats.to(self.device)
        if normalize_input:
            lr_quats = _normalize_quaternions(lr_quats)
        feat_lr_a1 = self.encode_a1(lr_quats)
        feat_hr_a1, _ = self._forward_sr_features(
            feat_lr_a1,
            lr_shape,
            lr_boundary_map=lr_boundary_map,
        )
        return self.decode(feat_hr_a1)


    def forward(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
        lr_boundary_map: torch.Tensor | None = None,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass: input quaternions (optionally with image shape), output HR quaternions.
        If img_shape is provided, runs SR pipeline; otherwise, decodes input features directly.
        In SR mode, lr_boundary_map is required.
        """
        quats = quats.to(self.device)
        if quats.dim() != 2 or quats.shape[-1] != 4:
            raise ValueError(f"IsoEmbeddingSRAttn expects (N,4), got {tuple(quats.shape)}")
        if normalize_input:
            quats = _normalize_quaternions(quats)

        if img_shape is not None:
            if lr_boundary_map is None:
                raise ValueError("lr_boundary_map is required in SR mode")
            return self.forward_sr(
                quats,
                lr_shape=img_shape,
                lr_boundary_map=lr_boundary_map,
                normalize_input=False,
            )

        feat_a1 = self.encode_a1(quats)
        return self.decode(feat_a1)

__all__ = [
    "AttentionBlock",
    "BoundaryAwareAttentionUpsampler",
    "CubochoricOptimizingLocalIsoDecoder",
    "EquivariantSpatialConv",
    "EquivariantTransposeConv",
    "GrainAttention",
    "IsoEmbeddingSRAttn",
    "LearnableA1QuaternionDecoder",
    "LocalIsoCrystalEncoder",
]
