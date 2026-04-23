from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps

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
            raise ValueError(
                f"RRCTP semiglobal model currently assumes isotropic SR, got scale={scale}"
            )
        scale = scale_y
    scale_int = int(scale)
    if scale_int < 1:
        raise ValueError(f"scale must be >= 1, got {scale_int}")
    return scale_int


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
    """
    Decode local-iso irreps features to passive quaternions by nearest-table seeding
    followed by feature-space optimization.
    """

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
        B, C = feat_target.shape
        if C != self.target_dim:
            raise ValueError(f"Expected target dim {self.target_dim} for {self.target_irreps}, got {C}")

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


class PhaseEmbedding4x4(nn.Module):
    """Learned phase embedding for subpixel positions inside an LR cell."""

    def __init__(self, upsample_factor: int | tuple[int, int] | list[int] = 4, emb_dim: int = 32):
        super().__init__()
        self.upsample_factor = _as_isotropic_scale(upsample_factor)
        self.num_phases = int(self.upsample_factor * self.upsample_factor)
        self.emb = nn.Embedding(self.num_phases, int(emb_dim))
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, phase_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(phase_ids)


class SharedWindowGate(nn.Module):
    """Shared visibility gate over a fixed LR candidate window."""

    def __init__(self, window_size: int = 5):
        super().__init__()
        self.window_size = int(window_size)
        if self.window_size < 1 or self.window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
        self.window_logits = nn.Parameter(torch.zeros(self.window_size * self.window_size))
        self.aff_temp = nn.Parameter(torch.tensor(1.0))
        self.aff_bias = nn.Parameter(torch.tensor(0.0))
        self.cross_boundary_floor = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        same_grain: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eps = 1e-6
        B, U, M, _ = candidates.shape

        g_win = torch.sigmoid(self.window_logits).view(1, 1, M).expand(B, U, M)

        qh = F.normalize(query, dim=-1)
        ch = F.normalize(candidates, dim=-1)
        cos = (qh.unsqueeze(2) * ch).sum(dim=-1)
        g_aff = torch.sigmoid(self.aff_temp * cos + self.aff_bias)

        if same_grain is None:
            g_boundary = torch.ones_like(g_aff)
        else:
            floor = torch.sigmoid(self.cross_boundary_floor)
            g_boundary = torch.where(same_grain, torch.ones_like(g_aff), floor * torch.ones_like(g_aff))

        g = (g_win * g_boundary * g_aff).clamp_min(eps)
        return g, {
            "g_win": g_win,
            "g_boundary": g_boundary,
            "g_aff": g_aff,
        }


class InvariantStatBuilder(nn.Module):
    """Builds blockwise invariant statistics from irrep-valued feature tensors."""

    def __init__(self, irreps_feat: Irreps | str):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.block_slices = _irrep_block_slices(self.irreps_feat)
        self.num_blocks = len(self.block_slices)
        self.stat_dim = 3 * self.num_blocks

    def pair_stats(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(f"Expected matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
        outs: list[torch.Tensor] = []
        for start, end in self.block_slices:
            aa = a[..., start:end]
            bb = b[..., start:end]
            outs.append(aa.norm(dim=-1, keepdim=True))
            outs.append(bb.norm(dim=-1, keepdim=True))
            outs.append((aa * bb).sum(dim=-1, keepdim=True))
        return torch.cat(outs, dim=-1)

    def query_candidates(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        q = query.unsqueeze(2).expand_as(candidates)
        return self.pair_stats(q, candidates)

    def query_summary(self, query: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        return self.pair_stats(query, summary)

    def query_contexts(self, query: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        q = query.unsqueeze(2).expand_as(contexts)
        return self.pair_stats(q, contexts)


class InvariantCandidateScorer(nn.Module):
    """Per-expert candidate scoring from invariant stats, offsets, and phase conditioning."""

    def __init__(
        self,
        stat_dim: int,
        num_experts: int,
        phase_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        in_dim = int(stat_dim) + 2 + int(phase_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.num_experts),
        )

    def forward(
        self,
        cand_stats: torch.Tensor,
        rel_offsets: torch.Tensor,
        phase_embed: torch.Tensor,
    ) -> torch.Tensor:
        B, U, M, _ = cand_stats.shape
        offs = rel_offsets.view(1, 1, M, 2).expand(B, U, M, 2)
        ph = phase_embed.unsqueeze(2).expand(B, U, M, phase_embed.shape[-1])
        x = torch.cat([cand_stats, offs, ph], dim=-1)
        return self.net(x).permute(0, 1, 3, 2).contiguous()


class InvariantRouter(nn.Module):
    """Per-pixel router from invariant summaries to expert logits."""

    def __init__(
        self,
        stat_dim: int,
        num_experts: int,
        phase_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        in_dim = int(stat_dim) + int(phase_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.num_experts),
        )

    def forward(self, summary_stats: torch.Tensor, phase_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([summary_stats, phase_embed], dim=-1))


class EquivariantQueryAnchor(nn.Module):
    """
    Phase-conditioned block scaling of the parent feature.

    This keeps the query in the same irrep space as the parent feature while
    letting the subpixel phase modulate the anchor through invariant scalars.
    """

    def __init__(
        self,
        irreps_feat: Irreps | str,
        phase_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.block_slices = _irrep_block_slices(self.irreps_feat)
        self.num_blocks = len(self.block_slices)
        self.net = nn.Sequential(
            nn.Linear(int(phase_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.num_blocks),
        )

    def forward(self, parent_feat: torch.Tensor, phase_embed: torch.Tensor) -> torch.Tensor:
        scales = 1.0 + 0.5 * torch.tanh(self.net(phase_embed))
        out = torch.zeros_like(parent_feat)
        for j, (start, end) in enumerate(self.block_slices):
            out[..., start:end] = scales[..., j : j + 1] * parent_feat[..., start:end]
        return out


class ExpertEquivariantSeedHead(nn.Module):
    """
    Batched expert bank for phase- and stats-conditioned equivariant seed heads.

    All experts are evaluated in one tensorized pass so seed computation stays
    GPU-friendly even when the number of experts grows.
    """

    def __init__(
        self,
        irreps_feat: Irreps | str,
        stat_dim: int,
        phase_dim: int,
        num_experts: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.block_slices = _irrep_block_slices(self.irreps_feat)
        self.num_blocks = len(self.block_slices)
        self.num_experts = int(num_experts)
        in_dim = int(stat_dim) + int(phase_dim)
        hid = int(hidden_dim)

        self.fc1_weight = nn.Parameter(torch.empty(self.num_experts, in_dim, hid))
        self.fc1_bias = nn.Parameter(torch.zeros(self.num_experts, hid))
        self.fc2_weight = nn.Parameter(torch.empty(self.num_experts, hid, 2 * self.num_blocks))
        self.fc2_bias = nn.Parameter(torch.zeros(self.num_experts, 2 * self.num_blocks))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for k in range(self.num_experts):
            nn.init.xavier_uniform_(self.fc1_weight[k])
            nn.init.xavier_uniform_(self.fc2_weight[k])
        nn.init.zeros_(self.fc1_bias)
        nn.init.zeros_(self.fc2_bias)

    def forward(
        self,
        query: torch.Tensor,
        contexts: torch.Tensor,
        phase_embed: torch.Tensor,
        pair_stats: torch.Tensor,
    ) -> torch.Tensor:
        B, U, K, C = contexts.shape
        if K != self.num_experts:
            raise ValueError(f"Expected num_experts={self.num_experts}, got {K}")

        ph = phase_embed.unsqueeze(2).expand(B, U, K, phase_embed.shape[-1])
        x = torch.cat([pair_stats, ph], dim=-1)
        hidden = torch.einsum("bukf,kfh->bukh", x, self.fc1_weight)
        hidden = hidden + self.fc1_bias.view(1, 1, K, -1)
        hidden = F.gelu(hidden)
        coeffs = torch.einsum("bukh,kho->buko", hidden, self.fc2_weight)
        coeffs = coeffs + self.fc2_bias.view(1, 1, K, -1)
        alpha, beta = coeffs.chunk(2, dim=-1)
        alpha = 1.0 + 0.5 * torch.tanh(alpha)
        beta = 0.5 * torch.tanh(beta)

        query_exp = query.unsqueeze(2).expand(B, U, K, C)
        out = torch.zeros_like(contexts)
        for j, (start, end) in enumerate(self.block_slices):
            out[..., start:end] = alpha[..., j : j + 1] * query_exp[..., start:end]
            out[..., start:end] = out[..., start:end] + beta[..., j : j + 1] * contexts[..., start:end]
        return out


class RRCTPSemiglobalUpsampler(nn.Module):
    """
    Semiglobal RRCTP upsampler that stops at the routed HR seed.

    Flow:
      1) build candidate bank
      2) score/select candidates with invariant selector logic
      3) build expert contexts
      4) build and mix expert seed proposals
    """

    def __init__(
        self,
        irreps_feat: Irreps | str,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        window_size: int = 5,
        num_experts: int = 12,
        top_k: int = 2,
        phase_dim: int = 32,
        score_hidden_dim: int = 64,
        router_hidden_dim: int = 128,
        query_hidden_dim: int = 64,
        seed_hidden_dim: int = 128,
        token_chunk_size: int = 1024,
    ):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feat_dim = int(self.irreps_feat.dim)
        self.upsample_factor = _as_isotropic_scale(upsample_factor)
        self.window_size = int(window_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.phase_dim = int(phase_dim)
        self.token_chunk_size = int(token_chunk_size)

        if self.window_size < 1 or self.window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got {self.top_k}")

        self.phase_embed = PhaseEmbedding4x4(
            upsample_factor=self.upsample_factor,
            emb_dim=self.phase_dim,
        )
        self.stat_builder = InvariantStatBuilder(self.irreps_feat)
        self.query_anchor = EquivariantQueryAnchor(
            self.irreps_feat,
            phase_dim=self.phase_dim,
            hidden_dim=int(query_hidden_dim),
        )
        self.window_gate = SharedWindowGate(window_size=self.window_size)
        self.candidate_scorer = InvariantCandidateScorer(
            stat_dim=self.stat_builder.stat_dim,
            num_experts=self.num_experts,
            phase_dim=self.phase_dim,
            hidden_dim=int(score_hidden_dim),
        )
        self.router = InvariantRouter(
            stat_dim=self.stat_builder.stat_dim,
            num_experts=self.num_experts,
            phase_dim=self.phase_dim,
            hidden_dim=int(router_hidden_dim),
        )
        self.seed_heads = ExpertEquivariantSeedHead(
            self.irreps_feat,
            stat_dim=self.stat_builder.stat_dim,
            phase_dim=self.phase_dim,
            num_experts=self.num_experts,
            hidden_dim=int(seed_hidden_dim),
        )

        self._cached_index_key: tuple[int, int, str] | None = None
        self._cached_parent_idx: torch.Tensor | None = None
        self._cached_phase_idx: torch.Tensor | None = None
        self._cached_rel_offsets: torch.Tensor | None = None

    def _get_index_tensors(
        self,
        lr_shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = int(lr_shape[0]), int(lr_shape[1])
        key = (H, W, str(device))
        if self._cached_index_key == key:
            if self._cached_parent_idx is None or self._cached_phase_idx is None or self._cached_rel_offsets is None:
                raise RuntimeError("Index cache key is set but tensors are missing.")
            return self._cached_parent_idx, self._cached_phase_idx, self._cached_rel_offsets

        r = self.upsample_factor
        Hr, Wr = H * r, W * r

        y_hr = torch.arange(Hr, device=device)
        x_hr = torch.arange(Wr, device=device)
        gy, gx = torch.meshgrid(y_hr, x_hr, indexing="ij")

        py = torch.div(gy, r, rounding_mode="floor")
        px = torch.div(gx, r, rounding_mode="floor")
        parent_idx = (py * W + px).reshape(-1)

        ph_y = torch.remainder(gy, r)
        ph_x = torch.remainder(gx, r)
        phase_idx = (ph_y * r + ph_x).reshape(-1)

        rad = self.window_size // 2
        offs = []
        den = float(max(1, rad))
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                offs.append((float(dy) / den, float(dx) / den))
        rel_offsets = torch.tensor(offs, device=device, dtype=torch.float32)

        self._cached_index_key = key
        self._cached_parent_idx = parent_idx
        self._cached_phase_idx = phase_idx
        self._cached_rel_offsets = rel_offsets
        return parent_idx, phase_idx, rel_offsets

    def _build_parent_patch_bank(self, feat_lr_img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_lr_img.shape
        p = self.window_size // 2
        feat_pad = F.pad(feat_lr_img, (p, p, p, p), mode="replicate")
        patches = feat_pad.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        return (
            patches.permute(0, 2, 3, 4, 5, 1)
            .contiguous()
            .reshape(B, H * W, self.window_size * self.window_size, C)
        )

    def _build_parent_label_bank(self, labels_lr: torch.Tensor) -> torch.Tensor:
        B, H, W = labels_lr.shape
        p = self.window_size // 2
        lab = labels_lr.unsqueeze(1).to(dtype=torch.float32)
        lab_pad = F.pad(lab, (p, p, p, p), mode="replicate")
        patches = lab_pad.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        bank = patches[:, 0].reshape(B, H * W, self.window_size * self.window_size)
        return bank.to(dtype=labels_lr.dtype)

    def forward(
        self,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        lr_labels: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int], dict[str, torch.Tensor]] | tuple[torch.Tensor, tuple[int, int]]:
        H, W = int(lr_shape[0]), int(lr_shape[1])
        r = self.upsample_factor
        Hr, Wr = H * r, W * r

        batched = feat_lr.dim() == 3
        if not batched:
            feat_lr = feat_lr.unsqueeze(0)
        B, N, C = feat_lr.shape
        if N != H * W:
            raise ValueError(f"Expected N={H * W}, got {N}")
        if C != self.feat_dim:
            raise ValueError(f"Expected feature dim {self.feat_dim}, got {C}")

        feat_lr_img = feat_lr.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        parent_bank = self._build_parent_patch_bank(feat_lr_img)

        label_bank = None
        if lr_labels is not None:
            if lr_labels.dim() == 2:
                lr_labels = lr_labels.unsqueeze(0)
            if lr_labels.shape[0] != B or lr_labels.shape[-2:] != (H, W):
                raise ValueError(
                    f"lr_labels must be (B,H,W) with {(B, H, W)}, got {tuple(lr_labels.shape)}"
                )
            label_bank = self._build_parent_label_bank(lr_labels.long())

        parent_idx, phase_idx, rel_offsets = self._get_index_tensors((H, W), feat_lr.device)
        rel_offsets = rel_offsets.to(device=feat_lr.device, dtype=feat_lr.dtype)

        U = Hr * Wr
        out_chunks: list[torch.Tensor] = []
        aux_accum: dict[str, list[torch.Tensor]] = {
            "router_logits": [],
            "topk_idx": [],
            "topk_pi": [],
            "query_anchor": [],
            "seed": [],
        }

        for start in range(0, U, self.token_chunk_size):
            end = min(start + self.token_chunk_size, U)
            u = end - start

            p_idx = parent_idx[start:end]
            ph_idx = phase_idx[start:end]

            cand = parent_bank.index_select(1, p_idx)
            parent_feat = feat_lr.index_select(1, p_idx)

            ph = self.phase_embed(ph_idx).to(dtype=feat_lr.dtype)
            ph = ph.unsqueeze(0).expand(B, u, -1)

            q0 = self.query_anchor(parent_feat, ph)

            same_grain = None
            if label_bank is not None and lr_labels is not None:
                lab_c = label_bank.index_select(1, p_idx)
                lab_p = lr_labels.view(B, H * W).index_select(1, p_idx).unsqueeze(-1)
                valid = (lab_c >= 0) & (lab_p >= 0)
                same_grain = valid & (lab_c == lab_p)

            g, _gate_terms = self.window_gate(q0, cand, same_grain=same_grain)

            cand_stats = self.stat_builder.query_candidates(q0, cand)
            score = self.candidate_scorer(cand_stats, rel_offsets=rel_offsets, phase_embed=ph)
            logits = score + torch.log(g.unsqueeze(2))
            alpha = torch.softmax(logits, dim=-1)

            ctx = torch.einsum("bukm,bumc->bukc", alpha, cand)

            g_norm = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            bank_sum = torch.einsum("bum,bumc->buc", g_norm, cand)
            router_stats = self.stat_builder.query_summary(q0, bank_sum)
            router_logits = self.router(router_stats, ph)
            topk_val, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
            topk_pi = torch.softmax(topk_val, dim=-1)

            ctx_stats = self.stat_builder.query_contexts(q0, ctx)
            seed_all = self.seed_heads(q0, ctx, ph, ctx_stats)

            idx_seed = topk_idx.unsqueeze(-1).expand(B, u, self.top_k, C)
            seed_sel = torch.gather(seed_all, dim=2, index=idx_seed)
            seed = (topk_pi.unsqueeze(-1) * seed_sel).sum(dim=2)
            out_chunks.append(seed)

            if return_aux:
                aux_accum["router_logits"].append(router_logits.detach())
                aux_accum["topk_idx"].append(topk_idx.detach())
                aux_accum["topk_pi"].append(topk_pi.detach())
                aux_accum["query_anchor"].append(q0.detach())
                aux_accum["seed"].append(seed.detach())

        feat_hr = torch.cat(out_chunks, dim=1)

        if not batched:
            feat_hr = feat_hr.squeeze(0)

        if not return_aux:
            return feat_hr, (Hr, Wr)

        aux = {
            "router_logits": torch.cat(aux_accum["router_logits"], dim=1),
            "topk_idx": torch.cat(aux_accum["topk_idx"], dim=1),
            "topk_pi": torch.cat(aux_accum["topk_pi"], dim=1),
            "query_anchor": torch.cat(aux_accum["query_anchor"], dim=1),
            "seed": torch.cat(aux_accum["seed"], dim=1),
        }
        if not batched:
            aux = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.shape[0] == 1 else v
                for k, v in aux.items()
            }
        return feat_hr, (Hr, Wr), aux


class IsoEmbeddingSRRRCTP(nn.Module):
    """
    Minimal RRCTP SR model with the simple flow:
      encoder -> RRCTP upsampler (through HR seed) -> decoder
    """

    def __init__(
        self,
        crystal: str = "fcc",
        d6_convention: str = "z_axis",
        device: str | torch.device | None = None,
        feature_irreps: str = "full",
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        window_size: int = 5,
        num_experts: int = 12,
        top_k_experts: int = 2,
        phase_dim: int = 32,
        rrctp_score_hidden_dim: int = 64,
        rrctp_router_hidden_dim: int = 128,
        rrctp_query_hidden_dim: int = 64,
        rrctp_seed_hidden_dim: int = 128,
        rrctp_token_chunk_size: int = 1024,
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

        self.upsample_factor = _as_isotropic_scale(upsample_factor)
        self.rrctp = RRCTPSemiglobalUpsampler(
            irreps_feat=self.irreps_feat,
            upsample_factor=self.upsample_factor,
            window_size=int(window_size),
            num_experts=int(num_experts),
            top_k=int(top_k_experts),
            phase_dim=int(phase_dim),
            score_hidden_dim=int(rrctp_score_hidden_dim),
            router_hidden_dim=int(rrctp_router_hidden_dim),
            query_hidden_dim=int(rrctp_query_hidden_dim),
            seed_hidden_dim=int(rrctp_seed_hidden_dim),
            token_chunk_size=int(rrctp_token_chunk_size),
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
            B, N, C = features.shape
            q = decoder(features.reshape(B * N, C))
            return self.reduce_to_fz(q).reshape(B, N, 4)
        q = decoder(features)
        return self.reduce_to_fz(q)

    def _forward_sr_features(
        self,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        lr_labels: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        return self.rrctp(
            feat_lr,
            lr_shape=lr_shape,
            lr_labels=lr_labels,
            return_aux=return_aux,
        )

    def forward_sr(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = True,
        lr_labels: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        lr_quats = lr_quats.to(self.device)
        if normalize_input:
            lr_quats = _normalize_quaternions(lr_quats)

        feat_lr = self.encode(lr_quats)
        if return_aux:
            feat_hr, _hr_shape, aux = self._forward_sr_features(
                feat_lr,
                lr_shape=lr_shape,
                lr_labels=lr_labels,
                return_aux=True,
            )
            return self.decode(feat_hr), aux

        feat_hr, _hr_shape = self._forward_sr_features(
            feat_lr,
            lr_shape=lr_shape,
            lr_labels=lr_labels,
            return_aux=False,
        )
        return self.decode(feat_hr)

    def forward(
        self,
        lr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = True,
        lr_labels: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        return self.forward_sr(
            lr_quats,
            lr_shape=lr_shape,
            normalize_input=normalize_input,
            lr_labels=lr_labels,
            return_aux=return_aux,
        )

    def feature_loss_sr(
        self,
        lr_quats: torch.Tensor,
        hr_quats: torch.Tensor,
        lr_shape: tuple[int, int],
        normalize_input: bool = True,
        lr_labels: torch.Tensor | None = None,
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
            feat_lr = self.encode(lr_flat).detach()
            feat_hr_tgt = self.encode(hr_flat).detach()

        if batched:
            feat_lr = feat_lr.reshape(B, -1, feat_lr.shape[-1])
            feat_hr_tgt = feat_hr_tgt.reshape(B, -1, feat_hr_tgt.shape[-1])

        feat_hr_pred, _ = self._forward_sr_features(
            feat_lr,
            lr_shape=lr_shape,
            lr_labels=lr_labels,
            return_aux=False,
        )
        return F.mse_loss(feat_hr_pred, feat_hr_tgt)


__all__ = [
    "LocalIsoCrystalEncoder",
    "CubochoricOptimizingLocalIsoDecoder",
    "PhaseEmbedding4x4",
    "SharedWindowGate",
    "InvariantStatBuilder",
    "InvariantCandidateScorer",
    "InvariantRouter",
    "EquivariantQueryAnchor",
    "ExpertEquivariantSeedHead",
    "RRCTPSemiglobalUpsampler",
    "IsoEmbeddingSRRRCTP",
]
