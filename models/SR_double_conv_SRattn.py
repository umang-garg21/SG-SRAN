"""
Standalone local-iso SR model with double LR conv and HR attention.

Key requirements addressed:
- Single model class: `IsoEmbeddingSRAttn`
- Uses local-iso embedding encoder
- Uses cubochoric-sampled optimizing decoder (feature-space optimization)
- Crystal family selected at top level (`crystal='fcc'` or `crystal='hcp'`)
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Decode local-iso irreps features (a1 or full) to passive quaternions.

    Seeds are taken from cubochoric fundamental-zone samples and refined with Adam
    to minimize feature-space MSE against target features.
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
        qn = (feat_target * feat_target).sum(dim=-1, keepdim=True)
        dist = qn + self.table_feat_norm.unsqueeze(0) - 2.0 * (feat_target @ self.table_feat.T)
        _, idx = torch.topk(dist, k=k, largest=False, dim=1)
        return idx

    def _encode_target_features(self, quats_passive: torch.Tensor) -> torch.Tensor:
        if self.target_irreps == "a1":
            return self.encoder.forward_a1(quats_passive)
        return self.encoder.forward_full(quats_passive)

    def forward(self, feat_target: torch.Tensor) -> torch.Tensor:
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
    """Learnable MLP decoder from a1 features to passive unit quaternions."""

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
    def __init__(
        self,
        kernel_size: int = 3,
        irreps_in: Irreps | str = "1x4e + 1x6e",
        irreps_out: Irreps | str | None = None,
        use_residual: bool = True,
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
        patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        w = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
        neigh = (patches * w).sum(dim=(-1, -2))

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
    def __init__(
        self,
        kernel_size: int = 3,
        upsample_factor: int = 4,
        use_residual: bool = False,
        irreps_io: Irreps | str = "1x4e + 1x6e",
    ):
        super().__init__()
        self.upsample_factor = int(upsample_factor)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.use_residual = bool(use_residual)

        self.irreps_io = Irreps(irreps_io)
        self.total_dim = int(self.irreps_io.dim)
        C = self.total_dim

        tp_kernel = self.upsample_factor + 2
        tp_pad = (tp_kernel - self.upsample_factor) // 2
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=C,
            out_channels=C,
            kernel_size=tp_kernel,
            stride=self.upsample_factor,
            padding=tp_pad,
            output_padding=0,
            groups=C,
            bias=False,
        )
        with torch.no_grad():
            self._init_bilinear()

        self.spatial_weights = nn.Parameter(
            torch.ones(self.kernel_size, self.kernel_size) / (self.kernel_size * self.kernel_size)
        )
        self.tp = FullyConnectedTensorProduct(
            self.irreps_io,
            self.irreps_io,
            self.irreps_io,
            shared_weights=True,
        )

    def _init_bilinear(self) -> None:
        r = self.upsample_factor
        k = self.transpose_conv.kernel_size[0]
        bilinear_1d = torch.zeros(k)
        center = (k - 1) / 2.0
        for i in range(k):
            bilinear_1d[i] = max(0.0, 1.0 - abs(i - center) / r)
        bilinear_2d = bilinear_1d.unsqueeze(1) * bilinear_1d.unsqueeze(0)
        bilinear_2d = bilinear_2d / bilinear_2d.sum()
        self.transpose_conv.weight.data[:] = bilinear_2d.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        features: torch.Tensor,
        img_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        H, W = img_shape
        r = self.upsample_factor
        Hr, Wr = H * r, W * r

        batched = features.dim() == 3
        if not batched:
            features = features.unsqueeze(0)
        B = features.shape[0]
        C = features.shape[-1]
        if C != self.total_dim:
            raise ValueError(f"Expected feature dim {self.total_dim}, got {C}")
        if features.shape[1] != H * W:
            raise ValueError(f"Expected N={H*W}, got N={features.shape[1]}")

        feat_img = features.view(B, H, W, C).permute(0, 3, 1, 2)
        feat_hr = self.transpose_conv(feat_img)[:, :, :Hr, :Wr]

        feat_padded = F.pad(feat_hr, [self.padding] * 4, mode="replicate")
        patches = feat_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        w = self.spatial_weights.view(1, 1, 1, 1, self.kernel_size, self.kernel_size)
        context = (patches * w).sum(dim=(-1, -2))

        N = Hr * Wr
        feat_flat = feat_hr.permute(0, 2, 3, 1).reshape(B * N, C)
        context_flat = context.permute(0, 2, 3, 1).reshape(B * N, C)
        out = self.tp(feat_flat, context_flat)
        if self.use_residual:
            out = out + feat_flat

        out = out.reshape(B, N, C)
        if not batched:
            out = out.squeeze(0)
        return out, (Hr, Wr)


class AttentionBlock(nn.Module):
    """Block-local equivariant attention on full-feature tensors."""

    def __init__(self, irreps_feat: Irreps | str, num_channels: int = 8):
        super().__init__()
        self.irreps_feat = Irreps(irreps_feat)
        self.feat_dim = int(self.irreps_feat.dim)
        self.sh_irreps = Irreps("1x0e + 1x2e")

        hidden_terms = []
        for mul, ir in self.irreps_feat:
            hidden_terms.append((int(mul) * int(num_channels), ir))
        self.irreps_h = Irreps(hidden_terms).simplify()

        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(float(self.feat_dim)))))
        self.pos_bias = nn.Linear(6, 1, bias=True)
        nn.init.zeros_(self.pos_bias.weight)
        nn.init.zeros_(self.pos_bias.bias)

        self.lin_in = IrrepsLinear(self.irreps_feat, self.irreps_h)
        self.tp_val = FullyConnectedTensorProduct(
            self.irreps_h, self.sh_irreps, self.irreps_h, shared_weights=True
        )
        self.tp_out = FullyConnectedTensorProduct(
            self.irreps_h, self.irreps_h, self.irreps_h, shared_weights=True
        )
        self.lin_out = IrrepsLinear(self.irreps_h, self.irreps_feat)
        with torch.no_grad():
            self.lin_out.weight.data.zero_()

    def forward(
        self,
        feat: torch.Tensor,
        sh_block: torch.Tensor,
        H: int,
        W: int,
        block_h: int,
        block_w: int,
    ) -> torch.Tensor:
        B, N, C = feat.shape
        if C != self.feat_dim:
            raise ValueError(f"Expected feature dim {self.feat_dim}, got {C}")
        if N != H * W:
            raise ValueError(f"Expected N={H*W}, got {N}")

        num_bh = H // block_h
        num_bw = W // block_w
        Nb = block_h * block_w
        Bb = B * num_bh * num_bw
        dtype = feat.dtype

        feat_blocks = (
            feat.reshape(B, num_bh, block_h, num_bw, block_w, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bb, Nb, C)
        )

        scale = torch.exp(self.log_scale)
        feat_n = F.normalize(feat_blocks, dim=-1)
        scores = scale * torch.bmm(feat_n, feat_n.transpose(-2, -1))

        pb = self.pos_bias(sh_block).squeeze(-1)
        scores = scores + pb.view(1, Nb, 1) + pb.view(1, 1, Nb)
        attn = torch.softmax(scores.float(), dim=-1).to(dtype)

        h = self.lin_in(feat_blocks.reshape(Bb * Nb, C)).reshape(Bb, Nb, -1)
        Ch = h.shape[-1]
        sh_flat = sh_block.unsqueeze(0).expand(Bb, Nb, -1).reshape(Bb * Nb, -1)
        vals = self.tp_val(h.reshape(Bb * Nb, Ch), sh_flat).reshape(Bb, Nb, Ch)
        ctx = torch.bmm(attn, vals)
        h_out = self.tp_out(h.reshape(Bb * Nb, Ch), ctx.reshape(Bb * Nb, Ch)).reshape(Bb, Nb, Ch)
        delta_blocks = self.lin_out(h_out.reshape(Bb * Nb, Ch)).reshape(Bb, Nb, C)

        delta = (
            delta_blocks.reshape(B, num_bh, num_bw, block_h, block_w, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, H * W, C)
        )
        return delta


class IsoEmbeddingSRAttn(nn.Module):
    """
    Standalone local-iso SR model with double LR conv + HR attention.

    Crystal family is selected at top-level with `crystal`:
    - `fcc` (group O)
    - `hcp` (group D6)
    """

    _SH_IRREPS = Irreps("1x0e + 1x2e")

    def __init__(
        self,
        crystal: str = "fcc",
        d6_convention: str = "z_axis",
        device: str | torch.device | None = None,
        upsample_factor: int = 4,
        upsample_residual: bool = True,
        num_hr_attn_blocks: int = 1,
        hr_attn_num_channels: int = 8,
        hr_attn_block_size: int = 16,
        decoder_cubochoric_resolution: int = 1,
        decoder_num_starts: int = 6,
        decoder_steps: int = 25,
        decoder_lr: float = 0.05,
        decoder_method: str = "cubochoric",
        decoder_max_table_rows: int | None = None,
        decoder_table_cache_dir: str | Path | None = "out/decoder_lookup_tables",
        decoder_backend: str = "optimizing",
        decoder_learnable_hidden_dim: int = 256,
        decoder_learnable_num_layers: int = 3,
        decoder_learnable_dropout: float = 0.0,
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
        self.feature_dim = int(self.encoder.out_dim_full)
        self.output_irreps = self.irreps_a1
        self.output_dim = self.feature_dim_a1
        self.upsample_factor = int(upsample_factor)
        self.hr_attn_block_size = int(hr_attn_block_size)

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

        # Requested architecture:
        # LR conv1 k=3: a1 -> full
        self.conv_lr1 = EquivariantSpatialConv(
            kernel_size=3,
            irreps_in=self.irreps_a1,
            irreps_out=self.irreps_full,
            # Keep a projected skip to avoid early feature collapse (a1 -> full).
            use_residual=True,
        )
        # LR conv2 k=9: full -> full
        self.conv_lr2 = EquivariantSpatialConv(
            kernel_size=9,
            irreps_in=self.irreps_full,
            irreps_out=self.irreps_full,
        )
        # Upsample k=3: full -> full
        self.upsample_conv = EquivariantTransposeConv(
            kernel_size=3,
            upsample_factor=self.upsample_factor,
            use_residual=bool(upsample_residual),
            irreps_io=self.irreps_full,
        )
        # HR conv1: full -> full
        self.conv_hr1 = EquivariantSpatialConv(
            kernel_size=3,
            irreps_in=self.irreps_full,
            irreps_out=self.irreps_full,
        )
        # Attention block(s)
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(self.irreps_full, num_channels=int(hr_attn_num_channels))
                for _ in range(max(1, int(num_hr_attn_blocks)))
            ]
        )
        # Final projection: full -> a1 (irreps output).
        self.final_proj = o3.Linear(self.irreps_full, self.irreps_a1)

        self._cached_hr_block_shape: tuple[int, int] | None = None
        self._cached_hr_sh_block: torch.Tensor | None = None

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

    def encode_a1(self, quats: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_a1(quats)

    def encode_full_target(self, quats: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_full(quats)

    def reduce_to_fz(
        self,
        quats: torch.Tensor,
        return_op_map: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
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
        if (
            self._cached_hr_block_shape == (block_h, block_w)
            and self._cached_hr_sh_block is not None
            and self._cached_hr_sh_block.device == device
        ):
            return self._cached_hr_sh_block.to(dtype)

        ys = torch.linspace(-1.0, 1.0, block_h, device=device)
        xs = torch.linspace(-1.0, 1.0, block_w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        dirs = torch.stack(
            [
                grid_x.reshape(-1),
                grid_y.reshape(-1),
                torch.zeros(block_h * block_w, device=device),
            ],
            dim=-1,
        )
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sh = o3.spherical_harmonics(self._SH_IRREPS, dirs, normalize=False)
        self._cached_hr_block_shape = (block_h, block_w)
        self._cached_hr_sh_block = sh
        return sh.to(dtype)

    def _apply_attention(
        self,
        features: torch.Tensor,
        hr_shape: tuple[int, int],
    ) -> torch.Tensor:
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
        if pad_h > 0 or pad_w > 0:
            feat_2d = feat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
            feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
            feat = feat_2d.permute(0, 2, 3, 1).reshape(B, Hr_pad * Wr_pad, C)

        sh_block = self._get_hr_sh_block(block_h, block_w, feat.device, feat.dtype)
        for block in self.attention_blocks:
            feat = feat + block(feat, sh_block, Hr_pad, Wr_pad, block_h, block_w)

        if pad_h > 0 or pad_w > 0:
            feat = feat.reshape(B, Hr_pad, Wr_pad, C)[:, :Hr, :Wr, :].reshape(B, Hr * Wr, C)

        if not batched:
            feat = feat.squeeze(0)
        return feat

    def _forward_sr_features(
        self,
        feat_lr_a1: torch.Tensor,
        lr_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        feat = self.conv_lr1(feat_lr_a1, lr_shape)
        feat = self.conv_lr2(feat, lr_shape)
        feat, hr_shape = self.upsample_conv(feat, lr_shape)
        feat = self.conv_hr1(feat, hr_shape)
        feat_full = self._apply_attention(feat, hr_shape)

        batched = feat_full.dim() == 3
        if not batched:
            feat_full = feat_full.unsqueeze(0)
        B, N, C = feat_full.shape
        feat_a1 = self.final_proj(feat_full.reshape(B * N, C)).reshape(B, N, self.feature_dim_a1)
        if not batched:
            feat_a1 = feat_a1.squeeze(0)
        return feat_a1, hr_shape

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
    ) -> torch.Tensor:
        lr_quats = lr_quats.to(self.device)
        if normalize_input:
            lr_quats = _normalize_quaternions(lr_quats)
        feat_lr_a1 = self.encode_a1(lr_quats)
        feat_hr_a1, _ = self._forward_sr_features(feat_lr_a1, lr_shape)
        return self.decode(feat_hr_a1)

    def forward(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
        normalize_input: bool = True,
    ) -> torch.Tensor:
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
    "CubochoricOptimizingLocalIsoDecoder",
    "EquivariantSpatialConv",
    "EquivariantTransposeConv",
    "IsoEmbeddingSRAttn",
    "LearnableA1QuaternionDecoder",
    "LocalIsoCrystalEncoder",
]
