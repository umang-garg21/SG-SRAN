# ---------------------------------------------------------------------------
# Irrep-respecting true-attention SR (FCC-focused)
# Standalone module version.
#
# Overview
# --------
# This variant keeps representation learning in the encoder's A1 feature space.
# HR queries stay at pixel resolution for boundary fidelity, while LR keys/values
# are patchified to reduce attention compute.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps

from models.SR_double_conv_SRattn_a1_boundary_refine import (
    CubochoricOptimizingLocalIsoDecoder,
    EquivariantSpatialConv,
    LocalIsoCrystalEncoder,
)
from models.local_iso_embedding import _quat_to_matrix_active

# ============================================================
# Small helpers
# ============================================================


def _flat_to_image(x: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
    """
    (B, H*W, C) or (H*W, C) -> (B, C, H, W) or (C, H, W)
    """
    h, w = img_shape
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)

    b, n, c = x.shape
    if n != h * w:
        raise ValueError(f"Expected N={h*w}, got {n}")

    y = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return y if batched else y.squeeze(0)


def _image_to_flat(x: torch.Tensor) -> torch.Tensor:
    """
    (B, C, H, W) or (C, H, W) -> (B, H*W, C) or (H*W, C)
    """
    batched = x.dim() == 4
    if not batched:
        x = x.unsqueeze(0)

    b, c, h, w = x.shape
    y = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    return y if batched else y.squeeze(0)


def _normalize_quaternions(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternion vectors with a numerically-stable denominator."""
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, eps: float = 1.0) -> torch.Tensor:
    """Average a tensor with an optional broadcastable spatial mask."""
    if mask is None:
        return x.mean()
    w = mask.to(dtype=x.dtype)
    return (x * w).sum() / w.sum().clamp_min(eps)


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
    probe_stages.append(
        {
            "name": str(name),
            # Probe tensors are diagnostic-only; move them to CPU immediately
            # to avoid retaining large HR feature maps on CUDA across the whole
            # notebook/demo flow.
            "feat": feat_store.detach().cpu().clone().contiguous(),
            "shape": (int(img_shape[0]), int(img_shape[1])),
        }
    )


def _as_scale_tuple(scale: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize scalar or anisotropic scale config into (scale_y, scale_x)."""
    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError(f"Expected 2D scale, got {scale!r}")
        return int(scale[0]), int(scale[1])
    s = int(scale)
    return s, s


def _build_normalized_coords(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build flattened 2D coordinates in [-1, 1] for positional encoding."""
    y = torch.linspace(-1.0, 1.0, steps=max(1, h), device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=max(1, w), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _feature_block_slices(irreps: Irreps | str) -> list[tuple[int, int]]:
    """Return slice boundaries for each irrep family in a flattened feature vector."""
    return [(sl.start, sl.stop) for sl in Irreps(irreps).slices()]


def _patchify_lr_kv(
    feat_lr_img: torch.Tensor,
    patch_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """
    Patchify LR feature map for KV tokens while keeping channel dimension unchanged.

    Returns:
      - kv_tokens: (B, N_patch, C)
      - kv_coords: (N_patch, 2) normalized in [-1,1]
      - patch_grid: (H_patch, W_patch)
    """
    b, c, h, w = feat_lr_img.shape
    py, px = patch_size
    if py <= 0 or px <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size!r}")

    pad_h = (py - (h % py)) % py
    pad_w = (px - (w % px)) % px
    feat_pad = F.pad(feat_lr_img, (0, pad_w, 0, pad_h), mode="replicate")

    patches = F.unfold(feat_pad, kernel_size=(py, px), stride=(py, px))
    n_patch = int(patches.shape[-1])
    patches = patches.view(b, c, py * px, n_patch)
    kv_tokens = patches.mean(dim=2).transpose(1, 2).contiguous()

    h_pad = h + pad_h
    w_pad = w + pad_w
    h_patch = h_pad // py
    w_patch = w_pad // px

    y_centers = torch.arange(h_patch, device=feat_lr_img.device, dtype=feat_lr_img.dtype) * py + (py - 1.0) / 2.0
    x_centers = torch.arange(w_patch, device=feat_lr_img.device, dtype=feat_lr_img.dtype) * px + (px - 1.0) / 2.0
    y_centers = y_centers.clamp(0.0, max(0.0, h - 1.0))
    x_centers = x_centers.clamp(0.0, max(0.0, w - 1.0))
    yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")

    if h > 1:
        y_norm = 2.0 * (yy / (h - 1.0)) - 1.0
    else:
        y_norm = torch.zeros_like(yy)
    if w > 1:
        x_norm = 2.0 * (xx / (w - 1.0)) - 1.0
    else:
        x_norm = torch.zeros_like(xx)

    kv_coords = torch.stack([x_norm.reshape(-1), y_norm.reshape(-1)], dim=-1)
    return kv_tokens, kv_coords, (h_patch, w_patch)


def _weighted_geometric_median(
    candidates: torch.Tensor,
    weights: torch.Tensor,
    num_iters: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Robust weighted geometric median for HR seed aggregation.

    Args:
      candidates: (B, M, C, H, W)
      weights:    (B, M, H, W), expected non-negative
    Returns:
      seed: (B, C, H, W)
    """
    w = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    z = (candidates * w.unsqueeze(2)).sum(dim=1)

    for _ in range(max(1, int(num_iters))):
        dist = (candidates - z.unsqueeze(1)).pow(2).sum(dim=2).sqrt().clamp_min(eps)
        alpha = w / dist
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(eps)
        z = (candidates * alpha.unsqueeze(2)).sum(dim=1)
    return z


def _channelwise_median_pool2d(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Channel-wise spatial median filter for robust HR seed cleanup."""
    k = int(kernel_size)
    if k <= 1:
        return x
    pad = k // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    patches = F.unfold(xp, kernel_size=k, stride=1)
    b, _, l = patches.shape
    c = int(x.shape[1])
    patches = patches.view(b, c, k * k, l)
    med = patches.median(dim=2).values
    return med.view(b, c, int(x.shape[-2]), int(x.shape[-1]))


def _candidate_channel_median(candidates: torch.Tensor) -> torch.Tensor:
    """Median-pool HR seed candidates along candidate axis (B,M,C,H,W -> B,C,H,W)."""
    if candidates.dim() != 5:
        raise ValueError(f"Expected candidates shape (B,M,C,H,W), got {tuple(candidates.shape)}")
    return candidates.median(dim=1).values


class LRTokenSelfAttentionBlock(nn.Module):
    """Lightweight self-attention block operating on LR block tokens in A1 space."""

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.feature_dim)
        hid = max(self.feature_dim, int(round(float(mlp_ratio) * self.feature_dim)))
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hid),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hid, self.feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Multi-path seeded cross-attention blocks
# ============================================================
class MultiPathCrossAttentionLayer(nn.Module):
    """One HR<-LR cross-attention layer with per-layer 3-path query seeding."""

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = float(self.head_dim) ** -0.5

        self.norm_q = nn.LayerNorm(self.feature_dim)
        self.norm_kv = nn.LayerNorm(self.feature_dim)
        self.norm_mlp = nn.LayerNorm(self.feature_dim)

        self.q_proj = nn.Linear(self.feature_dim, self.inner_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.inner_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.feature_dim)

        self.logit_scale = nn.Parameter(torch.zeros(self.num_heads))
        self.attn_dropout = nn.Dropout(float(dropout))

        mlp_hidden = max(self.feature_dim, int(round(float(mlp_ratio) * self.feature_dim)))
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden, self.feature_dim),
        )

        # Per-block multi-path seed fusion:
        # A = bilinear/content seed, B = positional seed, C = projected previous state
        self.seed_c_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.seed_gate = nn.Linear(self.feature_dim, 3)
        self.seed_alpha = nn.Parameter(torch.tensor(1.0))
        self.block_pos_bias = nn.Parameter(torch.zeros(1, 1, self.feature_dim))

    def forward(
        self,
        x: torch.Tensor,
        seed_content: torch.Tensor,
        seed_pos: torch.Tensor,
        kv_tokens: torch.Tensor,
        kv_pos: torch.Tensor,
        query_chunk_size: int = 2048,
        return_stats: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        b, nq, _ = x.shape
        nk = int(kv_tokens.shape[1])

        seed_b = seed_pos + self.block_pos_bias
        seed_c = self.seed_c_proj(x)
        gate = torch.softmax(self.seed_gate(x), dim=-1)
        fused_seed = (
            gate[..., 0:1] * seed_content
            + gate[..., 1:2] * seed_b
            + gate[..., 2:3] * seed_c
        )

        q_tokens = x + torch.tanh(self.seed_alpha) * fused_seed

        q_in = self.norm_q(q_tokens)
        kv_in = self.norm_kv(kv_tokens + kv_pos)

        q = self.q_proj(q_in).view(b, nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_in).view(b, nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_in).view(b, nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = self.scale * torch.exp(self.logit_scale).view(1, self.num_heads, 1, 1)
        scale = scale.to(device=q.device, dtype=q.dtype)

        ctx_chunks: list[torch.Tensor] = []
        entropy_chunks: list[torch.Tensor] = []
        peak_idx_chunks: list[torch.Tensor] = []
        peak_weight_chunks: list[torch.Tensor] = []
        source_importance = q.new_zeros((b, self.num_heads, nk), dtype=torch.float32) if return_stats else None

        chunk = max(1, int(query_chunk_size))
        for start in range(0, nq, chunk):
            end = min(start + chunk, nq)
            q_chunk = q[:, :, start:end, :]

            if return_stats:
                scores = torch.einsum("bhqd,bhkd->bhqk", q_chunk, k)
                scores = scores * scale
                attn = torch.softmax(scores, dim=-1)
                attn_drop = self.attn_dropout(attn)

                ctx = torch.einsum("bhqk,bhkd->bhqd", attn_drop, v)
                ctx_chunks.append(ctx.permute(0, 2, 1, 3).reshape(b, end - start, self.inner_dim))

                attn_stats = attn.float()
                peak_w, peak_i = attn_stats.max(dim=-1)
                ent = -(attn_stats.clamp_min(1e-8) * attn_stats.clamp_min(1e-8).log()).sum(dim=-1)
                entropy_chunks.append(ent)
                peak_idx_chunks.append(peak_i)
                peak_weight_chunks.append(peak_w)
                source_importance = source_importance + attn_stats.sum(dim=2)
            else:
                # Memory-efficient path for normal training/inference when attention
                # diagnostics are not requested.
                head_scale = torch.exp(self.logit_scale).view(1, self.num_heads, 1, 1)
                head_scale = head_scale.to(device=q_chunk.device, dtype=q_chunk.dtype)
                q_scaled = q_chunk * head_scale
                attn_dropout_p = float(self.attn_dropout.p) if self.training else 0.0
                ctx = F.scaled_dot_product_attention(
                    q_scaled,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=attn_dropout_p,
                    is_causal=False,
                    scale=self.scale,
                )
                ctx_chunks.append(ctx.permute(0, 2, 1, 3).reshape(b, end - start, self.inner_dim))

        ctx_all = torch.cat(ctx_chunks, dim=1)

        y = x + self.out_proj(ctx_all)
        y = y + self.mlp(self.norm_mlp(y))

        if not return_stats:
            return y, None

        stats = {
            "attention_entropy": torch.cat(entropy_chunks, dim=-1),
            "attention_peak_idx": torch.cat(peak_idx_chunks, dim=-1),
            "attention_peak_weight": torch.cat(peak_weight_chunks, dim=-1),
            "attention_source_importance": source_importance / float(max(1, nq)),
            "seed_gate_mean": gate.mean(dim=(0, 1)).detach(),
        }
        return y, stats

class ViTPixelQueryPatchKVAupsampler(nn.Module):
    """HR pixel-query / LR patch-KV attention upsampler with multi-path block seeding."""

    def __init__(
        self,
        feature_dim: int,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        lr_patch_size: int | tuple[int, int] | list[int] = (2, 2),
        num_heads: int = 8,
        head_dim: int = 32,
        num_layers: int = 4,
        mlp_ratio: float = 2.0,
        pos_hidden_dim: int = 64,
        dropout: float = 0.0,
        query_chunk_size: int = 2048,
        seed_mode: str = "bilinear",
        seed_num_candidates: int = 4,
        seed_lr_self_attn_layers: int = 1,
        seed_lr_self_attn_heads: int = 3,
        seed_lr_block_size: int | tuple[int, int] | list[int] = (8, 8),
        seed_gm_iters: int = 3,
        seed_use_hr_median_pool: bool = False,
        seed_hr_median_kernel_size: int = 3,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.lr_patch_size = _as_scale_tuple(lr_patch_size)
        self.num_heads = int(num_heads)
        self.num_layers = max(1, int(num_layers))
        self.query_chunk_size = max(1, int(query_chunk_size))

        mode = str(seed_mode).strip().lower()
        if mode not in {"bilinear", "lr_self_attn_median"}:
            raise ValueError(
                "seed_mode must be 'bilinear' or 'lr_self_attn_median'; "
                f"got {seed_mode!r}"
            )
        self.seed_mode = mode
        self.seed_num_candidates = max(1, int(seed_num_candidates))
        self.seed_lr_self_attn_layers = max(1, int(seed_lr_self_attn_layers))
        self.seed_lr_self_attn_heads = max(1, int(seed_lr_self_attn_heads))
        self.seed_lr_block_size = _as_scale_tuple(seed_lr_block_size)
        self.seed_gm_iters = max(1, int(seed_gm_iters))
        self.seed_use_hr_median_pool = bool(seed_use_hr_median_pool)
        self.seed_hr_median_kernel_size = max(1, int(seed_hr_median_kernel_size))

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, int(pos_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(pos_hidden_dim), self.feature_dim),
        )

        self.layers = nn.ModuleList(
            [
                MultiPathCrossAttentionLayer(
                    feature_dim=self.feature_dim,
                    num_heads=self.num_heads,
                    head_dim=int(head_dim),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.seed_mode == "lr_self_attn_median":
            if self.feature_dim % self.seed_lr_self_attn_heads != 0:
                raise ValueError(
                    "seed_lr_self_attn_heads must divide feature_dim for MultiheadAttention; "
                    f"got feature_dim={self.feature_dim}, heads={self.seed_lr_self_attn_heads}"
                )
            self.seed_lr_pos_mlp = nn.Sequential(
                nn.Linear(2, int(pos_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(pos_hidden_dim), self.feature_dim),
            )
            self.seed_lr_self_attn_blocks = nn.ModuleList(
                [
                    LRTokenSelfAttentionBlock(
                        feature_dim=self.feature_dim,
                        num_heads=self.seed_lr_self_attn_heads,
                        mlp_ratio=float(mlp_ratio),
                        dropout=float(dropout),
                    )
                    for _ in range(self.seed_lr_self_attn_layers)
                ]
            )
            self.seed_candidate_proj = nn.Conv2d(
                self.feature_dim,
                self.seed_num_candidates * self.feature_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.seed_weight_proj = nn.Conv2d(
                self.feature_dim,
                self.seed_num_candidates,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.seed_lr_pos_mlp = None
            self.seed_lr_self_attn_blocks = nn.ModuleList()
            self.seed_candidate_proj = None
            self.seed_weight_proj = None

    def _build_seed_content_img(
        self,
        feat_lr_img: torch.Tensor,
        hr_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, object]]:
        h_hr, w_hr = int(hr_shape[0]), int(hr_shape[1])
        h_lr, w_lr = int(feat_lr_img.shape[-2]), int(feat_lr_img.shape[-1])

        if self.seed_mode == "bilinear":
            seed_content_img = F.interpolate(
                feat_lr_img,
                size=(h_hr, w_hr),
                mode="bilinear",
                align_corners=False,
            )
            return seed_content_img, {}

        if self.seed_lr_pos_mlp is None or self.seed_candidate_proj is None or self.seed_weight_proj is None:
            raise RuntimeError("Robust seed modules are not initialized.")

        bsz = int(feat_lr_img.shape[0])
        block_tokens, block_coords, block_grid = _patchify_lr_kv(feat_lr_img, self.seed_lr_block_size)
        h_blk, w_blk = block_grid

        lr_block_pos = self.seed_lr_pos_mlp(block_coords).unsqueeze(0).expand(bsz, -1, -1)
        lr_tokens = block_tokens + lr_block_pos
        for blk in self.seed_lr_self_attn_blocks:
            lr_tokens = blk(lr_tokens)

        lr_ctx_img = lr_tokens.transpose(1, 2).reshape(bsz, self.feature_dim, h_blk, w_blk)
        lr_ctx_img = F.interpolate(lr_ctx_img, size=(h_lr, w_lr), mode="bilinear", align_corners=False)

        cand_lr = self.seed_candidate_proj(lr_ctx_img)
        cand_lr = cand_lr.view(bsz, self.seed_num_candidates, self.feature_dim, h_lr, w_lr)
        weight_lr_logits = self.seed_weight_proj(lr_ctx_img)

        cand_hr = F.interpolate(
            cand_lr.reshape(bsz, self.seed_num_candidates * self.feature_dim, h_lr, w_lr),
            size=(h_hr, w_hr),
            mode="bilinear",
            align_corners=False,
        ).view(bsz, self.seed_num_candidates, self.feature_dim, h_hr, w_hr)
        weight_hr_logits = F.interpolate(weight_lr_logits, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
        weight_hr = torch.softmax(weight_hr_logits, dim=1)

        # Aggregate multiple candidate seeds via robust median across candidate axis.
        seed_content_img = _candidate_channel_median(cand_hr)
        if self.seed_use_hr_median_pool:
            seed_content_img = _channelwise_median_pool2d(seed_content_img, self.seed_hr_median_kernel_size)

        seed_aux: dict[str, object] = {
            "seed_lr_ctx": lr_ctx_img,
            "seed_candidates_hr": cand_hr,
            "seed_candidate_weights_hr": weight_hr,
            "seed_block_grid": (h_blk, w_blk),
            "seed_block_size": self.seed_lr_block_size,
        }
        return seed_content_img, seed_aux

    def forward(
        self,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]] | torch.Tensor:
        h_lr, w_lr = lr_shape
        h_hr = int(h_lr * self.upsample_factor[0])
        w_hr = int(w_lr * self.upsample_factor[1])

        batched = feat_lr.dim() == 3
        if not batched:
            feat_lr = feat_lr.unsqueeze(0)

        b = int(feat_lr.shape[0])
        feat_lr_img = _flat_to_image(feat_lr, lr_shape)

        seed_content_img, seed_aux = self._build_seed_content_img(
            feat_lr_img=feat_lr_img,
            hr_shape=(h_hr, w_hr),
        )
        seed_content = _image_to_flat(seed_content_img)

        # LR patch tokens for K/V.
        kv_tokens, kv_coords, patch_grid = _patchify_lr_kv(feat_lr_img, self.lr_patch_size)
        h_patch, w_patch = patch_grid

        # Seed path B: positional seed (shared coordinates, block-specific bias inside layer).
        hr_coords = _build_normalized_coords(h_hr, w_hr, feat_lr.device, feat_lr.dtype)
        seed_pos = self.pos_mlp(hr_coords).unsqueeze(0).expand(b, -1, -1)
        kv_pos = self.pos_mlp(kv_coords).unsqueeze(0).expand(b, -1, -1)

        x = seed_content
        layer_outputs: list[torch.Tensor] = []
        layer_stats: list[dict[str, torch.Tensor]] = []

        for layer in self.layers:
            x, stats = layer(
                x=x,
                seed_content=seed_content,
                seed_pos=seed_pos,
                kv_tokens=kv_tokens,
                kv_pos=kv_pos,
                query_chunk_size=self.query_chunk_size,
                return_stats=return_aux,
            )
            if return_aux and stats is not None:
                layer_outputs.append(x.detach())
                layer_stats.append(stats)

        if not return_aux:
            return x if batched else x.squeeze(0)

        entropy = torch.stack([s["attention_entropy"] for s in layer_stats], dim=1)
        peak_idx = torch.stack([s["attention_peak_idx"] for s in layer_stats], dim=1)
        peak_weight = torch.stack([s["attention_peak_weight"] for s in layer_stats], dim=1)
        src_importance = torch.stack([s["attention_source_importance"] for s in layer_stats], dim=1)
        gate_means = torch.stack([s["seed_gate_mean"] for s in layer_stats], dim=0)

        peak_patch_y = torch.div(peak_idx, w_patch, rounding_mode="floor")
        peak_patch_x = torch.remainder(peak_idx, w_patch)

        py, px = self.lr_patch_size
        peak_lr_y = (peak_patch_y.float() * py + (py - 1.0) / 2.0).clamp(0.0, max(0.0, h_lr - 1.0))
        peak_lr_x = (peak_patch_x.float() * px + (px - 1.0) / 2.0).clamp(0.0, max(0.0, w_lr - 1.0))

        aux: dict[str, object] = {
            "feat_query_seed_hr": seed_content_img if batched else seed_content_img.squeeze(0),
            "seed_mode": self.seed_mode,
            "attention_entropy_hr": entropy.view(b, self.num_layers, self.num_heads, h_hr, w_hr),
            "attention_peak_patch_idx_hr": peak_idx.view(b, self.num_layers, self.num_heads, h_hr, w_hr),
            "attention_peak_weight_hr": peak_weight.view(b, self.num_layers, self.num_heads, h_hr, w_hr),
            "attention_peak_patch_y_hr": peak_patch_y.view(b, self.num_layers, self.num_heads, h_hr, w_hr),
            "attention_peak_patch_x_hr": peak_patch_x.view(b, self.num_layers, self.num_heads, h_hr, w_hr),
            "attention_peak_lr_y_hr": peak_lr_y.view(b, self.num_layers, self.num_heads, h_hr, w_hr),
            "attention_peak_lr_x_hr": peak_lr_x.view(b, self.num_layers, self.num_heads, h_hr, w_hr),
            "attention_source_importance_lrpatch": src_importance.view(
                b,
                self.num_layers,
                self.num_heads,
                h_patch,
                w_patch,
            ),
            "seed_path_gate_mean": gate_means,
            "attention_layer_outputs": layer_outputs,
            "lr_patch_grid": (h_patch, w_patch),
            "lr_patch_size": self.lr_patch_size,
            "hr_shape": (h_hr, w_hr),
        }
        aux.update(seed_aux)

        if not batched:
            x = x.squeeze(0)
            aux = {
                k: (v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1 else v)
                for k, v in aux.items()
            }

        return x, aux


class RRCTPUpsampler(nn.Module):
    """Nearest-seeded HR lift followed by rotation-routed conditional tensor products."""

    def __init__(
        self,
        feature_dim: int,
        irreps_feat: Irreps | str,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        num_experts: int = 4,
        top_k: int = 2,
        router_hidden_dim: int = 64,
        context_kernel_size: int = 3,
        gate_init_bias: float = -4.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.irreps_feat = Irreps(irreps_feat)
        if int(self.irreps_feat.dim) != self.feature_dim:
            raise ValueError(
                f"feature_dim={self.feature_dim} does not match irreps_feat.dim={self.irreps_feat.dim}"
            )
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.num_experts = max(2, int(num_experts))
        self.top_k = min(max(1, int(top_k)), self.num_experts)
        self.context_kernel_size = max(1, int(context_kernel_size))
        self.context_padding = self.context_kernel_size // 2
        self.context_logits = nn.Parameter(torch.zeros(self.context_kernel_size, self.context_kernel_size))

        self.block_slices = _feature_block_slices(self.irreps_feat)
        router_in_dim = max(1, 3 * len(self.block_slices))
        router_hidden = max(int(router_hidden_dim), router_in_dim)
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, self.num_experts + 1),
        )
        self.gate_init_bias = float(gate_init_bias)

        self.expert_quats = nn.Parameter(torch.zeros(self.num_experts, 4))
        self.tp = FullyConnectedTensorProduct(
            self.irreps_feat,
            self.irreps_feat,
            self.irreps_feat,
            shared_weights=True,
        )
        self.tp_update_proj = o3.Linear(self.irreps_feat, self.irreps_feat)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if isinstance(self.router[0], nn.Linear):
            nn.init.xavier_uniform_(self.router[0].weight)
            nn.init.zeros_(self.router[0].bias)
        if isinstance(self.router[2], nn.Linear):
            nn.init.zeros_(self.router[2].weight)
            nn.init.zeros_(self.router[2].bias)
            self.router[2].bias.data[-1] = float(self.gate_init_bias)
        if hasattr(self.tp_update_proj, "weight"):
            with torch.no_grad():
                self.tp_update_proj.weight.zero_()

        expert_quats = torch.zeros(self.num_experts, 4, dtype=self.expert_quats.dtype, device=self.expert_quats.device)
        expert_quats[0, 0] = 1.0
        preset_angles = [
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ]
        for idx in range(1, self.num_experts):
            axis = preset_angles[(idx - 1) % len(preset_angles)]
            theta = torch.tensor(torch.pi / 2.0, dtype=expert_quats.dtype, device=expert_quats.device)
            c = torch.cos(theta / 2.0)
            s = torch.sin(theta / 2.0)
            expert_quats[idx, 0] = c
            expert_quats[idx, 1:] = s * torch.tensor(axis[1:], dtype=expert_quats.dtype, device=expert_quats.device)
        with torch.no_grad():
            self.expert_quats.copy_(expert_quats)

    def configure_demo_identity(self) -> None:
        """Make checkpoint-free demos behave like a stable NN-seeded baseline."""
        with torch.no_grad():
            self.context_logits.zero_()
            if hasattr(self.tp, "weight"):
                self.tp.weight.zero_()
            if hasattr(self.tp_update_proj, "weight"):
                self.tp_update_proj.weight.zero_()
            if isinstance(self.router[2], nn.Linear):
                self.router[2].weight.zero_()
                self.router[2].bias.zero_()
                self.router[2].bias[-1] = float(self.gate_init_bias)

    def _nearest_seed_img(
        self,
        feat_lr_img: torch.Tensor,
        hr_shape: tuple[int, int],
    ) -> torch.Tensor:
        return F.interpolate(feat_lr_img, size=hr_shape, mode="nearest")

    def _build_context_img(self, seed_img: torch.Tensor) -> torch.Tensor:
        if self.context_kernel_size <= 1:
            return seed_img
        xp = F.pad(seed_img, (self.context_padding, self.context_padding, self.context_padding, self.context_padding), mode="replicate")
        patches = F.unfold(xp, kernel_size=self.context_kernel_size, stride=1)
        b, _, l = patches.shape
        c = int(seed_img.shape[1])
        k2 = int(self.context_kernel_size * self.context_kernel_size)
        patches = patches.view(b, c, k2, l)
        weights = torch.softmax(self.context_logits.reshape(-1), dim=0).view(1, 1, k2, 1)
        ctx = (patches * weights).sum(dim=2)
        return ctx.view(b, c, int(seed_img.shape[-2]), int(seed_img.shape[-1]))

    def _router_stats(self, feat_flat: torch.Tensor, ctx_flat: torch.Tensor) -> torch.Tensor:
        stats: list[torch.Tensor] = []
        for start, stop in self.block_slices:
            feat_blk = feat_flat[..., start:stop]
            ctx_blk = ctx_flat[..., start:stop]
            feat_norm = feat_blk.norm(dim=-1, keepdim=True)
            ctx_norm = ctx_blk.norm(dim=-1, keepdim=True)
            align = (feat_blk * ctx_blk).sum(dim=-1, keepdim=True)
            align = align / (feat_norm * ctx_norm).clamp_min(1e-8)
            stats.extend([feat_norm, ctx_norm, align])
        return torch.cat(stats, dim=-1)

    def _expert_d_mats(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        expert_quats = _normalize_quaternions(self.expert_quats.to(device=device, dtype=dtype))
        rot_mats = _quat_to_matrix_active(expert_quats)
        d_mats_cpu = self.irreps_feat.D_from_matrix(rot_mats.to(device="cpu", dtype=torch.float32))
        return d_mats_cpu.to(device=device, dtype=dtype)

    def forward(
        self,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]] | torch.Tensor:
        h_lr, w_lr = int(lr_shape[0]), int(lr_shape[1])
        h_hr = int(h_lr * self.upsample_factor[0])
        w_hr = int(w_lr * self.upsample_factor[1])

        batched = feat_lr.dim() == 3
        if not batched:
            feat_lr = feat_lr.unsqueeze(0)

        b = int(feat_lr.shape[0])
        feat_lr_img = _flat_to_image(feat_lr, lr_shape)
        seed_img = self._nearest_seed_img(feat_lr_img, (h_hr, w_hr))
        ctx_img = self._build_context_img(seed_img)

        seed_flat = _image_to_flat(seed_img)
        ctx_flat = _image_to_flat(ctx_img)
        router_stats = self._router_stats(seed_flat, ctx_flat)
        router_out = self.router(router_stats)
        expert_logits = router_out[..., : self.num_experts]
        gate = torch.sigmoid(router_out[..., self.num_experts : self.num_experts + 1])

        topk_logits, topk_idx = torch.topk(expert_logits, k=self.top_k, dim=-1)
        topk_weight = torch.softmax(topk_logits, dim=-1)

        d_mats = self._expert_d_mats(device=ctx_flat.device, dtype=ctx_flat.dtype)
        rotated_ctx_all = torch.einsum("kdc,bnc->bnkd", d_mats, ctx_flat)
        topk_ctx = torch.gather(
            rotated_ctx_all,
            dim=2,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.feature_dim),
        )

        skip_topk = seed_flat.unsqueeze(2).expand(-1, -1, self.top_k, -1)
        tp_update = self.tp(
            skip_topk.reshape(-1, self.feature_dim),
            topk_ctx.reshape(-1, self.feature_dim),
        )
        tp_update = self.tp_update_proj(tp_update).reshape(b, h_hr * w_hr, self.top_k, self.feature_dim)
        tp_candidate = skip_topk + tp_update
        tp_blend = (topk_weight.unsqueeze(-1) * tp_candidate).sum(dim=2)
        out = gate * tp_blend + (1.0 - gate) * seed_flat

        if not return_aux:
            return out if batched else out.squeeze(0)

        expert_logits_img = expert_logits.view(b, h_hr, w_hr, self.num_experts).permute(0, 3, 1, 2).contiguous()
        gate_img = gate.view(b, h_hr, w_hr, 1).permute(0, 3, 1, 2).contiguous()
        topk_idx_img = topk_idx.view(b, h_hr, w_hr, self.top_k).permute(0, 3, 1, 2).contiguous()
        topk_weight_img = topk_weight.view(b, h_hr, w_hr, self.top_k).permute(0, 3, 1, 2).contiguous()
        out_img = _flat_to_image(out, (h_hr, w_hr))

        aux: dict[str, object] = {
            "feat_query_seed_hr": seed_img if batched else seed_img.squeeze(0),
            "feat_rr_ctp_context_hr": ctx_img if batched else ctx_img.squeeze(0),
            "feat_rr_ctp_out_hr": out_img if batched else out_img.squeeze(0),
            "rr_ctp_router_logits_hr": expert_logits_img if batched else expert_logits_img.squeeze(0),
            "rr_ctp_gate_hr": gate_img if batched else gate_img.squeeze(0),
            "rr_ctp_topk_idx_hr": topk_idx_img if batched else topk_idx_img.squeeze(0),
            "rr_ctp_topk_weight_hr": topk_weight_img if batched else topk_weight_img.squeeze(0),
            "rr_ctp_expert_quats": _normalize_quaternions(self.expert_quats.detach()).cpu(),
            "seed_mode": "nearest",
            "hr_shape": (h_hr, w_hr),
            "lr_patch_grid": None,
            "lr_patch_size": None,
        }
        return (out if batched else out.squeeze(0)), aux

# ============================================================
# Full model
# ============================================================

class IsoEmbeddingSRAttn(nn.Module):
    """
    FCC-focused true-attention SR model:

      LR quats
        -> encoder.forward_a1
        -> optional equivariant LR blocks
        -> HR pixel queries with per-layer multi-path seeding
        -> global multi-head HR<-LRpatch attention
        -> optimizing decoder back to quats

    No explicit boundary-prep branch is used in this variant.
    """

    def __init__(
        self,
        crystal: str = "fcc",
        d6_convention: str = "z_axis",
        device: str | torch.device | None = None,
        upsample_factor: int | tuple[int, int] | list[int] = 4,
        num_lr_blocks: int = 1,
        use_pre_lr: Optional[bool] = None,
        decoder_cubochoric_resolution: int = 1,
        decoder_num_starts: int = 6,
        decoder_steps: int = 25,
        decoder_lr: float = 0.05,
        decoder_method: str = "cubochoric",
        decoder_max_table_rows: int | None = None,
        decoder_table_cache_dir: str | Path | None = "out/decoder_lookup_tables",
        feature_upsampler_type: str = "vit_attention",
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
        attention_num_heads: int = 8,
        attention_head_dim: int = 32,
        attention_num_layers: int = 4,
        attention_mlp_ratio: float = 2.0,
        attention_pos_hidden_dim: int = 64,
        attention_dropout: float = 0.0,
        attention_query_chunk_size: int = 2048,
        attention_lr_patch_size: int | tuple[int, int] | list[int] = (2, 2),
        seed_mode: str = "bilinear",
        seed_num_candidates: int = 4,
        seed_lr_self_attn_layers: int = 1,
        seed_lr_self_attn_heads: int = 3,
        seed_lr_block_size: int | tuple[int, int] | list[int] = (8, 8),
        seed_gm_iters: int = 3,
        seed_use_hr_median_pool: bool = False,
        seed_hr_median_kernel_size: int = 3,
        rr_ctp_num_experts: int = 4,
        rr_ctp_top_k: int = 2,
        rr_ctp_router_hidden_dim: int = 64,
        rr_ctp_context_kernel_size: int = 3,
        use_hr_conv1: bool = False,
    ):
        super().__init__()

        if str(crystal).lower() != "fcc":
            raise ValueError("This implementation is finalized for FCC only. Use crystal='fcc'.")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.encoder = LocalIsoCrystalEncoder(
            crystal="fcc",
            d6_convention=d6_convention,
            dtype=torch.float32,
            device=self.device,
        )
        self.irreps_a1 = self.encoder.irreps_a1
        self.feature_dim_a1 = int(self.encoder.out_dim_a1)

        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.lambda_feat = float(lambda_feat)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_lr_boundary = float(lambda_lr_boundary)
        self.lambda_side_correct = float(lambda_side_correct)
        self.lambda_side_entropy = float(lambda_side_entropy)
        self.boundary_thr_deg = float(boundary_thr_deg)
        self.boundary_connectivity = int(boundary_connectivity)
        self.use_focal_boundary = bool(use_focal_boundary)
        self.focal_gamma = float(focal_gamma)
        self.side_correct_rel_gap = float(side_correct_rel_gap)
        self.side_correct_band_kernel = _as_scale_tuple(side_correct_band_kernel)

        mode = str(feature_upsampler_type).strip().lower()
        if mode == "shifted_bilinear":
            mode = "bilinear"
        if mode not in {"vit_attention", "bilinear", "rr_ctp"}:
            raise ValueError(
                "feature_upsampler_type must be 'vit_attention', 'bilinear', 'rr_ctp', or alias 'shifted_bilinear'; "
                f"got {feature_upsampler_type!r}"
            )
        self.feature_upsampler_type = mode
        self.use_hr_conv1 = bool(use_hr_conv1)

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

        self.true_attention_upsampler: ViTPixelQueryPatchKVAupsampler | None = None
        self.rr_ctp_upsampler: RRCTPUpsampler | None = None
        if self.feature_upsampler_type == "vit_attention":
            self.true_attention_upsampler = ViTPixelQueryPatchKVAupsampler(
                feature_dim=self.feature_dim_a1,
                upsample_factor=self.upsample_factor,
                lr_patch_size=attention_lr_patch_size,
                num_heads=int(attention_num_heads),
                head_dim=int(attention_head_dim),
                num_layers=int(attention_num_layers),
                mlp_ratio=float(attention_mlp_ratio),
                pos_hidden_dim=int(attention_pos_hidden_dim),
                dropout=float(attention_dropout),
                query_chunk_size=int(attention_query_chunk_size),
                seed_mode=str(seed_mode),
                seed_num_candidates=int(seed_num_candidates),
                seed_lr_self_attn_layers=int(seed_lr_self_attn_layers),
                seed_lr_self_attn_heads=int(seed_lr_self_attn_heads),
                seed_lr_block_size=seed_lr_block_size,
                seed_gm_iters=int(seed_gm_iters),
                seed_use_hr_median_pool=bool(seed_use_hr_median_pool),
                seed_hr_median_kernel_size=int(seed_hr_median_kernel_size),
            )
        elif self.feature_upsampler_type == "rr_ctp":
            self.rr_ctp_upsampler = RRCTPUpsampler(
                feature_dim=self.feature_dim_a1,
                irreps_feat=self.irreps_a1,
                upsample_factor=self.upsample_factor,
                num_experts=int(rr_ctp_num_experts),
                top_k=int(rr_ctp_top_k),
                router_hidden_dim=int(rr_ctp_router_hidden_dim),
                context_kernel_size=int(rr_ctp_context_kernel_size),
            )

        self.conv_hr1: EquivariantSpatialConv | None = None
        if self.use_hr_conv1:
            self.conv_hr1 = EquivariantSpatialConv(
                kernel_size=3,
                irreps_in=self.irreps_a1,
                irreps_out=self.irreps_a1,
                use_residual=True,
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

    def configure_demo_identity(self) -> None:
        """Stabilize checkpoint-free demos around a nearest-seeded identity baseline."""
        for blk in self.lr_blocks:
            if hasattr(blk, "tp") and hasattr(blk.tp, "weight"):
                with torch.no_grad():
                    blk.tp.weight.zero_()
            if hasattr(blk, "spatial_logits"):
                with torch.no_grad():
                    blk.spatial_logits.zero_()
        if self.conv_hr1 is not None:
            if hasattr(self.conv_hr1, "tp") and hasattr(self.conv_hr1.tp, "weight"):
                with torch.no_grad():
                    self.conv_hr1.tp.weight.zero_()
            if hasattr(self.conv_hr1, "spatial_logits"):
                with torch.no_grad():
                    self.conv_hr1.spatial_logits.zero_()
        if self.rr_ctp_upsampler is not None:
            self.rr_ctp_upsampler.configure_demo_identity()

    def encode_a1(self, quats: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_a1(quats)

    def decode(self, features_a1: torch.Tensor) -> torch.Tensor:
        """Decode A1 feature vectors back to normalized quaternions."""
        batched = features_a1.dim() == 3
        if batched:
            b, n, c = features_a1.shape
            q = self.decoder(features_a1.reshape(b * n, c))
            return _normalize_quaternions(q).reshape(b, n, 4)
        return _normalize_quaternions(self.decoder(features_a1))

    def _bilinear_feature_upsample(
        self,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, object]] | torch.Tensor:
        batched = feat_lr.dim() == 3
        if not batched:
            feat_lr = feat_lr.unsqueeze(0)

        h_lr, w_lr = lr_shape
        h_hr = int(h_lr * self.upsample_factor[0])
        w_hr = int(w_lr * self.upsample_factor[1])

        feat_lr_img = _flat_to_image(feat_lr, lr_shape)
        feat_hr_img = F.interpolate(feat_lr_img, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
        feat_hr = _image_to_flat(feat_hr_img)

        if not return_aux:
            return feat_hr if batched else feat_hr.squeeze(0)

        aux: dict[str, object] = {
            "feat_query_seed_hr": feat_hr_img if batched else feat_hr_img.squeeze(0),
            "hr_shape": (h_hr, w_hr),
            "lr_patch_grid": None,
            "lr_patch_size": None,
        }
        return (feat_hr if batched else feat_hr.squeeze(0)), aux

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
        del lr_boundary_map, lr_quats, extra_material_stats

        feat = feat_lr_a1.to(self.device)
        probe_stages: list[dict[str, object]] | None = [] if return_probe else None
        _append_probe_stage(probe_stages, "encode_a1_lr", feat, lr_shape)

        for bi, blk in enumerate(self.lr_blocks, start=1):
            feat = blk(feat, lr_shape)
            _append_probe_stage(probe_stages, f"lr_block_{bi}", feat, lr_shape)

        need_up_aux = bool(return_aux or return_probe)
        if self.feature_upsampler_type == "vit_attention":
            if self.true_attention_upsampler is None:
                raise RuntimeError("true_attention_upsampler is not initialized.")
            upsampled = self.true_attention_upsampler(
                feat_lr=feat,
                lr_shape=lr_shape,
                return_aux=need_up_aux,
            )
            if need_up_aux:
                feat_hr, up_aux = upsampled
                hr_shape = tuple(up_aux["hr_shape"])
                _append_probe_stage(probe_stages, "query_seed_hr", up_aux["feat_query_seed_hr"], hr_shape)
                layer_outputs = up_aux.get("attention_layer_outputs", [])
                for li, layer_feat in enumerate(layer_outputs, start=1):
                    _append_probe_stage(probe_stages, f"attention_block_{li}_hr", layer_feat, hr_shape)
            else:
                feat_hr = upsampled
                hr_shape = (
                    int(lr_shape[0]) * int(self.upsample_factor[0]),
                    int(lr_shape[1]) * int(self.upsample_factor[1]),
                )
                up_aux = {}
            _append_probe_stage(probe_stages, "vit_attention_out", feat_hr, hr_shape)
        elif self.feature_upsampler_type == "rr_ctp":
            if self.rr_ctp_upsampler is None:
                raise RuntimeError("rr_ctp_upsampler is not initialized.")
            upsampled = self.rr_ctp_upsampler(
                feat_lr=feat,
                lr_shape=lr_shape,
                return_aux=need_up_aux,
            )
            if need_up_aux:
                feat_hr, up_aux = upsampled
                hr_shape = tuple(up_aux["hr_shape"])
                _append_probe_stage(probe_stages, "query_seed_hr", up_aux["feat_query_seed_hr"], hr_shape)
                _append_probe_stage(probe_stages, "rr_ctp_context_hr", up_aux["feat_rr_ctp_context_hr"], hr_shape)
                _append_probe_stage(probe_stages, "rr_ctp_out_hr", up_aux["feat_rr_ctp_out_hr"], hr_shape)
            else:
                feat_hr = upsampled
                hr_shape = (
                    int(lr_shape[0]) * int(self.upsample_factor[0]),
                    int(lr_shape[1]) * int(self.upsample_factor[1]),
                )
                up_aux = {}
            _append_probe_stage(probe_stages, "rr_ctp_upsampler_out", feat_hr, hr_shape)
        else:
            upsampled = self._bilinear_feature_upsample(
                feat_lr=feat,
                lr_shape=lr_shape,
                return_aux=need_up_aux,
            )
            if need_up_aux:
                feat_hr, up_aux = upsampled
                hr_shape = tuple(up_aux["hr_shape"])
            else:
                feat_hr = upsampled
                hr_shape = (
                    int(lr_shape[0]) * int(self.upsample_factor[0]),
                    int(lr_shape[1]) * int(self.upsample_factor[1]),
                )
                up_aux = {}
            _append_probe_stage(probe_stages, "bilinear_out", feat_hr, hr_shape)

        if self.conv_hr1 is not None:
            feat_hr = self.conv_hr1(feat_hr, hr_shape)
            _append_probe_stage(probe_stages, "hr_conv_1", feat_hr, hr_shape)

        _append_probe_stage(probe_stages, "final_feat_before_decode", feat_hr, hr_shape)

        if not return_aux:
            return feat_hr, hr_shape

        aux: dict[str, object] = {
            **up_aux,
            "feature_upsampler_type": self.feature_upsampler_type,
            "feat_lr_a1_post_lr": feat,
            "feat_hr_a1": feat_hr,
            "boundary_logits_hr_refined": None,
            "boundary_logits_hr": None,
            "boundary_logits_lr": None,
        }
        if probe_stages is not None:
            aux["probe_stages"] = probe_stages
        return feat_hr, hr_shape, aux

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
            return self.decode(feat_hr_a1), aux

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
            b = lr_quats.shape[0]
            lr_flat = lr_quats.reshape(-1, 4)
            hr_flat = hr_quats.reshape(-1, 4)
        else:
            b = 1
            lr_flat = lr_quats
            hr_flat = hr_quats

        if normalize_input:
            lr_flat = _normalize_quaternions(lr_flat)
            hr_flat = _normalize_quaternions(hr_flat)

        with torch.no_grad():
            feat_lr_a1_flat = self.encode_a1(lr_flat).detach()
            feat_hr_tgt_flat = self.encode_a1(hr_flat).detach()

        if batched:
            feat_lr_a1 = feat_lr_a1_flat.reshape(b, -1, feat_lr_a1_flat.shape[-1])
            feat_hr_tgt = feat_hr_tgt_flat.reshape(b, -1, feat_hr_tgt_flat.shape[-1])
        else:
            feat_lr_a1 = feat_lr_a1_flat
            feat_hr_tgt = feat_hr_tgt_flat

        h_hr = int(lr_shape[0]) * int(self.upsample_factor[0])
        w_hr = int(lr_shape[1]) * int(self.upsample_factor[1])
        expected_hr_n = int(h_hr * w_hr)
        actual_hr_n = int(hr_quats.shape[1] if batched else hr_quats.shape[0])
        if actual_hr_n != expected_hr_n:
            raise ValueError(
                f"Expected HR quaternions with N={expected_hr_n} from lr_shape={lr_shape} "
                f"and scale={self.upsample_factor}, got N={actual_hr_n}"
            )

        # Training/validation loss does not require attention diagnostics.
        # Avoiding return_aux here prevents large per-query attention-stat tensors
        # from being materialized, which significantly reduces VRAM usage.
        feat_hr, _ = self._forward_sr_features(
            feat_lr_a1=feat_lr_a1,
            lr_shape=lr_shape,
            lr_boundary_map=lr_boundary_map,
            lr_quats=lr_quats,
            extra_material_stats=extra_material_stats,
            return_aux=False,
        )
        loss_feat = F.mse_loss(feat_hr, feat_hr_tgt)
        total = self.lambda_feat * loss_feat

        info: dict[str, torch.Tensor] = {
            "loss_feat": loss_feat.detach(),
            "loss_boundary": total.new_zeros(()),
            "loss_side_correct": total.new_zeros(()),
            "loss_side_entropy": total.new_zeros(()),
            "loss_total": total.detach(),
        }
        return (total, info) if return_info else total

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
        """Compatibility no-op: no explicit boundary head in this variant."""
        del aux, lr_quats, lr_shape, thr_deg, connectivity, lambda_hr, lambda_lr, use_refined_hr, use_focal, focal_gamma

        b = int(hr_quats.shape[0]) if hr_quats.dim() == 3 else 1
        gb_hr = torch.zeros((b, 1, hr_shape[0], hr_shape[1]), device=self.device, dtype=torch.float32)
        zero = gb_hr.sum() * 0.0
        info: dict[str, torch.Tensor] = {
            "gb_target_hr": gb_hr.detach(),
            "loss_boundary_hr": zero.detach(),
            "loss_boundary_total": zero.detach(),
        }
        return zero, info

    def forward(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
        lr_boundary_map: torch.Tensor | None = None,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        quats = quats.to(self.device)
        if quats.dim() != 2 or quats.shape[-1] != 4:
            raise ValueError(f"IsoEmbeddingSRAttn expects (N,4), got {tuple(quats.shape)}")
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

# Backward-compatible alias expected by some runtime helpers.
IsoEmbeddingSRSDF = IsoEmbeddingSRAttn

# ============================================================
# Minimal feature-space training helpers
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
    """Compatibility helper retained for shared training utilities."""
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
    """Compatibility helper retained for shared training utilities."""
    d_plus = ((feat_plus - feat_target) ** 2).mean(dim=1, keepdim=True)
    d_minus = ((feat_minus - feat_target) ** 2).mean(dim=1, keepdim=True)
    rel_gap = (d_plus - d_minus).abs() / (d_plus + d_minus).clamp_min(eps)
    conf_mask = (rel_gap > float(rel_gap_threshold)).to(dtype=d_plus.dtype)
    mask = teacher_band.to(dtype=d_plus.dtype) * conf_mask

    target_side = (d_minus < d_plus).long().squeeze(1)
    ce = F.cross_entropy(side_logits, target_side, reduction="none").unsqueeze(1)
    return _masked_mean(ce, mask)


def train_step_feature_space(
    model: IsoEmbeddingSRAttn,
    batch: dict[str, torch.Tensor | tuple[int, int]],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Train in A1 feature space. Boundary and side losses are no-ops in this variant."""
    model.train()

    lr_quats = batch["lr_quats"].to(model.device)
    hr_quats = batch["hr_quats"].to(model.device)
    lr_shape = tuple(batch["lr_shape"])
    lr_boundary_map = batch.get("lr_boundary_map", None)
    if lr_boundary_map is not None:
        lr_boundary_map = lr_boundary_map.to(model.device)
    extra_stats = batch.get("stats", None)
    if extra_stats is not None:
        extra_stats = extra_stats.to(model.device)

    with torch.no_grad():
        feat_lr = model.encode_a1(lr_quats)
        feat_hr_tgt = model.encode_a1(hr_quats)

    feat_hr_pred, hr_shape, aux = model._forward_sr_features(
        feat_lr_a1=feat_lr,
        lr_shape=lr_shape,
        lr_boundary_map=lr_boundary_map,
        lr_quats=lr_quats,
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

    loss_side_correct = loss_feat.new_zeros(())
    loss_side = loss_feat.new_zeros(())
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
    model: IsoEmbeddingSRAttn,
    batch: dict[str, torch.Tensor | tuple[int, int]],
    cfg: TrainConfig,
) -> dict[str, float]:
    model.eval()

    lr_quats = batch["lr_quats"].to(model.device)
    hr_quats = batch["hr_quats"].to(model.device)
    lr_shape = tuple(batch["lr_shape"])
    lr_boundary_map = batch.get("lr_boundary_map", None)
    if lr_boundary_map is not None:
        lr_boundary_map = lr_boundary_map.to(model.device)
    extra_stats = batch.get("stats", None)
    if extra_stats is not None:
        extra_stats = extra_stats.to(model.device)

    feat_lr = model.encode_a1(lr_quats)
    feat_hr_tgt = model.encode_a1(hr_quats)
    feat_hr_pred, hr_shape, aux = model._forward_sr_features(
        feat_lr_a1=feat_lr,
        lr_shape=lr_shape,
        lr_boundary_map=lr_boundary_map,
        lr_quats=lr_quats,
        extra_material_stats=extra_stats,
        return_aux=True,
    )
    loss_feat = F.mse_loss(feat_hr_pred, feat_hr_tgt)

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

    loss_side_correct = loss_feat.new_zeros(())
    loss_side_entropy = loss_feat.new_zeros(())
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
    "MultiPathCrossAttentionLayer",
    "LRTokenSelfAttentionBlock",
    "ViTPixelQueryPatchKVAupsampler",
    "IsoEmbeddingSRAttn",
    "IsoEmbeddingSRSDF",
    "TrainConfig",
    "side_entropy_loss",
    "side_correctness_loss",
    "train_step_feature_space",
    "validate_batch_feature_space",
]
