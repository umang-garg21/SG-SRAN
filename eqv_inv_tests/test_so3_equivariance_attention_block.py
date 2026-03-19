"""
test_so3_equivariance_attention_block.py
==========================================
Validates SO(3) equivariance of AttentionBlock AND invariance of attention weights.

Two properties tested
---------------------
1. Equivariance of the block output:
       block( D·x ) + x  ≈  D · (block( x ) + x)
   i.e. delta( D·x ) ≈ D · delta( x )
   where delta is the residual output of the block.

2. Invariance of attention weights:
   Attention scores  s_ij = exp(log_s) · (f̂_i · f̂_j)  are SO(3)-invariant because
   D is orthogonal ⟹ (D·f̂_i)·(D·f̂_j) = f̂_i·f̂_j.
   We verify this numerically: attn_weights( D·x ) ≈ attn_weights( x ).

Run:  python eqv_inv_tests/test_so3_equivariance_attention_block.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import Irreps

from _helpers import rotate_features, rel_error, report, section, summary
from models.SR_double_conv_SRattn_a1 import AttentionBlock, LocalIsoCrystalEncoder

# ── config ────────────────────────────────────────────────────────────────────
# H and W must be multiples of block_size
BLOCK_SIZE      = 4
H, W            = 8, 8          # 2×2 blocks of size 4×4
N_TRIALS        = 12
TOL_EQUIV       = 1e-4
TOL_ATTN_INV    = 1e-5          # attention weights should be very close
DEVICE          = torch.device("cpu")
SEED            = 42


def _build_d_block(block_h: int, block_w: int, device: torch.device) -> torch.Tensor:
    """Pairwise L2 distance matrix for a block (Nb, Nb), Nb = block_h * block_w."""
    ys = torch.linspace(-1.0, 1.0, block_h, device=device)
    xs = torch.linspace(-1.0, 1.0, block_w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    return torch.cdist(coords, coords, p=2)


def _compute_attention_weights(
    block: AttentionBlock,
    feat: torch.Tensor,
    H: int, W: int,
    block_h: int, block_w: int,
    d_block: torch.Tensor,
) -> torch.Tensor:
    """
    Manually reproduce the attention weight matrix from AttentionBlock.forward.
    Returns attn of shape (Bb, Nb, Nb) where Bb = B * num_blocks.
    """
    B, _, C = feat.shape
    num_bh = H // block_h
    num_bw = W // block_w
    Nb = block_h * block_w
    Bb = B * num_bh * num_bw

    feat_blocks = (
        feat.reshape(B, num_bh, block_h, num_bw, block_w, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bb, Nb, C)
    )
    f_n    = F.normalize(feat_blocks, dim=-1)
    scores = torch.exp(block.log_s) * torch.bmm(f_n, f_n.transpose(-2, -1))
    pb     = block.pos_bias(d_block.unsqueeze(-1)).squeeze(-1)
    scores = scores + pb.unsqueeze(0)
    return torch.softmax(scores.float(), dim=-1)


def _run_equivariance(
    block: AttentionBlock,
    x: torch.Tensor,
    irreps: Irreps,
    d_block: torch.Tensor,
    n_trials: int,
    seed: int,
) -> tuple[float, float]:
    """
    Tests: delta( D·x ) ≈ D · delta( x )
    where delta = block.forward(feat, d_block, H, W, block_h, block_w).
    Returns (mean_rel_err, max_rel_err).
    """
    block_h = block_w = BLOCK_SIZE
    with torch.no_grad():
        delta_ref = block(x, d_block, H, W, block_h, block_w)

    errors = []
    torch.manual_seed(seed)
    for _ in range(n_trials):
        R   = o3.rand_matrix(dtype=torch.float32).to(DEVICE)
        D   = irreps.D_from_matrix(R).to(DEVICE)

        x_rot = rotate_features(x, D)
        with torch.no_grad():
            delta_in_rot  = block(x_rot, d_block, H, W, block_h, block_w)
            delta_out_rot = rotate_features(delta_ref, D)

        errors.append(rel_error(delta_in_rot, delta_out_rot))

    t = torch.tensor(errors)
    return float(t.mean()), float(t.max())


def _run_attn_invariance(
    block: AttentionBlock,
    x: torch.Tensor,
    irreps: Irreps,
    d_block: torch.Tensor,
    n_trials: int,
    seed: int,
) -> tuple[float, float]:
    """
    Tests: attn_weights( D·x ) ≈ attn_weights( x ).
    Invariance follows from D being orthogonal: (Df̂_i)·(Df̂_j) = f̂_i·f̂_j.
    Returns (mean_rel_err, max_rel_err).
    """
    block_h = block_w = BLOCK_SIZE
    with torch.no_grad():
        attn_ref = _compute_attention_weights(block, x, H, W, block_h, block_w, d_block)

    errors = []
    torch.manual_seed(seed + 500)
    for _ in range(n_trials):
        R     = o3.rand_matrix(dtype=torch.float32).to(DEVICE)
        D     = irreps.D_from_matrix(R).to(DEVICE)
        x_rot = rotate_features(x, D)
        with torch.no_grad():
            attn_rot = _compute_attention_weights(block, x_rot, H, W, block_h, block_w, d_block)

        errors.append(rel_error(attn_rot, attn_ref))

    t = torch.tensor(errors)
    return float(t.mean()), float(t.max())


def main() -> None:
    results: list[bool] = []
    d_block = _build_d_block(BLOCK_SIZE, BLOCK_SIZE, DEVICE)

    for crystal in ("fcc", "hcp"):
        enc    = LocalIsoCrystalEncoder(crystal=crystal, dtype=torch.float32, device=DEVICE).eval()
        irr_a1 = enc.irreps_a1

        section(f"Crystal={crystal.upper()}  irreps_a1={irr_a1}  block_size={BLOCK_SIZE}x{BLOCK_SIZE}")

        for num_channels in (4, 8):
            torch.manual_seed(SEED)
            block = AttentionBlock(
                irreps_feat=irr_a1,
                num_channels=num_channels,
            ).to(DEVICE).eval()

            torch.manual_seed(SEED + 99)
            x = torch.randn(1, H * W, irr_a1.dim, dtype=torch.float32, device=DEVICE)

            # 1. Block output equivariance
            mean_eq, max_eq = _run_equivariance(block, x, irr_a1, d_block, N_TRIALS, SEED)
            ok = report(
                f"{crystal.upper()} AttentionBlock(ch={num_channels}) delta-equivariance",
                max_eq, TOL_EQUIV,
                extra=f"mean={mean_eq:.2e}",
            )
            results.append(ok)

            # 2. Attention weight invariance
            mean_ai, max_ai = _run_attn_invariance(block, x, irr_a1, d_block, N_TRIALS, SEED)
            ok2 = report(
                f"{crystal.upper()} AttentionBlock(ch={num_channels}) attn-weight-invariance",
                max_ai, TOL_ATTN_INV,
                extra=f"mean={mean_ai:.2e}",
            )
            results.append(ok2)

    summary(results)


if __name__ == "__main__":
    main()
