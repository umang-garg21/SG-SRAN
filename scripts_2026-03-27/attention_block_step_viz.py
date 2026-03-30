#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Allow running directly: `python scripts/attention_block_step_viz.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SR_double_conv_SRattn_a1 import AttentionBlock


@dataclass
class TraceResult:
    feat_blocks: torch.Tensor
    dot_scores: torch.Tensor
    pb: torch.Tensor
    scores: torch.Tensor
    attn: torch.Tensor
    h: torch.Tensor
    ctx: torch.Tensor
    h_out: torch.Tensor
    delta_blocks: torch.Tensor
    delta: torch.Tensor
    delta_ref: torch.Tensor
    max_abs_diff: float
    B: int
    H: int
    W: int
    C_feat: int
    block_h: int
    block_w: int
    num_bh: int
    num_bw: int


def get_block_distance_matrix(
    block_h: int,
    block_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, block_h, device=device)
    xs = torch.linspace(-1.0, 1.0, block_w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    return torch.cdist(coords, coords, p=2).to(dtype)


def block_index(
    sample_idx: int,
    block_row: int,
    block_col: int,
    num_bh: int,
    num_bw: int,
) -> int:
    return sample_idx * (num_bh * num_bw) + block_row * num_bw + block_col


def trace_attention_block(
    block: AttentionBlock,
    feat: torch.Tensor,
    d_block: torch.Tensor,
    H: int,
    W: int,
    block_h: int,
    block_w: int,
) -> TraceResult:
    with torch.no_grad():
        B, _, C_feat = feat.shape
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

        f_n = F.normalize(feat_blocks, dim=-1, eps=1e-6)
        dot_scores = torch.exp(block.log_s) * torch.bmm(f_n, f_n.transpose(-2, -1))
        pb = block.pos_bias(d_block.unsqueeze(-1)).squeeze(-1)
        scores = dot_scores + pb.unsqueeze(0)
        attn = torch.softmax(scores.float(), dim=-1).to(dtype)

        feat_flat = feat_blocks.reshape(Bb * Nb, C_feat)
        h = block.lin_in(feat_flat).reshape(Bb, Nb, -1)
        ctx = torch.bmm(attn, h)
        h_out = block.tp_out(
            h.reshape(Bb * Nb, -1),
            ctx.reshape(Bb * Nb, -1),
        ).reshape(Bb, Nb, -1)
        delta_blocks = block.lin_out(h_out.reshape(Bb * Nb, -1)).reshape(Bb, Nb, C_feat)

        delta = (
            delta_blocks.reshape(B, num_bh, num_bw, block_h, block_w, C_feat)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, H * W, C_feat)
        )

        delta_ref = block(feat, d_block, H, W, block_h, block_w)
        max_abs_diff = float((delta - delta_ref).abs().max().item())

        return TraceResult(
            feat_blocks=feat_blocks,
            dot_scores=dot_scores,
            pb=pb,
            scores=scores,
            attn=attn,
            h=h,
            ctx=ctx,
            h_out=h_out,
            delta_blocks=delta_blocks,
            delta=delta,
            delta_ref=delta_ref,
            max_abs_diff=max_abs_diff,
            B=B,
            H=H,
            W=W,
            C_feat=C_feat,
            block_h=block_h,
            block_w=block_w,
            num_bh=num_bh,
            num_bw=num_bw,
        )


def _imshow(ax, data: torch.Tensor, title: str, cmap: str = "viridis") -> None:
    im = ax.imshow(data.detach().cpu().numpy(), cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_trace(
    out_path: Path,
    trace: TraceResult,
    sample_idx: int,
    block_row: int,
    block_col: int,
) -> None:
    bb = block_index(sample_idx, block_row, block_col, trace.num_bh, trace.num_bw)
    Nb = trace.block_h * trace.block_w
    center_idx = (trace.block_h // 2) * trace.block_w + (trace.block_w // 2)
    corner_idx = 0

    fig1, axes = plt.subplots(2, 4, figsize=(18, 8))
    _imshow(axes[0, 0], trace.pb, "Position Bias Matrix pb(d)")
    _imshow(axes[0, 1], trace.dot_scores[bb], "Dot Scores (selected block)")
    _imshow(axes[0, 2], trace.scores[bb], "Scores = dot + pb")
    _imshow(axes[0, 3], trace.attn[bb], "Attention Matrix")

    _imshow(
        axes[1, 0],
        trace.pb[center_idx].reshape(trace.block_h, trace.block_w),
        f"pb map from center idx={center_idx}",
        cmap="magma",
    )
    _imshow(
        axes[1, 1],
        trace.pb[corner_idx].reshape(trace.block_h, trace.block_w),
        f"pb map from corner idx={corner_idx}",
        cmap="magma",
    )
    _imshow(
        axes[1, 2],
        trace.attn[bb, center_idx].reshape(trace.block_h, trace.block_w),
        f"attn map from center idx={center_idx}",
        cmap="plasma",
    )
    _imshow(
        axes[1, 3],
        trace.attn[bb, corner_idx].reshape(trace.block_h, trace.block_w),
        f"attn map from corner idx={corner_idx}",
        cmap="plasma",
    )

    fig1.suptitle("AttentionBlock internals: position bias, scores, attention", fontsize=14)
    fig1.tight_layout()

    feat_norm = trace.feat_blocks[bb].norm(dim=-1).reshape(trace.block_h, trace.block_w)
    h_norm = trace.h[bb].norm(dim=-1).reshape(trace.block_h, trace.block_w)
    ctx_norm = trace.ctx[bb].norm(dim=-1).reshape(trace.block_h, trace.block_w)
    delta_norm = trace.delta_blocks[bb].norm(dim=-1).reshape(trace.block_h, trace.block_w)

    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
    _imshow(axes2[0], feat_norm, "||feat_blocks||", cmap="cividis")
    _imshow(axes2[1], h_norm, "||h = lin_in(feat)||", cmap="cividis")
    _imshow(axes2[2], ctx_norm, "||ctx = attn @ h||", cmap="cividis")
    _imshow(axes2[3], delta_norm, "||delta_blocks||", cmap="coolwarm")
    fig2.suptitle("Selected block: feature/hidden/context/delta norms", fontsize=14)
    fig2.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out_path.with_name(out_path.stem + "_scores.png"), dpi=180)
    fig2.savefig(out_path.with_name(out_path.stem + "_norms.png"), dpi=180)
    plt.close(fig1)
    plt.close(fig2)

    print(f"Saved plot: {out_path.with_name(out_path.stem + '_scores.png')}")
    print(f"Saved plot: {out_path.with_name(out_path.stem + '_norms.png')}")
    print(f"Nb (pixels per block): {Nb}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-through and visualize AttentionBlock internals.")
    parser.add_argument("--irreps", type=str, default="1x4e + 1x6e")
    parser.add_argument("--num-channels", type=int, default=8)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=8, help="Padded grid height (multiple of block_h).")
    parser.add_argument("--W", type=int, default=8, help="Padded grid width (multiple of block_w).")
    parser.add_argument("--block-h", type=int, default=4)
    parser.add_argument("--block-w", type=int, default=4)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--block-row", type=int, default=0)
    parser.add_argument("--block-col", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out/attention_block_trace.png"),
        help="Base path. Script writes '<stem>_scores.png' and '<stem>_norms.png'.",
    )
    args = parser.parse_args()

    if args.H % args.block_h != 0 or args.W % args.block_w != 0:
        raise ValueError("H/W must be multiples of block_h/block_w for AttentionBlock.forward.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    block = AttentionBlock(
        irreps_feat=args.irreps,
        num_channels=args.num_channels,
    ).to(device)
    block.eval()

    C_feat = block.irreps_feat.dim
    feat = torch.randn(args.B, args.H * args.W, C_feat, device=device, dtype=torch.float32)
    d_block = get_block_distance_matrix(args.block_h, args.block_w, device=device, dtype=feat.dtype)

    print("=== Input summary ===")
    print(f"feat shape: {tuple(feat.shape)}")
    print(f"C_feat (from irreps): {C_feat}")
    print(f"d_block shape: {tuple(d_block.shape)}")
    print(f"log_s (global scalar): {float(block.log_s.item()):.6f}")
    print(f"num_bh={args.H // args.block_h}, num_bw={args.W // args.block_w}")

    trace = trace_attention_block(
        block=block,
        feat=feat,
        d_block=d_block,
        H=args.H,
        W=args.W,
        block_h=args.block_h,
        block_w=args.block_w,
    )

    print("\n=== Step shapes ===")
    print(f"feat_blocks:   {tuple(trace.feat_blocks.shape)}  # (Bb, Nb, C_feat)")
    print(f"dot_scores:    {tuple(trace.dot_scores.shape)}  # (Bb, Nb, Nb)")
    print(f"pb:            {tuple(trace.pb.shape)}  # (Nb, Nb)")
    print(f"scores:        {tuple(trace.scores.shape)}  # (Bb, Nb, Nb)")
    print(f"attn:          {tuple(trace.attn.shape)}  # (Bb, Nb, Nb)")
    print(f"h:             {tuple(trace.h.shape)}  # (Bb, Nb, C_h)")
    print(f"ctx:           {tuple(trace.ctx.shape)}  # (Bb, Nb, C_h)")
    print(f"h_out:         {tuple(trace.h_out.shape)}  # (Bb, Nb, C_h)")
    print(f"delta_blocks:  {tuple(trace.delta_blocks.shape)}  # (Bb, Nb, C_feat)")
    print(f"delta:         {tuple(trace.delta.shape)}  # (B, H*W, C_feat)")
    print(f"forward diff:  max|manual - block.forward| = {trace.max_abs_diff:.6e}")

    args.sample_idx = min(args.sample_idx, trace.B - 1)
    args.block_row = min(args.block_row, trace.num_bh - 1)
    args.block_col = min(args.block_col, trace.num_bw - 1)
    bb = block_index(args.sample_idx, args.block_row, args.block_col, trace.num_bh, trace.num_bw)
    print(
        f"\nSelected block for plots: sample={args.sample_idx}, "
        f"block_row={args.block_row}, block_col={args.block_col}, bb={bb}"
    )

    plot_trace(
        out_path=args.out,
        trace=trace,
        sample_idx=args.sample_idx,
        block_row=args.block_row,
        block_col=args.block_col,
    )


if __name__ == "__main__":
    main()
