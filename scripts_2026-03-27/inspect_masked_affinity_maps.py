#!/usr/bin/env python
"""Inspect and visualize strict masked affinities for one sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SR_double_conv_SRattn_a1_masked import _local_irrep_affinity
from visualization.model_stage_walkthrough import load_lr_input, load_model_from_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load one model/sample and inspect affinity masks for masked conv/upsample/"
            "attention stages."
        )
    )
    p.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    p.add_argument("--config", type=str, default="config.json", help="Config file in exp_dir.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint in exp_dir/checkpoints or absolute path.",
    )
    p.add_argument("--device", type=str, default=None, help="Device override.")
    p.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"])
    p.add_argument("--sample_offset", type=int, default=0)
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--lr_npy", type=str, default=None)
    p.add_argument(
        "--crop_hw",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Optional top-left crop in LR coordinates.",
    )
    p.add_argument(
        "--query_yx",
        nargs=2,
        type=int,
        default=None,
        metavar=("Y", "X"),
        help="Optional LR query pixel. Defaults to image center.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <exp_dir>/affinity_inspect/<sample_label>",
    )
    return p.parse_args()


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _flat_to_bchw(feat_flat: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    h, w = shape
    return feat_flat.reshape(1, h, w, feat_flat.shape[-1]).permute(0, 3, 1, 2)


def _compute_local_affinity_from_feat_img(
    feat_img: torch.Tensor,
    module,
    *,
    hard_threshold_override: float | None = None,
) -> torch.Tensor:
    """Compute (H, W, k, k) local affinity map for a masked conv-like module."""
    k = int(module.kernel_size)
    pad = int(module.padding)
    eps = float(getattr(module, "mask_eps", 1e-6))
    tau = float(getattr(module, "mask_tau", 0.6))
    mask_min = float(getattr(module, "mask_min", 0.0))
    strength = float(getattr(module, "mask_strength", 1.0))
    hard = float(getattr(module, "hard_threshold", 0.0))
    if hard_threshold_override is not None:
        hard = float(hard_threshold_override)

    B, C, H, W = feat_img.shape
    if B != 1:
        raise ValueError(f"Expected B=1 for inspection, got {B}")

    feat_padded = F.pad(feat_img, (pad, pad, pad, pad), mode="replicate")
    patches = feat_padded.unfold(2, k, 1).unfold(3, k, 1)
    k2 = k * k
    neigh = patches.permute(0, 2, 3, 4, 5, 1).reshape(B, H, W, k2, C)
    center = feat_img.permute(0, 2, 3, 1).unsqueeze(3).expand(-1, -1, -1, k2, -1)

    aff = _local_irrep_affinity(
        center=center,
        neigh=neigh,
        irreps=module.irreps_in,
        tau=tau,
        mask_min=mask_min,
        mask_strength=strength,
        eps=eps,
        hard_threshold=hard,
    )
    return aff.reshape(H, W, k, k).detach().cpu()


def _upsample_pre_context_features(feat_lr2: torch.Tensor, lr_shape: tuple[int, int], up_module) -> tuple[torch.Tensor, tuple[int, int]]:
    """Recreate the upsampled HR feature map before masked neighborhood context."""
    h, w = lr_shape
    r = int(up_module.upsample_factor)
    hr_shape = (h * r, w * r)
    feat_img = _flat_to_bchw(feat_lr2, lr_shape)
    c = feat_img.shape[1]
    up_weight = up_module._expanded_transpose_weight()
    feat_hr = torch.nn.functional.conv_transpose2d(
        feat_img,
        up_weight,
        bias=None,
        stride=r,
        padding=int(up_module.transpose_padding),
        output_padding=0,
        groups=c,
    )[:, :, : hr_shape[0], : hr_shape[1]]
    return feat_hr, hr_shape


def _stage_stats(aff: torch.Tensor, stage_name: str) -> str:
    """Return a compact text summary for one affinity tensor (H,W,k,k)."""
    k = aff.shape[-1]
    keep = (aff > 0.0).float()
    keep_ratio = float(keep.mean().item())

    ctr = k // 2
    no_ctr = keep.clone()
    no_ctr[..., ctr, ctr] = 0.0
    keep_ratio_no_center = float(
        (no_ctr.sum(dim=(-1, -2)) / float(k * k - 1)).mean().item()
    )
    return (
        f"{stage_name}: mean={float(aff.mean().item()):.4f}, "
        f"std={float(aff.std(unbiased=False).item()):.4f}, "
        f"keep_ratio={keep_ratio:.4f}, keep_ratio_no_center={keep_ratio_no_center:.4f}"
    )


def _plot_local_affinity(
    out_dir: Path,
    stage_name: str,
    aff_soft: torch.Tensor,
    aff_hard: torch.Tensor,
    query_y: int,
    query_x: int,
) -> None:
    """Save local affinity diagnostics for one stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W, k, _ = aff_hard.shape
    qy = max(0, min(query_y, H - 1))
    qx = max(0, min(query_x, W - 1))
    ctr = k // 2

    keep = (aff_hard > 0.0).float()
    keep_ratio_offset = keep.mean(dim=(0, 1))
    mean_soft_offset = aff_soft.mean(dim=(0, 1))
    mean_hard_offset = aff_hard.mean(dim=(0, 1))

    mean_aff_spatial = aff_hard.mean(dim=(2, 3))
    keep_ratio_spatial = keep.mean(dim=(2, 3))

    q_soft = aff_soft[qy, qx]
    q_hard = aff_hard[qy, qx]
    q_keep = (q_hard > 0.0).float()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    im = axes[0, 0].imshow(mean_soft_offset.numpy(), cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(f"{stage_name}: mean soft affinity (k={k})")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.02)

    im = axes[0, 1].imshow(mean_hard_offset.numpy(), cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("mean hard affinity")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.02)

    im = axes[0, 2].imshow(keep_ratio_offset.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("keep ratio per offset")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.02)

    im = axes[0, 3].imshow(q_hard.numpy(), cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 3].set_title(f"query hard affinity at (y={qy}, x={qx})")
    axes[0, 3].scatter([ctr], [ctr], c="cyan", s=30)
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.02)

    im = axes[1, 0].imshow(mean_aff_spatial.numpy(), cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("mean affinity per pixel")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.02)

    im = axes[1, 1].imshow(keep_ratio_spatial.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("keep ratio per pixel")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.02)

    im = axes[1, 2].imshow(q_keep.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("query keep mask (hard)")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.02)

    axes[1, 3].hist(aff_soft.reshape(-1).numpy(), bins=60, alpha=0.7, label="soft")
    axes[1, 3].hist(aff_hard.reshape(-1).numpy(), bins=60, alpha=0.7, label="hard")
    axes[1, 3].set_title("affinity histogram")
    axes[1, 3].legend(loc="upper right")

    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_dir / f"{_safe(stage_name)}_affinity_diagnostics.png", dpi=180)
    plt.close(fig)

    np.save(out_dir / f"{_safe(stage_name)}_affinity_soft.npy", aff_soft.numpy())
    np.save(out_dir / f"{_safe(stage_name)}_affinity_hard.npy", aff_hard.numpy())


def _attention_debug(
    model,
    feat_hr1: torch.Tensor,
    hr_shape: tuple[int, int],
    query_hr_yx: tuple[int, int],
) -> dict[str, torch.Tensor] | None:
    """Build debug tensors for the first attention block."""
    if not bool(getattr(model, "use_attention", False)):
        return None
    if len(getattr(model, "attention_blocks", [])) == 0:
        return None

    block = model.attention_blocks[0]
    if feat_hr1.dim() == 2:
        feat_hr1 = feat_hr1.unsqueeze(0)
    B, N, C = feat_hr1.shape
    if B != 1:
        raise ValueError(f"Expected B=1 for inspection, got {B}")
    Hr, Wr = hr_shape
    if N != Hr * Wr:
        raise ValueError(f"Expected N={Hr*Wr}, got {N}")

    block_h = min(int(model.hr_attn_block_size), Hr)
    block_w = min(int(model.hr_attn_block_size), Wr)
    pad_h = (-Hr) % block_h
    pad_w = (-Wr) % block_w
    Hr_pad, Wr_pad = Hr + pad_h, Wr + pad_w

    feat = feat_hr1
    if pad_h > 0 or pad_w > 0:
        feat_2d = feat.reshape(B, Hr, Wr, C).permute(0, 3, 1, 2)
        feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
        feat = feat_2d.permute(0, 2, 3, 1).reshape(B, Hr_pad * Wr_pad, C)

    num_bh = Hr_pad // block_h
    num_bw = Wr_pad // block_w
    Nb = block_h * block_w

    feat_blocks = (
        feat.reshape(B, num_bh, block_h, num_bw, block_w, C)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(B * num_bh * num_bw, Nb, C)
    )
    d_block = model._get_hr_sh_block(block_h, block_w, feat.device, feat.dtype)

    scores = torch.exp(block.log_s) * block._invariant_scores(feat_blocks)
    pb = block.pos_bias(d_block.unsqueeze(-1)).squeeze(-1)
    scores = scores + pb.unsqueeze(0)

    mask = None
    if hasattr(block, "_irrep_affinity_mask"):
        mask = block._irrep_affinity_mask(feat_blocks, d_block)
        valid = mask > 0.0
        scores = scores.masked_fill(~valid, -1.0e4)
        if hasattr(block, "mask_strength"):
            mask_safe = torch.where(valid, mask.clamp_min(1e-12), torch.ones_like(mask))
            scores = scores + float(block.mask_strength) * torch.log(mask_safe)

    attn = torch.softmax(scores.float(), dim=-1)

    qy, qx = query_hr_yx
    qy = max(0, min(qy, Hr - 1))
    qx = max(0, min(qx, Wr - 1))
    bh = qy // block_h
    bw = qx // block_w
    bb_idx = bh * num_bw + bw
    local_y = qy % block_h
    local_x = qx % block_w
    q_idx = local_y * block_w + local_x

    out = {
        "d_block": d_block.detach().cpu(),
        "scores_block": scores[bb_idx].detach().cpu(),
        "attn_block": attn[bb_idx].detach().cpu(),
        "query_attn_map": attn[bb_idx, q_idx].reshape(block_h, block_w).detach().cpu(),
        "query_idx": torch.tensor([q_idx], dtype=torch.int64),
        "block_h": torch.tensor([block_h], dtype=torch.int64),
        "block_w": torch.tensor([block_w], dtype=torch.int64),
        "bb_idx": torch.tensor([bb_idx], dtype=torch.int64),
    }
    if mask is not None:
        out["mask_block"] = mask[bb_idx].detach().cpu()
        out["query_mask_map"] = mask[bb_idx, q_idx].reshape(block_h, block_w).detach().cpu()
    return out


def _plot_attention_debug(out_dir: Path, dbg: dict[str, torch.Tensor]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    block_h = int(dbg["block_h"].item())
    block_w = int(dbg["block_w"].item())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    q_idx = int(dbg["query_idx"].item())
    im = axes[0, 0].imshow(dbg["d_block"][q_idx].reshape(block_h, block_w).numpy(), cmap="magma")
    axes[0, 0].set_title("distance map from query (within block)")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.02)

    if "query_mask_map" in dbg:
        im = axes[0, 1].imshow(dbg["query_mask_map"].numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[0, 1].set_title("query affinity mask")
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.02)
    else:
        axes[0, 1].axis("off")

    im = axes[0, 2].imshow(dbg["query_attn_map"].numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("query attention weights")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.02)

    if "mask_block" in dbg:
        im = axes[1, 0].imshow(dbg["mask_block"].numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, 0].set_title("pair mask matrix")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.02)
    else:
        axes[1, 0].axis("off")

    im = axes[1, 1].imshow(dbg["attn_block"].numpy(), cmap="viridis")
    axes[1, 1].set_title("attention matrix")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.02)

    axes[1, 2].hist(dbg["attn_block"].reshape(-1).numpy(), bins=60, alpha=0.8)
    axes[1, 2].set_title("attention weights histogram")

    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_dir / "attention_affinity_diagnostics.png", dpi=180)
    plt.close(fig)

    for k, v in dbg.items():
        np.save(out_dir / f"attention_{k}.npy", v.numpy())


def main() -> None:
    args = parse_args()
    model, cfg, checkpoint_path = load_model_from_experiment(
        Path(args.exp_dir),
        config_name=args.config,
        checkpoint_name=args.checkpoint,
        device=args.device,
    )
    lr_arr, label = load_lr_input(
        cfg,
        split=args.split,
        sample_offset=int(args.sample_offset),
        dataset_root=args.dataset_root,
        lr_npy=args.lr_npy,
        crop_hw=None if args.crop_hw is None else (int(args.crop_hw[0]), int(args.crop_hw[1])),
    )

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else (Path(args.exp_dir).resolve() / "affinity_inspect" / label)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W, _ = lr_arr.shape
    if args.query_yx is None:
        qy_lr, qx_lr = H // 2, W // 2
    else:
        qy_lr = int(args.query_yx[0])
        qx_lr = int(args.query_yx[1])
    qy_lr = max(0, min(qy_lr, H - 1))
    qx_lr = max(0, min(qx_lr, W - 1))

    lr_quats = torch.from_numpy(lr_arr.reshape(H * W, 4)).to(model.device, dtype=torch.float32)
    lr_shape = (H, W)

    print(f"Using checkpoint: {checkpoint_path if checkpoint_path is not None else '[random init]'}")
    print(f"Sample: {label}")
    print(f"LR shape: {lr_shape}")
    print(f"LR query pixel: (y={qy_lr}, x={qx_lr})")

    with torch.no_grad():
        feat_a1 = model.encode_a1(lr_quats)
        feat_lr1 = model.conv_lr1(feat_a1, lr_shape) if bool(getattr(model, "use_lr_conv1", True)) else feat_a1
        feat_lr2 = model.conv_lr2(feat_lr1, lr_shape) if bool(getattr(model, "use_lr_conv2", True)) else feat_lr1
        feat_up, hr_shape = model.upsample_conv(feat_lr2, lr_shape)
        feat_hr1 = model.conv_hr1(feat_up, hr_shape)

    # LR conv1 affinity (input to conv1 is feat_a1)
    if hasattr(model.conv_lr1, "hard_threshold"):
        aff_lr1_soft = _compute_local_affinity_from_feat_img(
            _flat_to_bchw(feat_a1, lr_shape), model.conv_lr1, hard_threshold_override=0.0
        )
        aff_lr1_hard = _compute_local_affinity_from_feat_img(
            _flat_to_bchw(feat_a1, lr_shape), model.conv_lr1, hard_threshold_override=None
        )
        print(_stage_stats(aff_lr1_hard, "conv_lr1"))
        _plot_local_affinity(out_dir, "conv_lr1", aff_lr1_soft, aff_lr1_hard, qy_lr, qx_lr)

    # LR conv2 affinity (input to conv2 is feat_lr1)
    if hasattr(model.conv_lr2, "hard_threshold"):
        aff_lr2_soft = _compute_local_affinity_from_feat_img(
            _flat_to_bchw(feat_lr1, lr_shape), model.conv_lr2, hard_threshold_override=0.0
        )
        aff_lr2_hard = _compute_local_affinity_from_feat_img(
            _flat_to_bchw(feat_lr1, lr_shape), model.conv_lr2, hard_threshold_override=None
        )
        print(_stage_stats(aff_lr2_hard, "conv_lr2"))
        _plot_local_affinity(out_dir, "conv_lr2", aff_lr2_soft, aff_lr2_hard, qy_lr, qx_lr)

    # Upsample affinity (input is HR pre-context features right after transpose-conv)
    if hasattr(model.upsample_conv, "hard_threshold"):
        feat_hr_pre, hr_shape2 = _upsample_pre_context_features(feat_lr2, lr_shape, model.upsample_conv)
        if hr_shape2 != hr_shape:
            raise RuntimeError(f"HR shape mismatch: {hr_shape2} vs {hr_shape}")
        qy_hr = min(hr_shape[0] - 1, qy_lr * int(model.upsample_factor))
        qx_hr = min(hr_shape[1] - 1, qx_lr * int(model.upsample_factor))

        aff_up_soft = _compute_local_affinity_from_feat_img(
            feat_hr_pre, model.upsample_conv, hard_threshold_override=0.0
        )
        aff_up_hard = _compute_local_affinity_from_feat_img(
            feat_hr_pre, model.upsample_conv, hard_threshold_override=None
        )
        print(_stage_stats(aff_up_hard, "upsample"))
        _plot_local_affinity(out_dir, "upsample", aff_up_soft, aff_up_hard, qy_hr, qx_hr)
    else:
        qy_hr = min(hr_shape[0] - 1, qy_lr * int(model.upsample_factor))
        qx_hr = min(hr_shape[1] - 1, qx_lr * int(model.upsample_factor))

    # HR conv1 affinity (input to conv_hr1 is feat_up)
    if hasattr(model.conv_hr1, "hard_threshold"):
        aff_hr1_soft = _compute_local_affinity_from_feat_img(
            _flat_to_bchw(feat_up, hr_shape), model.conv_hr1, hard_threshold_override=0.0
        )
        aff_hr1_hard = _compute_local_affinity_from_feat_img(
            _flat_to_bchw(feat_up, hr_shape), model.conv_hr1, hard_threshold_override=None
        )
        print(_stage_stats(aff_hr1_hard, "conv_hr1"))
        _plot_local_affinity(out_dir, "conv_hr1", aff_hr1_soft, aff_hr1_hard, qy_hr, qx_hr)

    # Attention affinity diagnostics (first attention block only).
    attn_dbg = _attention_debug(model, feat_hr1, hr_shape, (qy_hr, qx_hr))
    if attn_dbg is not None:
        mask = attn_dbg.get("mask_block", None)
        if mask is not None:
            keep_ratio = float((mask > 0).float().mean().item())
            print(f"attention_block0: pair keep_ratio={keep_ratio:.4f}")
        _plot_attention_debug(out_dir, attn_dbg)
    else:
        print("attention_block0: skipped (attention disabled or no blocks)")

    print(f"Saved diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
