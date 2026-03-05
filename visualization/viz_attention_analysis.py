#!/usr/bin/env python3
"""
viz_attention_analysis.py

Visualize attention maps and learned kernels of FCCAutoEncoderSRDoubleConvAttn.

Usage (from project root):
    conda run -n material python visualization/viz_attention_analysis.py \\
        --exp_dir experiments/IN718/LAE_sr_double_conv_attn_01 \\
        [--sample_idx 0] [--out_dir <path>] [--device cpu]

Outputs (saved to <out_dir>/):
    01_spatial_weights.png      – learned spatial aggregation kernels (all conv layers)
    02_transpose_kernels.png    – 22 depthwise upsampling kernels of EquivariantTransposeConv
    03_attn_params.png          – per-block: s4/s6 scales, pos_bias weights, lin_out norms
    04_attn_entropy.png         – attention entropy / mean-received / self-attn spatial maps
    05_attn_block_matrices.png  – selected (Nb×Nb) attention matrices from each HR attn block
    06_feature_norms.png        – ||f4||, ||f6|| norm maps at each pipeline stage
"""

import argparse
import json
import sys
import os
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.SR_double_conv_SRattn import FCCAutoEncoderSRDoubleConvAttn
from models.SR_grain_attn import LRBlockAttentionBlock


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze attention maps & kernels of FCCAutoEncoderSRDoubleConvAttn"
    )
    p.add_argument(
        "--exp_dir",
        default="experiments/IN718/LAE_sr_double_conv_attn_01",
        help="Experiment directory (absolute or relative to project root)",
    )
    p.add_argument(
        "--run_dir",
        default=None,
        help="Specific run subdirectory (default: latest timestamped dir)",
    )
    p.add_argument(
        "--ckpt",
        default="best_model.pt",
        help="Checkpoint filename inside <run_dir>/checkpoints/",
    )
    p.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Which sample from the dataset split to use for dynamic analysis",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <run_dir>/attention_analysis/)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cpu' or 'cuda' (default: auto-detect)",
    )
    p.add_argument(
        "--no_data",
        action="store_true",
        help="Skip data-dependent analysis (only generate static kernel/weight plots)",
    )
    p.add_argument(
        "--split",
        default="Test",
        help="Dataset split to draw the sample from: Train, Val, Test",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config & model helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(exp_dir: Path) -> dict:
    path = exp_dir / "config.json"
    with open(path) as f:
        return json.load(f)


def find_latest_run(exp_dir: Path) -> Path:
    candidates = sorted([
        d for d in exp_dir.iterdir()
        if d.is_dir() and (d / "checkpoints").is_dir()
    ])
    if not candidates:
        raise FileNotFoundError(
            f"No run subdirs with checkpoints/ found in {exp_dir}"
        )
    return candidates[-1]


def build_model(cfg: dict, device: torch.device) -> FCCAutoEncoderSRDoubleConvAttn:
    decoder_config = {
        "decoder_lookup_resolution":   int(cfg.get("decoder_lookup_resolution", 3)),
        "decoder_lookup_refine_steps": int(cfg.get("decoder_lookup_refine_steps", 0)),
        "decoder_w6":                  float(cfg.get("decoder_w6", 0.5)),
    }
    lr_shape_raw = cfg.get("lr_shape", None)
    lr_shape = tuple(int(x) for x in lr_shape_raw) if lr_shape_raw else None

    def _oi(key):
        v = cfg.get(key)
        return int(v) if v is not None else None

    return FCCAutoEncoderSRDoubleConvAttn(
        device=device,
        upsample_factor=int(cfg.get("scale", 4)),
        upsample_residual=bool(cfg.get("upsample_residual", False)),
        upsampler="conv",
        lr_shape=lr_shape,
        lr_conv_kernel_size=_oi("lr_conv_kernel_size"),
        lr_conv_kernel_size_1=_oi("lr_conv_kernel_size_1"),
        lr_conv_kernel_size_2=_oi("lr_conv_kernel_size_2"),
        decoder_backend=str(cfg.get("decoder_backend", "lookup")),
        decoder_config=decoder_config,
        num_hr_attn_blocks=int(cfg.get("hr_attn_num_blocks", 2)),
        hr_attn_num_channels=int(cfg.get("hr_attn_num_channels", 8)),
        hr_attn_block_size=int(cfg.get("hr_attn_block_size", 16)),
    ).to(device)


def load_checkpoint(
    model: FCCAutoEncoderSRDoubleConvAttn,
    ckpt_path: Path,
    device: torch.device,
) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    # Training wraps the core in TrainableFCCAutoEncoder → keys start with "core."
    if all(k.startswith("core.") for k in state):
        state = {k[5:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return int(ckpt.get("epoch", 0))


# ─────────────────────────────────────────────────────────────────────────────
# Attention capture (monkey-patch each LRBlockAttentionBlock)
# ─────────────────────────────────────────────────────────────────────────────

def _capturing_block_forward(self, feat, sh_block, H, W, block_h, block_w):
    """Replacement LRBlockAttentionBlock.forward that saves attention weights."""
    B, N, C22 = feat.shape
    num_bh = H // block_h
    num_bw = W // block_w
    Nb  = block_h * block_w
    Bb  = B * num_bh * num_bw
    dtype = feat.dtype

    feat_blocks = (
        feat.reshape(B, num_bh, block_h, num_bw, block_w, C22)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bb, Nb, C22)
    )

    s4 = torch.exp(self.log_s4)
    s6 = torch.exp(self.log_s6)
    f4_n = F.normalize(feat_blocks[..., :9],  dim=-1)
    f6_n = F.normalize(feat_blocks[..., 9:],  dim=-1)

    scores_content = (
        s4 * torch.bmm(f4_n, f4_n.transpose(-2, -1))
        + s6 * torch.bmm(f6_n, f6_n.transpose(-2, -1))
    )
    pb     = self.pos_bias(sh_block)
    scores = scores_content + (pb + pb.T).unsqueeze(0)
    attn   = torch.softmax(scores.float(), dim=-1).to(dtype)

    self._attn_store.append({
        "block_idx":      self._attn_idx,
        "attn":           attn.detach().cpu().float(),          # (Bb, Nb, Nb)
        "scores_content": scores_content.detach().cpu().float(),# (Bb, Nb, Nb)
        "num_bh":   num_bh,
        "num_bw":   num_bw,
        "block_h":  block_h,
        "block_w":  block_w,
    })

    # Call original forward to produce the actual delta output
    return self._orig_forward(feat, sh_block, H, W, block_h, block_w)


class AttentionCapture:
    """Context manager: patches hr_attn_blocks to capture attention tensors."""

    def __init__(self, model: FCCAutoEncoderSRDoubleConvAttn):
        self.model = model
        self.data: List[Dict] = []

    def __enter__(self):
        self.data.clear()
        for i, blk in enumerate(self.model.hr_attn_blocks):
            blk._attn_store    = self.data
            blk._attn_idx      = i
            blk._orig_forward  = blk.forward
            blk.forward        = types.MethodType(_capturing_block_forward, blk)
        return self

    def __exit__(self, *_):
        for blk in self.model.hr_attn_blocks:
            blk.forward = blk._orig_forward
            del blk._orig_forward, blk._attn_store, blk._attn_idx


# ─────────────────────────────────────────────────────────────────────────────
# Feature norm capture (forward hooks on submodules)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureNormCapture:
    """Registers forward hooks on key submodules to record ||f4||, ||f6|| maps."""

    def __init__(self, model: FCCAutoEncoderSRDoubleConvAttn):
        self.model   = model
        self.stages: Dict[str, Dict] = {}
        self._handles: List = []

    def __enter__(self):
        self.stages.clear()
        self._handles.clear()

        capture = self  # closure reference

        def make_hook(name, shape_from_output=False):
            def hook(module, inp, output):
                # Determine output features
                if not isinstance(output, tuple):
                    return
                f4_out = output[0]
                f6_out = output[1]

                # Determine spatial shape
                if shape_from_output and len(output) > 2:
                    H, W = output[2]          # upsample returns (f4, f6, hr_shape)
                elif len(inp) > 2 and isinstance(inp[2], (tuple, list)):
                    H, W = inp[2]             # EquivariantSpatialConv: inp=(f4, f6, shape)
                else:
                    return

                # Strip batch dim if batched (B, N, C)
                f4 = f4_out.detach().cpu()
                f6 = f6_out.detach().cpu()
                if f4.dim() == 3:
                    f4 = f4[0]
                    f6 = f6[0]

                capture.stages[name] = {
                    "f4_norm": f4.norm(dim=-1).reshape(H, W).numpy(),
                    "f6_norm": f6.norm(dim=-1).reshape(H, W).numpy(),
                    "H": H, "W": W,
                }

                # Also capture the INPUT to the first conv as "encoded LR"
                if name == "conv1 (LR k=5)" and "encoded (LR)" not in capture.stages:
                    f4_in = inp[0].detach().cpu()
                    f6_in = inp[1].detach().cpu()
                    if f4_in.dim() == 3:
                        f4_in = f4_in[0]
                        f6_in = f6_in[0]
                    capture.stages["encoded (LR)"] = {
                        "f4_norm": f4_in.norm(dim=-1).reshape(H, W).numpy(),
                        "f6_norm": f6_in.norm(dim=-1).reshape(H, W).numpy(),
                        "H": H, "W": W,
                    }

            return hook

        targets = [
            ("conv1 (LR k=5)",   self.model.conv_layer,    False),
            ("conv2 (LR k=9)",   self.model.conv_lr2,      False),
            ("upsample",         self.model.upsample_conv, True),
            ("conv_hr (HR k=3)", self.model.conv_hr,       False),
        ]
        for name, mod, shape_from_out in targets:
            h = mod.register_forward_hook(make_hook(name, shape_from_out))
            self._handles.append(h)

        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Static plots
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_kernel(ax, w: np.ndarray, fontsize: int = 7):
    vmax = abs(w).max() if abs(w).max() > 0 else 1.0
    for (j, i), v in np.ndenumerate(w):
        color = "white" if abs(v) > 0.55 * vmax else "black"
        ax.text(i, j, f"{v:.3f}", ha="center", va="center",
                fontsize=fontsize, color=color)


def plot_spatial_weights(model: FCCAutoEncoderSRDoubleConvAttn, out_path: Path):
    """Plot learned spatial aggregation weights for all conv layers."""
    layers = [
        ("conv_layer\n(LR k=5)",     model.conv_layer.spatial_weights.detach().cpu().numpy()),
        ("conv_lr2\n(LR k=9)",       model.conv_lr2.spatial_weights.detach().cpu().numpy()),
        ("conv_hr\n(HR k=3)",        model.conv_hr.spatial_weights.detach().cpu().numpy()),
        ("upsample spatial\n(3×3)",  model.upsample_conv.spatial_weights.detach().cpu().numpy()),
    ]

    fig, axes = plt.subplots(1, len(layers), figsize=(4.5 * len(layers), 5.5))
    for ax, (name, w) in zip(axes, layers):
        vmax = max(abs(w.min()), abs(w.max()), 1e-9)
        im = ax.imshow(w, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fs = 6 if w.shape[0] > 5 else 7
        _annotate_kernel(ax, w, fontsize=fs)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(f"shape: {w.shape[0]}×{w.shape[1]}", fontsize=8)

    fig.suptitle("Learned Spatial Aggregation Weights (all conv layers)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_transpose_kernels(model: FCCAutoEncoderSRDoubleConvAttn, out_path: Path):
    """Plot the 22 depthwise kernels of EquivariantTransposeConv."""
    # weight shape: (22, 1, kH, kW) → squeeze to (22, kH, kW)
    kernels = model.upsample_conv.transpose_conv.weight.detach().cpu().squeeze(1).numpy()
    C, kH, kW = kernels.shape

    ncols = 11
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.1))
    axes_flat = axes.flatten()

    irrep_labels = (
        [f"l=4 [{m}]" for m in range(9)] +
        [f"l=6 [{m}]" for m in range(13)]
    )
    vabs = max(abs(kernels.min()), abs(kernels.max()), 1e-9)

    last_im = None
    for i, (ax, k, lbl) in enumerate(zip(axes_flat, kernels, irrep_labels)):
        last_im = ax.imshow(k, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                            interpolation="nearest")
        ax.set_title(lbl, fontsize=7)
        ax.axis("off")

    for ax in axes_flat[C:]:
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes_flat.tolist(), fraction=0.008, pad=0.01,
                     label="Kernel weight")

    fig.suptitle(
        f"EquivariantTransposeConv — Depthwise Kernels ({C} channels, {kH}×{kW})\n"
        f"First 9: l=4 irrep channels  |  Last 13: l=6 irrep channels",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_attn_params(model: FCCAutoEncoderSRDoubleConvAttn, out_path: Path):
    """Visualize learned parameters of each HR attention block."""
    blocks = list(model.hr_attn_blocks)
    n = len(blocks)

    fig, axes = plt.subplots(n, 4, figsize=(20, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    sh_labels = ["SH 0e", "SH 2e[0]", "SH 2e[1]", "SH 2e[2]", "SH 2e[3]", "SH 2e[4]"]

    for i, blk in enumerate(blocks):
        s4  = float(torch.exp(blk.log_s4))
        s6  = float(torch.exp(blk.log_s6))
        ls4 = float(blk.log_s4)
        ls6 = float(blk.log_s6)

        # ── Col 0: scale factors ──────────────────────────────────────────────
        ax = axes[i, 0]
        colors = ["#4C72B0", "#DD8452"]
        bars   = ax.bar(["s4  (l=4)", "s6  (l=6)"], [s4, s6], color=colors)
        ax.set_title(
            f"Block {i}: Attention Scales\nlog_s4={ls4:.3f}  log_s6={ls6:.3f}",
            fontsize=9,
        )
        ax.set_ylabel("exp(log_s)")
        ax.axhline(0, color="k", lw=0.5)
        for bar, v in zip(bars, [s4, s6]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v * 1.05,
                f"{v:.4f}",
                ha="center", va="bottom", fontsize=9,
            )

        # ── Col 1: position bias weights ──────────────────────────────────────
        ax = axes[i, 1]
        pb_w = blk.pos_bias.weight.detach().cpu().numpy().flatten()  # (6,)
        pb_b = float(blk.pos_bias.bias.item())
        ax.bar(range(6), pb_w, color="#55A868")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(sh_labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(
            f"Block {i}: Position Bias Weights\n(scalar bias = {pb_b:.4f})",
            fontsize=9,
        )
        ax.set_ylabel("Weight value")

        # ── Col 2: lin_out weights (1-D: e3nn IrrepsLinear stores flat weights)
        ax = axes[i, 2]
        lo_w = blk.lin_out.weight.detach().cpu().numpy().flatten()  # 1-D
        # Split by irrep type: first C weights for l=4, last C for l=6
        C_hidden = len(lo_w) // 2
        colors_w = ["#4C72B0"] * C_hidden + ["#DD8452"] * C_hidden
        ax.bar(range(len(lo_w)), lo_w, color=colors_w)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(C_hidden - 0.5, color="gray", lw=1.0, ls="--")
        ax.set_title(
            f"Block {i}: lin_out weights  (len={len(lo_w)})\n"
            f"Blue=l=4 ({C_hidden}), Orange=l=6 ({len(lo_w)-C_hidden})",
            fontsize=9,
        )
        ax.set_xlabel("Weight index")
        ax.set_ylabel("Weight value")

        # ── Col 3: lin_in vs lin_out weight magnitudes ────────────────────────
        ax = axes[i, 3]
        li_w  = blk.lin_in.weight.detach().cpu().numpy().flatten()
        lo_nz = np.abs(lo_w)
        li_nz = np.abs(li_w)
        ax.bar(range(len(lo_nz)), lo_nz, color="#C44E52", label=f"lin_out (n={len(lo_nz)})")
        ax2 = ax.twinx()
        ax2.bar(np.arange(len(li_nz)) + 0.35, li_nz, width=0.35,
                color="#55A868", alpha=0.7, label=f"lin_in (n={len(li_nz)})")
        ax.set_title(f"Block {i}: |lin_out| vs |lin_in| weights", fontsize=9)
        ax.set_xlabel("Weight index")
        ax.set_ylabel("|lin_out|", color="#C44E52")
        ax2.set_ylabel("|lin_in|",  color="#55A868")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    fig.suptitle("HR Attention Block Learned Parameters", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic plots (require a forward pass on real data)
# ─────────────────────────────────────────────────────────────────────────────

def _to_spatial(x: torch.Tensor, num_bh: int, num_bw: int,
                block_h: int, block_w: int) -> np.ndarray:
    """Reshape (Bb, Nb) → (Hr, Wr) spatial map."""
    Hr = num_bh * block_h
    Wr = num_bw * block_w
    return (
        x.reshape(num_bh, num_bw, block_h, block_w)
         .permute(0, 2, 1, 3)
         .reshape(Hr, Wr)
         .numpy()
    )


def _attn_spatial_maps(entry: Dict):
    """Return (entropy, mean_recv, self_attn) spatial maps for one captured block."""
    attn     = entry["attn"]           # (Bb, Nb, Nb)
    num_bh   = entry["num_bh"]
    num_bw   = entry["num_bw"]
    block_h  = entry["block_h"]
    block_w  = entry["block_w"]

    # Entropy of each query's attention distribution
    entropy  = -(attn * (attn + 1e-9).log()).sum(-1)           # (Bb, Nb)
    # Mean attention received (column average)
    mean_recv = attn.mean(1)                                    # (Bb, Nb)
    # Self-attention: how much each pixel attends to itself
    self_attn = torch.diagonal(attn, dim1=1, dim2=2)            # (Bb, Nb)

    kw = dict(num_bh=num_bh, num_bw=num_bw, block_h=block_h, block_w=block_w)
    return (
        _to_spatial(entropy,   **kw),
        _to_spatial(mean_recv, **kw),
        _to_spatial(self_attn, **kw),
    )


def plot_attn_entropy(
    attn_data: List[Dict],
    ipf_rgb: Optional[np.ndarray],
    out_path: Path,
):
    """Attention entropy, mean-received, and self-attn spatial maps."""
    n_blocks = len(attn_data)
    has_ipf  = ipf_rgb is not None
    ncols    = 4 if has_ipf else 3

    fig, axes = plt.subplots(
        n_blocks, ncols, figsize=(5.2 * ncols, 4.5 * n_blocks)
    )
    if n_blocks == 1:
        axes = axes[np.newaxis, :]

    for row, entry in enumerate(attn_data):
        ent_map, recv_map, self_map = _attn_spatial_maps(entry)
        bidx = entry["block_idx"]

        panels = [
            ("Attention Entropy\n[nats / pixel]",             ent_map,  "viridis"),
            ("Mean Attention Received\n[col-avg over queries]", recv_map, "hot"),
            ("Self-Attention Weight\n[diag(A)]",               self_map, "plasma"),
        ]
        for col, (title, m, cmap) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(m, cmap=cmap, interpolation="nearest")
            ax.set_title(f"Block {bidx}: {title}", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)

        if has_ipf:
            ax = axes[row, 3]
            ax.imshow(ipf_rgb)
            ent_norm = (ent_map - ent_map.min()) / (ent_map.max() - ent_map.min() + 1e-9)
            ax.imshow(ent_norm, cmap="hot", alpha=0.45, interpolation="nearest")
            ax.set_title(f"Block {bidx}: IPF-Z + entropy overlay", fontsize=9)
            ax.axis("off")

    fig.suptitle("HR Attention Block — Spatial Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_attn_block_matrices(attn_data: List[Dict], out_path: Path, n_samples: int = 6):
    """Show (Nb×Nb) attention matrices for a handful of representative spatial blocks."""
    n_blocks = len(attn_data)
    ncols    = n_samples + 1   # n_samples individual + 1 mean-over-all

    fig, axes = plt.subplots(
        n_blocks, ncols, figsize=(2.8 * ncols, 3.2 * n_blocks)
    )
    if n_blocks == 1:
        axes = axes[np.newaxis, :]

    for row, entry in enumerate(attn_data):
        attn   = entry["attn"]   # (Bb, Nb, Nb)
        Bb     = attn.shape[0]
        bidx   = entry["block_idx"]
        num_bh = entry["num_bh"]
        num_bw = entry["num_bw"]

        # Pick a variety of spatial blocks: four corners + centre + one extra
        center = num_bh // 2 * num_bw + num_bw // 2
        candidates = sorted(set([
            0,
            num_bw - 1,
            (num_bh - 1) * num_bw,
            Bb - 1,
            center,
            Bb // 3,
        ]))[:n_samples]

        for col, sp_idx in enumerate(candidates):
            ax       = axes[row, col]
            mat      = attn[sp_idx].numpy()       # (Nb, Nb)
            bh_i     = sp_idx // num_bw
            bw_i     = sp_idx % num_bw
            im       = ax.imshow(mat, cmap="hot", vmin=0.0, aspect="equal",
                                 interpolation="nearest")
            ax.set_title(f"B{bidx} [{bh_i},{bw_i}]", fontsize=7)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)

        # Final column: mean attention matrix across all spatial blocks
        ax       = axes[row, -1]
        mean_mat = attn.mean(0).numpy()
        im       = ax.imshow(mean_mat, cmap="hot", vmin=0.0, aspect="equal",
                             interpolation="nearest")
        ax.set_title(f"B{bidx} mean\n(all {Bb} blocks)", fontsize=7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)

    fig.suptitle(
        f"HR Attention Matrices — {n_samples} sampled blocks + mean  "
        f"(each axis = Nb pixels within one HR block)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_feature_norms(stages: Dict[str, Dict], out_path: Path):
    """||f4|| and ||f6|| norm maps at each pipeline stage."""
    stage_order = [
        "encoded (LR)",
        "conv1 (LR k=5)",
        "conv2 (LR k=9)",
        "upsample",
        "conv_hr (HR k=3)",
    ]
    # Keep only stages that were actually captured, in order
    keys = [k for k in stage_order if k in stages] + \
           [k for k in stages if k not in stage_order]

    n = len(keys)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(11, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(keys):
        info = stages[name]
        for col, (feat, lbl, cmap) in enumerate([
            (info["f4_norm"], "‖f4‖  (l=4,  9 comps)", "plasma"),
            (info["f6_norm"], "‖f6‖  (l=6, 13 comps)", "viridis"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(feat, cmap=cmap, interpolation="nearest")
            res_tag = f"[{info['H']}×{info['W']}]"
            ax.set_title(f"{name}  {res_tag}: {lbl}", fontsize=9)
            ax.axis("off")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
            cb.ax.tick_params(labelsize=7)

    fig.suptitle("Feature Norm Maps at Each Pipeline Stage", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = ROOT / exp_dir
    exp_dir = exp_dir.resolve()

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    run_dir = Path(args.run_dir).resolve() if args.run_dir else find_latest_run(exp_dir)
    ckpt_path = run_dir / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "attention_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Print header ──────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Experiment : {exp_dir.name}")
    print(f"  Run dir    : {run_dir.name}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {device}")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*64}\n")

    # ── Build & load model ────────────────────────────────────────────────────
    cfg = load_config(exp_dir)
    print("Building model ...")
    model = build_model(cfg, device)
    model.eval()
    epoch = load_checkpoint(model, ckpt_path, device)
    print(f"Loaded checkpoint (epoch {epoch})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Static plots ──────────────────────────────────────────────────────────
    print("\n[1/3] Static kernel / weight analysis ...")
    plot_spatial_weights(model,      out_dir / "01_spatial_weights.png")
    plot_transpose_kernels(model,    out_dir / "02_transpose_kernels.png")
    plot_attn_params(model,          out_dir / "03_attn_params.png")

    if args.no_data:
        print("\nSkipping data-dependent analysis (--no_data).")
        print(f"\nDone. Outputs in: {out_dir}")
        return

    # ── Load one data sample ──────────────────────────────────────────────────
    print(f"\n[2/3] Loading data (split={args.split}, sample_idx={args.sample_idx}) ...")
    lr_flat = hr_flat = None
    lr_shape = hr_shape = None

    try:
        from training.data_loading import build_dataloader

        loader = build_dataloader(
            dataset_root=cfg["dataset_root"],
            split=args.split,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            preload=False,
            preload_torch=False,
            pin_memory=False,
        )

        for idx, (lr_batch, hr_batch) in enumerate(loader):
            if idx == args.sample_idx:
                lr_b = lr_batch[0]   # (4, H_lr, W_lr)
                hr_b = hr_batch[0]   # (4, H_hr, W_hr)
                H_lr, W_lr = int(lr_b.shape[1]), int(lr_b.shape[2])
                H_hr, W_hr = int(hr_b.shape[1]), int(hr_b.shape[2])
                lr_shape = (H_lr, W_lr)
                hr_shape = (H_hr, W_hr)
                lr_flat  = lr_b.permute(1, 2, 0).reshape(-1, 4).to(device)
                lr_flat  = lr_flat / lr_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                hr_flat  = hr_b.permute(1, 2, 0).reshape(-1, 4)
                print(f"  LR: {lr_shape}   HR: {hr_shape}")
                break

        if lr_flat is None:
            raise ValueError(
                f"sample_idx={args.sample_idx} exceeds dataset size for split '{args.split}'"
            )

    except Exception as exc:
        print(f"  WARNING: Could not load data — {exc}")
        print("  Skipping data-dependent analysis.")
        print("  To suppress, pass --no_data or fix the dataset path in config.json.")
        print(f"\nDone. Outputs in: {out_dir}")
        return

    # ── Forward pass with attention & feature captures ────────────────────────
    print("\n[3/3] Running inference with attention & feature captures ...")

    feat_capture = FeatureNormCapture(model)
    attn_capture = AttentionCapture(model)

    with torch.no_grad(), feat_capture, attn_capture:
        q_sr = model.forward_sr(lr_flat, lr_shape=lr_shape)

    print(
        f"  Attention entries captured : {len(attn_capture.data)} "
        f"({len(model.hr_attn_blocks)} block(s))"
    )
    print(f"  Feature-norm stages        : {list(feat_capture.stages.keys())}")

    # ── IPF render for overlay (optional) ─────────────────────────────────────
    ipf_rgb = None
    try:
        from orix.quaternion.symmetry import Oh
        from visualization.ipf_render import render_ipf_rgb

        q_hr_np = hr_flat.reshape(H_hr, W_hr, 4).numpy()
        ipf_rgb = render_ipf_rgb(q_hr_np, Oh, ref_dir="Z")
        print("  IPF-Z render: OK")
    except Exception as exc:
        print(f"  IPF render skipped: {exc}")

    # ── Dynamic plots ──────────────────────────────────────────────────────────
    plot_attn_entropy(
        attn_capture.data,
        ipf_rgb,
        out_dir / "04_attn_entropy.png",
    )
    plot_attn_block_matrices(
        attn_capture.data,
        out_dir / "05_attn_block_matrices.png",
    )
    plot_feature_norms(
        feat_capture.stages,
        out_dir / "06_feature_norms.png",
    )

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
