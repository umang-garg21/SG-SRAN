"""
Trace IsoEmbeddingSRAttn layer-by-layer.

This script prints tensor statistics (and optionally full tensors) for each stage:
1) input LR quaternions
2) encode_a1
3) LR conv1
4) LR conv2
5) transpose upsample conv
6) HR conv1
7) each attention block output
8) final full->a1 projection
9) decoder raw output
10) FZ-reduced output
11) forward_sr output (consistency check)

It also saves spatial color plots for each stage.
No argparse: edit CONFIG directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

# Allow direct execution: `python scripts/trace_sr_conv_layers.py`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.SR_double_conv_SRattn import IsoEmbeddingSRAttn


CONFIG = {
    "crystal": "fcc",  # "fcc" or "hcp"
    "d6_convention": "z_axis",
    "lr_h": 2,
    "lr_w": 2,
    "upsample_factor": 2,
    "use_lr_conv1": False,
    "use_lr_conv2": False,
    "use_attention": False,
    "device": "cpu",
    "seed": 0,
    "head": 10,
    "num_hr_attn_blocks": 1,
    "hr_attn_num_channels": 8,
    "hr_attn_block_size": 16,
    "decoder_cubochoric_resolution": 1,
    "decoder_num_starts": 3,
    "decoder_steps": 2,
    "decoder_lr": 0.05,
    "decoder_method": "cubochoric",
    "decoder_backend": "optimizing",  # "optimizing" or "learnable"
    "decoder_learnable_hidden_dim": 256,
    "decoder_learnable_num_layers": 3,
    "decoder_learnable_dropout": 0.0,
    "print_full_tensors": True,
    "make_spatial_plots": True,
    "make_irrep_channel_plots": True,
    "show_plots": False,
    "plot_dir": "outputs/iso_embedding_sr_attn_trace_plots",
    "plot_max_channels": 14,
    "irrep_plot_dir": "outputs/iso_embedding_sr_attn_trace_plots/irrep_blocks",
    "irrep_plot_max_channels_per_block": 9,
}


def _random_unit_quats(n: int, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    q = torch.randn(n, 4, dtype=torch.float32, device=device)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(q[:, :1] < 0.0, -q, q)


def _print_tensor(name: str, x: torch.Tensor, head: int = 8) -> None:
    t = x.detach()
    print(f"\n{name}")
    print(f"  shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
    print(
        "  stats:"
        f" min={float(t.min().item()): .6e}"
        f" max={float(t.max().item()): .6e}"
        f" mean={float(t.mean().item()): .6e}"
        f" std={float(t.std(unbiased=False).item()): .6e}"
    )
    if t.ndim == 1:
        vec = t[: min(head, t.shape[0])].cpu().tolist()
        print(f"  values[:{len(vec)}]={vec}")
        return
    flat = t.reshape(-1, t.shape[-1])
    vec = flat[0, : min(head, flat.shape[-1])].cpu().tolist()
    print(f"  first_row[:{len(vec)}]={vec}")


def _print_tensor_full(name: str, x: torch.Tensor) -> None:
    t = x.detach().cpu()
    print(f"\n{name} (full)")
    torch.set_printoptions(threshold=10_000_000, linewidth=220, sci_mode=False)
    try:
        print(t)
    finally:
        torch.set_printoptions(profile="default")


def _reshape_grid(features: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    h, w = shape
    if features.ndim != 2:
        raise ValueError(f"Expected 2D tensor (N, C), got shape={tuple(features.shape)}")
    n, _ = features.shape
    if n != h * w:
        raise ValueError(f"Expected N={h*w} for shape={shape}, got N={n}")
    return features.detach().cpu().reshape(h, w, -1)


def _norm01(x: torch.Tensor) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()
    span = (x_max - x_min).clamp_min(1e-12)
    return (x - x_min) / span


def _save_spatial_plots(
    name: str,
    features: torch.Tensor,
    shape: tuple[int, int],
    out_dir: Path,
    max_channels: int,
) -> None:
    grid = _reshape_grid(features, shape)
    h, w, c = grid.shape
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")

    norm_map = torch.linalg.norm(grid, dim=-1)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(norm_map.numpy(), cmap="viridis", interpolation="nearest")
    ax.set_title(f"{name}: channel L2 norm ({h}x{w})")
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_name}_norm.png", dpi=180)
    plt.close(fig)

    if c >= 3:
        rgb = grid[..., :3]
    elif c == 2:
        rgb = torch.stack([grid[..., 0], grid[..., 1], torch.zeros_like(grid[..., 0])], dim=-1)
    else:
        rgb = torch.stack([grid[..., 0], grid[..., 0], grid[..., 0]], dim=-1)
    rgb = _norm01(rgb)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(rgb.numpy(), interpolation="nearest")
    ax.set_title(f"{name}: first-3ch RGB ({h}x{w})")
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_name}_rgb.png", dpi=180)
    plt.close(fig)

    n_ch = min(int(c), int(max_channels))
    n_cols = 4
    n_rows = (n_ch + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows))
    axes = axes.reshape(-1)
    for idx in range(n_rows * n_cols):
        ax = axes[idx]
        if idx < n_ch:
            chan = grid[..., idx]
            im = ax.imshow(chan.numpy(), cmap="coolwarm", interpolation="nearest")
            ax.set_title(f"ch {idx}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.axis("off")
    fig.suptitle(f"{name}: channel maps (first {n_ch}/{c})", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_name}_channels.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _irrep_block_specs(irreps) -> list[tuple[str, int, int]]:
    """Return per-copy irrep blocks in packed channel order."""
    blocks: list[tuple[str, int, int]] = []
    start = 0
    for mul, ir in irreps:
        mul_i = int(mul)
        dim_i = int(ir.dim)
        l_i = int(ir.l)
        p_i = "e" if int(ir.p) == 1 else "o"
        for copy_idx in range(mul_i):
            end = start + dim_i
            blocks.append((f"l{l_i}{p_i}_copy{copy_idx}", start, end))
            start = end
    return blocks


def _save_irrep_block_plots(
    name: str,
    features: torch.Tensor,
    shape: tuple[int, int],
    out_dir: Path,
    irreps,
    max_channels_per_block: int,
) -> None:
    """Save spatial maps split by irrep block for debugging."""
    grid = _reshape_grid(features, shape)
    _, _, c = grid.shape
    blocks = _irrep_block_specs(irreps)
    if len(blocks) == 0:
        return

    if int(blocks[-1][2]) != int(c):
        print(
            f"[warning] {name}: irrep block dim ({blocks[-1][2]}) "
            f"does not match tensor channels ({c}); skipping irrep block plots."
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")

    for block_name, start, end in blocks:
        block = grid[..., start:end]
        block_dim = int(end - start)
        print(
            f"  irrep-block {name}: {block_name} channels[{start}:{end}] "
            f"dim={block_dim} mean={float(block.mean().item()): .3e} std={float(block.std(unbiased=False).item()): .3e}"
        )

        norm_map = torch.linalg.norm(block, dim=-1)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(norm_map.numpy(), cmap="magma", interpolation="nearest")
        ax.set_title(f"{name}: {block_name} norm")
        ax.set_xlabel("W")
        ax.set_ylabel("H")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe_name}__{block_name}__norm.png", dpi=180)
        plt.close(fig)

        n_ch = min(block_dim, int(max_channels_per_block))
        n_cols = 4
        n_rows = (n_ch + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows))
        axes = axes.reshape(-1)
        for idx in range(n_rows * n_cols):
            ax = axes[idx]
            if idx < n_ch:
                chan = block[..., idx]
                im = ax.imshow(chan.numpy(), cmap="coolwarm", interpolation="nearest")
                ax.set_title(f"{block_name} ch{idx}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            else:
                ax.axis("off")
        fig.suptitle(f"{name}: {block_name} channels (first {n_ch}/{block_dim})", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe_name}__{block_name}__channels.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def _apply_attention_stepwise(model, feat: torch.Tensor, hr_shape: tuple[int, int]):
    hr_h, hr_w = hr_shape
    batched = feat.dim() == 3
    if not batched:
        feat = feat.unsqueeze(0)

    bsz, n, c = feat.shape
    if n != hr_h * hr_w:
        raise ValueError(f"Expected N={hr_h*hr_w}, got {n}")

    block_h = min(model.hr_attn_block_size, hr_h)
    block_w = min(model.hr_attn_block_size, hr_w)
    pad_h = (-hr_h) % block_h
    pad_w = (-hr_w) % block_w
    hr_h_pad, hr_w_pad = hr_h + pad_h, hr_w + pad_w

    feat_work = feat
    if pad_h > 0 or pad_w > 0:
        feat_2d = feat_work.reshape(bsz, hr_h, hr_w, c).permute(0, 3, 1, 2)
        feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
        feat_work = feat_2d.permute(0, 2, 3, 1).reshape(bsz, hr_h_pad * hr_w_pad, c)

    def _unpad(x: torch.Tensor) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return x
        return x.reshape(bsz, hr_h_pad, hr_w_pad, c)[:, :hr_h, :hr_w, :].reshape(bsz, hr_h * hr_w, c)

    sh_block = model._get_hr_sh_block(block_h, block_w, feat_work.device, feat_work.dtype)

    stage_outputs = []
    for idx, block in enumerate(model.attention_blocks):
        delta = block(feat_work, sh_block, hr_h_pad, hr_w_pad, block_h, block_w)
        feat_work = feat_work + delta
        cur = _unpad(feat_work)
        if not batched:
            cur = cur.squeeze(0)
        stage_outputs.append((f"attention_block_{idx}_output", cur))

    feat_final = _unpad(feat_work)
    if not batched:
        feat_final = feat_final.squeeze(0)
    return feat_final, stage_outputs


def main() -> None:
    crystal = str(CONFIG["crystal"])
    d6_convention = str(CONFIG["d6_convention"])
    lr_h = int(CONFIG["lr_h"])
    lr_w = int(CONFIG["lr_w"])
    upsample_factor = int(CONFIG["upsample_factor"])
    device_str = str(CONFIG["device"])
    seed = int(CONFIG["seed"])
    head = int(CONFIG["head"])
    print_full_tensors = bool(CONFIG["print_full_tensors"])
    make_spatial_plots = bool(CONFIG["make_spatial_plots"])
    make_irrep_channel_plots = bool(CONFIG["make_irrep_channel_plots"])
    show_plots = bool(CONFIG["show_plots"])
    plot_max_channels = int(CONFIG["plot_max_channels"])
    plot_dir_cfg = str(CONFIG["plot_dir"])
    irrep_plot_dir_cfg = str(CONFIG["irrep_plot_dir"])
    irrep_plot_max_channels_per_block = int(CONFIG["irrep_plot_max_channels_per_block"])

    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device(device_str)
    lr_shape = (lr_h, lr_w)
    n_lr = lr_h * lr_w
    plot_dir = (repo_root / plot_dir_cfg).resolve()
    irrep_plot_dir = (repo_root / irrep_plot_dir_cfg).resolve()

    model = IsoEmbeddingSRAttn(
        crystal=crystal,
        d6_convention=d6_convention,
        device=device,
        upsample_factor=upsample_factor,
        use_lr_conv1=bool(CONFIG["use_lr_conv1"]),
        use_lr_conv2=bool(CONFIG["use_lr_conv2"]),
        use_attention=bool(CONFIG["use_attention"]),
        num_hr_attn_blocks=int(CONFIG["num_hr_attn_blocks"]),
        hr_attn_num_channels=int(CONFIG["hr_attn_num_channels"]),
        hr_attn_block_size=int(CONFIG["hr_attn_block_size"]),
        decoder_cubochoric_resolution=int(CONFIG["decoder_cubochoric_resolution"]),
        decoder_num_starts=int(CONFIG["decoder_num_starts"]),
        decoder_steps=int(CONFIG["decoder_steps"]),
        decoder_lr=float(CONFIG["decoder_lr"]),
        decoder_method=str(CONFIG["decoder_method"]),
        decoder_backend=str(CONFIG["decoder_backend"]),
    ).eval()

    print("Model")
    print(f"  class         : {model.__class__.__name__}")
    print(f"  crystal       : {crystal}")
    print(f"  irreps_a1     : {model.irreps_a1}")
    print(f"  irreps_full   : {model.irreps_full}")
    print(f"  use_lr_conv1  : {model.use_lr_conv1}")
    print(f"  use_lr_conv2  : {model.use_lr_conv2}")
    print(f"  use_attention : {model.use_attention}")
    print(f"  has_lift_layer: {hasattr(model, 'lift_layer')}")
    print(
        "  conv_lr1 tp   :"
        f" {model.conv_lr1.tp.irreps_in1} x {model.conv_lr1.tp.irreps_in2} -> {model.conv_lr1.tp.irreps_out}"
    )
    print(
        "  conv_lr2 tp   :"
        f" {model.conv_lr2.tp.irreps_in1} x {model.conv_lr2.tp.irreps_in2} -> {model.conv_lr2.tp.irreps_out}"
    )
    print(
        "  upsample tp   :"
        f" {model.upsample_conv.tp.irreps_in1} x {model.upsample_conv.tp.irreps_in2} -> {model.upsample_conv.tp.irreps_out}"
    )
    print(
        "  conv_hr1 tp   :"
        f" {model.conv_hr1.tp.irreps_in1} x {model.conv_hr1.tp.irreps_in2} -> {model.conv_hr1.tp.irreps_out}"
    )
    print(f"  final_proj    : {model.final_proj.irreps_in} -> {model.final_proj.irreps_out}")

    lr_quats = _random_unit_quats(n_lr, device=device, seed=seed)

    with torch.no_grad():
        feat_a1_lr = model.encode_a1(lr_quats)

        if model.use_lr_conv1:
            feat_lr1 = model.conv_lr1(feat_a1_lr, lr_shape)
            stage_lr1_name = "conv_lr1_output"
        else:
            feat_lr1 = model._apply_pointwise_linear(model.a1_to_full_proj, feat_a1_lr)
            stage_lr1_name = "conv_lr1_bypass_proj_output"

        if model.use_lr_conv2:
            feat_lr2 = model.conv_lr2(feat_lr1, lr_shape)
            stage_lr2_name = "conv_lr2_output"
        else:
            feat_lr2 = feat_lr1
            stage_lr2_name = "conv_lr2_bypass_identity_output"

        feat_up, hr_shape = model.upsample_conv(feat_lr2, lr_shape)
        feat_hr1 = model.conv_hr1(feat_up, hr_shape)

        if model.use_attention and len(model.attention_blocks) > 0:
            feat_attn, attn_stage_outputs = _apply_attention_stepwise(model, feat_hr1, hr_shape)
        else:
            feat_attn = feat_hr1
            attn_stage_outputs = [("attention_bypass_identity_output", feat_attn)]

        feat_a1_hr = model.final_proj(feat_attn)

    q_dec_raw = model.decoder(feat_a1_hr)
    q_dec_fz = model.reduce_to_fz(q_dec_raw)
    q_forward = model.forward_sr(lr_quats, lr_shape=lr_shape, normalize_input=False)

    n_hr = hr_shape[0] * hr_shape[1]
    print(f"\nShapes: LR={lr_shape}, HR={hr_shape}, n_lr={n_lr}, n_hr={n_hr}")

    stages: list[tuple[str, torch.Tensor, tuple[int, int], int, object | None]] = [
        ("input_quats_lr", lr_quats, lr_shape, 4, None),
        ("encode_a1_lr", feat_a1_lr, lr_shape, plot_max_channels, model.irreps_a1),
        (stage_lr1_name, feat_lr1, lr_shape, plot_max_channels, model.irreps_full),
        (stage_lr2_name, feat_lr2, lr_shape, plot_max_channels, model.irreps_full),
        ("upsample_output", feat_up, hr_shape, plot_max_channels, model.irreps_full),
        ("conv_hr1_output", feat_hr1, hr_shape, plot_max_channels, model.irreps_full),
    ]
    for name, tensor in attn_stage_outputs:
        stages.append((name, tensor, hr_shape, plot_max_channels, model.irreps_full))
    stages.extend(
        [
            ("attention_output", feat_attn, hr_shape, plot_max_channels, model.irreps_full),
            ("final_proj_output_a1", feat_a1_hr, hr_shape, plot_max_channels, model.irreps_a1),
            ("decoder_raw_output", q_dec_raw, hr_shape, 4, None),
            ("decoder_fz_output", q_dec_fz, hr_shape, 4, None),
            ("forward_sr_output", q_forward, hr_shape, 4, None),
        ]
    )

    for name, tensor, _, _, _ in stages:
        _print_tensor(name, tensor, head=head)

    if print_full_tensors:
        for name, tensor, _, _, _ in stages:
            _print_tensor_full(name, tensor)

    if make_spatial_plots:
        for name, tensor, shape, max_ch, _ in stages:
            _save_spatial_plots(name, tensor, shape, plot_dir, max_channels=max_ch)
        print(f"\nSaved spatial plots to: {plot_dir}")

    if make_irrep_channel_plots:
        for name, tensor, shape, _, irreps_spec in stages:
            if irreps_spec is None:
                continue
            _save_irrep_block_plots(
                name=name,
                features=tensor,
                shape=shape,
                out_dir=irrep_plot_dir,
                irreps=irreps_spec,
                max_channels_per_block=irrep_plot_max_channels_per_block,
            )
        print(f"Saved irrep block plots to: {irrep_plot_dir}")

    if make_spatial_plots or make_irrep_channel_plots:
        if show_plots:
            plt.show()

    diff = (q_dec_fz - q_forward).abs().max().item()
    print(f"\nConsistency check: max|reduce_to_fz(decoder_out) - forward_sr| = {diff:.6e}")


if __name__ == "__main__":
    main()
