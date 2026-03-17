"""
Step through SR_conv (local-iso FCC SR pipeline) layer by layer.

Prints tensor shapes, summary stats, and a small value preview for:
1) input LR quaternions
2) encode_a1
3) encode_full_target
4) lift_layer (a1 -> full)
5) first conv_layer (a1 -> full)
6) upsample_conv
7) conv_hr
8) decoder raw output
9) FZ-reduced output
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
from matplotlib import pyplot as plt


# Edit these values directly (no argparse).
CONFIG = {
    "lr_h": 2,
    "lr_w": 2,
    "upsample_factor": 2,
    "upsampler": "conv",  # "conv" or "attention"
    "device": "cpu",
    "seed": 0,
    "head": 9,
    "print_full_tensors": True,
    "make_spatial_plots": True,
    "show_plots": True,
    "plot_dir": "tmp/sr_conv_trace_plots",
    "plot_max_channels": 14,
    # Leave empty to use repo default:
    #   <repo>/symmetry_groups/local_iso_lookup_O_res1_irreps.npy
    "lookup_path": "",
    "lookup_resolution": 1,
}


def _load_sr_conv_module(repo_root: Path):
    mod_path = repo_root / "models" / "SR_conv.py"
    spec = importlib.util.spec_from_file_location("repo_sr_conv", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    if t.ndim >= 2:
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
    n, c = features.shape
    if n != h * w:
        raise ValueError(f"Expected N={h*w} for shape={shape}, got N={n}")
    return features.detach().cpu().reshape(h, w, c)


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
    max_channels: int = 14,
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
    norm_path = out_dir / f"{safe_name}_norm.png"
    fig.savefig(norm_path, dpi=180)
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
    rgb_path = out_dir / f"{safe_name}_rgb.png"
    fig.savefig(rgb_path, dpi=180)
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
    ch_path = out_dir / f"{safe_name}_channels.png"
    fig.savefig(ch_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    lr_h = int(CONFIG["lr_h"])
    lr_w = int(CONFIG["lr_w"])
    upsample_factor = int(CONFIG["upsample_factor"])
    upsampler = str(CONFIG["upsampler"])
    if upsampler not in {"conv", "attention"}:
        raise ValueError(f"CONFIG['upsampler'] must be 'conv' or 'attention', got: {upsampler}")
    device_str = str(CONFIG["device"])
    seed = int(CONFIG["seed"])
    head = int(CONFIG["head"])
    print_full_tensors = bool(CONFIG["print_full_tensors"])
    make_spatial_plots = bool(CONFIG["make_spatial_plots"])
    show_plots = bool(CONFIG["show_plots"])
    plot_max_channels = int(CONFIG["plot_max_channels"])
    plot_dir_cfg = str(CONFIG["plot_dir"])
    lookup_resolution = int(CONFIG["lookup_resolution"])
    lookup_path_cfg = str(CONFIG["lookup_path"])

    repo_root = Path(__file__).resolve().parents[1]
    module = _load_sr_conv_module(repo_root)

    lookup_path = (
        Path(lookup_path_cfg).expanduser().resolve()
        if lookup_path_cfg.strip()
        else (repo_root / "symmetry_groups" / "local_iso_lookup_O_res1_irreps.npy").resolve()
    )
    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup table not found: {lookup_path}")

    device = torch.device(device_str)
    lr_shape = (lr_h, lr_w)
    n_lr = lr_h * lr_w
    plot_dir = (repo_root / plot_dir_cfg).resolve()

    model = module.FCCAutoEncoderSR(
        device=device,
        upsample_factor=upsample_factor,
        upsampler=upsampler,
        decoder_backend="lookup",
        decoder_config={
            "decoder_lookup_resolution": lookup_resolution,
            "decoder_lookup_npy_path": str(lookup_path),
        },
    ).eval()

    print("Model Irreps")
    print(f"  irreps_a1   : {model.irreps_a1}")
    print(f"  irreps_full : {model.irreps_full}")
    print(f"  lift_layer  : {model.lift_layer.irreps_in} -> {model.lift_layer.irreps_out}")
    print(
        "  first conv tp:"
        f" {model.conv_layer.irreps_in} x {model.conv_layer.irreps_in}"
        f" -> {model.conv_layer.irreps_out}"
    )
    print(f"  upsample irreps: {model.upsample_conv.irreps_feat if hasattr(model.upsample_conv, 'irreps_feat') else model.upsample_conv.irreps_io}")
    print(f"  conv_hr tp     : {model.conv_hr.irreps_in} x {model.conv_hr.irreps_in} -> {model.conv_hr.irreps_out}")

    lr_quats = _random_unit_quats(n_lr, device=device, seed=seed)

    with torch.no_grad():
        lr_quats = model._normalize_quaternions(lr_quats)
        feat_a1 = model.encode_a1(lr_quats)
        feat_full_target = model.encode_full_target(lr_quats)
        feat_lifted = model.lift_to_full(feat_a1)
        feat_lr = model.conv_layer(feat_a1, lr_shape)
        feat_up, hr_shape = model.upsample_conv(feat_lr, lr_shape)
        feat_hr = model.conv_hr(feat_up, hr_shape)
        q_dec_raw = model.decoder(feat_hr)
        q_dec_fz = model.reduce_to_fz(q_dec_raw)
        q_forward = model.forward_sr(lr_quats, lr_shape=lr_shape, normalize_input=False)

    print(f"\nShapes: LR={lr_shape}, HR={hr_shape}, n_lr={n_lr}, n_hr={hr_shape[0] * hr_shape[1]}")
    _print_tensor("input_quats_lr", lr_quats, head=head)
    _print_tensor("encode_a1", feat_a1, head=head)
    _print_tensor("encode_full_target", feat_full_target, head=head)
    _print_tensor("lift_layer_output", feat_lifted, head=head)
    _print_tensor("conv_layer_lr_output (a1 -> full)", feat_lr, head=head)
    _print_tensor("upsample_output", feat_up, head=head)
    _print_tensor("conv_hr_output", feat_hr, head=head)
    _print_tensor("decoder_raw_output", q_dec_raw, head=head)
    _print_tensor("decoder_fz_output", q_dec_fz, head=head)

    if print_full_tensors:
        _print_tensor_full("input_quats_lr", lr_quats)
        _print_tensor_full("encode_a1", feat_a1)
        _print_tensor_full("encode_full_target", feat_full_target)
        _print_tensor_full("lift_layer_output", feat_lifted)
        _print_tensor_full("conv_layer_lr_output (a1 -> full)", feat_lr)
        _print_tensor_full("upsample_output", feat_up)
        _print_tensor_full("conv_hr_output", feat_hr)
        _print_tensor_full("decoder_raw_output", q_dec_raw)
        _print_tensor_full("decoder_fz_output", q_dec_fz)

    if make_spatial_plots:
        _save_spatial_plots("input_quats_lr", lr_quats, lr_shape, plot_dir, max_channels=plot_max_channels)
        _save_spatial_plots("encode_a1", feat_a1, lr_shape, plot_dir, max_channels=plot_max_channels)
        _save_spatial_plots(
            "encode_full_target", feat_full_target, lr_shape, plot_dir, max_channels=plot_max_channels
        )
        _save_spatial_plots(
            "lift_layer_output", feat_lifted, lr_shape, plot_dir, max_channels=plot_max_channels
        )
        _save_spatial_plots(
            "conv_layer_lr_output (a1 -> full)", feat_lr, lr_shape, plot_dir, max_channels=plot_max_channels
        )
        _save_spatial_plots("upsample_output", feat_up, hr_shape, plot_dir, max_channels=plot_max_channels)
        _save_spatial_plots("conv_hr_output", feat_hr, hr_shape, plot_dir, max_channels=plot_max_channels)
        _save_spatial_plots("decoder_raw_output", q_dec_raw, hr_shape, plot_dir, max_channels=4)
        _save_spatial_plots("decoder_fz_output", q_dec_fz, hr_shape, plot_dir, max_channels=4)
        print(f"\nSaved spatial plots to: {plot_dir}")
        if show_plots:
            plt.show()

    diff = (q_dec_fz - q_forward).abs().max().item()
    print(f"\nConsistency check: max|reduce_to_fz(decoder_out) - forward_sr| = {diff:.6e}")


if __name__ == "__main__":
    # main()

    lr_h = int(CONFIG["lr_h"])
    lr_w = int(CONFIG["lr_w"])
    upsample_factor = int(CONFIG["upsample_factor"])
    upsampler = str(CONFIG["upsampler"])
    if upsampler not in {"conv", "attention"}:
        raise ValueError(f"CONFIG['upsampler'] must be 'conv' or 'attention', got: {upsampler}")
    device_str = str(CONFIG["device"])
    seed = int(CONFIG["seed"])
    head = int(CONFIG["head"])
    print_full_tensors = bool(CONFIG["print_full_tensors"])
    make_spatial_plots = bool(CONFIG["make_spatial_plots"])
    show_plots = bool(CONFIG["show_plots"])
    plot_max_channels = int(CONFIG["plot_max_channels"])
    plot_dir_cfg = str(CONFIG["plot_dir"])
    lookup_resolution = int(CONFIG["lookup_resolution"])
    lookup_path_cfg = str(CONFIG["lookup_path"])

    repo_root = Path(__file__).resolve().parents[1]
    module = _load_sr_conv_module(repo_root)

    lookup_path = (
        Path(lookup_path_cfg).expanduser().resolve()
        if lookup_path_cfg.strip()
        else (repo_root / "symmetry_groups" / "local_iso_lookup_O_res1_irreps.npy").resolve()
    )
    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup table not found: {lookup_path}")

    device = torch.device(device_str)
    lr_shape = (lr_h, lr_w)
    n_lr = lr_h * lr_w
    plot_dir = (repo_root / plot_dir_cfg).resolve()

    model = module.FCCAutoEncoderSR(
        device=device,
        upsample_factor=upsample_factor,
        upsampler=upsampler,
        decoder_backend="lookup",
        decoder_config={
            "decoder_lookup_resolution": lookup_resolution,
            "decoder_lookup_npy_path": str(lookup_path),
        },
    ).eval()

    print("Model Irreps")
    print(f"  irreps_a1   : {model.irreps_a1}")
    print(f"  irreps_full : {model.irreps_full}")
    print(f"  lift_layer  : {model.lift_layer.irreps_in} -> {model.lift_layer.irreps_out}")
    print(
        "  first conv tp:"
        f" {model.conv_layer.irreps_in} x {model.conv_layer.irreps_in}"
        f" -> {model.conv_layer.irreps_out}"
    )
    print(f"  upsample irreps: {model.upsample_conv.irreps_feat if hasattr(model.upsample_conv, 'irreps_feat') else model.upsample_conv.irreps_io}")
    print(f"  conv_hr tp     : {model.conv_hr.irreps_in} x {model.conv_hr.irreps_in} -> {model.conv_hr.irreps_out}")

    lr_quats = _random_unit_quats(n_lr, device=device, seed=seed)

    with torch.no_grad():
        lr_quats = model._normalize_quaternions(lr_quats)
        feat_a1 = model.encode_a1(lr_quats)
        feat_full_target = model.encode_full_target(lr_quats)
        feat_lifted = model.lift_to_full(feat_a1)
        feat_lr = model.conv_layer(feat_a1, lr_shape)
        feat_up, hr_shape = model.upsample_conv(feat_lr, lr_shape)
        feat_hr = model.conv_hr(feat_up, hr_shape)
        q_dec_raw = model.decoder(feat_hr)
        q_dec_fz = model.reduce_to_fz(q_dec_raw)
        q_forward = model.forward_sr(lr_quats, lr_shape=lr_shape, normalize_input=False)

    print(f"\nShapes: LR={lr_shape}, HR={hr_shape}, n_lr={n_lr}, n_hr={hr_shape[0] * hr_shape[1]}")
    _print_tensor("input_quats_lr", lr_quats, head=head)
    _print_tensor("encode_a1", feat_a1, head=head)
    _print_tensor("encode_full_target", feat_full_target, head=head)
    _print_tensor("lift_layer_output", feat_lifted, head=head)
    _print_tensor("conv_layer_lr_output (a1 -> full)", feat_lr, head=head)
    _print_tensor("upsample_output", feat_up, head=head)
    _print_tensor("conv_hr_output", feat_hr, head=head)
    _print_tensor("decoder_raw_output", q_dec_raw, head=head)
    _print_tensor("decoder_fz_output", q_dec_fz, head=head)

    if print_full_tensors:
        _print_tensor_full("input_quats_lr", lr_quats)
        _print_tensor_full("encode_a1", feat_a1)
        _print_tensor_full("encode_full_target", feat_full_target)
        _print_tensor_full("lift_layer_output", feat_lifted)
        _print_tensor_full("conv_layer_lr_output (a1 -> full)", feat_lr)
        _print_tensor_full("upsample_output", feat_up)
        _print_tensor_full("conv_hr_output", feat_hr)
        _print_tensor_full("decoder_raw_output", q_dec_raw)
        _print_tensor_full("decoder_fz_output", q_dec_fz)

    if make_spatial_plots:
        _save_spatial_plots("input_quats_lr", lr_quats, lr_shape, plot_dir, max_channels=plot_max_channels)
        _save_spatial_plots("encode_a1", feat_a1, lr_shape, plot_dir, max_channels=plot_max_channels)
        _save_spatial_plots(
            "encode_full_target", feat_full_target, lr_shape, plot_dir, max_channels=plot_max_channels
        )
        _save_spatial_plots(
            "lift_layer_output", feat_lifted, lr_shape, plot_dir, max_channels=plot_max_channels
        )
        _save_spatial_plots(
            "conv_layer_lr_output (a1 -> full)", feat_lr, lr_shape, plot_dir, max_channels=plot_max_channels
        )
        _save_spatial_plots("upsample_output", feat_up, hr_shape, plot_dir, max_channels=plot_max_channels)
        _save_spatial_plots("conv_hr_output", feat_hr, hr_shape, plot_dir, max_channels=plot_max_channels)
        _save_spatial_plots("decoder_raw_output", q_dec_raw, hr_shape, plot_dir, max_channels=4)
        _save_spatial_plots("decoder_fz_output", q_dec_fz, hr_shape, plot_dir, max_channels=4)
        print(f"\nSaved spatial plots to: {plot_dir}")
        if show_plots:
            plt.show()

    diff = (q_dec_fz - q_forward).abs().max().item()
    print(f"\nConsistency check: max|reduce_to_fz(decoder_out) - forward_sr| = {diff:.6e}")
