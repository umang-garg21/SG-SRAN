"""
Step through SR_conv (local-iso FCC SR pipeline) layer by layer.

Prints tensor shapes, summary stats, and a small value preview for:
1) input LR quaternions
2) encode_a1
3) lift_layer (a1 -> full)
4) conv_layer (LR)
5) upsample_conv
6) conv_hr
7) decoder raw output
8) FZ-reduced output
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


# Edit these values directly (no argparse).
CONFIG = {
    "lr_h": 2,
    "lr_w": 2,
    "upsample_factor": 2,
    "upsampler": "conv",  # "conv" or "attention"
    "device": "cpu",
    "seed": 0,
    "head": 8,
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
    print(f"  feature_irreps (SR backbone): {model.feature_irreps}")

    lr_quats = _random_unit_quats(n_lr, device=device, seed=seed)

    with torch.no_grad():
        lr_quats = model._normalize_quaternions(lr_quats)
        feat_a1 = model.encode_a1(lr_quats)
        feat_full_target = model.encode_full_target(lr_quats)
        feat_lifted = model.lift_to_full(feat_a1)
        feat_lr = model.conv_layer(feat_lifted, lr_shape)
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
    _print_tensor("conv_layer_lr_output", feat_lr, head=head)
    _print_tensor("upsample_output", feat_up, head=head)
    _print_tensor("conv_hr_output", feat_hr, head=head)
    _print_tensor("decoder_raw_output", q_dec_raw, head=head)
    _print_tensor("decoder_fz_output", q_dec_fz, head=head)

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
    print(f"  feature_irreps (SR backbone): {model.feature_irreps}")

    lr_quats = _random_unit_quats(n_lr, device=device, seed=seed)

    with torch.no_grad():
        lr_quats = model._normalize_quaternions(lr_quats)
        feat_a1 = model.encode_a1(lr_quats)
        feat_full_target = model.encode_full_target(lr_quats)
        feat_lifted = model.lift_to_full(feat_a1)
        feat_lr = model.conv_layer(feat_lifted, lr_shape)
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
    _print_tensor("conv_layer_lr_output", feat_lr, head=head)
    _print_tensor("upsample_output", feat_up, head=head)
    _print_tensor("conv_hr_output", feat_hr, head=head)
    _print_tensor("decoder_raw_output", q_dec_raw, head=head)
    _print_tensor("decoder_fz_output", q_dec_fz, head=head)

    diff = (q_dec_fz - q_forward).abs().max().item()
    print(f"\nConsistency check: max|reduce_to_fz(decoder_out) - forward_sr| = {diff:.6e}")
