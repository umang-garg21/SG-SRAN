"""
Trace IsoEmbeddingSRAttn layer-by-layer on real LR dataset samples (HCP + FCC).

This mirrors trace_sr_conv_layers.py but uses passive HWC quaternions loaded from
real datasets, similar to test_group_properties_real_data_both.py.

No argparse: edit CONFIG directly.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, SCRIPT_DIR):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from models.SR_double_conv_SRattn import IsoEmbeddingSRAttn
from trace_sr_conv_layers import (  # type: ignore
    _apply_attention_stepwise,
    _print_tensor,
    _print_tensor_full,
    _save_irrep_block_plots,
    _save_spatial_plots,
)


CONFIG = {
    "device": "cpu",
    "split": "Test",
    "samples_per_dataset": 1,
    "sample_offset": 0,
    "head": 10,
    "print_full_tensors": False,
    "make_spatial_plots": True,
    "make_irrep_channel_plots": True,
    "show_plots": False,
    "plot_max_channels": 14,
    "irrep_plot_max_channels_per_block": 9,
    "save_stage_tensors_npy": True,
    "seed": 0,
    "upsample_factor": 4,
    "num_hr_attn_blocks": 1,
    "hr_attn_num_channels": 8,
    "hr_attn_block_size": 16,
    "decoder_backend": "optimizing",  # "optimizing" or "learnable"
    "decoder_cubochoric_resolution": 1,
    "decoder_num_starts": 1,
    "decoder_steps": 0,
    "decoder_lr": 0.05,
    "decoder_method": "cubochoric",
    "decoder_learnable_hidden_dim": 256,
    "decoder_learnable_num_layers": 3,
    "decoder_learnable_dropout": 0.0,
    # Use full decoder lookup table for highest-fidelity traces.
    "decoder_max_table_rows": None,
    "decoder_table_cache_dir": "out/decoder_lookup_tables",
    "out_root": "outputs/iso_embedding_sr_attn_trace_real_data_both",
    "datasets": [
        {
            "name": "hcp_ti64",
            "crystal": "hcp",
            "d6_convention": "z_axis",
            "dataset_root": "/data/warren/materials/materials_data_mount/datasets/Ti64_DIC_Mclean_QSR_x4",
        },
        {
            "name": "fcc_in718",
            "crystal": "fcc",
            "d6_convention": "z_axis",
            "dataset_root": "/data/warren/materials/materials_data_mount/datasets/IN718_QSR_x4",
        },
    ],
}

_NAME_RE = re.compile(
    r"^(?P<ds>.+)_(?P<split>train|val|test)_(?P<which>hr|lr)_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
    re.IGNORECASE,
)


def _pair_key(path: Path) -> tuple[str, int] | None:
    m = _NAME_RE.match(path.name)
    if m is None:
        return None
    return m.group("axis").lower(), int(m.group("id"))


def _load_lr_pairs(dataset_root: Path, split: str) -> list[tuple[tuple[str, int], Path, Path]]:
    split_dir = dataset_root / str(split)
    lr_dir = split_dir / "LR_Data"
    hr_dir = split_dir / "HR_Data"
    if not lr_dir.exists():
        raise FileNotFoundError(f"Missing LR directory: {lr_dir}")
    if not hr_dir.exists():
        raise FileNotFoundError(f"Missing HR directory: {hr_dir}")

    lr_map = {}
    for fp in sorted(lr_dir.glob("*.npy")):
        k = _pair_key(fp)
        if k is not None:
            lr_map[k] = fp
    hr_map = {}
    for fp in sorted(hr_dir.glob("*.npy")):
        k = _pair_key(fp)
        if k is not None:
            hr_map[k] = fp

    common = sorted(set(lr_map.keys()).intersection(hr_map.keys()))
    return [(k, lr_map[k], hr_map[k]) for k in common]


def _ensure_hwc_quat(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D quaternion image, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        out = arr
    elif arr.shape[0] == 4:
        out = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"Could not locate quaternion axis of length 4 in {arr.shape}")
    return out.astype(np.float32, copy=False)


def _reshape_stage_tensor(
    tensor: torch.Tensor,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    h, w = shape_hw
    t = tensor.detach().cpu()
    if t.ndim != 2:
        raise ValueError(f"Expected stage tensor shape (N,C), got {tuple(t.shape)}")
    if int(t.shape[0]) != int(h * w):
        raise ValueError(f"Expected N={h*w}, got N={int(t.shape[0])}")
    return t.reshape(h, w, int(t.shape[1])).numpy().astype(np.float32, copy=False)


def _trace_one_sample(
    model: IsoEmbeddingSRAttn,
    lr_quats_passive_hwc: np.ndarray,
    out_dir: Path,
    head: int,
    print_full_tensors: bool,
    make_spatial_plots: bool,
    make_irrep_channel_plots: bool,
    show_plots: bool,
    plot_max_channels: int,
    irrep_plot_max_channels_per_block: int,
    save_stage_tensors_npy: bool,
) -> None:
    h, w, _ = lr_quats_passive_hwc.shape
    lr_shape = (h, w)
    lr_quats = torch.from_numpy(lr_quats_passive_hwc.reshape(h * w, 4)).to(
        device=model.device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        feat_a1_lr = model.encode_a1(lr_quats)
        feat_lr1 = model.conv_lr1(feat_a1_lr, lr_shape)
        feat_lr2 = model.conv_lr2(feat_lr1, lr_shape)
        feat_up, hr_shape = model.upsample_conv(feat_lr2, lr_shape)
        feat_hr1 = model.conv_hr1(feat_up, hr_shape)
        feat_attn, attn_stage_outputs = _apply_attention_stepwise(model, feat_hr1, hr_shape)
        feat_a1_hr = model.final_proj(feat_attn)

    # Optimizing decoder performs internal gradient-based refinement.
    with torch.enable_grad():
        q_dec_raw = model.decoder(feat_a1_hr)
        q_dec_fz = model.reduce_to_fz(q_dec_raw)
        q_forward = model.forward_sr(lr_quats, lr_shape=lr_shape, normalize_input=False)

    stages: list[tuple[str, torch.Tensor, tuple[int, int], int, object | None]] = [
        ("input_quats_lr", lr_quats, lr_shape, 4, None),
        ("encode_a1_lr", feat_a1_lr, lr_shape, plot_max_channels, model.irreps_a1),
        ("conv_lr1_output", feat_lr1, lr_shape, plot_max_channels, model.irreps_full),
        ("conv_lr2_output", feat_lr2, lr_shape, plot_max_channels, model.irreps_full),
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

    print(f"  LR shape={lr_shape}, HR shape={hr_shape}")
    for name, tensor, _, _, _ in stages:
        _print_tensor(name, tensor, head=head)
    if print_full_tensors:
        for name, tensor, _, _, _ in stages:
            _print_tensor_full(name, tensor)

    if save_stage_tensors_npy:
        npy_dir = out_dir / "stage_tensors_npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor, shape, _, _ in stages:
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")
            np.save(npy_dir / f"{safe_name}.npy", _reshape_stage_tensor(tensor, shape))
        print(f"  Saved stage tensors (.npy): {npy_dir}")

    if make_spatial_plots:
        plot_dir = out_dir / "spatial_plots"
        for name, tensor, shape, max_ch, _ in stages:
            _save_spatial_plots(name, tensor, shape, plot_dir, max_channels=max_ch)
        print(f"  Saved spatial plots: {plot_dir}")

    if make_irrep_channel_plots:
        irrep_dir = out_dir / "irrep_block_plots"
        for name, tensor, shape, _, irreps_spec in stages:
            if irreps_spec is None:
                continue
            _save_irrep_block_plots(
                name=name,
                features=tensor,
                shape=shape,
                out_dir=irrep_dir,
                irreps=irreps_spec,
                max_channels_per_block=irrep_plot_max_channels_per_block,
            )
        print(f"  Saved irrep block plots: {irrep_dir}")

    if make_spatial_plots or make_irrep_channel_plots:
        if show_plots:
            from matplotlib import pyplot as plt

            plt.show()

    diff = float((q_dec_fz - q_forward).abs().max().item())
    print(f"  Consistency max|reduce_to_fz(decoder_out)-forward_sr|={diff:.6e}")


def main() -> None:
    device = torch.device(str(CONFIG["device"]))
    split = str(CONFIG["split"])
    samples_per_dataset = max(1, int(CONFIG["samples_per_dataset"]))
    sample_offset = max(0, int(CONFIG["sample_offset"]))
    out_root = (REPO_ROOT / str(CONFIG["out_root"])).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for ds in CONFIG["datasets"]:
        ds_name = str(ds["name"])
        crystal = str(ds["crystal"])
        d6_convention = str(ds.get("d6_convention", "z_axis"))
        dataset_root = Path(str(ds["dataset_root"]))

        pairs = _load_lr_pairs(dataset_root, split)
        if len(pairs) == 0:
            raise RuntimeError(f"No LR/HR pairs found at {dataset_root}/{split}")
        selected = pairs[sample_offset : sample_offset + samples_per_dataset]
        if len(selected) == 0:
            raise RuntimeError(
                f"{ds_name}: requested offset={sample_offset}, but only {len(pairs)} pair(s)."
            )

        model = IsoEmbeddingSRAttn(
            crystal=crystal,
            d6_convention=d6_convention,
            device=device,
            upsample_factor=int(CONFIG["upsample_factor"]),
            num_hr_attn_blocks=int(CONFIG["num_hr_attn_blocks"]),
            hr_attn_num_channels=int(CONFIG["hr_attn_num_channels"]),
            hr_attn_block_size=int(CONFIG["hr_attn_block_size"]),
            decoder_cubochoric_resolution=int(CONFIG["decoder_cubochoric_resolution"]),
            decoder_num_starts=int(CONFIG["decoder_num_starts"]),
            decoder_steps=int(CONFIG["decoder_steps"]),
            decoder_lr=float(CONFIG["decoder_lr"]),
            decoder_method=str(CONFIG["decoder_method"]),
            decoder_max_table_rows=CONFIG.get("decoder_max_table_rows"),
            decoder_table_cache_dir=CONFIG.get("decoder_table_cache_dir"),
            decoder_backend=str(CONFIG["decoder_backend"]),
        ).eval()

        print("\n" + "=" * 96)
        print(
            f"Dataset={ds_name} crystal={crystal} split={split} "
            f"samples={len(selected)}/{len(pairs)} root={dataset_root}"
        )
        print(f"  irreps_a1={model.irreps_a1}")
        print(f"  irreps_full={model.irreps_full}")

        for idx, (pair_key, lr_fp, hr_fp) in enumerate(selected):
            lr_arr = _ensure_hwc_quat(np.load(lr_fp))
            axis, block_id = pair_key
            sample_tag = f"{idx:03d}_axis_{axis}_block_{block_id}"
            out_dir = out_root / ds_name / sample_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n[{ds_name}] sample={idx} key={pair_key} "
                f"lr_file={lr_fp.name} hr_file={hr_fp.name} lr_shape={tuple(lr_arr.shape)}"
            )
            _trace_one_sample(
                model=model,
                lr_quats_passive_hwc=lr_arr,
                out_dir=out_dir,
                head=int(CONFIG["head"]),
                print_full_tensors=bool(CONFIG["print_full_tensors"]),
                make_spatial_plots=bool(CONFIG["make_spatial_plots"]),
                make_irrep_channel_plots=bool(CONFIG["make_irrep_channel_plots"]),
                show_plots=bool(CONFIG["show_plots"]),
                plot_max_channels=int(CONFIG["plot_max_channels"]),
                irrep_plot_max_channels_per_block=int(CONFIG["irrep_plot_max_channels_per_block"]),
                save_stage_tensors_npy=bool(CONFIG["save_stage_tensors_npy"]),
            )

    print(f"\nDone. Outputs written under: {out_root}")


if __name__ == "__main__":
    main()
