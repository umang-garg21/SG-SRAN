"""
Trace IsoEmbeddingSRAttn layer-by-layer on real LR dataset samples (HCP + FCC).

This mirrors trace_sr_conv_layers.py but uses passive HWC quaternions loaded from
real datasets, similar to test_group_properties_real_data_both.py.

No argparse: edit CONFIG directly.
"""

from __future__ import annotations

import os
import re
import shutil
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

from models.SR_double_conv_SRattn_a1 import IsoEmbeddingSRAttn
from trace_sr_conv_layers import (  # type: ignore
    _apply_attention_stepwise,
    _print_tensor,
    _print_tensor_full,
    _save_irrep_block_plots,
    _save_spatial_plots,
)
from utils.symmetry_utils import resolve_symmetry
from visualization.ipf_render import render_ipf_image


CONFIG = {
    "device": "cpu",
    "split": "Test",
    "samples_per_dataset": 1,
    "sample_offset": 0,
    "lr_crop_hw": [16, 16],
    "head": 10,
    "print_full_tensors": False,
    "make_spatial_plots": False,
    "make_first3_rgb_plots": False,
    "make_irrep_channel_plots": True,
    "show_plots": False,
    "plot_max_channels": 14,
    "irrep_plot_max_channels_per_block": None,
    "save_stage_tensors_npy": False,
    "seed": 0,
    "upsample_factor": 4,
    "use_lr_conv1": True,
    "use_lr_conv2": True,
    "use_attention": True,
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
    "make_stage_ipf_decode_plots": True,
    "make_stage_ipf_row_figure": True,
    "ipf_ref_dir": "ALL",
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


def _save_stage_ipf_row_figure(
    rows: list[tuple[str, Path]],
    out_png: Path,
) -> Path | None:
    if len(rows) == 0:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except Exception:
        return None

    images: list[Image.Image] = []
    labels: list[str] = []
    for name, png_path in rows:
        if not png_path.exists():
            continue
        images.append(Image.open(png_path).convert("RGB"))
        labels.append(name)

    if len(images) == 0:
        return None

    target_w = max(im.width for im in images)
    target_h = max(im.height for im in images)
    images = [ImageOps.pad(im, (target_w, target_h), color=(245, 245, 245)) for im in images]

    pad = 16
    font_size = max(16, min(36, target_h // 8))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        title_font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, font_size + 2))
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Compute label-column width from actual row labels.
    probe = Image.new("RGB", (10, 10), (255, 255, 255))
    probe_draw = ImageDraw.Draw(probe)
    row_texts = [f"{i + 1:02d}. {label}" for i, label in enumerate(labels)]
    max_text_w = 0
    for txt in row_texts:
        bb = probe_draw.textbbox((0, 0), txt, font=font)
        max_text_w = max(max_text_w, int(bb[2] - bb[0]))
    label_w = max(300, max_text_w + 2 * pad)

    title = "Stage IPF Decoder Outputs"
    title_bb = probe_draw.textbbox((0, 0), title, font=title_font)
    title_h = int(title_bb[3] - title_bb[1]) + 2 * pad

    canvas_w = label_w + target_w + 3 * pad
    canvas_h = title_h + (len(images) * target_h) + ((len(images) + 1) * pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad // 2), title, fill=(0, 0, 0), font=title_font)

    for i, (im, txt) in enumerate(zip(images, row_texts)):
        y = title_h + pad + i * (target_h + pad)
        x = label_w + 2 * pad
        canvas.paste(im, (x, y))

        # Row label background panel.
        panel_x0 = pad // 2
        panel_x1 = label_w + pad // 2
        panel_y0 = y
        panel_y1 = y + target_h
        draw.rectangle(
            [(panel_x0, panel_y0), (panel_x1, panel_y1)],
            fill=(247, 247, 247),
            outline=(180, 180, 180),
            width=2,
        )
        bb = draw.textbbox((0, 0), txt, font=font)
        txt_h = int(bb[3] - bb[1])
        tx = pad
        ty = y + max(0, (target_h - txt_h) // 2)
        draw.text((tx, ty), txt, fill=(0, 0, 0), font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png


def _prune_trace_output_dirs(out_dir: Path, keep: set[str]) -> None:
    managed = {"stage_tensors_npy", "spatial_plots", "irrep_block_plots", "stage_ipf_decoder"}
    for name in sorted(managed - keep):
        p = out_dir / name
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def _trace_one_sample(
    model: IsoEmbeddingSRAttn,
    sym_class,
    lr_quats_passive_hwc: np.ndarray,
    out_dir: Path,
    head: int,
    print_full_tensors: bool,
    make_spatial_plots: bool,
    make_first3_rgb_plots: bool,
    make_irrep_channel_plots: bool,
    make_stage_ipf_decode_plots: bool,
    make_stage_ipf_row_figure: bool,
    ipf_ref_dir: str,
    show_plots: bool,
    plot_max_channels: int,
    irrep_plot_max_channels_per_block: int | None,
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

        if bool(getattr(model, "use_lr_conv1", True)):
            feat_lr1 = model.conv_lr1(feat_a1_lr, lr_shape)
            stage_lr1_name = "conv_lr1_output"
        else:
            feat_lr1 = feat_a1_lr
            stage_lr1_name = "conv_lr1_bypass_identity_output"

        if bool(getattr(model, "use_lr_conv2", True)):
            feat_lr2 = model.conv_lr2(feat_lr1, lr_shape)
            stage_lr2_name = "conv_lr2_output"
        else:
            feat_lr2 = feat_lr1
            stage_lr2_name = "conv_lr2_bypass_identity_output"

        feat_up, hr_shape = model.upsample_conv(feat_lr2, lr_shape)
        feat_hr1 = model.conv_hr1(feat_up, hr_shape)
        if bool(getattr(model, "use_attention", True)) and len(model.attention_blocks) > 0:
            feat_attn, attn_stage_outputs = _apply_attention_stepwise(model, feat_hr1, hr_shape)
        else:
            feat_attn = feat_hr1
            attn_stage_outputs = [("attention_bypass_identity_output", feat_attn)]
        feat_a1_hr = model.final_proj(feat_attn)

    # Optimizing decoder performs internal gradient-based refinement.
    with torch.enable_grad():
        q_dec_raw = model.decoder(feat_a1_hr)
        q_dec_fz = model.reduce_to_fz(q_dec_raw)
        q_forward = model.forward_sr(lr_quats, lr_shape=lr_shape, normalize_input=False)

    ir_lr1 = getattr(model.conv_lr1, "irreps_out", model.irreps_a1)
    ir_lr2 = getattr(model.conv_lr2, "irreps_out", ir_lr1)
    ir_up = getattr(model.upsample_conv, "irreps_out", ir_lr2)
    ir_hr1 = getattr(model.conv_hr1, "irreps_out", ir_up)
    ir_attn = ir_hr1

    stages: list[tuple[str, torch.Tensor, tuple[int, int], int, object | None]] = [
        ("input_quats_lr", lr_quats, lr_shape, 4, None),
        ("encode_a1_lr", feat_a1_lr, lr_shape, plot_max_channels, model.irreps_a1),
        (stage_lr1_name, feat_lr1, lr_shape, plot_max_channels, ir_lr1),
        (stage_lr2_name, feat_lr2, lr_shape, plot_max_channels, ir_lr2),
        ("upsample_output", feat_up, hr_shape, plot_max_channels, ir_up),
        ("conv_hr1_output", feat_hr1, hr_shape, plot_max_channels, ir_hr1),
    ]
    for name, tensor in attn_stage_outputs:
        stages.append((name, tensor, hr_shape, plot_max_channels, ir_attn))
    stages.extend(
        [
            ("attention_output", feat_attn, hr_shape, plot_max_channels, ir_attn),
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

    keep_dirs: set[str] = set()
    if save_stage_tensors_npy:
        keep_dirs.add("stage_tensors_npy")
    if make_spatial_plots:
        keep_dirs.add("spatial_plots")
    if make_irrep_channel_plots:
        keep_dirs.add("irrep_block_plots")
    if make_stage_ipf_decode_plots:
        keep_dirs.add("stage_ipf_decoder")
    _prune_trace_output_dirs(out_dir=out_dir, keep=keep_dirs)

    if save_stage_tensors_npy:
        npy_dir = out_dir / "stage_tensors_npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor, shape, _, _ in stages:
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")
            np.save(npy_dir / f"{safe_name}.npy", _reshape_stage_tensor(tensor, shape))
        print(f"  Saved stage tensors (.npy): {npy_dir}")

    if make_spatial_plots:
        plot_dir = out_dir / "spatial_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for stale in list(plot_dir.glob("*_norm.png")) + list(plot_dir.glob("*_rgb.png")):
            stale.unlink(missing_ok=True)
        for name, tensor, shape, max_ch, irreps_spec in stages:
            if irreps_spec is None:
                # Skip quaternion-stage channel plots.
                continue
            _save_spatial_plots(
                name,
                tensor,
                shape,
                plot_dir,
                max_channels=max_ch,
                save_first3_rgb=make_first3_rgb_plots,
            )
        print(f"  Saved spatial plots: {plot_dir}")

    if make_irrep_channel_plots:
        irrep_dir = out_dir / "irrep_block_plots"
        irrep_dir.mkdir(parents=True, exist_ok=True)
        for stale in irrep_dir.glob("*__norm.png"):
            stale.unlink(missing_ok=True)
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

    if make_stage_ipf_decode_plots:
        ipf_dir = out_dir / "stage_ipf_decoder"
        ipf_dir.mkdir(parents=True, exist_ok=True)
        stage_ipf_rows: list[tuple[str, Path]] = []
        for name, tensor, shape, _, irreps_spec in stages:
            q_stage: torch.Tensor | None = None
            if irreps_spec is None and int(tensor.shape[-1]) == 4:
                q_stage = tensor
            elif (
                irreps_spec is not None
                and str(irreps_spec) == str(model.irreps_a1)
                and int(tensor.shape[-1]) == int(model.feature_dim_a1)
            ):
                with torch.enable_grad():
                    q_stage = model.reduce_to_fz(model.decoder(tensor))

            if q_stage is None:
                continue

            q_hwc = _reshape_stage_tensor(q_stage, shape)
            safe_name = (
                name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("->", "_to_")
            )
            out_png = ipf_dir / f"{safe_name}_ipf.png"
            render_ipf_image(
                q_hwc,
                sym_class=sym_class,
                out_png=str(out_png),
                ref_dir=str(ipf_ref_dir),
                include_key=True,
                overwrite=True,
                format_input=True,
            )
            stage_ipf_rows.append((name, out_png))
        print(f"  Saved decoded stage IPF maps: {ipf_dir}")
        if make_stage_ipf_row_figure:
            row_png = _save_stage_ipf_row_figure(
                rows=stage_ipf_rows,
                out_png=ipf_dir / "stage_ipf_decode_rows.png",
            )
            if row_png is not None:
                print(f"  Saved stage IPF row figure: {row_png}")

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
            use_lr_conv1=bool(CONFIG.get("use_lr_conv1", True)),
            use_lr_conv2=bool(CONFIG.get("use_lr_conv2", True)),
            use_attention=bool(CONFIG.get("use_attention", True)),
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
        print(f"  use_lr_conv1={model.use_lr_conv1}")
        print(f"  use_lr_conv2={model.use_lr_conv2}")
        print(f"  use_attention={model.use_attention}")

        sym_name = "D6h" if crystal.lower() == "hcp" else "Oh"
        sym_class = resolve_symmetry(sym_name)

        for idx, (pair_key, lr_fp, hr_fp) in enumerate(selected):
            lr_arr = _ensure_hwc_quat(np.load(lr_fp))
            crop_hw = CONFIG.get("lr_crop_hw", None)
            if crop_hw is not None:
                ch, cw = int(crop_hw[0]), int(crop_hw[1])
                lr_arr = lr_arr[:ch, :cw, :]
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
                sym_class=sym_class,
                lr_quats_passive_hwc=lr_arr,
                out_dir=out_dir,
                head=int(CONFIG["head"]),
                print_full_tensors=bool(CONFIG["print_full_tensors"]),
                make_spatial_plots=bool(CONFIG["make_spatial_plots"]),
                make_first3_rgb_plots=bool(CONFIG.get("make_first3_rgb_plots", False)),
                make_irrep_channel_plots=bool(CONFIG["make_irrep_channel_plots"]),
                make_stage_ipf_decode_plots=bool(CONFIG.get("make_stage_ipf_decode_plots", True)),
                make_stage_ipf_row_figure=bool(CONFIG.get("make_stage_ipf_row_figure", True)),
                ipf_ref_dir=str(CONFIG.get("ipf_ref_dir", "ALL")),
                show_plots=bool(CONFIG["show_plots"]),
                plot_max_channels=int(CONFIG["plot_max_channels"]),
                irrep_plot_max_channels_per_block=(
                    None
                    if CONFIG.get("irrep_plot_max_channels_per_block", None) is None
                    else int(CONFIG["irrep_plot_max_channels_per_block"])
                ),
                save_stage_tensors_npy=bool(CONFIG["save_stage_tensors_npy"]),
            )

    print(f"\nDone. Outputs written under: {out_root}")


if __name__ == "__main__":
    main()
