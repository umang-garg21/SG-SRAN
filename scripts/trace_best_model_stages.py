"""
Trace stage-wise spatial outputs for the trained best IsoEmbeddingSRAttn model.

No argparse: edit CONFIG below.
"""

from __future__ import annotations

import json
import math
import os
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
from trace_sr_conv_layers_real_data_both import (  # type: ignore
    _ensure_hwc_quat,
    _load_lr_pairs,
    _trace_one_sample,
)


CONFIG = {
    "exp_dir": "experiments/Ti64/iso_embedding_sr_attn_hcp_01",
    "config_name": "config_eval.json",
    "checkpoint_name": "best_model.pt",
    "dataset_root": "/data/warren/materials/EBSD/Ti64_DIC_Mclean_QSR_x4",
    "split": "Val",
    "sample_offset": 0,
    "output_subdir": "visualizations/final_stage_spatial_outputs",
    "head": 12,
    "plot_max_channels": 14,
    "irrep_plot_max_channels_per_block": 9,
}


def _make_stage_contact_sheet(spatial_dir: Path, suffix: str, out_name: str, title: str) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        return None

    imgs = sorted(
        p
        for p in spatial_dir.glob(f"*{suffix}")
        if not p.name.startswith("stage_")
    )
    if len(imgs) == 0:
        return None

    tiles = []
    labels = []
    for p in imgs:
        im = Image.open(p).convert("RGB")
        tiles.append(im)
        labels.append(p.name.replace(suffix, ""))

    target_w = max(im.width for im in tiles)
    target_h = max(im.height for im in tiles)
    tiles = [ImageOps.pad(im, (target_w, target_h), color=(245, 245, 245)) for im in tiles]

    cols = 4
    rows = math.ceil(len(tiles) / cols)
    pad = 20
    label_h = 26
    title_h = 46

    canvas_w = cols * target_w + (cols + 1) * pad
    canvas_h = rows * (target_h + label_h) + (rows + 1) * pad + title_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), title, fill=(0, 0, 0))

    for i, (im, lbl) in enumerate(zip(tiles, labels)):
        r = i // cols
        c = i % cols
        x = pad + c * (target_w + pad)
        y = title_h + pad + r * (target_h + label_h + pad)
        canvas.paste(im, (x, y))
        draw.text((x, y + target_h + 4), lbl, fill=(0, 0, 0))

    out_path = spatial_dir / out_name
    canvas.save(out_path)
    return out_path


def main() -> None:
    exp_dir = (REPO_ROOT / str(CONFIG["exp_dir"])).resolve()
    cfg_path = exp_dir / str(CONFIG["config_name"])
    ckpt_path = exp_dir / "checkpoints" / str(CONFIG["checkpoint_name"])
    cfg = json.loads(cfg_path.read_text())

    dataset_root = Path(str(CONFIG["dataset_root"]))
    split = str(CONFIG["split"])
    sample_offset = max(0, int(CONFIG["sample_offset"]))
    pairs = _load_lr_pairs(dataset_root, split)
    if sample_offset >= len(pairs):
        raise IndexError(
            f"sample_offset={sample_offset} out of range for {len(pairs)} pairs in {dataset_root}/{split}"
        )

    pair_key, lr_fp, hr_fp = pairs[sample_offset]
    lr_arr = _ensure_hwc_quat(np.load(lr_fp))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsoEmbeddingSRAttn(
        crystal=str(cfg.get("crystal", "hcp")),
        d6_convention=str(cfg.get("d6_convention", "z_axis")),
        device=device,
        upsample_factor=int(cfg.get("scale", 4)),
        upsample_residual=bool(cfg.get("upsample_residual", True)),
        num_hr_attn_blocks=int(cfg.get("num_hr_attn_blocks", 1)),
        hr_attn_num_channels=int(cfg.get("hr_attn_num_channels", 4)),
        hr_attn_block_size=int(cfg.get("hr_attn_block_size", 16)),
        hr_attn_tp_out_chunk_size=cfg.get("hr_attn_tp_out_chunk_size", 128),
        hr_attn_checkpoint=bool(cfg.get("hr_attn_checkpoint", False)),
        decoder_cubochoric_resolution=int(cfg.get("decoder_cubochoric_resolution", 1)),
        decoder_num_starts=int(cfg.get("decoder_num_starts", 8)),
        decoder_steps=int(cfg.get("decoder_steps", 12)),
        decoder_lr=float(cfg.get("decoder_lr", 0.05)),
        decoder_method=str(cfg.get("decoder_method", "cubochoric")),
        decoder_max_table_rows=cfg.get("decoder_max_table_rows", None),
        decoder_table_cache_dir=cfg.get("decoder_table_cache_dir", "out/decoder_lookup_tables"),
        decoder_backend=str(cfg.get("decoder_backend", "optimizing")),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    out_root = exp_dir / str(CONFIG["output_subdir"])
    out_dir = out_root / f"{split.lower()}_axis_{pair_key[0]}_block_{pair_key[1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _trace_one_sample(
        model=model,
        lr_quats_passive_hwc=lr_arr,
        out_dir=out_dir,
        head=int(CONFIG["head"]),
        print_full_tensors=False,
        make_spatial_plots=True,
        make_irrep_channel_plots=True,
        show_plots=False,
        plot_max_channels=int(CONFIG["plot_max_channels"]),
        irrep_plot_max_channels_per_block=int(CONFIG["irrep_plot_max_channels_per_block"]),
        save_stage_tensors_npy=True,
    )

    spatial_dir = out_dir / "spatial_plots"
    norm_sheet = _make_stage_contact_sheet(
        spatial_dir=spatial_dir,
        suffix="_norm.png",
        out_name="stage_norm_contact_sheet.png",
        title="Stage Spatial Outputs (Norm)",
    )
    rgb_sheet = _make_stage_contact_sheet(
        spatial_dir=spatial_dir,
        suffix="_rgb.png",
        out_name="stage_rgb_contact_sheet.png",
        title="Stage Spatial Outputs (First-3ch RGB)",
    )

    print(f"LR file: {lr_fp}")
    print(f"HR file: {hr_fp}")
    print(f"Output dir: {out_dir}")
    if norm_sheet is not None:
        print(f"Norm contact sheet: {norm_sheet}")
    if rgb_sheet is not None:
        print(f"RGB contact sheet: {rgb_sheet}")


if __name__ == "__main__":
    main()

