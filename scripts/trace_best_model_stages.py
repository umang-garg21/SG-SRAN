"""
Trace stage-wise outputs for the trained best IsoEmbeddingSRAttn model.

Configured to emit only:
1) irrep_block_plots
2) stage_ipf_decoder

No argparse: edit CONFIG below.
"""

from __future__ import annotations

import json
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
from utils.symmetry_utils import resolve_symmetry


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
    "irrep_plot_max_channels_per_block": None,
    "make_stage_ipf_row_figure": True,
    "ipf_ref_dir": "ALL",
}


def _adapt_checkpoint_state_for_model(
    model: IsoEmbeddingSRAttn,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    sd = dict(state)

    # Legacy checkpoints (from earlier transpose-conv implementation) stored
    # one upsample kernel per channel. Current model stores one kernel per
    # irrep copy; convert by averaging channels that share a copy id.
    legacy_up_key = "upsample_conv.transpose_conv.weight"
    new_up_key = "upsample_conv.transpose_kernels"
    if legacy_up_key in sd and new_up_key not in sd:
        w = sd.pop(legacy_up_key)
        if (
            hasattr(model, "upsample_conv")
            and hasattr(model.upsample_conv, "channel_to_copy_idx")
            and hasattr(model.upsample_conv, "num_irrep_copies")
        ):
            idx = model.upsample_conv.channel_to_copy_idx.detach().cpu()
            ncopy = int(model.upsample_conv.num_irrep_copies)
            k_h = int(w.shape[-2])
            k_w = int(w.shape[-1])
            kernels = torch.empty((ncopy, 1, k_h, k_w), dtype=w.dtype)
            for copy_id in range(ncopy):
                ch = torch.nonzero(idx == copy_id, as_tuple=False).flatten()
                if int(ch.numel()) == 0:
                    kernels[copy_id].zero_()
                else:
                    kernels[copy_id] = w[ch].mean(dim=0)
            sd[new_up_key] = kernels

    # If these optional modules were introduced after checkpoint creation, avoid
    # random-weight impact by disabling/removing them before loading.
    if "a1_to_full_proj.weight" not in sd and hasattr(model, "a1_to_full_proj"):
        model.a1_to_full_proj = None
    if (
        "conv_lr1.residual_proj.weight" not in sd
        and hasattr(model, "conv_lr1")
        and hasattr(model.conv_lr1, "residual_proj")
    ):
        model.conv_lr1.use_residual = False
        model.conv_lr1.residual_proj = None

    return sd
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
        use_lr_conv1=bool(cfg.get("use_lr_conv1", True)),
        use_lr_conv2=bool(cfg.get("use_lr_conv2", True)),
        use_attention=bool(cfg.get("use_attention", True)),
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
    state = _adapt_checkpoint_state_for_model(model, state)
    load_res = model.load_state_dict(state, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print(f"load_state_dict(strict=False): missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}")
        if load_res.missing_keys:
            print("  missing_keys:", load_res.missing_keys)
        if load_res.unexpected_keys:
            print("  unexpected_keys:", load_res.unexpected_keys)
    model = model.to(device).eval()

    out_root = exp_dir / str(CONFIG["output_subdir"])
    out_dir = out_root / f"{split.lower()}_axis_{pair_key[0]}_block_{pair_key[1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    crystal = str(cfg.get("crystal", "hcp")).lower()
    sym_name = "D6h" if crystal == "hcp" else "Oh"
    sym_class = resolve_symmetry(sym_name)

    _trace_one_sample(
        model=model,
        sym_class=sym_class,
        lr_quats_passive_hwc=lr_arr,
        out_dir=out_dir,
        head=int(CONFIG["head"]),
        print_full_tensors=False,
        make_spatial_plots=False,
        make_first3_rgb_plots=False,
        make_irrep_channel_plots=True,
        make_stage_ipf_decode_plots=True,
        make_stage_ipf_row_figure=bool(CONFIG.get("make_stage_ipf_row_figure", True)),
        ipf_ref_dir=str(CONFIG.get("ipf_ref_dir", "ALL")),
        show_plots=False,
        plot_max_channels=int(CONFIG["plot_max_channels"]),
        irrep_plot_max_channels_per_block=(
            None
            if CONFIG.get("irrep_plot_max_channels_per_block", None) is None
            else int(CONFIG["irrep_plot_max_channels_per_block"])
        ),
        save_stage_tensors_npy=False,
    )

    print(f"LR file: {lr_fp}")
    print(f"HR file: {hr_fp}")
    print(f"Output dir: {out_dir}")
    print(f"Irrep block dir: {out_dir / 'irrep_block_plots'}")
    print(f"Stage IPF dir: {out_dir / 'stage_ipf_decoder'}")


if __name__ == "__main__":
    main()
