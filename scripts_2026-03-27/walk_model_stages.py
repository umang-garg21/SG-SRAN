#!/usr/bin/env python
"""Trace the active IsoEmbeddingSRAttn model stage-by-stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk through IsoEmbeddingSRAttn stages, decode A1 irreps back to "
            "quaternions/IPF, and save per-stage channel visualizations."
        )
    )
    parser.add_argument(
        "--exp_dir",
        required=True,
        type=str,
        help="Experiment directory containing the config and checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config file name inside exp_dir (default: config.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename in exp_dir/checkpoints, or absolute path. Defaults to best_model.pt when present.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Test",
        choices=["Train", "Val", "Test"],
        help="Dataset split used when resolving an LR sample from dataset_root.",
    )
    parser.add_argument(
        "--sample_offset",
        type=int,
        default=0,
        help="Which LR/HR pair to inspect within the split.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional dataset_root override. Defaults to the resolved config value.",
    )
    parser.add_argument(
        "--lr_npy",
        type=str,
        default=None,
        help="Optional direct path to one LR quaternion .npy file. Bypasses dataset split lookup.",
    )
    parser.add_argument(
        "--crop_hw",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Optional LR crop size to trace only the top-left HxW region.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <exp_dir>/stage_walkthrough/<sample_label>.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="How many values to print from the first tensor row.",
    )
    parser.add_argument(
        "--plot_max_channels",
        type=int,
        default=14,
        help="Maximum number of channels to show in each stage spatial plot.",
    )
    parser.add_argument(
        "--irrep_plot_max_channels_per_block",
        type=int,
        default=None,
        help="Optional cap per irrep block plot. Defaults to all channels in each block.",
    )
    parser.add_argument(
        "--ipf_ref_dir",
        type=str,
        default="ALL",
        help="IPF reference direction for decoded stage plots: X, Y, Z, or ALL.",
    )
    parser.add_argument(
        "--print_full_tensors",
        action="store_true",
        help="Print full stage tensors in addition to summaries.",
    )
    parser.add_argument(
        "--make_first3_rgb_plots",
        action="store_true",
        help="Also save first-3-channel RGB previews for feature stages.",
    )
    parser.add_argument(
        "--save_stage_tensors_npy",
        action="store_true",
        help="Save each stage tensor reshaped onto its spatial grid as .npy.",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display matplotlib windows after saving plots.",
    )
    parser.add_argument(
        "--no_spatial_plots",
        action="store_true",
        help="Skip per-stage feature channel heatmaps.",
    )
    parser.add_argument(
        "--no_irrep_channel_plots",
        action="store_true",
        help="Skip per-irrep block channel plots.",
    )
    parser.add_argument(
        "--no_stage_ipf_plots",
        action="store_true",
        help="Skip decoded quaternion/IPF plots for A1 stages.",
    )
    parser.add_argument(
        "--no_stage_ipf_row_figure",
        action="store_true",
        help="Skip the combined row figure of decoded stage IPF outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from visualization.model_stage_walkthrough import run_stage_walkthrough

    run_stage_walkthrough(
        exp_dir=Path(args.exp_dir),
        config_name=args.config,
        checkpoint_name=args.checkpoint,
        device=args.device,
        split=args.split,
        sample_offset=int(args.sample_offset),
        dataset_root=args.dataset_root,
        lr_npy=args.lr_npy,
        crop_hw=None if args.crop_hw is None else (int(args.crop_hw[0]), int(args.crop_hw[1])),
        out_dir=None if args.out_dir is None else Path(args.out_dir),
        head=int(args.head),
        plot_max_channels=int(args.plot_max_channels),
        irrep_plot_max_channels_per_block=args.irrep_plot_max_channels_per_block,
        print_full_tensors=bool(args.print_full_tensors),
        make_spatial_plots=not bool(args.no_spatial_plots),
        make_first3_rgb_plots=bool(args.make_first3_rgb_plots),
        make_irrep_channel_plots=not bool(args.no_irrep_channel_plots),
        make_stage_ipf_decode_plots=not bool(args.no_stage_ipf_plots),
        make_stage_ipf_row_figure=not bool(args.no_stage_ipf_row_figure),
        ipf_ref_dir=str(args.ipf_ref_dir),
        show_plots=bool(args.show_plots),
        save_stage_tensors_npy=bool(args.save_stage_tensors_npy),
    )


if __name__ == "__main__":
    main()
