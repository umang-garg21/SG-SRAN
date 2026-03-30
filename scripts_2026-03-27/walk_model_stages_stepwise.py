#!/usr/bin/env python
"""Step through IsoEmbeddingSRAttn stages and write viz after each stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _safe_stage_name(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk through model stages one-by-one and write stage-specific "
            "plots/tensors into separate folders."
        )
    )
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config filename inside exp_dir (default: config.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename in exp_dir/checkpoints or absolute path.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument(
        "--split",
        type=str,
        default="Test",
        choices=["Train", "Val", "Test"],
        help="Dataset split used for sample lookup.",
    )
    parser.add_argument(
        "--sample_offset",
        type=int,
        default=0,
        help="Which sample index to trace in the chosen split.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional dataset root override.",
    )
    parser.add_argument(
        "--lr_npy",
        type=str,
        default=None,
        help="Optional direct LR quaternion npy file path.",
    )
    parser.add_argument(
        "--crop_hw",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Optional top-left LR crop size.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output root. Default: <exp_dir>/stage_walkthrough_stepwise/<sample_label>.",
    )
    parser.add_argument("--head", type=int, default=10, help="Tensor print head length.")
    parser.add_argument(
        "--plot_max_channels",
        type=int,
        default=14,
        help="Max channels in spatial plots.",
    )
    parser.add_argument(
        "--irrep_plot_max_channels_per_block",
        type=int,
        default=None,
        help="Optional max channels per irrep block plot.",
    )
    parser.add_argument(
        "--ipf_ref_dir",
        type=str,
        default="ALL",
        help="IPF ref dir for decoded plots (X/Y/Z/ALL).",
    )
    parser.add_argument(
        "--print_full_tensors",
        action="store_true",
        help="Print full tensor values.",
    )
    parser.add_argument(
        "--make_first3_rgb_plots",
        action="store_true",
        help="Save first-3-channel RGB previews for feature stages.",
    )
    parser.add_argument(
        "--save_stage_tensors_npy",
        action="store_true",
        help="Save stage tensors as npy.",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots (if backend supports it).",
    )
    parser.add_argument(
        "--no_spatial_plots",
        action="store_true",
        help="Skip stage channel heatmaps.",
    )
    parser.add_argument(
        "--no_irrep_channel_plots",
        action="store_true",
        help="Skip irrep-block channel plots.",
    )
    parser.add_argument(
        "--no_stage_ipf_plots",
        action="store_true",
        help="Skip decoded stage IPF plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from utils.symmetry_utils import resolve_symmetry
    from visualization.model_stage_walkthrough import (
        collect_stage_records,
        load_lr_input,
        load_model_from_experiment,
        write_stage_outputs,
    )

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

    crystal = str(getattr(cfg, "crystal", "fcc")).lower()
    sym_name = "D6h" if crystal == "hcp" else "Oh"
    sym_class = resolve_symmetry(sym_name)
    stages, diff = collect_stage_records(
        model,
        lr_arr,
        plot_max_channels=int(args.plot_max_channels),
    )

    if args.out_dir is None:
        out_root = Path(args.exp_dir).resolve() / "stage_walkthrough_stepwise" / label
    else:
        out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Using checkpoint: {checkpoint_path if checkpoint_path is not None else '[random init]'}")
    print(f"Tracing sample: {label}")
    print(f"Model crystal: {crystal}")
    print(f"Model irreps_a1: {model.irreps_a1}")
    print(
        "Model irreps_full: "
        f"{getattr(model, 'irreps_full', getattr(getattr(model, 'encoder', None), 'irreps_full', 'n/a'))}"
    )
    print(f"Use LR conv1: {getattr(model, 'use_lr_conv1', 'n/a')}")
    print(f"Use LR conv2: {getattr(model, 'use_lr_conv2', 'n/a')}")
    print(f"Use attention: {getattr(model, 'use_attention', 'n/a')}")
    if hasattr(model, "lr_blocks"):
        print(f"LR blocks: {len(getattr(model, 'lr_blocks', []))}")
    if hasattr(model, "hr_blocks"):
        print(f"HR blocks: {len(getattr(model, 'hr_blocks', []))}")
    if hasattr(model, "snap_mode"):
        print(f"Snap mode: {getattr(model, 'snap_mode', 'n/a')}")
    print(f"Total stages: {len(stages)}")

    stage_index_lines: list[str] = []
    for idx, stage in enumerate(stages):
        safe = _safe_stage_name(stage.name)
        step_dir = out_root / f"{idx:02d}_{safe}"
        stage_index_lines.append(f"{idx:02d}  {stage.name}  ->  {step_dir}")
        print(f"\n===== Stage {idx:02d}: {stage.name} =====")

        write_stage_outputs(
            model=model,
            stages=[stage],
            out_dir=step_dir,
            sym_class=sym_class,
            head=int(args.head),
            print_full_tensors=bool(args.print_full_tensors),
            make_spatial_plots=not bool(args.no_spatial_plots),
            make_first3_rgb_plots=bool(args.make_first3_rgb_plots),
            make_irrep_channel_plots=not bool(args.no_irrep_channel_plots),
            make_stage_ipf_decode_plots=not bool(args.no_stage_ipf_plots),
            make_stage_ipf_row_figure=False,
            ipf_ref_dir=str(args.ipf_ref_dir),
            show_plots=bool(args.show_plots),
            irrep_plot_max_channels_per_block=args.irrep_plot_max_channels_per_block,
            save_stage_tensors_npy=bool(args.save_stage_tensors_npy),
        )

    index_path = out_root / "stage_index.txt"
    index_path.write_text(
        "\n".join(
            [
                f"sample: {label}",
                f"checkpoint: {checkpoint_path if checkpoint_path is not None else '[random init]'}",
                f"consistency_max_abs_diff: {diff:.6e}",
                "",
                "stages:",
                *stage_index_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nConsistency max|reduce_to_fz(decoder_out)-forward_sr| = {diff:.6e}")
    print(f"Stepwise outputs written to: {out_root}")
    print(f"Stage index written to: {index_path}")


if __name__ == "__main__":
    main()
