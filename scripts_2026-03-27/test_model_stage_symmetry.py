#!/usr/bin/env python
"""Test stagewise symmetry of the active IsoEmbeddingSRAttn model."""

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
            "Test symmetry one stage at a time by applying a crystal symmetry to the "
            "output of each stage and feeding that transformed state into the next stage."
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
        "--fresh_init",
        action="store_true",
        help="Use a freshly initialized model and skip loading any checkpoint.",
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
        help="Optional LR crop size to test only the top-left HxW region.",
    )
    parser.add_argument(
        "--sym_index",
        type=int,
        default=1,
        help="Which crystal symmetry operator to apply from model.encoder.sym_ops.",
    )
    parser.add_argument(
        "--tol_rel",
        type=float,
        default=5e-3,
        help="Relative-error tolerance for stagewise symmetry checks.",
    )
    parser.add_argument(
        "--tol_rms",
        type=float,
        default=2e-4,
        help="RMS-error tolerance for stagewise symmetry checks.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional path to save the stagewise symmetry report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from visualization.model_stage_symmetry import run_stage_symmetry_test

    run_stage_symmetry_test(
        exp_dir=Path(args.exp_dir),
        config_name=args.config,
        checkpoint_name=args.checkpoint,
        fresh_init=bool(args.fresh_init),
        device=args.device,
        split=args.split,
        sample_offset=int(args.sample_offset),
        dataset_root=args.dataset_root,
        lr_npy=args.lr_npy,
        crop_hw=None if args.crop_hw is None else (int(args.crop_hw[0]), int(args.crop_hw[1])),
        sym_index=int(args.sym_index),
        tol_rel=float(args.tol_rel),
        tol_rms=float(args.tol_rms),
        out_json=None if args.out_json is None else Path(args.out_json),
    )


if __name__ == "__main__":
    main()
