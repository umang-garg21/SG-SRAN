#!/usr/bin/env python3
"""Interactive live grain-attention probe for the modified one-sided SR model."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")


def _preparse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    return parser.parse_known_args(argv)[0]


_PRE_ARGS = _preparse_cli(sys.argv[1:])
if _PRE_ARGS.gpu_ids is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_PRE_ARGS.gpu_ids)

import matplotlib

def _tk_display_works() -> bool:
    try:
        import tkinter as tk
    except Exception:
        return False
    try:
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:
        return False


def _auto_select_interactive_backend(requested: str | None) -> str | None:
    if requested:
        req = str(requested).lower()
        if req == "tkagg" and not _tk_display_works():
            return "WebAgg"
        return requested
    if _tk_display_works():
        return "TkAgg"
    return "WebAgg"


_SELECTED_BACKEND = _auto_select_interactive_backend(_PRE_ARGS.backend)
if _SELECTED_BACKEND:
    matplotlib.use(_SELECTED_BACKEND)
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.infer_iso_embedding_sr_attn import (
    _flatten_quat_chw,
    _load_model_from_checkpoint,
    _resolve_checkpoint,
    _resolve_model_class,
    _unpack_batch,
)
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from utils.stage_probe_utils import InteractiveAttentionProbeFigure, pick_most_free_cuda_gpu
from utils.symmetry_utils import resolve_symmetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive live grain-attention probe")
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    parser.add_argument("--config", type=str, default="config_new.json", help="Config file inside exp_dir.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename or absolute path.")
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"], help="Dataset split.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Dataset sample index.")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=_PRE_ARGS.gpu_ids,
        help="Optional CUDA_VISIBLE_DEVICES value. By default the most-free GPU is selected.",
    )
    parser.add_argument("--backend", type=str, default=_PRE_ARGS.backend, help="Matplotlib interactive backend, e.g. TkAgg or QtAgg.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for initial/random probe selection.")
    parser.add_argument("--init_y", type=int, default=None, help="Optional initial HR y-coordinate.")
    parser.add_argument("--init_x", type=int, default=None, help="Optional initial HR x-coordinate.")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional directory for saved interactive snapshots.")
    return parser.parse_args()


def _resolve_loader_flags(cfg, model_cls) -> tuple[bool, float, bool]:
    forward_sr_params_cls = inspect.signature(model_cls.forward_sr).parameters
    model_supports_lr_boundary = "lr_boundary_map" in forward_sr_params_cls
    model_requires_lr_boundary = (
        model_supports_lr_boundary
        and forward_sr_params_cls["lr_boundary_map"].default is inspect._empty
    )
    feature_upsampler_type = str(getattr(cfg, "feature_upsampler_type", "shifted_bilinear")).strip().lower()
    use_lr_boundary_map = bool(getattr(cfg, "use_lr_boundary_map", model_supports_lr_boundary))
    if feature_upsampler_type == "grain_attention":
        use_lr_boundary_map = True
    if model_requires_lr_boundary:
        use_lr_boundary_map = True
    lr_boundary_angle_deg = float(getattr(cfg, "lr_boundary_angle_deg", 5.0))
    lr_boundary_mark_both_sides = bool(getattr(cfg, "lr_boundary_mark_both_sides", True))
    return use_lr_boundary_map, lr_boundary_angle_deg, lr_boundary_mark_both_sides


def _default_out_dir(exp_dir: Path, split: str, sample_idx: int) -> Path:
    return exp_dir / "analysis" / "interactive_attention_probe" / f"{str(split).lower()}_sample{int(sample_idx):04d}"


def main() -> None:
    args = parse_args()
    selected_gpu: int | None = None
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        print(f"Using requested GPU(s): {args.gpu_ids}")
    elif torch.cuda.is_available():
        selected_gpu = pick_most_free_cuda_gpu()
        if selected_gpu is not None:
            print(f"Auto-selected most-free GPU: {selected_gpu}")

    backend_name = str(matplotlib.get_backend()).lower()
    interactive_backends = {
        "tkagg",
        "qtagg",
        "qt5agg",
        "webagg",
        "nbagg",
        "gtk3agg",
        "wxagg",
        "macosx",
    }
    if backend_name not in interactive_backends:
        raise RuntimeError(
            "Matplotlib is using a non-interactive backend. "
            f"Detected DISPLAY={os.environ.get('DISPLAY')!r}. "
            "Try running with --backend TkAgg or --backend WebAgg."
        )

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{selected_gpu}" if selected_gpu is not None else "cuda")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{selected_gpu}" if selected_gpu is not None else "cuda")
        else:
            device = torch.device("cpu")

    exp_dir = Path(args.exp_dir).resolve()
    config_path = exp_dir / args.config
    sample_idx = max(0, int(args.sample_idx))
    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else _default_out_dir(exp_dir, args.split, sample_idx)

    cfg = load_and_prepare_config(config_path, exp_dir / "logs" / "interactive_probe_run_config.json")
    checkpoint_path = _resolve_checkpoint(cfg, exp_dir, args.checkpoint)
    model_cls, _, _ = _resolve_model_class(cfg)
    use_lr_boundary_map, lr_boundary_angle_deg, lr_boundary_mark_both_sides = _resolve_loader_flags(cfg, model_cls)

    loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split=str(args.split).capitalize(),
        batch_size=1,
        num_workers=0,
        preload=False,
        preload_torch=False,
        pin_memory=False,
        shuffle=False,
        take_first=sample_idx + 1,
        seed=int(getattr(cfg, "seed", 42)),
        return_lr_boundary_map=use_lr_boundary_map,
        lr_boundary_angle_deg=lr_boundary_angle_deg,
        lr_boundary_mark_both_sides=lr_boundary_mark_both_sides,
    )

    model = _load_model_from_checkpoint(cfg, checkpoint_path, device=device)
    model.eval()

    selected_batch = None
    for idx, batch in enumerate(loader):
        if idx == sample_idx:
            selected_batch = batch
            break
    if selected_batch is None:
        raise IndexError(f"sample_idx={sample_idx} is out of range for split={args.split!r}")

    lr_batch, _, lr_boundary_batch = _unpack_batch(selected_batch)
    lr = lr_batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
    lr_boundary = None
    if lr_boundary_batch is not None:
        lr_boundary = lr_boundary_batch[0].to(device=device, dtype=torch.float32, non_blocking=True)

    lr_flat, lr_shape = _flatten_quat_chw(lr)
    forward_kwargs = {
        "lr_shape": lr_shape,
        "normalize_input": True,
        "return_aux": True,
        "return_probe": False,
    }
    if lr_boundary is not None:
        forward_kwargs["lr_boundary_map"] = lr_boundary

    with torch.enable_grad():
        sr_flat, aux = model.forward_sr(lr_flat, **forward_kwargs)

    if str(aux.get("feature_upsampler_type", "")).strip().lower() != "grain_attention":
        raise RuntimeError("Interactive attention probe requires feature_upsampler_type='grain_attention'.")

    boundary_prob_hr = aux.get("boundary_prob_hr")
    if not isinstance(boundary_prob_hr, torch.Tensor):
        raise RuntimeError("Could not infer HR output shape from aux['boundary_prob_hr'].")
    Hr, Wr = int(boundary_prob_hr.shape[-2]), int(boundary_prob_hr.shape[-1])
    sr_hwc = sr_flat.reshape(Hr, Wr, 4).detach().cpu()

    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))
    viewer = InteractiveAttentionProbeFigure(
        model_obj=model,
        aux=aux,
        sr_quat_hwc=sr_hwc,
        sym_class=sym_class,
        sample_index=0,
        out_dir=out_dir,
        seed=int(args.seed),
    )

    initial_point = None
    if args.init_y is not None and args.init_x is not None:
        initial_point = (int(args.init_y), int(args.init_x))

    print(
        "Interactive attention probe ready.\n"
        "Click on the SR image to recompute the live trace for that HR pixel.\n"
        "Keys: b=random boundary, n=random non-boundary, s=save snapshot, q=close."
    )
    viewer.show(initial_point=initial_point, block=True)


if __name__ == "__main__":
    main()
