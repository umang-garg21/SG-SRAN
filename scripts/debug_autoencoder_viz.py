import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from models.autoencoder import FCCAutoEncoder
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from utils.symmetry_utils import resolve_symmetry
from visualization.visualize_sr_results import render_input_output_side_by_side


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug autoencoder reconstruction and visualize input/output IPF maps"
    )
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config filename inside exp_dir (default: config.json)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last",
        help="Checkpoint to load: 'last', 'best', or explicit checkpoint path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to sample from",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index within selected split",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: exp_dir/visualizations/debug)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., cpu, cuda:0). Default follows config/runtime",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(exp_dir: Path, checkpoint_arg: str) -> Path:
    if checkpoint_arg == "last":
        return exp_dir / "checkpoints" / "last_checkpoint.pt"
    if checkpoint_arg == "best":
        return exp_dir / "checkpoints" / "best_model.pt"
    return Path(checkpoint_arg)


def _load_model_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        blob = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(str(ckpt_path), map_location=device)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state_dict = blob["model_state_dict"]
    elif isinstance(blob, dict):
        state_dict = blob
    else:
        raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")

    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    if missing_keys and len(unexpected_keys) > 0:
        core_prefix = "core."
        if all(k.startswith(core_prefix) for k in state_dict.keys()):
            stripped = {k[len(core_prefix):]: v for k, v in state_dict.items()}
            load_result = model.load_state_dict(stripped, strict=False)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)

    if len(unexpected_keys) > 0:
        print(f"[debug] Unexpected checkpoint keys (showing up to 10): {unexpected_keys[:10]}")
    if len(missing_keys) > 0:
        print(f"[debug] Missing model keys (showing up to 10): {missing_keys[:10]}")


def _grab_sample_hr(loader, sample_idx: int) -> torch.Tensor:
    if sample_idx < 0:
        raise ValueError("sample_idx must be >= 0")

    seen = 0
    for batch in loader:
        _, hr = batch
        bsz = int(hr.shape[0])
        if sample_idx < seen + bsz:
            local_idx = sample_idx - seen
            return hr[local_idx]
        seen += bsz

    raise IndexError(f"sample_idx={sample_idx} is out of range for selected split")


def _misorientation_deg(
    model: torch.nn.Module,
    q_reconstructed: torch.Tensor,
    q_truth: torch.Tensor,
) -> torch.Tensor:
    if hasattr(model, "_to_active_convention"):
        qA = model._to_active_convention(q_reconstructed)
        qB = model._to_active_convention(q_truth)
    else:
        qA = q_reconstructed
        qB = q_truth

    delta = model.quat_mul(qA, FCCAutoEncoder._quat_conjugate(qB))
    delta = delta / torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)
    w_abs = delta[:, 0].abs().clamp(max=1.0)
    return 2.0 * torch.acos(w_abs) * 180.0 / torch.pi


def _summary_stats(values: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def _save_quaternion_visualizations(
    q_input_flat: torch.Tensor,
    q_output_flat: torch.Tensor,
    h: int,
    w: int,
    out_dir: Path,
    prefix: str,
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[debug] Skipping quaternion plots (matplotlib unavailable): {exc}")
        return {}

    q_in_np = q_input_flat.detach().cpu().numpy()
    q_out_np = q_output_flat.detach().cpu().numpy()

    q_in_img = q_in_np.reshape(h, w, 4)
    q_out_img = q_out_np.reshape(h, w, 4)

    comp_names = ["w", "x", "y", "z"]

    components_png = out_dir / f"{prefix}_quat_components.png"
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for i, comp in enumerate(comp_names):
        vmin = float(min(q_in_img[..., i].min(), q_out_img[..., i].min()))
        vmax = float(max(q_in_img[..., i].max(), q_out_img[..., i].max()))

        im0 = axes[0, i].imshow(q_in_img[..., i], cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"Input {comp}")
        axes[0, i].axis("off")
        fig.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

        im1 = axes[1, i].imshow(q_out_img[..., i], cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"Output {comp}")
        axes[1, i].axis("off")
        fig.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

    fig.suptitle(f"Quaternion Components: {prefix}")
    fig.savefig(components_png, dpi=220)
    plt.close(fig)

    scatter_png = out_dir / f"{prefix}_quat_xyz_scatter.png"
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    pairs = [(1, 2, "x vs y"), (1, 3, "x vs z"), (2, 3, "y vs z")]
    for ax, (ix, iy, ttl) in zip(axes2, pairs):
        ax.scatter(q_in_np[:, ix], q_in_np[:, iy], s=8, alpha=0.5, label="input")
        ax.scatter(q_out_np[:, ix], q_out_np[:, iy], s=8, alpha=0.5, label="output")
        ax.set_xlabel(comp_names[ix])
        ax.set_ylabel(comp_names[iy])
        ax.set_title(ttl)
        ax.grid(alpha=0.2)
    axes2[0].legend(loc="best")
    fig2.suptitle(f"Quaternion Vector-Part Scatter: {prefix}")
    fig2.savefig(scatter_png, dpi=220)
    plt.close(fig2)

    return {
        "components_png": str(components_png),
        "scatter_png": str(scatter_png),
    }


def main() -> None:
    args = parse_args()

    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    run_cfg_path = exp_dir / "logs" / "run_config.json"
    cfg = load_and_prepare_config(config_path, run_cfg_path)

    if args.device is not None:
        cfg.device = args.device

    device = torch.device(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    model_type = str(getattr(cfg, "model_type", "")).lower()
    if model_type not in {"fcc_autoencoder", "fcc_autoencoder_learnable_decoder"}:
        raise ValueError(
            f"This debug script supports autoencoder model types only. Got model_type='{model_type}'."
        )

    model = build_model(cfg).to(device)
    ckpt_path = _resolve_checkpoint_path(exp_dir, args.checkpoint)
    _load_model_state(model, ckpt_path, device)
    model.eval()

    loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split=args.split.capitalize(),
        batch_size=int(getattr(cfg, "batch_size", 8)),
        num_workers=int(getattr(cfg, "num_workers", 0)),
        preload=bool(getattr(cfg, "preload", True)),
        preload_torch=bool(getattr(cfg, "preload_torch", True)),
        pin_memory=bool(getattr(cfg, "pin_memory", True)),
        shuffle=False,
        seed=int(getattr(cfg, "seed", 42)),
    )

    hr_sample = _grab_sample_hr(loader, args.sample_idx)
    h, w = int(hr_sample.shape[1]), int(hr_sample.shape[2])
    q_flat = hr_sample.permute(1, 2, 0).reshape(-1, 4).to(device)
    q_flat = q_flat / torch.norm(q_flat, dim=1, keepdim=True).clamp_min(1e-12)

    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "visualizations" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))

    q_decoded_default = model(q_flat, normalize_input=True)
    q_reconstructed_default, errors_default, symmetry_idx_default = model.match_closest_symmetry(
        q_decoded_default,
        q_flat,
    )
    mis_deg_default = _misorientation_deg(model, q_reconstructed_default, q_flat)

    q_in = q_flat.reshape(h, w, 4).detach().cpu().numpy()
    q_out_default = q_reconstructed_default.reshape(h, w, 4).detach().cpu().numpy()

    default_png = out_dir / "input_output_default.png"
    render_input_output_side_by_side(
        input_q_arr=q_in,
        output_q_arr=q_out_default,
        sym_class=sym_class,
        out_png=str(default_png),
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=False,
        dpi=300,
    )

    default_quat_plots = _save_quaternion_visualizations(
        q_input_flat=q_flat,
        q_output_flat=q_reconstructed_default,
        h=h,
        w=w,
        out_dir=out_dir,
        prefix="default",
    )

    report: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "sample_idx": int(args.sample_idx),
        "model_type": model_type,
        "default": {
            "errors_rad": _summary_stats(errors_default),
            "misorientation_deg": _summary_stats(mis_deg_default),
            "output_png": str(default_png),
            "quat_plots": default_quat_plots,
            "symmetry_index_hist": torch.bincount(symmetry_idx_default.cpu(), minlength=24).tolist(),
        },
    }

    report_path = out_dir / "debug_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[debug_autoencoder_viz] Saved outputs:")
    print(f"  - {default_png}")
    if "default" in report and report["default"].get("quat_plots"):
        dp = report["default"]["quat_plots"]
        if "components_png" in dp:
            print(f"  - {dp['components_png']}")
        if "scatter_png" in dp:
            print(f"  - {dp['scatter_png']}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
