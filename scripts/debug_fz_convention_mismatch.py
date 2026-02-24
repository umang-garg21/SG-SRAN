import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_model
from models.autoencoder import FCCAutoEncoder
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from utils.quat_ops import enforce_hemisphere, reduce_to_fz_min_angle
from utils.symmetry_utils import resolve_symmetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug mismatch between model.reduce_to_fz and quat_ops.reduce_to_fz_min_angle"
    )
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint to load: best, last, or explicit path",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=50)
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
    elif isinstance(blob, dict) and "decoder_state_dict" in blob:
        if not hasattr(model, "decoder"):
            raise ValueError("Checkpoint has decoder_state_dict but model has no decoder")
        model.decoder.load_state_dict(blob["decoder_state_dict"], strict=False)
        return
    elif isinstance(blob, dict):
        state_dict = blob
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        print(f"[debug] Unexpected keys (up to 10): {load_result.unexpected_keys[:10]}")
    if load_result.missing_keys:
        print(f"[debug] Missing keys (up to 10): {load_result.missing_keys[:10]}")


def _grab_sample_hr(loader, sample_idx: int) -> torch.Tensor:
    seen = 0
    for batch in loader:
        _, hr = batch
        bsz = int(hr.shape[0])
        if sample_idx < seen + bsz:
            return hr[sample_idx - seen]
        seen += bsz
    raise IndexError(f"sample_idx={sample_idx} out of range")


def _misorientation_deg(model: torch.nn.Module, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    delta = model.quat_mul(q1, FCCAutoEncoder._quat_conjugate(q2))
    delta = delta / torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)
    w_abs = delta[:, 0].abs().clamp(max=1.0)
    return 2.0 * torch.acos(w_abs) * 180.0 / torch.pi


def _symmetry_min_misorientation_deg(
    model: torch.nn.Module,
    q_ref: torch.Tensor,
    q_tgt: torch.Tensor,
) -> torch.Tensor:
    syms = model.physics.fcc_syms.to(q_ref.device)
    n = q_ref.shape[0]
    m = syms.shape[0]

    q_ref_exp = q_ref.unsqueeze(1).expand(-1, m, -1)
    sym_exp = syms.unsqueeze(0).expand(n, -1, -1)
    family = model.quat_mul(
        q_ref_exp.reshape(-1, 4),
        sym_exp.reshape(-1, 4),
    ).reshape(n, m, 4)
    family = family / torch.norm(family, dim=-1, keepdim=True).clamp_min(1e-12)

    q_tgt_exp = q_tgt.unsqueeze(1).expand(-1, m, -1)
    delta = model.quat_mul(
        family.reshape(-1, 4),
        FCCAutoEncoder._quat_conjugate(q_tgt_exp.reshape(-1, 4)),
    ).reshape(n, m, 4)
    delta = delta / torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)

    w_abs = delta[..., 0].abs().clamp(max=1.0)
    mis = 2.0 * torch.acos(w_abs) * 180.0 / torch.pi
    return mis.min(dim=1).values


def _stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def _save_debug_plots(
    out_dir: Path,
    h: int,
    w: int,
    mis_model_vs_np_direct: torch.Tensor,
    mis_model_vs_np_sym: torch.Tensor,
    model_idx: torch.Tensor,
    np_idx_t: torch.Tensor,
    top_rows: list[dict[str, Any]],
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[debug] Skipping plots (matplotlib unavailable): {exc}")
        return {}

    out: dict[str, str] = {}

    direct_map = mis_model_vs_np_direct.detach().cpu().numpy().reshape(h, w)
    sym_map = mis_model_vs_np_sym.detach().cpu().numpy().reshape(h, w)
    model_idx_map = model_idx.detach().cpu().numpy().reshape(h, w)
    np_idx_map = np_idx_t.detach().cpu().numpy().reshape(h, w)

    vmax = float(np.percentile(direct_map, 99.5)) if direct_map.size > 0 else 1.0
    vmax = max(vmax, 1e-6)

    heat_png = out_dir / "mis_model_vs_np_direct_deg.png"
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(direct_map, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title("model.reduce_to_fz vs reduce_to_fz_min_angle (deg)")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(heat_png, dpi=220)
    plt.close(fig)
    out["mis_model_vs_np_direct_deg_png"] = str(heat_png)

    sym_png = out_dir / "mis_model_vs_np_sym_min_deg.png"
    fig2, ax2 = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im2 = ax2.imshow(sym_map, cmap="viridis")
    ax2.set_title("Symmetry-min mismatch (deg)")
    ax2.set_xlabel("col")
    ax2.set_ylabel("row")
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig2.savefig(sym_png, dpi=220)
    plt.close(fig2)
    out["mis_model_vs_np_sym_min_deg_png"] = str(sym_png)

    mask_png = out_dir / "mismatch_masks.png"
    fig3, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    thresholds = [1.0, 10.0, 45.0]
    for ax_i, thr in zip(axes, thresholds):
        mask = (direct_map > thr).astype(np.float32)
        ax_i.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax_i.set_title(f"Mismatch > {thr:g}°")
        ax_i.set_axis_off()
    fig3.savefig(mask_png, dpi=220)
    plt.close(fig3)
    out["mismatch_masks_png"] = str(mask_png)

    idx_png = out_dir / "symmetry_index_difference.png"
    fig4, ax4 = plt.subplots(figsize=(7, 6), constrained_layout=True)
    idx_diff = (model_idx_map != np_idx_map).astype(np.float32)
    ax4.imshow(idx_diff, cmap="gray", vmin=0, vmax=1)
    ax4.set_title("Symmetry index mismatch map (model != np)")
    ax4.set_axis_off()
    fig4.savefig(idx_png, dpi=220)
    plt.close(fig4)
    out["symmetry_index_mismatch_png"] = str(idx_png)

    top_png = out_dir / "top_mismatch_overlay.png"
    fig5, ax5 = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax5.imshow(direct_map, cmap="magma", vmin=0.0, vmax=vmax)
    rows = [r["row"] for r in top_rows[:25]]
    cols = [r["col"] for r in top_rows[:25]]
    if rows and cols:
        ax5.scatter(cols, rows, s=18, c="cyan", edgecolors="black", linewidths=0.4)
    ax5.set_title("Top mismatch pixels (first 25)")
    ax5.set_xlabel("col")
    ax5.set_ylabel("row")
    fig5.savefig(top_png, dpi=220)
    plt.close(fig5)
    out["top_mismatch_overlay_png"] = str(top_png)

    return out


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "visualizations" / "fz_convention_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_and_prepare_config(config_path, exp_dir / "logs" / "run_config.json")
    if args.device is not None:
        cfg.device = args.device

    device = torch.device(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg).to(device)
    ckpt_path = _resolve_checkpoint_path(exp_dir, args.checkpoint)
    _load_model_state(model, ckpt_path, device)
    model.eval()

    if not hasattr(model, "reduce_to_fz"):
        raise ValueError("Model does not expose reduce_to_fz; cannot compare conventions.")

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
    q_in = hr_sample.permute(1, 2, 0).reshape(-1, 4).to(device)
    q_in = q_in / torch.norm(q_in, dim=1, keepdim=True).clamp_min(1e-12)

    with torch.no_grad():
        q_dec = model(q_in, normalize_input=True)
        q_dec = q_dec / torch.norm(q_dec, dim=1, keepdim=True).clamp_min(1e-12)

        q_model_fz, model_idx = model.reduce_to_fz(q_dec, return_op_map=True)
        q_model_fz = q_model_fz / torch.norm(q_model_fz, dim=1, keepdim=True).clamp_min(1e-12)

        sym = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))
        q_np, np_idx = reduce_to_fz_min_angle(
            q_dec.detach().cpu().numpy(),
            sym=sym,
            normalize=True,
            hemisphere=True,
            return_op_map=True,
        )
        q_np = enforce_hemisphere(q_np, scalar_first=True)
        q_np_fz = torch.from_numpy(q_np).to(device=device, dtype=torch.float32)
        q_np_fz = q_np_fz / torch.norm(q_np_fz, dim=1, keepdim=True).clamp_min(1e-12)
        np_idx_t = torch.from_numpy(np_idx.reshape(-1)).to(device=device, dtype=torch.int64)

        mis_model_vs_np_direct = _misorientation_deg(model, q_model_fz, q_np_fz)
        mis_model_vs_np_sym = _symmetry_min_misorientation_deg(model, q_model_fz, q_np_fz)

        mis_model_vs_in_direct = _misorientation_deg(model, q_model_fz, q_in)
        mis_model_vs_in_sym = _symmetry_min_misorientation_deg(model, q_model_fz, q_in)

        mis_np_vs_in_direct = _misorientation_deg(model, q_np_fz, q_in)
        mis_np_vs_in_sym = _symmetry_min_misorientation_deg(model, q_np_fz, q_in)

    thresholds = [1.0, 5.0, 10.0, 45.0, 80.0]
    counts_model_vs_np = {str(t): int((mis_model_vs_np_direct > t).sum().item()) for t in thresholds}

    order = torch.argsort(mis_model_vs_np_direct, descending=True)
    top_k = min(int(args.top_k), int(order.numel()))
    top_rows: list[dict[str, Any]] = []
    for rank, flat_idx in enumerate(order[:top_k].tolist(), start=1):
        r = flat_idx // w
        c = flat_idx % w
        top_rows.append(
            {
                "rank": rank,
                "row": int(r),
                "col": int(c),
                "mis_model_vs_np_direct_deg": float(mis_model_vs_np_direct[flat_idx].item()),
                "mis_model_vs_np_sym_min_deg": float(mis_model_vs_np_sym[flat_idx].item()),
                "mis_model_vs_input_direct_deg": float(mis_model_vs_in_direct[flat_idx].item()),
                "mis_np_vs_input_direct_deg": float(mis_np_vs_in_direct[flat_idx].item()),
                "model_sym_idx": int(model_idx[flat_idx].item()),
                "np_sym_idx": int(np_idx_t[flat_idx].item()),
                "w_model": float(q_model_fz[flat_idx, 0].item()),
                "w_np": float(q_np_fz[flat_idx, 0].item()),
            }
        )

    report = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "sample_idx": int(args.sample_idx),
        "shape_hw": [h, w],
        "metrics": {
            "model_vs_np_direct_deg": _stats(mis_model_vs_np_direct),
            "model_vs_np_sym_min_deg": _stats(mis_model_vs_np_sym),
            "model_vs_input_direct_deg": _stats(mis_model_vs_in_direct),
            "model_vs_input_sym_min_deg": _stats(mis_model_vs_in_sym),
            "np_vs_input_direct_deg": _stats(mis_np_vs_in_direct),
            "np_vs_input_sym_min_deg": _stats(mis_np_vs_in_sym),
        },
        "counts_model_vs_np_direct": counts_model_vs_np,
        "negative_scalar_counts": {
            "model_fz_w_lt_0": int((q_model_fz[:, 0] < 0).sum().item()),
            "np_fz_w_lt_0": int((q_np_fz[:, 0] < 0).sum().item()),
        },
        "symmetry_hist": {
            "model_reduce_to_fz": torch.bincount(model_idx.cpu(), minlength=24).tolist(),
            "np_reduce_to_fz_min_angle": torch.bincount(np_idx_t.cpu(), minlength=24).tolist(),
        },
        "top_mismatch_pixels": top_rows,
    }

    plot_paths = _save_debug_plots(
        out_dir=out_dir,
        h=h,
        w=w,
        mis_model_vs_np_direct=mis_model_vs_np_direct,
        mis_model_vs_np_sym=mis_model_vs_np_sym,
        model_idx=model_idx,
        np_idx_t=np_idx_t,
        top_rows=top_rows,
    )
    if plot_paths:
        report["plots"] = plot_paths

    out_json = out_dir / "fz_convention_debug_report.json"
    out_txt = out_dir / "fz_convention_debug_top_pixels.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Top mismatch pixels: model.reduce_to_fz vs reduce_to_fz_min_angle\n")
        f.write(f"sample_idx={args.sample_idx}, shape={h}x{w}\n\n")
        f.write("rank row col mis_model_np_direct mis_model_np_symmin mis_model_input mis_np_input model_idx np_idx w_model w_np\n")
        for item in top_rows:
            f.write(
                f"{item['rank']:4d} {item['row']:4d} {item['col']:4d} "
                f"{item['mis_model_vs_np_direct_deg']:18.6f} "
                f"{item['mis_model_vs_np_sym_min_deg']:20.6f} "
                f"{item['mis_model_vs_input_direct_deg']:16.6f} "
                f"{item['mis_np_vs_input_direct_deg']:12.6f} "
                f"{item['model_sym_idx']:9d} {item['np_sym_idx']:6d} "
                f"{item['w_model']:8.5f} {item['w_np']:8.5f}\n"
            )

    print(f"Saved: {out_json}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
