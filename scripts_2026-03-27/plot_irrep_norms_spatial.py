#!/usr/bin/env python
"""Plot per-L irrep norm maps and individual irrep-copy maps for model stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from visualization.model_stage_walkthrough import (
    collect_stage_records,
    load_lr_input,
    load_model_from_experiment,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot per-L irrep norms spatially and individual irrep-copy maps "
            "for one traced sample."
        )
    )
    p.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    p.add_argument("--config", type=str, default="config.json", help="Config filename in exp_dir.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename in exp_dir/checkpoints or absolute path.",
    )
    p.add_argument("--device", type=str, default=None, help="Torch device override.")
    p.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"])
    p.add_argument("--sample_offset", type=int, default=0)
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--lr_npy", type=str, default=None)
    p.add_argument(
        "--crop_hw",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Optional top-left LR crop size.",
    )
    p.add_argument(
        "--stages",
        type=str,
        default="encode_a1_lr,conv_lr1_output,conv_lr2_output,upsample_output,conv_hr1_output,attention_output",
        help="Comma-separated stage names or 'all'.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <exp_dir>/irrep_norm_plots/<sample_label>",
    )
    p.add_argument(
        "--max_copy_plots",
        type=int,
        default=0,
        help="If >0, limit number of individual irrep-copy figures per stage.",
    )
    p.add_argument(
        "--plot_components",
        action="store_true",
        help="Also save component maps for each irrep copy.",
    )
    return p.parse_args()


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def _reshape_stage_grid(stage_tensor: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    h, w = shape
    if stage_tensor.ndim != 2:
        raise ValueError(f"Expected stage tensor rank-2 (N,C), got {tuple(stage_tensor.shape)}")
    if stage_tensor.shape[0] != h * w:
        raise ValueError(f"Expected N={h*w}, got N={stage_tensor.shape[0]}")
    return stage_tensor.detach().cpu().reshape(h, w, -1)


def _decompose_irrep_maps(
    stage_grid: torch.Tensor,
    irreps,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    """
    Returns:
      per_l_mean: dict[label -> (H,W)] where label is like l2e
      copies: list of dicts:
        {
          "label": str,
          "l_label": str,
          "norm_map": (H,W),
          "components": (H,W,d),
          "copy_idx": int,
          "d": int
        }
    """
    h, w, c = stage_grid.shape
    start = 0
    per_l_lists: dict[str, list[torch.Tensor]] = {}
    copies: list[dict[str, object]] = []

    for mul, ir in irreps:
        mul_i = int(mul)
        d_i = int(ir.dim)
        l_i = int(ir.l)
        p_i = "e" if int(ir.p) == 1 else "o"
        l_label = f"l{l_i}{p_i}"
        n = mul_i * d_i
        block = stage_grid[..., start : start + n].reshape(h, w, mul_i, d_i)
        norms = torch.sqrt((block**2).sum(dim=-1).clamp_min(1e-12))  # (H,W,mul)

        for copy_idx in range(mul_i):
            norm_map = norms[..., copy_idx]
            components = block[..., copy_idx, :]
            per_l_lists.setdefault(l_label, []).append(norm_map)
            copies.append(
                {
                    "label": f"{l_label}_copy{copy_idx}",
                    "l_label": l_label,
                    "norm_map": norm_map,
                    "components": components,
                    "copy_idx": copy_idx,
                    "d": d_i,
                }
            )

        start += n

    if start != c:
        raise RuntimeError(f"Irrep partition mismatch: consumed {start}, expected {c}")

    per_l_mean = {k: torch.stack(v, dim=0).mean(dim=0) for k, v in per_l_lists.items()}
    return per_l_mean, copies


def _plot_map_dict(map_dict: dict[str, torch.Tensor], title: str, out_png: Path) -> None:
    labels = sorted(map_dict.keys())
    n = len(labels)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.6 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        label = labels[i]
        arr = map_dict[label].numpy()
        im = ax.imshow(arr, cmap="magma", interpolation="nearest")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_component_grid(components: torch.Tensor, title: str, out_png: Path) -> None:
    d = int(components.shape[-1])
    n_cols = min(5, d)
    n_rows = (d + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.2 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i >= d:
            ax.axis("off")
            continue
        arr = components[..., i].numpy()
        vmax = float(np.max(np.abs(arr)))
        if vmax < 1e-12:
            vmax = 1e-12
        im = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"m={i}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    model, cfg, checkpoint_path = load_model_from_experiment(
        exp_dir,
        config_name=args.config,
        checkpoint_name=args.checkpoint,
        device=args.device,
    )
    lr_arr, sample_label = load_lr_input(
        cfg,
        split=args.split,
        sample_offset=int(args.sample_offset),
        dataset_root=args.dataset_root,
        lr_npy=args.lr_npy,
        crop_hw=None if args.crop_hw is None else (int(args.crop_hw[0]), int(args.crop_hw[1])),
    )

    stages, _ = collect_stage_records(model, lr_arr, plot_max_channels=16)
    stage_map = {s.name: s for s in stages}
    available = [s.name for s in stages if s.irreps is not None]

    requested = [x.strip() for x in str(args.stages).split(",") if x.strip()]
    if len(requested) == 1 and requested[0].lower() == "all":
        selected_names = available
    else:
        selected_names = [name for name in requested if name in stage_map and stage_map[name].irreps is not None]

    if not selected_names:
        raise ValueError(
            "No valid irrep stages selected. "
            f"Available: {available}"
        )

    if args.out_dir is None:
        out_root = exp_dir / "irrep_norm_plots" / sample_label
    else:
        out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path if checkpoint_path is not None else '[random init]'}")
    print(f"Sample: {sample_label}")
    print(f"Stages: {selected_names}")
    print(f"Output: {out_root}")

    index_lines = [
        f"checkpoint: {checkpoint_path if checkpoint_path is not None else '[random init]'}",
        f"sample: {sample_label}",
        f"selected_stages: {selected_names}",
        "",
    ]

    for stage_name in selected_names:
        stage = stage_map[stage_name]
        stage_dir = out_root / _safe(stage_name)
        stage_dir.mkdir(parents=True, exist_ok=True)

        grid = _reshape_stage_grid(stage.tensor, stage.shape)
        per_l_mean, copies = _decompose_irrep_maps(grid, stage.irreps)

        # Per-L aggregated norm maps.
        per_l_png = stage_dir / "per_L_norm_maps.png"
        _plot_map_dict(
            per_l_mean,
            title=f"{stage_name}: per-L mean copy norms",
            out_png=per_l_png,
        )
        np.savez(stage_dir / "per_L_norm_maps.npz", **{k: v.numpy() for k, v in per_l_mean.items()})

        # Individual irrep-copy norm maps.
        copy_map = {str(c["label"]): c["norm_map"] for c in copies}
        _plot_map_dict(
            copy_map,
            title=f"{stage_name}: individual irrep-copy norms",
            out_png=stage_dir / "individual_irrep_copy_norms.png",
        )
        np.savez(stage_dir / "individual_irrep_copy_norms.npz", **{k: v.numpy() for k, v in copy_map.items()})

        # Optional per-copy component maps.
        if bool(args.plot_components):
            comp_dir = stage_dir / "components"
            max_copy_plots = int(args.max_copy_plots)
            for i, rec in enumerate(copies):
                if max_copy_plots > 0 and i >= max_copy_plots:
                    break
                label = str(rec["label"])
                comps = rec["components"]
                _plot_component_grid(
                    comps,
                    title=f"{stage_name}: {label} components",
                    out_png=comp_dir / f"{_safe(label)}_components.png",
                )
                np.save(comp_dir / f"{_safe(label)}_components.npy", comps.numpy())

        index_lines.append(f"{stage_name}:")
        index_lines.append(f"  per_L_norm_maps.png")
        index_lines.append(f"  individual_irrep_copy_norms.png")
        if bool(args.plot_components):
            index_lines.append(f"  components/*.png")
        index_lines.append("")

        print(f"[ok] {stage_name}: wrote {stage_dir}")

    (out_root / "index.txt").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"Done. Index: {out_root / 'index.txt'}")


if __name__ == "__main__":
    main()

