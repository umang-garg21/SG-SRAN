#!/usr/bin/env python
"""Visualize SDF SR model stages: irreps, boundary evidence/maps, and IPF outputs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from e3nn.o3 import Irreps
from orix import plot as orix_plot
from orix.quaternion import Orientation
from orix.vector import Vector3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.symmetry_utils import resolve_symmetry
from visualization.model_stage_walkthrough import load_lr_input, load_model_from_experiment


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def _ensure_hwc_quaternion_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 quaternion image, got {arr.shape}")
    if arr.shape[-1] == 4:
        out = arr
    elif arr.shape[0] == 4:
        out = np.moveaxis(arr, 0, -1)
    else:
        raise ValueError(f"Could not find quaternion axis in shape {arr.shape}")
    return out.astype(np.float32, copy=False)


def _load_lr_pairs(dataset_root: Path, split: str) -> list[tuple[tuple[str, int], Path, Path]]:
    name_re = re.compile(
        r"^(?P<ds>.+)_(?P<split>train|val|test)_(?P<which>hr|lr)_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
        re.IGNORECASE,
    )

    def _pair_key(path: Path) -> tuple[str, int] | None:
        m = name_re.match(path.name)
        if m is None:
            return None
        return m.group("axis").lower(), int(m.group("id"))

    split_dir = dataset_root / str(split)
    lr_dir = split_dir / "LR_Data"
    hr_dir = split_dir / "HR_Data"
    if not lr_dir.exists() or not hr_dir.exists():
        raise FileNotFoundError(f"Missing LR/HR split dirs under: {split_dir}")

    lr_map: dict[tuple[str, int], Path] = {}
    for fp in sorted(lr_dir.glob("*.npy")):
        key = _pair_key(fp)
        if key is not None:
            lr_map[key] = fp

    hr_map: dict[tuple[str, int], Path] = {}
    for fp in sorted(hr_dir.glob("*.npy")):
        key = _pair_key(fp)
        if key is not None:
            hr_map[key] = fp

    common = sorted(set(lr_map).intersection(hr_map))
    return [(key, lr_map[key], hr_map[key]) for key in common]


def _load_hr_input(
    cfg: Any,
    *,
    split: str,
    sample_offset: int,
    dataset_root: str | None,
    lr_npy: str | None,
    hr_npy: str | None,
    crop_hw: tuple[int, int] | None,
    scale: int,
) -> np.ndarray | None:
    if hr_npy is not None:
        hr = _ensure_hwc_quaternion_image(np.load(hr_npy))
    elif lr_npy is not None:
        return None
    else:
        root = Path(dataset_root or str(getattr(cfg, "dataset_root", ""))).resolve()
        if not str(root):
            return None
        pairs = _load_lr_pairs(root, split)
        if sample_offset < 0 or sample_offset >= len(pairs):
            return None
        _, _, hr_fp = pairs[sample_offset]
        hr = _ensure_hwc_quaternion_image(np.load(hr_fp))

    if crop_hw is not None:
        ch, cw = int(crop_hw[0]), int(crop_hw[1])
        hr = hr[: ch * int(scale), : cw * int(scale), :]
    return hr


def _reshape_feature_grid(features: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    h, w = shape
    if features.ndim != 2:
        raise ValueError(f"Expected (N,C) feature tensor, got {tuple(features.shape)}")
    n, _ = features.shape
    if n != h * w:
        raise ValueError(f"Expected N={h*w}, got N={n}")
    return features.detach().cpu().reshape(h, w, -1)


def _block_labels(irreps: Irreps | str) -> list[str]:
    irr = Irreps(irreps)
    labels: list[str] = []
    seen: dict[str, int] = {}
    for mul_ir in irr:
        ir = mul_ir.ir
        label = f"l{int(ir.l)}{'e' if int(ir.p) == 1 else 'o'}"
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}_blk{seen[label]}"
        labels.append(label)
    return labels


def _compute_irrep_norm_maps(
    features: torch.Tensor,
    shape: tuple[int, int],
    irreps: Irreps | str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    grid = _reshape_feature_grid(features, shape)
    irr = Irreps(irreps)
    labels = _block_labels(irr)

    by_block: dict[str, np.ndarray] = {}
    by_l_group: dict[str, list[np.ndarray]] = {}
    scalars_0e: dict[str, np.ndarray] = {}

    for sl, mul_ir, label in zip(irr.slices(), irr, labels):
        mul = int(mul_ir.mul)
        d = int(mul_ir.ir.dim)
        l = int(mul_ir.ir.l)
        p_label = "e" if int(mul_ir.ir.p) == 1 else "o"
        l_group = f"l{l}{p_label}"

        blk = grid[..., sl].reshape(shape[0], shape[1], mul, d)
        blk_norm = torch.sqrt((blk * blk).mean(dim=(-1, -2)).clamp_min(1e-12))
        by_block[label] = blk_norm.numpy()
        by_l_group.setdefault(l_group, []).append(blk_norm.numpy())

        if l == 0 and p_label == "e":
            for copy_idx in range(mul):
                scalars_0e[f"{label}_copy{copy_idx}"] = blk[..., copy_idx, 0].numpy()

    by_l = {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in by_l_group.items()}
    return by_block, by_l, scalars_0e


def _tensor_image_channels(x: torch.Tensor) -> np.ndarray:
    t = x.detach().cpu()
    if t.ndim == 4:
        if int(t.shape[0]) != 1:
            raise ValueError(f"Expected batch size 1 for image tensor, got {tuple(t.shape)}")
        t = t.squeeze(0)
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3:
        raise ValueError(f"Expected 2D/3D/4D image tensor, got {tuple(t.shape)}")
    return t.numpy().astype(np.float32, copy=False)


def _first_channel_map(x: torch.Tensor) -> np.ndarray:
    return _tensor_image_channels(x)[0]


def _norm_map(x: torch.Tensor, eps: float = 1e-12) -> np.ndarray:
    chw = _tensor_image_channels(x)
    sq = np.mean(chw * chw, axis=0)
    return np.sqrt(np.maximum(sq, eps))


def _plot_map_grid(
    maps: list[tuple[str, np.ndarray]],
    out_png: Path,
    title: str,
    *,
    cmap: str = "magma",
    ncols: int = 4,
    symmetric: bool = False,
) -> None:
    if not maps:
        return

    n = len(maps)
    ncols = max(1, min(int(ncols), n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        name, arr = maps[i]
        if symmetric:
            vmax = max(float(np.max(np.abs(arr))), 1e-8)
            im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        else:
            im = ax.imshow(arr, cmap=cmap, interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_ipf_rgb_safe(
    q_hwc: np.ndarray,
    sym_class,
    ref_dir: str = "ALL",
) -> np.ndarray | list[np.ndarray]:
    q = _ensure_hwc_quaternion_image(q_hwc)

    ori = Orientation(q)
    ori.symmetry = sym_class
    ckey = orix_plot.IPFColorKeyTSL(sym_class.laue)

    dirs = {
        "X": Vector3d((1, 0, 0)),
        "Y": Vector3d((0, 1, 0)),
        "Z": Vector3d((0, 0, 1)),
    }
    ref = str(ref_dir).upper()
    wanted = ("X", "Y", "Z") if ref == "ALL" else (ref,)

    out: list[np.ndarray] = []
    for key in wanted:
        if key not in dirs:
            raise ValueError(f"Invalid IPF ref_dir={ref_dir!r}; expected X, Y, Z, or ALL.")
        ckey.direction = dirs[key]
        out.append(ckey.orientation2color(ori))

    return out if ref == "ALL" else out[0]


def _plot_lr_ipf_with_pairwise(
    lr_q_hwc: np.ndarray,
    pairwise_norm_lr: np.ndarray,
    sym_class,
    out_png: Path,
) -> None:
    ipf_rgb = _render_ipf_rgb_safe(lr_q_hwc, sym_class=sym_class, ref_dir="ALL")
    if not isinstance(ipf_rgb, list):
        ipf_rgb = [ipf_rgb]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.2))
    dirs = ("X", "Y", "Z")
    for i, dname in enumerate(dirs):
        ax_ipf = axes[2 * i]
        ax_pair = axes[2 * i + 1]

        ax_ipf.imshow(ipf_rgb[i])
        ax_ipf.set_title(f"LR IPF-{dname}")
        ax_ipf.axis("off")

        im = ax_pair.imshow(pairwise_norm_lr, cmap="magma", interpolation="nearest")
        ax_pair.set_title(f"Pairwise Norm ({dname})")
        ax_pair.axis("off")
        fig.colorbar(im, ax=ax_pair, fraction=0.046, pad=0.02)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_ipf_rows(
    *,
    lr_q_hwc: np.ndarray,
    sr_q_hwc: np.ndarray,
    hr_q_hwc: np.ndarray | None,
    sym_class,
    out_png: Path,
) -> None:
    rows: list[tuple[str, list[np.ndarray]]] = []

    lr_rgb = _render_ipf_rgb_safe(lr_q_hwc, sym_class=sym_class, ref_dir="ALL")
    sr_rgb = _render_ipf_rgb_safe(sr_q_hwc, sym_class=sym_class, ref_dir="ALL")
    if not isinstance(lr_rgb, list):
        lr_rgb = [lr_rgb]
    if not isinstance(sr_rgb, list):
        sr_rgb = [sr_rgb]
    rows.append(("LR", lr_rgb))
    rows.append(("SR", sr_rgb))

    if hr_q_hwc is not None:
        hr_rgb = _render_ipf_rgb_safe(hr_q_hwc, sym_class=sym_class, ref_dir="ALL")
        if not isinstance(hr_rgb, list):
            hr_rgb = [hr_rgb]
        rows.append(("HR", hr_rgb))

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(12.5, 4.2 * nrows))
    axes = np.asarray(axes).reshape(nrows, 3)

    for r, (row_name, rgbs) in enumerate(rows):
        for c, dname in enumerate(("X", "Y", "Z")):
            ax = axes[r, c]
            ax.imshow(rgbs[c])
            ax.set_title(f"{row_name} IPF-{dname}")
            ax.axis("off")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _extract_module_build_evidence(model) -> Any:
    module_name = model.__class__.__module__
    mod = __import__(module_name, fromlist=["build_irrep_boundary_evidence"])
    build_fn = getattr(mod, "build_irrep_boundary_evidence", None)
    if build_fn is None:
        raise AttributeError(
            f"Model module {module_name} does not expose build_irrep_boundary_evidence"
        )
    return build_fn


def _collect_sdf_stages(
    model,
    lr_q_hwc: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], torch.Tensor, torch.Tensor, tuple[int, int]]:
    h, w, _ = lr_q_hwc.shape
    lr_shape = (h, w)

    lr_flat = torch.from_numpy(lr_q_hwc.reshape(h * w, 4)).to(
        device=model.device,
        dtype=torch.float32,
    )

    build_evidence = _extract_module_build_evidence(model)

    stage_feats: list[dict[str, Any]] = []
    with torch.no_grad():
        feat = model.encode_a1(lr_flat)
        stage_feats.append({"name": "encode_a1_lr", "feat": feat, "shape": lr_shape})

        if hasattr(model, "lr_blocks"):
            for idx, blk in enumerate(model.lr_blocks, start=1):
                feat = blk(feat, lr_shape)
                stage_feats.append({"name": f"lr_block_{idx:02d}_output", "feat": feat, "shape": lr_shape})
        elif getattr(model, "pre_lr", None) is not None:
            feat = model.pre_lr(feat, lr_shape)
            stage_feats.append({"name": "pre_lr_output", "feat": feat, "shape": lr_shape})

        evidence = build_evidence(
            feat_lr=feat,
            lr_shape=lr_shape,
            irreps_feat=model.irreps_a1,
            offsets=getattr(model, "evidence_offsets", None),
            radius=int(getattr(model, "evidence_radius", 1)),
        )

        boundary_head = getattr(model, "boundary_sdf_head", getattr(model, "sdf_head", None))
        if boundary_head is None:
            raise AttributeError("Model is missing boundary SDF head attribute")

        sdf_out = boundary_head(
            evidence_tensor_lr=evidence["tensor"],
            lr_shape=lr_shape,
            extra_stats=None,
        )

        feat_hr, up_aux = model.sdf_upsample(
            feat_lr=feat,
            lr_shape=lr_shape,
            sdf_out=sdf_out,
            return_aux=True,
        )

        hr_shape = tuple(int(v) for v in sdf_out["hr_shape"])
        stage_feats.append({"name": "sdf_upsample_output", "feat": feat_hr, "shape": hr_shape})

        if hasattr(model, "hr_blocks"):
            for idx, blk in enumerate(model.hr_blocks, start=1):
                feat_hr = blk(
                    feat_hr,
                    hr_shape,
                    guidance=sdf_out["guidance_hr"],
                    boundary_logits=sdf_out["boundary_logits_hr"],
                )
                stage_feats.append({"name": f"hr_block_{idx:02d}_output", "feat": feat_hr, "shape": hr_shape})
        elif getattr(model, "post_hr", None) is not None:
            feat_hr = model.post_hr(
                feat_hr,
                hr_shape,
                guidance=sdf_out["guidance_hr"],
                boundary_logits=sdf_out["boundary_logits_hr"],
            )
            stage_feats.append({"name": "post_hr_output", "feat": feat_hr, "shape": hr_shape})

        boundary_logits_hr_refined = None
        if getattr(model, "refinement_head", None) is not None:
            feat_hr, boundary_logits_hr_refined = model.refinement_head(
                feat_hr,
                hr_shape,
                guidance=sdf_out["guidance_hr"],
                boundary_logits=sdf_out["boundary_logits_hr"],
            )
            stage_feats.append({"name": "refinement_output", "feat": feat_hr, "shape": hr_shape})

    with torch.enable_grad():
        q_sr_flat = model.decode(feat_hr)

    aux: dict[str, Any] = {}
    aux.update(evidence)
    aux.update(sdf_out)
    aux.update(up_aux)
    aux["boundary_logits_hr_refined"] = boundary_logits_hr_refined

    return stage_feats, aux, q_sr_flat.detach(), lr_flat.detach(), lr_shape


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualize SDF SR stages spatially as irrep norms, boundary evidence/maps, and IPF outputs."
        )
    )
    p.add_argument("--exp_dir", required=True, type=str, help="Experiment directory.")
    p.add_argument("--config", type=str, default="config.json", help="Config filename in exp_dir.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint filename under exp_dir/checkpoints or absolute path. "
            "Defaults to best_model.pt when available."
        ),
    )
    p.add_argument("--device", type=str, default=None, help="Torch device override (e.g. cuda:0 or cpu).")
    p.add_argument("--split", type=str, default="Test", choices=["Train", "Val", "Test"])
    p.add_argument("--sample_offset", type=int, default=0)
    p.add_argument("--dataset_root", type=str, default=None, help="Optional dataset root override.")
    p.add_argument("--lr_npy", type=str, default=None, help="Optional direct LR quaternion .npy.")
    p.add_argument("--hr_npy", type=str, default=None, help="Optional direct HR quaternion .npy.")
    p.add_argument(
        "--crop_hw",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Optional top-left LR crop size.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <exp_dir>/sdf_stage_viz/<sample_label>",
    )
    p.add_argument(
        "--max_evidence_maps",
        type=int,
        default=24,
        help="Maximum number of E channels and C channels to plot each.",
    )
    p.add_argument(
        "--max_irrep_maps",
        type=int,
        default=24,
        help="Maximum number of per-block irrep maps to plot for each stage.",
    )
    p.add_argument(
        "--boundary_teacher_thr_deg",
        type=float,
        default=3.0,
        help="Misorientation threshold for teacher boundary map (if model supports it).",
    )
    p.add_argument(
        "--boundary_teacher_connectivity",
        type=int,
        default=4,
        choices=[4, 8],
        help="Connectivity for teacher boundary map (if model supports it).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    model, cfg, ckpt_path = load_model_from_experiment(
        exp_dir,
        config_name=args.config,
        checkpoint_name=args.checkpoint,
        device=args.device,
    )

    if not hasattr(model, "sdf_upsample"):
        raise RuntimeError(
            "This script expects an SDF SR model with `sdf_upsample`. "
            f"Got model class: {model.__class__.__module__}.{model.__class__.__name__}"
        )

    crop_hw = None if args.crop_hw is None else (int(args.crop_hw[0]), int(args.crop_hw[1]))
    lr_q_hwc, sample_label = load_lr_input(
        cfg,
        split=args.split,
        sample_offset=int(args.sample_offset),
        dataset_root=args.dataset_root,
        lr_npy=args.lr_npy,
        crop_hw=crop_hw,
    )

    scale = int(getattr(model, "upsample_factor", int(getattr(cfg, "scale", 4))))
    hr_q_hwc = _load_hr_input(
        cfg,
        split=args.split,
        sample_offset=int(args.sample_offset),
        dataset_root=args.dataset_root,
        lr_npy=args.lr_npy,
        hr_npy=args.hr_npy,
        crop_hw=crop_hw,
        scale=scale,
    )

    crystal = str(getattr(cfg, "crystal", "fcc")).lower()
    sym_name = "D6h" if crystal == "hcp" else "Oh"
    sym_class = resolve_symmetry(sym_name)

    stages, aux, q_sr_flat, lr_flat, lr_shape = _collect_sdf_stages(model, lr_q_hwc)
    hr_shape = (int(lr_shape[0] * scale), int(lr_shape[1] * scale))
    q_sr_hwc = q_sr_flat.detach().cpu().reshape(hr_shape[0], hr_shape[1], 4).numpy().astype(np.float32, copy=False)

    out_root = (
        Path(args.out_dir).resolve()
        if args.out_dir is not None
        else exp_dir / "sdf_stage_viz" / sample_label
    )
    out_root.mkdir(parents=True, exist_ok=True)

    stage_irrep_dir = out_root / "stage_irreps"
    boundary_dir = out_root / "boundary"
    ipf_dir = out_root / "ipf"
    npy_dir = out_root / "npy"
    stage_irrep_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)
    ipf_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    # 1) Irrep-stage plots
    for rec in stages:
        name = str(rec["name"])
        feat = rec["feat"]
        shape = tuple(int(v) for v in rec["shape"])

        by_block, by_l, scalars_0e = _compute_irrep_norm_maps(feat, shape, model.irreps_a1)

        _plot_map_grid(
            sorted(by_block.items())[: int(args.max_irrep_maps)],
            stage_irrep_dir / f"{_safe_name(name)}__irrep_block_norms.png",
            title=f"{name}: per-irrep-block spatial norm",
            cmap="magma",
            ncols=4,
        )

        _plot_map_grid(
            sorted(by_l.items()),
            stage_irrep_dir / f"{_safe_name(name)}__irrep_l_group_norms.png",
            title=f"{name}: per-l/parity spatial norm",
            cmap="magma",
            ncols=4,
        )

        if scalars_0e:
            _plot_map_grid(
                sorted(scalars_0e.items()),
                stage_irrep_dir / f"{_safe_name(name)}__0e_scalar_maps.png",
                title=f"{name}: 0e scalar copies",
                cmap="viridis",
                ncols=4,
            )

        np.savez(
            npy_dir / f"{_safe_name(name)}__irrep_norms.npz",
            **{f"by_block__{k}": v for k, v in by_block.items()},
            **{f"by_l__{k}": v for k, v in by_l.items()},
            **{f"zeroe__{k}": v for k, v in scalars_0e.items()},
        )

    # 2) Boundary evidence + boundary maps
    evidence_tensor = aux["tensor"]
    if isinstance(evidence_tensor, torch.Tensor) and evidence_tensor.dim() == 4:
        evidence_tensor = evidence_tensor.squeeze(0)
    if not isinstance(evidence_tensor, torch.Tensor) or evidence_tensor.dim() != 3:
        raise ValueError("Expected evidence tensor with shape (S,H,W)")

    pairwise_norm_lr = torch.linalg.norm(evidence_tensor, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
    np.save(npy_dir / "pairwise_scalar_lr_norm.npy", pairwise_norm_lr)

    map_np: dict[str, np.ndarray] = {}
    for k, v in aux.get("maps", {}).items():
        if isinstance(v, torch.Tensor):
            vv = v.detach().cpu()
            if vv.ndim == 3 and int(vv.shape[0]) == 1:
                vv = vv.squeeze(0)
            if vv.ndim == 2:
                map_np[str(k)] = vv.numpy().astype(np.float32, copy=False)

    e_items = sorted((k, v) for k, v in map_np.items() if k.startswith("E_"))
    c_items = sorted((k, v) for k, v in map_np.items() if k.startswith("C_"))

    if e_items:
        e_stack = np.stack([v for _, v in e_items], axis=0)
        e_mean = np.mean(e_stack, axis=0)
        e_max = np.max(e_stack, axis=0)
    else:
        e_mean = pairwise_norm_lr
        e_max = pairwise_norm_lr

    if c_items:
        c_stack = np.stack([v for _, v in c_items], axis=0)
        c_mean = np.mean(c_stack, axis=0)
        c_max = np.max(c_stack, axis=0)
    else:
        c_mean = pairwise_norm_lr
        c_max = pairwise_norm_lr

    _plot_map_grid(
        [
            ("Pairwise scalar norm LR", pairwise_norm_lr),
            ("E mean", e_mean),
            ("E max", e_max),
            ("C mean", c_mean),
            ("C max", c_max),
        ],
        boundary_dir / "evidence_summary.png",
        title="Boundary evidence summary",
        cmap="magma",
        ncols=3,
    )

    if e_items:
        _plot_map_grid(
            e_items[: int(args.max_evidence_maps)],
            boundary_dir / "evidence_E_maps.png",
            title="E maps (pairwise mismatch)",
            cmap="magma",
            ncols=4,
        )
    if c_items:
        _plot_map_grid(
            c_items[: int(args.max_evidence_maps)],
            boundary_dir / "evidence_C_maps.png",
            title="C maps (pairwise cosine mismatch)",
            cmap="magma",
            ncols=4,
        )

    boundary_maps: list[tuple[str, np.ndarray]] = [
        ("pairwise_norm_lr", pairwise_norm_lr),
        ("guidance_lr_norm", _norm_map(aux["guidance_lr"])),
        ("boundary_logits_lr", _first_channel_map(aux["boundary_logits_lr"])),
        ("boundary_prob_lr", 1.0 / (1.0 + np.exp(-_first_channel_map(aux["boundary_logits_lr"])))),
        ("guidance_hr_norm", _norm_map(aux["guidance_hr"])),
        ("boundary_logits_hr", _first_channel_map(aux["boundary_logits_hr"])),
        ("boundary_prob_hr", _first_channel_map(aux["boundary_prob_hr"])),
        ("sdf_hr", _first_channel_map(aux["sdf_hr"])),
        ("nx_hr", _first_channel_map(aux["nx_hr"])),
        ("ny_hr", _first_channel_map(aux["ny_hr"])),
        ("boundary_band_hr", _first_channel_map(aux["boundary_band_hr"])),
        ("side_prob_plus", _tensor_image_channels(aux["side_probs_hr"])[0]),
        ("side_prob_minus", _tensor_image_channels(aux["side_probs_hr"])[1]),
        ("feat_center_hr_norm", _norm_map(aux["feat_center_hr"])),
        ("feat_plus_hr_norm", _norm_map(aux["feat_plus_hr"])),
        ("feat_minus_hr_norm", _norm_map(aux["feat_minus_hr"])),
    ]

    if isinstance(aux.get("boundary_logits_hr_refined"), torch.Tensor):
        boundary_maps.append(("boundary_logits_hr_refined", _first_channel_map(aux["boundary_logits_hr_refined"])))
        boundary_maps.append(
            (
                "boundary_prob_hr_refined",
                1.0 / (1.0 + np.exp(-_first_channel_map(aux["boundary_logits_hr_refined"]))),
            )
        )

    _plot_map_grid(
        boundary_maps,
        boundary_dir / "boundary_and_sdf_maps.png",
        title="Boundary/SDF stage maps",
        cmap="magma",
        ncols=4,
    )

    if hasattr(model, "boundary_teacher_from_lr_quats"):
        try:
            labels_lr, gb_lr = model.boundary_teacher_from_lr_quats(
                lr_quats=lr_flat,
                lr_shape=lr_shape,
                thr_deg=float(args.boundary_teacher_thr_deg),
                connectivity=int(args.boundary_teacher_connectivity),
            )
            lbl_np = labels_lr.detach().cpu().numpy()
            gb_np = gb_lr.detach().cpu().numpy().astype(np.float32, copy=False)
            np.save(npy_dir / "teacher_grain_labels_lr.npy", lbl_np)
            np.save(npy_dir / "teacher_grain_boundary_lr.npy", gb_np)
            _plot_map_grid(
                [
                    ("Teacher grain labels (LR)", lbl_np.astype(np.float32, copy=False)),
                    ("Teacher grain boundary (LR)", gb_np),
                ],
                boundary_dir / "teacher_grain_maps_lr.png",
                title="Teacher grain segmentation/boundary from LR quats",
                cmap="viridis",
                ncols=2,
            )
        except Exception as exc:
            print(f"[warn] boundary_teacher_from_lr_quats failed: {exc}")

    # 3) IPF views
    _plot_ipf_rows(
        lr_q_hwc=lr_q_hwc,
        sr_q_hwc=q_sr_hwc,
        hr_q_hwc=hr_q_hwc,
        sym_class=sym_class,
        out_png=ipf_dir / "ipf_lr_sr_hr.png",
    )
    _plot_lr_ipf_with_pairwise(
        lr_q_hwc=lr_q_hwc,
        pairwise_norm_lr=pairwise_norm_lr,
        sym_class=sym_class,
        out_png=ipf_dir / "lr_ipf_with_pairwise_norm.png",
    )

    # 4) Text summary
    def _scalar_list(x: Any) -> list[float] | None:
        if not isinstance(x, torch.Tensor):
            return None
        return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]

    summary_lines = [
        f"experiment_dir: {exp_dir}",
        f"checkpoint: {ckpt_path if ckpt_path is not None else '[random init]'}",
        f"sample: {sample_label}",
        f"model_class: {model.__class__.__module__}.{model.__class__.__name__}",
        f"model_irreps_a1: {model.irreps_a1}",
        f"lr_shape: {lr_shape}",
        f"hr_shape: {hr_shape}",
        f"pairwise_scalar_channels: {int(evidence_tensor.shape[0])}",
        f"evidence_offsets: {getattr(model, 'evidence_offsets', 'n/a')}",
        f"shift_px: {_scalar_list(aux.get('shift_px', None))}",
        f"band_center: {_scalar_list(aux.get('band_center', None))}",
        f"band_sharpness: {_scalar_list(aux.get('band_sharpness', None))}",
        f"side_temp: {_scalar_list(aux.get('side_temp', None))}",
        "",
        "output_dirs:",
        f"  - stage_irreps: {stage_irrep_dir}",
        f"  - boundary: {boundary_dir}",
        f"  - ipf: {ipf_dir}",
        f"  - npy: {npy_dir}",
        "",
    ]
    (out_root / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Checkpoint: {ckpt_path if ckpt_path is not None else '[random init]'}")
    print(f"Sample: {sample_label}")
    print(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"LR shape: {lr_shape}, HR shape: {hr_shape}")
    print(f"Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
