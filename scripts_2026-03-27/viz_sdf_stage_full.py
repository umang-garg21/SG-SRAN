#!/usr/bin/env python
"""Full SDF stage visualization: irreps, decoded IPF, and grain-boundary losses."""

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
import torch.nn.functional as F
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


def _plot_stage_decoded_ipf_rows(
    rows: list[tuple[str, list[np.ndarray]]],
    out_png: Path,
) -> None:
    if not rows:
        return
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(12.5, 3.6 * nrows))
    axes = np.asarray(axes).reshape(nrows, 3)

    for r, (name, rgbs) in enumerate(rows):
        for c, dname in enumerate(("X", "Y", "Z")):
            ax = axes[r, c]
            ax.imshow(rgbs[c])
            ax.set_title(f"{name} -> IPF-{dname}", fontsize=9)
            ax.axis("off")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
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


def _set_decoder_settings(
    model,
    *,
    num_starts: int,
    steps: int,
    lr: float,
    method: str | None,
) -> dict[str, tuple[Any, Any]]:
    dec = getattr(model, "decoder", None)
    if dec is None:
        return {}

    changed: dict[str, tuple[Any, Any]] = {}

    if hasattr(dec, "num_starts"):
        old = getattr(dec, "num_starts")
        new = max(1, int(num_starts))
        setattr(dec, "num_starts", new)
        changed["num_starts"] = (old, new)

    if hasattr(dec, "steps"):
        old = getattr(dec, "steps")
        new = max(0, int(steps))
        setattr(dec, "steps", new)
        changed["steps"] = (old, new)

    if hasattr(dec, "lr"):
        old = getattr(dec, "lr")
        new = float(lr)
        setattr(dec, "lr", new)
        changed["lr"] = (old, new)

    if method is not None and hasattr(dec, "method"):
        old = getattr(dec, "method")
        new = str(method)
        setattr(dec, "method", new)
        changed["method"] = (old, new)

    return changed


def _flat_quats_to_teacher(
    model,
    q_flat: torch.Tensor,
    img_shape: tuple[int, int],
    thr_deg: float,
    connectivity: int,
) -> torch.Tensor | None:
    if hasattr(model, "batch_boundary_teacher_from_quats"):
        _, gb = model.batch_boundary_teacher_from_quats(
            quats=q_flat,
            img_shape=img_shape,
            thr_deg=float(thr_deg),
            connectivity=int(connectivity),
        )
        return gb

    if hasattr(model, "boundary_teacher_from_quats"):
        _, gb = model.boundary_teacher_from_quats(
            quats=q_flat,
            img_shape=img_shape,
            thr_deg=float(thr_deg),
            connectivity=int(connectivity),
        )
        return gb.unsqueeze(0).unsqueeze(0)

    return None


def _flat_quats_to_teacher_lr(
    model,
    q_flat: torch.Tensor,
    img_shape: tuple[int, int],
    thr_deg: float,
    connectivity: int,
) -> torch.Tensor | None:
    if hasattr(model, "boundary_teacher_from_lr_quats"):
        _, gb = model.boundary_teacher_from_lr_quats(
            lr_quats=q_flat,
            lr_shape=img_shape,
            thr_deg=float(thr_deg),
            connectivity=int(connectivity),
        )
        return gb.unsqueeze(0).unsqueeze(0)
    return _flat_quats_to_teacher(
        model=model,
        q_flat=q_flat,
        img_shape=img_shape,
        thr_deg=thr_deg,
        connectivity=connectivity,
    )


def _binary_metrics(pred_prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> dict[str, float]:
    pred = (pred_prob >= float(thr)).to(dtype=target.dtype)
    tp = float((pred * target).sum().item())
    fp = float((pred * (1.0 - target)).sum().item())
    fn = float(((1.0 - pred) * target).sum().item())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def _to_b1hw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        if int(x.shape[0]) == 1:
            return x.unsqueeze(1)
        raise ValueError(f"Expected (1,H,W) or (H,W), got {tuple(x.shape)}")
    if x.dim() == 4:
        if int(x.shape[0]) != 1:
            raise ValueError(f"Expected batch size 1, got {tuple(x.shape)}")
        return x
    raise ValueError(f"Expected 2D/3D/4D map tensor, got {tuple(x.shape)}")


def _plot_boundary_predictions(
    *,
    lr_ipf_z: np.ndarray,
    hr_ipf_z: np.ndarray | None,
    pairwise_norm_lr: np.ndarray,
    pred_lr_prob: np.ndarray,
    pred_hr_prob: np.ndarray,
    teacher_lr: np.ndarray | None,
    teacher_hr: np.ndarray | None,
    bce_lr_map: np.ndarray | None,
    bce_hr_map: np.ndarray | None,
    metrics_text: str,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.5))

    # Row 1 (HR)
    axes[0, 0].imshow(hr_ipf_z if hr_ipf_z is not None else lr_ipf_z)
    axes[0, 0].set_title("HR IPF-Z")
    axes[0, 0].axis("off")

    if teacher_hr is not None:
        im = axes[0, 1].imshow(teacher_hr, cmap="gray", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.02)
        axes[0, 1].set_title("Teacher GB (HR)")
    else:
        axes[0, 1].imshow(np.zeros_like(pred_hr_prob), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, 1].set_title("Teacher GB (HR) N/A")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(pred_hr_prob, cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.02)
    axes[0, 2].set_title("Pred GB Prob (HR)")
    axes[0, 2].axis("off")

    pred_hr_bin = (pred_hr_prob >= 0.5).astype(np.float32)
    im = axes[0, 3].imshow(pred_hr_bin, cmap="gray", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.02)
    axes[0, 3].set_title("Pred GB Binary (HR)")
    axes[0, 3].axis("off")

    if bce_hr_map is not None:
        im = axes[0, 4].imshow(bce_hr_map, cmap="viridis")
        fig.colorbar(im, ax=axes[0, 4], fraction=0.046, pad=0.02)
        axes[0, 4].set_title("Per-pixel BCE (HR)")
    else:
        axes[0, 4].imshow(np.zeros_like(pred_hr_prob), cmap="viridis")
        axes[0, 4].set_title("Per-pixel BCE (HR) N/A")
    axes[0, 4].axis("off")

    # Row 2 (LR)
    axes[1, 0].imshow(lr_ipf_z)
    axes[1, 0].set_title("LR IPF-Z")
    axes[1, 0].axis("off")

    if teacher_lr is not None:
        im = axes[1, 1].imshow(teacher_lr, cmap="gray", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.02)
        axes[1, 1].set_title("Teacher GB (LR)")
    else:
        axes[1, 1].imshow(np.zeros_like(pred_lr_prob), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Teacher GB (LR) N/A")
    axes[1, 1].axis("off")

    im = axes[1, 2].imshow(pred_lr_prob, cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.02)
    axes[1, 2].set_title("Pred GB Prob (LR)")
    axes[1, 2].axis("off")

    if bce_lr_map is not None:
        im = axes[1, 3].imshow(bce_lr_map, cmap="viridis")
        fig.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.02)
        axes[1, 3].set_title("Per-pixel BCE (LR)")
    else:
        im = axes[1, 3].imshow(pairwise_norm_lr, cmap="magma")
        fig.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.02)
        axes[1, 3].set_title("Pairwise Scalar Norm (LR)")
    axes[1, 3].axis("off")

    im = axes[1, 4].imshow(pairwise_norm_lr, cmap="magma")
    fig.colorbar(im, ax=axes[1, 4], fraction=0.046, pad=0.02)
    axes[1, 4].set_title("Pairwise Scalar Norm (LR)")
    axes[1, 4].axis("off")

    fig.suptitle(metrics_text, y=1.02, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot SDF model stages: irreps per stage, decoded stage IPF, LR/SR/HR IPF, "
            "and grain-boundary predictions with losses."
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
        help="Output directory. Default: <exp_dir>/sdf_stage_full_viz/<sample_label>",
    )
    p.add_argument("--max_evidence_maps", type=int, default=24)
    p.add_argument("--max_irrep_maps", type=int, default=24)
    p.add_argument(
        "--boundary_teacher_thr_deg",
        type=float,
        default=4.0,
        help="Misorientation threshold for boundary teacher maps.",
    )
    p.add_argument(
        "--boundary_teacher_connectivity",
        type=int,
        default=8,
        choices=[4, 8],
        help="Connectivity for boundary teacher maps.",
    )

    # Good decoder defaults for visualization quality.
    p.add_argument("--decoder_num_starts", type=int, default=16)
    p.add_argument("--decoder_steps", type=int, default=20)
    p.add_argument("--decoder_lr", type=float, default=0.03)
    p.add_argument("--decoder_method", type=str, default=None, help="Optional decoder method override.")
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

    decoder_changes = _set_decoder_settings(
        model,
        num_starts=int(args.decoder_num_starts),
        steps=int(args.decoder_steps),
        lr=float(args.decoder_lr),
        method=args.decoder_method,
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
        else exp_dir / "sdf_stage_full_viz" / sample_label
    )
    out_root.mkdir(parents=True, exist_ok=True)

    stage_irrep_dir = out_root / "stage_irreps"
    stage_ipf_dir = out_root / "stage_ipf"
    boundary_dir = out_root / "boundary"
    ipf_dir = out_root / "ipf"
    npy_dir = out_root / "npy"
    stage_irrep_dir.mkdir(parents=True, exist_ok=True)
    stage_ipf_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)
    ipf_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    # 1) Irrep-stage plots + decoded stage IPFs
    stage_ipf_rows: list[tuple[str, list[np.ndarray]]] = []
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

        # Decode each stage with A1 feature size into quats and render IPF.
        q_stage_hwc: np.ndarray | None = None
        if feat.ndim == 2 and int(feat.shape[-1]) == int(getattr(model, "feature_dim_a1", feat.shape[-1])):
            try:
                with torch.enable_grad():
                    q_stage = model.decode(feat)
                q_stage_hwc = q_stage.detach().cpu().reshape(shape[0], shape[1], 4).numpy().astype(np.float32, copy=False)
            except Exception:
                q_stage_hwc = None
        elif feat.ndim == 2 and int(feat.shape[-1]) == 4:
            q_stage_hwc = feat.detach().cpu().reshape(shape[0], shape[1], 4).numpy().astype(np.float32, copy=False)

        if q_stage_hwc is not None:
            rgbs = _render_ipf_rgb_safe(q_stage_hwc, sym_class=sym_class, ref_dir="ALL")
            if isinstance(rgbs, list):
                stage_ipf_rows.append((name, rgbs))

    if stage_ipf_rows:
        _plot_stage_decoded_ipf_rows(
            rows=stage_ipf_rows,
            out_png=stage_ipf_dir / "stage_decoded_ipf_rows.png",
        )

    # 2) Evidence and boundary maps
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

    _plot_map_grid(
        [("Pairwise scalar norm LR", pairwise_norm_lr)],
        boundary_dir / "pairwise_scalar_norm_lr.png",
        title="LR pairwise scalar norm",
        cmap="magma",
        ncols=1,
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

    # 3) LR/SR/HR IPF references
    _plot_ipf_rows(
        lr_q_hwc=lr_q_hwc,
        sr_q_hwc=q_sr_hwc,
        hr_q_hwc=hr_q_hwc,
        sym_class=sym_class,
        out_png=ipf_dir / "ipf_lr_sr_hr.png",
    )

    # 4) Boundary predictions and losses (HR + LR)
    pred_hr_key = "boundary_logits_hr_refined" if isinstance(aux.get("boundary_logits_hr_refined"), torch.Tensor) else "boundary_logits_hr"
    pred_hr_logits = _to_b1hw(aux[pred_hr_key]).detach()
    pred_lr_logits = _to_b1hw(aux["boundary_logits_lr"]).detach()

    pred_hr_prob = torch.sigmoid(pred_hr_logits)
    pred_lr_prob = torch.sigmoid(pred_lr_logits)

    teacher_hr = None
    teacher_lr = None
    bce_hr = None
    bce_lr = None
    bce_hr_map = None
    bce_lr_map = None
    metrics_hr: dict[str, float] | None = None
    metrics_lr: dict[str, float] | None = None

    if hr_q_hwc is not None:
        hr_flat = torch.from_numpy(hr_q_hwc.reshape(-1, 4)).to(model.device, dtype=torch.float32)
        gb_hr = _flat_quats_to_teacher(
            model=model,
            q_flat=hr_flat,
            img_shape=hr_shape,
            thr_deg=float(args.boundary_teacher_thr_deg),
            connectivity=int(args.boundary_teacher_connectivity),
        )
        if isinstance(gb_hr, torch.Tensor):
            gb_hr = _to_b1hw(gb_hr).to(device=pred_hr_logits.device, dtype=pred_hr_logits.dtype)
            if tuple(gb_hr.shape[-2:]) != tuple(pred_hr_logits.shape[-2:]):
                gb_hr = F.interpolate(gb_hr, size=pred_hr_logits.shape[-2:], mode="nearest")
            teacher_hr = gb_hr.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
            bce_hr_t = F.binary_cross_entropy_with_logits(pred_hr_logits, gb_hr)
            bce_hr = float(bce_hr_t.item())
            bce_hr_map = F.binary_cross_entropy_with_logits(pred_hr_logits, gb_hr, reduction="none").squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
            metrics_hr = _binary_metrics(pred_prob=pred_hr_prob, target=gb_hr)

    gb_lr = _flat_quats_to_teacher_lr(
        model=model,
        q_flat=lr_flat,
        img_shape=lr_shape,
        thr_deg=float(args.boundary_teacher_thr_deg),
        connectivity=int(args.boundary_teacher_connectivity),
    )
    if isinstance(gb_lr, torch.Tensor):
        gb_lr = _to_b1hw(gb_lr).to(device=pred_lr_logits.device, dtype=pred_lr_logits.dtype)
        if tuple(gb_lr.shape[-2:]) != tuple(pred_lr_logits.shape[-2:]):
            gb_lr = F.interpolate(gb_lr, size=pred_lr_logits.shape[-2:], mode="nearest")
        teacher_lr = gb_lr.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
        bce_lr_t = F.binary_cross_entropy_with_logits(pred_lr_logits, gb_lr)
        bce_lr = float(bce_lr_t.item())
        bce_lr_map = F.binary_cross_entropy_with_logits(pred_lr_logits, gb_lr, reduction="none").squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
        metrics_lr = _binary_metrics(pred_prob=pred_lr_prob, target=gb_lr)

    lr_rgb = _render_ipf_rgb_safe(lr_q_hwc, sym_class=sym_class, ref_dir="ALL")
    hr_rgb = None
    if hr_q_hwc is not None:
        hr_rgb = _render_ipf_rgb_safe(hr_q_hwc, sym_class=sym_class, ref_dir="ALL")
    if not isinstance(lr_rgb, list):
        lr_rgb = [lr_rgb]
    if hr_rgb is not None and not isinstance(hr_rgb, list):
        hr_rgb = [hr_rgb]

    metrics_bits = []
    if bce_hr is not None:
        metrics_bits.append(f"BCE_HR={bce_hr:.4e}")
    if bce_lr is not None:
        metrics_bits.append(f"BCE_LR={bce_lr:.4e}")
    if metrics_hr is not None:
        metrics_bits.append(f"HR F1={metrics_hr['f1']:.3f} IoU={metrics_hr['iou']:.3f}")
    if metrics_lr is not None:
        metrics_bits.append(f"LR F1={metrics_lr['f1']:.3f} IoU={metrics_lr['iou']:.3f}")
    metrics_text = " | ".join(metrics_bits) if metrics_bits else "Boundary teacher/loss unavailable"

    _plot_boundary_predictions(
        lr_ipf_z=lr_rgb[2],
        hr_ipf_z=(hr_rgb[2] if isinstance(hr_rgb, list) else None),
        pairwise_norm_lr=pairwise_norm_lr,
        pred_lr_prob=pred_lr_prob.squeeze().detach().cpu().numpy().astype(np.float32, copy=False),
        pred_hr_prob=pred_hr_prob.squeeze().detach().cpu().numpy().astype(np.float32, copy=False),
        teacher_lr=teacher_lr,
        teacher_hr=teacher_hr,
        bce_lr_map=bce_lr_map,
        bce_hr_map=bce_hr_map,
        metrics_text=metrics_text,
        out_png=boundary_dir / "boundary_predictions_and_losses.png",
    )

    # 5) Text summary
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
        f"boundary_teacher_thr_deg: {float(args.boundary_teacher_thr_deg)}",
        f"boundary_teacher_connectivity: {int(args.boundary_teacher_connectivity)}",
        f"decoder_changes: {decoder_changes}",
        f"bce_hr: {bce_hr}",
        f"bce_lr: {bce_lr}",
        f"metrics_hr: {metrics_hr}",
        f"metrics_lr: {metrics_lr}",
        f"shift_px: {_scalar_list(aux.get('shift_px', None))}",
        f"band_center: {_scalar_list(aux.get('band_center', None))}",
        f"band_sharpness: {_scalar_list(aux.get('band_sharpness', None))}",
        f"side_temp: {_scalar_list(aux.get('side_temp', None))}",
        "",
        "output_dirs:",
        f"  - stage_irreps: {stage_irrep_dir}",
        f"  - stage_ipf: {stage_ipf_dir}",
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
    print(f"Decoder changes: {decoder_changes}")
    print(f"Boundary metrics: {metrics_text}")
    print(f"Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
