"""
Plot local-iso feature space with IPF color correspondence.

Outputs a figure with:
1) IPF color key overlaid with sampled orientations (true IPF colors)
2) 2D PCA projection of feature vectors, colored by IPF color
3) IPF-key overlays where points are colored by selected feature channels

No argparse: edit CONFIG directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from orix import plot as orix_plot
from orix.quaternion import Orientation, symmetry
from orix.sampling import get_sample_fundamental
from orix.vector import Vector3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.SR_double_conv_SRattn import LocalIsoCrystalEncoder


CONFIG = {
    "crystal": "fcc",  # "fcc" or "hcp"
    "d6_convention": "z_axis",
    "feature_space": "a1",  # "a1" or "full"
    "reference_dir": "Z",  # "X", "Y", or "Z"
    "sample_resolution": 2,
    "sample_method": "cubochoric",
    "max_samples": 4000,
    "seed": 0,
    "point_size": 8,
    "alpha": 0.8,
    "channels_to_plot": [0, 1, 2, 3],
    "out_png": "diagnostics/feature_space_ipf_overlay.png",
    "out_npz": "diagnostics/feature_space_ipf_overlay_data.npz",
    "out_json": "diagnostics/feature_space_ipf_overlay_meta.json",
}


_REF_DIRS = {
    "X": Vector3d((1, 0, 0)),
    "Y": Vector3d((0, 1, 0)),
    "Z": Vector3d((0, 0, 1)),
}


def _sample_fz_passive_quats(
    crystal: str,
    resolution: int,
    method: str,
    max_samples: int | None,
) -> np.ndarray:
    crystal_l = str(crystal).lower()
    if crystal_l == "fcc":
        point_group = symmetry.Oh
    elif crystal_l == "hcp":
        point_group = symmetry.D6h
    else:
        raise ValueError(f"crystal must be 'fcc' or 'hcp', got: {crystal}")

    rot = get_sample_fundamental(int(resolution), point_group=point_group, method=str(method))
    raw = np.asarray(getattr(rot, "data", rot), dtype=np.float32)
    if raw.ndim != 2:
        raw = raw.reshape(-1, 4)
    if raw.shape[-1] != 4 and raw.shape[0] == 4:
        raw = raw.T
    if raw.shape[-1] != 4:
        raise ValueError(f"Unexpected quaternion sample shape: {tuple(raw.shape)}")

    # Normalize + hemisphere.
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    raw = raw / np.clip(norms, 1e-12, None)
    raw[raw[:, 0] < 0.0] *= -1.0
    if max_samples is not None:
        raw = raw[: int(max_samples)]
    return raw


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # x: (N, C)
    x_center = x - x.mean(axis=0, keepdims=True)
    # SVD-based PCA
    u, s, vt = np.linalg.svd(x_center, full_matrices=False)
    coords = u[:, :2] * s[:2]
    components = vt[:2]
    explained = (s[:2] ** 2) / np.maximum((s**2).sum(), 1e-12)
    return coords, components, explained


def main() -> None:
    crystal = str(CONFIG["crystal"]).lower()
    d6_convention = str(CONFIG["d6_convention"])
    feature_space = str(CONFIG["feature_space"]).lower()
    ref_dir = str(CONFIG["reference_dir"]).upper()
    sample_resolution = int(CONFIG["sample_resolution"])
    sample_method = str(CONFIG["sample_method"])
    max_samples = CONFIG["max_samples"]
    point_size = float(CONFIG["point_size"])
    alpha = float(CONFIG["alpha"])
    channels = [int(c) for c in CONFIG["channels_to_plot"]]
    seed = int(CONFIG["seed"])

    if ref_dir not in _REF_DIRS:
        raise ValueError(f"reference_dir must be one of {tuple(_REF_DIRS.keys())}, got: {ref_dir}")
    if feature_space not in {"a1", "full"}:
        raise ValueError(f"feature_space must be 'a1' or 'full', got: {feature_space}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    encoder = LocalIsoCrystalEncoder(
        crystal=crystal,
        d6_convention=d6_convention,
        dtype=torch.float32,
        device=device,
    ).eval()

    q_np = _sample_fz_passive_quats(
        crystal=crystal,
        resolution=sample_resolution,
        method=sample_method,
        max_samples=None if max_samples is None else int(max_samples),
    )
    q_t = torch.from_numpy(q_np).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if feature_space == "a1":
            feat_t = encoder.forward_a1(q_t)
            irreps_str = str(encoder.irreps_a1)
        else:
            feat_t = encoder.forward_full(q_t)
            irreps_str = str(encoder.irreps_full)
    feat = feat_t.detach().cpu().numpy()
    n, c = feat.shape

    # IPF colors for each orientation.
    sym_obj = symmetry.Oh if crystal == "fcc" else symmetry.D6h
    ori = Orientation(q_np)
    ori.symmetry = sym_obj
    ckey = orix_plot.IPFColorKeyTSL(sym_obj.laue)
    ckey.direction = _REF_DIRS[ref_dir]
    colors = np.asarray(ckey.orientation2color(ori), dtype=np.float32)
    colors = np.clip(colors, 0.0, 1.0)

    pca_xy, pca_components, pca_explained = _pca_2d(feat)

    channels_valid = [ch for ch in channels if 0 <= ch < c]
    if len(channels_valid) == 0:
        channels_valid = list(range(min(4, c)))

    ncols = max(2, len(channels_valid))
    fig = plt.figure(figsize=(5.0 * ncols, 10.0))
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.28)

    # Top-left: IPF key + orientation points in their own IPF colors.
    ax_ipf = fig.add_subplot(gs[0, 0], projection="ipf", symmetry=sym_obj.laue)
    ax_ipf.plot_ipf_color_key()
    ax_ipf.scatter(ori, c=colors, s=point_size, alpha=alpha)
    ax_ipf.set_title(f"IPF-{ref_dir} Key + Sampled Orientations")

    # Top-middle: PCA(feature) colored by IPF color.
    ax_pca = fig.add_subplot(gs[0, 1])
    ax_pca.scatter(pca_xy[:, 0], pca_xy[:, 1], c=colors, s=point_size, alpha=alpha, linewidths=0)
    ax_pca.set_xlabel(f"PC1 ({100.0 * pca_explained[0]:.1f}% var)")
    ax_pca.set_ylabel(f"PC2 ({100.0 * pca_explained[1]:.1f}% var)")
    ax_pca.set_title("Feature PCA colored by IPF color")
    ax_pca.grid(True, alpha=0.2)

    # Fill unused top-row axes, if any.
    for j in range(2, ncols):
        ax = fig.add_subplot(gs[0, j])
        ax.axis("off")

    # Bottom row: channel value overlays on IPF key.
    for j, ch in enumerate(channels_valid):
        ax = fig.add_subplot(gs[1, j], projection="ipf", symmetry=sym_obj.laue)
        ax.plot_ipf_color_key()
        vals = feat[:, ch]
        sc = ax.scatter(ori, c=vals, cmap="coolwarm", s=point_size, alpha=alpha)
        ax.set_title(f"IPF overlay: feature ch {ch}")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("value", rotation=90)

    for j in range(len(channels_valid), ncols):
        ax = fig.add_subplot(gs[1, j])
        ax.axis("off")

    fig.suptitle(
        f"Feature/IPF Correspondence | crystal={crystal} | space={feature_space} | "
        f"irreps={irreps_str} | N={n}",
        fontsize=14,
        y=0.98,
    )

    out_png = Path(CONFIG["out_png"]).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    out_npz = Path(CONFIG["out_npz"]).resolve()
    np.savez_compressed(
        out_npz,
        quats_passive_wxyz=q_np,
        feature=feat,
        pca_xy=pca_xy,
        pca_components=pca_components,
        pca_explained=pca_explained,
        ipf_color_rgb=colors,
        channels_plotted=np.asarray(channels_valid, dtype=np.int64),
    )

    out_json = Path(CONFIG["out_json"]).resolve()
    with open(out_json, "w") as f:
        json.dump(
            {
                "crystal": crystal,
                "d6_convention": d6_convention,
                "feature_space": feature_space,
                "irreps": irreps_str,
                "reference_dir": ref_dir,
                "sample_resolution": sample_resolution,
                "sample_method": sample_method,
                "num_samples": int(n),
                "feature_dim": int(c),
                "channels_plotted": channels_valid,
                "out_png": str(out_png),
                "out_npz": str(out_npz),
            },
            f,
            indent=2,
        )

    print(f"Saved figure: {out_png}")
    print(f"Saved data:   {out_npz}")
    print(f"Saved meta:   {out_json}")


if __name__ == "__main__":
    main()

