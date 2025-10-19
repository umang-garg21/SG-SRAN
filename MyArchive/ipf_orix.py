# -*-coding:utf-8 -*-
"""
File:        ipf_orix.py
Created at:  2025/10/01 12:06:28
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""

import os, re, glob
from typing import List, Tuple, Optional, Union, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib as mpl
import matplotlib.pyplot as plt
from orix.quaternion import Orientation
from orix.quaternion import symmetry as SYM
from orix.vector import Vector3d
from orix import plot as orix_plot

# MPL defaults
if not getattr(mpl, "_ipf_defaults_set", False):
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "figure.dpi": 500,
        }
    )
    mpl._ipf_defaults_set = True

# Utils
_LAST_INT_RE = re.compile(r"(\d+)(?=\.npy$)")


def _last_int_key(fp: str) -> int:
    m = _LAST_INT_RE.search(os.path.basename(fp))
    return int(m.group(1)) if m else -1


def _as_hw4(arr: np.ndarray) -> np.ndarray:
    """(H,W,4) or (4,H,W) -> (4,H,W) float32, no unnecessary copies."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D quaternion array; got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = np.moveaxis(arr, -1, 0)
    elif arr.shape[0] != 4:
        raise ValueError(f"Expected (H,W,4) or (4,H,W); got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _ensure_hw4_for_orix(arr: np.ndarray) -> np.ndarray:
    """(H,W,4) or (4,H,W) -> (H,W,4) for Orix (no copy if possible)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D quaternion array; got {arr.shape}")
    return arr if arr.shape[-1] == 4 else np.moveaxis(arr, 0, -1)


def _to_torch_4hw(arr: np.ndarray) -> torch.Tensor:
    """Safe tensor from (4,H,W) numpy (handles read-only memmaps)."""
    if (
        (arr.dtype != np.float32)
        or (not arr.flags["C_CONTIGUOUS"])
        or (not arr.flags["WRITEABLE"])
    ):
        arr = np.array(arr, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(arr)


# Symmetry aliases
_SYM_ALIASES = {
    # cubic (FCC/BCC Laue m-3m)
    "oh": "Oh",
    "cubic": "Oh",
    "fcc": "Oh",
    "bcc": "Oh",
    "m-3m": "Oh",
    # hexagonal (HCP Laue 6/mmm)
    "hcp": "D6h",
    "hex": "D6h",
    "6/mmm": "D6h",
    "d6h": "D6h",
    # a few common others (extend as needed)
    "d4h": "D4h",
    "d3d": "D3d",
    "d2h": "D2h",
    "td": "Td",
    "o": "O",
}


def _resolve_symmetry(symmetry: Union[str, object]):
    if not isinstance(symmetry, str):
        return symmetry
    key = symmetry.strip().lower()
    canon = _SYM_ALIASES.get(key, symmetry.strip())
    if hasattr(SYM, canon):
        return getattr(SYM, canon)
    tname = canon[:1].upper() + canon[1:]
    if hasattr(SYM, tname):
        return getattr(SYM, tname)
    raise ValueError(f"Unknown symmetry '{symmetry}'")


_DIRS = {"X": Vector3d((1, 0, 0)), "Y": Vector3d((0, 1, 0)), "Z": Vector3d((0, 0, 1))}


# Dataset Class
class QuaternionPairDataset(Dataset):
    """
    LR/HR quaternion .npy pairs matched by the last integer in filename.
      - Returns tensors shaped (4,h,w) and (4,H,W) for training (float32).
      - Provides IPF previews using a single, dataset-wide symmetry.
    """

    def __init__(
        self,
        lr_glob: str,
        hr_glob: str,
        take_first: Optional[int] = None,
        symmetry: Union[str, object] = "Oh",
    ):
        lr_files = sorted(glob.glob(lr_glob, recursive=True), key=_last_int_key)
        hr_files = sorted(glob.glob(hr_glob, recursive=True), key=_last_int_key)
        if not lr_files:
            raise FileNotFoundError(f"No LR files matched glob:\n  {lr_glob}")
        if not hr_files:
            raise FileNotFoundError(f"No HR files matched glob:\n  {hr_glob}")

        lr_map = {k: f for f in lr_files if (k := _last_int_key(f)) >= 0}
        hr_map = {k: f for f in hr_files if (k := _last_int_key(f)) >= 0}
        common = sorted(lr_map.keys() & hr_map.keys())
        if not common:
            raise FileNotFoundError("No matching LR/HR quaternion .npy pairs found.")

        if take_first is not None:
            common = common[:take_first]

        self.pairs: List[Tuple[str, str]] = [(lr_map[k], hr_map[k]) for k in common]

        # Symmetry for IPF previews
        self.sym_class = _resolve_symmetry(symmetry)  # e.g., Oh or D6h
        self.ckey = orix_plot.IPFColorKeyTSL(self.sym_class.laue)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lr_fp, hr_fp = self.pairs[idx]
        lr_np = _as_hw4(np.load(lr_fp, mmap_mode="r"))  # (4,h,w)
        hr_np = _as_hw4(np.load(hr_fp, mmap_mode="r"))  # (4,H,W)
        return _to_torch_4hw(lr_np), _to_torch_4hw(hr_np)

    # IPF preview for a single index
    def save_ipf_preview(
        self,
        idx: int,
        out_png: str,
        which: str = "HR",
        ref_dir: str = "ALL",
        include_key: bool = True,
    ):
        """Fast IPF save for a single LR/HR image using the dataset's symmetry."""
        lr_fp, hr_fp = self.pairs[idx]
        arr = np.load(hr_fp if which.upper() == "HR" else lr_fp, mmap_mode="r")
        arr_hw4 = _ensure_hw4_for_orix(arr)  # (H,W,4)

        ori = Orientation(arr_hw4)
        ori.symmetry = self.sym_class

        ckey = self.ckey
        show_all = ref_dir.upper() == "ALL"
        ncols = 3 if show_all else 1

        key_cols = 1 if include_key else 0
        fig_cols = ncols + key_cols
        wr = [1] * ncols + ([0.9] if include_key else [])
        fig = plt.figure(
            constrained_layout=False,
            figsize=(5.2 * ncols + (2.6 if include_key else 0), 4.8),
        )
        gs = fig.add_gridspec(1, fig_cols, width_ratios=wr, wspace=0.05)
        axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]

        if show_all:
            for name, ax in zip(("X", "Y", "Z"), axes):
                ckey.direction = _DIRS[name]
                img = ckey.orientation2color(~ori)  # ~ori: lab->crystal
                ax.imshow(img)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"IPF-{name}")
                ax.axis("off")
        else:
            ref = ref_dir.upper()
            if ref not in _DIRS:
                raise ValueError("ref_dir must be 'X','Y','Z', or 'ALL'")
            ckey.direction = _DIRS[ref]
            img = ckey.orientation2color(~ori)
            a0 = axes[0]
            a0.imshow(img)
            a0.set_aspect("equal", adjustable="box")
            a0.set_title(f"IPF-{ref}")
            a0.axis("off")

        if include_key:
            ax_ipf = fig.add_subplot(
                gs[0, -1], projection="ipf", symmetry=ori.symmetry.laue
            )
            ax_ipf.plot_ipf_color_key()
            ax_ipf.set_title("")
            # for txt in getattr(ax_ipf, "texts", []):
            #     txt.set_fontsize(12)

        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)


# Example usage
if __name__ == "__main__":
    ds = QuaternionPairDataset(
        lr_glob="/data/warren/materials/materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/LR_Images/**/Open_718_Train_lr_x_normal_*.npy",
        hr_glob="/data/warren/materials/materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/HR_Images/**/Open_718_Train_hr_x_normal_*.npy",
        take_first=5,
        symmetry="Oh",
    )

    print("Pairs:", len(ds))
    lr, hr = ds[0]  # (4,h,w), (4,H,W)

    ds.save_ipf_preview(
        0, "ipf_out/sample0_ALL.png", which="HR", ref_dir="ALL", include_key=True
    )

    # Batch previews (Z only, no color key)
    ds.save_ipf_many(
        indices=range(len(ds)),
        out_dir="ipf_out",
        which="HR",
        ref_dir="Z",
        include_key=True,
    )
