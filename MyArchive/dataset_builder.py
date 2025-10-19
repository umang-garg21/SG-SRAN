# -*-coding:utf-8 -*-
"""
File:        dataset_builder.py
Created at:  2025/10/02 18:40:51
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""

from importlib_metadata import files
import os, re, glob, json, random, datetime, pytz, warnings
from typing import List, Tuple, Optional, Dict, Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from orix.quaternion import Orientation, symmetry as SYM
from orix.vector import Vector3d
from orix import plot as orix_plot
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from copy import deepcopy

from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d

if not getattr(mpl, "_ipf_defaults_set", False):
    mpl.rcParams.update({"font.size": 12, "axes.titlesize": 16, "figure.dpi": 500})
    mpl._ipf_defaults_set = True

_LAST_INT_RE = re.compile(r"(\d+)(?=\.npy$)")

# Aliases -> Orix symmetry canonical names
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
    # others
    "d4h": "D4h",
    "d3d": "D3d",
    "d2h": "D2h",
    "td": "Td",
    "o": "O",
}

_DIRS = {"X": Vector3d((1, 0, 0)), "Y": Vector3d((0, 1, 0)), "Z": Vector3d((0, 0, 1))}


# -------------------
# Helper Functions
# -------------------
def _canon_symmetry_str(symmetry: str) -> str:
    """Normalize symmetry alias into canonical string (e.g. 'Oh', 'D6h')."""
    if not isinstance(symmetry, str):
        return getattr(symmetry, "__name__", str(symmetry))
    key = symmetry.strip().lower()
    return _SYM_ALIASES.get(key, symmetry.strip())


def _resolve_symmetry(symmetry: str):
    """Resolve symmetry string into actual Orix symmetry class."""
    if not isinstance(symmetry, str):
        return symmetry
    canon = _canon_symmetry_str(symmetry)
    if hasattr(SYM, canon):
        return getattr(SYM, canon)
    raise ValueError(f"Unknown symmetry: {symmetry}")


def _last_int_key(fp: str) -> int:
    m = _LAST_INT_RE.search(os.path.basename(fp))
    return int(m.group(1)) if m else -1


def _to_spatial_quat(arr: np.ndarray) -> np.ndarray:
    """
    Reorder array so the quaternion axis (size=4) is last: (*spatial, 4).

    Accepts layouts such as:
        (4, H, W)
        (H, W, 4)
        (4, H, W, D)
        (H, W, D, 4)
        etc.

    Parameters
    ----------
    arr : np.ndarray
        Quaternion array containing exactly one axis of length 4.

    Returns
    -------
    q : np.ndarray
        Quaternion-last array (*spatial, 4), dtype float32.

    Raises
    ------
    ValueError
        If no quaternion axis (size=4) found or multiple exist.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}")

    shape = arr.shape
    ndim = arr.ndim
    if ndim == 0:
        raise ValueError("Input array must have at least one dimension")

    # Fast-path: already quaternion-last or quaternion-first
    if shape[-1] == 4:
        q = arr
    elif shape[0] == 4:
        q = np.moveaxis(arr, 0, -1)
    else:
        # Fallback: search for quaternion axis
        quat_axes = [i for i, s in enumerate(shape) if s == 4]
        if len(quat_axes) == 0:
            raise ValueError(f"No quaternion axis (size=4) found in shape {shape}")
        if len(quat_axes) > 1:
            raise ValueError(f"Multiple dimensions of size=4 found in {shape}")
        q = np.moveaxis(arr, quat_axes[0], -1)

    # Ensure float32 without unnecessary copying
    if q.dtype != np.float32:
        q = q.astype(np.float32, copy=False)
    return q


def _to_quat_spatial(arr: np.ndarray) -> np.ndarray:
    """
    Reorder array so the quaternion channel (size 4) comes first: (4, *spatial).

    Common layouts:
        (4, H, W, ...)
        (H, W, ..., 4)

    Falls back to scanning all axes for the quaternion dimension if not found
    in the first or last position.

    Parameters
    ----------
    arr : np.ndarray
        Quaternion array containing exactly one axis of length 4.

    Returns
    -------
    q : np.ndarray
        Quaternion-first array (4, *spatial), float32 dtype preserved.

    Raises
    ------
    ValueError
        If no dimension of length 4 exists or if multiple do.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}")

    shape = arr.shape
    ndim = arr.ndim

    if ndim < 1:
        raise ValueError(f"Invalid shape {shape}: array must have at least 1 dimension")

    if shape[0] == 4:
        q = arr  # Already quat-first
    elif shape[-1] == 4:
        q = np.moveaxis(arr, -1, 0)  # Move last axis to first
    else:
        # Fallback: search all axes
        quat_axes = [i for i, s in enumerate(shape) if s == 4]
        if len(quat_axes) == 0:
            raise ValueError(f"No quaternion dimension (size 4) found in shape {shape}")
        if len(quat_axes) > 1:
            raise ValueError(
                f"Multiple dimensions of size 4 found in {shape}; cannot infer quaternion axis"
            )
        q = np.moveaxis(arr, quat_axes[0], 0)  # Move found axis to first

    return q.astype(np.float32, copy=False)


def _to_torch_quat_spatial(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy (4,H,W) -> torch Tensor safely and efficiently."""
    if (
        arr.dtype != np.float32
        or not arr.flags["C_CONTIGUOUS"]
        or not arr.flags["WRITEABLE"]
    ):
        arr = np.array(arr, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(arr)


def _format_quaternions(
    q: np.ndarray,
    normalize: bool = True,
    enforce_hemisphere: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Standardize quaternion layout to scalar-first [s,x,y,z] in (4, *spatial) form.

    Auto-detects vector-first order ([x,y,z,s]) and performs optional normalization
    and hemisphere canonicalization (s >= 0).

    Parameters
    ----------
    q : np.ndarray
        Quaternion array of shape (...,4) or (4,...).
    normalize : bool, default=True
        Normalize each quaternion to unit norm.
    enforce_hemisphere : bool, default=True
        Flip quaternions so scalar part (s) >= 0.
    eps : float, default=1e-12
        Numerical floor for normalization.

    Returns
    -------
    q_out : np.ndarray
        Canonicalized scalar-first quaternions (4,*spatial), float32 dtype.
    """
    q_out = _to_quat_spatial(q)

    # Detect scalar order
    if float(np.mean(np.abs(q_out[0]))) < float(np.mean(np.abs(q_out[3]))):
        q_out = np.concatenate([q_out[3:4], q_out[0:3]], axis=0)

    # Normalize
    if normalize:
        norms = np.linalg.norm(q_out, axis=0, keepdims=True)
        norms = np.where(norms < eps, 1.0, norms)
        q_out /= norms

    # Hemisphere canonicalization
    if enforce_hemisphere:
        flip = q_out[0] < 0
        if np.any(flip):
            q_out[:, flip] *= -1.0

    return q_out.astype(np.float32, copy=False)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _save_npy(path: str, arr: np.ndarray):
    _ensure_dir(os.path.dirname(path) or ".")
    np.save(path, arr)


def get_patch_spatial(q: np.ndarray, patch_shape: tuple, scale: int):
    """
    Randomly crop a square/cubic patch from a quaternion field and produce
    aligned high- and low-resolution pairs via strided downsampling.

    Parameters
    ----------
    q : np.ndarray
        Quaternion array of shape (4,*spatial), (4,H,W), (4,H,W,D), etc.
    patch_shape : tuple
        Shape of the patch to extract (must be match the spatial rank of q).
    scale : int
        Downsampling scale factor.


    Returns
    -------
    lr_patch : np.ndarray
        Low-resolution quaternion patch after strided downsampling,
        with shape (4, H/scale, W/scale, ... ), dtype float32,
        and C-contiguous layout for efficient PyTorch conversion.

    hr_patch : np.ndarray
        High-resolution quaternion patch of shape (4, *patch_shape),
        dtype float32, and C-contiguous layout.
    """
    ndim = q.ndim - 1
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion-first (4, ...), got {q.shape}")
    if len(patch_shape) != ndim:
        raise ValueError(f"Patch shape {patch_shape} incompatible with data {q.shape}")

    starts = [
        random.randrange(0, q.shape[i + 1] - patch_shape[i] + 1) for i in range(ndim)
    ]
    slices_hr = (slice(None),) + tuple(
        slice(st, st + ps) for st, ps in zip(starts, patch_shape)
    )
    hr = q[slices_hr]
    stride = (slice(None),) + tuple(slice(None, None, scale) for _ in range(ndim))
    lr = hr[stride]
    return (
        np.asarray(lr, dtype=np.float32, order="C"),
        np.asarray(hr, dtype=np.float32, order="C"),
    )


def _pick_patch_size_all(
    files: List[str], scale: int, cap: Optional[int] = None
) -> int:
    """
    Compute the largest power-of-2 patch size fitting all quaternion images,
    divisible by the given scale. Uses memmap headers only (not loading data).

    Supports (4,H,W), (H,W,4), (4,H,W,D), (H,W,D,4), etc.

    Parameters
    ----------
    files : list of str
        List of .npy file paths.
    scale : int
        Downsampling scale (must divide patch size evenly).
    cap : int, optional
        Upper bound for patch size.

    Returns
    -------
    patch_shape : tuple[int]
        Equal-length patch dimensions (P,), (P, P), or (P, P, P).
    """
    min_side = np.inf  # smallest spatial extent across dataset
    ndim = None  # number of spatial dimensions

    for fp in files:
        try:
            mm = np.lib.format.open_memmap(fp, mode="r")
            shp = mm.shape
            del mm  # explicitly release the memmap reference
        except Exception as e:
            raise ValueError(f"Failed to read {fp}: {e}")

        # Identify quaternion axis (dim=4)
        if shp[0] == 4:
            spatial = shp[1:]
        elif shp[-1] == 4:
            spatial = shp[:-1]
        else:
            axes = [i for i, s in enumerate(shp) if s == 4]
            if not axes:
                raise ValueError(f"No quaternion dimension (size=4) in shape {shp}")
            ax = axes[0]
            spatial = tuple(s for i, s in enumerate(shp) if i != ax)

        if len(spatial) < 1:
            raise ValueError(f"No spatial dimensions found in {fp}, shape={shp}")

        ndim = len(spatial) if ndim is None else ndim
        if len(spatial) != ndim:
            raise ValueError(
                f"Inconsistent dimensionality across files ({len(spatial)} vs {ndim})"
            )

        # Track smallest spatial dimension
        min_side = min(min_side, *spatial)

    if not np.isfinite(min_side):
        raise ValueError("No valid spatial shapes found in dataset")

    # Cap and quantize to nearest power of two divisible by scale
    lim = int(min_side if cap is None else min(min_side, cap))
    P = 1 << (lim.bit_length() - 1)  # largest power of 2 <= lim
    while P % scale != 0:
        P //= 2
    if P <= 0:
        raise ValueError(f"No valid patch size for scale={scale}")

    # Square/cubic patch shape
    return tuple([P] * ndim)


# -------------------
# Dataset Builder
# -------------------
def build_quaternion_sr_dataset(
    hr_glob: Optional[str] = None,
    out_root: str = "datasets",
    dataset_name: str = "IN718",
    scale: int = 4,
    hr_dirs: Optional[
        Dict[str, str]
    ] = None,  # {"Train": "...", "Val": "...", "Test": "..."}
    split: Dict[str, float] = {"Train": 0.8, "Val": 0.1, "Test": 0.1},
    take_first: Optional[int] = None,
    patch_cap: Optional[int] = None,
    seed: int = 1234,
    symmetry: str = "Oh",
    creator: str = "Unknown",
    contact: str = "unknown@example.com",
):
    """
    Prepare a quaternion super-resolution dataset from high-resolution .npy images.

    Parameters
    ----------
    hr_glob : str, optional
        Glob pattern for HR .npy files (used if hr_dirs not provided).
    out_root : str, default="datasets"
        Root directory where dataset will be created.
    dataset_name : str, default="IN718"
        Name of the dataset folder.
    scale : int, default=4
        Downsampling scale (applied in both H and W).
    hr_dirs : dict, optional
        Explicit HR dirs: {"Train": "...", "Val": "...", "Test": "..."}.
    split : dict, default={"Train":0.8, "Val":0.1, "Test":0.1}
        Split ratios if auto-splitting HR files.
    take_first : int, optional
        If set, only use the first N HR files.
    patch_cap : int, optional
        Maximum patch size cap.
    seed : int, default=1234
        Random seed for reproducibility.
    symmetry : str, default="Oh"
        Orix symmetry label (saved in metadata).
    creator : str, default="Unknown"
        Dataset creator name.
    contact : str, default="unknown@example.com"
        Contact email.

    Returns
    -------
    dataset_info : dict
        Metadata about the prepared dataset, also written to dataset_info.json.
    """
    root = os.path.join(out_root, dataset_name)

    # Check if dataset already exists
    info_path = os.path.join(root, "dataset_info.json")
    if os.path.exists(info_path):
        print(f"[Preparing Dataset] Dataset already exists: {root}")
        with open(info_path, "r") as f:
            dataset_info = json.load(f)
        return dataset_info

    print(f"[Preparing Dataset] Preparing {dataset_name} dataset.")
    if take_first:
        print(
            f"[Preparing Dataset] take_first = {take_first} --> limiting total HR files to first {take_first}."
        )

    random.seed(seed)

    splits: Dict[str, List[str]] = {}
    # Case 1: explicit dirs
    if hr_dirs:
        for split_name, path in hr_dirs.items():
            paths = sorted(glob.glob(path, recursive=True), key=_last_int_key)
            if take_first:
                paths = paths[:take_first]
            splits[split_name.capitalize()] = paths
    # Case 2/3: glob
    elif hr_glob:
        all_files = sorted(glob.glob(hr_glob, recursive=True), key=_last_int_key)
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"[Preparing Dataset] No HR files matched: {hr_glob}"
            )

        if take_first:
            all_files = all_files[:take_first]

        print(f"[Preparing Dataset] Found {len(all_files)} HR files.")

        keys = ["Train", "Val", "Test"]
        ratios = [split.get(k, 0) for k in keys]
        s = sum(ratios)
        ratios = [r / s for r in ratios]
        n = len(all_files)
        n_train, n_val = int(round(n * ratios[0])), int(round(n * ratios[1]))
        splits = {
            "Train": all_files[:n_train],
            "Val": all_files[n_train : n_train + n_val],
            "Test": all_files[n_train + n_val :],
        }
    else:
        raise ValueError("Must provide hr_dirs or hr_glob")

    all_files = [f for flist in splits.values() for f in flist]
    print(f"[Preparing Dataset] Found {len(all_files)} total HR files.")

    if len(all_files) == 0:
        raise FileNotFoundError(
            "[Preparing Dataset] No HR files found across all splits. "
            "Please check your hr_glob or hr_dirs paths."
        )

    patch_shape = _pick_patch_size_all(all_files, scale, patch_cap)
    print(f"[Preparing Dataset] Patch shape = {patch_shape}, scale = {scale}.")

    # Create dirs
    for split_name in splits:
        for sub in ("Original_Data", "HR_Data", "LR_Data"):
            _ensure_dir(os.path.join(root, split_name, sub))
    counts = {k: {"hr": 0, "lr": 0} for k in splits}

    # Process files
    for split_name, files in splits.items():
        print(f"[Preparing Dataset] Processing {split_name} ({len(files)} files).")
        for fp in files:
            # Load and canonicalize quaternion data
            arr = np.load(fp, mmap_mode="r")  # (4,*spatial) or (H,W,4), etc.
            # Save original data in root/Original_Data/{split}/ for reference
            orig_out = os.path.join(
                root, split_name, "Original_Data", os.path.basename(fp)
            )
            _save_npy(orig_out, arr)

            arr = _format_quaternions(arr)  # (4,*spatial) or (4,H,W), etc.

            # Generate HR/LR patches
            lr_patch, hr_patch = get_patch_spatial(arr, patch_shape, scale)

            # Derive filenames
            base = os.path.splitext(os.path.basename(fp))[0]
            m = re.search(r"_([xyz])_block_(\d+)", base, re.IGNORECASE)
            if m:
                axis = m.group(1).lower()
                block_id = int(m.group(2))
            else:
                axis = "x"
                block_id = counts[split_name]["hr"] + 1

            hr_tag = (
                f"{dataset_name}_{split_name.lower()}_hr_{axis}_block_{block_id}.npy"
            )
            lr_tag = (
                f"{dataset_name}_{split_name.lower()}_lr_{axis}_block_{block_id}.npy"
            )

            hr_out = os.path.join(root, split_name, "HR_Data", hr_tag)
            lr_out = os.path.join(root, split_name, "LR_Data", lr_tag)

            _save_npy(hr_out, hr_patch)
            _save_npy(lr_out, lr_patch)
            counts[split_name]["hr"] += 1
            counts[split_name]["lr"] += 1

    # Metadata
    # Get proper symmetry string
    sym_canon = _canon_symmetry_str(symmetry)

    # Metadata
    created_at = (
        datetime.datetime.now(datetime.timezone.utc)
        .astimezone(pytz.timezone("America/Los_Angeles"))
        .isoformat()
    )

    dataset_info = {
        "dataset": dataset_name,
        "patch_shape": patch_shape,
        "scale": scale,
        "symmetry": sym_canon,
        "creator": creator,
        "contact": contact,
        "created_at": created_at,
        "counts": counts,
        "splits": {
            k: {
                "HR_glob": os.path.join(root, k, "HR_Data", "*.npy"),
                "LR_glob": os.path.join(root, k, "LR_Data", "*.npy"),
            }
            for k in splits
        },
    }

    with open(os.path.join(root, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(
        f"[Preparing Dataset] Done. dataset = {dataset_name}, patch_shape = {patch_shape}, scale = {scale}, symmetry = {sym_canon}. Saved to {root}.\n"
    )
    return dataset_info


def render_ipf_image(
    arr_hw4: np.ndarray,
    sym_class,
    out_png: Optional[str] = None,
    ref_dir: str = "ALL",
    include_key: bool = True,
    overwrite: bool = False,
):
    """Render quaternion orientation array to an IPF image with consistent formatting."""

    if arr_hw4.shape[-1] != 4:
        raise ValueError(f"Expected (H,W,4) input, got {arr_hw4.shape}")

    if out_png and not overwrite and os.path.exists(out_png):
        return out_png

    ori = Orientation(arr_hw4)
    ori.symmetry = sym_class
    ckey = orix_plot.IPFColorKeyTSL(sym_class.laue)

    ref_dir_lc = ref_dir.lower()
    show_all = ref_dir_lc == "all"

    ncols = 3 if show_all else 1
    key_cols = 1 if include_key else 0
    fig_cols = ncols + key_cols
    wr = [1] * ncols + ([0.9] if include_key else [])

    fig = plt.figure(
        constrained_layout=False,
        figsize=(5.2 * ncols + (2.6 if include_key else 0), 4.8),
    )
    gs = fig.add_gridspec(1, fig_cols, width_ratios=wr, wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]

    if show_all:
        for name, ax in zip(("X", "Y", "Z"), axes):
            ckey.direction = _DIRS[name]
            ax.imshow(ckey.orientation2color(ori))
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"IPF-{name}")
            ax.axis("off")
    else:
        ref = ref_dir.upper()
        if ref not in _DIRS:
            raise ValueError("ref_dir must be 'X','Y','Z','ALL'")
        ckey.direction = _DIRS[ref]
        axes[0].imshow(ckey.orientation2color(ori))
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_title(f"IPF-{ref}")
        axes[0].axis("off")

    if include_key:
        ax_ipf = fig.add_subplot(
            gs[0, -1], projection="ipf", symmetry=ori.symmetry.laue
        )
        ax_ipf.plot_ipf_color_key()
        ax_ipf.set_title("")

    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return out_png


class QuaternionDataset(Dataset):
    """
    Quaternion LR/HR dataset loader for structured datasets created with build_quaternion_sr_dataset.

    Parameters
    ----------
    dataset_root : str
        Path to dataset folder containing dataset_info.json
        (e.g. /data/warren/materials/EBSD/IN718_2D_SR_x4).
    split : {"Train","Val","Test"}, default="Train"
        Which split to load.
    take_first : int, optional
        If set, only use the first N HR files.
    check_integrity : bool, default=True
        If True, verify LR/HR shape consistency on first 20 pairs.
    fix_on_warn: bool = False
        If True, fix quaternions.

    Returns
    -------
    LR : torch.float32 (4,h,w)
        Low-resolution quaternion image.
    HR : torch.float32 (4,H,W)
        High-resolution quaternion image.
    """

    _NAME_RE = re.compile(
        r"^(?P<ds>.+)_(?P<split>train|val|test)_(?P<which>hr|lr)_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
        re.IGNORECASE,
    )

    @staticmethod
    def _parse_filename(fname: str) -> str:
        """Extract axis and block ID key from dataset filename."""
        m = QuaternionDataset._NAME_RE.match(os.path.basename(fname))
        if not m:
            raise ValueError(f"Unexpected file format: {fname}")
        return f"{m.group('axis').lower()}_{m.group('id')}"

    @staticmethod
    def _quat_first_shape(a) -> Tuple[int, ...]:
        """Ensure quaternion-first (4, *spatial) layout."""
        shp = a.shape if isinstance(a, np.ndarray) else tuple(a)
        if 4 not in shp:
            raise ValueError(
                f"[Integrity Error] No quaternion axis (size=4) in shape {shp}"
            )
        if shp[0] != 4:
            raise ValueError(f"[Integrity Error] Quaternion dimension not first: {shp}")
        return shp

    @staticmethod
    def _memmap_shape(path: str) -> Tuple[int, ...]:
        """Return the shape of a .npy file via memmap header (fast, no data load)."""
        try:
            mm = np.lib.format.open_memmap(path, mode="r")
            shape = mm.shape
            del mm
            return shape
        except Exception as e:
            raise ValueError(f"Failed to read shape of {path}: {e}")

    @staticmethod
    def _sample_quaternions(q: np.ndarray, n: int = 10000) -> np.ndarray:
        """Sample up to n quaternions from (4, *spatial) efficiently."""
        q = _to_quat_spatial(q)
        flat = q.reshape(4, -1)
        total = flat.shape[1]
        if total <= n:
            return flat
        idx = np.random.choice(total, n, replace=False)
        return flat[:, idx]

    @staticmethod
    def _fix_quaternion_file(path: str):
        """Normalize and hemisphere-align quaternion array in-place (safe overwrite)."""
        q = np.load(path, mmap_mode="r")
        q_fixed = _format_quaternions(q, normalize=True, enforce_hemisphere=True)
        tmp_path = f"{path}.fixed.npy"
        np.save(tmp_path, q_fixed)
        os.replace(tmp_path, path)  # atomic overwrite
        print(f"[Auto-Fix] Rewritten normalized file: {path}")

    def __init__(
        self,
        dataset_root: str,
        split: str = "Train",
        take_first: Optional[int] = None,
        check_integrity: bool = True,
        fix_on_warn: bool = True,
    ):
        # Allow passing dataset_info.json path directly
        if dataset_root.endswith("dataset_info.json"):
            info_path = dataset_root
        else:
            info_path = os.path.join(dataset_root, "dataset_info.json")

        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"Missing dataset_info.json: {info_path}")

        with open(info_path, "r") as f:
            self.info = json.load(f)

        self.split = split.capitalize()
        if self.split not in ("Train", "Val", "Test"):
            raise ValueError("split must be 'Train', 'Val', or 'Test'")

        # Resolve HR/LR globs from metadata
        hr_glob = self.info["splits"][self.split]["HR_glob"]
        lr_glob = self.info["splits"][self.split]["LR_glob"]
        hr_files = sorted(glob.glob(hr_glob))
        lr_files = sorted(glob.glob(lr_glob))
        if take_first:
            hr_files, lr_files = hr_files[:take_first], lr_files[:take_first]

        hr_map = {self._parse_filename(f): f for f in hr_files}
        lr_map = {self._parse_filename(f): f for f in lr_files}
        common_keys = sorted(hr_map.keys() & lr_map.keys())
        if not common_keys:
            raise RuntimeError("No matching HR/LR pairs found")

        self.pairs: List[Tuple[str, str]] = [
            (lr_map[k], hr_map[k]) for k in common_keys
        ]

        # Symmetry from metadata
        sym_str = self.info.get("symmetry", "Oh")
        self.sym_class = _resolve_symmetry(sym_str)
        self.ckey = orix_plot.IPFColorKeyTSL(self.sym_class.laue)

        # Optional integrity check
        if check_integrity:
            self.check_integrity(fix_on_warn=fix_on_warn)

    def check_integrity(
        self,
        n_check: int = 20,
        fix_on_warn: bool = False,
        check_normalization: bool = True,
        check_hemisphere: bool = True,
        check_fz: bool = True,
        fz_tol_deg: float = 0.5,
        sample_n: int = 10000,
        sample_all: bool = False,
        save_integrity_report: bool = True,
        plot_problem_ipfs: bool = True,
    ):
        """
        Validate dataset structure and quaternion correctness.

        Checks:
        - Quaternion-first layout (4, *spatial)
        - LR/HR scale consistency
        - Quaternion normalization (norm ~ 1) if check_normalization=True
        - Hemisphere canonicalization (s >= 0) if check_hemisphere=True
        - Fundamental zone inclusion if check_fz=True
        - Consistent shapes across all HR/LR pairs

        Sampling:
        - Uses up to `sample_n` quaternions per file by default.
        - If sample_all=True, all quaternions in each HR file are checked (slower).

        Reporting:
        - Automatically names CSV as `dataset_root/<split>_dataset_integrity_report.csv`.
        Example: "train_dataset_integrity_report.csv"
        - Report includes only files that failed one or more checks:
            index, file_path, n_sampled, needs_hemi_fix, needs_norm_fix,
            frac_outside_fz, mean_delta_deg, max_delta_deg
        """

        if not self.pairs:
            print("[Integrity Check] No file pairs to check.")
            return

        split_name = self.split.lower()

        scale = self.info.get("scale", None)
        n_check = min(n_check, len(self.pairs))
        print(
            f"[Integrity Check] Verifying first {n_check} LR/HR pairs ({split_name.upper()} split)..."
        )

        n_shape_warn, n_norm_warn, n_hemi_warn, n_fz_warn = 0, 0, 0, 0
        shape_hr_ref, shape_lr_ref = None, None
        integrity_records = []  # only problematic files

        fz_tol_rad = np.deg2rad(fz_tol_deg)
        sym = self.sym_class

        for i in range(n_check):
            lr_fp, hr_fp = self.pairs[i]

            try:
                hr_shape = self._memmap_shape(hr_fp)
                lr_shape = self._memmap_shape(lr_fp)
            except ValueError as e:
                raise ValueError(f"[Integrity Error] Shape read failed: {e}")

            # Check layout and scale
            self._quat_first_shape(hr_shape)
            self._quat_first_shape(lr_shape)

            if shape_hr_ref is None:
                shape_hr_ref, shape_lr_ref = hr_shape, lr_shape
            else:
                if hr_shape != shape_hr_ref or lr_shape != shape_lr_ref:
                    n_shape_warn += 1
                    warnings.warn(
                        f"[Integrity Warning] Shape mismatch at pair {i}: "
                        f"HR {hr_shape} vs ref {shape_hr_ref}, LR {lr_shape} vs ref {shape_lr_ref}"
                    )

            if scale and len(hr_shape) == len(lr_shape):
                hr_spatial, lr_spatial = hr_shape[1:], lr_shape[1:]
                expected = tuple(h // scale for h in hr_spatial)
                if lr_spatial != expected:
                    raise ValueError(
                        f"[Integrity Error] LR spatial {lr_spatial} != expected {expected} "
                        f"from HR {hr_spatial} (scale={scale})"
                    )

            # Sample quaternions
            try:
                hr = np.load(hr_fp, mmap_mode="r")
                n_total = np.prod(hr.shape[1:])
                if sample_all:
                    q_sample = hr.reshape(4, -1)
                    n_used = q_sample.shape[1]
                else:
                    n_used = min(sample_n, n_total)
                    idx = np.random.choice(n_total, n_used, replace=False)
                    q_sample = hr.reshape(4, -1)[:, idx]
                del hr
            except Exception as e:
                warnings.warn(
                    f"[Integrity Warning] Could not read quaternions from {hr_fp}: {e}"
                )
                continue

            needs_norm_fix = False
            needs_hemi_fix = False
            frac_out_fz, mean_angle_deg, max_angle_deg = 0.0, 0.0, 0.0

            # Normalization check
            if check_normalization:
                norms = np.linalg.norm(q_sample, axis=0)
                mean_norm = float(norms.mean())
                max_dev = float(np.abs(norms - 1).max())
                if not (0.98 <= mean_norm <= 1.02 and max_dev < 0.1):
                    warnings.warn(
                        f"[Integrity Warning] Non-unit quaternions in {os.path.basename(hr_fp)}: "
                        f"mean_norm={mean_norm:.3f}, max_dev={max_dev:.3f}"
                    )
                    n_norm_warn += 1
                    needs_norm_fix = True

            # Hemisphere check
            if check_hemisphere:
                mean_scalar = float(q_sample[0].mean())
                if mean_scalar < -0.01:
                    warnings.warn(
                        f"[Integrity Warning] Hemisphere not canonicalized in {os.path.basename(hr_fp)}: "
                        f"mean_scalar={mean_scalar:.3f}"
                    )
                    n_hemi_warn += 1
                    needs_hemi_fix = True

            # Fundamental zone check
            if check_fz:
                try:
                    ori = Orientation(q_sample.T, symmetry=sym)
                    ori_fz = ori.map_into_symmetry_reduced_zone()
                    mis = ori_fz.inv() * ori
                    ang = mis.angle
                    outside_mask = ang > fz_tol_rad
                    frac_out_fz = float(outside_mask.mean())
                    mean_angle_deg = float(np.rad2deg(ang.mean()))
                    max_angle_deg = float(np.rad2deg(ang.max()))

                    if frac_out_fz > 0.01:
                        warnings.warn(
                            f"[Integrity Warning] {frac_out_fz*100:.2f}% of orientations in "
                            f"{os.path.basename(hr_fp)} lie outside the FZ "
                            f"({sym.name}, tol={fz_tol_deg:.2f} deg)."
                        )
                        n_fz_warn += 1
                except Exception as e:
                    warnings.warn(
                        f"[Integrity Warning] FZ check failed for {hr_fp}: {e}"
                    )

            # Only record files that failed one or more checks
            if needs_norm_fix or needs_hemi_fix or frac_out_fz > 0.01:
                integrity_records.append(
                    (
                        i,
                        hr_fp,
                        n_used,
                        int(needs_hemi_fix),
                        int(needs_norm_fix),
                        frac_out_fz * 100,
                        mean_angle_deg,
                        max_angle_deg,
                    )
                )

                if fix_on_warn and (needs_norm_fix or needs_hemi_fix):
                    self._fix_quaternion_file(hr_fp)
                    if os.path.exists(lr_fp):
                        self._fix_quaternion_file(lr_fp)

        # Save CSV only if there are any problems
        if save_integrity_report and integrity_records:
            df = pd.DataFrame(
                integrity_records,
                columns=[
                    "index",
                    "file_path",
                    "n_sampled",
                    "needs_hemi_fix",
                    "needs_norm_fix",
                    "frac_outside_fz (%)",
                    "mean_delta_deg",
                    "max_delta_deg",
                ],
            )

            # Derive dataset root
            try:
                split_name = getattr(self, "split", None) or self.info.get(
                    "split", "unknown"
                )
                hr_glob = self.info["splits"][split_name]["HR_glob"]
                dataset_root = hr_glob.split(f"/{split_name}/")[0]
            except Exception:
                dataset_root = "."
            report_path = os.path.join(
                dataset_root, f"{split_name.lower()}_dataset_integrity_report.csv"
            )
            df.to_csv(report_path, index=False)

            print(f"\n[Integrity Check] Saved report -> {report_path}")

            if plot_problem_ipfs:
                self.plot_problem_ipfs_from_report(report_path)

        elif save_integrity_report:
            print(
                f"[Integrity Check] All {n_check} checked files passed. No report written."
            )

        # Summary
        total_warn = n_shape_warn + n_norm_warn + n_hemi_warn + n_fz_warn
        print("\n[Integrity Check] Summary:")
        print(f"  Verified pairs: {n_check}")
        print(f"  Reference HR shape: {shape_hr_ref}")
        print(f"  Reference LR shape: {shape_lr_ref}")
        print(
            f"  Sampling: {'ALL quaternions' if sample_all else f'{sample_n} per file'}"
        )

        if total_warn == 0:
            print(
                "  All shapes consistent, quaternions normalized, hemisphere aligned, and in FZ.\n"
            )
        else:
            print("  Issues detected:")
            if n_shape_warn:
                print(f"    Shape mismatches: {n_shape_warn} files")
            if check_normalization and n_norm_warn:
                print(f"    Normalization warnings: {n_norm_warn} files")
            if check_hemisphere and n_hemi_warn:
                print(f"    Hemisphere warnings: {n_hemi_warn} files")
            if check_fz and n_fz_warn:
                print(f"    FZ warnings: {n_fz_warn} files")

            print(
                f"\n  Auto-fix mode: {'ENABLED' if fix_on_warn else 'DISABLED (warnings only)'}\n"
            )

    def plot_problem_ipfs_from_report(
        self,
        csv_path: str = None,
        out_dir: str = None,
        ref_dir: str = "ALL",
        include_key: bool = True,
        overwrite: bool = True,
    ):
        """
        Render IPF maps for problematic quaternion files flagged in the integrity report.

        Parameters
        ----------
        csv_path : str, optional
            Path to the integrity report CSV. If None, it will be inferred from dataset info.
        out_dir : str, optional
            Output directory to save IPF previews.
            Defaults to "<dataset_root>/Problem_Ipfs/<split_name>/".
        ref_dir : {"X","Y","Z","ALL"}, default="ALL"
            Reference direction(s) for IPF coloring.
        include_key : bool, default=True
            Whether to include the IPF color key.
        overwrite : bool, default=True
            If False, skip rendering if preview already exists.

        Returns
        -------
        list[str]
            Paths to saved preview images.
        """
        # ---------------------
        # Determine dataset root and split name
        # ---------------------
        split_name = getattr(self, "split", None) or self.info.get("split", "unknown")

        if csv_path is None:
            try:
                hr_glob = self.info["splits"][split_name]["HR_glob"]
                dataset_root = hr_glob.split(f"/{split_name}/")[0]
                csv_path = os.path.join(
                    dataset_root, f"{split_name.lower()}_dataset_integrity_report.csv"
                )
            except Exception:
                raise FileNotFoundError(
                    "[Integrity Plot] Could not auto-detect report path. Please provide csv_path explicitly."
                )
        else:
            dataset_root = os.path.dirname(os.path.abspath(csv_path))

        # ---------------------
        # Default output directory
        # ---------------------
        if out_dir is None:
            out_dir = os.path.join(dataset_root, "Problem_Ipfs", split_name)

        # ---------------------
        # Load CSV
        # ---------------------
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[Integrity Plot] No entries found in {csv_path}.")
            return []

        # ---------------------
        # Detect frac_out_fz column
        # ---------------------
        possible_cols = [c for c in df.columns if "frac_out" in c.lower()]
        if not possible_cols:
            raise KeyError(
                f"No 'frac_out_fz' column found in {csv_path}. Columns = {list(df.columns)}"
            )
        frac_col = possible_cols[0]

        # ---------------------
        # Detect file column
        # ---------------------
        if "file" in df.columns:
            file_col = "file"
        elif "file_path" in df.columns:
            file_col = "file_path"
        else:
            raise KeyError(
                f"No file path column found in {csv_path}. Expected 'file' or 'file_path'."
            )

        # ---------------------
        # Build mask for problematic files
        # ---------------------
        mask = df[frac_col] > 0
        if "needs_normalization" in df.columns:
            mask |= df["needs_normalization"].astype(bool)
        if "needs_hemisphere_fix" in df.columns:
            mask |= df["needs_hemisphere_fix"].astype(bool)

        problem_df = df[mask]
        if problem_df.empty:
            print("[Integrity Plot] No problematic files found. Dataset appears clean.")
            return []

        os.makedirs(out_dir, exist_ok=True)
        saved_paths = []

        print(
            f"[Integrity Plot] Rendering {len(problem_df)} problematic files → {out_dir}"
        )
        for _, row in tqdm(problem_df.iterrows(), total=len(problem_df)):
            idx = int(row["index"])
            file_path = str(row[file_col])
            frac_val = float(row[frac_col])

            # --- Construct output file name ---
            base = os.path.basename(file_path).replace(".npy", "")
            out_png = os.path.join(out_dir, f"{idx:03d}_{base}_fz_{frac_val:.2f}.png")

            try:
                self.save_ipf_preview(
                    idx,
                    out_png,
                    which="HR",
                    ref_dir=ref_dir,
                    include_key=include_key,
                    overwrite=overwrite,
                )
                saved_paths.append(out_png)
            except Exception as e:
                print(f"[Integrity Plot] Skipping index {idx} ({file_path}): {e}")

        print(f"[Integrity Plot] Done. Saved {len(saved_paths)} previews → {out_dir}")
        return saved_paths

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lr_fp, hr_fp = self.pairs[idx]
        lr = np.load(lr_fp, mmap_mode="r")
        hr = np.load(hr_fp, mmap_mode="r")
        return _to_torch_quat_spatial(lr), _to_torch_quat_spatial(hr)

    def get_numpy_spatial_quat(self, idx: int):
        """
        Return LR and HR samples as NumPy arrays in (*spatial, 4) quaternion-last layout.

        This is analogous to __getitem__, but returns NumPy arrays instead of
        PyTorch tensors and ensures quaternion-last ordering for visualization
        or analysis (e.g. IPF coloring).

        Parameters
        ----------
        idx : int
            Dataset index.

        Returns
        -------
        (lr_spatial_quat, hr_spatial_quat) : tuple of np.ndarray
            Low- and high-resolution quaternion arrays, shape (*spatial, 4), dtype float32.
        """
        lr_fp, hr_fp = self.pairs[idx]

        # Efficiently load via memmap (no full RAM copy until needed)
        lr_mm = np.lib.format.open_memmap(lr_fp, mode="r")
        hr_mm = np.lib.format.open_memmap(hr_fp, mode="r")

        # Convert to quaternion-last layout
        lr = _to_spatial_quat(lr_mm)
        hr = _to_spatial_quat(hr_mm)

        # Ensure float32 and contiguous layout (for downstream ops)
        if lr.dtype != np.float32 or not lr.flags["C_CONTIGUOUS"]:
            lr = np.ascontiguousarray(lr, dtype=np.float32)
        if hr.dtype != np.float32 or not hr.flags["C_CONTIGUOUS"]:
            hr = np.ascontiguousarray(hr, dtype=np.float32)

        return lr, hr

    def save_ipf_preview(
        self,
        idx: int,
        out_png: str = "IPF_Preview.png",
        which: str = "HR",
        ref_dir: str = "ALL",
        include_key: bool = True,
        overwrite: bool = True,
    ):
        """
        Save a single IPF preview image for a given dataset sample.

        Loads quaternion data in (*spatial, 4) form using the dataset's symmetry class,
        then renders an IPF image (optionally including the color key).

        Parameters
        ----------
        idx : int
            Dataset index.
        out_png : str
            Output PNG path.
        which : {"HR", "LR"}, default="HR"
            Whether to render the high-resolution (HR) or low-resolution (LR) sample.
        ref_dir : {"X", "Y", "Z", "ALL"}, default="ALL"
            Reference direction(s) for IPF coloring.
        include_key : bool, default=True
            Whether to include the IPF color key in the figure.
        overwrite : bool, default=True
            If False, skip rendering if the image already exists.

        Returns
        -------
        out_png : str
            Path to the saved IPF image.
        """
        # Skip if file already exists and overwrite is False
        if not overwrite and os.path.exists(out_png):
            print(f"[IPF Preview] Skipping existing file: {out_png}")
            return out_png

        # Load quaternion data as (*spatial, 4)
        lr_np, hr_np = self.get_numpy_spatial_quat(idx)
        arr_spatial_quat = hr_np if which.upper() == "HR" else lr_np

        # Render and save
        out_path = render_ipf_image(
            arr_spatial_quat,
            self.sym_class,
            out_png=out_png,
            ref_dir=ref_dir,
            include_key=include_key,
            overwrite=overwrite,
        )

        print(f"[IPF Preview] Saved {which.upper()} image -> {out_path}")
        return out_path

    # Batch helper for many previews
    def save_ipf_many(
        self,
        indices: Iterable[int],
        out_dir: str,
        which: str = "HR",
        ref_dir: str = "ALL",
        include_key: bool = False,
    ):
        """
        Save many IPF maps efficiently:
          - same symmetry reused
          - default: single direction 'Z' and no color key (fast)
        """
        os.makedirs(out_dir, exist_ok=True)
        for i in indices:
            out_png = os.path.join(
                out_dir,
                f"{self.info.get('dataset')}_{which.lower()}_{i:04d}_ref_dir_{ref_dir}.png",
            )

            self.save_ipf_preview(
                i, out_png, which=which, ref_dir=ref_dir, include_key=include_key
            )


def save_dataset_ipfs(
    dataset_root: str,
    splits=("Train", "Val", "Test"),
    which_list=("HR", "LR"),
    ref_dir: str = "ALL",
    include_key: bool = True,
    overwrite: bool = False,
    num_workers: int = 4,
):
    """
    Save IPF images for all splits/which in a quaternion SR dataset.

    Skips execution entirely if IPF image folders already exist and contain images.
    """
    info_path = os.path.join(dataset_root, "dataset_info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"Missing dataset_info.json at {info_path}")
    with open(info_path, "r") as f:
        info = json.load(f)

    sym_class = _resolve_symmetry(info.get("symmetry", "Oh"))

    ref_dir = ref_dir.upper()
    if ref_dir not in ("X", "Y", "Z", "ALL"):
        raise ValueError("ref_dir must be 'X','Y','Z','ALL'")

    # ---- Early exit check: if all split/which IPF_Images dirs already exist & populated ----
    already_done = True
    for split in splits:
        for which in which_list:
            ipf_dir = os.path.join(
                dataset_root, split.capitalize(), f"{which.upper()}_IPF_Images"
            )
            if not os.path.isdir(ipf_dir):
                already_done = False
                break
            if not any(fname.endswith(".png") for fname in os.listdir(ipf_dir)):
                already_done = False
                break
        if not already_done:
            break

    if already_done and not overwrite:
        print(
            f"[Preparing IPF Images] Skipping: IPF images already exist in {dataset_root}"
        )
        return

    # ---- Build list of all files to process ----
    tasks = []
    for split in splits:
        split = split.capitalize()
        if split not in ("Train", "Val", "Test"):
            raise ValueError(f"Invalid split: {split}")
        for which in which_list:
            which = which.upper()
            if which not in ("HR", "LR"):
                raise ValueError(f"Invalid which: {which}")
            glob_pat = info["splits"][split][f"{which}_glob"]
            files = sorted(glob.glob(glob_pat))
            if not files:
                print(
                    f"[Preparing IPF Images] No files for split={split}, which={which}"
                )
                continue

            out_dir = os.path.join(dataset_root, split, f"{which}_IPF_Images")
            os.makedirs(out_dir, exist_ok=True)

            for fp in files:
                base = os.path.splitext(os.path.basename(fp))[0]
                out_png = os.path.join(out_dir, f"{base}_ref_{ref_dir.lower()}.png")
                tasks.append((fp, out_png))

    print(f"[Preparing IPF Images] {len(tasks)} total images to process")

    def process_file(fp, out_png):
        if not overwrite and os.path.exists(out_png):
            return None
        arr_hw4 = _to_spatial_quat(np.load(fp, mmap_mode="r"))
        return render_ipf_image(
            arr_hw4,
            sym_class,
            out_png=out_png,
            ref_dir=ref_dir,
            include_key=include_key,
            overwrite=overwrite,
        )

    # ---- Run tasks ----
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(process_file, *task) for task in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    _ = fut.result()
                except Exception as e:
                    print(f"[Preparing IPF Images] Error: {e}")
                if i % 50 == 0:
                    print(f"[Preparing IPF Images] {i}/{len(tasks)} done...")
    else:
        for i, task in enumerate(tasks, 1):
            try:
                _ = process_file(*task)
            except Exception as e:
                print(f"[Preparing IPF Images] Error on {task[0]}: {e}")
            if i % 50 == 0:
                print(f"[Preparing IPF Images] {i}/{len(tasks)} done...")

    print(f"[Preparing IPF Images] Completed saving IPFs -> {dataset_root}")


def save_dataset_ipf_summary(
    dataset,
    out_png: str,
    which: str = "HR",
    ref_dir: str = "Z",
    n_total: int = 100_000,
    per_file_max: int = 2000,
    include_key: bool = True,
    overwrite: bool = True,
    figsize: tuple = (7, 6),
):
    """
    Render and save a single IPF summary image representing all orientations
    in a quaternion SR dataset split (e.g., Train HR or LR).

    Parameters
    ----------
    dataset : QuaternionDataset
        Loaded dataset instance (e.g., train_ds, val_ds, test_ds).
    out_png : str
        Output PNG path for the saved IPF summary.
    which : {"HR", "LR"}, default="HR"
        Whether to visualize high-resolution (HR) or low-resolution (LR) data.
    ref_dir : {"X", "Y", "Z"}, default="Z"
        Reference direction for IPF coloring.
    n_total : int, default=100_000
        Approximate total number of orientations to plot across all files.
    per_file_max : int, default=2000
        Maximum number of orientations to sample from each file.
    include_key : bool, default=True
        Whether to include the IPF color key.
    overwrite : bool, default=True
        If False, skip rendering if the output file already exists.
    figsize : tuple, default=(7, 6)
        Figure size.
    grid_alpha : float, default=0.25
        Transparency for grid lines in the IPF plot.

    Returns
    -------
    out_png : str
        Path to the saved PNG file.
    """
    # Skip if file already exists and overwrite is False
    if not overwrite and os.path.exists(out_png):
        print(f"[IPF Summary] Skipping existing file: {out_png}")
        return out_png

    sym_class = dataset.sym_class
    ref_dir = ref_dir.upper()
    if ref_dir not in ("X", "Y", "Z"):
        raise ValueError("ref_dir must be 'X', 'Y', or 'Z'")

    # Initialize figure and IPF axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="ipf", symmetry=sym_class.laue)
    ax.set_aspect("equal", adjustable="box")

    title = f"{dataset.info['dataset']} {which.upper()} IPF-{ref_dir}"
    ax.set_title(title, pad=14)

    ipfkey = orix_plot.IPFColorKeyTSL(sym_class.laue)
    ipfkey.direction = Vector3d(
        {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[ref_dir]
    )

    # Determine sampling
    n_files = len(dataset.pairs)
    per_file = min(max(1, n_total // n_files), per_file_max)
    print(
        f"[IPF Summary] Sampling ~{per_file} orientations per file across {n_files} files..."
    )

    total_plotted = 0
    for i in range(n_files):
        try:
            lr_np, hr_np = dataset.get_numpy_spatial_quat(i)
            q = hr_np if which.upper() == "HR" else lr_np
            q_flat = q.reshape(-1, 4)
            total = q_flat.shape[0]
            if total > per_file:
                q_flat = q_flat[np.random.choice(total, per_file, replace=False)]

            O = Orientation(q_flat, symmetry=sym_class)
            rgb = ipfkey.orientation2color(O)
            O.scatter(
                projection="ipf",
                c=rgb,
                s=2,
                alpha=0.6,
                direction=ipfkey.direction,
                figure=fig,
            )

            total_plotted += q_flat.shape[0]
        except Exception as e:
            print(f"[IPF Summary] Skipping index {i}: {e}")

    print(f"[IPF Summary] Done. Total orientations plotted: {total_plotted}")

    # Include color key if requested
    if include_key:
        ax_ipf = fig.add_subplot(111, projection="ipf", symmetry=sym_class.laue)
        ax_ipf.plot_ipf_color_key()
        ax_ipf.set_title("")

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout(pad=1.4)
    fig.savefig(out_png, bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"[IPF Summary] Saved {which.upper()} summary -> {out_png}")
    return out_png


# -----------------------------------------------------------
# Render quaternion orientation array to RGB(s)
# -----------------------------------------------------------
def render_ipf_rgb(
    arr_hw4: np.ndarray,
    sym_class,
    ref_dir: str = "ALL",
):
    """Render quaternion orientation array to RGB image(s)."""
    if arr_hw4.shape[-1] != 4:
        raise ValueError(f"Expected (H,W,4) input, got {arr_hw4.shape}")

    ori = Orientation(arr_hw4)
    ori.symmetry = sym_class
    ckey = orix_plot.IPFColorKeyTSL(sym_class.laue)

    show_all = ref_dir.lower() == "all"
    directions = ("X", "Y", "Z") if show_all else (ref_dir.upper(),)
    rgb_list = []

    for d in directions:
        ckey.direction = _DIRS[d]
        rgb = ckey.orientation2color(ori)
        rgb_list.append(rgb)

    return rgb_list if show_all else rgb_list[0]


# -----------------------------------------------------------
# Save 3-panel IPF figure for XYZ directions
# -----------------------------------------------------------
def save_ipf_xyz_figure(rgb_list, out_path: str, title: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, rgb, d in zip(axes, rgb_list, ("X", "Y", "Z")):
        ax.imshow(rgb)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"IPF-{d}", fontsize=12)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=0.93)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# -----------------------------------------------------------
# Main visualization function
# -----------------------------------------------------------
def find_and_render_worst_fz_region(
    q_spatial: np.ndarray,
    sym_class,
    patch_size: int = 64,
    out_dir: str = "fz_patch_debug",
    base_name: str = "fz_region",
    ref_dir: str = "ALL",
    tol_deg: float = 0.5,
    stride: int = None,
):
    """
    Find patch with highest fraction of orientations outside FZ and render:
      1. Original IPF XYZ
      2. FZ-reduced IPF XYZ
      3. Original IPF XYZ with black overlay on outside-FZ pixels
      4. Red/black outside FZ mask
      5. Misorientation heatmap from FZ
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W, _ = q_spatial.shape
    stride = stride or patch_size

    # --- Compute outside mask ---
    ori_full = Orientation(q_spatial.reshape(-1, 4), symmetry=sym_class)

    q_spatial_fz, _ = reduce_to_fz_min_angle_fast(q_spatial, symmetry=sym_class)
    ori_fz_full = Orientation(q_spatial_fz.reshape(-1, 4), symmetry=sym_class)
    # ori_fz_full = ori_full.map_into_symmetry_reduced_zone()
    mis_full = ori_fz_full.inv() * ori_full
    ang_full = mis_full.angle.reshape(H, W)
    outside_mask_full = ang_full > np.deg2rad(tol_deg)

    # --- Find worst patch ---
    best_frac = -1.0
    best_yx = (0, 0)
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            frac = outside_mask_full[y : y + patch_size, x : x + patch_size].mean()
            if frac > best_frac:
                best_frac = frac
                best_yx = (y, x)

    y0, x0 = best_yx
    patch = q_spatial[y0 : y0 + patch_size, x0 : x0 + patch_size, :]
    outside_mask_patch = outside_mask_full[y0 : y0 + patch_size, x0 : x0 + patch_size]
    ang_patch = ang_full[y0 : y0 + patch_size, x0 : x0 + patch_size]

    print(f"[FZ] Best patch at (y={y0}, x={x0}) with {best_frac*100:.2f}% outside FZ")

    # --- FZ reduced patch ---

    # q_spatial_fz, _ = reduce_to_fz_min_angle_fast(q_spatial, symmetry=sym_class)
    # ori_fz_full = Orientation(q_spatial_fz.reshape(-1, 4), symmetry=sym_class)

    # ori_patch = Orientation(patch.reshape(-1, 4), symmetry=sym_class)

    patch_fz, _ = reduce_to_fz_min_angle_fast(patch, symmetry=sym_class)
    # ori_patch_fz = ori_patch.map_into_symmetry_reduced_zone()
    # patch_fz = ori_patch_fz.data.reshape(patch.shape)

    frac_tag = f"{best_frac*100:.2f}"

    # --- IPF RGB render ---
    rgb_orig_list = render_ipf_rgb(patch, sym_class, ref_dir="ALL")
    rgb_fz_list = render_ipf_rgb(patch_fz, sym_class, ref_dir="ALL")

    # --- Blackout overlay ---
    blackout_list = []
    for rgb in rgb_orig_list:
        blackout = rgb.copy()
        blackout[outside_mask_patch] = [0.0, 0.0, 0.0]
        blackout_list.append(blackout)

    # --- Save figures ---
    orig_out = os.path.join(out_dir, f"{base_name}_orig_xyz_y{y0}_x{x0}_{frac_tag}.png")
    fz_out = os.path.join(out_dir, f"{base_name}_fz_xyz_y{y0}_x{x0}_{frac_tag}.png")
    blackout_out = os.path.join(
        out_dir, f"{base_name}_black_xyz_y{y0}_x{x0}_{frac_tag}.png"
    )
    unfolded_out = os.path.join(out_dir, f"{base_name}")
    mask_png = os.path.join(out_dir, f"{base_name}_FZ_mask_y{y0}_x{x0}_{frac_tag}.png")
    heatmap_png = os.path.join(
        out_dir, f"{base_name}_FZ_misorientation_y{y0}_x{x0}_{frac_tag}.png"
    )

    save_ipf_xyz_figure(
        rgb_orig_list, orig_out, title=f"Original Patch (Outside FZ: {frac_tag}%)"
    )
    save_ipf_xyz_figure(rgb_fz_list, fz_out, title="FZ Reduced Patch")
    save_ipf_xyz_figure(
        blackout_list, blackout_out, title="Original with Outside FZ Blacked Out"
    )

    # --- Red mask ---
    mask_rgb = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
    mask_rgb[outside_mask_patch] = [1.0, 0.0, 0.0]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask_rgb)
    ax.axis("off")
    ax.set_title(f"Outside FZ ({frac_tag}%)")
    fig.savefig(mask_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # --- Misorientation heatmap ---
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(np.degrees(ang_patch), cmap="inferno")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Angle to FZ [°]", fontsize=10)
    ax.axis("off")
    ax.set_title("FZ Misorientation Heatmap")
    fig.savefig(heatmap_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    plot_unfolded_ipf_with_symmetry_and_fz(
        q_spatial,
        ref_dir="X",
        tol_deg=tol_deg,
        out_png=os.path.join(
            out_dir,
            f"{base_name}_ipf_X_unfolded.png",
        ),
    )

    plot_unfolded_ipf_with_symmetry_and_fz(
        q_spatial,
        ref_dir="Y",
        tol_deg=tol_deg,
        out_png=os.path.join(
            out_dir,
            f"{base_name}_ipf_Y_unfolded.png",
        ),
    )
    plot_unfolded_ipf_with_symmetry_and_fz(
        q_spatial,
        ref_dir="Z",
        tol_deg=tol_deg,
        out_png=os.path.join(
            out_dir,
            f"{base_name}_ipf_Z_unfolded.png",
        ),
    )

    print(f"[FZ] Saved multi-panel IPF + blackout + mask + heatmap -> {out_dir}")

    return {
        "best_yx": (y0, x0),
        "frac_outside": best_frac,
        "outside_mask_patch": outside_mask_patch,
        "patch_original": patch,
        "patch_fz": patch_fz,
        "mask_png": mask_png,
        "orig_png": orig_out,
        "fz_png": fz_out,
        "blackout_png": blackout_out,
        "heatmap_png": heatmap_png,
    }


def plot_unfolded_ipf_with_symmetry_and_fz(
    arr_hw4,
    sym_class="Oh",
    ref_dir="Z",
    max_points=None,
    highlight_outside_fz=True,
    tol_deg=0.5,
    out_png="unfolded_ipf_fz.png",
):
    """
    Plot unfolded IPF directions with symmetry axes and FZ boundary for cubic (or any) symmetry.

    Parameters
    ----------
    arr_hw4 : ndarray
        Quaternion array, (H,W,4) or (4,H,W), scalar-last or first.
    sym_class : str or orix symmetry, default="Oh"
        Symmetry class to use.
    ref_dir : str, default="Z"
        Crystal direction to project.
    max_points : int, default=None
        Max orientations to plot, default is all.
    highlight_outside_fz : bool, default=True
        Whether to highlight orientations lying outside FZ.
    tol_deg : float, default=0.5
        Misorientation tolerance (deg) for outside FZ classification.
    out_png : str
        Path to save PNG figure.
    """
    # --- Handle input array ---
    arr_hw4_shape = arr_hw4.shape
    if arr_hw4.ndim == 3 and arr_hw4.shape[0] == 4:
        arr_hw4 = np.moveaxis(arr_hw4, 0, -1)
    q_flat = arr_hw4.reshape(-1, 4)
    # Normalize
    q_flat /= np.linalg.norm(q_flat, axis=1, keepdims=True)
    flip = q_flat[:, 0] < 0
    q_flat[flip] *= -1

    if max_points and q_flat.shape[0] > max_points:
        idx = np.random.choice(q_flat.shape[0], max_points, replace=False)
        q_flat = q_flat[idx]

    # --- ORIX orientation ---
    sym = getattr(symmetry, sym_class) if isinstance(sym_class, str) else sym_class
    ori = Orientation(q_flat, symmetry=sym)

    q_flat_FZ, _ = reduce_to_fz_min_angle_fast(
        q_flat.reshape(arr_hw4_shape), symmetry=sym
    )

    ori_fz = Orientation(q_flat_FZ.reshape(q_flat.shape), symmetry=sym)

    # ori_fz = ori.map_into_symmetry_reduced_zone()

    # --- Ref direction ---
    ref_map = {
        "X": Vector3d.xvector(),
        "Y": Vector3d.yvector(),
        "Z": Vector3d.zvector(),
    }
    v_ref = ref_map[ref_dir.upper()]

    v_unfolded = ori * v_ref
    v_reduced = ori_fz * v_ref

    # Determine which points lie outside FZ
    outside_mask = None
    if highlight_outside_fz:
        mis = ori_fz.inv() * ori
        ang = np.rad2deg(mis.angle)
        outside_mask = ang > tol_deg
        frac_outside = outside_mask.mean() * 100
    else:
        frac_outside = 0.0

    # --- Plot ---
    fig, ax = plt.subplots(subplot_kw={"projection": "stereographic"}, figsize=(6, 6))
    ax.set_title(
        f"{sym.name} Unfolded IPF ({ref_dir}) — Outside FZ: {frac_outside:.2f}%",
        pad=20,
    )

    # Plot points
    if highlight_outside_fz and outside_mask is not None:
        ax.scatter(
            v_unfolded[~outside_mask], c="grey", s=5, alpha=0.5, label="Inside FZ"
        )
        if np.any(outside_mask):
            ax.scatter(
                v_unfolded[outside_mask], c="green", s=5, alpha=0.7, label="Outside FZ"
            )
    else:
        ax.scatter(v_unfolded, c="grey", s=5, alpha=0.5)

    # Draw symmetry axes (432 example)
    marker_size = 200
    v4fold = Vector3d([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    ax.draw_circle(v4fold, color="blue")

    v3fold = Vector3d([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])

    ax.draw_circle(v3fold, color="red")

    v2fold = Vector3d(
        [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [1, 1, 0],
            [-1, -1, 0],
            [-1, 1, 0],
            [1, -1, 0],
        ]
    )
    ax.draw_circle(v2fold, color="blue")

    # Draw FZ boundary manually
    sector = sym.fundamental_sector
    original_pole = deepcopy(sector._pole)
    sector._pole = ax.pole
    edges = sector.edges
    sector._pole = original_pole
    x, y, _ = ax._pretransform_input((edges,))
    patch = mpatches.PathPatch(
        mpath.Path(np.column_stack([x, y]), closed=True),
        facecolor="none",
        edgecolor="grey",
        linewidth=1.5,
        alpha=0.9,
        zorder=5,
    )
    ax.add_patch(patch)

    ax.set_labels("RD", "TD", None)
    ax.show_hemisphere_label()
    ax.legend(loc="upper right", fontsize=10)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from orix.quaternion import Orientation, symmetry as SYM


def plot_symmetry_operator_map(q_spatial, sym_class=SYM.Oh, return_indices=True):
    """
    Compute and plot the symmetry operator used to map each quaternion
    to the fundamental zone for cubic symmetry, and return the pixel
    indices of non-identity operations.

    Parameters
    ----------
    q_spatial : ndarray (H, W, 4)
        Quaternion array representing orientations (scalar first).
    sym_class : orix symmetry, optional
        Symmetry class (default: SYM.Oh).
    return_indices : bool, default=True
        Whether to return indices of pixels that required FZ mapping.

    Returns
    -------
    results : dict
        {
            "op_map": 2D array of operator IDs per pixel,
            "unique_ids": unique symmetry operator IDs present,
            "counts": pixel counts per operator,
            "non_identity_indices": list of (y,x) for non-identity pixels (if return_indices=True)
        }
    """
    # -------------------------------------------------------------------------
    # 1. Compute misorientation between original and FZ-reduced orientation
    # -------------------------------------------------------------------------
    H, W, _ = q_spatial.shape
    q_flat = q_spatial.reshape(-1, 4)
    o_orig = Orientation(q_flat, sym_class)
    o_fz = o_orig.map_into_symmetry_reduced_zone()
    mis = o_fz * o_orig.inv()
    mis_quats = mis.data.reshape(-1, 4)

    # -------------------------------------------------------------------------
    # 2. Determine which symmetry operator matches each misorientation
    # -------------------------------------------------------------------------
    ops = sym_class.data
    sym_quats = ops.copy()

    def closest_operator_index(q):
        q = q / np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        dots = np.abs(np.dot(sym_quats, q))
        return np.argmax(dots)

    op_idx = np.array([closest_operator_index(q) for q in mis_quats])

    # -------------------------------------------------------------------------
    # 3. Build operator index map (2D)
    # -------------------------------------------------------------------------
    op_map = op_idx.reshape(H, W)
    unique_ids, counts = np.unique(op_map, return_counts=True)
    num_present = len(unique_ids)

    # -------------------------------------------------------------------------
    # 4. Remap IDs to 0..num_present-1 for plotting
    # -------------------------------------------------------------------------
    id_to_idx = {int(op_id): idx for idx, op_id in enumerate(unique_ids)}
    remapped = np.vectorize(id_to_idx.get)(op_map)

    # -------------------------------------------------------------------------
    # 5. Plot operator index map
    # -------------------------------------------------------------------------
    cmap = plt.get_cmap("tab20", num_present)
    bounds = np.arange(num_present + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(remapped, cmap=cmap, norm=norm, interpolation="nearest")

    # -------------------------------------------------------------------------
    # 6. Build colorbar labels (quaternion → angle & axis)
    # -------------------------------------------------------------------------
    labels = []
    for op_id in unique_ids:
        op_id = int(op_id)
        w, x, y, z = ops[op_id]
        angle = 2 * np.arccos(w)
        angle_deg = np.degrees(angle)
        if np.isclose(angle_deg, 0, atol=1e-8):
            axis = np.array([0.0, 0.0, 0.0])
        else:
            axis = np.array([x, y, z]) / np.sin(angle / 2.0)
            axis /= np.linalg.norm(axis)
        axis_str = f"[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]"
        labels.append(f"{op_id:02d} ({angle_deg:.0f}°, {axis_str})")

    cbar = plt.colorbar(im, ticks=np.arange(num_present), ax=ax)
    cbar.ax.set_yticklabels(labels)
    ax.set_title("Symmetry Operator Applied")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 7. Get pixel indices for non-identity operator (if requested)
    # -------------------------------------------------------------------------
    non_identity_indices = None
    if return_indices:
        mask = op_map != 0  # 0 corresponds to identity operator
        non_identity_indices = np.argwhere(mask)

    return {
        "op_map": op_map,
        "unique_ids": unique_ids,
        "counts": counts,
        "non_identity_indices": non_identity_indices,
    }


import numpy as np
from orix.quaternion import Orientation, symmetry as SYM
from orix.quaternion.orientation_region import OrientationRegion


def reduce_to_fz_min_angle_fast(q_spatial, symmetry=SYM.Oh, batch_size=500000):
    """
    DREAM.3D-style minimum-angle reduction to the fundamental zone (FZ)
    with vectorized symmetry operations for large orientation maps.

    Parameters
    ----------
    q_spatial : (H, W, 4) ndarray (float)
        Input quaternion array (scalar-first, normalized).
    sym : orix symmetry (default: SYM.Oh)
        Symmetry group.
    batch_size : int
        Number of pixels to process per batch (controls memory use).

    Returns
    -------
    q_fz_min : (H, W, 4) ndarray
        Minimum-angle FZ-reduced quaternion field.
    op_idx_map : (H, W) ndarray (int)
        Symmetry operator index used for each pixel.
    """
    H, W, _ = q_spatial.shape
    n_pix = H * W
    q_flat = q_spatial.reshape(n_pix, 4)
    q_flat /= np.linalg.norm(q_flat, axis=1, keepdims=True)

    ops = symmetry.data  # (Nops, 4)
    n_ops = ops.shape[0]
    region = OrientationRegion.from_symmetry(symmetry)

    q_out = np.empty_like(q_flat)
    op_idx_out = np.zeros(n_pix, dtype=np.int32)

    for start in range(0, n_pix, batch_size):
        end = min(start + batch_size, n_pix)
        batch = q_flat[start:end]  # (B,4)
        B = batch.shape[0]

        # Expand to (B, Nops, 4)
        bq = batch[:, None, :]  # (B,1,4)
        sops = ops[None, :, :]  # (1,Nops,4)

        # Quaternion multiply: cand = s * q
        w0, x0, y0, z0 = np.moveaxis(sops, -1, 0)
        w1, x1, y1, z1 = np.moveaxis(bq, -1, 0)

        cand = np.empty((B, n_ops, 4), dtype=np.float64)
        cand[..., 0] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        cand[..., 1] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        cand[..., 2] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        cand[..., 3] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

        # Normalize candidates (vectorized)
        cand /= np.linalg.norm(cand, axis=2, keepdims=True)

        # Check FZ inclusion for all candidates
        cand_flat = cand.reshape(-1, 4)
        o_cand = Orientation(cand_flat, symmetry)
        inside_mask = (o_cand < region).reshape(B, n_ops)

        # Compute misorientation angle from identity: 2*arccos(w)
        angles = 2 * np.arccos(np.clip(cand[..., 0], -1.0, 1.0))
        angles[~inside_mask] = np.inf

        # Pick min angle operator for each pixel
        best_idx = np.argmin(angles, axis=1)
        best_q = cand[np.arange(B), best_idx]

        q_out[start:end] = best_q
        op_idx_out[start:end] = best_idx

    return q_out.reshape(H, W, 4), op_idx_out.reshape(H, W)


if __name__ == "__main__":
    # Example usage
    dataset_out_root = "/data/warren/materials/EBSD"
    dataset_name = "IN718_2D_SR_x4"
    dataset_dir = os.path.join(dataset_out_root, dataset_name)

    # dataset_info = build_quaternion_sr_dataset(
    #     hr_dirs={
    #         "Train": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Train/HR_Images/*.npy",
    #         "Val": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Val/HR_Images/preprocessed_imgs_all_Blocks/*.npy",
    #         "Test": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/*.npy",
    #     },
    #     out_root=dataset_out_root,
    #     dataset_name=dataset_name,
    #     scale=4,
    #     symmetry="Oh",
    #     creator="Warren Zamudio",
    #     contact="wzamudio@ucsb.edu",
    # )

    # save_dataset_ipfs(
    #     dataset_root=dataset_dir,
    #     splits=("Train", "Val", "Test"),
    #     which_list=("HR", "LR"),
    #     ref_dir="ALL",
    #     include_key=True,
    #     overwrite=False,
    #     num_workers=16,  # adjust for CPU cores
    # )

    train_ds = QuaternionDataset(dataset_dir, split="Train")

    # plot_unfolded_ipf_with_symmetry_and_fz(
    #     q_spatial,
    #     ref_dir="X",
    #     tol_deg=0,
    #     out_png="fz_patch_debug/indx_179/ipf_X_unfolded.png",
    # )

    # indx = 254
    indx = 179
    _, q_spatial = train_ds.get_numpy_spatial_quat(indx)

    render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)
    
    q_fz_min, op_map = reduce_to_fz_min_angle_fast(
    q_spatial, symmetry=SYM.Oh, batch_size=100000000
)

    render_ipf_image(q_fz_min, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

    result = find_and_render_worst_fz_region(
        q_spatial,
        train_ds.sym_class,
        patch_size=128,
        out_dir=f"fz_debug/indx_{indx}",
        base_name="quats",
        ref_dir="ALL",  # X, Y, Z IPF renders, but only ONE mask image
        tol_deg=0.1,
    )
    q_fz_min, op_map = reduce_to_fz_min_angle_fast(
        q_spatial, symmetry=SYM.Oh, batch_size=100000000
    )
    result = find_and_render_worst_fz_region(
        q_fz_min,
        train_ds.sym_class,
        patch_size=128,
        out_dir=f"fz_debug/indx_{indx}_FZ",
        base_name="quats_fz",
        ref_dir="ALL",  # X, Y, Z IPF renders, but only ONE mask image
        tol_deg=0.1,
    )
    # quat_scalar_first = _format_quaternions(
    #     np.load(
    #         "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Train/HR_Images/Open_718_Train_hr_x_block_327.npy"
    #     ),
    #     normalize=False,
    #     enforce_hemisphere=False,
    # )

    # q_spatial = _to_spatial_quat(quat_scalar_first)

    q_fz_min, op_map = reduce_to_fz_min_angle_fast(
        q_spatial, symmetry=SYM.Oh, batch_size=100000000
    )
    q_fz_min_format = _to_spatial_quat(_format_quaternions(q_fz_min))

    print("Unique operators used:", np.unique(op_map))
    print("Swapped pixel count:", np.sum(op_map != 0))
    render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)
    render_ipf_image(q_fz_min, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

    result = find_and_render_worst_fz_region(
        q_fz_min_format,
        train_ds.sym_class,
        patch_size=300,
        out_dir=f"fz_debug/indx_327",
        base_name="quats",
        ref_dir="ALL",  # X, Y, Z IPF renders, but only ONE mask image
        tol_deg=0.1,
    )

#     # q_spatial.shape

#     # q_spatial[:, :, 0]

#     render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

#     q_fz_norm = _to_spatial_quat(_format_quaternions(q_spatial_fz))

#     render_ipf_image(q_fz_norm, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

#     q_spatial.shape

#     ori = Orientation(q_spatial.reshape(-1, 4), symmetry=train_ds.sym_class)

#     ori_fz, idx_left, idx_right = ori.map_into_symmetry_reduced_zone_with_ops(True)

#     q_spatial_fz = ori_fz.data.reshape(q_spatial.shape)

#     test_q = q_spatial[72, 75]
#     ori = Orientation(test_q, symmetry=SYM.Oh)

#     ori_fz = ori.map_into_symmetry_reduced_zone()

#     test_q_fz = ori_fz.data

#     test_q * SYM.Oh.data[1]


# q_fz_min, op_map = reduce_to_fz_min_angle_fast(q_spatial, symmetry=SYM.Oh, batch_size=250000)
# print("Unique operators used:", np.unique(op_map))
# print("Swapped pixel count:", np.sum(op_map != 0))
# render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)


# ori = Orientation(q_spatial.reshape(-1, 4), symmetry=SYM.Oh)
# ori_fz = ori.map_into_symmetry_reduced_zone()

# q_spatial_fz = ori_fz.data.reshape(q_spatial.shape)

# render_ipf_image(q_spatial, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)

# render_ipf_image(q_fz_min, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)
# render_ipf_image(q_spatial_fz, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)


# plot_unfolded_ipf_with_symmetry_and_fz(
#     q_spatial,
#     ref_dir="Z",
#     tol_deg=0,
#     out_png="fz_patch_debug/indx_179_FZ/ipf_Z_unfolded.png",
# )

# plot_unfolded_ipf_with_symmetry_and_fz(
#     q_spatial,
#     ref_dir="X",
#     tol_deg=0,
#     out_png="fz_patch_debug/indx_179/ipf_X_unfolded.png",
# )

# plot_unfolded_ipf_with_symmetry_and_fz(
#     q_spatial,
#     ref_dir="Y",
#     tol_deg=0,
#     out_png="fz_patch_debug/indx_179/ipf_Y_unfolded.png",
# )

# train_ds.check_integrity(10000, sample_all=True)

# save_dataset_ipf_summary(
#     train_ds,
#     out_png=os.path.join(dataset_dir, "Train", "IPF_Z_HR.png"),
#     which="HR",
#     ref_dir="Z",
#     n_total=5214208,
#     per_file_max=4096,
#     include_key=False,
# )

# val_ds = QuaternionDataset(dataset_dir, split="Val")
# test_ds = QuaternionDataset(dataset_dir, split="Test")

# # a = np.load(
# #     "/data/warren/materials/EBSD/IN718_2D_SR_x4/Train/HR_Data/IN718_2d_sr_x4_train_hr_x_block_1.npy"
# # )

# a = np.load(
#     "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/preprocessed_imgs_all_Block/Open_718_Test_hr_x_block_0.npy"
# )

# q = np.concatenate([a[..., 3:4], a[..., :3]], axis=-1)

# render_ipf_image(a, SYM.O, ref_dir="ALL", include_key=True, overwrite=True)
# render_ipf_image(q, SYM.Oh, ref_dir="ALL", include_key=True, overwrite=True)
