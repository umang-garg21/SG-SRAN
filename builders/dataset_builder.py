# -*- coding:utf-8 -*-
"""
File:        dataset_builder.py
Created at:  2025/10/17
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Quaternion SR dataset builder.
"""

import os
import re
import glob
import json
import random
import datetime
import numpy as np
import pytz
from typing import Dict, Optional
from tqdm import tqdm

# ---------------------------
# Imports from utils
# ---------------------------
import sys
from pathlib import Path

# Add parent directory to path to allow imports from utils
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.quat_ops import format_quaternions
from utils.symmetry_utils import canon_symmetry_str, resolve_symmetry
from utils.dataset_utils import (
    last_int_key,
    ensure_dir,
    save_npy,
    pick_patch_size_all,
    random_aligned_patches,
)

# ---------------------------
# Dataset Builder
# ---------------------------


def build_quaternion_sr_dataset(
    hr_glob: Optional[str] = None,
    out_root: str = "datasets",
    dataset_name: str = "IN718",
    scale: int = 4,
    hr_dirs: Optional[Dict[str, str]] = None,
    split: Dict[str, float] = {"Train": 0.8, "Val": 0.1, "Test": 0.1},
    take_first: Optional[int] = None,
    patch_cap: Optional[int] = None,
    seed: int = 1234,
    symmetry: str = "Oh",
    normalize: bool = True,
    hemisphere: bool = True,
    reduce_to_fz: bool = False,
    creator: str = "Unknown",
    contact: str = "unknown@example.com",
) -> Dict:
    """
    Build quaternion super-resolution dataset.

    Parameters
    ----------
    hr_glob : str, optional
        Glob pattern for HR .npy files (used if hr_dirs not provided).
    out_root : str
        Root directory for dataset.
    dataset_name : str
        Name of dataset folder.
    scale : int
        Downsampling scale.
    hr_dirs : dict, optional
        Explicit HR dirs: {"Train": "...", "Val": "...", "Test": "..."}.
    split : dict
        Ratios for train/val/test split if auto-splitting HR files.
    take_first : int, optional
        Limit number of HR files.
    patch_cap : int, optional
        Cap maximum patch size.
    seed : int
        Random seed for reproducibility.
    symmetry : str
        Symmetry group label.
    normalize : bool
        Whether to normalize quaternions.
    hemisphere : bool
        Whether to enforce hemisphere.
    reduce_to_fz : bool
        Whether to reduce to Fundamental Zone (FZ).
    creator : str
        Dataset creator metadata.
    contact : str
        Contact email metadata.

    Returns
    -------
    dict
        Dataset metadata dictionary.
    """
    # ---------------------------
    # Setup and metadata
    # ---------------------------
    root = os.path.join(out_root, dataset_name)
    info_path = os.path.join(root, "dataset_info.json")

    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            return json.load(f)

    random.seed(seed)
    sym_canon = canon_symmetry_str(symmetry)
    sym_obj = resolve_symmetry(symmetry)

    # ---------------------------
    # Gather file list / splits
    # ---------------------------
    if hr_dirs:
        splits = {
            name.capitalize(): sorted(
                glob.glob(path, recursive=True), key=last_int_key
            )[: take_first or None]
            for name, path in hr_dirs.items()
        }
    elif hr_glob:
        all_files = sorted(glob.glob(hr_glob, recursive=True), key=last_int_key)[
            : take_first or None
        ]
        n = len(all_files)
        n_train = int(round(n * split["Train"]))
        n_val = int(round(n * split["Val"]))
        splits = {
            "Train": all_files[:n_train],
            "Val": all_files[n_train : n_train + n_val],
            "Test": all_files[n_train + n_val :],
        }
    else:
        raise ValueError("Provide either hr_dirs or hr_glob.")

    all_files = [f for fl in splits.values() for f in fl]
    if not all_files:
        raise FileNotFoundError("No HR files found.")

    # ---------------------------
    # Patch size calculation
    # ---------------------------
    patch_shape = pick_patch_size_all(all_files, scale, patch_cap)
    print(
        f"[Creating Dataset] Files={len(all_files)} | Patch={patch_shape} | Scale={scale}"
    )

    # Create output structure
    for s in splits:
        for sub in ("Original_Data", "HR_Data", "LR_Data"):
            ensure_dir(os.path.join(root, s, sub))
    counts = {k: {"hr": 0, "lr": 0} for k in splits}

    # ---------------------------
    # Main processing loop
    # ---------------------------
    for split_name, files_list in splits.items():
        print(f"[Creating Dataset] {split_name}: {len(files_list)} files")

        for fp in tqdm(files_list, desc=f"Processing {split_name}", unit="file"):
            # for fp in files_list:
            q_arr = np.load(fp, mmap_mode="r")
            save_npy(
                os.path.join(root, split_name, "Original_Data", os.path.basename(fp)),
                q_arr,
            )

            # Quaternion formatting (normalize + hemisphere + FZ)
            q_cf = format_quaternions(
                q_arr,
                normalize=normalize,
                hemisphere=hemisphere,
                reduce_fz=reduce_to_fz,
                sym=sym_canon if reduce_to_fz else None,
                quat_first=True,
            )

            # Extract HR/LR patches
            lr_patch, hr_patch = random_aligned_patches(q_cf, patch_shape, scale)

            # Naming
            base = os.path.splitext(os.path.basename(fp))[0]
            m = re.search(r"_([xyz])_block_(\d+)", base, re.IGNORECASE)
            axis = m.group(1).lower() if m else "x"
            block_id = int(m.group(2)) if m else counts[split_name]["hr"] + 1
            hr_tag = (
                f"{dataset_name}_{split_name.lower()}_hr_{axis}_block_{block_id}.npy"
            )
            lr_tag = (
                f"{dataset_name}_{split_name.lower()}_lr_{axis}_block_{block_id}.npy"
            )

            save_npy(os.path.join(root, split_name, "HR_Data", hr_tag), hr_patch)
            save_npy(os.path.join(root, split_name, "LR_Data", lr_tag), lr_patch)

            counts[split_name]["hr"] += 1
            counts[split_name]["lr"] += 1

    # ---------------------------
    # Metadata
    # ---------------------------
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
        "formatting": {
            "normalize": normalize,
            "hemisphere": hemisphere,
            "reduce_fz": reduce_to_fz,
        },
    }

    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"[Creating Dataset] Completed, saved json -> {info_path}")

    return dataset_info
