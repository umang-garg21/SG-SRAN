# -*-coding:utf-8 -*-
"""
File:        quaternion_dataset.py
Created at:  2025/10/18 12:17:06
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Quaternion super-resolution dataset class.
"""


import math
import os
import re
import glob
import json
import torch
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from utils.quat_ops import to_spatial_quat


class QuaternionDataset(Dataset):
    """
    Quaternion LR/HR dataset loader for structured datasets created with build_quaternion_sr_dataset.

    Parameters
    ----------
    dataset_root : str
        Path to dataset folder containing dataset_info.json.
    split : {"Train","Val","Test"}, default="Train"
        Which split to load.
    take_first : int, optional
        If set, only use the first N HR files.
    preload : bool, default=False
        Preload all data into CPU RAM at init (fastest per step).
    preload_torch : bool, default=False
        Preload as torch tensors (skip conversion in __getitem__).
    preload_workers : int, default=8
        Number of threads used for parallel preload.
    pin_memory : bool, default=True
        Pin tensor memory to accelerate GPU transfer.

    Returns
    -------
    LR : torch.float32 (4,h,w) or (h,w,4)
        Low-resolution quaternion image.
    HR : torch.float32 (4,H,W) or (H,W,4)
        High-resolution quaternion image.
    LR boundary map : torch.float32 (h,w), optional
        Returned when `return_lr_boundary_map=True`.
    """

    _NAME_RE = re.compile(
        r"^(?P<ds>.+)_(?P<split>train|val|test)_(?P<which>hr|lr)_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        dataset_root: str,
        split: str = "Train",
        take_first: Optional[int] = None,
        preload: bool = False,
        preload_torch: bool = False,
        preload_workers: int = 8,
        pin_memory: bool = True,
        return_lr_boundary_map: bool = False,
        lr_boundary_angle_deg: float = 5.0,
        lr_boundary_mark_both_sides: bool = False,
    ):
        self.root = dataset_root
        self.split = split.capitalize()
        self.preload = preload
        self.preload_torch = preload_torch
        self.preload_workers = preload_workers
        self.pin_memory = pin_memory
        self.return_lr_boundary_map = bool(return_lr_boundary_map)
        self.lr_boundary_angle_deg = float(lr_boundary_angle_deg)
        self.lr_boundary_mark_both_sides = bool(lr_boundary_mark_both_sides)

        # ------------------------------------------------------------
        # Load metadata
        # ------------------------------------------------------------
        info_path = (
            dataset_root
            if dataset_root.endswith("dataset_info.json")
            else os.path.join(dataset_root, "dataset_info.json")
        )
        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"Missing dataset_info.json: {info_path}")

        with open(info_path, "r") as f:
            info = json.load(f)

        self.symmetry = info["symmetry"]

        hr_glob = info["splits"][self.split]["HR_glob"]
        lr_glob = info["splits"][self.split]["LR_glob"]
        hr_files = sorted(glob.glob(hr_glob))
        lr_files = sorted(glob.glob(lr_glob))
        if len(hr_files) == 0 or len(lr_files) == 0:
            # Some dataset_info.json files contain machine-specific absolute globs.
            # Fall back to paths relative to dataset_root when those globs are stale.
            dataset_dir = (
                os.path.dirname(info_path)
                if info_path.endswith("dataset_info.json")
                else dataset_root
            )
            fallback_hr_glob = os.path.join(dataset_dir, self.split, "HR_Data", "*.npy")
            fallback_lr_glob = os.path.join(dataset_dir, self.split, "LR_Data", "*.npy")
            hr_fallback = sorted(glob.glob(fallback_hr_glob))
            lr_fallback = sorted(glob.glob(fallback_lr_glob))
            if len(hr_fallback) > 0 and len(lr_fallback) > 0:
                hr_files = hr_fallback
                lr_files = lr_fallback
        if take_first:
            hr_files = hr_files[:take_first]
            lr_files = lr_files[:take_first]

        hr_map = {self._parse_filename(f): f for f in hr_files}
        lr_map = {self._parse_filename(f): f for f in lr_files}
        common_keys = sorted(hr_map.keys() & lr_map.keys())
        if not common_keys:
            raise RuntimeError(f"No matching HR/LR pairs found for split={self.split}")

        self.pairs: List[Tuple[str, str]] = [
            (lr_map[k], hr_map[k]) for k in common_keys
        ]

        # ------------------------------------------------------------
        # Optional parallel preload
        # ------------------------------------------------------------
        self._preloaded_lr = None
        self._preloaded_hr = None
        self._preloaded_lr_boundary = None

        if preload:

            def load_pair(pair):
                lr_fp, hr_fp = pair
                lr_arr = np.load(lr_fp, mmap_mode=None)
                hr_arr = np.load(hr_fp, mmap_mode=None)
                # Ensure preloaded numpy arrays are writable. np.load may
                # sometimes return read-only arrays depending on platform and
                # file format; copying here avoids later warnings from
                # torch.from_numpy when converting to tensors.
                if not lr_arr.flags.writeable:
                    lr_arr = np.array(lr_arr, copy=True)
                if not hr_arr.flags.writeable:
                    hr_arr = np.array(hr_arr, copy=True)
                if preload_torch:
                    lr_t = torch.from_numpy(lr_arr)
                    hr_t = torch.from_numpy(hr_arr)
                    if self.return_lr_boundary_map:
                        lr_bmap = self._compute_lr_boundary_map_torch(lr_t)
                        return lr_t, hr_t, lr_bmap
                    return lr_t, hr_t
                if self.return_lr_boundary_map:
                    lr_bmap = self._compute_lr_boundary_map_torch(torch.from_numpy(lr_arr))
                    return lr_arr, hr_arr, lr_bmap.numpy()
                return lr_arr, hr_arr

            n_workers = min(self.preload_workers, len(self.pairs), os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(load_pair, self.pairs))

            if self.return_lr_boundary_map:
                self._preloaded_lr, self._preloaded_hr, self._preloaded_lr_boundary = zip(*results)
            else:
                self._preloaded_lr, self._preloaded_hr = zip(*results)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    @staticmethod
    def _parse_filename(fname: str) -> str:
        m = QuaternionDataset._NAME_RE.match(os.path.basename(fname))
        if not m:
            raise ValueError(f"Unexpected file format: {fname}")
        return f"{m.group('axis').lower()}_{m.group('id')}"

    @staticmethod
    def _open_memmap(fp: str) -> np.ndarray:
        arr = np.lib.format.open_memmap(fp, mode="r")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    @staticmethod
    def _to_hwc_quat_torch(q: torch.Tensor) -> torch.Tensor:
        """
        Convert rank-3 quaternion image to (H,W,4).
        Accepts (4,H,W) or (H,W,4).
        """
        if q.dim() != 3:
            raise ValueError(f"Expected rank-3 quaternion image, got {tuple(q.shape)}")
        if q.shape[0] == 4:
            return q.permute(1, 2, 0)
        if q.shape[-1] == 4:
            return q
        raise ValueError(f"Expected quaternion axis of size 4 in dim=0 or dim=-1, got {tuple(q.shape)}")

    def _compute_lr_boundary_map_torch(self, lr_q: torch.Tensor) -> torch.Tensor:
        """
        Build a binary LR boundary map from adjacent quaternion dissimilarity.
        """
        q = self._to_hwc_quat_torch(lr_q).to(dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        h, w, _ = q.shape
        boundary = torch.zeros((h, w), dtype=torch.float32)

        theta = max(0.0, min(180.0, float(self.lr_boundary_angle_deg)))
        cos_half = float(math.cos(math.radians(0.5 * theta)))

        dot_x = (q[:, 1:, :] * q[:, :-1, :]).sum(dim=-1).abs().clamp(0.0, 1.0)
        dot_y = (q[1:, :, :] * q[:-1, :, :]).sum(dim=-1).abs().clamp(0.0, 1.0)
        b_x = (dot_x < cos_half).to(dtype=torch.float32)
        b_y = (dot_y < cos_half).to(dtype=torch.float32)

        if self.lr_boundary_mark_both_sides:
            boundary[:, 1:] = torch.maximum(boundary[:, 1:], b_x)
            boundary[:, :-1] = torch.maximum(boundary[:, :-1], b_x)
            boundary[1:, :] = torch.maximum(boundary[1:, :], b_y)
            boundary[:-1, :] = torch.maximum(boundary[:-1, :], b_y)
        else:
            boundary[:, 1:] = torch.maximum(boundary[:, 1:], b_x)
            boundary[1:, :] = torch.maximum(boundary[1:, :], b_y)
        return boundary

    # ------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fastest path: fully preloaded as torch tensors
        if self._preloaded_lr is not None and self.preload_torch:
            lr_t = self._preloaded_lr[idx]
            hr_t = self._preloaded_hr[idx]
            lr_bmap_t = self._preloaded_lr_boundary[idx] if self.return_lr_boundary_map else None

        # Preloaded as numpy arrays
        elif self._preloaded_lr is not None:
            lr_t = torch.from_numpy(self._preloaded_lr[idx])
            hr_t = torch.from_numpy(self._preloaded_hr[idx])
            lr_bmap_t = (
                torch.from_numpy(self._preloaded_lr_boundary[idx])
                if self.return_lr_boundary_map
                else None
            )

        # Lazy load with memmap fallback
        else:
            lr_fp, hr_fp = self.pairs[idx]
            lr_arr = self._open_memmap(lr_fp)
            hr_arr = self._open_memmap(hr_fp)

            # Some numpy memmap objects may be non-writable. torch.from_numpy
            # requires a writable array; if the memmap is read-only, make a
            # writable copy. We only copy when necessary to avoid extra memory
            # overhead when the array is already writable.
            if not lr_arr.flags.writeable:
                lr_arr = np.array(lr_arr, copy=True)
            if not hr_arr.flags.writeable:
                hr_arr = np.array(hr_arr, copy=True)

            lr_t = torch.from_numpy(lr_arr)
            hr_t = torch.from_numpy(hr_arr)
            lr_bmap_t = (
                self._compute_lr_boundary_map_torch(lr_t)
                if self.return_lr_boundary_map
                else None
            )

        # Note: do NOT call .pin_memory() here inside the worker process.
        # Let the DataLoader handle pinning (it will pin the collated batch
        # in the main process when DataLoader(pin_memory=True) is used).
        if self.return_lr_boundary_map:
            if lr_bmap_t is None:
                lr_bmap_t = self._compute_lr_boundary_map_torch(lr_t)
            return lr_t, hr_t, lr_bmap_t
        return lr_t, hr_t

    def get_numpy_spatial_quat(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return LR and HR samples in spatial-last (H,W,4) for visualization."""
        if self._preloaded_lr is not None:
            lr = (
                self._preloaded_lr[idx].numpy()
                if self.preload_torch
                else self._preloaded_lr[idx]
            )
            hr = (
                self._preloaded_hr[idx].numpy()
                if self.preload_torch
                else self._preloaded_hr[idx]
            )
        else:
            lr_fp, hr_fp = self.pairs[idx]
            lr = self._open_memmap(lr_fp)
            hr = self._open_memmap(hr_fp)

        return to_spatial_quat(lr), to_spatial_quat(hr)
