# -*-coding:utf-8 -*-
"""
File:        quaternion_dataset.py
Created at:  2025/10/18 12:17:06
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Quaternion super-resolution dataset class.
"""


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
    LR : torch.float32 (4,h,w)
        Low-resolution quaternion image.
    HR : torch.float32 (4,H,W)
        High-resolution quaternion image.
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
    ):
        self.root = dataset_root
        self.split = split.capitalize()
        self.preload = preload
        self.preload_torch = preload_torch
        self.preload_workers = preload_workers
        self.pin_memory = pin_memory

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
                    return torch.from_numpy(lr_arr), torch.from_numpy(hr_arr)
                return lr_arr, hr_arr

            n_workers = min(self.preload_workers, len(self.pairs), os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(load_pair, self.pairs))

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

    # ------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fastest path: fully preloaded as torch tensors
        if self._preloaded_lr is not None and self.preload_torch:
            lr_t = self._preloaded_lr[idx]
            hr_t = self._preloaded_hr[idx]

        # Preloaded as numpy arrays
        elif self._preloaded_lr is not None:
            lr_t = torch.from_numpy(self._preloaded_lr[idx])
            hr_t = torch.from_numpy(self._preloaded_hr[idx])

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

        # Note: do NOT call .pin_memory() here inside the worker process.
        # Let the DataLoader handle pinning (it will pin the collated batch
        # in the main process when DataLoader(pin_memory=True) is used).
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
