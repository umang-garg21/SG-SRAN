# -*-coding:utf-8 -*-
"""
File:        data_loading.py
Created at:  2025/10/18 13:00:44
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Data loading utilities for quaternion super-resolution using the quaternion_dataset object.
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

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional

# Import your optimized QuaternionDataset
from training.quaternion_dataset import QuaternionDataset


# ============================================================
# 🔸 Reproducibility Helpers
# ============================================================


def seed_worker(worker_id: int):
    """Ensure deterministic behavior in DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# 🔸 DataLoader Builder
# ============================================================


def build_dataloader(
    dataset_root: str,
    split: str = "Train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    preload: bool = False,
    preload_torch: bool = False,
    persistent_workers: bool = False,
    take_first: Optional[int] = None,
) -> DataLoader:
    """
    Build a DataLoader for quaternion SR datasets.

    Parameters
    ----------
    dataset_root : str
        Path to dataset folder containing dataset_info.json
    split : str
        "Train", "Val", or "Test"
    batch_size : int
        Number of patches per batch.
    shuffle : bool
        Shuffle samples (usually True for training).
    num_workers : int
        Number of CPU workers for background data loading.
    pin_memory : bool
        Pin memory to speed up host→GPU transfer.
    preload : bool
        Preload entire dataset into CPU RAM at init.
    preload_torch : bool
        Preload directly as torch tensors.
    persistent_workers : bool
        Keep workers alive between epochs for performance.
    take_first : int, optional
        For debugging: limit dataset size.

    Returns
    -------
    DataLoader
    """
    ds = QuaternionDataset(
        dataset_root=dataset_root,
        split=split,
        preload=preload,
        preload_torch=preload_torch,
        pin_memory=pin_memory,
        take_first=take_first,
    )

    # Generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(42)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if split == "Train" else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return dl
