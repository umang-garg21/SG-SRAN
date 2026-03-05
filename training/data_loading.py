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
    prefetch_factor: int = 2,
    take_first: Optional[int] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,  # Seed parameter for reproducibility
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
    prefetch_factor : int
        Number of batches to prefetch per worker.
    take_first : int, optional
        For debugging: limit dataset size.
    distributed : bool
        Enable DDP-aware sampling
    rank : int
        Process rank for DDP
    world_size : int
        Total number of processes for DDP
    seed : int
        Random seed for reproducibility (default: 42)

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
    g.manual_seed(seed)

    # Setup sampler for DDP
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle if split == "Train" else False,
            seed=seed,
        )
        # When using DistributedSampler, shuffle must be False in DataLoader
        shuffle = False

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if split == "Train" and sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g if sampler is None else None,
        sampler=sampler,
    )

    return dl
