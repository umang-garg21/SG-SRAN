# # -*-coding:utf-8 -*-
# """
# File:        quaternion_dataset.py
# Created at:  2025/10/18 12:11:03
# Author:      Warren Zamudio
# Contact:     wzamudio@ucsb.edu
# Description: Quaternion Dataset Loader for SR Training
# """

# import os
# import re
# import glob
# import json
# import torch
# import numpy as np
# from typing import List, Tuple, Optional
# from torch.utils.data import Dataset
# from concurrent.futures import ThreadPoolExecutor
# from utils.quat_ops import to_spatial_quat


# class QuaternionDataset(Dataset):
#     """
#     Quaternion LR/HR dataset loader for structured datasets created with build_quaternion_sr_dataset.

#     Parameters
#     ----------
#     dataset_root : str
#         Path to dataset folder containing dataset_info.json.
#     split : {"Train","Val","Test"}, default="Train"
#         Which split to load.
#     take_first : int, optional
#         If set, only use the first N HR files.
#     preload : bool, default=False
#         Preload all data into CPU RAM at init (fastest per step).
#     preload_torch : bool, default=False
#         Preload as torch tensors (skip conversion in __getitem__).
#     preload_workers : int, default=8
#         Number of threads used for parallel preload.
#     pin_memory : bool, default=True
#         Pin tensor memory to accelerate GPU transfer.

#     Returns
#     -------
#     LR : torch.float32 (4,h,w)
#         Low-resolution quaternion image.
#     HR : torch.float32 (4,H,W)
#         High-resolution quaternion image.
#     """

#     _NAME_RE = re.compile(
#         r"^(?P<ds>.+)_(?P<split>train|val|test)_(?P<which>hr|lr)_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
#         re.IGNORECASE,
#     )

#     def __init__(
#         self,
#         dataset_root: str,
#         split: str = "Train",
#         take_first: Optional[int] = None,
#         preload: bool = False,
#         preload_torch: bool = False,
#         preload_workers: int = 8,
#         pin_memory: bool = True,
#     ):
#         self.root = dataset_root
#         self.split = split.capitalize()
#         self.preload = preload
#         self.preload_torch = preload_torch
#         self.preload_workers = preload_workers
#         self.pin_memory = pin_memory

#         # ------------------------------------------------------------
#         # Load metadata
#         # ------------------------------------------------------------
#         info_path = (
#             dataset_root
#             if dataset_root.endswith("dataset_info.json")
#             else os.path.join(dataset_root, "dataset_info.json")
#         )
#         if not os.path.isfile(info_path):
#             raise FileNotFoundError(f"Missing dataset_info.json: {info_path}")

#         with open(info_path, "r") as f:
#             info = json.load(f)

#         hr_glob = info["splits"][self.split]["HR_glob"]
#         lr_glob = info["splits"][self.split]["LR_glob"]
#         hr_files = sorted(glob.glob(hr_glob))
#         lr_files = sorted(glob.glob(lr_glob))
#         if take_first:
#             hr_files = hr_files[:take_first]
#             lr_files = lr_files[:take_first]

#         hr_map = {self._parse_filename(f): f for f in hr_files}
#         lr_map = {self._parse_filename(f): f for f in lr_files}
#         common_keys = sorted(hr_map.keys() & lr_map.keys())
#         if not common_keys:
#             raise RuntimeError(f"No matching HR/LR pairs found for split={self.split}")

#         self.pairs: List[Tuple[str, str]] = [
#             (lr_map[k], hr_map[k]) for k in common_keys
#         ]

#         # ------------------------------------------------------------
#         # Optional parallel preload
#         # ------------------------------------------------------------
#         self._preloaded_lr = None
#         self._preloaded_hr = None

#         if preload:

#             def load_pair(pair):
#                 lr_fp, hr_fp = pair
#                 lr_arr = np.load(lr_fp, mmap_mode=None)
#                 hr_arr = np.load(hr_fp, mmap_mode=None)
#                 if preload_torch:
#                     return torch.from_numpy(lr_arr), torch.from_numpy(hr_arr)
#                 return lr_arr, hr_arr

#             n_workers = min(self.preload_workers, len(self.pairs), os.cpu_count() or 1)
#             with ThreadPoolExecutor(max_workers=n_workers) as ex:
#                 results = list(ex.map(load_pair, self.pairs))

#             self._preloaded_lr, self._preloaded_hr = zip(*results)

#     # ------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------
#     @staticmethod
#     def _parse_filename(fname: str) -> str:
#         m = QuaternionDataset._NAME_RE.match(os.path.basename(fname))
#         if not m:
#             raise ValueError(f"Unexpected file format: {fname}")
#         return f"{m.group('axis').lower()}_{m.group('id')}"

#     @staticmethod
#     def _open_memmap(fp: str) -> np.ndarray:
#         arr = np.lib.format.open_memmap(fp, mode="r")
#         if arr.dtype != np.float32:
#             arr = arr.astype(np.float32)
#         return arr

#     # ------------------------------------------------------------
#     # Dataset API
#     # ------------------------------------------------------------
#     def __len__(self) -> int:
#         return len(self.pairs)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Fastest path: fully preloaded as torch tensors
#         if self._preloaded_lr is not None and self.preload_torch:
#             lr_t = self._preloaded_lr[idx]
#             hr_t = self._preloaded_hr[idx]

#         # Preloaded as numpy arrays
#         elif self._preloaded_lr is not None:
#             lr_t = torch.from_numpy(self._preloaded_lr[idx])
#             hr_t = torch.from_numpy(self._preloaded_hr[idx])

#         # Lazy load with memmap fallback
#         else:
#             lr_fp, hr_fp = self.pairs[idx]
#             lr_t = torch.from_numpy(self._open_memmap(lr_fp))
#             hr_t = torch.from_numpy(self._open_memmap(hr_fp))

#         if self.pin_memory:
#             return lr_t.pin_memory(), hr_t.pin_memory()
#         return lr_t, hr_t

#     def get_numpy_spatial_quat(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
#         """Return LR and HR samples in spatial-last (H,W,4) for visualization."""
#         if self._preloaded_lr is not None:
#             lr = (
#                 self._preloaded_lr[idx].numpy()
#                 if self.preload_torch
#                 else self._preloaded_lr[idx]
#             )
#             hr = (
#                 self._preloaded_hr[idx].numpy()
#                 if self.preload_torch
#                 else self._preloaded_hr[idx]
#             )
#         else:
#             lr_fp, hr_fp = self.pairs[idx]
#             lr = self._open_memmap(lr_fp)
#             hr = self._open_memmap(hr_fp)

#         return to_spatial_quat(lr), to_spatial_quat(hr)


# # """
# # datasets/quaternion_dataset.py

# # QuaternionDataset:
# #     - Efficient quaternion dataset loader for LR/HR .npy blocks
# #     - Clean separation between data access and visualization
# #     - Optional integrity checking for normalization, hemisphere alignment, and shape consistency
# #     - Works with quaternion-first format (4, H, W)

# # Author: Warren Zamudio
# # """

# # import os
# # import re
# # import glob
# # import json
# # import warnings
# # from typing import List, Tuple, Optional

# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset
# # from orix.quaternion import Orientation

# # from utils.quat_ops import (
# #     to_spatial_quat,
# #     to_torch_quat_spatial,
# #     format_quaternions,
# #     reduce_to_fz_min_angle,
# # )

# # from utils.symmetry_utils import resolve_symmetry as _resolve_symmetry


# # class QuaternionDataset(Dataset):
# #     """
# #     Quaternion LR/HR dataset loader for structured datasets created with build_quaternion_sr_dataset.

# #     Parameters
# #     ----------
# #     dataset_root : str
# #         Path to dataset folder containing dataset_info.json.
# #     split : {"Train","Val","Test"}, default="Train"
# #         Which split to load.
# #     take_first : int, optional
# #         If set, only use the first N HR files.
# #     check_integrity : bool, default=True
# #         If True, verify LR/HR shape and quaternion validity on load.
# #     fix_on_warn : bool, default=True
# #         If True, auto-fix normalization and hemisphere issues.

# #     Returns
# #     -------
# #     LR : torch.float32 (4,h,w)
# #         Low-resolution quaternion image.
# #     HR : torch.float32 (4,H,W)
# #         High-resolution quaternion image.
# #     """

# #     _NAME_RE = re.compile(
# #         r"^(?P<ds>.+)_(?P<split>train|val|test)_(?P<which>hr|lr)_(?P<axis>[xyz])_block_(?P<id>\d+)\.npy$",
# #         re.IGNORECASE,
# #     )

# #     # ----------------------------
# #     # Static utility functions
# #     # ----------------------------
# #     @staticmethod
# #     def _parse_filename(fname: str) -> str:
# #         """Extract axis and block ID key from dataset filename."""
# #         m = QuaternionDataset._NAME_RE.match(os.path.basename(fname))
# #         if not m:
# #             raise ValueError(f"Unexpected file format: {fname}")
# #         return f"{m.group('axis').lower()}_{m.group('id')}"

# #     @staticmethod
# #     def _quat_first_shape(a) -> Tuple[int, ...]:
# #         """Ensure quaternion-first (4, *spatial) layout."""
# #         shp = a.shape if isinstance(a, np.ndarray) else tuple(a)
# #         if 4 not in shp:
# #             raise ValueError(
# #                 f"[Integrity Error] No quaternion axis (size=4) in shape {shp}"
# #             )
# #         if shp[0] != 4:
# #             raise ValueError(f"[Integrity Error] Quaternion dimension not first: {shp}")
# #         return shp

# #     @staticmethod
# #     def _memmap_shape(path: str) -> Tuple[int, ...]:
# #         """Return the shape of a .npy file via memmap header (fast, no data load)."""
# #         try:
# #             mm = np.lib.format.open_memmap(path, mode="r")
# #             shape = mm.shape
# #             del mm
# #             return shape
# #         except Exception as e:
# #             raise ValueError(f"Failed to read shape of {path}: {e}")

# #     @staticmethod
# #     def _sample_quaternions(q: np.ndarray, n: int = 10000) -> np.ndarray:
# #         """Sample up to n quaternions from (4, *spatial) efficiently."""
# #         q = q.reshape(4, -1)
# #         total = q.shape[1]
# #         if total <= n:
# #             return q
# #         idx = np.random.choice(total, n, replace=False)
# #         return q[:, idx]

# #     @staticmethod
# #     def _fix_quaternion_file(path: str, sym=None, fz_reduce: bool = False):
# #         """
# #         Normalize, hemisphere-align, and optionally FZ-reduce quaternion array in-place.
# #         """
# #         q = np.load(path, mmap_mode="r")

# #         # Normalize + hemisphere
# #         q_fixed = format_quaternions(q, normalize=True, enforce_hemisphere=True)

# #         # Optional FZ reduction
# #         if fz_reduce and sym is not None:
# #             q_fixed = reduce_to_fz_min_angle(
# #                 q_fixed.transpose(1, 2, 0), sym
# #             )  # if H,W,4
# #             # or if already (4,H,W)
# #             if q_fixed.shape[0] != 4:
# #                 q_fixed = np.moveaxis(q_fixed, -1, 0)

# #         tmp_path = f"{path}.fixed.npy"
# #         np.save(tmp_path, q_fixed)
# #         os.replace(tmp_path, path)
# #         print(f"[Auto-Fix] Rewritten normalized (and FZ-reduced) file: {path}")

# #     # ----------------------------
# #     # Initialization
# #     # ----------------------------
# #     def __init__(
# #         self,
# #         dataset_root: str,
# #         split: str = "Train",
# #         take_first: Optional[int] = None,
# #         check_integrity: bool = True,
# #         fix_on_warn: bool = True,
# #     ):
# #         # Load metadata
# #         info_path = (
# #             dataset_root
# #             if dataset_root.endswith("dataset_info.json")
# #             else os.path.join(dataset_root, "dataset_info.json")
# #         )
# #         if not os.path.isfile(info_path):
# #             raise FileNotFoundError(f"Missing dataset_info.json: {info_path}")

# #         with open(info_path, "r") as f:
# #             self.info = json.load(f)

# #         self.split = split.capitalize()
# #         if self.split not in ("Train", "Val", "Test"):
# #             raise ValueError("split must be 'Train', 'Val', or 'Test'")

# #         # Resolve HR/LR file pairs
# #         hr_glob = self.info["splits"][self.split]["HR_glob"]
# #         lr_glob = self.info["splits"][self.split]["LR_glob"]
# #         hr_files = sorted(glob.glob(hr_glob))
# #         lr_files = sorted(glob.glob(lr_glob))
# #         if take_first:
# #             hr_files, lr_files = hr_files[:take_first], lr_files[:take_first]

# #         hr_map = {self._parse_filename(f): f for f in hr_files}
# #         lr_map = {self._parse_filename(f): f for f in lr_files}
# #         common_keys = sorted(hr_map.keys() & lr_map.keys())
# #         if not common_keys:
# #             raise RuntimeError("No matching HR/LR pairs found")

# #         self.pairs: List[Tuple[str, str]] = [
# #             (lr_map[k], hr_map[k]) for k in common_keys
# #         ]
# #         self.sym_class = _resolve_symmetry(self.info.get("symmetry", "Oh"))

# #         if check_integrity:
# #             self.check_integrity(fix_on_warn=fix_on_warn)

# #     # ----------------------------
# #     # Integrity checking
# #     # ----------------------------
# #     def check_integrity(
# #         self,
# #         n_check: int = 20,
# #         fix_on_warn: bool = False,
# #         check_normalization: bool = True,
# #         check_hemisphere: bool = True,
# #         check_fz: bool = True,
# #         fz_tol_deg: float = 0.5,
# #         sample_n: int = 10000,
# #     ):
# #         """Validate dataset structure and quaternion correctness."""
# #         n_check = min(n_check, len(self.pairs))
# #         sym = self.sym_class
# #         fz_tol_rad = np.deg2rad(fz_tol_deg)

# #         n_shape_warn = 0
# #         n_norm_warn = 0
# #         n_hemi_warn = 0
# #         n_fz_warn = 0

# #         shape_hr_ref, shape_lr_ref = None, None

# #         for i in range(n_check):
# #             lr_fp, hr_fp = self.pairs[i]

# #             hr_shape = self._memmap_shape(hr_fp)
# #             lr_shape = self._memmap_shape(lr_fp)
# #             self._quat_first_shape(hr_shape)
# #             self._quat_first_shape(lr_shape)

# #             if shape_hr_ref is None:
# #                 shape_hr_ref, shape_lr_ref = hr_shape, lr_shape
# #             else:
# #                 if hr_shape != shape_hr_ref or lr_shape != shape_lr_ref:
# #                     n_shape_warn += 1
# #                     warnings.warn(f"[Shape Warning] {i}: HR={hr_shape}, LR={lr_shape}")

# #             # Sample quaternions
# #             hr = np.load(hr_fp, mmap_mode="r")
# #             q_sample = self._sample_quaternions(hr, sample_n)
# #             del hr

# #             # Normalization
# #             if check_normalization:
# #                 norms = np.linalg.norm(q_sample, axis=0)
# #                 if not np.allclose(norms.mean(), 1.0, atol=0.02):
# #                     n_norm_warn += 1
# #                     warnings.warn(f"[Norm Warning] {os.path.basename(hr_fp)}")
# #                     if fix_on_warn:
# #                         self._fix_quaternion_file(hr_fp)

# #             # Hemisphere
# #             if check_hemisphere and q_sample[0].mean() < -0.01:
# #                 n_hemi_warn += 1
# #                 warnings.warn(f"[Hemisphere Warning] {os.path.basename(hr_fp)}")
# #                 if fix_on_warn:
# #                     self._fix_quaternion_file(hr_fp)

# #             # Fundamental zone
# #             if check_fz:
# #                 q_fz, _ = reduce_to_fz_min_angle(q_sample.T, sym=sym)
# #                 dot = np.abs(np.sum(q_fz * q_sample.T, axis=1))
# #                 dot = np.clip(dot, -1.0, 1.0)
# #                 angles = np.arccos(dot)
# #                 frac_out = (angles > fz_tol_rad).mean()

# #                 if frac_out > 0.01:
# #                     n_fz_warn += 1
# #                     warnings.warn(
# #                         f"[FZ Warning] {os.path.basename(hr_fp)} {frac_out*100:.2f}% outside FZ"
# #                     )

# #         print(
# #             f"[Integrity] shape={n_shape_warn}, norm={n_norm_warn}, hemi={n_hemi_warn}, fz={n_fz_warn}"
# #         )

# #     # ----------------------------
# #     # Dataset API
# #     # ----------------------------
# #     def __len__(self) -> int:
# #         return len(self.pairs)

# #     def __getitem__(self, idx: int):
# #         """Return LR and HR samples as PyTorch tensors (4,H,W)."""
# #         lr_fp, hr_fp = self.pairs[idx]
# #         lr = np.load(lr_fp, mmap_mode="r")
# #         hr = np.load(hr_fp, mmap_mode="r")
# #         return to_torch_quat_spatial(lr), to_torch_quat_spatial(hr)

# #     def get_numpy_spatial_quat(self, idx: int):
# #         """
# #         Return LR and HR samples as NumPy arrays in (*spatial, 4) quaternion-last layout.
# #         """
# #         lr_fp, hr_fp = self.pairs[idx]
# #         lr = to_spatial_quat(np.lib.format.open_memmap(lr_fp, mode="r"))
# #         hr = to_spatial_quat(np.lib.format.open_memmap(hr_fp, mode="r"))

# #         if lr.dtype != np.float32 or not lr.flags["C_CONTIGUOUS"]:
# #             lr = np.ascontiguousarray(lr, dtype=np.float32)
# #         if hr.dtype != np.float32 or not hr.flags["C_CONTIGUOUS"]:
# #             hr = np.ascontiguousarray(hr, dtype=np.float32)

# #         return lr, hr
