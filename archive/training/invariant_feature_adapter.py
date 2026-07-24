# -*- coding:utf-8 -*-
"""
Dataset adapter that converts Bunge quaternions to symmetry-invariant features.

This wrapper is optional and intended for experiments where the model consumes
invariant feature maps instead of raw quaternion channels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from bunge_invariant import encode_bunge_invariant_features


class InvariantFeatureAdapterDataset(Dataset):
    """
    Wrap a base dataset yielding (lr, hr) and encode either side with
    Bunge-invariant features.

    Parameters
    ----------
    base_dataset : Dataset
        Underlying dataset yielding (lr, hr) tensors or numpy arrays.
    method : {"fz_logmap","soft_orbit","hybrid"}
        Invariant encoding method.
    beta : float
        Soft-orbit temperature factor.
    apply_to : {"lr","hr","both"}
        Which side(s) to encode.
    channel_first : bool
        If True, return encoded maps as (C,H,W). If False, keep (...,C).
    cache_features : bool
        Cache encoded samples by index to avoid recomputation each epoch.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        method: str = "hybrid",
        beta: float = 64.0,
        apply_to: str = "lr",
        channel_first: bool = True,
        cache_features: bool = False,
    ):
        self.base_dataset = base_dataset
        self.method = str(method).strip().lower()
        self.beta = float(beta)
        self.apply_to = str(apply_to).strip().lower()
        self.channel_first = bool(channel_first)
        self.cache_features = bool(cache_features)
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        if self.method not in {"fz_logmap", "soft_orbit", "hybrid"}:
            raise ValueError(f"Unsupported invariant method: {method}")
        if self.apply_to not in {"lr", "hr", "both"}:
            raise ValueError(f"apply_to must be one of {{'lr','hr','both'}}, got {apply_to}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    @staticmethod
    def _as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _encode(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        arr = self._as_numpy(x)
        result = encode_bunge_invariant_features(arr, method=self.method, beta=self.beta)
        feat = np.asarray(result["features"], dtype=np.float32)

        if self.channel_first and feat.ndim >= 3:
            feat = np.moveaxis(feat, -1, 0)

        feat = np.ascontiguousarray(feat, dtype=np.float32)
        return torch.from_numpy(feat)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_features and idx in self._cache:
            return self._cache[idx]

        lr, hr = self.base_dataset[idx]
        lr_t = lr if isinstance(lr, torch.Tensor) else torch.from_numpy(np.asarray(lr))
        hr_t = hr if isinstance(hr, torch.Tensor) else torch.from_numpy(np.asarray(hr))

        if self.apply_to in {"lr", "both"}:
            lr_out = self._encode(lr_t)
        else:
            lr_out = lr_t.to(torch.float32)

        if self.apply_to in {"hr", "both"}:
            hr_out = self._encode(hr_t)
        else:
            hr_out = hr_t.to(torch.float32)

        if self.cache_features:
            self._cache[idx] = (lr_out, hr_out)

        return lr_out, hr_out

