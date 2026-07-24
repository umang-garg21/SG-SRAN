from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_scale_tuple(scale: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(scale, int):
        if scale <= 0:
            raise ValueError(f"upsample_factor must be positive, got {scale}")
        return (int(scale), int(scale))

    scale = tuple(int(v) for v in scale)
    if len(scale) != 2:
        raise ValueError(f"upsample_factor must have length 2, got {scale}")
    if scale[0] <= 0 or scale[1] <= 0:
        raise ValueError(f"upsample_factor values must be positive, got {scale}")
    return scale


def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return quats / quats.norm(dim=-1, keepdim=True).clamp_min(eps)


class QuaternionBicubicFInterpolateSR(nn.Module):
    """
    Lightweight quaternion SR baseline using torch F.interpolate bicubic resize.

    This keeps the interpolation backend inside PyTorch so it can be compared
    directly against the existing OpenCV componentwise bicubic baseline.
    """

    def __init__(
        self,
        *,
        upsample_factor: int | Iterable[int] = 4,
        align_corners: bool = False,
        normalize_output: bool = True,
        canonicalize_output: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.align_corners = bool(align_corners)
        self.normalize_output = bool(normalize_output)
        self.canonicalize_output = bool(canonicalize_output)
        self.device = torch.device("cpu")
        if device is not None:
            self.device = torch.device(device)

    def _reshape_flat_quaternions(
        self,
        quats: torch.Tensor,
        lr_shape: tuple[int, int] | list[int],
    ) -> torch.Tensor:
        h, w = (int(lr_shape[0]), int(lr_shape[1]))
        if quats.ndim != 2 or quats.shape[-1] != 4:
            raise ValueError(f"Expected flat quaternion tensor of shape (H*W, 4), got {tuple(quats.shape)}")
        expected = h * w
        if int(quats.shape[0]) != expected:
            raise ValueError(f"Expected {expected} quaternions for lr_shape={lr_shape}, got {tuple(quats.shape)}")
        return quats.reshape(h, w, 4)

    def _postprocess_output(self, quats: torch.Tensor) -> torch.Tensor:
        if self.normalize_output:
            quats = _normalize_quaternions(quats)
        if self.canonicalize_output:
            quats = torch.where(quats[..., :1] < 0.0, -quats, quats)
        return quats

    def forward_sr(
        self,
        lr_quats: torch.Tensor,
        *,
        lr_shape: tuple[int, int] | list[int],
        normalize_input: bool = True,
        **_: object,
    ) -> torch.Tensor:
        q_hwc = self._reshape_flat_quaternions(lr_quats, lr_shape)
        if normalize_input:
            q_hwc = _normalize_quaternions(q_hwc)

        q_chw = q_hwc.permute(2, 0, 1).unsqueeze(0)
        sr_chw = F.interpolate(
            q_chw,
            scale_factor=self.upsample_factor,
            mode="bicubic",
            align_corners=self.align_corners,
        )
        sr_hwc = sr_chw.squeeze(0).permute(1, 2, 0)
        sr_hwc = self._postprocess_output(sr_hwc)
        return sr_hwc.reshape(-1, 4)

    def forward(
        self,
        lr_quats: torch.Tensor,
        *,
        img_shape: tuple[int, int] | list[int] | None = None,
        lr_shape: tuple[int, int] | list[int] | None = None,
        normalize_input: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        input_shape = lr_shape if lr_shape is not None else img_shape
        if input_shape is None:
            raise TypeError("forward requires lr_shape=... or img_shape=...")
        return self.forward_sr(
            lr_quats,
            lr_shape=input_shape,
            normalize_input=normalize_input,
            **kwargs,
        )
