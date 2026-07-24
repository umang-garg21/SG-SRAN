"""Phase-kernel pixel-shuffle SR with true offset-wise equivariant kernels.

This is an ablation of ``SR_phase_kernel_pixelshuffle``.  Instead of the
center-conditioned update

    TP(feature_center, weighted_neighbor_summary),

each local convolution uses the translation-convolution form

    sum_offset Linear_offset(feature_at_offset),

where every ``Linear_offset`` is an SO(3)-equivariant e3nn linear map.  Thus the
spatial kernel has independent learned parameters per offset without making the
center pixel a mandatory argument in every neighbor interaction.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import Irreps
from torch import nn

from models.SR_4x4_from_4x1_ocrp_anchorless import (
    IsoEmbedding4x1SROCRP,
    _as_scale_tuple,
)
from models.SR_phase_kernel_pixelshuffle import (
    IsoEmbeddingPhaseKernelPixelShuffleSR,
    _phase_signature,
)


class EquivariantOffsetKernelSpatialConv(nn.Module):
    """Offset-wise equivariant local convolution.

    The layer is the e3nn analogue of a small image convolution over an
    orientation-feature field: every spatial offset owns an equivariant linear
    map and the offset responses are summed.  No center-feature tensor product
    or center-neighbor similarity mask is used.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        irreps_in: Irreps | str = "1x4e",
        irreps_out: Irreps | str | None = None,
        use_residual: bool = False,
        residual_weight: float = 1.0,
        dilation: int = 1,
        output_scale: float | None = None,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding = (self.kernel_size // 2) * self.dilation
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out) if irreps_out is not None else self.irreps_in
        self.in_dim = int(self.irreps_in.dim)
        self.out_dim = int(self.irreps_out.dim)
        self.num_offsets = int(self.kernel_size * self.kernel_size)
        self.use_residual = bool(use_residual)
        self.residual_weight = float(residual_weight)
        self.output_scale = (
            1.0 / math.sqrt(float(self.num_offsets))
            if output_scale is None
            else float(output_scale)
        )

        self.offset_linears = nn.ModuleList(
            [o3.Linear(self.irreps_in, self.irreps_out) for _ in range(self.num_offsets)]
        )
        self.residual_proj: o3.Linear | None = None
        if self.use_residual and self.irreps_in != self.irreps_out:
            self.residual_proj = o3.Linear(self.irreps_in, self.irreps_out)

    def _extract_patches(self, feat_img: torch.Tensor) -> torch.Tensor:
        bsz, cdim, h, w = feat_img.shape
        feat_padded = F.pad(
            feat_img,
            (self.padding, self.padding, self.padding, self.padding),
            mode="replicate",
        )
        patches = F.unfold(
            feat_padded,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            stride=1,
        )
        return patches.view(bsz, cdim, self.num_offsets, h * w).permute(0, 3, 2, 1)

    def forward(self, features: torch.Tensor, img_shape: tuple[int, int]) -> torch.Tensor:
        h, w = int(img_shape[0]), int(img_shape[1])
        batched = features.dim() == 3
        if not batched:
            features = features.unsqueeze(0)
        bsz, n, cdim = features.shape
        if cdim != self.in_dim:
            raise ValueError(f"Expected feature dim {self.in_dim}, got {cdim}")
        if n != h * w:
            raise ValueError(f"Expected N={h*w}, got {n}")

        feat_img = features.view(bsz, h, w, cdim).permute(0, 3, 1, 2).contiguous()
        patches = self._extract_patches(feat_img)
        flat_patches = patches.reshape(bsz * n, self.num_offsets, cdim)

        out = flat_patches.new_zeros((bsz * n, self.out_dim))
        for offset_idx, linear in enumerate(self.offset_linears):
            out = out + linear(flat_patches[:, offset_idx, :])
        out = out * self.output_scale

        center_flat = features.reshape(bsz * n, cdim)
        if self.use_residual:
            if self.residual_proj is None:
                out = out + self.residual_weight * center_flat
            else:
                out = out + self.residual_weight * self.residual_proj(center_flat)

        out = out.reshape(bsz, n, self.out_dim)
        if not batched:
            out = out.squeeze(0)
        return out


class PhaseOffsetKernelPixelShuffleUpsampler(nn.Module):
    """Independent true offset kernels for each HR subpixel phase."""

    def __init__(
        self,
        irreps_feat,
        upsample_factor: int | tuple[int, int] | list[int] = (4, 4),
        kernel_size: int = 3,
        use_residual: bool = True,
        residual_weight: float = 1.0,
    ):
        super().__init__()
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.kernel_size = int(kernel_size)
        self.num_phases = int(self.upsample_factor[0] * self.upsample_factor[1])
        self.phase_kernels = nn.ModuleList(
            [
                EquivariantOffsetKernelSpatialConv(
                    kernel_size=self.kernel_size,
                    irreps_in=irreps_feat,
                    irreps_out=irreps_feat,
                    use_residual=bool(use_residual),
                    residual_weight=float(residual_weight),
                )
                for _ in range(self.num_phases)
            ]
        )

    def forward(self, feat_lr: torch.Tensor, lr_shape: tuple[int, int], return_aux: bool = False):
        batched = feat_lr.dim() == 3
        if not batched:
            feat_lr = feat_lr.unsqueeze(0)
        bsz, n_lr, cdim = feat_lr.shape
        h_lr, w_lr = int(lr_shape[0]), int(lr_shape[1])
        if n_lr != h_lr * w_lr:
            raise ValueError(f"Expected LR feature length {h_lr * w_lr}, got {n_lr}")

        phase_feats = [kernel(feat_lr, (h_lr, w_lr)) for kernel in self.phase_kernels]
        sy, sx = self.upsample_factor
        phase_stack = torch.stack(phase_feats, dim=2).view(bsz, h_lr, w_lr, sy, sx, cdim)
        feat_hr = (
            phase_stack.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, h_lr * sy * w_lr * sx, cdim)
        )
        hr_shape = (h_lr * sy, w_lr * sx)

        if not batched:
            feat_hr = feat_hr.squeeze(0)
        if not return_aux:
            return feat_hr, hr_shape
        aux = {
            "phase_kernel_outputs": phase_stack.detach(),
            "upsample_factor": self.upsample_factor,
            "num_phase_kernels": self.num_phases,
            "phase_kernel_size": self.kernel_size,
            "phase_kernel_type": "offset_linear_sum",
        }
        return feat_hr, hr_shape, aux


class IsoEmbeddingPhaseKernelPixelShuffleOffsetConvSR(IsoEmbeddingPhaseKernelPixelShuffleSR):
    """Encoder -> offset-wise phase kernels -> pixel shuffle -> decoder."""

    def __init__(
        self,
        *args,
        upsample_factor: int | tuple[int, int] | list[int] = (4, 4),
        phase_kernel_size: int = 3,
        phase_use_residual: bool = True,
        phase_residual_weight: float = 1.0,
        phase_feature_mask_cosine_threshold: float | None = None,
        phase_feature_mask_soft: bool | None = None,
        phase_feature_mask_temperature: float | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            upsample_factor=upsample_factor,
            phase_kernel_size=phase_kernel_size,
            phase_use_residual=phase_use_residual,
            phase_residual_weight=phase_residual_weight,
            phase_feature_mask_cosine_threshold=phase_feature_mask_cosine_threshold,
            phase_feature_mask_soft=phase_feature_mask_soft,
            phase_feature_mask_temperature=phase_feature_mask_temperature,
            **kwargs,
        )

        self.offset_kernel_ablation = "offset_linear_sum_no_center_conditioning"
        self.conv_lr1 = EquivariantOffsetKernelSpatialConv(
            kernel_size=int(self.conv_lr1.kernel_size),
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=self.use_residual_lr1,
            residual_weight=self.lr_conv1_residual_weight,
        )
        self.conv_hr1 = EquivariantOffsetKernelSpatialConv(
            kernel_size=int(self.conv_hr1.kernel_size),
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=self.use_residual_hr1,
            residual_weight=self.hr_conv1_residual_weight,
        )
        self.conv_hr2 = EquivariantOffsetKernelSpatialConv(
            kernel_size=int(self.conv_hr2.kernel_size),
            irreps_in=self.irreps_feat,
            irreps_out=self.irreps_feat,
            use_residual=self.use_residual_hr2,
            residual_weight=self.hr_conv2_residual_weight,
        )
        if self.conv_hr3 is not None:
            self.conv_hr3 = EquivariantOffsetKernelSpatialConv(
                kernel_size=int(self.conv_hr3.kernel_size),
                irreps_in=self.irreps_feat,
                irreps_out=self.irreps_feat,
                use_residual=self.use_residual_hr3,
                residual_weight=self.hr_conv3_residual_weight,
            )
        self.phase_upsampler = PhaseOffsetKernelPixelShuffleUpsampler(
            irreps_feat=self.irreps_feat,
            upsample_factor=self.upsample_factor,
            kernel_size=int(phase_kernel_size),
            use_residual=bool(phase_use_residual),
            residual_weight=float(phase_residual_weight),
        )


class IsoEmbedding4x4PhaseKernelPixelShuffleOffsetConvSR(
    IsoEmbeddingPhaseKernelPixelShuffleOffsetConvSR
):
    """Readable 4x4 default wrapper for experiment configs."""

    def __init__(self, *args, upsample_factor=(4, 4), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


IsoEmbeddingPhaseKernelPixelShuffleOffsetConvSR.__init__.__signature__ = _phase_signature(
    IsoEmbedding4x1SROCRP.__init__
)
IsoEmbedding4x4PhaseKernelPixelShuffleOffsetConvSR.__init__.__signature__ = _phase_signature(
    IsoEmbedding4x1SROCRP.__init__, upsample_default=(4, 4)
)


IsoEmbeddingSRAttn = IsoEmbeddingPhaseKernelPixelShuffleOffsetConvSR


__all__ = [
    "EquivariantOffsetKernelSpatialConv",
    "PhaseOffsetKernelPixelShuffleUpsampler",
    "IsoEmbeddingPhaseKernelPixelShuffleOffsetConvSR",
    "IsoEmbedding4x4PhaseKernelPixelShuffleOffsetConvSR",
    "IsoEmbeddingSRAttn",
]
