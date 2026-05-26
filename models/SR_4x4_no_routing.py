"""No-routing counterfactual to the 4x4 OCRP pipeline.

This module defines `IsoEmbedding4x4NoRoutingBicubic` and
`IsoEmbedding4x4NoRoutingNN`: drop-in replacements for
`IsoEmbedding4x4FromSROCRP` in which the routed-patch upsampler is replaced by a
fixed bicubic (resp. nearest-neighbour) interpolation in latent space. Every
other component --- the local-iso encoder, the LR refinement convolution, the
HR refinement convolutions and the cubochoric decoder --- is unchanged, so any
metric difference relative to the OCRP backbone isolates the contribution of
the routing decision itself.

This is the [STAGE-EVIDENCE] counterfactual called for by the BSM comment in
Results: if OCRP is replaced by a non-routed latent upsampler followed by the
same HR convolutions, can the residual stack alone recover the OCRP gain?
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SR_4x4_from_4x1_ocrp_anchorless import (
    IsoEmbedding4x4FromSROCRP,
    _as_scale_tuple,
)


class _LatentInterpolationUpsampler(nn.Module):
    """Fixed spatial interpolation in latent space. Zero trainable parameters.

    Matches the calling signature of `OCRP4x1PatchUpsampler.forward`:
    inputs `(lr_quats, feat_lr, lr_shape, return_aux)` and outputs
    `(feat_hr, hr_shape)` or `(feat_hr, hr_shape, aux)`.
    """

    def __init__(self, upsample_factor, mode: str = "bicubic"):
        super().__init__()
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        if mode not in {"bicubic", "nearest"}:
            raise ValueError(f"mode must be 'bicubic' or 'nearest', got {mode}")
        self.mode = mode

    def forward(self, lr_quats, feat_lr, lr_shape, return_aux: bool = False):
        batched = feat_lr.dim() == 3
        if not batched:
            feat_lr = feat_lr.unsqueeze(0)
        bsz, n_lr, feat_dim = feat_lr.shape
        h_lr, w_lr = int(lr_shape[0]), int(lr_shape[1])
        if n_lr != h_lr * w_lr:
            raise ValueError(
                f"feat_lr token count {n_lr} != H_lr*W_lr = {h_lr * w_lr}"
            )
        r_h, r_w = self.upsample_factor
        h_hr, w_hr = h_lr * r_h, w_lr * r_w

        feat_grid = feat_lr.transpose(1, 2).reshape(bsz, feat_dim, h_lr, w_lr)
        kwargs = {}
        if self.mode == "bicubic":
            kwargs["align_corners"] = False
        feat_grid = F.interpolate(
            feat_grid, size=(h_hr, w_hr), mode=self.mode, **kwargs
        )
        feat_hr = feat_grid.reshape(bsz, feat_dim, h_hr * w_hr).transpose(1, 2)
        if not batched:
            feat_hr = feat_hr.squeeze(0)
        hr_shape = (h_hr, w_hr)
        if return_aux:
            return feat_hr, hr_shape, {"no_routing_mode": self.mode}
        return feat_hr, hr_shape


class _IsoEmbedding4x4NoRoutingBase(IsoEmbedding4x4FromSROCRP):
    """Shared init that replaces ``self.ocrp`` after the parent's construction.

    The parent constructor still builds the OCRP module, but it is immediately
    overwritten with a zero-parameter interpolation upsampler. This keeps
    constructor-argument compatibility with the existing config files.
    """

    _no_routing_mode: str = "bicubic"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ocrp = _LatentInterpolationUpsampler(
            upsample_factor=self.upsample_factor,
            mode=self._no_routing_mode,
        )
        self.no_routing_mode = self._no_routing_mode


class IsoEmbedding4x4NoRoutingBicubic(_IsoEmbedding4x4NoRoutingBase):
    _no_routing_mode = "bicubic"


class IsoEmbedding4x4NoRoutingNN(_IsoEmbedding4x4NoRoutingBase):
    _no_routing_mode = "nearest"
