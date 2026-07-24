"""Phase-kernel equivariant pixel-shuffle SR model.

This model is a routing-free alternative to OCRP. It applies one independent
equivariant LR feature convolution per HR subpixel phase, rearranges the phase
outputs into an HR feature field, then refines/decodes in HR feature space.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn

from models.SR_4x4_from_4x1_ocrp_anchorless import (
    CosineMaskedEquivariantSpatialConv,
    IsoEmbedding4x1SROCRP,
    IsoEmbeddingFromSROCRP,
    _as_scale_tuple,
)


class PhaseKernelPixelShuffleUpsampler(nn.Module):
    """Independent symmetry-valid feature kernels for each HR subpixel phase."""

    def __init__(
        self,
        irreps_feat,
        upsample_factor: int | tuple[int, int] | list[int] = (4, 4),
        kernel_size: int = 3,
        use_residual: bool = True,
        residual_weight: float = 1.0,
        feature_mask_cosine_threshold: float = 0.98,
        feature_mask_soft: bool = False,
        feature_mask_temperature: float = 32.0,
    ):
        super().__init__()
        self.upsample_factor = _as_scale_tuple(upsample_factor)
        self.kernel_size = int(kernel_size)
        self.num_phases = int(self.upsample_factor[0] * self.upsample_factor[1])
        self.phase_kernels = nn.ModuleList(
            [
                CosineMaskedEquivariantSpatialConv(
                    kernel_size=self.kernel_size,
                    irreps_in=irreps_feat,
                    irreps_out=irreps_feat,
                    use_residual=bool(use_residual),
                    residual_weight=float(residual_weight),
                    feature_mask_cosine_threshold=float(feature_mask_cosine_threshold),
                    feature_mask_soft=bool(feature_mask_soft),
                    feature_mask_temperature=float(feature_mask_temperature),
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
        }
        return feat_hr, hr_shape, aux


class IsoEmbeddingPhaseKernelPixelShuffleSR(IsoEmbeddingFromSROCRP):
    """Encoder -> phase-specific equivariant kernels -> pixel shuffle -> decoder."""

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
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)
        if hasattr(self, "ocrp"):
            delattr(self, "ocrp")
        phase_thr = (
            self.lr_conv_feature_mask_cosine_threshold
            if phase_feature_mask_cosine_threshold is None
            else float(phase_feature_mask_cosine_threshold)
        )
        phase_soft = (
            self.lr_conv_feature_mask_soft
            if phase_feature_mask_soft is None
            else bool(phase_feature_mask_soft)
        )
        phase_temp = (
            self.lr_conv_feature_mask_temperature
            if phase_feature_mask_temperature is None
            else float(phase_feature_mask_temperature)
        )
        self.phase_upsampler = PhaseKernelPixelShuffleUpsampler(
            irreps_feat=self.irreps_feat,
            upsample_factor=self.upsample_factor,
            kernel_size=int(phase_kernel_size),
            use_residual=bool(phase_use_residual),
            residual_weight=float(phase_residual_weight),
            feature_mask_cosine_threshold=phase_thr,
            feature_mask_soft=phase_soft,
            feature_mask_temperature=phase_temp,
        )

    def _filter_incompatible_state_dict(self, state_dict):
        filtered = super()._filter_incompatible_state_dict(state_dict)
        return {key: value for key, value in filtered.items() if not key.startswith("ocrp.")}

    def _forward_sr_features(
        self,
        lr_quats: torch.Tensor,
        feat_lr: torch.Tensor,
        lr_shape: tuple[int, int],
        return_aux: bool = False,
    ):
        feat_pre = feat_lr
        if self.use_lr_conv1:
            feat_pre = self.conv_lr1(feat_pre, lr_shape)

        out = self.phase_upsampler(feat_pre, lr_shape, return_aux=return_aux)
        if return_aux:
            feat_hr, hr_shape, aux = out
            feat_hr_raw_phase = feat_hr
        else:
            feat_hr, hr_shape = out

        feat_hr_after_conv1 = feat_hr
        if self.use_hr_conv1:
            feat_hr_after_conv1 = self.conv_hr1(feat_hr_after_conv1, hr_shape)
        feat_hr_after_conv2 = feat_hr_after_conv1
        if self.use_hr_conv2:
            feat_hr_after_conv2 = self.conv_hr2(feat_hr_after_conv2, hr_shape)
        feat_hr = feat_hr_after_conv2
        if self.conv_hr3 is not None:
            feat_hr = self.conv_hr3(feat_hr, hr_shape)

        if not return_aux:
            return feat_hr, hr_shape

        aux["feat_lr_encode"] = feat_lr
        aux["feat_lr_pre_phase_kernel"] = feat_pre
        aux["feat_hr_raw_phase_kernel"] = feat_hr_raw_phase
        aux["feat_hr_post_hr_conv1"] = feat_hr_after_conv1
        aux["feat_hr_post_hr_conv2"] = feat_hr_after_conv2
        aux["feat_hr_post_hr_conv"] = feat_hr

        probe_stages: list[dict[str, object]] = [
            {"name": "encode_lr", "feat": feat_lr.detach(), "shape": tuple(lr_shape)}
        ]
        if self.use_lr_conv1:
            probe_stages.append(
                {"name": "lr_conv1_post", "feat": feat_pre.detach(), "shape": tuple(lr_shape)}
            )
        probe_stages.append(
            {"name": "phase_kernel_hr_raw", "feat": feat_hr_raw_phase.detach(), "shape": tuple(hr_shape)}
        )
        if self.use_hr_conv1:
            probe_stages.append(
                {"name": "hr_conv1_post", "feat": feat_hr_after_conv1.detach(), "shape": tuple(hr_shape)}
            )
        if self.use_hr_conv2:
            probe_stages.append(
                {"name": "hr_conv2_post", "feat": feat_hr_after_conv2.detach(), "shape": tuple(hr_shape)}
            )
        if self.conv_hr3 is not None:
            probe_stages.append(
                {"name": "hr_conv3_post", "feat": feat_hr.detach(), "shape": tuple(hr_shape)}
            )
        aux["probe_stages"] = probe_stages
        return feat_hr, hr_shape, aux


class IsoEmbedding4x4PhaseKernelPixelShuffleSR(IsoEmbeddingPhaseKernelPixelShuffleSR):
    """Readable 4x4 default wrapper for experiment configs."""

    def __init__(self, *args, upsample_factor=(4, 4), **kwargs):
        super().__init__(*args, upsample_factor=upsample_factor, **kwargs)


def _phase_signature(base_fn, *, upsample_default=None):
    sig = inspect.signature(base_fn)
    params = list(sig.parameters.values())
    if upsample_default is not None:
        params = [
            param.replace(default=upsample_default)
            if param.name == "upsample_factor"
            else param
            for param in params
        ]
    insert_at = len(params)
    for idx, param in enumerate(params):
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            insert_at = idx
            break
    extras = [
        inspect.Parameter("phase_kernel_size", inspect.Parameter.KEYWORD_ONLY, default=3, annotation=int),
        inspect.Parameter("phase_use_residual", inspect.Parameter.KEYWORD_ONLY, default=True, annotation=bool),
        inspect.Parameter("phase_residual_weight", inspect.Parameter.KEYWORD_ONLY, default=1.0, annotation=float),
        inspect.Parameter(
            "phase_feature_mask_cosine_threshold",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=float | None,
        ),
        inspect.Parameter(
            "phase_feature_mask_soft",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=bool | None,
        ),
        inspect.Parameter(
            "phase_feature_mask_temperature",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=float | None,
        ),
    ]
    return sig.replace(parameters=params[:insert_at] + extras + params[insert_at:])


IsoEmbeddingPhaseKernelPixelShuffleSR.__init__.__signature__ = _phase_signature(
    IsoEmbedding4x1SROCRP.__init__
)
IsoEmbedding4x4PhaseKernelPixelShuffleSR.__init__.__signature__ = _phase_signature(
    IsoEmbedding4x1SROCRP.__init__, upsample_default=(4, 4)
)


IsoEmbeddingSRAttn = IsoEmbeddingPhaseKernelPixelShuffleSR


__all__ = [
    "PhaseKernelPixelShuffleUpsampler",
    "IsoEmbeddingPhaseKernelPixelShuffleSR",
    "IsoEmbedding4x4PhaseKernelPixelShuffleSR",
    "IsoEmbeddingSRAttn",
]
