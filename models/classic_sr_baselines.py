"""Classical Euclidean SR baselines adapted to quaternion-valued EBSD maps.

The RCAN body follows the authors' public implementation.  Only the RGB
mean-shift layers are omitted (quaternions have four normalized channels) and
the output upsampler is selected for the requested 4x1 or 4x4 task.  The same
tail helpers are also used to adapt the published SAN and HAN bodies.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class PixelShuffle1D(nn.Module):
    """Pixel shuffle along the final spatial axis only."""

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = int(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        factor = self.upscale_factor
        if channels % factor:
            raise ValueError(f"Channel count {channels} is not divisible by factor {factor}")
        out_channels = channels // factor
        x = x.contiguous().view(batch, factor, out_channels, height, width)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x.view(batch, out_channels, height, width * factor)


class HeightUpsampler1D(nn.Module):
    """UCSB-style one-axis tail used for an anisotropic ``[scale, 1]`` task.

    As in the released Q-RBSA/HAN one-axis implementation, the x2 convolution
    is shared when a power-of-two scale contains multiple shuffle stages.
    """

    def __init__(self, scale: int, n_feats: int):
        super().__init__()
        scale = int(scale)
        if scale <= 0 or scale & (scale - 1):
            raise NotImplementedError(f"Only power-of-two 1D scales are supported, got {scale}")
        self.scale = scale
        self.conv = default_conv(n_feats, 2 * n_feats, 3)
        self.shuffle = PixelShuffle1D(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The released UCSB tail shuffles width after transposing H and W.
        x = x.permute(0, 1, 3, 2)
        for _ in range(int(math.log2(self.scale))):
            x = self.shuffle(self.conv(x))
        return x.permute(0, 1, 3, 2)


class Upsampler2D(nn.Sequential):
    """Standard sub-pixel upsampler from EDSR/RCAN/SAN/HAN."""

    def __init__(self, scale: int, n_feats: int):
        scale = int(scale)
        layers: list[nn.Module] = []
        if scale > 0 and not scale & (scale - 1):
            for _ in range(int(math.log2(scale))):
                layers.extend((default_conv(n_feats, 4 * n_feats, 3), nn.PixelShuffle(2)))
        elif scale == 3:
            layers.extend((default_conv(n_feats, 9 * n_feats, 3), nn.PixelShuffle(3)))
        else:
            raise NotImplementedError(f"Unsupported 2D scale: {scale}")
        super().__init__(*layers)


def make_tail(scale_pair: tuple[int, int], n_feats: int, n_colors: int = 4) -> nn.Sequential:
    scale_h, scale_w = (int(scale_pair[0]), int(scale_pair[1]))
    if scale_h == scale_w:
        upsampler: nn.Module = Upsampler2D(scale_h, n_feats)
    elif scale_w == 1:
        upsampler = HeightUpsampler1D(scale_h, n_feats)
    else:
        raise NotImplementedError(f"Unsupported scale pair: {scale_pair}")
    return nn.Sequential(upsampler, default_conv(n_feats, n_colors, 3))


class CALayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv_du(self.avg_pool(x))


class RCAB(nn.Module):
    def __init__(self, n_feats: int, reduction: int):
        super().__init__()
        self.body = nn.Sequential(
            default_conv(n_feats, n_feats, 3),
            nn.ReLU(inplace=True),
            default_conv(n_feats, n_feats, 3),
            CALayer(n_feats, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class ResidualGroup(nn.Module):
    def __init__(self, n_feats: int, reduction: int, n_resblocks: int):
        super().__init__()
        self.body = nn.Sequential(
            *[RCAB(n_feats, reduction) for _ in range(int(n_resblocks))],
            default_conv(n_feats, n_feats, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class RCAN(nn.Module):
    """Authors' RCAN body with quaternion I/O and a task-matched SR tail."""

    def __init__(
        self,
        *,
        scale_pair: tuple[int, int],
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        n_feats: int = 64,
        reduction: int = 16,
        n_colors: int = 4,
    ):
        super().__init__()
        self.head = nn.Sequential(default_conv(n_colors, n_feats, 3))
        self.body = nn.Sequential(
            *[
                ResidualGroup(n_feats, reduction, n_resblocks)
                for _ in range(int(n_resgroups))
            ],
            default_conv(n_feats, n_feats, 3),
        )
        self.tail = make_tail(scale_pair, n_feats, n_colors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        return self.tail(self.body(x) + x)


def efficient_covpool(x: torch.Tensor) -> torch.Tensor:
    """Exact covariance used by SAN without constructing an ``HW x HW`` matrix."""

    batch, channels, height, width = x.shape
    flat = x.reshape(batch, channels, height * width)
    centered = flat - flat.mean(dim=2, keepdim=True)
    return centered.bmm(centered.transpose(1, 2)) / float(height * width)


def stable_sqrtm(matrix: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """Stable autograd form of SAN's Newton--Schulz covariance square root.

    The released custom backward becomes non-finite on some EBSD batches.  This
    uses the same trace-normalized Newton--Schulz iteration, with a tiny
    scale-relative diagonal regularizer for rank-deficient covariance matrices,
    and lets current PyTorch differentiate the matrix products directly.
    """

    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    batch, channels, _ = matrix.shape
    eye = torch.eye(channels, dtype=matrix.dtype, device=matrix.device).expand(
        batch, channels, channels
    )
    raw_trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    eps = torch.finfo(matrix.dtype).eps * 64.0
    relative_scale = (raw_trace / float(channels)).clamp_min(eps)
    matrix = matrix + (eps * relative_scale).view(batch, 1, 1) * eye
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(eps)
    y = matrix / trace.view(batch, 1, 1)
    z = eye
    for _ in range(int(num_iters)):
        update = 0.5 * (3.0 * eye - z.bmm(y))
        y = y.bmm(update)
        z = update.bmm(z)
    return y * torch.sqrt(trace).view(batch, 1, 1)
