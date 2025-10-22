import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

# ========================================================================
# Quaternion lifting / projection (same as before)
# ========================================================================


def quat_to_lmat(q):
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    L = torch.stack(
        [
            torch.stack([a, -b, -c, -d], dim=1),
            torch.stack([b, a, -d, c], dim=1),
            torch.stack([c, d, a, -b], dim=1),
            torch.stack([d, -c, b, a], dim=1),
        ],
        dim=1,
    )
    return L.view(q.size(0), 16, q.size(2), q.size(3))


def lmat_to_quat(L):
    B, _, H, W = L.shape
    Lm = L.view(B, 4, 4, H, W)
    q = Lm[:, :, 0, :, :]
    return q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)


# ========================================================================
# Quaternion Conv / Transpose Conv helpers (same as before)
# ========================================================================


def _fan_in_fan_out(weight):
    fan_in = weight.size(1)
    fan_out = weight.size(0)
    for s in weight.shape[2:]:
        fan_in *= s
        fan_out *= s
    return fan_in, fan_out


def _he_init_like(wr, wi, wj, wk, criterion="glorot"):
    fan_in, fan_out = _fan_in_fan_out(wr)
    if criterion.lower() == "he":
        s = math.sqrt(2.0 / fan_in)
    else:
        s = math.sqrt(2.0 / (fan_in + fan_out))
    for p in (wr, wi, wj, wk):
        nn.init.normal_(p, mean=0.0, std=s / 2.0)


def quaternion_block_weight(r, i, j, k):
    k_rr = torch.cat([r, -i, -j, -k], dim=1)
    k_ri = torch.cat([i, r, -k, j], dim=1)
    k_rj = torch.cat([j, k, r, -i], dim=1)
    k_rk = torch.cat([k, -j, i, r], dim=1)
    return torch.cat([k_rr, k_ri, k_rj, k_rk], dim=0)


class QuaternionConv(nn.Module):
    def __init__(
        self,
        in_q_channels,
        out_q_channels,
        kernel_size,
        padding=None,
        groups=None,
        bias=True,
    ):
        super().__init__()
        assert in_q_channels % 4 == 0 and out_q_channels % 4 == 0
        self.in_q = in_q_channels // 4
        self.out_q = out_q_channels // 4
        self.groups = self.in_q if groups is None else groups
        assert self.groups == self.in_q

        kshape = (kernel_size, kernel_size)
        wshape = (self.out_q, self.in_q // self.groups, *kshape)
        self.r = nn.Parameter(torch.empty(wshape))
        self.i = nn.Parameter(torch.empty(wshape))
        self.j = nn.Parameter(torch.empty(wshape))
        self.k = nn.Parameter(torch.empty(wshape))
        _he_init_like(self.r, self.i, self.j, self.k)

        self.bias = nn.Parameter(torch.zeros(out_q_channels)) if bias else None
        self.padding = kernel_size // 2 if padding is None else padding

    def forward(self, x):
        w = quaternion_block_weight(self.r, self.i, self.j, self.k)
        return F.conv2d(x, w, self.bias, 1, self.padding, 1, self.groups)


class QuaternionTransposeConv(nn.Module):
    def __init__(
        self,
        in_q_channels,
        out_q_channels,
        scale_factor=2,
        overlap=False,
        kernel_size=None,
        groups=None,
        bias=True,
    ):
        super().__init__()
        assert in_q_channels % 4 == 0 and out_q_channels % 4 == 0
        self.in_q = in_q_channels // 4
        self.out_q = out_q_channels // 4
        self.groups = self.in_q if groups is None else groups
        s = scale_factor

        if not overlap:
            kernel_size = s
            padding = 0
            output_padding = 0
        else:
            kernel_size = s + 3 if kernel_size is None else kernel_size
            padding = (kernel_size - s) // 2
            output_padding = 0
            warnings.warn(f"[QTC] Overlap ON — blending will occur.")

        kshape = (kernel_size, kernel_size)
        wshape = (self.in_q, self.out_q // self.groups, *kshape)
        self.r = nn.Parameter(torch.empty(wshape))
        self.i = nn.Parameter(torch.empty(wshape))
        self.j = nn.Parameter(torch.empty(wshape))
        self.k = nn.Parameter(torch.empty(wshape))
        _he_init_like(self.r, self.i, self.j, self.k)

        self.bias = nn.Parameter(torch.zeros(out_q_channels)) if bias else None
        self.stride = s
        self.padding = padding
        self.output_padding = output_padding
        self.groups = self.groups

    def forward(self, x):
        w = quaternion_block_weight(self.r, self.i, self.j, self.k)
        return F.conv_transpose2d(
            x, w, self.bias, self.stride, self.padding, self.output_padding, self.groups
        )


# ========================================================================
# Residual block with adjustable kernel
# ========================================================================


class QuaternionResBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        g = channels // 4
        self.block = nn.Sequential(
            QuaternionConv(channels, channels, kernel_size, groups=g),
            # nn.GELU(),
            QuaternionConv(channels, channels, kernel_size, groups=g),
        )

    def forward(self, x):
        return x + self.block(x)


# ========================================================================
# Progressive Kernel Quaternion SR Net
# ========================================================================


class ProgressiveQuaternionSRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        in_ch = 16
        mid_ch = getattr(cfg, "n_feats", 64)
        out_ch = 16
        scale = getattr(cfg, "scale", 4)
        overlap = getattr(cfg, "overlap", False)
        base_k = getattr(cfg, "kernel_size", 3)
        num_res = getattr(cfg, "use_resblocks", 4)

        g_in = in_ch // 4
        g_mid = mid_ch // 4

        # Encoder with small kernel
        self.enc = nn.Sequential(
            QuaternionConv(in_ch, mid_ch, base_k, groups=g_in),
            # nn.GELU(),
            QuaternionConv(mid_ch, mid_ch, base_k, groups=g_mid),
            # nn.GELU(),
        )

        # Upsample
        self.up = QuaternionTransposeConv(
            mid_ch, mid_ch, scale_factor=scale, overlap=overlap, groups=g_mid
        )

        # Progressive residual refinement: kernel grows with depth
        res_layers = []
        for i in range(num_res):
            k_i = base_k + 2 * min(i, 2)  # e.g., 3,5,7,7,7...
            res_layers.append(QuaternionResBlock(mid_ch, k_i))
            # res_layers.append(nn.GELU())

        res_layers.append(QuaternionConv(mid_ch, out_ch, base_k, groups=g_mid))
        self.refine = nn.Sequential(*res_layers)

    def forward(self, q_in):
        x = quat_to_lmat(q_in)
        x = self.enc(x)
        x = self.up(x)
        x = self.refine(x)
        q_out = lmat_to_quat(x)
        q_out = q_out / q_out.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return q_out
