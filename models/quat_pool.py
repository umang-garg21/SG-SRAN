import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings


# ------------------------------------------------------------------------------ #
# Quaternion lifting and projection
# ------------------------------------------------------------------------------ #
def quat_to_lmat(q: torch.Tensor) -> torch.Tensor:
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


def lmat_to_quat(L: torch.Tensor) -> torch.Tensor:
    B, _, H, W = L.shape
    Lm = L.view(B, 4, 4, H, W)
    q = Lm[:, :, 0, :, :]
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    return q


# ------------------------------------------------------------------------------ #
# Quaternion Conv core helpers
# ------------------------------------------------------------------------------ #
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


# ------------------------------------------------------------------------------ #
# Quaternion Conv and Transpose Conv
# ------------------------------------------------------------------------------ #
class QuaternionConv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=None,
        bias=True,
    ):
        super().__init__()
        assert in_ch % 4 == 0 and out_ch % 4 == 0
        self.in_q = in_ch // 4
        self.out_q = out_ch // 4
        self.groups = self.in_q if groups is None else groups
        assert self.groups == self.in_q

        if isinstance(kernel_size, int):
            kshape = (kernel_size,) * 2
        else:
            kshape = tuple(kernel_size)

        wshape = (self.out_q, self.in_q // self.groups, *kshape)
        self.r = nn.Parameter(torch.empty(wshape))
        self.i = nn.Parameter(torch.empty(wshape))
        self.j = nn.Parameter(torch.empty(wshape))
        self.k = nn.Parameter(torch.empty(wshape))
        _he_init_like(self.r, self.i, self.j, self.k)

        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        w = quaternion_block_weight(self.r, self.i, self.j, self.k)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class QuaternionTransposeConv(nn.Module):
    def __init__(
        self,
        in_q_channels,
        out_q_channels,
        stride=2,
        scale_factor=None,
        overlap=False,
        kernel_size=None,
        padding=None,
        output_padding=None,
        dilation=1,
        groups=None,
        bias=True,
    ):
        super().__init__()
        assert in_q_channels % 4 == 0 and out_q_channels % 4 == 0
        self.in_q = in_q_channels // 4
        self.out_q = out_q_channels // 4
        self.groups = self.in_q if groups is None else groups
        assert self.groups == self.in_q

        s = scale_factor if scale_factor is not None else stride
        self.stride = s

        if not overlap:
            kernel_size = s
            padding = 0
            output_padding = 0
        else:
            kernel_size = kernel_size if kernel_size is not None else s + 3
            padding = padding if padding is not None else (kernel_size - s) // 2
            output_padding = output_padding if output_padding is not None else 0

        if kernel_size > s:
            warnings.warn(
                f"[QuaternionTransposeConv] kernel_size ({kernel_size}) > stride ({s}) → overlapping patches."
            )

        required_output_padding = s - kernel_size + 2 * padding
        if required_output_padding >= 0:
            output_padding = required_output_padding
        else:
            warnings.warn(
                f"[QuaternionTransposeConv] required_output_padding={required_output_padding} < 0"
            )

        self.padding = padding
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.dilation = dilation

        kshape = (kernel_size, kernel_size)
        wshape = (self.in_q, self.out_q // self.groups, *kshape)
        self.r = nn.Parameter(torch.empty(wshape))
        self.i = nn.Parameter(torch.empty(wshape))
        self.j = nn.Parameter(torch.empty(wshape))
        self.k = nn.Parameter(torch.empty(wshape))
        _he_init_like(self.r, self.i, self.j, self.k)

        self.bias = nn.Parameter(torch.zeros(out_q_channels)) if bias else None

    def forward(self, x):
        w = quaternion_block_weight(self.r, self.i, self.j, self.k)
        return F.conv_transpose2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


# ------------------------------------------------------------------------------ #
# Quaternion-safe pooling
# ------------------------------------------------------------------------------ #
class QuaternionMagnitudeAvgPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, q):
        mag = torch.linalg.norm(q, dim=1, keepdim=True)
        weighted = q * mag
        pooled_weighted = F.avg_pool2d(
            weighted, self.kernel_size, self.stride, self.padding
        )
        pooled_mag = F.avg_pool2d(mag, self.kernel_size, self.stride, self.padding)
        out = pooled_weighted / (pooled_mag + 1e-8)
        out = out / (out.norm(dim=1, keepdim=True) + 1e-8)
        return out


# ------------------------------------------------------------------------------ #
# Optional Residual Block
# ------------------------------------------------------------------------------ #
class QuaternionResBlock(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        groups = ch // 4
        self.conv1 = QuaternionConv(ch, ch, k, padding=k // 2, groups=groups)
        self.conv2 = QuaternionConv(ch, ch, k, padding=k // 2, groups=groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return out


# ------------------------------------------------------------------------------ #
# Quaternion SR U-Net Style Network
# ------------------------------------------------------------------------------ #
class QuaternionPoolSRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # -----------------------------
        # Core hyperparameters
        # -----------------------------
        in_ch = 16  # lifted quaternion (4x4)
        mid_ch = getattr(cfg, "n_feats", 64)
        out_ch = 16
        base_scale = getattr(cfg, "scale", 4)
        overlap = getattr(cfg, "overlap", False)
        k = getattr(cfg, "kernel_size", 3)
        num_resblocks = getattr(cfg, "num_resblocks", 3)

        # -----------------------------
        # Auto-compute effective scale
        # -----------------------------
        self.down_factor = 2  # since we use 1 pooling layer (2x)
        self.effective_scale = base_scale * self.down_factor

        # -----------------------------
        # Groups
        # -----------------------------
        g_in = in_ch // 4
        g_mid = mid_ch // 4

        # -----------------------------
        # Encoder
        # -----------------------------
        self.enc1 = QuaternionConv(in_ch, mid_ch, k, padding=k // 2, groups=g_in)
        self.pool1 = QuaternionMagnitudeAvgPool2d(2)
        self.enc2 = QuaternionConv(mid_ch, mid_ch, k, padding=k // 2, groups=g_mid)

        # -----------------------------
        # Residual blocks
        # -----------------------------
        self.resblocks = nn.Sequential(
            *[QuaternionResBlock(mid_ch, k) for _ in range(num_resblocks)]
        )

        # -----------------------------
        # Decoder / Upsampling
        # -----------------------------
        self.up = QuaternionTransposeConv(
            in_q_channels=mid_ch,
            out_q_channels=mid_ch,
            scale_factor=self.effective_scale,
            overlap=overlap,
            groups=g_mid,
        )

        self.outc = QuaternionConv(mid_ch, out_ch, k, padding=k // 2, groups=g_mid)

    def forward(self, q_in):
        B, C, H, W = q_in.shape
        target_h = H * self.cfg.scale
        target_w = W * self.cfg.scale

        # -----------------------------
        # Forward pass
        # -----------------------------
        x = quat_to_lmat(q_in)
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.resblocks(x)

        # Force output shape to match exactly expected scale
        x = self.up(x)
        x = self.outc(x)

        q_out = lmat_to_quat(x)
        q_out = q_out / q_out.norm(dim=1, keepdim=True).clamp_min(
            1e-8
        )  # unit quaternion
        return q_out


# ------------------------------------------------------------------------------ #
# Example
# ------------------------------------------------------------------------------ #
if __name__ == "__main__":
    B, H, W = 2, 32, 32
    scale = 4
    q_lr = torch.randn(B, 4, H, W)
    q_lr = q_lr / (q_lr.norm(dim=1, keepdim=True) + 1e-8)

    class DummyCfg:
        n_feats = 64
        scale = scale
        overlap = False
        kernel_size = 3
        num_resblocks = 3

    net = QuaternionPoolSRNet(DummyCfg())
    q_sr = net(q_lr)
    print(f"✅ Output shape: {q_sr.shape}")
