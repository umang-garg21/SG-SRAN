import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------ #
# Quaternion lifting and projection
# ------------------------------------------------------------------------------ #


def quat_to_lmat(q: torch.Tensor) -> torch.Tensor:
    """Lift quaternion image (B,4,H,W) to (B,16,H,W) via left multiplication matrix."""
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
    """Project lifted L-matrix back to quaternion (B,4,H,W) using the first column."""
    B, _, H, W = L.shape
    Lm = L.view(B, 4, 4, H, W)
    q = Lm[:, :, 0, :, :]
    q = q / (q.norm(dim=1, keepdim=True).clamp_min(1e-8))  # unit quaternion
    return q


# ------------------------------------------------------------------------------ #
# Quaternion convolution helpers
# ------------------------------------------------------------------------------ #


def _fan_in_fan_out(weight: torch.Tensor):
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
        in_q_channels,
        out_q_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=None,
        bias=True,
    ):
        super().__init__()
        assert in_q_channels % 4 == 0 and out_q_channels % 4 == 0
        self.in_q = in_q_channels // 4
        self.out_q = out_q_channels // 4
        self.groups = self.in_q if groups is None else groups
        # Allow groups that evenly divide the quaternion input count, and ensure
        # the raw output channel count (4 * out_q) is divisible by groups so
        # the expanded conv weight shape is compatible with torch.nn.functional.conv2d.
        assert (
            self.in_q % self.groups == 0
        ), f"groups must divide in_q={self.in_q}, got groups={self.groups}"
        assert (
            (4 * self.out_q) % self.groups == 0
        ), f"groups must divide raw out channels (4*out_q)={4*self.out_q}, got groups={self.groups}"

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

        self.bias = nn.Parameter(torch.zeros(out_q_channels)) if bias else None
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
        overlap=False,  # 👈 NEW FLAG
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
        assert self.groups == self.in_q, f"groups must equal in_q={self.in_q}"

        # Determine scale
        s = scale_factor if scale_factor is not None else stride
        self.stride = s

        # Auto kernel / padding for clean upsampling
        if not overlap:
            kernel_size = s
            padding = 0
            output_padding = 0
        else:
            # example: make kernel slightly larger than stride to induce blending
            kernel_size = kernel_size if kernel_size is not None else s + 3
            padding = padding if padding is not None else (kernel_size - s) // 2
            output_padding = output_padding if output_padding is not None else 0

        # Warn if overlap occurs
        if kernel_size > s:
            warnings.warn(
                f"[QuaternionTransposeConv] kernel_size ({kernel_size}) > stride ({s}) → overlapping patches, quaternion blending will occur."
            )

        # Adjust output_padding to ensure shape match
        required_output_padding = s - kernel_size + 2 * padding
        if required_output_padding >= 0:
            if output_padding != required_output_padding:
                output_padding = required_output_padding
        else:
            warnings.warn(
                f"[QuaternionTransposeConv] required_output_padding={required_output_padding} < 0, output shape may not match exactly."
            )

        self.padding = padding
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.dilation = dilation

        if isinstance(kernel_size, int):
            kshape = (kernel_size,) * 2
        else:
            kshape = tuple(kernel_size)

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
# Quaternion SR Net
# ------------------------------------------------------------------------------ #
class Quaternion_DeepConv_SRNet(nn.Module):
    def __init__(self, cfg):
        """
        Quaternion super-resolution network built from config.

        Required in cfg:
          - n_feats (int)
          - scale (int)

        Optional in cfg:
          - kernel_size (int, default=3)
          - overlap (bool, default=False)
          - dropout (float, currently unused)
        """
        super().__init__()
        self.cfg = cfg
        
        in_ch = 16  # lifted quaternion channels (4x4)
        mid_ch = getattr(cfg, "n_feats", 32)
        out_ch = 16
        scale_factor = getattr(cfg, "scale", 4)
        overlap = getattr(cfg, "overlap", False)
        k = getattr(cfg, "kernel_size", 3)
        n_resblocks = getattr(cfg, "n_resblocks", 4)

        # auto groups to respect quaternion structure
        g_in = in_ch // 4
        g_mid = mid_ch // 4
        g_out = out_ch // 4

        self.enc1 = QuaternionConv(in_ch, mid_ch, k, padding=k // 2, groups=g_in)
        self.enc2 = QuaternionConv(mid_ch, mid_ch, k, padding=k // 2, groups=g_mid)

        # Hyena-like gated conv blocks:
        # These blocks emulate attention-like multiplicative gating and long-range
        # mixing using inexpensive convolutional primitives. The design below
        # keeps quaternion channel divisibility and strives for the ``effectiveness
        # of attention with the compute profile of convolutions'' by combining
        # pointwise gating (GLU-like) and a spatial mixer (depthwise grouped conv).
        class HyenaResidualBlock(nn.Module):
            def __init__(self, channels, kernel=k, reduction=2, groups=None):
                super().__init__()
                # ensure bottleneck is divisible by 4 (quaternion groups)
                bottleneck = max(4, (channels // reduction) // 4 * 4)
                if bottleneck == 0:
                    bottleneck = 4

                # pointwise projections to compute content and gate (GLU-style)
                # both outputs maintain quaternion channel alignment
                self.content_proj = QuaternionConv(channels, bottleneck, 1, padding=0, groups=groups)
                self.gate_proj = QuaternionConv(channels, bottleneck, 1, padding=0, groups=groups)

                # spatial mixer: grouped quaternion conv with a reasonably large kernel
                # acts like a long-range depthwise mixer when groups == bottleneck//4
                grp = bottleneck // 4 if groups is None else groups
                self.spatial_mixer = QuaternionConv(
                    bottleneck, bottleneck, kernel, padding=kernel // 2, groups=grp
                )

                # final pointwise projection back to channels
                self.out_proj = QuaternionConv(bottleneck, channels, 1, padding=0, groups=groups)

                # small normalization & non-linearity to stabilise gating + residual
                self.act = nn.SiLU()
                # optional lightweight channel-wise normalization to help training
                # use GroupNorm with groups divisible by quaternion groups for stability
                gn_groups = max(1, (bottleneck // 4))
                # GroupNorm expects channel dimension as raw channels (4 * q channels)
                self.norm = nn.GroupNorm(gn_groups * 4, bottleneck)

            def forward(self, x):
                # compute content and gate pathways
                content = self.content_proj(x)
                gate = self.gate_proj(x)

                # multiplicative gating (GLU-like but using sigmoid gate)
                gated = content * torch.sigmoid(gate)

                # spatial mixing (cheap convolutional alternative to attention)
                mixed = self.spatial_mixer(gated)

                # normalization + non-linearity before projection back
                # (norm expects raw channel count; our QuaternionConv returns raw channels)
                mixed = self.norm(mixed)
                mixed = self.act(mixed)

                out = self.out_proj(mixed)

                # residual connection
                return x + out

        # Lightweight global attention block that computes attention on a small
        # downsampled spatial grid (e.g. 8x8) and broadcasts the result back.
        # This captures global context cheaply and helps produce smooth boundaries
        # similar to self-attention but with much lower cost.
        class GlobalDownsampleAttention(nn.Module):
            def __init__(self, channels, attn_spatial=8):
                super().__init__()
                assert channels % 4 == 0
                self.channels = channels
                # keep quaternion alignment by using a QuaternionConv before/after
                self.pre_proj = QuaternionConv(channels, channels, 1, padding=0)
                # use regular 1x1 convs for q/k/v on raw channels
                self.qkv = nn.Conv2d(channels, channels * 3, 1)
                self.out_proj = QuaternionConv(channels, channels, 1, padding=0)
                self.attn_spatial = attn_spatial

            def forward(self, x):
                # x: (B, C, H, W) where C is raw channels (4*q)
                B, C, H, W = x.shape
                y = self.pre_proj(x)
                qkv = self.qkv(y)
                # downsample spatially to small grid for attention
                ps = min(self.attn_spatial, H, W)
                if ps <= 1:
                    # degenerate case: fallback to identity
                    return x
                qkv_ds = F.adaptive_avg_pool2d(qkv, (ps, ps))
                # split q,k,v and reshape to (B, N, C)
                Craw = C
                q, k, v = torch.chunk(qkv_ds, 3, dim=1)
                N = ps * ps
                q = q.view(B, Craw, N).permute(0, 2, 1)  # (B, N, C)
                k = k.view(B, Craw, N).permute(0, 2, 1)
                v = v.view(B, Craw, N).permute(0, 2, 1)

                # scaled dot-product attention over small N
                scale = Craw ** 0.5
                attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / scale, dim=-1)  # (B,N,N)
                out = torch.bmm(attn, v)  # (B,N,C)

                # project back to spatial grid and upsample
                out = out.permute(0, 2, 1).contiguous().view(B, Craw, ps, ps)
                out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
                out = self.out_proj(out)
                return x + out

        # Build a mixed stack: interleave HyenaResidualBlock and occasional
        # lightweight global attention blocks to get smoothing similar to
        # full self-attention while keeping compute low.
        blocks = []
        for i in range(n_resblocks):
            blocks.append(HyenaResidualBlock(mid_ch, kernel=k))
            # insert an attention block every 2 residuals (configurable)
            if (i + 1) % 2 == 0:
                blocks.append(GlobalDownsampleAttention(mid_ch, attn_spatial=8))

        self.body = nn.Sequential(*blocks)

        self.up = QuaternionTransposeConv(
            in_q_channels=mid_ch,
            out_q_channels=mid_ch,
            scale_factor=scale_factor,
            overlap=overlap,
            groups=g_mid,
        )
        self.outc = QuaternionConv(mid_ch, out_ch, k, padding=k // 2, groups=g_out)
        # self.act = nn.ReLU(inplace=True)

    def forward(self, q_in):
        # Lift quaternions to 16-channel real representation
        x = quat_to_lmat(q_in)
        # x = self.act(self.enc1(x))
        # x = self.act(self.enc2(x))
        # x = self.act(self.up(x))
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.body(x)
        x = self.up(x)
        x = self.outc(x)
        q_out = lmat_to_quat(x)
        # Normalize to unit quaternion after possible blending
        # q_out = q_out / q_out.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return q_out


# class QuaternionSRNet(nn.Module):
#     def __init__(self, base_q_channels=16, scale_factor=4, overlap=False):
#         super().__init__()
#         in_ch = 16
#         mid_ch = base_q_channels
#         out_ch = 16

#         self.enc1 = QuaternionConv(in_ch, mid_ch, 3, padding=1)
#         self.enc2 = QuaternionConv(mid_ch, mid_ch, 3, padding=1)
#         self.up = QuaternionTransposeConv(
#             in_q_channels=mid_ch,
#             out_q_channels=mid_ch,
#             scale_factor=scale_factor,
#             overlap=overlap,
#         )
#         self.outc = QuaternionConv(mid_ch, out_ch, 3, padding=1)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, q_in):
#         x = quat_to_lmat(q_in)
#         x = self.act(self.enc1(x))
#         x = self.act(self.enc2(x))
#         x = self.act(self.up(x))
#         x = self.outc(x)
#         q_out = lmat_to_quat(x)
#         # Normalize to unit quaternion after possible blending
#         q_out = q_out / q_out.norm(dim=1, keepdim=True).clamp_min(1e-8)
#         return q_out


# ------------------------------------------------------------------------------ #
# Example usage
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":
    B = 2
    H = W = 32
    scale = 4

    print("\nNon-overlapping SR")
    net_clean = Quaternion_res_SRNet(base_q_channels=32, scale_factor=scale, overlap=False)
    q_lr = torch.randn(B, 4, H, W)
    q_lr = q_lr / q_lr.norm(dim=1, keepdim=True).clamp_min(1e-8)
    q_sr = net_clean(q_lr)
    print("Output shape (clean):", q_sr.shape)  # expected (B, 4, 128, 128)

    print("\nOverlapping SR")
    net_overlap = Quaternion_res_SRNet(base_q_channels=64, scale_factor=scale, overlap=True)
    q_sr2 = net_overlap(q_lr)
    print("Output shape (overlap):", q_sr2.shape)
