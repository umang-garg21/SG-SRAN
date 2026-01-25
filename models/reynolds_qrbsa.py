import math
import warnings

# from Archive.model.quat_utils.Qops_with_QSN import Residual_SA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


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
            4 * self.out_q
        ) % self.groups == 0, f"groups must divide raw out channels (4*out_q)={4*self.out_q}, got groups={self.groups}"

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
            # kernel_size = kernel_size if kernel_size is not None else s + 3
            # padding = padding if padding is not None else (kernel_size - s) // 2
            # output_padding = output_padding if output_padding is not None else 0
            kernel_size = s*2
            padding = padding if padding is not None else kernel_size // 4
            output_padding = output_padding if output_padding is not None else s // 2

        # Warn if overlap occurs
        if kernel_size > s:
            warnings.warn(
                f"[QuaternionTransposeConv] kernel_size ({kernel_size}) > stride ({s}) → overlapping patches, quaternion blending will occur."
            )

        assert kernel_size == 8
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


class QuatLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        assert num_channels % 4 == 0, "Channels must be multiple of 4"
        self.num_q = num_channels // 4
        self.weight = nn.Parameter(torch.ones(self.num_q, 1, 1))
        self.bias = nn.Parameter(torch.zeros(self.num_q, 1, 1))
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.num_q, 4, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.unsqueeze(2) + self.bias.unsqueeze(2)
        return x.view(B, C, H, W)


class QAttention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = QuaternionConv(dim, dim * 3, kernel_size=1, stride=1, padding=0)
        self.qkv_dw = QuaternionConv(dim * 3, dim * 3, kernel_size=3, stride=1)
        self.proj = QuaternionConv(dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (h c) h1 w1 -> b h c (h1 w1)", h=self.num_heads)
        k = rearrange(k, "b (h c) h1 w1 -> b h c (h1 w1)", h=self.num_heads)
        v = rearrange(v, "b (h c) h1 w1 -> b h c (h1 w1)", h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(
            out, "b h c (h1 w1) -> b (h c) h1 w1", h=self.num_heads, h1=H, w1=W
        )

        return self.proj(out)


def q_gelu(x, eps=1e-8):
    """
    Quaternion GELU: CHECK https://link.springer.com/article/10.1007/s00006-024-01350-x
    Applies GELU to quaternion norm and rescales q.
    x: (B, C, H, W), C divisible by 4
    """
    B, C, H, W = x.shape
    q = x.view(B, C // 4, 4, H, W)  # (B, n, 4, H, W)
    norm = torch.sqrt((q**2).sum(dim=2, keepdim=True) + eps)  # (B, n, 1, H, W)
    gate = F.gelu(norm)  # scalar gate
    q = q * gate / (norm + eps)  # scale quaternion packet
    return q.view(B, C, H, W)


class QFeedForward(nn.Module):
    def __init__(self, dim, group_tensor, group_tensor_inv, ffn_expansion_factor=2.0):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)

        self.project_in = EquivariantReynoldsWrap(
            QuaternionConv(dim, hidden * 2, kernel_size=1, stride=1, padding=0),
            group_tensor,
            group_tensor_inv,
        )
        self.dwconv = EquivariantReynoldsWrap(
            QuaternionConv(hidden * 2, hidden * 2, kernel_size=3, stride=1),
            group_tensor,
            group_tensor_inv,
        )
        self.project_out = EquivariantReynoldsWrap(
            QuaternionConv(hidden * 2, dim, kernel_size=1, stride=1, padding=0),
            group_tensor,
            group_tensor_inv,
        )

    def forward(self, x):
        x = self.project_in(x)
        x = q_gelu(x)  # <--- EQUIVARIANT NONLINEARITY
        x = self.dwconv(x)
        x = q_gelu(x)  # <--- optional second activation
        x = self.project_out(x)
        return x


class QTransformerBlock(nn.Module):
    """
    Quaternion Transformer Block:
      x = x + Attention(LN(x))
      x = x + FeedForward(LN(x))
    """

    def __init__(
        self, dim, group_tensor, group_tensor_inv, num_heads=1, ffn_expansion_factor=2.0
    ):
        super().__init__()

        self.norm1 = EquivariantReynoldsWrap(
            QuatLayerNorm(
                dim,
            ),
            group_tensor,
            group_tensor_inv,
        )

        self.attn = EquivariantReynoldsWrap(
            QAttention(dim, num_heads),
            group_tensor,
            group_tensor_inv,
        )
        self.norm2 = EquivariantReynoldsWrap(
            QuatLayerNorm(
                dim,
            ),
            group_tensor,
            group_tensor_inv,
        )
        self.ffn = QFeedForward(
            dim,
            group_tensor,
            group_tensor_inv,
            ffn_expansion_factor,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def q_relu(x, eps=1e-8):
    """
    Quaternion ReLU: https://link.springer.com/article/10.1007/s00006-024-01350-x
        Scale quaternion by ReLU(norm).
    x: (B, C, H, W), C divisible by 4
    """
    B, C, H, W = x.shape
    q = x.view(B, C // 4, 4, H, W)  # (B, n, 4, H, W)
    norm = torch.sqrt((q**2).sum(dim=2, keepdim=True) + eps)  # (B, n, 1, H, W)

    gate = F.relu(norm)  # Apply ReLU to scalar norm
    q = q * gate / (norm + eps)  # Scale quaternion packet

    return q.view(B, C, H, W)


class Residual_SA(nn.Module):
    """
    Quaternion Residual Block with Self-Attention
    Matches functionality of the original:
        conv2d → ReLU → conv2d → QTransformerBlock → residual add

    Requirements:
      - in_channels and out_channels must be multiples of 4 (quaternion format).
      - QuaternionConv is your current one (in_q_channels, out_q_channels, ...).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        group_tensor: torch.Tensor,
        group_tensor_inv: torch.Tensor,
        kernel: int = 3,
        stride: int = 1,
        num_heads: int = 1,
        ffn_expansion_factor: float = 1.0,
    ):
        super().__init__()

        assert in_channels % 4 == 0
        assert out_channels % 4 == 0

        pad = kernel // 2
        g_in = in_channels // 4
        g_out = out_channels // 4

        # First quaternion conv
        self.conv1 = EquivariantReynoldsWrap(
            QuaternionConv(
                in_q_channels=in_channels,
                out_q_channels=in_channels,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                groups=g_in,
                bias=True,
            ),
            group_tensor,
            group_tensor_inv,
        )

        self.relu = q_relu

        # Second quaternion conv
        self.conv2 = EquivariantReynoldsWrap(
            QuaternionConv(
                in_q_channels=in_channels,
                out_q_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                groups=g_out,
                bias=True,
            ),
            group_tensor,
            group_tensor_inv,
        )

        self.sa = QTransformerBlock(
            dim=out_channels,
            group_tensor=group_tensor,
            group_tensor_inv=group_tensor_inv,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sa(y)
        y += x
        return y


class EquivariantReynoldsWrap(nn.Module):
    """
    Reynolds operator wrapper: enforces equivariance for any module `fn`
    under a group action represented by group_tensor (G, Cg, Cg).
    Input/output channel dims must be multiples of Cg (=4 for quats).
    Works with inputs (B, C, *spatial).
    """

    def __init__(
        self, fn: nn.Module, group_tensor: torch.Tensor, group_tensor_inv: torch.Tensor
    ):
        super().__init__()
        self.fn = fn
        self.register_buffer("group_tensor", group_tensor)  # (G, Cg, Cg)
        self.register_buffer("group_tensor_inv", group_tensor_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *spatial = x.shape
        G, Cg, _ = self.group_tensor.shape
        assert C % Cg == 0, f"Channels {C} must be multiple of {Cg}"
        n_feats = C // Cg

        # Lift (apply group action on quaternion axis)
        x = x.view(B, n_feats, Cg, *spatial)  # (B,n,Cg,*)
        # gamma_x[b,g,n,c,...] = sum_i group[g,c,i] * x[b,n,i,...]
        gamma_x = torch.einsum("gci,bni...->bgnc...", self.group_tensor, x).reshape(
            B * G, n_feats * Cg, *spatial
        )

        # Apply wrapped op
        fx = self.fn(gamma_x)  # (B*G, Cout, *s')
        BG, Cout, *spatial_out = fx.shape
        assert BG == B * G and Cout % Cg == 0
        n_out = Cout // Cg

        fx = fx.view(B, G, n_out, Cg, *spatial_out)  # (B,G,n_out,Cg,*)
        # project back: sum_i group_inv[g,c,i] * fx[b,g,n,i,...]
        fx = torch.einsum("gci,bgni...->bgnc...", self.group_tensor_inv, fx)

        # Average over group
        return fx.mean(dim=1).reshape(B, Cout, *spatial_out)


# ------------------------------------------------------------------------------ #
# Quaternion SR Net
# ------------------------------------------------------------------------------ #
class Equivariant_QRBSA(nn.Module):
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

        gt = torch.tensor(np.load(cfg.sym_np_path), dtype=torch.float32)
        gti = torch.tensor(np.load(cfg.sym_inv_np_path), dtype=torch.float32)
        self.register_buffer("group_tensor", gt)
        self.register_buffer("group_tensor_inv", gti)

        in_ch = 4  # lifted quaternion channels (4x4)
        mid_ch = getattr(cfg, "n_feats", 32)
        out_ch = 4
        scale_factor = getattr(cfg, "scale", 4)
        overlap = getattr(cfg, "overlap", False)
        k = getattr(cfg, "kernel_size", 3)
        n_resblocks = getattr(cfg, "n_resblocks", 4)

        # auto groups to respect quaternion structure
        g_in = in_ch // 4
        g_mid = mid_ch // 4
        # g_out = out_ch // 4

        self.act = q_relu

        self.enc1 = EquivariantReynoldsWrap(
            QuaternionConv(in_ch, mid_ch, k, padding=k // 2, groups=g_in),
            self.group_tensor,
            self.group_tensor_inv,
        )

        self.enc2 = EquivariantReynoldsWrap(
            QuaternionConv(mid_ch, mid_ch, k, padding=k // 2, groups=g_mid),
            self.group_tensor,
            self.group_tensor_inv,
        )

        self.body = nn.Sequential(
            *[
                Residual_SA(
                    mid_ch,
                    mid_ch,
                    group_tensor=self.group_tensor,
                    group_tensor_inv=self.group_tensor_inv,
                )
                for _ in range(n_resblocks)
            ]
        )

        self.up = EquivariantReynoldsWrap(
            QuaternionTransposeConv(
                in_q_channels=mid_ch,
                out_q_channels=mid_ch,
                scale_factor=scale_factor,
                overlap=overlap,
                groups=g_mid,
            ),
            self.group_tensor,
            self.group_tensor_inv,
        )

        self.outc = EquivariantReynoldsWrap(
            QuaternionConv(
                mid_ch, in_ch, k, padding=k // 2, groups=g_in
            ),  # MAYBE  groups= mid_ch ?? CHECK
            self.group_tensor,
            self.group_tensor_inv,
        )

    def forward(self, x):
        # Store input dimensions for cropping
        B, C, H_in, W_in = x.shape
        expected_H_out = H_in * self.cfg.scale
        expected_W_out = W_in * self.cfg.scale
        
        # x = self.act(self.enc1(x))
        x = self.enc1(x)
        # x = self.act(self.enc2(x))
        x = self.enc2(x)
        x = self.body(x) + x
        x = self.up(x)
        x = self.outc(x)

        # Crop output to match expected HR dimensions (4x input size)
        H_out, W_out = x.shape[2], x.shape[3]
        if H_out > expected_H_out or W_out > expected_W_out:
            # Center crop to expected dimensions
            h_start = (H_out - expected_H_out) // 2
            w_start = (W_out - expected_W_out) // 2
            x = x[:, :, h_start:h_start + expected_H_out, w_start:w_start + expected_W_out]

        # Normalize to unit quaternion after possible blending
        # x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return x


# ------------------------------------------------------------------------------ #
# Example usage
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":

    class _Args:
        sym_np_path = "/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group.npy"
        sym_inv_np_path = (
            "/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group_inv.npy"
        )
        dropout = 0.0

    args = _Args()
    B = 2
    H = W = 32
    scale = 4

    print("\nNon-overlapping SR")
    model = Equivariant_QRBSA(cfg=args)

    q_lr = torch.randn(B, 4, H, W)
    q_lr = q_lr / q_lr.norm(dim=1, keepdim=True).clamp_min(1e-8)
    q_sr = model(q_lr)

    print("Output shape (clean):", q_sr.shape)  # expected (B, 4, 128, 128)

    def test_model_equivariance(
        model,
        x: torch.Tensor,
        n_check: int | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        """
        Test equivariance of the model under the loaded group action:
            f(g·x) ≈ g·f(x)

        Parameters
        ----------
        model : nn.Module
            Model with .group_tensor (G, Cg, Cg)
        x : torch.Tensor
            Input tensor (B, C, H, W)
        n_check : int or None
            Number of group elements to test; if None, checks all
        atol : float
            Absolute tolerance
        rtol : float
            Relative tolerance

        Returns
        -------
        passed : bool
            True if equivariance holds within tolerance
        max_err : float
            Maximum error across all group elements tested
        errs : list[float]
            Per-group element max error
        """
        model.eval()
        device = next(model.parameters()).device
        x = x.to(device)

        with torch.inference_mode():
            fx = model(x)
            G = model.group_tensor.shape[0]
            Cg = model.group_tensor.shape[1]

            if n_check is None or n_check > G:
                n_check = G
            idx = torch.arange(n_check)

            errs = []
            for g_idx in idx:
                gmat = model.group_tensor[g_idx].to(device)  # (Cg, Cg)
                # g·x
                gx = torch.einsum("ci,bi...->bc...", gmat, x)
                # f(g·x)
                f_gx = model(gx)
                # g·f(x)
                g_fx = torch.einsum("ci,bi...->bc...", gmat, fx)
                # max error for this element
                err = (f_gx - g_fx).abs().max().item()
                errs.append(err)

            max_err = max(errs)
            tol = atol + rtol * fx.abs().max().item()
            passed = max_err <= tol

        print(f"\n[Equivariance Test]")
        print(f"Checked {n_check}/{G} group elements")
        print(f"Tolerance: {tol:.3e}")
        print(f"Max error: {max_err:.3e}")
        print(f"Per-group errors: {[round(e, 6) for e in errs]}")
        print(f"✅ Equivariant: {passed}\n")

        return passed, max_err, errs

    passed, max_err, errs = test_model_equivariance(model, q_lr, n_check=None)

    # print("\nOverlapping SR")
    # args.overlap
    # net_overlap = Quaternion_res_SRNet(cfg=args)
    # q_sr2 = net_overlap(q_lr)
    # print("Output shape (overlap):", q_sr2.shape)
