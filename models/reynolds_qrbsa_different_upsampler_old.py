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


class QuaternionNormalize(nn.Module):
    """Normalize quaternion channels to unit length."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return F.normalize(x, dim=1, eps=self.eps)


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

def quat_log(q, eps=1e-8):
    """
    Quaternion logarithm for unit quaternion q:
      q = [w, v], w = cos(theta/2), v = axis * sin(theta/2)
    log(q) = axis * (theta/2)
    Input:
      q: (..., 4)
    Output:
      (..., 3) axis-angle vector (axis * (theta/2))
    """
    q = F.normalize(q, dim=-1, eps=eps)
    w = q[..., 0]
    v = q[..., 1:]
    # theta_half = acos(w)  (this equals theta/2)
    theta_half = torch.acos(torch.clamp(w, -1.0, 1.0))
    sin_theta_half = torch.sin(theta_half)
    # When sin_theta_half ~ 0, axis is undefined; divide safely
    axis = v / (sin_theta_half.unsqueeze(-1) + eps)
    # log(q) should be theta_half * axis (NOT theta_half/2)
    log_q = theta_half.unsqueeze(-1) * axis
    return log_q

def quat_exp(v, eps=1e-8):
    """
    Quaternion exponential / axis-angle -> quaternion.
    Input:
      v: (..., 3) where v = axis * (theta/2)  (i.e. same convention as quat_log)
    Output:
      (..., 4) quaternion unit
    If theta = ||v||, then quaternion = [cos(theta), sin(theta) * (v/theta)]
    """
    theta = torch.norm(v, dim=-1, keepdim=True)  # theta == theta_half
    axis = v / (theta + eps)
    cos_comp = torch.cos(theta)       # cos(theta_half)
    sin_comp = torch.sin(theta)       # sin(theta_half)
    q = torch.cat([cos_comp, sin_comp * axis], dim=-1)
    return F.normalize(q, dim=-1, eps=eps)

class SlerpUpsample(nn.Module):
    """
    Vectorized SLERP upsampling for quaternions with symmetry groups.

    group_tensor: (G, 4, 4) rotation matrices (applied to quaternion column vectors)
    group_tensor_inv: (G, 4, 4) inverse matrices
    """

    def __init__(self, scale_factor, group_tensor, group_tensor_inv):
        super().__init__()
        self.scale_factor = int(scale_factor)
        # Expect group_tensor shape (G,4,4) and group_tensor_inv shape (G,4,4)
        self.register_buffer("group_tensor", group_tensor)         # (G,4,4)
        self.register_buffer("group_tensor_inv", group_tensor_inv) # (G,4,4)

    def slerp_vectorized(self, q0, q1, t, eps=1e-8):
        """
        q0, q1: (B, n, 4, H, W)
        t: either scalar or (H, W) or (H*W,) interpolation factor in [0,1]
        returns: (B, n, 4, H, W)
        """
        B, n, _, H, W = q0.shape
        G = self.group_tensor.shape[0]
        S = H * W

        # normalize
        q0 = F.normalize(q0, dim=2, eps=eps)
        q1 = F.normalize(q1, dim=2, eps=eps)

        # flatten spatial dims: (B, n, S, 4)
        q0_flat = q0.permute(0, 1, 3, 4, 2).reshape(B, n, S, 4)
        q1_flat = q1.permute(0, 1, 3, 4, 2).reshape(B, n, S, 4)

        # Apply all group elements (matrix multiply): result shape (B, G, n, S, 4)
        # einsum: group 'gij', q 'bnsj' -> 'bgnsi' where 'i' is output component
        gamma_q0_all = torch.einsum('gij,bnsj->bgnsi', self.group_tensor, q0_flat)  # (B,G,n,S,4)
        gamma_q1_all = torch.einsum('gij,bnsj->bgnsi', self.group_tensor, q1_flat)  # (B,G,n,S,4)

        # Compute dot product across quaternion components -> (B,G,n,S)
        dot_all = (gamma_q0_all * gamma_q1_all).sum(dim=-1)

        # Handle double cover: flip sign of q1 if dot < 0
        sign_flip = (dot_all < 0).unsqueeze(-1)  # (B,G,n,S,1)
        gamma_q1_all = torch.where(sign_flip, -gamma_q1_all, gamma_q1_all)
        dot_all = torch.abs(dot_all)

        # clamp and compute angle
        dot_clamped = torch.clamp(dot_all, -1.0 + 1e-7, 1.0 - 1e-7)
        theta_all = torch.acos(dot_clamped)  # (B,G,n,S)

        # Pick minimal theta across group axis
        min_theta, min_idx = theta_all.min(dim=1)  # (B,n,S), (B,n,S)

        # Gather best quaternions for q0 and q1
        # permute to (B, n, G, S, 4) to gather on dim=2
        gamma_q0_perm = gamma_q0_all.permute(0, 2, 1, 3, 4)  # (B,n,G,S,4)
        gamma_q1_perm = gamma_q1_all.permute(0, 2, 1, 3, 4)  # (B,n,G,S,4)

        # min_idx shape (B,n,S) -> expand for gather
        gather_idx = min_idx.unsqueeze(-1).unsqueeze(-1)  # (B,n,S,1,1)
        gather_idx = gather_idx.expand(-1, -1, -1, 1, 4)  # (B,n,S,1,4)
        # need indices on dim=2 (G), so reorder dims to (B, n, G, S, 4) and gather dim=2
        best_q0 = torch.gather(gamma_q0_perm, 2, gather_idx).squeeze(2)  # (B,n,S,4)
        best_q1 = torch.gather(gamma_q1_perm, 2, gather_idx).squeeze(2)  # (B,n,S,4)

        # reshape back to (B,n,4,H,W)
        q0_best = best_q0.view(B, n, S, 4).permute(0, 1, 3, 2).reshape(B, n, 4, H, W)
        q1_best = best_q1.view(B, n, S, 4).permute(0, 1, 3, 2).reshape(B, n, 4, H, W)

        # Prepare theta_best: (B,n,S) -> (B,n,1,H,W)
        theta_best = min_theta.view(B, n, 1, H, W)
        sin_theta = torch.sin(theta_best)

        # prepare t: accept (H,W) or scalar
        if torch.is_tensor(t):
            t_tensor = t.to(q0.device).view(1, 1, 1, H, W)
        else:
            t_tensor = torch.tensor(float(t), device=q0.device).view(1, 1, 1, 1, 1)

        # SLERP weights, safe divide
        w0 = torch.sin((1.0 - t_tensor) * theta_best) / (sin_theta + eps)
        w1 = torch.sin(t_tensor * theta_best) / (sin_theta + eps)

        # fallback to linear when theta small
        use_linear = (sin_theta.abs() < eps) | (theta_best.abs() < eps)
        w0 = torch.where(use_linear, 1.0 - t_tensor, w0)
        w1 = torch.where(use_linear, t_tensor, w1)

        # Result before FZ reduction: (B,n,4,H,W)
        result = w0 * q0_best + w1 * q1_best
        result = F.normalize(result, dim=2, eps=eps)

        # FZ reduce: apply inverse group matrix corresponding to chosen min_idx
        # min_idx flattened -> (B*n*S,)
        idx_flat = min_idx.view(-1)  # (B*n*S,)
        inv_selected = self.group_tensor_inv[idx_flat]  # (B*n*S,4,4)

        # result_flat: (B*n*S, 4)
        result_flat = result.permute(0, 1, 3, 4, 2).reshape(-1, 4)  # (B*n*S,4)

        # batch multiply: (B*n*S,4,4) x (B*n*S,4,1) -> (B*n*S,4,1)
        fz_out = torch.bmm(inv_selected, result_flat.unsqueeze(-1)).squeeze(-1)  # (B*n*S,4)

        # reshape back to (B, n, 4, H, W)
        fz_result = fz_out.view(B, n, H, W, 4).permute(0, 1, 4, 2, 3)

        return fz_result

    def forward(self, x):
        """
        x: (B, C, H, W) where C = n_quats * 4
        returns: (B, C, new_H, new_W)
        """
        B, C, H, W = x.shape
        assert C % 4 == 0, "Channels must be multiple of 4"
        n_quats = C // 4
        x = x.view(B, n_quats, 4, H, W)

        new_H = H * self.scale_factor
        new_W = W * self.scale_factor

        # build continuous coordinates (output pixel centers projected back to input)
        yy = (torch.arange(new_H, device=x.device, dtype=torch.float32) / self.scale_factor)
        xx = (torch.arange(new_W, device=x.device, dtype=torch.float32) / self.scale_factor)

        y0 = torch.floor(yy).long().clamp(0, H - 1)
        x0 = torch.floor(xx).long().clamp(0, W - 1)
        y1 = torch.clamp(y0 + 1, max=H - 1)
        x1 = torch.clamp(x0 + 1, max=W - 1)

        # bilinear weights
        wy = (yy - y0.float()).view(new_H, 1)  # (new_H,1)
        wx = (xx - x0.float()).view(1, new_W)  # (1,new_W)
        w_xy = (wy.unsqueeze(-1) * (1 - wx).unsqueeze(0))  # not used directly; we'll do separable slerps

        # Build index grids for sampling: (new_H, new_W)
        y0g, x0g = torch.meshgrid(y0, x0, indexing='ij')
        y1g, x0g2 = torch.meshgrid(y1, x0, indexing='ij')
        y0g2, x1g = torch.meshgrid(y0, x1, indexing='ij')
        y1g2, x1g2 = torch.meshgrid(y1, x1, indexing='ij')

        # sample four corner quaternions (B, n, 4, new_H, new_W)
        q00 = x[:, :, :, y0g, x0g]
        q01 = x[:, :, :, y0g2, x1g]
        q10 = x[:, :, :, y1g, x0g2]
        q11 = x[:, :, :, y1g2, x1g2]

        # interpolate horizontally (along x) first
        wx_tensor = wx  # (1, new_W)
        wx_full = wx_tensor.expand(new_H, new_W)  # (new_H, new_W)
        q_top = self.slerp_vectorized(q00, q01, wx_full)
        q_bottom = self.slerp_vectorized(q10, q11, wx_full)

        # then interpolate vertically using wy
        wy_full = wy.expand(new_H, new_W)
        q_final = self.slerp_vectorized(q_top, q_bottom, wy_full)

        return q_final.reshape(B, C, new_H, new_W)


class DirectSlerpGridUpsample(nn.Module):
    """
    Direct SLERP Grid Upsampling: Full geodesic interpolation across input grid.
    Uses kernel-based weighted average in quaternion log space for orientation-preserving upsampling.
    """

    def __init__(self, scale_factor, kernel_size=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        B, C, H, W = x.shape
        n_quats = C // 4
        x = x.view(B, n_quats, 4, H, W)  # (B, n, 4, H, W)
        
        new_H = H * self.scale_factor
        new_W = W * self.scale_factor
        
        # Normalize input quaternions
        x = F.normalize(x, dim=2, eps=1e-8)
        
        # Compute output positions in input coordinate space
        out_y = torch.arange(new_H, device=x.device, dtype=torch.float32) / self.scale_factor
        out_x = torch.arange(new_W, device=x.device, dtype=torch.float32) / self.scale_factor
        
        # Create output tensor
        output = torch.zeros(B, n_quats, 4, new_H, new_W, device=x.device, dtype=x.dtype)
        
        # For each output position, compute kernel-based geodesic interpolation
        for i in range(new_H):
            for j in range(new_W):
                # Input position
                y_in = out_y[i]
                x_in = out_x[j]
                
                # Kernel bounds
                y_start = max(0, int(y_in) - self.padding)
                y_end = min(H, int(y_in) + self.padding + 1)
                x_start = max(0, int(x_in) - self.padding)
                x_end = min(W, int(x_in) + self.padding + 1)
                
                # Collect kernel quaternions
                kernel_quats = x[:, :, :, y_start:y_end, x_start:x_end]  # (B, n, 4, kh, kw)
                kh, kw = kernel_quats.shape[-2:]
                
                # Compute distances and weights
                y_coords = torch.arange(y_start, y_end, device=x.device, dtype=torch.float32)
                x_coords = torch.arange(x_start, x_end, device=x.device, dtype=torch.float32)
                y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                
                dist_y = (y_grid - y_in).abs()
                dist_x = (x_grid - x_in).abs()
                dist = torch.sqrt(dist_y**2 + dist_x**2)
                
                # Gaussian weights
                sigma = 1.0  # Adjustable
                weights = torch.exp(-dist**2 / (2 * sigma**2))
                weights = weights / (weights.sum() + 1e-8)
                
                # Expand weights to match kernel_quats
                weights = weights.view(1, 1, 1, kh, kw)
                
                # Compute weighted average in log space
                log_quats = quat_log(kernel_quats.view(B, n_quats, 4, -1).permute(0, 1, 3, 2))  # (B, n, num_pixels, 3)
                weights_flat = weights.view(1, 1, -1, 1)
                
                weighted_log = (log_quats * weights_flat).sum(dim=2)  # (B, n, 3)
                
                # Exponential map back to quaternions
                interp_quat = quat_exp(weighted_log)  # (B, n, 4)
                
                # Store in output
                output[:, :, :, i, j] = interp_quat
        
        return output.view(B, C, new_H, new_W)

    def slerp_vectorized(self, q0, q1, t, eps=1e-8):
        """
        Vectorized spherical linear interpolation between two quaternions, using symmetries.
        
        Args:
            q0: First quaternion (B, n, 4, H_out, W_out)
            q1: Second quaternion (B, n, 4, H_out, W_out)
            t: Interpolation parameter [0, 1] (H_out, W_out)
            
        Returns:
            Interpolated quaternion, FZ reduced (B, n, 4, H_out, W_out)
        """
        B, n, _, H, W = q0.shape
        G = self.group_tensor.shape[0]
        
        # Ensure unit quaternions
        q0 = F.normalize(q0, dim=2, eps=eps)
        q1 = F.normalize(q1, dim=2, eps=eps)
        
        B, n, _, H, W = q0.shape
        G = self.group_tensor.shape[0]
        
        # Apply all symmetries at once for vectorization
        gamma_q0_all = torch.einsum('gci, bnihw -> bgnchw', self.group_tensor, q0)  # (B, G, n, 4, H, W)
        gamma_q1_all = torch.einsum('gci, bnihw -> bgnchw', self.group_tensor, q1)  # (B, G, n, 4, H, W)
        
        # Compute dot product
        dot_all = (gamma_q0_all * gamma_q1_all).sum(dim=3)  # (B, G, n, H, W)
        
        # Handle double cover
        gamma_q1_all = torch.where(dot_all.unsqueeze(3) < 0, -gamma_q1_all, gamma_q1_all)
        dot_all = torch.where(dot_all < 0, -dot_all, dot_all)
        
        # Compute theta
        dot_clamped = torch.clamp(dot_all, -1.0 + 1e-7, 1.0 - 1e-7)
        theta_all = torch.acos(dot_clamped)  # (B, G, n, H, W)
        
        # Find min theta and corresponding index over symmetries
        min_theta, min_theta_idx = theta_all.min(dim=1)  # (B, n, H, W), (B, n, H, W)
        
        # Gather best quaternions using advanced indexing
        idx_b = torch.arange(B, device=q0.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        idx_n = torch.arange(n, device=q0.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, n, 1, 1)
        idx_h = torch.arange(H, device=q0.device).unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # (1, 1, H, 1)
        idx_w = torch.arange(W, device=q0.device).unsqueeze(0).unsqueeze(1).unsqueeze(0)  # (1, 1, 1, W)
        
        q0_best = gamma_q0_all[idx_b, min_theta_idx, idx_n, :, idx_h, idx_w]  # (B, n, H, W, 4)
        q0_best = q0_best.permute(0, 1, 4, 2, 3)  # (B, n, 4, H, W)
        q1_best = gamma_q1_all[idx_b, min_theta_idx, idx_n, :, idx_h, idx_w]  # (B, n, H, W, 4)
        q1_best = q1_best.permute(0, 1, 4, 2, 3)  # (B, n, 4, H, W)
        
        # Now SLERP between q0_best and q1_best
        theta_best = min_theta.unsqueeze(2)  # (B, n, 1, H, W)
        sin_theta_best = torch.sin(theta_best)
        dot_best = torch.cos(theta_best)
        
        # Reshape t
        t = t.view(1, 1, 1, H, W)
        
        # SLERP
        w0 = torch.sin((1.0 - t) * theta_best) / (sin_theta_best + eps)
        w1 = torch.sin(t * theta_best) / (sin_theta_best + eps)
        
        # Linear fallback
        use_linear = (sin_theta_best.abs() < eps) | (theta_best.abs() < eps)
        w0 = torch.where(use_linear, 1.0 - t, w0)
        w1 = torch.where(use_linear, t, w1)
        
        result = w0 * q0_best + w1 * q1_best
        result = F.normalize(result, dim=2, eps=eps)
        
        # FZ reduce: apply inverse symmetry
        inv_g = self.group_tensor_inv[min_theta_idx]  # (B, n, H, W, 4, 4) wait, need to index properly
        # min_theta_idx is (B, n, H, W), group_tensor_inv is (G, 4, 4)
        # Need to gather inv_g for each position
        inv_g = self.group_tensor_inv[min_theta_idx]  # This won't work directly
        
        # Better: result is (B, n, 4, H, W), need to apply inv_g per position
        # Since inv_g depends on position, need to einsum with gathered inv
        # First, expand min_theta_idx to select inv matrices
        inv_selected = self.group_tensor_inv[min_theta_idx.view(-1)]  # (B*n*H*W, 4, 4)
        result_flat = result.permute(0,1,3,4,2).reshape(-1, 4)  # (B*n*H*W, 4)
        fz_result_flat = torch.einsum('bij,bj->bi', inv_selected, result_flat)
        fz_result = fz_result_flat.view(B, n, H, W, 4).permute(0,1,4,2,3)
        
        return fz_result

    def forward(self, x):
        B, C, H, W = x.shape
        n_quats = C // 4
        x = x.view(B, n_quats, 4, H, W)  # (B, n, 4, H, W)
        
        new_H = H * self.scale_factor
        new_W = W * self.scale_factor
        
        # Create coordinate grids for output positions
        out_y = torch.arange(new_H, device=x.device, dtype=torch.float32) / self.scale_factor
        out_x = torch.arange(new_W, device=x.device, dtype=torch.float32) / self.scale_factor
        
        y0 = torch.floor(out_y).long()
        y1 = torch.clamp(y0 + 1, max=H - 1)
        x0 = torch.floor(out_x).long()
        x1 = torch.clamp(x0 + 1, max=W - 1)
        
        wy = (out_y - y0.float()).view(new_H, 1)
        wx = (out_x - x0.float()).view(1, new_W)
        
        wy = wy.expand(new_H, new_W)
        wx = wx.expand(new_H, new_W)
        
        y0_grid, x0_grid = torch.meshgrid(y0, x0, indexing='ij')
        y0_grid, x1_grid = torch.meshgrid(y0, x1, indexing='ij')
        y1_grid, x0_grid_2 = torch.meshgrid(y1, x0, indexing='ij')
        y1_grid_2, x1_grid_2 = torch.meshgrid(y1, x1, indexing='ij')
        
        q00 = x[:, :, :, y0_grid, x0_grid]
        q01 = x[:, :, :, y0_grid, x1_grid]
        q10 = x[:, :, :, y1_grid, x0_grid_2]
        q11 = x[:, :, :, y1_grid_2, x1_grid_2]
        
        q_top = self.slerp_vectorized(q00, q01, wx)
        q_bottom = self.slerp_vectorized(q10, q11, wx)
        q_final = self.slerp_vectorized(q_top, q_bottom, wy)
        
        return q_final.reshape(B, C, new_H, new_W)

# ------------------------------------------------------------------------------ #
# Quaternion SR Net
# ------------------------------------------------------------------------------ #
class Reynolds_QRBSA_Different_Upsampler(nn.Module):
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

        sym_np_path = "/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group.npy"
        sym_inv_np_path = "/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group_inv.npy"

        gt = torch.tensor(np.load(sym_np_path), dtype=torch.float32)
        gti = torch.tensor(np.load(sym_inv_np_path), dtype=torch.float32)
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

        # self.up = EquivariantReynoldsWrap(
        #     QuaternionTransposeConv(
        #         in_q_channels=mid_ch,
        #         out_q_channels=mid_ch,
        #         scale_factor=scale_factor,
        #         overlap=overlap,
        #         groups=g_mid,
        #     ),
        #     self.group_tensor,
        #     self.group_tensor_inv,
        # )

        # NEW
        # self.up_interp = EquivariantReynoldsWrap(
        #     nn.Upsample(
        #         scale_factor=scale_factor,
        #         mode='bilinear',
        #         align_corners=False
        #     ),
        #     self.group_tensor,
        #     self.group_tensor_inv,
        # )

        # self.normalize = EquivariantReynoldsWrap(
        #     QuaternionNormalize(eps=1e-8),
        #     self.group_tensor,
        #     self.group_tensor_inv,
        # )

        self.up_interp = DirectSlerpGridUpsample(scale_factor=scale_factor)
            
        self.up_conv = EquivariantReynoldsWrap(
            QuaternionConv(
                in_q_channels=mid_ch,
                out_q_channels=mid_ch,
                kernel_size=3,
                padding=1,
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
        scale_factor = getattr(self.cfg, "scale", 4)
        expected_H_out = H_in * scale_factor
        expected_W_out = W_in * scale_factor
        
        # x = self.act(self.enc1(x))
        x = self.enc1(x)
        # x = self.act(self.enc2(x))
        x = self.enc2(x)
        x = self.body(x) + x

        # Normalize to unit quaternions before SLERP upsampling
        x = F.normalize(x, dim=1, eps=1e-8)  # (B, C, H, W)
        #x = self.normalize(x)
        # OLD
        # x = self.up(x) 
        # NEW
        x = self.up_interp(x) # (B, C, H*4, W*4)
        x = self.up_conv(x)   # (B, C, H*4, W*4)
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
    #model = Equivariant_QRBSA(cfg=args)
    model = Reynolds_QRBSA_Different_Upsampler(cfg=args)

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
