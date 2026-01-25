import math
import warnings

# from Archive.model.quat_utils.Qops_with_QSN import Residual_SA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from numpy import dtype

inv_sqrt_2 = 1.0 / math.sqrt(2.0)
half = 0.5
eps= 1e-8

# Your original 24 cubic symmetry quaternions
fcc_syms = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [inv_sqrt_2, inv_sqrt_2, 0, 0],
        [inv_sqrt_2, 0, inv_sqrt_2, 0],
        [inv_sqrt_2, 0, 0, inv_sqrt_2],
        [inv_sqrt_2, -inv_sqrt_2, 0, 0],
        [inv_sqrt_2, 0, -inv_sqrt_2, 0],
        [inv_sqrt_2, 0, 0, -inv_sqrt_2],
        [0, inv_sqrt_2, inv_sqrt_2, 0],
        [0, inv_sqrt_2, 0, inv_sqrt_2],
        [0, 0, inv_sqrt_2, inv_sqrt_2],
        [0, inv_sqrt_2, -inv_sqrt_2, 0],
        [0, 0, inv_sqrt_2, -inv_sqrt_2],
        [0, inv_sqrt_2, 0, -inv_sqrt_2],
        [half, half, half, half],
        [half, -half, -half, half],
        [half, -half, half, -half],
        [half, half, -half, -half],
        [half, half, half, -half],
        [half, half, -half, half],
        [half, -half, half, half],
        [half, -half, -half, -half],
        ])


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
        out = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return out

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

        self.project_in = QuaternionConv(dim, hidden * 2, kernel_size=1, stride=1, padding=0)

        self.dwconv = QuaternionConv(hidden * 2, hidden * 2, kernel_size=3, stride=1)

        self.project_out = QuaternionConv(hidden * 2, dim, kernel_size=1, stride=1, padding=0)

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

        self.norm1 = QuatLayerNorm(
                dim,
            )
        self.attn = QAttention(dim, num_heads)
        self.norm2 = QuatLayerNorm(
                dim,)
        
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
        self.conv1 =QuaternionConv(
                in_q_channels=in_channels,
                out_q_channels=in_channels,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                groups=g_in,
                bias=True,
            )

        self.relu = q_relu

        # Second quaternion conv
        self.conv2 = QuaternionConv(
                in_q_channels=in_channels,
                out_q_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                groups=g_out,
                bias=True,
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
        out = fx.mean(dim=1).reshape(B, Cout, *spatial_out)
        # Normalize each quaternion (group of 4 channels)
        # b, c = out.shape[:2]
        # spatial_dims = out.shape[2:]
        # out = out.view(b, c // 4, 4, *spatial_dims)
        # norm = out.norm(dim=2, keepdim=True) + 1e-8
        # out = out / norm
        # out = out.view(b, c, *spatial_dims)
        return out

# -------------------------------------------------------------------------
# JIT Compiled Math Helpers (Fuse operations for speed)
# -------------------------------------------------------------------------

@torch.jit.script
def hamilton_prod(q1, q2):
    """
    Computes q1 * q2 efficiently.
    Supports broadcasting if shapes align (e.g., q1(N,1,4) * q2(1,M,4))
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)

@torch.jit.script
def inverse2(q):
    """Conjugate/Inverse for unit quaternions."""
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)

@torch.jit.script
def quat_log(q, eps: float = 1e-8):
    q = F.normalize(q, dim=-1)
    w = q[..., 0]
    v = q[..., 1:]
    
    # Clamp w to avoid nan in acos
    theta_half = torch.acos(torch.clamp(w, -1.0 + 1e-7, 1.0 - 1e-7))
    sin_theta_half = torch.sin(theta_half)
    
    small_mask = sin_theta_half.abs() < eps
    
    # Avoid division by zero
    scale = theta_half.unsqueeze(-1) / (sin_theta_half.unsqueeze(-1) + eps)
    out = v * scale
    
    # Handle small angles (limit approaches 0)
    # We can't use in-place ops in JIT easily with masking sometimes, 
    # but strict masking is fine here.
    out = torch.where(small_mask.unsqueeze(-1), torch.zeros_like(out), out)
    return out

@torch.jit.script
def quat_exp(v, eps: float = 1e-8):
    theta_half = torch.norm(v, dim=-1, keepdim=True)
    small_mask = theta_half.abs() < eps

    w = torch.cos(theta_half)
    # Avoid division by zero
    scale = torch.sin(theta_half) / (theta_half + eps)
    xyz = v * scale

    q = torch.cat([w, xyz], dim=-1)
    
    # Identity quaternion for small angles
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype)
    q = torch.where(small_mask, identity, q)

    return F.normalize(q, dim=-1)

# -------------------------------------------------------------------------
# Main Module
# -------------------------------------------------------------------------

class SlerpUpsample(nn.Module):
    def __init__(self, scale_factor, group_quats):
        """
        scale_factor: int
        group_quats: Tensor (G, 4) of symmetry quaternions
        """
        super().__init__()
        self.scale_factor = int(scale_factor)
        
        # Register symmetries as a buffer (persistent state, not a parameter)
        if not torch.is_tensor(group_quats):
            group_quats = torch.tensor(group_quats)
        self.register_buffer("group_quats", group_quats.float())
    
    def slerp(self, q1, q2, t, syms):
        """
        Batched Symmetry-aware SLERP.
        q1, q2: (N_total, 4)
        t: (Scale,)
        syms: (G, 4)
        """
        # 1. Correct relative rotation in LOCAL frame: A = q2 * q1^-1
        # This represents the rotation to get from q1 to q2
        A = hamilton_prod(q2, inverse2(q1))

        # 2. Apply symmetries: A_syms = S * A (LEFT ACTION)
        # FIX IS HERE: Swapped inputs to hamilton_prod.
        # syms (1, G, 4) * A (N, 1, 4) -> (N, G, 4)
        A_syms = hamilton_prod(syms.unsqueeze(0), A.unsqueeze(1))

        # 3. Ensure positive w (shortest path on 4D sphere)
        # (N, G, 1) mask
        mask = A_syms[..., 0:1] < 0
        A_syms = torch.where(mask, -A_syms, A_syms)

        # 4. Find symmetry with largest real part (smallest angle)
        # Argmax over the G dimension (dim=1)
        best_w, best_indices = torch.max(A_syms[..., 0], dim=1) # (N,)
        
        # Gather the best quaternion
        idx = best_indices.view(-1, 1, 1).expand(-1, 1, 4)
        q_rel = torch.gather(A_syms, 1, idx).squeeze(1) # (N, 4)

        # 5. Interpolate
        v = quat_log(q_rel) # (N, 3)
        
        # Broadcast t
        K = t.shape[0]
        v_scaled = v.unsqueeze(1) * t.view(1, K, 1) # (N, K, 3)
        
        # Flatten to apply exp
        v_flat = v_scaled.reshape(-1, 3)
        q_interp = quat_exp(v_flat).reshape(q1.shape[0], K, 4)

        # 6. Reconstruct: q(t) = q_interp * q1
        # Apply the interpolated relative rotation to q1
        q3 = hamilton_prod(q_interp, q1.unsqueeze(1))

        return q3
    
    def forward(self, x):
        """
        x: (B, C, H, W) where C = 4*N
        """
        B, C, H, W = x.shape
        assert C % 4 == 0, "Channels must be divisible by 4"
        
        S = self.scale_factor
        if S == 1:
            return x

        # 1. MASSIVE FLATTENING
        # View as (Batch*NumQuats, 4, H, W) then permute to (TotalInstances, H, W, 4)
        # This removes all Python loops over B and N.
        x_reshaped = x.view(B, C // 4, 4, H, W).permute(0, 1, 3, 4, 2).reshape(-1, H, W, 4).contiguous()
        
        # x_reshaped is now (M, H, W, 4) where M = B * (C/4)
        
        # Pre-calculate interpolation fractions t
        # We exclude the endpoint 1.0 because it is the start of the next pixel
        t = torch.linspace(0, 1 - 1/S, steps=S, device=x.device, dtype=x.dtype)

        # -------------------------------------------------------
        # Pass 1: Horizontal (Upsample Width)
        # Input: (M, H, W, 4) -> Output: (M, H, W*S, 4)
        # -------------------------------------------------------
        if W > 1:
            # Prepare pairs for all rows in all images at once
            # q_left: (M, H, W-1, 4) -> Flatten to (TotalSegments, 4)
            q_left  = x_reshaped[..., :-1, :].reshape(-1, 4)
            q_right = x_reshaped[..., 1:, :].reshape(-1, 4)
            
            # Run SLERP on millions of pixels simultaneously
            # Output: (TotalSegments, S, 4)
            q_h_interp = self.slerp(q_left, q_right, t, self.group_quats)
            
            # Reshape and Organize
            # (M, H, W-1, S, 4) -> (M, H, (W-1)*S, 4)
            q_h_interp = q_h_interp.view(x_reshaped.shape[0], H, (W-1)*S, 4)
            
            # Handle Boundary (Last Column)
            # Replicate last column S times
            q_last_col = x_reshaped[..., -1:, :].expand(-1, -1, S, 4).reshape(x_reshaped.shape[0], H, S, 4)
            
            # Concat
            xs = torch.cat([q_h_interp, q_last_col], dim=2) # (M, H, W*S, 4)
        else:
            # Width is 1, just replicate S times
            xs = x_reshaped.repeat_interleave(S, dim=2)

        # -------------------------------------------------------
        # Pass 2: Vertical (Upsample Height)
        # Input: (M, H, W*S, 4) -> Output: (M, H*S, W*S, 4)
        # -------------------------------------------------------
        M, _, W_new, _ = xs.shape
        
        if H > 1:
            # Transpose to treat Height as the dimension to interpolate
            # (M, H, W_new, 4) -> (M, W_new, H, 4)
            xs_t = xs.transpose(1, 2)
            
            # q_top: (M, W_new, H-1, 4) -> Flatten
            q_top = xs_t[..., :-1, :].reshape(-1, 4)
            q_bot = xs_t[..., 1:, :].reshape(-1, 4)
            
            # SLERP
            q_v_interp = self.slerp(q_top, q_bot, t, self.group_quats)
            
            # Reshape: (M, W_new, (H-1)*S, 4)
            q_v_interp = q_v_interp.view(M, W_new, (H-1)*S, 4)
            
            # Handle Boundary (Last Row)
            q_last_row = xs_t[..., -1:, :].expand(-1, -1, S, 4).reshape(M, W_new, S, 4)
            
            # Concat
            xs_final_t = torch.cat([q_v_interp, q_last_row], dim=2) # (M, W_new, H*S, 4)
            
            # Transpose back: (M, H*S, W_new, 4)
            xs_final = xs_final_t.transpose(1, 2)
        else:
            xs_final = xs.repeat_interleave(S, dim=1)

        # -------------------------------------------------------
        # Restore Dimensions
        # -------------------------------------------------------
        # (M, H*S, W*S, 4) -> (B, N, H*S, W*S, 4) -> (B, N, 4, H*S, W*S) -> (B, C, H*S, W*S)
        output = xs_final.view(B, C // 4, H*S, W*S, 4).permute(0, 1, 4, 2, 3).reshape(B, C, H*S, W*S)
        
        return output

# class MultiHypothesisSlerp(nn.Module):
#     def __init__(self, in_channels, num_hypotheses, scale_factor, group_quats):
#         super().__init__()
#         self.K = num_hypotheses
#         self.scale = scale_factor
        
#         total_out = (4 * self.K) + self.K
        
#         # FIX 1: Add padding=0
#         self.splitter = QuaternionConv(
#             in_channels, 
#             total_out, 
#             kernel_size=1, 
#             padding=0,   # <--- CRITICAL FIX
#             groups=1     
#         )
        
#         self.geom_upsampler = SlerpUpsample(scale_factor, group_quats)
#         self.weight_upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
#         # FIX 2: Add padding=0
#         self.fusion = QuaternionConv(
#             4 * self.K, 
#             in_channels, 
#             kernel_size=1,
#             padding=0,   # <--- CRITICAL FIX
#             groups=1
#         )
#     def forward(self, x):
#         # x: (B, C, H, W) generic latent features
        
#         # --- A. Decompose ---
#         # (B, 5K, H, W)
#         components = self.splitter(x)
        
#         # Split into Quats and Weights
#         # quats_lr: (B, 4K, H, W)
#         # weights_lr: (B, K, H, W)
#         quats_lr, weights_lr = torch.split(components, [4 * self.K, self.K], dim=1)
        
#         # --- B. Enforce Physics on Quaternions ---
#         # We MUST normalize here so SLERP works. 
#         # But unlike your previous attempt, we have separate 'weights' to track magnitude,
#         # so it's okay to destroy magnitude information in 'quats_lr'.
#         B, _, H, W = quats_lr.shape
#         quats_lr = quats_lr.reshape(B, self.K, 4, H, W)
#         quats_lr = F.normalize(quats_lr, dim=2) # Normalize along the quaternion axis
#         quats_lr = quats_lr.reshape(B, 4 * self.K, H, W)
        
#         # --- C. Upsample ---
        
#         # Path 1: Physics Upsample (The "Latent Slerp")
#         # Apply Symmetry-Aware SLERP to these K latent hypotheses
#         # Output: (B, 4K, H*S, W*S)
#         quats_hr = self.geom_upsampler(quats_lr)
        
#         # Path 2: Standard Upsample (The "Selection Mask")
#         # Output: (B, K, H*S, W*S)
#         weights_hr = self.weight_upsampler(weights_lr)
#         weights_hr = torch.sigmoid(weights_hr) # Force weights to 0-1 range
        
#         # --- D. Weighted Fusion ---
#         # We blend the Slerped quaternions based on the upsampled weights.
#         # This allows the network to "switch" between latent hypotheses smoothly.
        
#         # Reshape for broadcasting
#         # Q: (B, K, 4, H_out, W_out)
#         # W: (B, K, 1, H_out, W_out)
#         quats_hr = quats_hr.view(B, self.K, 4, H * self.scale, W * self.scale)
#         weights_hr = weights_hr.view(B, self.K, 1, H * self.scale, W * self.scale)
        
#         # Weighted combination
#         # (B, K, 4, ...) * (B, K, 1, ...) -> sum over K -> (B, 4, ...)
#         # BUT: We want to keep latent dimension for the next layer, 
#         # so let's just modulate them and concatenate.
        
#         modulated_quats = quats_hr * weights_hr # (B, K, 4, H_out, W_out)
#         modulated_quats = modulated_quats.view(B, 4 * self.K, H * self.scale, W * self.scale)
        
#         # --- E. Project back to Feature Space ---
#         out = self.fusion(modulated_quats)
        
#         return out

class MultiHypothesisSlerp(nn.Module):
    def __init__(self, in_channels, num_hypotheses, scale_factor, group_quats):
        """
        Symmetry-Aware Geometric Upsampler.
        
        Args:
            in_channels: Number of input channels (e.g., n_feats).
            num_hypotheses (K): Number of latent crystal orientations to predict per pixel.
                                Higher K = more flexibility at boundaries.
            scale_factor: Upsampling factor (e.g., 4).
            group_quats: Tensor of symmetry group quaternions (24, 4).
        """
        super().__init__()
        self.K = num_hypotheses
        self.scale = scale_factor
        
        # Output channels for splitter: 
        # K quaternions (4 * K) + K confidence weights (K)
        total_out = (4 * self.K) + self.K
        
        # 1. The Splitter (Context Aware)
        # Uses kernel_size=3 so the decision to pick a specific symmetry 
        # is informed by neighboring pixels.
        self.splitter = QuaternionConv(
            in_channels, 
            total_out, 
            kernel_size=3, 
            padding=1,     # Preserves H, W
            groups=1       # Full mixing (not depthwise)
        )
        
        # 2. Geometric Upsampler (Physics)
        self.geom_upsampler = SlerpUpsample(scale_factor, group_quats)
        
        # 3. Weight Upsampler (Selection)
        self.weight_upsampler = nn.Upsample(
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 4. Fusion (Collapse hypotheses)
        # Maps the weighted combination back to feature space.
        self.fusion = QuaternionConv(
            4 * self.K, 
            in_channels, 
            kernel_size=1,
            padding=0,     # 1x1 conv needs pad 0 to preserve dims
            groups=1
        )
        
        # 5. Smoother (Artifact Removal)
        # A raw QuaternionConv (NO Reynolds Wrapper) to smooth spatial noise
        # without re-averaging symmetries.
        self.smoother = QuaternionConv(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels // 4 # Grouped Q-Conv for efficiency/structure
        )

    def forward(self, x):
        # x: (B, C, H, W)
        
        # --- A. Decompose into Hypotheses ---
        # (B, 5K, H, W)
        components = self.splitter(x)
        
        # Split into Quaternions and Weights
        # quats_lr: (B, 4K, H, W)
        # weights_lr: (B, K, H, W)
        quats_lr, weights_lr = torch.split(components, [4 * self.K, self.K], dim=1)
        
        # --- B. Enforce Physics (Normalization) ---
        # We must normalize each group of 4 channels independently so they are
        # valid rotations for SLERP.
        B, _, H, W = quats_lr.shape
        quats_lr = quats_lr.reshape(B, self.K, 4, H, W)
        quats_lr = F.normalize(quats_lr, dim=2, eps=1e-8)
        quats_lr = quats_lr.reshape(B, 4 * self.K, H, W)
        
        # --- C. Upsample ---
        
        # Path 1: Physics Upsample (The "Latent Slerp")
        # Apply Symmetry-Aware SLERP to these K latent hypotheses
        # Output: (B, 4K, H*S, W*S)
        quats_hr = self.geom_upsampler(quats_lr)
        
        # Path 2: Weights Upsample (The "Selection Mask")
        # Output: (B, K, H*S, W*S)
        weights_hr = self.weight_upsampler(weights_lr)
        
        # --- D. Competition (Softmax) ---
        # Use Softmax with Temperature to force a decision.
        # Temperature 0.2 makes it "sharp" (approaching argmax), reducing ghosting.
        # Temperature 1.0 makes it "soft" (blending).
        weights_hr = F.softmax(weights_hr / 0.2, dim=1) 
        
        # --- E. Weighted Fusion ---
        # Reshape for broadcasting
        # Q: (B, K, 4, H_out, W_out)
        # W: (B, K, 1, H_out, W_out)
        quats_hr = quats_hr.view(B, self.K, 4, H * self.scale, W * self.scale)
        weights_hr = weights_hr.view(B, self.K, 1, H * self.scale, W * self.scale)
        
        # Modulate: "Turn off" the hypotheses that represent the wrong symmetry
        modulated_quats = quats_hr * weights_hr # (B, K, 4, H_out, W_out)
        
        # Flatten back to channels
        modulated_quats = modulated_quats.view(B, 4 * self.K, H * self.scale, W * self.scale)
        
        # Fuse back to feature dimension
        out = self.fusion(modulated_quats)
        
        # --- F. Smooth ---
        # Clean up any remaining stitching artifacts
        out = self.smoother(out)
        
        return out

# ------------------------------------------------------------------------------ #
# Quaternion SR Net
# ------------------------------------------------------------------------------ #
class Network_Slerp(nn.Module):
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

        self.enc1 = QuaternionConv(in_ch, mid_ch, k, padding=k // 2, groups=g_in)

        self.enc2 = QuaternionConv(mid_ch, mid_ch, k, padding=k // 2, groups=g_mid)

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

        self.outc = QuaternionConv(mid_ch, in_ch, k, padding=k // 2, groups=g_in)

        self.latent_slerp = MultiHypothesisSlerp(
            in_channels=mid_ch, 
            num_hypotheses=4, # Try 4 or 8 latent orientations
            scale_factor=scale_factor, 
            group_quats=fcc_syms
        )

    def forward(self, x):
        # Store input dimensions for cropping
        B, C, H_in, W_in = x.shape
        scale_factor = getattr(self.cfg, "scale", 4)
        
        x = self.enc1(x)
        x = self.enc2(x)
        x = (self.body(x) + x) / 2
        x = self.latent_slerp(x) # Performs split -> norm -> slerp -> fuse
        x = self.outc(x)
        return x

# ------------------------------------------------------------------------------ #
# Example usage
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":

    class _Args:
        sym_np_path = "/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group.npy"
        sym_inv_np_path = "/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group_inv.npy"
        dropout = 0.0
        n_feats = 128  # Increase to match previous large model
        n_resblocks = 16  # Increase to match previous large model
        scale = 4
        kernel_size = 3
        overlap = False

    args = _Args()
    B = 2
    H = W = 32
    scale = args.scale

    print("\nNon-overlapping SR")
    model = Reynolds_QRBSA_Different_Upsampler(cfg=args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    q_lr = torch.randn(B, 4, H, W)
    q_lr = q_lr / q_lr.norm(dim=1, keepdim=True).clamp_min(1e-8)
    q_sr = model(q_lr)

    print("Output shape (clean):", q_sr.shape)  # expected (B, 4, 128, 128)
