import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# -------------------------------------------------------------------------
# Global Symmetry Definition (Hardcoded for Safety)
# -------------------------------------------------------------------------
inv_sqrt_2 = 1.0 / math.sqrt(2.0)
half = 0.5

# Standard Cubic (m3m) Symmetry Group Quaternions (24, 4)
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
    ], dtype=torch.float32)

# -------------------------------------------------------------------------
# Math Helpers
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# JIT Compiled Math Helpers
# -------------------------------------------------------------------------

@torch.jit.script
def hamilton_prod(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

@torch.jit.script
def hamilton_product_weights(r, i, j, k, q_sym):
    """
    Rotates the quaternion weights by the symmetry quaternion.
    q_sym: (4,) tensor [w, x, y, z]
    """
    # Robustly split the (4,) vector into 4 scalar tensors (0-d)
    w2, x2, y2, z2 = q_sym.unbind(-1)
    
    # Scalar * Tensor broadcasting is safe
    r_new = r * w2 - i * x2 - j * y2 - k * z2
    i_new = r * x2 + i * w2 + j * z2 - k * y2
    j_new = r * y2 - i * z2 + j * w2 + k * x2
    k_new = r * z2 + i * y2 - j * x2 + k * w2
    
    return r_new, i_new, j_new, k_new

@torch.jit.script
def inverse2(q):
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)

@torch.jit.script
def quat_log(q, eps: float = 1e-8):
    q = F.normalize(q, dim=-1)
    w = q[..., 0]
    v = q[..., 1:]
    theta_half = torch.acos(torch.clamp(w, -1.0 + 1e-7, 1.0 - 1e-7))
    sin_theta_half = torch.sin(theta_half)
    small_mask = sin_theta_half.abs() < eps
    scale = theta_half.unsqueeze(-1) / (sin_theta_half.unsqueeze(-1) + eps)
    out = v * scale
    out = torch.where(small_mask.unsqueeze(-1), torch.zeros_like(out), out)
    return out

@torch.jit.script
def quat_exp(v, eps: float = 1e-8):
    theta_half = torch.norm(v, dim=-1, keepdim=True)
    small_mask = theta_half.abs() < eps
    w = torch.cos(theta_half)
    scale = torch.sin(theta_half) / (theta_half + eps)
    xyz = v * scale
    q = torch.cat([w, xyz], dim=-1)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype)
    q = torch.where(small_mask, identity, q)
    return F.normalize(q, dim=-1)

def generate_group_filters(conv_layer, group_quats):
    """
    Takes a QuaternionConv layer and returns a stack of 24 filters.
    """
    # Force input to be (N, 4).
    # Since we use fcc_syms (24, 4), this will stay (24, 4).
    # If the user passed matrices (24, 4, 4), this check prevents silent flattening.
    if group_quats.dim() == 2:
        pass
    elif group_quats.dim() == 3 and group_quats.shape[0] == 1:
        group_quats = group_quats.squeeze(0)
    else:
        # If we got here, something is wrong with dimensions.
        # Flattening blindly caused the 9600 channel bug.
        # We try to trust reshape only if it makes sense (divisible by 4)
        group_quats = group_quats.reshape(-1, 4)
    
    r, i, j, k = conv_layer.r, conv_layer.i, conv_layer.j, conv_layer.k
    weights_list = []
    
    for g in range(group_quats.shape[0]):
        q_sym = group_quats[g] # (4,)
        rn, in_, jn, kn = hamilton_product_weights(r, i, j, k, q_sym)
        w_sym = quaternion_block_weight(rn, in_, jn, kn) 
        weights_list.append(w_sym)
        
    return torch.cat(weights_list, dim=0)

 
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
# Layers
# ------------------------------------------------------------------------------ #

class QuaternionConv(nn.Module):
    def __init__(self, in_q_channels, out_q_channels, kernel_size, stride=1, padding=1, dilation=1, groups=None, bias=True):
        super().__init__()
        assert in_q_channels % 4 == 0 and out_q_channels % 4 == 0
        self.in_q = in_q_channels // 4
        self.out_q = out_q_channels // 4
        self.groups = 1 if groups is None else groups
        
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
        out = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

def q_relu(x, eps=1e-8):
    B, C, H, W = x.shape
    q = x.view(B, C // 4, 4, H, W)
    norm = torch.sqrt((q**2).sum(dim=2, keepdim=True) + eps)
    gate = F.relu(norm)
    q = q * gate / (norm + eps)
    return q.view(B, C, H, W)

class QuaternionReLU(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return q_relu(x, eps=self.eps)

class QAttention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
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
        out = rearrange(out, "b h c (h1 w1) -> b (h c) h1 w1", h=self.num_heads, h1=H, w1=W)
        return self.proj(out)

class Attentionmultiply(nn.Module):
    def __init__(self, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        # temperature per head
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, q, k, v):
        # q,k,v expected shape: (b, h, c, seq_len)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out

class PackedQKVAttention(nn.Module):
    """Attention module that accepts packed qkv input for EquivariantReynoldsWrap compatibility.
    
    This module is designed to be wrapped with EquivariantReynoldsWrap. It takes a single
    input tensor containing packed q,k,v (concatenated along channel dim), performs attention,
    and returns a single output tensor.
    """
    def __init__(self, dim, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        # temperature per head
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, qkv_packed):
        # qkv_packed shape: (B*G, 3*dim, H, W) where dim is the feature dimension
        B_times_G, C_packed, H, W = qkv_packed.shape
        assert C_packed == 3 * self.dim, f"Expected {3*self.dim} channels, got {C_packed}"
        
        # Split into q, k, v
        q, k, v = qkv_packed.chunk(3, dim=1)  # each is (B*G, dim, H, W)
        
        # Rearrange for multi-head attention
        # Note: self.dim must be divisible by num_heads
        q = rearrange(q, "b (h c) h1 w1 -> b h c (h1 w1)", h=self.num_heads)
        k = rearrange(k, "b (h c) h1 w1 -> b h c (h1 w1)", h=self.num_heads)
        v = rearrange(v, "b (h c) h1 w1 -> b h c (h1 w1)", h=self.num_heads)
        
        # Attention computation
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        
        # Rearrange back to spatial format
        out = rearrange(out, "b h c (h1 w1) -> b (h c) h1 w1", h=self.num_heads, h1=H, w1=W)
        return out

class Reynolds_QAttention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        gt=torch.tensor(np.load("/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group.npy"), dtype=torch.float32)
        gt_inv=torch.tensor(np.load("/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group_inv.npy"), dtype=torch.float32)
        
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = EquivariantReynoldsWrap(
            QuaternionConv(dim, dim * 3, kernel_size=1, stride=1, padding=0),
            group_tensor=gt,
            group_tensor_inv=gt_inv
        )
        
        self.qkv_dw = EquivariantReynoldsWrap(
            QuaternionConv(dim * 3, dim * 3, kernel_size=3, stride=1),
            group_tensor=gt,
            group_tensor_inv=gt_inv
        )
        
        self.proj = EquivariantReynoldsWrap(
            QuaternionConv(dim, dim, kernel_size=1, stride=1, padding=0),
            group_tensor=gt,
            group_tensor_inv=gt_inv
        )

        # Wrap attention with EquivariantReynoldsWrap to ensure equivariance
        # under group actions. Attention mixes channel information via matrix
        # multiplication, which can break quaternion structure without wrapping.
        self.attn = EquivariantReynoldsWrap(
            PackedQKVAttention(dim=dim, num_heads=self.num_heads),
            group_tensor=gt,
            group_tensor_inv=gt_inv
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # qkv is already equivariant (produced by wrapped convs)
        qkv = self.qkv_dw(self.qkv(x))  # (B, 3*dim, H, W)
        
        # Apply wrapped attention to the packed qkv
        # The EquivariantReynoldsWrap will:
        # 1. Lift qkv by applying group actions
        # 2. Apply PackedQKVAttention on lifted data
        # 3. Project back and average over group
        out = self.attn(qkv)  # (B, dim, H, W)
        
        return self.proj(out)

class GroupAttention(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.groups = 24
        self.attn = QAttention(dim=channels, num_heads=num_heads)
        
    def forward(self, x):
        B, GC, H, W = x.shape
        C = GC // self.groups
        x_reshaped = x.view(B, self.groups, C, H, W).reshape(B * self.groups, C, H, W)
        out = self.attn(x_reshaped)
        out = out.view(B, self.groups, C, H, W).reshape(B, GC, H, W)
        return out

class SlerpUpsample(nn.Module):
    def __init__(self, scale_factor, group_quats):
        super().__init__()
        self.scale_factor = int(scale_factor)
        if not torch.is_tensor(group_quats):
            group_quats = torch.tensor(group_quats)
        self.register_buffer("group_quats", group_quats.float())
    
    def slerp(self, q1, q2, t, syms):
        A = hamilton_prod(q2, inverse2(q1))
        # Ensure correct broadcasting for syms
        if syms.dim() == 3: syms = syms.squeeze(0) 
        A_syms = hamilton_prod(syms.unsqueeze(0), A.unsqueeze(1))
        mask = A_syms[..., 0:1] < 0
        A_syms = torch.where(mask, -A_syms, A_syms)
        best_w, best_indices = torch.max(A_syms[..., 0], dim=1)
        idx = best_indices.view(-1, 1, 1).expand(-1, 1, 4)
        q_rel = torch.gather(A_syms, 1, idx).squeeze(1)
        v = quat_log(q_rel)
        K = t.shape[0]
        v_scaled = v.unsqueeze(1) * t.view(1, K, 1)
        v_flat = v_scaled.reshape(-1, 3)
        q_interp = quat_exp(v_flat).reshape(q1.shape[0], K, 4)
        q3 = hamilton_prod(q_interp, q1.unsqueeze(1))
        return q3
    
    def forward(self, x):
        B, C, H, W = x.shape
        S = self.scale_factor
        if S == 1: return x
        
        x_reshaped = x.view(B, C // 4, 4, H, W).permute(0, 1, 3, 4, 2).reshape(-1, H, W, 4).contiguous()
        t = torch.linspace(0, 1 - 1/S, steps=S, device=x.device, dtype=x.dtype)

        if W > 1:
            q_left  = x_reshaped[..., :-1, :].reshape(-1, 4)
            q_right = x_reshaped[..., 1:, :].reshape(-1, 4)
            q_h_interp = self.slerp(q_left, q_right, t, self.group_quats)
            q_h_interp = q_h_interp.view(x_reshaped.shape[0], H, (W-1)*S, 4)
            q_last_col = x_reshaped[..., -1:, :].expand(-1, -1, S, 4).reshape(x_reshaped.shape[0], H, S, 4)
            xs = torch.cat([q_h_interp, q_last_col], dim=2)
        else:
            xs = x_reshaped.repeat_interleave(S, dim=2)

        M, _, W_new, _ = xs.shape
        if H > 1:
            xs_t = xs.transpose(1, 2)
            q_top = xs_t[..., :-1, :].reshape(-1, 4)
            q_bot = xs_t[..., 1:, :].reshape(-1, 4)
            q_v_interp = self.slerp(q_top, q_bot, t, self.group_quats)
            q_v_interp = q_v_interp.view(M, W_new, (H-1)*S, 4)
            q_last_row = xs_t[..., -1:, :].expand(-1, -1, S, 4).reshape(M, W_new, S, 4)
            xs_final_t = torch.cat([q_v_interp, q_last_row], dim=2)
            xs_final = xs_final_t.transpose(1, 2)
        else:
            xs_final = xs.repeat_interleave(S, dim=1)

        output = xs_final.view(B, C // 4, H*S, W*S, 4).permute(0, 1, 4, 2, 3).reshape(B, C, H*S, W*S)
        return output

# class MultiHypothesisSlerp(nn.Module):
#     def __init__(self, in_channels, num_hypotheses, scale_factor, group_quats):
#         super().__init__()
#         self.K = num_hypotheses
#         self.scale = scale_factor
#         total_out = (4 * self.K) + self.K
        
#         self.splitter = QuaternionConv(in_channels, total_out, kernel_size=3, padding=1, groups=1)
#         self.geom_upsampler = SlerpUpsample(scale_factor, group_quats)
#         self.weight_upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
#         self.fusion = QuaternionConv(4 * self.K, in_channels, kernel_size=1, padding=0, groups=1)
#         self.smoother = QuaternionConv(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels // 4)

#     def forward(self, x):
#         # A. Split
#         components = self.splitter(x)
#         quats_lr, weights_lr = torch.split(components, [4 * self.K, self.K], dim=1)
        
#         # B. Normalize Quats (Physics)
#         B, _, H, W = quats_lr.shape
#         quats_lr = quats_lr.reshape(B, self.K, 4, H, W)
#         quats_lr = F.normalize(quats_lr, dim=2, eps=1e-8)
#         quats_lr = quats_lr.reshape(B, 4 * self.K, H, W)
        
#         # C. Upsample
#         quats_hr = self.geom_upsampler(quats_lr)     # Path 1: Geometry
#         weights_hr = self.weight_upsampler(weights_lr) # Path 2: Selection
        
#         # D. Softmax (The Fix: Remove the /0.2 divisor)
#         # Let the network learn the sharpness. 
#         # If you want it sharp at the end, anneal this value during training.
#         weights_hr = F.softmax(weights_hr, dim=1) 
        
#         # E. Fuse
#         quats_hr = quats_hr.view(B, self.K, 4, H * self.scale, W * self.scale)
#         weights_hr = weights_hr.view(B, self.K, 1, H * self.scale, W * self.scale)
        
#         # Weighted Linear Sum
#         modulated_quats = (quats_hr * weights_hr).sum(dim=1) # (B, 4, H, W)
        
#         # F. Renormalize (The Fix: Pull back to sphere surface)
#         # This prevents "muddy" colors at the blend points
#         modulated_quats = F.normalize(modulated_quats, dim=1, eps=1e-8)
        
#         # G. Final Polish
#         # Note: self.fusion input dim needs to change to 'in_channels' since we summed
#         # or you can keep your modulation logic if self.fusion expects 4*K channels.
#         # Assuming you want to project back to features:
        
#         # If your self.fusion expects 4*K, we need to expand again (wasteful).
#         # Better: Change self.fusion to accept 4 channels (the summed result).
        
#         # LET'S STICK TO YOUR LOGIC TO MINIMIZE CODE CHANGE:
#         # Re-expand for your current fusion layer
#         modulated_expanded = modulated_quats.unsqueeze(1).repeat(1, self.K, 1, 1, 1)
#         modulated_expanded = modulated_expanded.view(B, 4 * self.K, H * self.scale, W * self.scale)
        
#         out = self.fusion(modulated_expanded)
#         out = self.smoother(out)
        
#         return out

# class MultiHypothesisSlerp(nn.Module):
#     def __init__(self, in_channels, num_hypotheses, scale_factor, group_quats):
#         super().__init__()
#         self.K = num_hypotheses
#         self.scale = scale_factor
        
#         # -------------------------------------------------------
#         # 1. THE DEEP SPLITTER (High-Dimensional Reasoning)
#         # -------------------------------------------------------
#         # Instead of projecting immediately to 5*K, we keep the dimensions high 
#         # to allow the network to "think" about the decomposition.
        
#         # Layer A: Process Context (Preserve dimensionality)
#         # Input: k * mid_ch -> Output: k * mid_ch
#         # We use QuaternionConv to respect the input feature geometry.
#         self.splitter_body = nn.Sequential(
#             QuaternionConv(in_channels, in_channels, kernel_size=3, padding=1),
#             QuaternionReLU() 
#         )
        
#         # Layer B: The Bottleneck (Project to Physical Coordinates)
#         # Input: k * mid_ch -> Output: 5 * K (4 Quats + 1 Weight per K)
#         # We use a standard 1x1 Conv here because the output (5*K) 
#         # is not a pure quaternion structure (it contains scalar weights).
#         self.splitter_head = nn.Conv2d(in_channels, 5 * self.K, kernel_size=1)

#         # -------------------------------------------------------
#         # 2. UPSAMPLERS
#         # -------------------------------------------------------
#         self.geom_upsampler = SlerpUpsample(scale_factor, group_quats)
#         self.weight_upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

#     def forward(self, x):
#         # A. Deep Split
#         # 1. Reason in high dimensions
#         feat = self.splitter_body(x) 
#         # 2. Project to physical hypotheses
#         components = self.splitter_head(feat)
        
#         # Split into Geometry (4*K) and Probability (K)
#         quats_lr, weights_lr = torch.split(components, [4 * self.K, self.K], dim=1)
        
#         # --- STEP 3: Tanh Bounding (Stabilizer) ---
#         weights_lr = torch.tanh(weights_lr) 

#         # B. Normalize Quats (Physics Enforcement)
#         # Ensure predictions are on the hypersphere
#         B, _, H, W = quats_lr.shape
#         quats_lr = quats_lr.reshape(B, self.K, 4, H, W)
#         quats_lr = F.normalize(quats_lr, dim=2, eps=1e-8)
#         quats_lr = quats_lr.reshape(B, 4 * self.K, H, W)
        
#         # C. Upsample
#         quats_hr = self.geom_upsampler(quats_lr)       # SLERP (Physics)
#         weights_hr = self.weight_upsampler(weights_lr) # Bilinear (Selection)
        
#         # D. Softmax (The Decision Maker)
#         # Scale by 5.0 (Inverse Temp) to encourage sharp decisions
#         weights_hr = F.softmax(weights_hr * 5.0, dim=1) 
        
#         # E. Feature Space Modulation (Do NOT Sum)
#         # Prepare for broadcasting
#         # quats: (B, K, 4, H_up, W_up)
#         # weights: (B, K, 1, H_up, W_up)
#         quats_hr = quats_hr.view(B, self.K, 4, H * self.scale, W * self.scale)
#         weights_hr = weights_hr.view(B, self.K, 1, H * self.scale, W * self.scale)
        
#         # Modulate: Determine the strength of each hypothesis at each pixel
#         modulated_quats = quats_hr * weights_hr 
        
#         # F. Return Stack
#         # Flatten back to (B, 4*K, H_up, W_up).
#         # We hand off the separate hypotheses to the next layer for blending/smoothing.
#         out_stack = modulated_quats.view(B, 4 * self.K, H * self.scale, W * self.scale)
        
#         return out_stack, weights_hr  

class MultiHypothesisSlerp(nn.Module):
    def __init__(self, in_channels, num_hypotheses, scale_factor, group_quats):
        super().__init__()
        self.K = num_hypotheses
        self.scale = scale_factor
        
        # Splitter Logic
        self.splitter_body = nn.Sequential(
            QuaternionConv(in_channels, in_channels, kernel_size=3, padding=1),
            QuaternionReLU() 
        )
        self.splitter_head = nn.Conv2d(in_channels, 5 * self.K, kernel_size=1)
        nn.init.normal_(self.splitter_head.weight, std=0.01)

        self.geom_upsampler = SlerpUpsample(scale_factor, group_quats)
        self.weight_upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        # A. Split
        feat = self.splitter_body(x) 
        components = self.splitter_head(feat)
        quats_lr, weights_lr = torch.split(components, [4 * self.K, self.K], dim=1)
        
        # B. Normalize Physics
        B, _, H, W = quats_lr.shape
        quats_lr = quats_lr.reshape(B, self.K, 4, H, W)
        quats_lr = F.normalize(quats_lr, dim=2, eps=1e-8)
        quats_lr = quats_lr.reshape(B, 4 * self.K, H, W)
        
        # C. Upsample
        quats_hr = self.geom_upsampler(quats_lr)
        weights_hr = self.weight_upsampler(weights_lr)
        
        # D. Hard Selection (Gumbel) - PREVENTS BLUR
        if self.training:
            # Soft but sharp during training to allow gradients
            weights_hr = F.gumbel_softmax(weights_hr, tau=0.5, hard=False, dim=1)
        else:
            # Hard argmax during inference
            indices = weights_hr.argmax(dim=1, keepdim=True)
            weights_hr = torch.zeros_like(weights_hr).scatter_(1, indices, 1.0)
        
        # E. Collapse Stack to Single Map
        # (B, K, 4, H_up, W_up)
        quats_hr = quats_hr.view(B, self.K, 4, H * self.scale, W * self.scale)
        weights_hr = weights_hr.unsqueeze(2)
        
        # Weighted Sum (Selection)
        out_single_map = (quats_hr * weights_hr).sum(dim=1) # (B, 4, H, W)
        out_single_map = F.normalize(out_single_map, dim=1, eps=1e-8)
        
        # Return Single Map (4 channels) + Weights (for visualization)
        return out_single_map, weights_hr.squeeze(2)
        
class LiftingConv(nn.Module):
    def __init__(self, in_q, out_q, kernel_size, padding, group_quats):
        super().__init__()
        self.conv = QuaternionConv(in_q, out_q, kernel_size, padding=padding)
        self.register_buffer('group_quats', group_quats)
        self.out_q = out_q 
        
    def forward(self, x):
        W_stack = generate_group_filters(self.conv, self.group_quats)
        
        # FIX: Handle Bias being disconnected
        # self.conv.bias is (Out_Q). The stack produces (24 * Out_Q).
        # We must repeat the bias 24 times.
        if self.conv.bias is not None:
            b_stack = self.conv.bias.repeat(24)
        else:
            b_stack = None
            
        out = F.conv2d(x, W_stack, bias=b_stack, padding=self.conv.padding, stride=self.conv.stride)
        return out

class TopKGroupSelector(nn.Module):
    def __init__(self, in_channels, k=4, groups=24):
        super().__init__()
        self.k = k
        self.groups = groups
        # Scorer to rate "how good" each symmetry group is at this pixel
        self.scorer = nn.Sequential(
            nn.Conv2d(in_channels * groups, groups, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 24*C, H, W)
        B, GC, H, W = x.shape
        C = GC // self.groups
        
        # 1. Score the Groups
        # scores: (B, 24, H, W)
        scores = self.scorer(x)
        
        # 2. Find indices of the Top K groups
        # indices: (B, K, H, W)
        top_scores, indices = torch.topk(scores, k=self.k, dim=1)
        
        # 3. Gather the features corresponding to these Top K groups
        # We need to do some fancy indexing to grab the chunks
        x_view = x.view(B, self.groups, C, H, W)
        
        # Expand indices to match feature dims for gathering
        # indices_exp: (B, K, C, H, W)
        indices_exp = indices.unsqueeze(2).expand(-1, -1, C, -1, -1)
        
        # gathered_features: (B, K, C, H, W) -> (B, K*C, H, W)
        gathered_features = torch.gather(x_view, 1, indices_exp).view(B, self.k * C, H, W)
        
        return gathered_features

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=16, num_heads=4):
        super().__init__()
        self.window_size = window_size
        
        # We reuse your existing QAttention
        # It will operate on patches of size (window_size x window_size)
        self.attn = QAttention(dim, num_heads=num_heads)

    def window_partition(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            windows: (B * num_windows, C, window_size, window_size)
        """
        B, C, H, W = x.shape
        ws = self.window_size
        
        # Reshape to (B, C, h, ws, w, ws)
        # We assume H and W are divisible by ws (we pad in forward to ensure this)
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        
        # Permute to (B, h, w, C, ws, ws) -> Combine B,h,w -> (Batch*Windows, C, ws, ws)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, ws, ws)
        return windows

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (B * num_windows, C, window_size, window_size)
        Returns:
            x: (B, C, H, W)
        """
        ws = self.window_size
        # Calculate original Batch size from total windows
        # Total windows = (H // ws) * (W // ws) * B
        B = int(windows.shape[0] / (H * W / ws / ws))
        C = windows.shape[1]
        
        x = windows.view(B, H // ws, W // ws, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        
        # 1. Pad if H or W is not divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        Hp, Wp = x.shape[2], x.shape[3]

        # 2. Partition into windows
        # Shape becomes: (Batch * Num_Windows, C, 16, 16)
        x_windows = self.window_partition(x)

        # 3. Run Self-Attention on windows
        # The attention thinks it is working on a large batch of small 16x16 images.
        attn_windows = self.attn(x_windows)

        # 4. Merge windows back to image
        x = self.window_reverse(attn_windows, Hp, Wp)

        # 5. Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
            
        return x

class ReynoldsWindowAttention(nn.Module):
    def __init__(self, dim, window_size=16, num_heads=4):
        super().__init__()
        self.window_size = window_size
        
        # We reuse your existing QAttention
        # It will operate on patches of size (window_size x window_size)
        self.attn = Reynolds_QAttention(dim, num_heads=num_heads)

    def window_partition(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            windows: (B * num_windows, C, window_size, window_size)
        """
        B, C, H, W = x.shape
        ws = self.window_size
        
        # Reshape to (B, C, h, ws, w, ws)
        # We assume H and W are divisible by ws (we pad in forward to ensure this)
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        
        # Permute to (B, h, w, C, ws, ws) -> Combine B,h,w -> (Batch*Windows, C, ws, ws)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, ws, ws)
        return windows

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (B * num_windows, C, window_size, window_size)
        Returns:
            x: (B, C, H, W)
        """
        ws = self.window_size
        # Calculate original Batch size from total windows
        # Total windows = (H // ws) * (W // ws) * B
        B = int(windows.shape[0] / (H * W / ws / ws))
        C = windows.shape[1]
        
        x = windows.view(B, H // ws, W // ws, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        
        # 1. Pad if H or W is not divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        Hp, Wp = x.shape[2], x.shape[3]

        # 2. Partition into windows
        # Shape becomes: (Batch * Num_Windows, C, 16, 16)
        x_windows = self.window_partition(x)

        # 3. Run Self-Attention on windows
        # The attention thinks it is working on a large batch of small 16x16 images.
        attn_windows = self.attn(x_windows)

        # 4. Merge windows back to image
        x = self.window_reverse(attn_windows, Hp, Wp)

        # 5. Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
            
        return x

class RobustIterativeSmoother(nn.Module):
    def __init__(self, in_channels, group_quats, iterations=5, kernel_size=9, start_temp=1.0, end_temp=40.0):
        super().__init__()
        self.iterations = iterations
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.start_temp = start_temp
        self.end_temp = end_temp
        
        if not torch.is_tensor(group_quats):
            group_quats = torch.tensor(group_quats)
        self.register_buffer("syms", group_quats.float())

    def align_and_smooth(self, x, current_temp):
        B, C, H, W = x.shape
        N_q = C // 4
        
        unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        K_sq = self.kernel_size * self.kernel_size
        patches = unfold.view(B, N_q, 4, K_sq, H, W) 
        
        anchor = patches.mean(dim=3, keepdim=True) # (B, Nq, 4, 1, H, W)
        anchor = F.normalize(anchor, dim=2, eps=1e-8)
        
        # --- ROBUST DOT PRODUCT CALCULATION ---
        a0, a1, a2, a3 = anchor[:,:,0], anchor[:,:,1], anchor[:,:,2], anchor[:,:,3]
        n0, n1, n2, n3 = patches[:,:,0], patches[:,:,1], patches[:,:,2], patches[:,:,3]
        
        r0 = a0*n0 - a1*(-n1) - a2*(-n2) - a3*(-n3)
        r1 = a0*(-n1) + a1*n0 + a2*(-n3) - a3*(-n2)
        r2 = a0*(-n2) - a1*(-n3) + a2*n0 + a3*(-n1)
        r3 = a0*(-n3) + a1*(-n2) - a2*(-n1) + a3*n0
        rel = torch.stack([r0, r1, r2, r3], dim=2) # (B, Nq, 4, K_sq, H, W)
        
        # Manual Expansion to avoid einsum confusion
        # rel: (B, Nq, 4, K_sq, H, W) -> (..., 1)
        rel_exp = rel.unsqueeze(-1)
        
        # syms: (24, 4) -> (1, 1, 4, 1, 1, 1, 24)
        syms_exp = self.syms.t().view(1, 1, 4, 1, 1, 1, 24)
        
        # Dot product over quaternion dim (dim=2)
        dots = (rel_exp * syms_exp).sum(dim=2) # (B, Nq, K_sq, H, W, 24)
        
        max_dots, best_sym_idx = torch.max(torch.abs(dots), dim=-1) # (B, Nq, K_sq, H, W)
        
        # Gating
        weights = F.softmax(max_dots * current_temp, dim=2) 
        weights = weights.unsqueeze(2) 
        
        # Alignment
        indices_flat = best_sym_idx.flatten()
        selected_syms_flat = self.syms[indices_flat]
        selected_syms = selected_syms_flat.view(B, N_q, K_sq, H, W, 4)
        selected_syms = selected_syms.permute(0, 1, 5, 2, 3, 4)
        
        s0, s1, s2, s3 = selected_syms[:,:,0], selected_syms[:,:,1], selected_syms[:,:,2], selected_syms[:,:,3]
        
        new0 = s0*n0 - s1*n1 - s2*n2 - s3*n3
        new1 = s0*n1 + s1*n0 + s2*n3 - s3*n2
        new2 = s0*n2 - s1*n3 + s2*n0 + s3*n1
        new3 = s0*n3 + s1*n2 - s2*n1 + s3*n0
        aligned = torch.stack([new0, new1, new2, new3], dim=2)
        
        dot_check = (anchor * aligned).sum(dim=2, keepdim=True)
        aligned = aligned * torch.sign(dot_check)
        
        out = (aligned * weights).sum(dim=3)
        return F.normalize(out, dim=2, eps=1e-8).view(B, C, H, W)

    def forward(self, x):
        for i in range(self.iterations):
            progress = i / max(1, self.iterations - 1)
            curr_temp = self.start_temp + (self.end_temp - self.start_temp) * (progress ** 2)
            x = self.align_and_smooth(x, curr_temp)
        return x
    
class RefinementHead(nn.Module):
    def __init__(self, group_quats):
        super().__init__()
        
        # Pure Physics-Based Smoothing
        # No convolutions here prevents the network from learning to blur edges
        # to minimize MSE loss.
        self.final_smoother = RobustIterativeSmoother(
            in_channels=4, 
            group_quats=group_quats, 
            iterations=5,     
            kernel_size=9,    
            start_temp=1.0,
            end_temp=40.0
        )

    def forward(self, x):
        # Input x is (B, 4, H, W)
        out = self.final_smoother(x)
        return out
# ------------------------------------------------------------------------------ #
# Main Network
# ------------------------------------------------------------------------------ #
class Network_Slerp_Lifted_Ops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.k = getattr(cfg, "top_k", 4)
        
        gt = fcc_syms.clone()
        self.register_buffer("group_tensor", gt) 

        in_ch = 4
        mid_ch = getattr(cfg, "n_feats", 32)
        
        # 1. Lifting Layer
        self.lift = LiftingConv(in_ch, mid_ch, kernel_size=3, padding=1, group_quats=gt)
        
        # 2. Body
        self.body = nn.Sequential(
            nn.Conv2d(24*mid_ch, 24*mid_ch, 3, padding=1, groups=24), 
            QuaternionReLU(),
            GroupAttention(channels=mid_ch, num_heads=2),
            nn.Conv2d(24*mid_ch, 24*mid_ch, 3, padding=1, groups=24), 
            QuaternionReLU()
        )
        
        # Projection
        self.group_proj = TopKGroupSelector(in_channels=mid_ch, k=self.k, groups=24)

        # 3. Super-Resolution 
        self.latent_slerp = MultiHypothesisSlerp(
            in_channels=self.k * mid_ch, 
            num_hypotheses=self.k, 
            scale_factor=cfg.scale, 
            group_quats=gt
        )
        
        # 4. Refinement Head 
        # FIX IS HERE: Remove in_channels, etc. Just pass group_quats.
        self.outc = RefinementHead(group_quats=gt)

    def forward(self, x):
        x = self.lift(x)
        res = x
        x = self.body(x)
        x = x + res
        x_best = self.group_proj(x)

        # Upsample: Returns single map (B, 4, H, W)
        x_up_single, selection_weights = self.latent_slerp(x_best)
        
        # Refine: Physics smoothing only
        out = self.outc(x_up_single)

        return out, selection_weights
    
# ------------------------------------------------------------------------------ #
# Example usage
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":
    class _Args:
        dropout = 0.0
        n_feats = 128
        n_resblocks = 16
        scale = 4
        kernel_size = 3
        overlap = False

    args = _Args()
    B = 2
    H = W = 32
    
    print("\nInitializing Network_Slerp_Lifted_Ops...")
    model = Network_Slerp_Lifted_Ops(cfg=args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    q_lr = torch.randn(B, 4, H, W)
    q_lr = q_lr / q_lr.norm(dim=1, keepdim=True).clamp_min(1e-8)
    q_sr = model(q_lr)

    print("Output shape (clean):", q_sr.shape)