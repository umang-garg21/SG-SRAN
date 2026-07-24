# boundary_aware_so3o_sr.py
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import _rotation, _wigner


# ============================================================
# 1) Quaternion helpers
# ============================================================

def normalize_quaternion(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def standardize_quaternion_sign(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    sign = torch.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quaternion_inverse(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return quaternion_conjugate(q)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product, scalar-first quaternions.
    q1, q2: (..., 4)
    returns: (..., 4)
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def passive_to_active_quaternion(q_passive: torch.Tensor) -> torch.Tensor:
    """
    Passive specimen->crystal -> active rotation.
    """
    return quaternion_inverse(normalize_quaternion(q_passive))


def rand_quaternion_grid(
    B: int,
    H: int,
    W: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    q = o3.rand_quaternion(B * H * W, device=device, dtype=dtype)
    q = q.view(B, H, W, 4)
    return standardize_quaternion_sign(q)


# ============================================================
# 2) Device-safe Wigner D
# ============================================================

def wigner_D_device(
    l: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    device = alpha.device
    dtype = alpha.dtype

    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)

    X = _wigner.so3_generators(l).to(device=device, dtype=dtype)

    return (
        torch.matrix_exp(alpha * X[1])
        @ torch.matrix_exp(beta * X[0])
        @ torch.matrix_exp(gamma * X[1])
    )


def D_from_quaternion_device(irrep: o3.Irrep, q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion(q)
    R = o3.quaternion_to_matrix(q)
    alpha, beta, gamma = _rotation.matrix_to_angles(R)
    return wigner_D_device(irrep.l, alpha, beta, gamma)


# ============================================================
# 3) Proper cubic group O (24 rotations)
# ============================================================

def proper_cubic_group_O(
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    mats = []
    eye = torch.eye(3, device=device, dtype=dtype)

    for perm in itertools.permutations(range(3)):
        P = eye[list(perm), :]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            S = torch.diag(torch.tensor(signs, device=device, dtype=dtype))
            R = S @ P
            if torch.det(R) > 0:
                mats.append(R)

    G = torch.stack(mats, dim=0)

    keep = []
    for i in range(G.shape[0]):
        if not any(torch.allclose(G[i], G[j], atol=1e-12, rtol=0.0) for j in keep):
            keep.append(i)
    return G[keep]


# ============================================================
# 4) Unique cubic A1 seed in e3nn 4e
# ============================================================

def cubic_A1_seed_l4(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Compute the unique cubic-invariant seed a4 in e3nn's 4e basis.

    Done on CPU for robustness.
    """
    cpu = torch.device("cpu")
    G = proper_cubic_group_O(device=cpu, dtype=dtype)
    ir4 = o3.Irrep("4e")

    Dg = ir4.D_from_matrix(G)
    P = Dg.mean(dim=0)
    P = 0.5 * (P + P.T)

    evals, evecs = torch.linalg.eigh(P)
    idx = torch.argmax(evals)
    a4 = evecs[:, idx]
    a4 = a4 / a4.norm()

    j = torch.argmax(torch.abs(a4))
    if a4[j] < 0:
        a4 = -a4
    return a4


# ============================================================
# 5) Codec: quaternion <-> cubic quotient 4e descriptor
# ============================================================

@dataclass
class DecodeResult:
    quaternions: torch.Tensor
    descriptor_nn: torch.Tensor
    distances: torch.Tensor


class CubicQuotient4eCodec(nn.Module):
    """
    phi(q) = D^4(q_active) a4
    """

    def __init__(
        self,
        passive_input: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.passive_input = passive_input
        self.ir4 = o3.Irrep("4e")
        a4 = cubic_A1_seed_l4(dtype=dtype)
        self.register_buffer("a4", a4)

    @property
    def descriptor_dim(self) -> int:
        return 9

    @property
    def descriptor_radius(self) -> float:
        return float(self.a4.norm().item())

    def to_active(self, q: torch.Tensor) -> torch.Tensor:
        q = normalize_quaternion(q)
        if self.passive_input:
            q = passive_to_active_quaternion(q)
        return standardize_quaternion_sign(q)

    def encode_quaternion(self, q: torch.Tensor) -> torch.Tensor:
        q = self.to_active(q)
        D4 = D_from_quaternion_device(self.ir4, q)         # [..., 9, 9]
        x4 = torch.einsum("...ij,j->...i", D4, self.a4)    # [..., 9]
        return x4

    def encode_map(self, q_map: torch.Tensor) -> torch.Tensor:
        """
        q_map: [B,4,H,W]
        returns: [B,9,H,W]
        """
        B, C, H, W = q_map.shape
        assert C == 4, f"Expected [B,4,H,W], got {q_map.shape}"
        q = q_map.permute(0, 2, 3, 1).contiguous()
        x = self.encode_quaternion(q)
        return x.permute(0, 3, 1, 2).contiguous()

    def build_dictionary(
        self,
        n: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_dict = o3.rand_quaternion(n, device=device, dtype=dtype)
        q_dict = standardize_quaternion_sign(q_dict)
        x_dict = self.encode_quaternion(q_dict)
        return q_dict, x_dict

    @torch.no_grad()
    def decode_by_dictionary(
        self,
        x: torch.Tensor,
        q_dict: torch.Tensor,
        x_dict: Optional[torch.Tensor] = None,
        chunk: int = 4096,
    ) -> DecodeResult:
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, 9)

        if x_dict is None:
            x_dict = self.encode_quaternion(q_dict)

        x_dict = x_dict.to(device=x_flat.device, dtype=x_flat.dtype)
        q_dict = q_dict.to(device=x_flat.device, dtype=x_flat.dtype)

        best_dist = torch.empty(x_flat.shape[0], device=x_flat.device, dtype=x_flat.dtype)
        best_idx = torch.empty(x_flat.shape[0], device=x_flat.device, dtype=torch.long)

        for start in range(0, x_flat.shape[0], chunk):
            stop = min(start + chunk, x_flat.shape[0])
            xi = x_flat[start:stop]
            d2 = torch.cdist(xi, x_dict)
            di, ii = torch.min(d2, dim=1)
            best_dist[start:stop] = di
            best_idx[start:stop] = ii

        q_nn = q_dict[best_idx].reshape(*shape, 4)
        x_nn = x_dict[best_idx].reshape(*shape, 9)
        d_nn = best_dist.reshape(*shape)

        return DecodeResult(
            quaternions=q_nn,
            descriptor_nn=x_nn,
            distances=d_nn,
        )


# ============================================================
# 6) Standard fiber-preserving ops
# ============================================================

class SharedOverMConv2d(nn.Module):
    """
    Apply the same 2D conv to every m-component of a degree-l irrep block.

    Input/output:
        x: [B, mul*(2l+1), H, W]
    """

    def __init__(
        self,
        mul_in: int,
        mul_out: int,
        l: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.mul_in = mul_in
        self.mul_out = mul_out
        self.l = l
        self.d = 2 * l + 1
        self.conv = nn.Conv2d(
            mul_in,
            mul_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.mul_in * self.d, f"Expected {self.mul_in*self.d} channels, got {C}"

        x = x.view(B, self.mul_in, self.d, H, W).permute(0, 2, 1, 3, 4)
        x = x.reshape(B * self.d, self.mul_in, H, W)

        y = self.conv(x)
        H2, W2 = y.shape[-2:]

        y = y.view(B, self.d, self.mul_out, H2, W2).permute(0, 2, 1, 3, 4)
        y = y.reshape(B, self.mul_out * self.d, H2, W2)
        return y


class SharedOverMInterpolation(nn.Module):
    """
    Componentwise interpolation tied across m-components.
    """

    def __init__(
        self,
        mul: int,
        l: int,
        scale_factor: int = 2,
        mode: str = "nearest",
    ):
        super().__init__()
        self.mul = mul
        self.l = l
        self.d = 2 * l + 1
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.mul * self.d, f"Expected {self.mul*self.d} channels, got {C}"

        x = x.view(B, self.mul, self.d, H, W).permute(0, 2, 1, 3, 4)
        x = x.reshape(B * self.d, self.mul, H, W)

        if self.mode in ("bilinear", "bicubic"):
            y = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        else:
            y = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        H2, W2 = y.shape[-2:]
        y = y.view(B, self.d, self.mul, H2, W2).permute(0, 2, 1, 3, 4)
        y = y.reshape(B, self.mul * self.d, H2, W2)
        return y


# ============================================================
# 7) Affinity / boundary helpers
# ============================================================

def affinity_to_boundary(aff: torch.Tensor, center_idx: int) -> torch.Tensor:
    """
    Convert local kernel affinities [B,K2,H,W] into a scalar boundary map [B,1,H,W].

    boundary = 1 - mean(neighbor affinities excluding the center)
    """
    B, K2, H, W = aff.shape
    idx = [k for k in range(K2) if k != center_idx]
    if len(idx) == 0:
        return torch.zeros(B, 1, H, W, device=aff.device, dtype=aff.dtype)
    neigh = aff[:, idx]
    boundary = 1.0 - neigh.mean(dim=1, keepdim=True)
    return boundary.clamp(0.0, 1.0)


class AffinityHead4e(nn.Module):
    """
    Predict local affinities for a KxK masked convolution.

    The head uses only invariant scalar norms of each hidden 4e copy,
    optionally concatenated with a boundary prior.

    Output:
        aff: [B, K*K, H, W] in [0,1]
    """

    def __init__(
        self,
        mul4: int,
        kernel_size: int = 3,
        hidden: Optional[int] = None,
        use_boundary_prior: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.mul4 = mul4
        self.kernel_size = kernel_size
        self.k2 = kernel_size * kernel_size
        self.center_idx = self.k2 // 2
        self.use_boundary_prior = use_boundary_prior

        in_ch = mul4 + (1 if use_boundary_prior else 0)
        hidden = hidden or max(16, in_ch)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, self.k2, kernel_size=1),
        )

    def forward(
        self,
        x4: torch.Tensor,
        boundary_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x4: [B, mul4*9, H, W]
        boundary_prior: [B,1,H,W] or None
        """
        B, C, H, W = x4.shape
        assert C == self.mul4 * 9, f"Expected {self.mul4*9} channels, got {C}"

        xg = x4.view(B, self.mul4, 9, H, W)
        norms = torch.sqrt((xg ** 2).sum(dim=2).clamp_min(1e-12))  # [B,mul4,H,W]

        if self.use_boundary_prior:
            if boundary_prior is None:
                boundary_prior = torch.zeros(B, 1, H, W, device=x4.device, dtype=x4.dtype)
            inp = torch.cat([norms, boundary_prior], dim=1)
        else:
            inp = norms

        aff = torch.sigmoid(self.net(inp))  # [B,K2,H,W]
        aff[:, self.center_idx:self.center_idx + 1] = 1.0
        return aff


# ============================================================
# 8) Masked, renormalized convolution on 4e fibers
# ============================================================

class MaskedSharedOverMConv2d(nn.Module):
    """
    Boundary-aware normalized convolution tied across the m-components of 4e copies.

    Input:
        x:   [B, mul_in*9, H, W]
        aff: [B, K*K, H, W], predicted affinity for each kernel offset

    Output:
        y:   [B, mul_out*9, H, W]

    This layer:
      - unfolds x into KxK patches,
      - multiplies each local patch by the local affinity weights,
      - applies learned convolution weights,
      - renormalizes by the local affinity mass.
    """

    def __init__(
        self,
        mul_in: int,
        mul_out: int,
        kernel_size: int = 3,
        bias: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.mul_in = mul_in
        self.mul_out = mul_out
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.k2 = kernel_size * kernel_size
        self.d = 9
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(mul_out, mul_in, self.k2))
        if bias:
            self.bias = nn.Parameter(torch.zeros(mul_out))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = mul_in * self.k2
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, aff: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.mul_in * self.d, f"Expected {self.mul_in*self.d} channels, got {C}"
        assert aff.shape == (B, self.k2, H, W), (
            f"Expected aff shape {(B, self.k2, H, W)}, got {aff.shape}"
        )

        # x -> [B*d, mul_in, H, W]
        x = x.view(B, self.mul_in, self.d, H, W).permute(0, 2, 1, 3, 4).reshape(B * self.d, self.mul_in, H, W)

        # unfold patches: [B*d, mul_in*K2, H*W]
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        L = patches.shape[-1]
        patches = patches.view(B, self.d, self.mul_in, self.k2, L)  # [B,d,m_in,K2,L]

        aff_flat = aff.view(B, 1, 1, self.k2, L)                    # [B,1,1,K2,L]
        masked = patches * aff_flat

        # learned weighted sum over mul_in and K2
        # masked: [B,d,m_in,K2,L]
        # weight: [m_out,m_in,K2]
        y = torch.einsum("bdikl,oik->bdol", masked, self.weight)     # [B,d,m_out,L]

        # renormalize by local affinity mass so fewer valid pixels do not shrink output
        denom = aff_flat.sum(dim=3).clamp_min(self.eps)              # [B,1,1,L]
        y = y * (self.k2 / denom)

        if self.bias is not None:
            y = y + self.bias.view(1, 1, self.mul_out, 1)

        y = y.view(B, self.d, self.mul_out, H, W).permute(0, 2, 1, 3, 4).reshape(B, self.mul_out * self.d, H, W)
        return y


# ============================================================
# 9) Masked residual blocks
# ============================================================

class NormGate4e(nn.Module):
    """
    Equivariant nonlinearity: gate each 4e copy using its invariant norm.
    """

    def __init__(self, mul: int, hidden: Optional[int] = None):
        super().__init__()
        self.mul = mul
        hidden = hidden or max(16, mul)
        self.net = nn.Sequential(
            nn.Conv2d(mul, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, mul, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.mul * 9, f"Expected {self.mul*9} channels, got {C}"

        xg = x.view(B, self.mul, 9, H, W)
        norms = torch.sqrt((xg ** 2).sum(dim=2).clamp_min(1e-12))    # [B,mul,H,W]
        gates = self.net(norms)
        y = xg * gates.unsqueeze(2)
        return y.view(B, C, H, W)


class MaskedResidual4eBlock(nn.Module):
    def __init__(self, mul: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = MaskedSharedOverMConv2d(mul, mul, kernel_size=kernel_size)
        self.gate = NormGate4e(mul)
        self.conv2 = MaskedSharedOverMConv2d(mul, mul, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, aff: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x, aff)
        y = self.gate(y)
        y = self.conv2(y, aff)
        return x + y


# ============================================================
# 10) Boundary-aware dual-head SR model
# ============================================================

class BoundaryAwareSO3OSRNet(nn.Module):
    """
    Dual-head SR model:
      - descriptor head: HR cubic quotient 4e descriptor
      - affinity/boundary head: HR local affinities and derived boundary map

    Design:
      1) encode LR quaternions -> LR 4e descriptor
      2) lift to hidden 4e copies
      3) predict LR affinities and do masked LR residual blocks
      4) nearest-neighbor upsample hidden field (safe: no averaging)
      5) predict HR affinities and do masked HR residual blocks
      6) project back to HR 4e descriptor

    Optional:
      You can feed an LR boundary prior [B,1,H,W] if available.
    """

    def __init__(
        self,
        hidden_mul4: int = 24,
        num_blocks_lr: int = 3,
        num_blocks_hr: int = 4,
        sr_scale: int = 4,
        passive_input: bool = True,
        dtype: torch.dtype = torch.float64,
        use_boundary_prior: bool = True,
    ):
        super().__init__()
        self.sr_scale = sr_scale
        self.hidden_mul4 = hidden_mul4

        self.codec = CubicQuotient4eCodec(
            passive_input=passive_input,
            dtype=dtype,
        )

        # descriptor lift
        self.in_proj = SharedOverMConv2d(
            mul_in=1,
            mul_out=hidden_mul4,
            l=4,
            kernel_size=3,
            padding=1,
        )

        # LR affinity head + LR masked blocks
        self.aff_lr_head = AffinityHead4e(
            mul4=hidden_mul4,
            kernel_size=3,
            use_boundary_prior=use_boundary_prior,
        )
        self.lr_blocks = nn.ModuleList([
            MaskedResidual4eBlock(hidden_mul4, kernel_size=3)
            for _ in range(num_blocks_lr)
        ])

        # safe upsample: no cross-grain averaging
        self.up_hidden = SharedOverMInterpolation(
            mul=hidden_mul4,
            l=4,
            scale_factor=sr_scale,
            mode="nearest",
        )
        self.up_skip = SharedOverMInterpolation(
            mul=1,
            l=4,
            scale_factor=sr_scale,
            mode="nearest",
        )

        # HR affinity head + HR masked blocks
        self.aff_hr_head = AffinityHead4e(
            mul4=hidden_mul4,
            kernel_size=3,
            use_boundary_prior=use_boundary_prior,
        )
        self.hr_blocks = nn.ModuleList([
            MaskedResidual4eBlock(hidden_mul4, kernel_size=3)
            for _ in range(num_blocks_hr)
        ])

        # output descriptor head
        self.out_proj = SharedOverMConv2d(
            mul_in=hidden_mul4,
            mul_out=1,
            l=4,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        q_lr: torch.Tensor,
        lr_boundary_prior: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        q_lr: [B,4,H,W]
        lr_boundary_prior: optional [B,1,H,W]

        returns dict with:
            descriptor_hr : [B,9,sH,sW]
            affinity_lr   : [B,9,H,W]      (for K=3, K^2=9)
            affinity_hr   : [B,9,sH,sW]
            boundary_lr   : [B,1,H,W]
            boundary_hr   : [B,1,sH,sW]
        """
        # encode LR quaternions to quotient descriptor
        x0 = self.codec.encode_map(q_lr)                          # [B,9,H,W]
        skip = self.up_skip(x0)                                   # [B,9,sH,sW]

        # hidden LR field
        f = self.in_proj(x0)                                      # [B,m*9,H,W]

        # LR affinities + masked LR refinement
        aff_lr = self.aff_lr_head(f, boundary_prior=lr_boundary_prior)
        for block in self.lr_blocks:
            f = block(f, aff_lr)

        # upsample hidden field without averaging across grains
        f_hr = self.up_hidden(f)                                  # [B,m*9,sH,sW]

        # optional upsampled LR prior
        hr_boundary_prior = None
        if lr_boundary_prior is not None:
            hr_boundary_prior = F.interpolate(
                lr_boundary_prior,
                scale_factor=self.sr_scale,
                mode="nearest",
            )

        # HR affinities + masked HR refinement
        aff_hr = self.aff_hr_head(f_hr, boundary_prior=hr_boundary_prior)
        for block in self.hr_blocks:
            f_hr = block(f_hr, aff_hr)

        # descriptor output
        y = self.out_proj(f_hr) + skip                            # [B,9,sH,sW]

        # project back toward the constant-radius shell of valid embeddings
        target_norm = self.codec.a4.norm()
        ynorm = torch.sqrt((y ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12))
        y = target_norm * y / ynorm

        # derived boundary maps
        boundary_lr = affinity_to_boundary(aff_lr, self.aff_lr_head.center_idx)
        boundary_hr = affinity_to_boundary(aff_hr, self.aff_hr_head.center_idx)

        return {
            "descriptor_hr": y,
            "affinity_lr": aff_lr,
            "affinity_hr": aff_hr,
            "boundary_lr": boundary_lr,
            "boundary_hr": boundary_hr,
        }

    @torch.no_grad()
    def decode_descriptor_nn(
        self,
        descriptor_hr: torch.Tensor,
        q_dict: torch.Tensor,
        x_dict: Optional[torch.Tensor] = None,
        chunk: int = 4096,
    ) -> DecodeResult:
        """
        descriptor_hr: [B,9,H,W]
        """
        x = descriptor_hr.permute(0, 2, 3, 1).contiguous()  # [B,H,W,9]
        return self.codec.decode_by_dictionary(x, q_dict=q_dict, x_dict=x_dict, chunk=chunk)


# ============================================================
# 11) Losses
# ============================================================

def descriptor_mse_loss(x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_pred, x_gt)


def descriptor_shell_loss(x_pred: torch.Tensor, target_radius: float = 1.0) -> torch.Tensor:
    r = torch.sqrt((x_pred ** 2).sum(dim=1).clamp_min(1e-12))
    return ((r - target_radius) ** 2).mean()


def boundary_bce_loss(pred_boundary: torch.Tensor, gt_boundary: torch.Tensor) -> torch.Tensor:
    gt_boundary = gt_boundary.to(dtype=pred_boundary.dtype, device=pred_boundary.device)
    return F.binary_cross_entropy(pred_boundary.clamp(1e-6, 1 - 1e-6), gt_boundary)


def combined_boundary_aware_loss(
    model: BoundaryAwareSO3OSRNet,
    pred: dict,
    hr_q_gt: torch.Tensor,
    gt_boundary_hr: Optional[torch.Tensor] = None,
    lam_shell: float = 1e-2,
    lam_boundary: float = 1.0,
) -> dict:
    """
    pred is the model output dict.
    hr_q_gt: [B,4,sH,sW]
    gt_boundary_hr: optional [B,1,sH,sW]

    returns dict with individual terms and total.
    """
    with torch.no_grad():
        x_gt = model.codec.encode_map(hr_q_gt)

    x_pred = pred["descriptor_hr"]
    target_radius = model.codec.descriptor_radius

    loss_desc = descriptor_mse_loss(x_pred, x_gt)
    loss_shell = descriptor_shell_loss(x_pred, target_radius=target_radius)

    total = loss_desc + lam_shell * loss_shell

    out = {
        "total": total,
        "descriptor": loss_desc,
        "shell": loss_shell,
    }

    if gt_boundary_hr is not None:
        loss_b = boundary_bce_loss(pred["boundary_hr"], gt_boundary_hr)
        total = total + lam_boundary * loss_b
        out["boundary"] = loss_b
        out["total"] = total

    return out


# ============================================================
# 12) Demo
# ============================================================

def _demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    B, H, W = 2, 32, 48
    sr_scale = 4
    sH, sW = H * sr_scale, W * sr_scale

    # fake LR quaternion map
    q_lr = rand_quaternion_grid(B, H, W, device=device, dtype=dtype)
    q_lr = q_lr.permute(0, 3, 1, 2).contiguous()   # [B,4,H,W]

    # optional fake LR boundary prior
    lr_boundary_prior = torch.rand(B, 1, H, W, device=device, dtype=dtype)

    model = BoundaryAwareSO3OSRNet(
        hidden_mul4=16,
        num_blocks_lr=2,
        num_blocks_hr=3,
        sr_scale=sr_scale,
        passive_input=False,   # random quats here are active
        dtype=dtype,
        use_boundary_prior=True,
    ).to(device=device, dtype=dtype)

    out = model(q_lr, lr_boundary_prior=lr_boundary_prior)

    print("descriptor_hr:", tuple(out["descriptor_hr"].shape))  # [B,9,sH,sW]
    print("affinity_lr  :", tuple(out["affinity_lr"].shape))    # [B,9,H,W] for K=3
    print("affinity_hr  :", tuple(out["affinity_hr"].shape))    # [B,9,sH,sW]
    print("boundary_lr  :", tuple(out["boundary_lr"].shape))    # [B,1,H,W]
    print("boundary_hr  :", tuple(out["boundary_hr"].shape))    # [B,1,sH,sW]

    # fake HR GT for loss demo
    hr_q_gt = rand_quaternion_grid(B, sH, sW, device=device, dtype=dtype)
    hr_q_gt = hr_q_gt.permute(0, 3, 1, 2).contiguous()

    # fake HR boundary GT
    gt_boundary_hr = torch.rand(B, 1, sH, sW, device=device, dtype=dtype)

    losses = combined_boundary_aware_loss(
        model=model,
        pred=out,
        hr_q_gt=hr_q_gt,
        gt_boundary_hr=gt_boundary_hr,
        lam_shell=1e-2,
        lam_boundary=1.0,
    )

    print("loss total     :", float(losses["total"].item()))
    print("loss descriptor:", float(losses["descriptor"].item()))
    print("loss shell     :", float(losses["shell"].item()))
    print("loss boundary  :", float(losses["boundary"].item()))

    # optional NN decode
    q_dict, x_dict = model.codec.build_dictionary(
        n=10000,
        device=device,
        dtype=dtype,
    )
    dec = model.decode_descriptor_nn(out["descriptor_hr"], q_dict=q_dict, x_dict=x_dict, chunk=2048)
    print("decoded quaternions:", tuple(dec.quaternions.shape))
    print("mean NN distance    :", float(dec.distances.mean().item()))


if __name__ == "__main__":
    _demo()