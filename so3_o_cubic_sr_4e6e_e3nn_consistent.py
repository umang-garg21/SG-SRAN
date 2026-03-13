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
    return torch.where(q[..., :1] < 0.0, -q, q)


def passive_to_active_quaternion(q_passive: torch.Tensor) -> torch.Tensor:
    """
    Convert a passive specimen->crystal quaternion into the corresponding active
    rotation quaternion used internally by this model.
    """
    q_passive = normalize_quaternion(q_passive)
    return standardize_quaternion_sign(o3.inverse_quaternion(q_passive))


def rand_quaternion_grid(
    B: int,
    H: int,
    W: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
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
# 3) Proper cubic group and cubic 4e seed
# ============================================================

def proper_cubic_group_O(
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
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
    keep: list[int] = []
    for i in range(G.shape[0]):
        if not any(torch.allclose(G[i], G[j], atol=1e-12, rtol=0.0) for j in keep):
            keep.append(i)
    return G[keep]


def cubic_A1_seed_l4(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Compute the unique cubic A1 seed in the e3nn 4e basis from the projector
    P = |O|^{-1} sum_{g in O} D^4(g).
    """
    cpu = torch.device("cpu")
    G = proper_cubic_group_O(device=cpu, dtype=dtype)
    ir4 = o3.Irrep("4e")
    Dg = ir4.D_from_matrix(G)
    P = 0.5 * (Dg.mean(dim=0) + Dg.mean(dim=0).T)
    evals, evecs = torch.linalg.eigh(P)
    a4 = evecs[:, torch.argmax(evals)]
    a4 = a4 / a4.norm()
    j = torch.argmax(torch.abs(a4))
    if a4[j] < 0:
        a4 = -a4
    return a4


# ============================================================
# 3) Codec: quaternion <-> cubic quotient descriptor in 4e
# ============================================================

@dataclass
class DecodeResult:
    quaternions: torch.Tensor
    descriptor_nn: torch.Tensor
    distances: torch.Tensor


class CubicQuotient4eCodec(nn.Module):
    """
    phi(q) = D^4(q_active) a4

    Internally we use e3nn's official Irrep.D_from_quaternion on active unit
    quaternions. If your inputs are passive specimen->crystal quaternions,
    set passive_input=True so they are inverted before encoding.
    """

    def __init__(
        self,
        passive_input: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.passive_input = passive_input
        self.ir4 = o3.Irrep("4e")
        self.register_buffer("a4", cubic_A1_seed_l4(dtype=dtype))

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
        D4 = D_from_quaternion_device(self.ir4, q)
        return torch.einsum("...ij,j->...i", D4, self.a4.to(device=D4.device, dtype=D4.dtype))

    def encode_map(self, q_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q_map.shape
        assert C == 4, f"Expected [B,4,H,W], got {q_map.shape}"
        q = q_map.permute(0, 2, 3, 1).contiguous()
        x = self.encode_quaternion(q)
        return x.permute(0, 3, 1, 2).contiguous()

    def build_dictionary(
        self,
        n: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
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
        topk: int = 1,
    ) -> DecodeResult:
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, 9)

        if x_dict is None:
            x_dict = self.encode_quaternion(q_dict)

        x_dict = x_dict.to(device=x_flat.device, dtype=x_flat.dtype)
        q_dict = q_dict.to(device=x_flat.device, dtype=x_flat.dtype)
        x_dict_n = F.normalize(x_dict, dim=-1, eps=1e-12)

        k = min(max(1, int(topk)), int(x_dict.shape[0]))
        q_out = torch.empty(x_flat.shape[0], 4, device=x_flat.device, dtype=x_flat.dtype)
        x_out = torch.empty(x_flat.shape[0], 9, device=x_flat.device, dtype=x_flat.dtype)
        d_out = torch.empty(x_flat.shape[0], device=x_flat.device, dtype=x_flat.dtype)

        for start in range(0, x_flat.shape[0], chunk):
            stop = min(start + chunk, x_flat.shape[0])
            xi = x_flat[start:stop]
            sims = F.normalize(xi, dim=-1, eps=1e-12) @ x_dict_n.T
            _, top_idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
            if k == 1:
                idx = top_idx[:, 0]
            else:
                xk = x_dict[top_idx]
                d2 = ((xk - xi.unsqueeze(1)) ** 2).sum(dim=-1)
                j = torch.argmin(d2, dim=1)
                idx = top_idx[torch.arange(xi.shape[0], device=xi.device), j]
            q_sel = q_dict[idx]
            x_sel = x_dict[idx]
            q_out[start:stop] = q_sel
            x_out[start:stop] = x_sel
            d_out[start:stop] = torch.norm(x_sel - xi, dim=-1)

        return DecodeResult(
            quaternions=q_out.reshape(*shape, 4),
            descriptor_nn=x_out.reshape(*shape, 9),
            distances=d_out.reshape(*shape),
        )


# ============================================================
# 4) Fiber-preserving spatial operators
# ============================================================

class SharedOverMConv2d(nn.Module):
    """
    Apply the same 2D scalar convolution to each m-component of a single irrep.
    This mixes multiplicities but never mixes the m-basis inside an irrep copy.
    """

    def __init__(
        self,
        mul_in: int,
        mul_out: int,
        l: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.mul_in = mul_in
        self.mul_out = mul_out
        self.d = 2 * l + 1
        self.conv = nn.Conv2d(mul_in, mul_out, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.mul_in * self.d, f"Expected {self.mul_in * self.d}, got {C}"
        x = x.view(B, self.mul_in, self.d, H, W).permute(0, 2, 1, 3, 4).reshape(B * self.d, self.mul_in, H, W)
        y = self.conv(x)
        H2, W2 = y.shape[-2:]
        y = y.view(B, self.d, self.mul_out, H2, W2).permute(0, 2, 1, 3, 4).reshape(B, self.mul_out * self.d, H2, W2)
        return y


class MaskedSharedOverMConv2d(nn.Module):
    """
    Boundary-aware normalized convolution tied over m-components.
    """

    def __init__(
        self,
        mul_in: int,
        mul_out: int,
        d: int,
        kernel_size: int = 3,
        bias: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.mul_in = mul_in
        self.mul_out = mul_out
        self.d = d
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.k2 = kernel_size * kernel_size
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(mul_out, mul_in, self.k2))
        self.bias = nn.Parameter(torch.zeros(mul_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = mul_in * self.k2
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, aff: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.mul_in * self.d
        assert aff.shape == (B, self.k2, H, W)

        x = x.view(B, self.mul_in, self.d, H, W).permute(0, 2, 1, 3, 4).reshape(B * self.d, self.mul_in, H, W)
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        L = patches.shape[-1]
        patches = patches.view(B, self.d, self.mul_in, self.k2, L)

        aff_flat = aff.view(B, 1, 1, self.k2, L)
        valid = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        valid = F.unfold(valid, kernel_size=self.kernel_size, padding=self.padding, stride=1).view(B, 1, 1, self.k2, L)
        masked_aff = aff_flat * valid
        masked = patches * masked_aff

        y = torch.einsum("bdikl,oik->bdol", masked, self.weight)
        denom = masked_aff.sum(dim=3).clamp_min(self.eps)
        y = y * (self.k2 / denom)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, self.mul_out, 1)
        y = y.view(B, self.d, self.mul_out, H, W).permute(0, 2, 1, 3, 4).reshape(B, self.mul_out * self.d, H, W)
        return y


class MaskedLearnableSharedOverMUpsample2d(nn.Module):
    """
    Learnable subpixel upsampler tied over m-components and modulated by a
    local affinity mask.
    """

    def __init__(
        self,
        mul_in: int,
        mul_out: int,
        d: int,
        scale_factor: int = 2,
        kernel_size: int = 3,
        bias: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.mul_in = mul_in
        self.mul_out = mul_out
        self.d = d
        self.scale_factor = scale_factor
        self.num_phases = scale_factor * scale_factor
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.k2 = kernel_size * kernel_size
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.num_phases, mul_out, mul_in, self.k2))
        self.bias = nn.Parameter(torch.zeros(self.num_phases, mul_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = mul_in * self.k2
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, aff: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.scale_factor
        assert C == self.mul_in * self.d
        assert aff.shape == (B, self.k2, H, W)

        x = x.view(B, self.mul_in, self.d, H, W).permute(0, 2, 1, 3, 4).reshape(B * self.d, self.mul_in, H, W)
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        L = patches.shape[-1]
        patches = patches.view(B, self.d, self.mul_in, self.k2, L)

        aff_flat = aff.view(B, 1, 1, self.k2, L)
        valid = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        valid = F.unfold(valid, kernel_size=self.kernel_size, padding=self.padding, stride=1).view(B, 1, 1, self.k2, L)
        masked_aff = aff_flat * valid
        masked = patches * masked_aff

        y = torch.einsum("bdikl,poik->bdpol", masked, self.weight)
        denom = masked_aff.sum(dim=3).clamp_min(self.eps)
        y = y * (self.k2 / denom).unsqueeze(2)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, self.num_phases, self.mul_out, 1)

        y = y.view(B, self.d, self.num_phases, self.mul_out, H, W)
        y = y.permute(0, 3, 1, 2, 4, 5).reshape(B, self.mul_out * self.d * self.num_phases, H, W)
        return F.pixel_shuffle(y, upscale_factor=s)


# ============================================================
# 5) e3nn tensor-product lifts and couplings
# ============================================================

class PerPixelSelfTensorLift4to6(nn.Module):
    """
    1x4e ⊗ 1x4e -> mul6 x 6e, per pixel.
    """

    def __init__(self, mul6: int):
        super().__init__()
        self.mul6 = mul6
        self.tp = o3.FullyConnectedTensorProduct("1x4e", "1x4e", f"{mul6}x6e", shared_weights=True, internal_weights=True)

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x4.shape
        assert C == 9
        x = x4.permute(0, 2, 3, 1).contiguous().view(-1, 9)
        y = self.tp(x, x)
        return y.view(B, H, W, self.mul6 * 13).permute(0, 3, 1, 2).contiguous()


class CrossIrrepCoupling46(nn.Module):
    """
    Valid e3nn couplings used repeatedly inside residual blocks:
      (m4 x 4e) ⊗ (m6 x 6e) -> (m4 x 4e)
      (m4 x 4e) ⊗ (m4 x 4e) -> (m6 x 6e)
    """

    def __init__(self, mul4: int, mul6: int):
        super().__init__()
        self.mul4 = mul4
        self.mul6 = mul6
        self.tp46_to4 = o3.FullyConnectedTensorProduct(
            f"{mul4}x4e", f"{mul6}x6e", f"{mul4}x4e", shared_weights=True, internal_weights=True
        )
        self.tp44_to6 = o3.FullyConnectedTensorProduct(
            f"{mul4}x4e", f"{mul4}x4e", f"{mul6}x6e", shared_weights=True, internal_weights=True
        )
        self.scale_4 = nn.Parameter(torch.tensor(0.1))
        self.scale_6 = nn.Parameter(torch.tensor(0.1))

    def forward(self, x4: torch.Tensor, x6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x4.shape
        x4p = x4.permute(0, 2, 3, 1).contiguous().view(-1, self.mul4 * 9)
        x6p = x6.permute(0, 2, 3, 1).contiguous().view(-1, self.mul6 * 13)
        d4 = self.tp46_to4(x4p, x6p)
        d6 = self.tp44_to6(x4p, x4p)
        d4 = d4.view(B, H, W, self.mul4 * 9).permute(0, 3, 1, 2).contiguous()
        d6 = d6.view(B, H, W, self.mul6 * 13).permute(0, 3, 1, 2).contiguous()
        return x4 + self.scale_4.to(dtype=x4.dtype, device=x4.device) * d4, x6 + self.scale_6.to(dtype=x6.dtype, device=x6.device) * d6


class Project6eTo4eViaTP(nn.Module):
    """
    Valid 6e contribution to the descriptor head:
      (m6 x 6e) ⊗ (m6 x 6e) -> (m4_out x 4e)
    """

    def __init__(self, mul6: int, mul4_out: int = 1):
        super().__init__()
        self.mul6 = mul6
        self.mul4_out = mul4_out
        self.tp = o3.FullyConnectedTensorProduct(
            f"{mul6}x6e", f"{mul6}x6e", f"{mul4_out}x4e", shared_weights=True, internal_weights=True
        )

    def forward(self, x6: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x6.shape
        x = x6.permute(0, 2, 3, 1).contiguous().view(-1, self.mul6 * 13)
        y = self.tp(x, x)
        return y.view(B, H, W, self.mul4_out * 9).permute(0, 3, 1, 2).contiguous()


# ============================================================
# 6) Affinity and boundary helpers
# ============================================================

def replace_affinity_center(aff: torch.Tensor, center_idx: int, value: float = 1.0) -> torch.Tensor:
    left = aff[:, :center_idx]
    center = torch.full((aff.shape[0], 1, aff.shape[2], aff.shape[3]), float(value), device=aff.device, dtype=aff.dtype)
    right = aff[:, center_idx + 1:]
    return torch.cat([left, center, right], dim=1)


def descriptor_to_local_affinity(
    x_desc: torch.Tensor,
    kernel_size: int = 3,
    tau: float = 0.6,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, C, H, W = x_desc.shape
    assert C == 9
    p = kernel_size // 2
    k2 = kernel_size * kernel_size
    center_idx = k2 // 2
    x_pad = F.pad(x_desc, (p, p, p, p), mode="replicate")
    nei = F.unfold(x_pad, kernel_size=kernel_size).view(B, C, k2, H, W)
    ctr = x_desc.unsqueeze(2)
    d2 = ((nei - ctr) ** 2).sum(dim=1)
    aff = torch.exp(-d2 / (tau * tau)).clamp(min=eps, max=1.0 - eps)
    return replace_affinity_center(aff, center_idx=center_idx, value=1.0)


def affinity_to_boundary(aff: torch.Tensor, center_idx: int) -> torch.Tensor:
    idx = [k for k in range(aff.shape[1]) if k != center_idx]
    if not idx:
        return torch.zeros((aff.shape[0], 1, aff.shape[2], aff.shape[3]), device=aff.device, dtype=aff.dtype)
    return (1.0 - aff[:, idx].mean(dim=1, keepdim=True)).clamp(0.0, 1.0)


def harden_affinity(
    aff: torch.Tensor,
    center_idx: int,
    mode: str = "none",
    threshold: float = 0.5,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Enforce boundary isolation in local neighborhoods.

    - ``none``: keep the soft affinity.
    - ``hard``: binary gate at ``threshold``.
    - ``soft``: differentiable sharp gate around ``threshold``.
    """
    gate_mode = str(mode).strip().lower()
    if gate_mode in ("none", "off", "false", "0"):
        return replace_affinity_center(aff, center_idx=center_idx, value=1.0)

    thr = float(max(1e-6, min(1.0 - 1e-6, threshold)))
    if gate_mode in ("hard", "binary", "strict"):
        gate = (aff >= thr).to(dtype=aff.dtype)
    elif gate_mode in ("soft", "sigmoid", "st"):
        temp = float(max(1e-4, temperature))
        logit_thr = math.log(thr / (1.0 - thr))
        gate = torch.sigmoid((torch.logit(aff.clamp(1e-6, 1.0 - 1e-6), eps=1e-6) - logit_thr) / temp)
    else:
        raise ValueError(f"Unknown affinity hardening mode: {mode}")

    hardened = (aff * gate).clamp(0.0, 1.0)
    return replace_affinity_center(hardened, center_idx=center_idx, value=1.0)


def local_boundary_pair_target(gt_boundary: torch.Tensor, kernel_size: int = 3, sharpness: float = 6.0) -> torch.Tensor:
    B, _, H, W = gt_boundary.shape
    k2 = kernel_size * kernel_size
    center_idx = k2 // 2
    patches = F.unfold(gt_boundary, kernel_size=kernel_size, padding=kernel_size // 2).view(B, 1, k2, H * W)
    center = patches[:, :, center_idx:center_idx + 1, :]
    pair_b = torch.maximum(center, patches)
    tgt = torch.exp(-sharpness * pair_b).view(B, k2, H, W)
    return replace_affinity_center(tgt, center_idx=center_idx, value=1.0).clamp(0.0, 1.0)


class BoundaryAffinityHead46(nn.Module):
    """
    HR/LR affinity head from invariant 4e/6e norms plus a descriptor-geometry prior.
    """

    def __init__(
        self,
        mul4: int,
        mul6: int,
        kernel_size: int = 3,
        hidden: Optional[int] = None,
        use_boundary_prior: bool = True,
        tau: float = 0.6,
        hard_mask_mode: str = "none",
        hard_mask_threshold: float = 0.5,
        hard_mask_temperature: float = 0.05,
    ):
        super().__init__()
        self.mul4 = mul4
        self.mul6 = mul6
        self.kernel_size = kernel_size
        self.k2 = kernel_size * kernel_size
        self.center_idx = self.k2 // 2
        self.use_boundary_prior = use_boundary_prior
        self.tau = tau
        self.hard_mask_mode = hard_mask_mode
        self.hard_mask_threshold = float(hard_mask_threshold)
        self.hard_mask_temperature = float(hard_mask_temperature)

        in_ch = mul4 + mul6 + (1 if use_boundary_prior else 0)
        hidden = hidden or max(16, in_ch)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.aff_head = nn.Conv2d(hidden, self.k2, kernel_size=1)
        self.boundary_head = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.aff_head.weight)
        nn.init.zeros_(self.aff_head.bias)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.zeros_(self.boundary_head.bias)
        self.gamma_raw = nn.Parameter(torch.tensor(0.5))

    def boundary_pair_penalty(self, boundary_prob: torch.Tensor) -> torch.Tensor:
        B, _, H, W = boundary_prob.shape
        patches = F.unfold(boundary_prob, kernel_size=self.kernel_size, padding=self.kernel_size // 2).view(B, 1, self.k2, H * W)
        center = patches[:, :, self.center_idx:self.center_idx + 1, :]
        return torch.maximum(center, patches).view(B, self.k2, H, W)

    def forward(
        self,
        x4: torch.Tensor,
        x6: torch.Tensor,
        x_desc: torch.Tensor,
        boundary_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = x4.shape
        n4 = torch.sqrt((x4.view(B, self.mul4, 9, H, W) ** 2).sum(dim=2).clamp_min(1e-12))
        n6 = torch.sqrt((x6.view(B, self.mul6, 13, H, W) ** 2).sum(dim=2).clamp_min(1e-12))
        parts = [n4, n6]
        if self.use_boundary_prior:
            if boundary_prior is None:
                boundary_prior = torch.zeros(B, 1, H, W, device=x4.device, dtype=x4.dtype)
            parts.append(boundary_prior)

        h = self.trunk(torch.cat(parts, dim=1))
        learned_logits = self.aff_head(h)
        boundary_logit = self.boundary_head(h)
        boundary_prob = torch.sigmoid(boundary_logit)

        prior = descriptor_to_local_affinity(x_desc, kernel_size=self.kernel_size, tau=self.tau)
        logits = torch.logit(prior, eps=1e-6) + learned_logits
        logits = logits - F.softplus(self.gamma_raw).to(dtype=x4.dtype, device=x4.device) * self.boundary_pair_penalty(boundary_prob)
        aff = harden_affinity(
            torch.sigmoid(logits),
            center_idx=self.center_idx,
            mode=self.hard_mask_mode,
            threshold=self.hard_mask_threshold,
            temperature=self.hard_mask_temperature,
        )
        return aff, boundary_prob, boundary_logit


# ============================================================
# 7) Gating, descriptor head, and residual blocks
# ============================================================

class NormGate46(nn.Module):
    def __init__(self, mul4: int, mul6: int, hidden: Optional[int] = None):
        super().__init__()
        self.mul4 = mul4
        self.mul6 = mul6
        total = mul4 + mul6
        hidden = hidden or max(16, total)
        self.net = nn.Sequential(
            nn.Conv2d(total, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, total, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x4: torch.Tensor, x6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x4.shape
        n4 = torch.sqrt((x4.view(B, self.mul4, 9, H, W) ** 2).sum(dim=2).clamp_min(1e-12))
        n6 = torch.sqrt((x6.view(B, self.mul6, 13, H, W) ** 2).sum(dim=2).clamp_min(1e-12))
        gates = self.net(torch.cat([n4, n6], dim=1))
        g4 = gates[:, :self.mul4].unsqueeze(2)
        g6 = gates[:, self.mul4:].unsqueeze(2)
        y4 = (x4.view(B, self.mul4, 9, H, W) * g4).view_as(x4)
        y6 = (x6.view(B, self.mul6, 13, H, W) * g6).view_as(x6)
        return y4, y6


class DescriptorHead46(nn.Module):
    """
    Valid descriptor head:
      hidden 4e -> 1x4e   via multiplicity mixing
      hidden 6e -> 1x4e   via TP: (m6 x 6e) ⊗ (m6 x 6e) -> 1x4e
      fuse both 4e contributions with a scalar multiplicity mixer
    """

    def __init__(self, mul4: int, mul6: int):
        super().__init__()
        self.from4 = SharedOverMConv2d(mul4, 1, l=4, kernel_size=3, padding=1)
        self.from6 = Project6eTo4eViaTP(mul6=mul6, mul4_out=1)
        self.mix = SharedOverMConv2d(2, 1, l=4, kernel_size=1, padding=0)

    def forward(self, x4: torch.Tensor, x6: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.from4(x4), self.from6(x6)], dim=1)
        return self.mix(y)


class CoupledMaskedResidual46Block(nn.Module):
    def __init__(self, mul4: int, mul6: int, kernel_size: int = 3):
        super().__init__()
        self.conv4_1 = MaskedSharedOverMConv2d(mul4, mul4, d=9, kernel_size=kernel_size)
        self.conv6_1 = MaskedSharedOverMConv2d(mul6, mul6, d=13, kernel_size=kernel_size)
        self.couple = CrossIrrepCoupling46(mul4, mul6)
        self.gate = NormGate46(mul4, mul6)
        self.conv4_2 = MaskedSharedOverMConv2d(mul4, mul4, d=9, kernel_size=kernel_size)
        self.conv6_2 = MaskedSharedOverMConv2d(mul6, mul6, d=13, kernel_size=kernel_size)

    def forward(self, x4: torch.Tensor, x6: torch.Tensor, aff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y4 = self.conv4_1(x4, aff)
        y6 = self.conv6_1(x6, aff)
        y4, y6 = self.couple(y4, y6)
        y4, y6 = self.gate(y4, y6)
        y4 = self.conv4_2(y4, aff)
        y6 = self.conv6_2(y6, aff)
        return x4 + y4, x6 + y6


class CoupledMaskedResidual46Stack(nn.Module):
    def __init__(self, mul4: int, mul6: int, num_blocks: int, kernel_size: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([CoupledMaskedResidual46Block(mul4, mul6, kernel_size=kernel_size) for _ in range(num_blocks)])

    def forward(self, x4: torch.Tensor, x6: torch.Tensor, aff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x4, x6 = block(x4, x6, aff)
        return x4, x6


# ============================================================
# 8) Multistage learnable masked upsampling
# ============================================================

def factorize_scale(scale: int) -> list[int]:
    if not isinstance(scale, int) or scale < 2:
        raise ValueError(f"scale must be integer >= 2, got {scale}")
    factors: list[int] = []
    s = scale
    while s % 2 == 0:
        factors.append(2)
        s //= 2
    while s % 3 == 0:
        factors.append(3)
        s //= 3
    if s > 1:
        factors.append(s)
    return factors


class StageMaskedLearnableUpsample46(nn.Module):
    def __init__(
        self,
        mul4: int,
        mul6: int,
        scale_factor: int = 2,
        kernel_size: int = 3,
        affinity_hidden: int = 32,
        affinity_tau: float = 0.6,
        hard_mask_mode: str = "none",
        hard_mask_threshold: float = 0.5,
        hard_mask_temperature: float = 0.05,
        stage_refine_blocks: int = 1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.up4 = MaskedLearnableSharedOverMUpsample2d(mul4, mul4, d=9, scale_factor=scale_factor, kernel_size=kernel_size)
        self.up6 = MaskedLearnableSharedOverMUpsample2d(mul6, mul6, d=13, scale_factor=scale_factor, kernel_size=kernel_size)
        self.up_skip = MaskedLearnableSharedOverMUpsample2d(1, 1, d=9, scale_factor=scale_factor, kernel_size=kernel_size)
        self.affinity = BoundaryAffinityHead46(
            mul4,
            mul6,
            kernel_size=kernel_size,
            hidden=affinity_hidden,
            use_boundary_prior=True,
            tau=affinity_tau,
            hard_mask_mode=hard_mask_mode,
            hard_mask_threshold=hard_mask_threshold,
            hard_mask_temperature=hard_mask_temperature,
        )
        self.refine = CoupledMaskedResidual46Stack(mul4, mul6, num_blocks=max(0, int(stage_refine_blocks)), kernel_size=kernel_size)
        self.desc_head = DescriptorHead46(mul4, mul6)

    def forward(
        self,
        x4: torch.Tensor,
        x6: torch.Tensor,
        x_desc: torch.Tensor,
        aff_lr: torch.Tensor,
        boundary_lr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x4_hr = self.up4(x4, aff_lr)
        x6_hr = self.up6(x6, aff_lr)
        skip_hr = self.up_skip(x_desc, aff_lr)

        boundary_prior_hr = None
        if boundary_lr is not None:
            boundary_prior_hr = F.interpolate(boundary_lr, scale_factor=self.scale_factor, mode="nearest")

        aff_hr0, boundary_hr0, boundary_logit_hr0 = self.affinity(x4_hr, x6_hr, skip_hr, boundary_prior=boundary_prior_hr)
        x4_hr, x6_hr = self.refine(x4_hr, x6_hr, aff_hr0)
        provisional = self.desc_head(x4_hr, x6_hr) + skip_hr

        aff_hr, boundary_hr, boundary_logit_hr = self.affinity(x4_hr, x6_hr, provisional, boundary_prior=boundary_hr0)
        x4_hr, x6_hr = self.refine(x4_hr, x6_hr, aff_hr)
        desc_out = self.desc_head(x4_hr, x6_hr) + skip_hr

        aux = {
            "skip_hr": skip_hr,
            "affinity_hr0": aff_hr0,
            "boundary_hr0": boundary_hr0,
            "boundary_logit_hr0": boundary_logit_hr0,
            "provisional_descriptor_hr": provisional,
            "affinity_hr": aff_hr,
            "boundary_hr": boundary_hr,
            "boundary_logit_hr": boundary_logit_hr,
        }
        return x4_hr, x6_hr, desc_out, aux


# ============================================================
# 9) Main network
# ============================================================

class SO3OQuotientSRNet4e6eE3NN(nn.Module):
    """
    Best e3nn-consistent version:
      - internal active-rotation convention
      - 4e quotient descriptor output
      - hidden 4e + 6e fibers
      - only valid e3nn tensor-product couplings
      - learnable affinity-aware upsampler
      - multistage x2/x3 factorization
      - boundary-aware masked refinement at every stage
    """

    def __init__(
        self,
        hidden_mul4: int = 16,
        hidden_mul6: int = 8,
        num_blocks_lr: int = 2,
        num_blocks_hr: int = 1,
        sr_scale: int = 4,
        passive_input: bool = True,
        affinity_kernel_size: int = 3,
        affinity_tau: float = 0.6,
        affinity_hidden: int = 32,
        hard_mask_mode: str = "none",
        hard_mask_threshold: float = 0.5,
        hard_mask_temperature: float = 0.05,
        stage_refine_blocks: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.sr_scale = sr_scale
        self.hidden_mul4 = hidden_mul4
        self.hidden_mul6 = hidden_mul6

        self.codec = CubicQuotient4eCodec(passive_input=passive_input, dtype=dtype)
        self.in4 = SharedOverMConv2d(1, hidden_mul4, l=4, kernel_size=3, padding=1)
        self.in6 = PerPixelSelfTensorLift4to6(hidden_mul6)

        self.lr_affinity = BoundaryAffinityHead46(
            hidden_mul4,
            hidden_mul6,
            kernel_size=affinity_kernel_size,
            hidden=affinity_hidden,
            use_boundary_prior=False,
            tau=affinity_tau,
            hard_mask_mode=hard_mask_mode,
            hard_mask_threshold=hard_mask_threshold,
            hard_mask_temperature=hard_mask_temperature,
        )
        self.lr_blocks = CoupledMaskedResidual46Stack(hidden_mul4, hidden_mul6, num_blocks=num_blocks_lr, kernel_size=affinity_kernel_size)

        self.stage_blocks = nn.ModuleList([
            StageMaskedLearnableUpsample46(
                hidden_mul4,
                hidden_mul6,
                scale_factor=sf,
                kernel_size=affinity_kernel_size,
                affinity_hidden=affinity_hidden,
                affinity_tau=affinity_tau,
                hard_mask_mode=hard_mask_mode,
                hard_mask_threshold=hard_mask_threshold,
                hard_mask_temperature=hard_mask_temperature,
                stage_refine_blocks=stage_refine_blocks,
            )
            for sf in factorize_scale(sr_scale)
        ])

        self.hr_post = CoupledMaskedResidual46Stack(hidden_mul4, hidden_mul6, num_blocks=num_blocks_hr, kernel_size=affinity_kernel_size)
        self.final_desc_head = DescriptorHead46(hidden_mul4, hidden_mul6)

    def encode(self, q_lr: torch.Tensor) -> torch.Tensor:
        return self.codec.encode_map(q_lr)

    def forward(self, q_lr: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.encode(q_lr)
        x4 = self.in4(x0)
        x6 = self.in6(x0)

        aff_lr, boundary_lr, boundary_logit_lr = self.lr_affinity(x4, x6, x0, boundary_prior=None)
        x4, x6 = self.lr_blocks(x4, x6, aff_lr)

        desc = x0
        out: dict[str, torch.Tensor] = {
            "descriptor_lr": x0,
            "affinity_lr": aff_lr,
            "boundary_lr": boundary_lr,
            "boundary_logit_lr": boundary_logit_lr,
        }

        for i, stage in enumerate(self.stage_blocks):
            x4, x6, desc, aux = stage(x4, x6, desc, aff_lr, boundary_lr)
            aff_lr = aux["affinity_hr"]
            boundary_lr = aux["boundary_hr"]
            desc = desc
            for key, value in aux.items():
                out[f"stage{i+1}_{key}"] = value

        x4, x6 = self.hr_post(x4, x6, aff_lr)
        skip_final = out[f"stage{len(self.stage_blocks)}_skip_hr"]
        descriptor_hr_raw = self.final_desc_head(x4, x6) + skip_final
        descriptor_hr = self.codec.a4.norm() * descriptor_hr_raw / torch.sqrt((descriptor_hr_raw ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12))

        out.update({
            "descriptor_hr_raw": descriptor_hr_raw,
            "descriptor_hr": descriptor_hr,
            "affinity_hr": aff_lr,
            "boundary_hr": boundary_lr,
        })
        return out

    def forward_descriptor(self, q_lr: torch.Tensor) -> torch.Tensor:
        return self.forward(q_lr)["descriptor_hr"]

    @torch.no_grad()
    def forward_quaternion_nn(
        self,
        q_lr: torch.Tensor,
        q_dict: torch.Tensor,
        x_dict: Optional[torch.Tensor] = None,
        chunk: int = 4096,
        topk: int = 1,
    ) -> DecodeResult:
        x_hr = self.forward_descriptor(q_lr)
        x_hr_last = x_hr.permute(0, 2, 3, 1).contiguous()
        return self.codec.decode_by_dictionary(x_hr_last, q_dict=q_dict, x_dict=x_dict, chunk=chunk, topk=topk)


SO3OQuotientSRNet4e6e = SO3OQuotientSRNet4e6eE3NN


class SO3OQuotientSRNet4e6eE3NNLite(SO3OQuotientSRNet4e6eE3NN):
    """
    Lightweight variant with reduced channel counts and shallower refinement.
    Designed for faster experimentation while preserving the same API.
    """

    def __init__(
        self,
        hidden_mul4: int = 8,
        hidden_mul6: int = 4,
        num_blocks_lr: int = 1,
        num_blocks_hr: int = 0,
        sr_scale: int = 4,
        passive_input: bool = True,
        affinity_kernel_size: int = 3,
        affinity_tau: float = 0.35,
        affinity_hidden: int = 16,
        hard_mask_mode: str = "hard",
        hard_mask_threshold: float = 0.5,
        hard_mask_temperature: float = 0.05,
        stage_refine_blocks: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            hidden_mul4=hidden_mul4,
            hidden_mul6=hidden_mul6,
            num_blocks_lr=num_blocks_lr,
            num_blocks_hr=num_blocks_hr,
            sr_scale=sr_scale,
            passive_input=passive_input,
            affinity_kernel_size=affinity_kernel_size,
            affinity_tau=affinity_tau,
            affinity_hidden=affinity_hidden,
            hard_mask_mode=hard_mask_mode,
            hard_mask_threshold=hard_mask_threshold,
            hard_mask_temperature=hard_mask_temperature,
            stage_refine_blocks=stage_refine_blocks,
            dtype=dtype,
        )


# ============================================================
# 10) Losses
# ============================================================

def descriptor_mse_loss(x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_pred, x_gt)


def descriptor_shell_loss(x_pred_raw: torch.Tensor, target_radius: float = 1.0) -> torch.Tensor:
    r = torch.sqrt((x_pred_raw ** 2).sum(dim=1).clamp_min(1e-12))
    return ((r - target_radius) ** 2).mean()


def boundary_bce_loss(pred_boundary: torch.Tensor, gt_boundary: torch.Tensor) -> torch.Tensor:
    target = gt_boundary.to(device=pred_boundary.device, dtype=pred_boundary.dtype)
    return F.binary_cross_entropy(pred_boundary.clamp(1e-6, 1 - 1e-6), target)


def boundary_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def local_contrastive_descriptor_loss(
    x_desc: torch.Tensor,
    gt_boundary: torch.Tensor,
    kernel_size: int = 3,
    margin: float = 0.35,
    same_w: float = 1.0,
    cross_w: float = 1.0,
) -> torch.Tensor:
    B, C, H, W = x_desc.shape
    k2 = kernel_size * kernel_size
    center_idx = k2 // 2

    x_pad = F.pad(x_desc, (kernel_size // 2,) * 4, mode="replicate")
    x_nei = F.unfold(x_pad, kernel_size=kernel_size).view(B, C, k2, H, W)
    x_ctr = x_desc.unsqueeze(2)
    dist = torch.sqrt(((x_nei - x_ctr) ** 2).sum(dim=1).clamp_min(1e-12))

    b = gt_boundary.to(device=x_desc.device, dtype=x_desc.dtype)
    b_pad = F.pad(b, (kernel_size // 2,) * 4, mode="replicate")
    b_nei = F.unfold(b_pad, kernel_size=kernel_size).view(B, 1, k2, H, W)
    b_ctr = b.unsqueeze(2)
    pair_b = torch.maximum(b_ctr, b_nei).squeeze(1)

    same_mask = (pair_b < 0.5).to(dtype=x_desc.dtype)
    cross_mask = (pair_b >= 0.5).to(dtype=x_desc.dtype)
    keep = torch.ones((1, k2, 1, 1), device=x_desc.device, dtype=x_desc.dtype)
    keep[:, center_idx:center_idx + 1] = 0.0
    same_mask = same_mask * keep
    cross_mask = cross_mask * keep

    same_loss = (same_mask * dist.pow(2)).sum() / same_mask.sum().clamp_min(1.0)
    cross_loss = (cross_mask * F.relu(margin - dist).pow(2)).sum() / cross_mask.sum().clamp_min(1.0)
    return same_w * same_loss + cross_w * cross_loss


def combined_e3nn_consistent_loss(
    model: SO3OQuotientSRNet4e6eE3NN,
    pred: dict[str, torch.Tensor],
    hr_q_gt: torch.Tensor,
    gt_boundary_hr: Optional[torch.Tensor] = None,
    lam_shell: float = 1e-2,
    lam_boundary: float = 1.0,
    lam_dice: float = 0.5,
    lam_affinity: float = 0.5,
    lam_contrast: float = 0.25,
    lam_cross_aff: float = 0.0,
    lam_interior: float = 0.0,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        x_gt = model.codec.encode_map(hr_q_gt)

    x_pred = pred["descriptor_hr"]
    x_pred_raw = pred["descriptor_hr_raw"]
    total = descriptor_mse_loss(x_pred, x_gt)
    out = {"descriptor": total}

    shell = descriptor_shell_loss(x_pred_raw, target_radius=model.codec.descriptor_radius)
    total = total + lam_shell * shell
    out["shell"] = shell

    if gt_boundary_hr is not None:
        bce = boundary_bce_loss(pred["boundary_hr"], gt_boundary_hr)
        dice = boundary_dice_loss(pred["boundary_hr"], gt_boundary_hr)
        k = int(math.isqrt(pred["affinity_hr"].shape[1]))
        aff_tgt = local_boundary_pair_target(gt_boundary_hr, kernel_size=k)
        aff = F.binary_cross_entropy(pred["affinity_hr"].clamp(1e-6, 1 - 1e-6), aff_tgt.to(device=pred["affinity_hr"].device, dtype=pred["affinity_hr"].dtype))
        contrast = local_contrastive_descriptor_loss(x_pred, gt_boundary_hr, kernel_size=k)
        total = total + lam_boundary * bce + lam_dice * dice + lam_affinity * aff + lam_contrast * contrast
        out.update({"boundary": bce, "dice": dice, "affinity": aff, "contrast": contrast})

        if lam_cross_aff > 0:
            tau_cross = float(getattr(getattr(model, "lr_affinity", None), "tau", 0.6))
            aff_geo = descriptor_to_local_affinity(x_gt, kernel_size=k, tau=tau_cross)
            cross_aff = F.binary_cross_entropy(
                pred["affinity_hr"].clamp(1e-6, 1 - 1e-6),
                aff_geo.to(device=pred["affinity_hr"].device, dtype=pred["affinity_hr"].dtype),
            )
            total = total + lam_cross_aff * cross_aff
            out["cross_aff"] = cross_aff

        if lam_interior > 0:
            interior_mask = (1.0 - gt_boundary_hr).to(
                device=pred["boundary_hr"].device,
                dtype=pred["boundary_hr"].dtype,
            )
            interior = (
                pred["boundary_hr"].clamp(0.0, 1.0) * interior_mask
            ).sum() / interior_mask.sum().clamp_min(1.0)
            total = total + lam_interior * interior
            out["interior"] = interior

    out["total"] = total
    return out


# ============================================================
# 11) Demo
# ============================================================

def _demo() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    B, H, W = 2, 16, 20
    sr_scale = 4

    q_lr = rand_quaternion_grid(B, H, W, device=device, dtype=dtype)
    q_lr = q_lr.permute(0, 3, 1, 2).contiguous()

    model = SO3OQuotientSRNet4e6eE3NN(
        hidden_mul4=8,
        hidden_mul6=4,
        num_blocks_lr=1,
        num_blocks_hr=1,
        sr_scale=sr_scale,
        passive_input=False,
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    out = model(q_lr)
    print("descriptor_hr:", tuple(out["descriptor_hr"].shape))
    print("affinity_lr:", tuple(out["affinity_lr"].shape))
    print("affinity_hr:", tuple(out["affinity_hr"].shape))
    print("boundary_hr:", tuple(out["boundary_hr"].shape))

    q_dict, x_dict = model.codec.build_dictionary(n=2000, device=device, dtype=dtype)
    dec = model.forward_quaternion_nn(q_lr, q_dict=q_dict, x_dict=x_dict, topk=3)
    print("decoded quaternions:", tuple(dec.quaternions.shape))


if __name__ == "__main__":
    _demo()
