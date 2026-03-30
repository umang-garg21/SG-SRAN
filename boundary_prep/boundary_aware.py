from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from utils.symmetry_utils import resolve_symmetry
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
# 4) Unique cubic A1 seed in e3nn 4e basis
# ============================================================

def cubic_A1_seed_l4(dtype: torch.dtype = torch.float64) -> torch.Tensor:
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
# 5) Codec: quaternion <-> 4e cubic quotient descriptor
# ============================================================

@dataclass
class DecodeResult:
    quaternions: torch.Tensor
    descriptor_nn: torch.Tensor
    distances: torch.Tensor


class CubicQuotient4eCodec(nn.Module):
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
        D4 = D_from_quaternion_device(self.ir4, q)
        x4 = torch.einsum("...ij,j->...i", D4, self.a4)
        return x4

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
        dtype: torch.dtype = torch.float64,
        sampling: str = "random",
        fz_resolution: int = 3,
        fz_method: str = "cubochoric",
        fz_point_group: str = "O",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sampling = str(sampling).strip().lower()
        dev = device if device is not None else self.a4.device
        sampled_convention = "active"

        if sampling == "random":
            if n <= 0:
                raise ValueError("For sampling='random', n must be > 0.")
            q_dict = o3.rand_quaternion(n, device=dev, dtype=dtype)
            sampled_convention = "active"
        elif sampling == "fz":
            if int(fz_resolution) < 1:
                raise ValueError(f"fz_resolution must be >=1, got {fz_resolution}")
            try:
                import numpy as np
                from orix.sampling import get_sample_fundamental
            except Exception as exc:
                raise ImportError(
                    "FZ dictionary sampling requires `orix` and `numpy`."
                ) from exc

            try:
                point_group = resolve_symmetry(fz_point_group)
            except Exception as exc:
                raise ValueError(
                    f"Unknown point group '{fz_point_group}'. Examples: 'O', 'Oh', '432', 'fcc', 'm-3m'."
                ) from exc

            rot = get_sample_fundamental(
                int(fz_resolution),
                point_group=point_group,
                method=str(fz_method),
            )
            raw = np.asarray(getattr(rot, "data", rot), dtype=np.float64)
            if raw.ndim != 2:
                raw = raw.reshape(-1, 4)
            if raw.shape[-1] != 4 and raw.shape[0] == 4:
                raw = raw.T
            if raw.shape[-1] != 4:
                raise ValueError(f"Unexpected FZ quaternion shape: {tuple(raw.shape)}")

            q_all = torch.as_tensor(raw, device=dev, dtype=dtype)
            q_all = standardize_quaternion_sign(q_all)
            n_all = int(q_all.shape[0])
            if n_all == 0:
                raise RuntimeError("FZ sampler produced zero quaternions.")
            sampled_convention = "passive"

            if n <= 0:
                q_dict = q_all
            elif n <= n_all:
                idx = torch.randperm(n_all, device=dev)[:n]
                q_dict = q_all[idx]
            else:
                reps = n // n_all
                rem = n % n_all
                blocks = [q_all] * reps
                if rem > 0:
                    idx = torch.randperm(n_all, device=dev)[:rem]
                    blocks.append(q_all[idx])
                q_dict = torch.cat(blocks, dim=0)
        else:
            raise ValueError(f"Unknown sampling mode '{sampling}'. Use 'random' or 'fz'.")

        if sampled_convention == "active" and self.passive_input:
            q_dict = quaternion_inverse(normalize_quaternion(q_dict))
        elif sampled_convention == "passive" and not self.passive_input:
            q_dict = passive_to_active_quaternion(q_dict)

        q_dict = standardize_quaternion_sign(q_dict)
        x_dict = self.encode_quaternion(q_dict)
        return q_dict, x_dict

    def _refine_quaternion_chunk(
        self,
        x_target: torch.Tensor,
        q_init: torch.Tensor,
        steps: int = 4,
        lr: float = 1e-2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if steps <= 0:
            with torch.no_grad():
                x_out = self.encode_quaternion(q_init)
            return q_init, x_out

        x_target = x_target.detach()
        with torch.no_grad():
            q_best = q_init.detach().clone()
            x_best = self.encode_quaternion(q_best)
            err_best = ((x_best - x_target) ** 2).sum(dim=-1)

        q = q_best
        for _ in range(int(steps)):
            q = q.detach().requires_grad_(True)
            with torch.enable_grad():
                x_pred = self.encode_quaternion(q)
                loss = F.mse_loss(x_pred, x_target)
                grad = torch.autograd.grad(loss, q, create_graph=False, retain_graph=False)[0]

            with torch.no_grad():
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                gnorm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                q_cand = q - float(lr) * (grad / gnorm)
                q_cand = torch.nan_to_num(q_cand, nan=0.0, posinf=0.0, neginf=0.0)
                qnorm = q_cand.norm(dim=-1, keepdim=True)
                q_cand = q_cand / qnorm.clamp_min(1e-12)
                bad = qnorm.squeeze(-1) < 1e-10
                if bad.any():
                    q_cand = q_cand.clone()
                    q_cand[bad] = q_cand.new_tensor([1.0, 0.0, 0.0, 0.0])
                q_cand = standardize_quaternion_sign(q_cand)

                x_cand = self.encode_quaternion(q_cand)
                err_cand = ((x_cand - x_target) ** 2).sum(dim=-1)
                better = err_cand < err_best

                q_best = torch.where(better.unsqueeze(-1), q_cand, q_best)
                x_best = torch.where(better.unsqueeze(-1), x_cand, x_best)
                err_best = torch.where(better, err_cand, err_best)
                q = q_best

        return q_best.detach(), x_best.detach()

    @torch.no_grad()
    def decode_by_dictionary(
        self,
        x: torch.Tensor,
        q_dict: torch.Tensor,
        x_dict: Optional[torch.Tensor] = None,
        chunk: int = 4096,
        topk: int = 1,
        refine_steps: int = 0,
        refine_lr: float = 1e-2,
    ) -> DecodeResult:
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, 9)

        if x_dict is None:
            x_dict = self.encode_quaternion(q_dict)

        x_dict = x_dict.to(device=x_flat.device, dtype=x_flat.dtype)
        q_dict = q_dict.to(device=x_flat.device, dtype=x_flat.dtype)
        x_dict_n = F.normalize(x_dict, dim=-1, eps=1e-12)

        n_total = x_flat.shape[0]
        q_out = torch.empty(n_total, 4, device=x_flat.device, dtype=x_flat.dtype)
        x_out = torch.empty(n_total, 9, device=x_flat.device, dtype=x_flat.dtype)
        best_dist = torch.empty(n_total, device=x_flat.device, dtype=x_flat.dtype)

        k = max(1, int(topk))
        k = min(k, int(x_dict.shape[0]))
        refine_steps = max(0, int(refine_steps))

        for start in range(0, x_flat.shape[0], chunk):
            stop = min(start + chunk, x_flat.shape[0])
            xi = x_flat[start:stop]

            xi_n = F.normalize(xi, dim=-1, eps=1e-12)
            sims = xi_n @ x_dict_n.T
            _, top_idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)

            if k == 1:
                init_idx = top_idx[:, 0]
            else:
                xk = x_dict[top_idx]
                d2k = ((xk - xi.unsqueeze(1)) ** 2).sum(dim=-1)
                j = torch.argmin(d2k, dim=1)
                init_idx = top_idx[torch.arange(xi.shape[0], device=xi.device), j]

            q_init = q_dict[init_idx]

            if refine_steps > 0:
                q_chunk, x_chunk = self._refine_quaternion_chunk(
                    x_target=xi,
                    q_init=q_init,
                    steps=refine_steps,
                    lr=refine_lr,
                )
            else:
                q_chunk = q_init
                x_chunk = x_dict[init_idx]

            d_chunk = torch.norm(x_chunk - xi, dim=-1)
            q_out[start:stop] = q_chunk
            x_out[start:stop] = x_chunk
            best_dist[start:stop] = d_chunk

        q_nn = q_out.reshape(*shape, 4)
        x_nn = x_out.reshape(*shape, 9)
        d_nn = best_dist.reshape(*shape)
        return DecodeResult(quaternions=q_nn, descriptor_nn=x_nn, distances=d_nn)


# ============================================================
# 6) Fiber-preserving operators and affinity helpers
# ============================================================

class SharedOverMConv2d(nn.Module):
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
        x = x.view(B, self.mul_in, self.d, H, W).permute(0, 2, 1, 3, 4).reshape(B * self.d, self.mul_in, H, W)
        y = self.conv(x)
        H2, W2 = y.shape[-2:]
        y = y.view(B, self.d, self.mul_out, H2, W2).permute(0, 2, 1, 3, 4).reshape(B, self.mul_out * self.d, H2, W2)
        return y


class SharedOverMInterpolation(nn.Module):
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
        x = x.view(B, self.mul, self.d, H, W).permute(0, 2, 1, 3, 4).reshape(B * self.d, self.mul, H, W)
        if self.mode in ("bilinear", "bicubic"):
            y = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        else:
            y = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        H2, W2 = y.shape[-2:]
        y = y.view(B, self.d, self.mul, H2, W2).permute(0, 2, 1, 3, 4).reshape(B, self.mul * self.d, H2, W2)
        return y


class MaskedSharedOverMConv2d(nn.Module):
    def __init__(
        self,
        mul_in: int,
        mul_out: int,
        l: int,
        kernel_size: int = 3,
        bias: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.mul_in = mul_in
        self.mul_out = mul_out
        self.l = l
        self.d = 2 * l + 1
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.k2 = kernel_size * kernel_size
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(mul_out, mul_in, self.k2))
        if bias:
            self.bias = nn.Parameter(torch.zeros(mul_out))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = mul_in * self.k2
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, aff: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.mul_in * self.d, f"Expected {self.mul_in*self.d} channels, got {C}"
        assert aff.shape == (B, self.k2, H, W), f"Expected aff {(B,self.k2,H,W)}, got {tuple(aff.shape)}"

        x = x.view(B, self.mul_in, self.d, H, W).permute(0, 2, 1, 3, 4).reshape(B * self.d, self.mul_in, H, W)
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        L = patches.shape[-1]
        patches = patches.view(B, self.d, self.mul_in, self.k2, L)

        aff_flat = aff.view(B, 1, 1, self.k2, L)

        valid = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
        valid = F.unfold(valid, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        valid = valid.view(B, 1, 1, self.k2, L)

        masked_aff = aff_flat * valid
        masked = patches * masked_aff
        y = torch.einsum("bdikl,oik->bdol", masked, self.weight)
        denom = masked_aff.sum(dim=3).clamp_min(self.eps)
        y = y * (self.k2 / denom)

        if self.bias is not None:
            y = y + self.bias.view(1, 1, self.mul_out, 1)

        y = y.view(B, self.d, self.mul_out, H, W).permute(0, 2, 1, 3, 4).reshape(B, self.mul_out * self.d, H, W)
        return y


def affinity_to_boundary(aff: torch.Tensor, center_idx: int) -> torch.Tensor:
    B, K2, H, W = aff.shape
    idx = [k for k in range(K2) if k != center_idx]
    if len(idx) == 0:
        return torch.zeros(B, 1, H, W, device=aff.device, dtype=aff.dtype)
    boundary = 1.0 - aff[:, idx].mean(dim=1, keepdim=True)
    return boundary.clamp(0.0, 1.0)


def local_boundary_pair_target(gt_boundary: torch.Tensor, kernel_size: int = 3, sharpness: float = 6.0) -> torch.Tensor:
    assert gt_boundary.ndim == 4 and gt_boundary.shape[1] == 1
    B, _, H, W = gt_boundary.shape
    k2 = kernel_size * kernel_size
    center_idx = k2 // 2
    patches = F.unfold(gt_boundary, kernel_size=kernel_size, padding=kernel_size // 2)
    patches = patches.view(B, 1, k2, H * W)
    center = patches[:, :, center_idx:center_idx + 1, :]
    pair_b = torch.maximum(center, patches)
    tgt = torch.exp(-sharpness * pair_b).view(B, k2, H, W)

    center = torch.zeros_like(tgt)
    center[:, center_idx:center_idx + 1] = 1.0
    neigh_mask = torch.ones(1, k2, 1, 1, device=tgt.device, dtype=tgt.dtype)
    neigh_mask[:, center_idx:center_idx + 1] = 0.0
    tgt = tgt * neigh_mask + center
    return tgt.clamp(0.0, 1.0)


class BoundaryAffinityHead46(nn.Module):
    def __init__(
        self,
        mul4: int,
        mul6: int,
        kernel_size: int = 3,
        hidden: Optional[int] = None,
        use_boundary_prior: bool = True,
        alpha_init: float = 8.0,
        beta_init: float = 4.0,
        gamma_init: float = 2.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.mul4 = mul4
        self.mul6 = mul6
        self.kernel_size = kernel_size
        self.k2 = kernel_size * kernel_size
        self.center_idx = self.k2 // 2
        self.use_boundary_prior = use_boundary_prior

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

        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def descriptor_affinity_logits(self, x_desc: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_desc.shape
        assert C == 9, f"Expected [B,9,H,W], got {x_desc.shape}"

        patches = F.unfold(x_desc, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.view(B, 9, self.k2, H * W)
        center = patches[:, :, self.center_idx:self.center_idx + 1, :]
        d2 = ((patches - center) ** 2).sum(dim=1).view(B, self.k2, H, W)

        alpha = self.log_alpha.exp().to(dtype=x_desc.dtype, device=x_desc.device)
        beta = self.beta.to(dtype=x_desc.dtype, device=x_desc.device)
        logits = -alpha * d2 + beta
        return logits

    def boundary_pair_penalty(self, boundary_prob: torch.Tensor) -> torch.Tensor:
        B, _, H, W = boundary_prob.shape
        patches = F.unfold(boundary_prob, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.view(B, 1, self.k2, H * W)
        center = patches[:, :, self.center_idx:self.center_idx + 1, :]
        pair_b = torch.maximum(center, patches).view(B, self.k2, H, W)
        return pair_b

    def forward(
        self,
        x4: torch.Tensor,
        x6: torch.Tensor,
        x_desc: torch.Tensor,
        boundary_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C4, H, W = x4.shape
        B2, C6, H2, W2 = x6.shape
        assert B == B2 and H == H2 and W == W2
        assert C4 == self.mul4 * 9, f"Expected {self.mul4*9} channels for x4, got {C4}"
        assert C6 == self.mul6 * 13, f"Expected {self.mul6*13} channels for x6, got {C6}"
        assert x_desc.shape == (B, 9, H, W), f"Expected x_desc {(B,9,H,W)}, got {tuple(x_desc.shape)}"

        x4g = x4.view(B, self.mul4, 9, H, W)
        x6g = x6.view(B, self.mul6, 13, H, W)
        n4 = torch.sqrt((x4g ** 2).sum(dim=2).clamp_min(1e-12))
        n6 = torch.sqrt((x6g ** 2).sum(dim=2).clamp_min(1e-12))

        if self.use_boundary_prior:
            if boundary_prior is None:
                boundary_prior = torch.zeros(B, 1, H, W, device=x4.device, dtype=x4.dtype)
            inp = torch.cat([n4, n6, boundary_prior], dim=1)
        else:
            inp = torch.cat([n4, n6], dim=1)

        h = self.trunk(inp)
        learned_logits = self.aff_head(h)
        boundary_logit = self.boundary_head(h)
        boundary_prob = torch.sigmoid(boundary_logit)

        geom_logits = self.descriptor_affinity_logits(x_desc)
        pair_penalty = self.boundary_pair_penalty(boundary_prob)
        gamma = self.gamma.to(dtype=x4.dtype, device=x4.device)
        logits = geom_logits + learned_logits - gamma * pair_penalty
        aff = torch.sigmoid(logits)

        center = torch.zeros_like(aff)
        center[:, self.center_idx:self.center_idx + 1] = 1.0
        neigh_mask = torch.ones(1, self.k2, 1, 1, device=aff.device, dtype=aff.dtype)
        neigh_mask[:, self.center_idx:self.center_idx + 1] = 0.0
        aff = aff * neigh_mask + center
        return aff, boundary_prob, boundary_logit


# ============================================================
# 7) 4e/6e lifting, gating, and masked residual blocks
# ============================================================

class PerPixelSelfTensorLift4to6(nn.Module):
    def __init__(self, mul6: int):
        super().__init__()
        self.mul6 = mul6
        self.tp = o3.FullyConnectedTensorProduct(
            "1x4e",
            "1x4e",
            f"{mul6}x6e",
            shared_weights=True,
            internal_weights=True,
        )

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x4.shape
        assert C == 9, f"Expected [B,9,H,W], got {x4.shape}"
        x = x4.permute(0, 2, 3, 1).contiguous().view(-1, 9)
        y = self.tp(x, x)
        y = y.view(B, H, W, self.mul6 * 13).permute(0, 3, 1, 2).contiguous()
        return y


class PerPixelIrrepLinear(nn.Module):
    """
    Per-pixel equivariant linear map between irrep fibers.

    Input/output layout:
        x: [B, C, H, W]
    where C is the total irrep dimension.
    """

    def __init__(self, irreps_in: str, irreps_out: str):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.linear = o3.Linear(
            self.irreps_in,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        self.dim_in = self.irreps_in.dim
        self.dim_out = self.irreps_out.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.dim_in, f"Expected {self.dim_in} channels, got {C}"
        y = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        y = self.linear(y)
        y = y.view(B, H, W, self.dim_out).permute(0, 3, 1, 2).contiguous()
        return y


class DescriptorHead46(nn.Module):
    """
    Direct 6e -> 4e feedback before predicting a 1x4e descriptor.
    """

    def __init__(self, mul4: int, mul6: int):
        super().__init__()
        self.mul4 = mul4
        self.mul6 = mul6
        self.fuse6_to4 = PerPixelIrrepLinear(f"{mul6}x6e", f"{mul4}x4e")
        self.mix4 = SharedOverMConv2d(mul4, mul4, l=4, kernel_size=1, padding=0)
        self.out4 = SharedOverMConv2d(mul4, 1, l=4, kernel_size=1, padding=0)

    def forward(
        self,
        x4: torch.Tensor,
        x6: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        target_radius: Optional[float] = None,
    ) -> torch.Tensor:
        z4 = x4 + self.fuse6_to4(x6)
        z4 = self.mix4(z4)
        y = self.out4(z4)
        if skip is not None:
            y = y + skip
        if target_radius is not None:
            radius = torch.as_tensor(target_radius, device=y.device, dtype=y.dtype)
            ynorm = torch.sqrt((y ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12))
            y = radius * y / ynorm
        return y


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
        B, C4, H, W = x4.shape
        B2, C6, H2, W2 = x6.shape
        assert B == B2 and H == H2 and W == W2
        assert C4 == self.mul4 * 9, f"Expected {self.mul4*9} channels for x4, got {C4}"
        assert C6 == self.mul6 * 13, f"Expected {self.mul6*13} channels for x6, got {C6}"

        x4g = x4.view(B, self.mul4, 9, H, W)
        x6g = x6.view(B, self.mul6, 13, H, W)
        n4 = torch.sqrt((x4g ** 2).sum(dim=2).clamp_min(1e-12))
        n6 = torch.sqrt((x6g ** 2).sum(dim=2).clamp_min(1e-12))

        gates = self.net(torch.cat([n4, n6], dim=1))
        g4 = gates[:, :self.mul4]
        g6 = gates[:, self.mul4:]
        y4 = (x4g * g4.unsqueeze(2)).view(B, C4, H, W)
        y6 = (x6g * g6.unsqueeze(2)).view(B, C6, H, W)
        return y4, y6


class MaskedResidual46Block(nn.Module):
    def __init__(self, mul4: int, mul6: int, kernel_size: int = 3):
        super().__init__()
        self.conv4_1 = MaskedSharedOverMConv2d(mul4, mul4, l=4, kernel_size=kernel_size)
        self.conv6_1 = MaskedSharedOverMConv2d(mul6, mul6, l=6, kernel_size=kernel_size)
        self.gate = NormGate46(mul4, mul6)
        self.conv4_2 = MaskedSharedOverMConv2d(mul4, mul4, l=4, kernel_size=kernel_size)
        self.conv6_2 = MaskedSharedOverMConv2d(mul6, mul6, l=6, kernel_size=kernel_size)

    def forward(self, x4: torch.Tensor, x6: torch.Tensor, aff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y4 = self.conv4_1(x4, aff)
        y6 = self.conv6_1(x6, aff)
        y4, y6 = self.gate(y4, y6)
        y4 = self.conv4_2(y4, aff)
        y6 = self.conv6_2(y6, aff)
        return x4 + y4, x6 + y6


class MaskedResidual46Stack(nn.Module):
    def __init__(self, mul4: int, mul6: int, num_blocks: int, kernel_size: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([MaskedResidual46Block(mul4, mul6, kernel_size=kernel_size) for _ in range(num_blocks)])

    def forward(self, x4: torch.Tensor, x6: torch.Tensor, aff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x4, x6 = block(x4, x6, aff)
        return x4, x6


# ============================================================
# 8) Mask-aware upsampler
# ============================================================

class DualFiberMaskedUpsampler(nn.Module):
    """
    Nearest upsample -> provisional HR affinity -> provisional HR descriptor
    -> recomputed HR affinity -> masked refinement.

    This keeps the HR geometric affinity term from being locked to the
    blocky nearest-upsampled LR descriptor.
    """

    def __init__(
        self,
        mul4: int,
        mul6: int,
        sr_scale: int,
        kernel_size: int = 3,
        num_refine_blocks: int = 2,
        use_boundary_prior: bool = True,
    ):
        super().__init__()
        self.up4 = SharedOverMInterpolation(mul4, l=4, scale_factor=sr_scale, mode="nearest")
        self.up6 = SharedOverMInterpolation(mul6, l=6, scale_factor=sr_scale, mode="nearest")
        self.up_desc = SharedOverMInterpolation(1, l=4, scale_factor=sr_scale, mode="nearest")

        self.aff_head = BoundaryAffinityHead46(
            mul4=mul4,
            mul6=mul6,
            kernel_size=kernel_size,
            use_boundary_prior=use_boundary_prior,
        )
        self.mix4 = MaskedSharedOverMConv2d(mul4, mul4, l=4, kernel_size=kernel_size)
        self.mix6 = MaskedSharedOverMConv2d(mul6, mul6, l=6, kernel_size=kernel_size)
        self.provisional_desc = DescriptorHead46(mul4, mul6)
        self.refine = MaskedResidual46Stack(mul4, mul6, num_blocks=num_refine_blocks, kernel_size=kernel_size)

    def forward(
        self,
        x4: torch.Tensor,
        x6: torch.Tensor,
        x_desc: torch.Tensor,
        hr_boundary_prior: Optional[torch.Tensor] = None,
        target_radius: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x4u = self.up4(x4)
        x6u = self.up6(x6)
        x_desc_u = self.up_desc(x_desc)

        aff_hr0, _, _ = self.aff_head(
            x4u,
            x6u,
            x_desc=x_desc_u,
            boundary_prior=hr_boundary_prior,
        )

        x4u = self.mix4(x4u, aff_hr0)
        x6u = self.mix6(x6u, aff_hr0)

        x_desc_hr = self.provisional_desc(
            x4u,
            x6u,
            skip=x_desc_u,
            target_radius=target_radius,
        )

        aff_hr, boundary_hr, boundary_hr_logit = self.aff_head(
            x4u,
            x6u,
            x_desc=x_desc_hr,
            boundary_prior=hr_boundary_prior,
        )
        x4u, x6u = self.refine(x4u, x6u, aff_hr)

        x_desc_refined = self.provisional_desc(
            x4u,
            x6u,
            skip=x_desc_u,
            target_radius=target_radius,
        )
        return x4u, x6u, x_desc_refined, aff_hr, boundary_hr, boundary_hr_logit


# ============================================================
# 9) Boundary-aware 4e + 6e SR model
# ============================================================

class SO3OQuotientSRNet4e6e(nn.Module):
    def __init__(
        self,
        hidden_mul4: int = 16,
        hidden_mul6: int = 8,
        num_blocks_pre: int = 3,
        num_blocks_post: int = 3,
        sr_scale: int = 2,
        passive_input: bool = True,
        dtype: torch.dtype = torch.float64,
        affinity_kernel_size: int = 3,
        use_boundary_prior: bool = True,
        upsample_mode: str = "masked_nearest",
    ):
        super().__init__()
        self.sr_scale = sr_scale
        self.hidden_mul4 = hidden_mul4
        self.hidden_mul6 = hidden_mul6
        self.upsample_mode = str(upsample_mode).lower()

        self.codec = CubicQuotient4eCodec(passive_input=passive_input, dtype=dtype)

        self.in4 = SharedOverMConv2d(mul_in=1, mul_out=hidden_mul4, l=4, kernel_size=3, padding=1)
        self.in6 = PerPixelSelfTensorLift4to6(hidden_mul6)

        self.aff_lr_head = BoundaryAffinityHead46(
            mul4=hidden_mul4,
            mul6=hidden_mul6,
            kernel_size=affinity_kernel_size,
            use_boundary_prior=use_boundary_prior,
        )
        self.pre = MaskedResidual46Stack(hidden_mul4, hidden_mul6, num_blocks=num_blocks_pre, kernel_size=affinity_kernel_size)

        if self.upsample_mode == "masked_nearest":
            self.up = DualFiberMaskedUpsampler(
                hidden_mul4,
                hidden_mul6,
                sr_scale=sr_scale,
                kernel_size=affinity_kernel_size,
                num_refine_blocks=max(1, num_blocks_post),
                use_boundary_prior=use_boundary_prior,
            )
            self.post = None
        else:
            self.up4 = SharedOverMInterpolation(hidden_mul4, l=4, scale_factor=sr_scale, mode=self.upsample_mode)
            self.up6 = SharedOverMInterpolation(hidden_mul6, l=6, scale_factor=sr_scale, mode=self.upsample_mode)
            self.up_desc = SharedOverMInterpolation(1, l=4, scale_factor=sr_scale, mode=self.upsample_mode)
            self.aff_hr_head = BoundaryAffinityHead46(
                mul4=hidden_mul4,
                mul6=hidden_mul6,
                kernel_size=affinity_kernel_size,
                use_boundary_prior=use_boundary_prior,
            )
            self.hr_mix4 = MaskedSharedOverMConv2d(hidden_mul4, hidden_mul4, l=4, kernel_size=affinity_kernel_size)
            self.hr_mix6 = MaskedSharedOverMConv2d(hidden_mul6, hidden_mul6, l=6, kernel_size=affinity_kernel_size)
            self.provisional_desc = DescriptorHead46(hidden_mul4, hidden_mul6)
            self.post = MaskedResidual46Stack(hidden_mul4, hidden_mul6, num_blocks=num_blocks_post, kernel_size=affinity_kernel_size)

        self.out_desc = DescriptorHead46(hidden_mul4, hidden_mul6)
        self.skip_up = SharedOverMInterpolation(mul=1, l=4, scale_factor=sr_scale, mode="nearest")

    def encode(self, q_lr: torch.Tensor) -> torch.Tensor:
        return self.codec.encode_map(q_lr)

    def forward(self, q_lr: torch.Tensor, lr_boundary_prior: Optional[torch.Tensor] = None) -> dict:
        x0 = self.encode(q_lr)
        x4 = self.in4(x0)
        x6 = self.in6(x0)

        aff_lr, boundary_lr, boundary_lr_logit = self.aff_lr_head(
            x4,
            x6,
            x_desc=x0,
            boundary_prior=lr_boundary_prior,
        )
        x4, x6 = self.pre(x4, x6, aff_lr)

        hr_boundary_prior = None
        if lr_boundary_prior is not None:
            hr_boundary_prior = F.interpolate(lr_boundary_prior, scale_factor=self.sr_scale, mode="nearest")

        if self.upsample_mode == "masked_nearest":
            x4, x6, x_desc_hr, aff_hr, boundary_hr, boundary_hr_logit = self.up(
                x4,
                x6,
                x_desc=x0,
                hr_boundary_prior=hr_boundary_prior,
                target_radius=self.codec.descriptor_radius,
            )
        else:
            x4 = self.up4(x4)
            x6 = self.up6(x6)
            x_desc_hr0 = self.up_desc(x0)
            aff_hr0, _, _ = self.aff_hr_head(
                x4,
                x6,
                x_desc=x_desc_hr0,
                boundary_prior=hr_boundary_prior,
            )
            x4 = self.hr_mix4(x4, aff_hr0)
            x6 = self.hr_mix6(x6, aff_hr0)
            x_desc_hr = self.provisional_desc(
                x4,
                x6,
                skip=x_desc_hr0,
                target_radius=self.codec.descriptor_radius,
            )
            aff_hr, boundary_hr, boundary_hr_logit = self.aff_hr_head(
                x4,
                x6,
                x_desc=x_desc_hr,
                boundary_prior=hr_boundary_prior,
            )
            x4, x6 = self.post(x4, x6, aff_hr)
            x_desc_hr = self.provisional_desc(
                x4,
                x6,
                skip=x_desc_hr0,
                target_radius=self.codec.descriptor_radius,
            )

        y_raw = self.out_desc(x4, x6, skip=self.skip_up(x0), target_radius=None)
        target_norm = self.codec.a4.norm()
        ynorm = torch.sqrt((y_raw ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12))
        y = target_norm * y_raw / ynorm

        return {
            "descriptor_hr": y,
            "descriptor_hr_raw": y_raw,
            "affinity_lr": aff_lr,
            "affinity_hr": aff_hr,
            "boundary_lr": boundary_lr,
            "boundary_hr": boundary_hr,
            "boundary_lr_logit": boundary_lr_logit,
            "boundary_hr_logit": boundary_hr_logit,
            "boundary_lr_from_affinity": affinity_to_boundary(aff_lr, self.aff_lr_head.center_idx),
            "boundary_hr_from_affinity": affinity_to_boundary(aff_hr, self.aff_lr_head.center_idx),
            "provisional_descriptor_hr": x_desc_hr,
        }

    def forward_descriptor(self, q_lr: torch.Tensor, lr_boundary_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(q_lr, lr_boundary_prior=lr_boundary_prior)["descriptor_hr"]

    @torch.no_grad()
    def forward_quaternion_nn(
        self,
        q_lr: torch.Tensor,
        q_dict: torch.Tensor,
        x_dict: Optional[torch.Tensor] = None,
        chunk: int = 4096,
        topk: int = 1,
        refine_steps: int = 0,
        refine_lr: float = 1e-2,
        lr_boundary_prior: Optional[torch.Tensor] = None,
    ) -> DecodeResult:
        x_hr = self.forward_descriptor(q_lr, lr_boundary_prior=lr_boundary_prior)
        x_hr_last = x_hr.permute(0, 2, 3, 1).contiguous()
        return self.codec.decode_by_dictionary(
            x_hr_last,
            q_dict=q_dict,
            x_dict=x_dict,
            chunk=chunk,
            topk=topk,
            refine_steps=refine_steps,
            refine_lr=refine_lr,
        )


# ============================================================
# 10) Losses
# ============================================================

def descriptor_mse_loss(x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_pred, x_gt)


def descriptor_shell_loss(x_pred: torch.Tensor, target_radius: float = 1.0) -> torch.Tensor:
    r = torch.sqrt((x_pred ** 2).sum(dim=1).clamp_min(1e-12))
    return ((r - target_radius) ** 2).mean()


def boundary_bce_loss(pred_boundary: torch.Tensor, gt_boundary: torch.Tensor) -> torch.Tensor:
    gt_boundary = gt_boundary.to(dtype=pred_boundary.dtype, device=pred_boundary.device)
    return F.binary_cross_entropy(pred_boundary.clamp(1e-6, 1.0 - 1e-6), gt_boundary)


def boundary_dice_loss(pred_boundary: torch.Tensor, gt_boundary: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gt_boundary = gt_boundary.to(dtype=pred_boundary.dtype, device=pred_boundary.device)
    pred = pred_boundary.reshape(pred_boundary.shape[0], -1)
    tgt = gt_boundary.reshape(gt_boundary.shape[0], -1)
    inter = (pred * tgt).sum(dim=1)
    denom = pred.sum(dim=1) + tgt.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def affinity_bce_from_boundary_loss(
    pred_affinity: torch.Tensor,
    gt_boundary: torch.Tensor,
    kernel_size: int = 3,
    sharpness: float = 6.0,
) -> torch.Tensor:
    tgt_aff = local_boundary_pair_target(gt_boundary, kernel_size=kernel_size, sharpness=sharpness)
    return F.binary_cross_entropy(pred_affinity.clamp(1e-6, 1.0 - 1e-6), tgt_aff)


def combined_boundary_aware_quotient_loss(
    model: SO3OQuotientSRNet4e6e,
    pred: dict,
    hr_q_gt: torch.Tensor,
    gt_boundary_hr: Optional[torch.Tensor] = None,
    lam_shell: float = 1e-2,
    lam_boundary_bce: float = 1.0,
    lam_boundary_dice: float = 0.5,
    lam_affinity: float = 0.25,
) -> dict:
    with torch.no_grad():
        x_gt = model.codec.encode_map(hr_q_gt)

    x_pred = pred["descriptor_hr"]
    x_pred_raw = pred.get("descriptor_hr_raw", x_pred)
    target_radius = model.codec.descriptor_radius

    loss_desc = descriptor_mse_loss(x_pred, x_gt)
    loss_shell = descriptor_shell_loss(x_pred_raw, target_radius=target_radius)
    total = loss_desc + lam_shell * loss_shell

    out = {
        "total": total,
        "descriptor": loss_desc,
        "shell": loss_shell,
    }

    if gt_boundary_hr is not None:
        loss_b_bce = boundary_bce_loss(pred["boundary_hr"], gt_boundary_hr)
        loss_b_dice = boundary_dice_loss(pred["boundary_hr"], gt_boundary_hr)
        k = int(round(math.sqrt(pred["affinity_hr"].shape[1])))
        loss_aff = affinity_bce_from_boundary_loss(pred["affinity_hr"], gt_boundary_hr, kernel_size=k)
        total = total + lam_boundary_bce * loss_b_bce + lam_boundary_dice * loss_b_dice + lam_affinity * loss_aff
        out.update({
            "boundary_bce": loss_b_bce,
            "boundary_dice": loss_b_dice,
            "affinity": loss_aff,
            "total": total,
        })

    return out


# ============================================================
# 11) Debug / sanity utilities
# ============================================================

@torch.no_grad()
def apply_global_specimen_rotation_to_descriptor(x: torch.Tensor, q_rot_active: torch.Tensor) -> torch.Tensor:
    ir4 = o3.Irrep("4e")
    D4 = D_from_quaternion_device(ir4, normalize_quaternion(q_rot_active))
    if D4.ndim == 2:
        D4 = D4.unsqueeze(0).expand(x.shape[0], -1, -1)
    y = torch.einsum("bij,bjhw->bihw", D4.to(x.dtype).to(x.device), x)
    return y


# ============================================================
# 12) Demo
# ============================================================

def _demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    B, H, W = 2, 24, 32
    sr_scale = 4
    sH, sW = H * sr_scale, W * sr_scale

    q_lr = rand_quaternion_grid(B, H, W, device=device, dtype=dtype)
    q_lr = q_lr.permute(0, 3, 1, 2).contiguous()
    lr_boundary_prior = torch.rand(B, 1, H, W, device=device, dtype=dtype)

    model = SO3OQuotientSRNet4e6e(
        hidden_mul4=12,
        hidden_mul6=6,
        num_blocks_pre=2,
        num_blocks_post=2,
        sr_scale=sr_scale,
        passive_input=False,
        dtype=dtype,
        affinity_kernel_size=3,
        use_boundary_prior=True,
        upsample_mode="masked_nearest",
    ).to(device=device, dtype=dtype)

    out = model(q_lr, lr_boundary_prior=lr_boundary_prior)
    print("descriptor_hr:", tuple(out["descriptor_hr"].shape))
    print("affinity_lr  :", tuple(out["affinity_lr"].shape))
    print("affinity_hr  :", tuple(out["affinity_hr"].shape))
    print("boundary_lr  :", tuple(out["boundary_lr"].shape))
    print("boundary_hr  :", tuple(out["boundary_hr"].shape))

    hr_q_gt = rand_quaternion_grid(B, sH, sW, device=device, dtype=dtype)
    hr_q_gt = hr_q_gt.permute(0, 3, 1, 2).contiguous()
    gt_boundary_hr = torch.rand(B, 1, sH, sW, device=device, dtype=dtype)

    losses = combined_boundary_aware_quotient_loss(
        model=model,
        pred=out,
        hr_q_gt=hr_q_gt,
        gt_boundary_hr=gt_boundary_hr,
    )
    for k, v in losses.items():
        print(f"{k:>14s}: {float(v.item()):.6f}")

    q_dict, x_dict = model.codec.build_dictionary(n=10000, device=device, dtype=dtype)
    dec = model.forward_quaternion_nn(q_lr, q_dict=q_dict, x_dict=x_dict, chunk=2048)
    print("decoded quaternions:", tuple(dec.quaternions.shape))
    print("mean NN distance    :", float(dec.distances.mean().item()))

    qg = o3.rand_quaternion(1, device=device, dtype=dtype)[0]
    x_lr = model.encode(q_lr)
    x_rot = apply_global_specimen_rotation_to_descriptor(x_lr, qg)

    qg_expand = qg.view(1, 1, 1, 4).expand(B, H, W, 4)
    q_lr_last = q_lr.permute(0, 2, 3, 1).contiguous()
    q_lr_rot = quaternion_multiply(qg_expand, q_lr_last)
    q_lr_rot = normalize_quaternion(q_lr_rot)
    q_lr_rot = q_lr_rot.permute(0, 3, 1, 2).contiguous()

    x_lr_direct = model.encode(q_lr_rot)
    err = torch.norm(x_rot - x_lr_direct) / torch.norm(x_lr_direct)
    print("Relative descriptor equivariance error:", float(err.item()))


if __name__ == "__main__":
    _demo()