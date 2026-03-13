# so3_o_cubic_sr_4e6e_full.py
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from utils.symmetry_utils import resolve_symmetry

# private e3nn helpers used only to make Wigner-D device-safe on GPU
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
    If q_passive is a passive specimen->crystal orientation,
    convert to the corresponding active rotation.
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
    """
    Device-safe Wigner D matrix for proper rotations.
    """
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
    """
    Device-safe replacement for irrep.D_from_quaternion(q) for proper rotations.
    """
    q = normalize_quaternion(q)
    R = o3.quaternion_to_matrix(q)
    alpha, beta, gamma = _rotation.matrix_to_angles(R)
    D = wigner_D_device(irrep.l, alpha, beta, gamma)

    # proper rotations only -> parity factor is 1
    return D


# ============================================================
# 3) Proper cubic group O (24 rotations)
# ============================================================

def proper_cubic_group_O(
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    24 proper cubic rotations as signed permutation matrices with det=+1.
    Returns [24, 3, 3]
    """
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
    """
    Compute the unique cubic-invariant seed a4 in the e3nn 4e basis.

    Computed on CPU for robustness; the module buffer is moved later by model.to(...).
    Returns [9]
    """
    cpu = torch.device("cpu")
    G = proper_cubic_group_O(device=cpu, dtype=dtype)
    ir4 = o3.Irrep("4e")

    Dg = ir4.D_from_matrix(G)  # CPU-safe
    P = Dg.mean(dim=0)
    P = 0.5 * (P + P.T)

    evals, evecs = torch.linalg.eigh(P)
    idx = torch.argmax(evals)
    a4 = evecs[:, idx]
    a4 = a4 / a4.norm()

    # deterministic sign
    j = torch.argmax(torch.abs(a4))
    if a4[j] < 0:
        a4 = -a4
    return a4


# ============================================================
# 5) Codec: quaternion <-> 4e cubic quotient descriptor
# ============================================================

@dataclass
class DecodeResult:
    quaternions: torch.Tensor      # [..., 4]
    descriptor_nn: torch.Tensor    # [..., 9]
    distances: torch.Tensor        # [...]


class CubicQuotient4eCodec(nn.Module):
    """
    Encode quaternions into the cubic quotient descriptor:
        phi(q) = D^4(q_active) a4

    If passive_input=True, input quaternions are interpreted as passive
    specimen->crystal and inverted before applying D^4.
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
        """
        q: [..., 4]
        returns: [..., 9]
        """
        q = self.to_active(q)
        D4 = D_from_quaternion_device(self.ir4, q)         # [..., 9, 9]
        x4 = torch.einsum("...ij,j->...i", D4, self.a4)    # [..., 9]
        return x4

    def encode_map(self, q_map: torch.Tensor) -> torch.Tensor:
        """
        q_map: [B, 4, H, W]
        returns: [B, 9, H, W]
        """
        B, C, H, W = q_map.shape
        assert C == 4, f"Expected [B,4,H,W], got {q_map.shape}"
        q = q_map.permute(0, 2, 3, 1).contiguous()         # [B,H,W,4]
        x = self.encode_quaternion(q)                      # [B,H,W,9]
        return x.permute(0, 3, 1, 2).contiguous()          # [B,9,H,W]

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
        """
        Build descriptor decode dictionary.

        Parameters
        ----------
        n : int
            Requested dictionary size. For `sampling="fz"`, set `n<=0` to use
            all FZ samples generated at `fz_resolution`.
        sampling : {"random", "fz"}
            - "random": sample quaternions uniformly in SO(3) via e3nn.
            - "fz": sample quaternions from ORIX fundamental zone.
        fz_resolution : int
            ORIX FZ sampling resolution (used when `sampling="fz"`).
        fz_method : str
            ORIX FZ sampling method (e.g. "cubochoric").
        fz_point_group : str
            ORIX symmetry name (e.g. "O", "Oh").
        """
        sampling = str(sampling).strip().lower()
        dev = device if device is not None else self.a4.device
        sampled_convention = "active"

        if sampling == "random":
            if n <= 0:
                raise ValueError("For sampling='random', n must be > 0.")
            # e3nn random quaternions are active-rotation convention.
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
                    f"Unknown point group '{fz_point_group}'. "
                    "Examples: 'O', 'Oh', '432', 'fcc', 'm-3m'."
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
            # ORIX fundamental-zone quaternions follow passive convention.
            sampled_convention = "passive"

            if n <= 0:
                q_dict = q_all
            elif n <= n_all:
                # Subsample without replacement.
                idx = torch.randperm(n_all, device=dev)[:n]
                q_dict = q_all[idx]
            else:
                # Tile + partial random fill to hit requested size.
                reps = n // n_all
                rem = n % n_all
                blocks = [q_all] * reps
                if rem > 0:
                    idx = torch.randperm(n_all, device=dev)[:rem]
                    blocks.append(q_all[idx])
                q_dict = torch.cat(blocks, dim=0)
        else:
            raise ValueError(f"Unknown sampling mode '{sampling}'. Use 'random' or 'fz'.")

        # Convert sampled quaternions to the convention expected by this codec.
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
        """
        Local descriptor-space refinement on SO(3), starting from q_init.
        """
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
                # Normalize per-sample gradient for stable step magnitudes.
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
        """
        Descriptor-to-quaternion decode via cosine top-k retrieval + optional
        local refinement in descriptor space.

        x: [..., 9]
        q_dict: [N, 4]
        x_dict: [N, 9] optional
        """
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

            # On-shell descriptors make cosine retrieval an efficient NN proxy.
            xi_n = F.normalize(xi, dim=-1, eps=1e-12)
            sims = xi_n @ x_dict_n.T
            _, top_idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)

            if k == 1:
                init_idx = top_idx[:, 0]
            else:
                # Use cosine top-k as shortlist, then choose best L2 descriptor match.
                xk = x_dict[top_idx]                         # [m,k,9]
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

        return DecodeResult(
            quaternions=q_nn,
            descriptor_nn=x_nn,
            distances=d_nn,
        )


# ============================================================
# 6) Fiber-preserving 2D operators for one irrep block
# ============================================================

class SharedOverMConv2d(nn.Module):
    """
    Apply the same 2D conv to every m-component of a degree-l irrep block.

    Input/output layout:
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
        mode: str = "bilinear",
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
            y = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=False,
            )
        else:
            y = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode=self.mode,
            )

        H2, W2 = y.shape[-2:]
        y = y.view(B, self.d, self.mul, H2, W2).permute(0, 2, 1, 3, 4)
        y = y.reshape(B, self.mul * self.d, H2, W2)
        return y


# ============================================================
# 7) Multi-stage transpose-conv upsampler
# ============================================================

def factorize_scale(scale: int) -> list[int]:
    """
    Prefer repeated x2 stages when possible.
    Examples:
        2  -> [2]
        4  -> [2, 2]
        8  -> [2, 2, 2]
        6  -> [2, 3]
        12 -> [2, 2, 3]
        9  -> [3, 3]
        5  -> [5]
    """
    if not isinstance(scale, int) or scale < 2:
        raise ValueError(f"scale must be an integer >= 2, got {scale}")

    factors = []
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


class SharedOverMTransposeConv2d(nn.Module):
    """
    One transpose-conv stage tied across the m-components of one irrep block.
    """

    def __init__(
        self,
        mul_in: int,
        mul_out: int,
        l: int,
        scale_factor: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        if not isinstance(scale_factor, int) or scale_factor < 2:
            raise ValueError(f"scale_factor must be an integer >= 2, got {scale_factor}")

        self.mul_in = mul_in
        self.mul_out = mul_out
        self.l = l
        self.d = 2 * l + 1
        self.scale_factor = scale_factor

        s = scale_factor
        kernel_size = 2 * s - (s % 2)
        padding = s // 2
        output_padding = 0

        self.deconv = nn.ConvTranspose2d(
            mul_in,
            mul_out,
            kernel_size=kernel_size,
            stride=s,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.mul_in * self.d, f"Expected {self.mul_in*self.d} channels, got {C}"

        x = x.view(B, self.mul_in, self.d, H, W).permute(0, 2, 1, 3, 4)
        x = x.reshape(B * self.d, self.mul_in, H, W)

        y = self.deconv(x)
        H2, W2 = y.shape[-2:]

        y = y.view(B, self.d, self.mul_out, H2, W2).permute(0, 2, 1, 3, 4)
        y = y.reshape(B, self.mul_out * self.d, H2, W2)
        return y


class MultiStageSharedOverMTransposeConv2d(nn.Module):
    """
    Multi-stage learned upsampler.
    """

    def __init__(
        self,
        mul: int,
        l: int,
        sr_scale: int,
        refine_block_factory: Optional[Callable[[int], nn.Module]] = None,
        refine_per_stage: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.mul = mul
        self.l = l
        self.sr_scale = sr_scale
        self.factors = factorize_scale(sr_scale)

        stages = []
        for sf in self.factors:
            stages.append(
                SharedOverMTransposeConv2d(
                    mul_in=mul,
                    mul_out=mul,
                    l=l,
                    scale_factor=sf,
                    bias=bias,
                )
            )
            if refine_block_factory is not None and refine_per_stage > 0:
                for _ in range(refine_per_stage):
                    stages.append(refine_block_factory(mul))

        self.net = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 8) Tensor-product lift: 1x4e -> m6 x 6e
# ============================================================

class PerPixelSelfTensorLift4to6(nn.Module):
    """
    Per-pixel self tensor product:
        1x4e ⊗ 1x4e -> mul6 x 6e

    Input:
        x4: [B, 9, H, W]
    Output:
        x6: [B, mul6*13, H, W]
    """

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

        x = x4.permute(0, 2, 3, 1).contiguous().view(-1, 9)   # [BHW, 9]
        y = self.tp(x, x)                                     # [BHW, mul6*13]
        y = y.view(B, H, W, self.mul6 * 13).permute(0, 3, 1, 2).contiguous()
        return y


# ============================================================
# 9) Joint 4e/6e gating and residual blocks
# ============================================================

class NormGate46(nn.Module):
    """
    Joint equivariant nonlinearity for 4e and 6e fibers.

    Gates are functions only of invariant norms of each copy, so equivariance is preserved.
    """

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

        n4 = torch.sqrt((x4g ** 2).sum(dim=2).clamp_min(1e-12))   # [B,m4,H,W]
        n6 = torch.sqrt((x6g ** 2).sum(dim=2).clamp_min(1e-12))   # [B,m6,H,W]

        norms = torch.cat([n4, n6], dim=1)
        gates = self.net(norms)

        g4 = gates[:, :self.mul4]
        g6 = gates[:, self.mul4:]

        y4 = (x4g * g4.unsqueeze(2)).view(B, C4, H, W)
        y6 = (x6g * g6.unsqueeze(2)).view(B, C6, H, W)
        return y4, y6


class Residual46Block(nn.Module):
    """
    Residual block on hidden 4e + 6e fibers.
    """

    def __init__(self, mul4: int, mul6: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.conv4_1 = SharedOverMConv2d(mul4, mul4, l=4, kernel_size=kernel_size, padding=padding)
        self.conv6_1 = SharedOverMConv2d(mul6, mul6, l=6, kernel_size=kernel_size, padding=padding)

        self.gate = NormGate46(mul4, mul6)

        self.conv4_2 = SharedOverMConv2d(mul4, mul4, l=4, kernel_size=kernel_size, padding=padding)
        self.conv6_2 = SharedOverMConv2d(mul6, mul6, l=6, kernel_size=kernel_size, padding=padding)

    def forward(self, x4: torch.Tensor, x6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y4 = self.conv4_1(x4)
        y6 = self.conv6_1(x6)

        y4, y6 = self.gate(y4, y6)

        y4 = self.conv4_2(y4)
        y6 = self.conv6_2(y6)

        return x4 + y4, x6 + y6


class Residual46Stack(nn.Module):
    def __init__(self, mul4: int, mul6: int, num_blocks: int, kernel_size: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Residual46Block(mul4, mul6, kernel_size=kernel_size) for _ in range(num_blocks)]
        )

    def forward(self, x4: torch.Tensor, x6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x4, x6 = block(x4, x6)
        return x4, x6


# ============================================================
# 10) Dual-fiber upsamplers
# ============================================================

class DualFiberInterpolation(nn.Module):
    """
    Componentwise interpolation for 4e and 6e fibers.
    """

    def __init__(self, mul4: int, mul6: int, scale_factor: int, mode: str = "bilinear"):
        super().__init__()
        self.up4 = SharedOverMInterpolation(mul4, l=4, scale_factor=scale_factor, mode=mode)
        self.up6 = SharedOverMInterpolation(mul6, l=6, scale_factor=scale_factor, mode=mode)

    def forward(self, x4: torch.Tensor, x6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.up4(x4), self.up6(x6)


class DualFiberMultiStageTranspose(nn.Module):
    """
    Multi-stage learned transpose-conv for 4e and 6e fibers.
    """

    def __init__(
        self,
        mul4: int,
        mul6: int,
        sr_scale: int,
        refine_per_stage: int = 1,
    ):
        super().__init__()
        self.factors = factorize_scale(sr_scale)

        stages = []
        for sf in self.factors:
            stage_up4 = SharedOverMTransposeConv2d(mul4, mul4, l=4, scale_factor=sf)
            stage_up6 = SharedOverMTransposeConv2d(mul6, mul6, l=6, scale_factor=sf)

            refiners = Residual46Stack(mul4, mul6, num_blocks=refine_per_stage) if refine_per_stage > 0 else None
            stages.append(nn.ModuleDict({"up4": stage_up4, "up6": stage_up6, "refine": refiners}))

        self.stages = nn.ModuleList(stages)

    def forward(self, x4: torch.Tensor, x6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for stage in self.stages:
            x4 = stage["up4"](x4)
            x6 = stage["up6"](x6)
            if stage["refine"] is not None:
                x4, x6 = stage["refine"](x4, x6)
        return x4, x6


# ============================================================
# 11) 4e + 6e hidden-fiber SR model
# ============================================================

class SO3OQuotientSRNet4e6e(nn.Module):
    """
    Quaternion map -> cubic quotient 4e descriptor -> hidden 4e+6e fibers -> SR -> HR 4e descriptor.

    Input/output are still the injective cubic quotient embedding in 4e.
    Hidden 6e features are introduced via a self tensor product of the input 4e field.
    """

    def __init__(
        self,
        hidden_mul4: int = 16,
        hidden_mul6: int = 8,
        num_blocks_pre: int = 4,
        num_blocks_post: int = 4,
        upsample_mode: str = "bilinear",   # "bilinear", "nearest", "bicubic", "transpose"
        sr_scale: int = 2,
        refine_per_stage: int = 1,
        passive_input: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.sr_scale = sr_scale
        self.hidden_mul4 = hidden_mul4
        self.hidden_mul6 = hidden_mul6

        self.codec = CubicQuotient4eCodec(
            passive_input=passive_input,
            dtype=dtype,
        )

        # input 4e descriptor -> hidden 4e copies
        self.in4 = SharedOverMConv2d(
            mul_in=1,
            mul_out=hidden_mul4,
            l=4,
            kernel_size=3,
            padding=1,
        )

        # input 4e descriptor -> hidden 6e copies via tensor product
        self.in6 = PerPixelSelfTensorLift4to6(hidden_mul6)

        self.pre = Residual46Stack(hidden_mul4, hidden_mul6, num_blocks=num_blocks_pre)

        if upsample_mode == "transpose":
            self.up = DualFiberMultiStageTranspose(
                hidden_mul4,
                hidden_mul6,
                sr_scale=sr_scale,
                refine_per_stage=refine_per_stage,
            )
        else:
            self.up = DualFiberInterpolation(
                hidden_mul4,
                hidden_mul6,
                scale_factor=sr_scale,
                mode=upsample_mode,
            )

        self.post = Residual46Stack(hidden_mul4, hidden_mul6, num_blocks=num_blocks_post)

        # 4e output head
        self.out4 = SharedOverMConv2d(
            mul_in=hidden_mul4,
            mul_out=1,
            l=4,
            kernel_size=3,
            padding=1,
        )

        # residual/skip path from input descriptor to output descriptor
        self.skip_up = SharedOverMInterpolation(
            mul=1,
            l=4,
            scale_factor=sr_scale,
            mode="bilinear",
        )

    def encode(self, q_lr: torch.Tensor) -> torch.Tensor:
        """
        q_lr: [B,4,H,W]
        returns: [B,9,H,W]
        """
        return self.codec.encode_map(q_lr)

    def forward_descriptor(self, q_lr: torch.Tensor) -> torch.Tensor:
        """
        q_lr: [B,4,H,W]
        returns: [B,9,sH,sW]
        """
        x0 = self.encode(q_lr)              # [B,9,H,W]

        x4 = self.in4(x0)                   # [B, hidden_mul4*9, H, W]
        x6 = self.in6(x0)                   # [B, hidden_mul6*13, H, W]

        x4, x6 = self.pre(x4, x6)
        x4, x6 = self.up(x4, x6)
        x4, x6 = self.post(x4, x6)

        y = self.out4(x4)                   # [B,9,sH,sW]
        skip = self.skip_up(x0)             # [B,9,sH,sW]
        y = y + skip

        # project toward the constant-radius shell of valid embeddings
        target_norm = self.codec.a4.norm()
        ynorm = torch.sqrt((y ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12))
        y = target_norm * y / ynorm
        return y

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
    ) -> DecodeResult:
        x_hr = self.forward_descriptor(q_lr)
        x_hr_last = x_hr.permute(0, 2, 3, 1).contiguous()   # [B,sH,sW,9]
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
# 12) Losses
# ============================================================

def descriptor_mse_loss(x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_pred, x_gt)


def descriptor_shell_loss(x_pred: torch.Tensor, target_radius: float = 1.0) -> torch.Tensor:
    r = torch.sqrt((x_pred ** 2).sum(dim=1).clamp_min(1e-12))
    return ((r - target_radius) ** 2).mean()


def quotient_descriptor_loss(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    target_radius: float = 1.0,
    lam_shell: float = 1e-2,
) -> torch.Tensor:
    return descriptor_mse_loss(x_pred, x_gt) + lam_shell * descriptor_shell_loss(
        x_pred,
        target_radius=target_radius,
    )


# ============================================================
# 13) Debug / sanity utilities
# ============================================================

@torch.no_grad()
def apply_global_specimen_rotation_to_descriptor(
    x: torch.Tensor,
    q_rot_active: torch.Tensor,
) -> torch.Tensor:
    """
    x: [B,9,H,W]
    q_rot_active: [4] or [B,4]
    """
    ir4 = o3.Irrep("4e")
    D4 = D_from_quaternion_device(ir4, normalize_quaternion(q_rot_active))  # [...,9,9]

    if D4.ndim == 2:
        D4 = D4.unsqueeze(0).expand(x.shape[0], -1, -1)

    y = torch.einsum("bij,bjhw->bihw", D4.to(x.dtype).to(x.device), x)
    return y


# ============================================================
# 14) Demo
# ============================================================

def _demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    B, H, W = 2, 32, 48
    q_lr = rand_quaternion_grid(B, H, W, device=device, dtype=dtype)   # [B,H,W,4]
    q_lr = q_lr.permute(0, 3, 1, 2).contiguous()                       # [B,4,H,W]

    sr_scale = 4

    model = SO3OQuotientSRNet4e6e(
        hidden_mul4=16,
        hidden_mul6=8,
        num_blocks_pre=3,
        num_blocks_post=3,
        upsample_mode="transpose",   # "transpose" or "bilinear"
        sr_scale=sr_scale,
        refine_per_stage=1,
        passive_input=False,         # rand_quaternion_grid gives active quats
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    print("Model created.")
    if hasattr(model.up, "factors"):
        print("Upsample stage factors:", model.up.factors)

    x_hr = model.forward_descriptor(q_lr)
    print("Predicted HR descriptor shape:", tuple(x_hr.shape))

    q_dict, x_dict = model.codec.build_dictionary(
        n=20000,
        device=device,
        dtype=dtype,
    )

    dec = model.forward_quaternion_nn(
        q_lr,
        q_dict=q_dict,
        x_dict=x_dict,
        chunk=2048,
    )
    print("Decoded quaternion shape:", tuple(dec.quaternions.shape))
    print("Mean NN descriptor distance:", dec.distances.mean().item())

    # descriptor equivariance sanity check on the encoder
    qg = o3.rand_quaternion(1, device=device, dtype=dtype)[0]   # [4]
    x_lr = model.encode(q_lr)
    x_rot = apply_global_specimen_rotation_to_descriptor(x_lr, qg)

    qg_expand = qg.view(1, 1, 1, 4).expand(B, H, W, 4)
    q_lr_last = q_lr.permute(0, 2, 3, 1).contiguous()           # [B,H,W,4]
    q_lr_rot = quaternion_multiply(qg_expand, q_lr_last)
    q_lr_rot = normalize_quaternion(q_lr_rot)
    q_lr_rot = q_lr_rot.permute(0, 3, 1, 2).contiguous()

    x_lr_direct = model.encode(q_lr_rot)
    err = torch.norm(x_rot - x_lr_direct) / torch.norm(x_lr_direct)
    print("Relative descriptor equivariance error:", err.item())


if __name__ == "__main__":
    _demo()
