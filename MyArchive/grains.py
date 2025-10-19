"""
train_eqsr_with_vis.py
Quaternion-equivariant SR training with JSON-configured hyperparameters,
PSNR reporting, and side-by-side visualization (GT vs SR).

Create a config.json next to this file, e.g.:

{
  "epochs": 5,
  "batch_size": 8,
  "lr": 0.0003,
  "scale": 2,
  "hr": 64,
  "train_size": 2000,
  "val_size": 200,
  "n_feats": 8,
  "kernel_size": 3,
  "group": "432",
  "include_improper": true,
  "save": "checkpoints",
  "amp": false,
  "clip": 1.0,
  "vis_every": 1,
  "grains_min": 12,
  "grains_max": 28,
  "motif_freq_min": 3.0,
  "motif_freq_max": 7.0,
  "boundary_thickness": 1
}
"""

import os, json, math, random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# ---------- Optional dep (pymatgen for general groups; not required for 432) ----------
_HAS_PYMATGEN = True
try:
    from pymatgen.symmetry.groups import PointGroup
except Exception:
    _HAS_PYMATGEN = False

# =========================================================
# Utilities: PSNR + visualization
# =========================================================


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val**2) / mse)


def show_sr_vs_gt(
    HR_rgb: torch.Tensor,
    SR_rgb: torch.Tensor,
    psnr_val: float = None,
    save_path: str = None,
):
    """
    Display GT and SR side-by-side. HR_rgb, SR_rgb: (3,H,W), in [0,1].
    If save_path is provided, saves PNG as well.
    """
    gt = HR_rgb.permute(1, 2, 0).detach().cpu().numpy()
    sr = SR_rgb.permute(1, 2, 0).detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gt)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    title = "Super-resolved"
    if psnr_val is not None:
        title += f"\nPSNR: {psnr_val:.2f} dB"
    plt.title(title)
    plt.imshow(sr)
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# =========================================================
# Group reps: C rep from 3x3 rotations; FCC hard-coded fallback
# =========================================================


def C_from_R3_torch(R3: torch.Tensor) -> torch.Tensor:
    """
    R3: (...,3,3) orthogonal (det ±1). Return (...,4,4) = diag(1, R3).
    """
    *batch, _, _ = R3.shape
    I1 = torch.ones(*batch, 1, 1, device=R3.device, dtype=R3.dtype)
    Z13 = torch.zeros(*batch, 1, 3, device=R3.device, dtype=R3.dtype)
    Z31 = torch.zeros(*batch, 3, 1, device=R3.device, dtype=R3.dtype)
    top = torch.cat([I1, Z13], dim=-1)
    bot = torch.cat([Z31, R3], dim=-1)
    return torch.cat([top, bot], dim=-2)


def _hardcoded_fcc_quats_numpy() -> np.ndarray:
    """
    FCC (432) unit quaternions (S,X,Y,Z).
    """
    h = 1.0 / 2.0
    i = 1.0 / np.sqrt(2.0)
    fcc_rows = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [i, i, 0, 0],
            [i, 0, i, 0],
            [i, 0, 0, i],
            [i, -i, 0, 0],
            [i, 0, -i, 0],
            [i, 0, 0, -i],
            [0, i, i, 0],
            [0, i, 0, i],
            [0, 0, i, i],
            [0, i, -i, 0],
            [0, 0, i, -i],
            [0, i, 0, -i],
            [h, h, h, h],
            [h, -h, -h, h],
            [h, -h, h, -h],
            [h, h, -h, -h],
            [h, h, h, -h],
            [h, h, -h, h],
            [h, -h, h, h],
            [h, -h, -h, -h],
        ],
        dtype=float,
    )
    q = fcc_rows / np.linalg.norm(fcc_rows, axis=1, keepdims=True)
    q = np.unique(np.round(q, 12), axis=0)
    assert q.shape[0] == 24, f"FCC set expected 24, got {q.shape[0]}"
    return q


def rep_mats_from_pointgroup(
    group_name: str = "432",
    include_improper: bool = True,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Returns rho(g) as a tensor of shape (G, 4, 4) using the "C" rep: diag(1, R3).
    If pymatgen not available and group is "432", falls back to hard-coded FCC set.
    """
    if _HAS_PYMATGEN:
        pg = PointGroup(group_name)
        R3s = []
        for op in pg.symmetry_ops:
            M = np.array(op.rotation_matrix, dtype=float)
            det = np.linalg.det(M)
            if np.isclose(det, 1.0, atol=1e-9):
                det = 1.0
            elif np.isclose(det, -1.0, atol=1e-9):
                det = -1.0
            if det < 0 and not include_improper:
                continue
            R3s.append(M)
        if len(R3s) == 0:
            raise ValueError(
                f"No ops remain for group={group_name} with include_improper={include_improper}"
            )
        R3 = np.stack(R3s, axis=0)
        R3 = np.round(R3, 12)
        R3uniq = np.unique(R3.reshape(R3.shape[0], -1), axis=0).reshape(-1, 3, 3)
        R3_t = torch.tensor(R3uniq, device=device, dtype=dtype)
        return C_from_R3_torch(R3_t)

    # Fallback only for 432: convert quats -> R3 manually
    if group_name != "432":
        raise RuntimeError("pymatgen is required for groups other than 432.")
    q = _hardcoded_fcc_quats_numpy()  # (24,4) [S,X,Y,Z]
    a = q[:, 0]
    b = q[:, 1]
    c = q[:, 2]
    d = q[:, 3]
    R3 = np.stack(
        [
            np.stack(
                [1 - 2 * (c * c + d * d), 2 * (b * c - a * d), 2 * (b * d + a * c)],
                axis=-1,
            ),
            np.stack(
                [2 * (b * c + a * d), 1 - 2 * (b * b + d * d), 2 * (c * d - a * b)],
                axis=-1,
            ),
            np.stack(
                [2 * (b * d - a * c), 2 * (c * d + a * b), 1 - 2 * (b * b + c * c)],
                axis=-1,
            ),
        ],
        axis=1,
    )
    R3 = np.transpose(R3, (1, 0, 2))  # (24, 3, 3)
    R3_t = torch.tensor(R3, device=device, dtype=dtype)
    return C_from_R3_torch(R3_t)


def make_group_tensors(
    num_blocks: int,
    group_name: str = "432",
    include_improper: bool = True,
    device=None,
    dtype=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build block-diagonal reps with num_blocks copies of base 4x4 "C" rep.
    Returns (rho, rho_T) of shape (G, 4*num_blocks, 4*num_blocks).
    """
    base = rep_mats_from_pointgroup(
        group_name=group_name,
        include_improper=include_improper,
        device=device,
        dtype=dtype,
    )  # (G,4,4)
    I = torch.eye(num_blocks, device=base.device, dtype=base.dtype)
    rho = torch.kron(I, base)  # (G, 4B, 4B)
    return rho, rho.transpose(1, 2)


# =========================================================
# Quaternion kernel packing + Reynolds projection
# =========================================================


def pack_quaternion_kernel_conv(r, i, j, k):
    cat_r = torch.cat([r, -i, -j, -k], dim=1)
    cat_i = torch.cat([i, r, -k, j], dim=1)
    cat_j = torch.cat([j, k, r, -i], dim=1)
    cat_k = torch.cat([k, -j, i, r], dim=1)
    return torch.cat([cat_r, cat_i, cat_j, cat_k], dim=0)


def pack_quaternion_kernel_deconv(r, i, j, k):
    cat_r = torch.cat([r, -i, -j, -k], dim=1)
    cat_i = torch.cat([i, r, -k, j], dim=1)
    cat_j = torch.cat([j, k, r, -i], dim=1)
    cat_k = torch.cat([k, -j, i, r], dim=1)
    return torch.cat([cat_r, cat_i, cat_j, cat_k], dim=0)


def project_conv_weight(W, rho_out, rho_in):
    """
    W: (Co, Ci, kH, kW)
    Π(W) = (1/|G|) Σ_g Rout(g) W Rin(g)^T
    """
    G = rho_in.shape[0]
    Co, Ci, kH, kW = W.shape
    Wp = torch.zeros_like(W)
    for g in range(G):
        Rout = rho_out[g]
        Rin = rho_in[g]
        left = torch.einsum("ab,bcij->acij", Rout, W)
        right = torch.einsum("acij,dc->adij", left, Rin)
        Wp += right
    return Wp / G


def project_deconv_weight(W, rho_in, rho_out):
    """
    For conv_transpose2d, weight W has shape (Ci, Co, kH, kW).
    Π(W) = (1/|G|) Σ_g Rout(g) W Rin(g)^T
    """
    G = rho_in.shape[0]
    Ci, Co, kH, kW = W.shape
    Wp = torch.zeros_like(W)
    for g in range(G):
        Rout = rho_out[g]
        Rin = rho_in[g]
        left = torch.einsum("ab,cbij->caij", Rout, W)  # multiply output axis (dim=1)
        right = torch.einsum("daij,cd->caij", left, Rin)
        Wp += right
    return Wp / G


# =========================================================
# Quaternion conv modules + wrapper + upsampler
# =========================================================


class QuaternionConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.in_q = in_channels
        self.out_q = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        kH, kW = self.kernel_size
        self.r_weight = nn.Parameter(torch.randn(self.out_q, self.in_q, kH, kW) * 0.02)
        self.i_weight = nn.Parameter(torch.randn(self.out_q, self.in_q, kH, kW) * 0.02)
        self.j_weight = nn.Parameter(torch.randn(self.out_q, self.in_q, kH, kW) * 0.02)
        self.k_weight = nn.Parameter(torch.randn(self.out_q, self.in_q, kH, kW) * 0.02)
        self.stride, self.padding, self.dilation, self.groups = (
            stride,
            padding,
            dilation,
            groups,
        )
        self.bias = None if not bias else nn.Parameter(torch.zeros(4 * self.out_q))


class QuaternionTransposeConv(nn.Module):
    """
    For strict equivariance in upsampling: set kernel_size=stride=scale, padding=0, output_padding=0.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.in_q = in_channels
        self.out_q = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        kH, kW = self.kernel_size
        self.r_weight = nn.Parameter(torch.randn(self.in_q, self.out_q, kH, kW) * 0.02)
        self.i_weight = nn.Parameter(torch.randn(self.in_q, self.out_q, kH, kW) * 0.02)
        self.j_weight = nn.Parameter(torch.randn(self.in_q, self.out_q, kH, kW) * 0.02)
        self.k_weight = nn.Parameter(torch.randn(self.in_q, self.out_q, kH, kW) * 0.02)
        self.stride, self.padding, self.output_padding = stride, padding, output_padding
        self.dilation, self.groups = dilation, groups
        self.bias = None if not bias else nn.Parameter(torch.zeros(4 * self.out_q))


class EquivariantReynoldsWrap(nn.Module):
    """
    Wraps quaternion conv / deconv and applies Reynolds projection to PACKED real weight
    under the block-diagonal channel rep rho (same in/out fibers here).
    """

    def __init__(self, inner_module, group_tensor, group_tensor_inv, op_type="conv2d"):
        super().__init__()
        self.m = inner_module
        self.register_buffer("rho", group_tensor)
        self.register_buffer("rho_inv", group_tensor_inv)
        assert op_type in ("conv2d", "deconv2d")
        self.op_type = op_type

    def forward(self, x):
        if self.op_type == "conv2d":
            W = pack_quaternion_kernel_conv(
                self.m.r_weight, self.m.i_weight, self.m.j_weight, self.m.k_weight
            )
            Wp = project_conv_weight(W, rho_out=self.rho, rho_in=self.rho)
            return F.conv2d(
                x,
                Wp,
                bias=None,
                stride=self.m.stride,
                padding=self.m.padding,
                dilation=self.m.dilation,
                groups=self.m.groups,
            )
        else:
            W = pack_quaternion_kernel_deconv(
                self.m.r_weight, self.m.i_weight, self.m.j_weight, self.m.k_weight
            )
            Wp = project_deconv_weight(W, rho_in=self.rho, rho_out=self.rho)
            return F.conv_transpose2d(
                x,
                Wp,
                bias=None,
                stride=self.m.stride,
                padding=self.m.padding,
                output_padding=self.m.output_padding,
                dilation=self.m.dilation,
                groups=self.m.groups,
            )


class UpsamplerQuaternionTransposeConv(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        scale: int,
        n_feats: int,
        group_tensor: torch.Tensor,
        group_tensor_inv: torch.Tensor,
        dropout_prob: float = 0.0,
        out_channels: int = 3,  # RGB
    ):
        """
        n_feats: number of quaternion blocks (channels = 4 * n_feats)
        scale: upsampling factor (2 or 4 recommended)
        """
        super().__init__()
        self.scale = scale
        self.n_feat = n_feats

        self.conv_layer = EquivariantReynoldsWrap(
            QuaternionConv(
                n_feats,
                n_feats,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
            ),
            group_tensor=group_tensor,
            group_tensor_inv=group_tensor_inv,
            op_type="conv2d",
        )

        self.dropout = (
            nn.Identity() if dropout_prob <= 0 else nn.Dropout2d(p=dropout_prob)
        )

        self.transposed_conv = EquivariantReynoldsWrap(
            QuaternionTransposeConv(
                in_channels=n_feats,
                out_channels=n_feats,
                kernel_size=scale,
                stride=scale,
                padding=0,
                output_padding=0,
                bias=False,
            ),
            group_tensor=group_tensor,
            group_tensor_inv=group_tensor_inv,
            op_type="deconv2d",
        )

        self.post_conv_layer = EquivariantReynoldsWrap(
            QuaternionConv(
                n_feats,
                n_feats,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
            ),
            group_tensor=group_tensor,
            group_tensor_inv=group_tensor_inv,
            op_type="conv2d",
        )

        # Final real head (not equivariant): mix 4*n_feats -> RGB
        self.head = nn.Conv2d(4 * n_feats, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (N, 4*n_feats, h, w)
        x = self.conv_layer(x)
        x = self.dropout(x)
        x = self.transposed_conv(x)
        x = self.post_conv_layer(x)
        y = self.head(x)  # (N, out_channels, H*scale, W*scale)
        return y


# =========================================================
# FCC Microstructure dataset
# =========================================================


def _fcc_R3_set(device=None, dtype=None) -> torch.Tensor:
    """Return the set of 3×3 rotations for 432 (24 elements)."""
    base = rep_mats_from_pointgroup(
        "432", include_improper=True, device=device, dtype=dtype
    )  # (24,4,4) diag(1,R3)
    # Extract the 3×3 part
    return base[:, 1:, 1:]  # (24,3,3)


def _render_rotated_cubic_motif(
    H: int, W: int, theta: float, freq: float, phase: float = 0.0
) -> np.ndarray:
    """
    Simple 2D cubic-looking motif: product of two orthogonal stripe fields,
    rotated by theta. Returns grayscale in [0,1].
    f(x,y) = 0.5 + 0.5 * sin(2π f u) * sin(2π f v), with (u,v) the rotated coords.
    """
    ys, xs = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    ct, st = math.cos(theta), math.sin(theta)
    u = ct * xs + st * ys
    v = -st * xs + ct * ys
    pat = np.sin(2 * np.pi * freq * u + phase) * np.sin(2 * np.pi * freq * v + phase)
    pat = 0.5 + 0.5 * pat
    return np.clip(pat, 0.0, 1.0)


def _voronoi_grains(
    H: int, W: int, n_grains: int, rng: np.random.RandomState
) -> np.ndarray:
    """Return integer grain labels (H,W) via brute-force nearest seeds."""
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    seeds_y = rng.randint(0, H, size=n_grains)
    seeds_x = rng.randint(0, W, size=n_grains)
    # Compute squared distances to all seeds
    d2 = (ys[..., None] - seeds_y[None, None, :]) ** 2 + (
        xs[..., None] - seeds_x[None, None, :]
    ) ** 2
    labels = np.argmin(d2, axis=-1).astype(np.int32)  # (H,W)
    return labels


def _grain_boundaries(labels: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Binary mask (H,W) indicating grain boundaries."""
    H, W = labels.shape
    b = np.zeros((H, W), np.uint8)
    # 4-neighborhood difference
    b[1:] |= labels[1:] != labels[:-1]
    b[:-1] |= labels[:-1] != labels[1:]
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    if thickness > 1:
        from scipy.ndimage import (
            maximum_filter,
        )  # optional; if not available, it will fail—set thickness=1

        b = maximum_filter(b, size=thickness)
    return b


class FCCMicrostructureSR(Dataset):
    """
    Procedural FCC microstructure images:
      - Random Voronoi tessellation -> grains
      - Each grain assigned one of the 24 cubic rotations (432)
      - Inside each grain, render a rotated cubic motif with frequency jitter
      - Optional dark grain boundaries
      - HR images in [0,1], 3 channels (colored by orientation hue)
      - LR images via avg pooling by 'scale'
    """

    def __init__(
        self,
        n_samples=2000,
        hr_size=64,
        scale=2,
        grains_min=12,
        grains_max=28,
        motif_freq_min=3.0,
        motif_freq_max=7.0,
        boundary_thickness=1,
        seed=0,
    ):
        assert hr_size % scale == 0
        self.n = n_samples
        self.hr = hr_size
        self.lr = hr_size // scale
        self.scale = scale
        self.grains_min = grains_min
        self.grains_max = grains_max
        self.freq_min = motif_freq_min
        self.freq_max = motif_freq_max
        self.boundary_thickness = boundary_thickness

        self.rng = np.random.RandomState(seed)
        # Precompute the 432 rotations and their in-plane angles (about z)
        R3 = _fcc_R3_set()  # (24,3,3) on CPU float32
        R = R3.numpy()
        # Project to an in-plane rotation angle φ = atan2(R[1,0], R[0,0])
        self.angles = np.arctan2(R[:, 1, 0], R[:, 0, 0])  # 24 angles in [-pi, pi]

    def __len__(self):
        return self.n

    def _one_microstructure(self) -> torch.Tensor:
        H, W = self.hr, self.hr
        n_grains = self.rng.randint(self.grains_min, self.grains_max + 1)
        labels = _voronoi_grains(H, W, n_grains, self.rng)  # (H,W)
        boundary = (
            _grain_boundaries(labels, thickness=self.boundary_thickness)
            if self.boundary_thickness > 0
            else 0
        )

        # Assign an orientation to each grain from 24 rotations
        grain_angles = self.rng.choice(self.angles, size=n_grains, replace=True)

        # Render per-grain motif
        img = np.zeros((H, W), np.float32)
        hue = np.zeros((H, W), np.float32)  # store angle for coloring

        for g in range(n_grains):
            theta = grain_angles[g]
            freq = self.rng.uniform(self.freq_min, self.freq_max)
            phase = self.rng.uniform(0, 2 * np.pi)
            mask = labels == g
            # For efficiency, render motif once full-frame, then mask; small H keeps this cheap
            motif = _render_rotated_cubic_motif(H, W, theta, freq, phase)  # [0,1]
            img[mask] = motif[mask]
            hue[mask] = (theta + np.pi) / (2 * np.pi)  # map [-pi,pi] -> [0,1]

        # Darken grain boundaries
        if isinstance(boundary, np.ndarray):
            img[boundary > 0] *= 0.5

        # Simple HSV -> RGB (S=1,V=img) for orientation coloring
        S = np.ones_like(img)
        V = np.clip(img, 0.0, 1.0)
        Hh = hue  # [0,1]
        rgb = _hsv_to_rgb_numpy(Hh, S, V)  # (H,W,3) in [0,1]
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32)  # (3,H,W)
        return torch.from_numpy(rgb)

    def __getitem__(self, idx):
        HR = self._one_microstructure()  # (3,H,W)
        k = self.scale
        LR = F.avg_pool2d(HR.unsqueeze(0), kernel_size=k, stride=k).squeeze(
            0
        )  # (3, H/scale, W/scale)
        return LR, HR


def _hsv_to_rgb_numpy(H: np.ndarray, S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Vectorized HSV->RGB for arrays in [0,1].
    Returns (H,W,3).
    """
    h = H * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = V * (1.0 - S)
    q = V * (1.0 - S * f)
    t = V * (1.0 - S * (1.0 - f))

    i_mod = i % 6
    r = np.choose(i_mod, [V, q, p, p, t, V])
    g = np.choose(i_mod, [t, V, V, q, p, p])
    b = np.choose(i_mod, [p, p, t, V, V, q])
    return np.stack([r, g, b], axis=-1)


# =========================================================
# Quaternion input stem (simple)
# =========================================================


def make_quat_input_from_rgb(LR_rgb: torch.Tensor, n_feats: int) -> torch.Tensor:
    """
    LR_rgb: (N,3,h,w) in [0,1]
    Produce (N, 4*n_feats, h, w) by simple channel replication (placeholder stem).
    """
    N, C, h, w = LR_rgb.shape
    assert C == 3
    mono = LR_rgb.mean(dim=1, keepdim=True)  # (N,1,h,w)
    S = mono
    X = mono
    Y = mono
    Z = mono
    one_quat = torch.cat([S, X, Y, Z], dim=1)  # (N,4,h,w)
    return one_quat.repeat(1, n_feats, 1, 1)  # (N, 4*n_feats, h, w)


def apply_group_channels(x, Rblock):
    """
    x: (N, C, H, W), Rblock: (C, C) block-diagonal rep
    """
    N, C, H, W = x.shape
    xf = x.view(N, C, -1)
    yf = torch.einsum("ab,nbt->nat", Rblock, xf)
    return yf.view(N, C, H, W)


# =========================================================
# Training
# =========================================================


def train_with_config(cfg: Dict[str, Any]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    os.makedirs(cfg["save"], exist_ok=True)

    # Data (FCC microstructures)
    train_set = FCCMicrostructureSR(
        n_samples=cfg["train_size"],
        hr_size=cfg["hr"],
        scale=cfg["scale"],
        grains_min=cfg.get("grains_min", 12),
        grains_max=cfg.get("grains_max", 28),
        motif_freq_min=cfg.get("motif_freq_min", 3.0),
        motif_freq_max=cfg.get("motif_freq_max", 7.0),
        boundary_thickness=cfg.get("boundary_thickness", 1),
        seed=0,
    )
    val_set = FCCMicrostructureSR(
        n_samples=cfg["val_size"],
        hr_size=cfg["hr"],
        scale=cfg["scale"],
        grains_min=cfg.get("grains_min", 12),
        grains_max=cfg.get("grains_max", 28),
        motif_freq_min=cfg.get("motif_freq_min", 3.0),
        motif_freq_max=cfg.get("motif_freq_max", 7.0),
        boundary_thickness=cfg.get("boundary_thickness", 1),
        seed=1,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,  # <- no subprocesses
        persistent_workers=False,  # <- important when num_workers>0, harmless here
        pin_memory=False,  # <- can leave True on CUDA; False is simplest
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    # Group reps
    rho, rho_inv = make_group_tensors(
        num_blocks=cfg["n_feats"],
        group_name=cfg["group"],
        include_improper=bool(cfg["include_improper"]),
        device=device,
        dtype=dtype,
    )

    # Model
    model = UpsamplerQuaternionTransposeConv(
        kernel_size=cfg["kernel_size"],
        scale=cfg["scale"],
        n_feats=cfg["n_feats"],
        group_tensor=rho,
        group_tensor_inv=rho_inv,
        dropout_prob=0.0,
        out_channels=3,
    ).to(device=device, dtype=dtype)

    # Optim/sched
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    best_psnr = -1.0
    vis_every = int(cfg.get("vis_every", 1))

    for epoch in range(1, cfg["epochs"] + 1):
        # ---------------- Train ----------------
        model.train()
        total_loss = 0.0
        for LR_rgb, HR_rgb in train_loader:
            LR_rgb = LR_rgb.to(device=device, dtype=dtype, non_blocking=True)
            HR_rgb = HR_rgb.to(device=device, dtype=dtype, non_blocking=True)

            x = make_quat_input_from_rgb(LR_rgb, n_feats=cfg["n_feats"])

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                SR = model(x)
                loss = F.l1_loss(SR, HR_rgb)

            scaler.scale(loss).backward()
            if cfg["clip"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item() * LR_rgb.size(0)

        sched.step()
        train_loss = total_loss / len(train_set)

        # ---------------- Validate + PSNR ----------------
        model.eval()
        mse_sum, n_pix = 0.0, 0
        first_pair = None  # (HR, SR) for visualization
        with torch.no_grad():
            for LR_rgb, HR_rgb in val_loader:
                LR_rgb = LR_rgb.to(device=device, dtype=dtype, non_blocking=True)
                HR_rgb = HR_rgb.to(device=device, dtype=dtype, non_blocking=True)
                x = make_quat_input_from_rgb(LR_rgb, n_feats=cfg["n_feats"])
                SR = model(x).clamp(0.0, 1.0)
                if first_pair is None:
                    first_pair = (HR_rgb[0].detach().cpu(), SR[0].detach().cpu())
                mse_sum += F.mse_loss(SR, HR_rgb, reduction="sum").item()
                n_pix += np.prod(HR_rgb.shape)

        mse = mse_sum / n_pix
        cur_psnr = psnr_from_mse(mse, max_val=1.0)

        print(
            f"Epoch {epoch:03d} | train L1: {train_loss:.4f} | PSNR: {cur_psnr:.2f} dB"
        )

        # ------------- Visualize GT vs SR (first sample) -------------
        if (epoch % vis_every == 0) and (first_pair is not None):
            save_img = os.path.join(cfg["save"], f"val_epoch_{epoch:03d}.png")
            show_sr_vs_gt(
                first_pair[0], first_pair[1], psnr_val=cur_psnr, save_path=save_img
            )
            print(f"  Saved visualization: {save_img}")

        # ---------------- Save best ----------------
        if cur_psnr > best_psnr:
            best_psnr = cur_psnr
            ckpt_path = os.path.join(
                cfg["save"], f"best_sr_{cfg['group']}_x{cfg['scale']}.pt"
            )
            torch.save(
                {"model": model.state_dict(), "config": cfg, "psnr": best_psnr},
                ckpt_path,
            )
            print(f"  Saved: {ckpt_path}")

    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB")


# =========================================================
# Entry point: load config.json
# =========================================================

if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg = json.load(f)
    train_with_config(cfg)
