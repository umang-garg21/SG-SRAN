# train_eqsr_quat_pairs.py
# Quaternion super-resolution on paired LR/HR .npy files (quaternions).

import os, re, json, math, glob
from typing import Any, Dict, Tuple, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- Optional dep (pymatgen for general groups; not required for 432) ----------
_HAS_PYMATGEN = True
try:
    from pymatgen.symmetry.groups import PointGroup
except Exception:
    _HAS_PYMATGEN = False

# ---------- Orix for IPF previews ----------
try:
    from orix.quaternion import Orientation
    from orix.quaternion import symmetry as SYM
    from orix.vector import Vector3d
    from orix import plot as orix_plot

    _HAS_ORIX = True
except Exception:
    _HAS_ORIX = False

# One-time MPL defaults (high-res, readable)
if not getattr(mpl, "_qsr_mpl_defaults_set", False):
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "figure.dpi": 500,
        }
    )
    mpl._qsr_mpl_defaults_set = True


# =========================================================
# Utilities: PSNR + quaternion helpers + visualization
# =========================================================


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val**2) / mse)


def _unit_quat(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / n


def _hemisphere_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    d = (pred * gt).sum(dim=-1, keepdim=True)
    sign = torch.where(
        d < 0,
        torch.tensor(-1.0, device=pred.device, dtype=pred.dtype),
        torch.tensor(1.0, device=pred.device, dtype=pred.dtype),
    )
    return pred * sign


def _quat_loss_L1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (pred - gt).abs().mean()


def _quat_to_rgb_preview(q: torch.Tensor) -> torch.Tensor:
    single = q.dim() == 3
    if single:
        q = q.unsqueeze(0)
    b, c, d = q[:, 1], q[:, 2], q[:, 3]
    R = (b + 1.0) * 0.5
    G = (c + 1.0) * 0.5
    B = (d + 1.0) * 0.5
    rgb = torch.stack([R, G, B], dim=1)
    if single:
        rgb = rgb[0]
    return rgb.clamp(0, 1)


def show_sr_vs_gt_quat(
    HR_q: torch.Tensor,
    SR_q: torch.Tensor,
    psnr_val: float = None,
    save_path: str = None,
):
    gt_rgb = _quat_to_rgb_preview(HR_q).permute(1, 2, 0).detach().cpu().numpy()
    sr_rgb = _quat_to_rgb_preview(SR_q).permute(1, 2, 0).detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gt_rgb)
    plt.title("GT (quat preview)")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    title = "SR (quat preview)"
    if psnr_val is not None:
        title += f"\nPSNR: {psnr_val:.2f} dB"
    plt.title(title)
    plt.imshow(sr_rgb)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================
# Group reps: C rep from 3x3 rotations; FCC hard-coded fallback
# =========================================================


def C_from_R3_torch(R3: torch.Tensor) -> torch.Tensor:
    *batch, _, _ = R3.shape
    I1 = torch.ones(*batch, 1, 1, device=R3.device, dtype=R3.dtype)
    Z13 = torch.zeros(*batch, 1, 3, device=R3.device, dtype=R3.dtype)
    Z31 = torch.zeros(*batch, 3, 1, device=R3.device, dtype=R3.dtype)
    top = torch.cat([I1, Z13], dim=-1)
    bot = torch.cat([Z31, R3], dim=-1)
    return torch.cat([top, bot], dim=-2)


def _hardcoded_fcc_quats_numpy() -> np.ndarray:
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
    group_name: str = "432", include_improper: bool = True, device=None, dtype=None
) -> torch.Tensor:
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
        if not R3s:
            raise ValueError(
                f"No ops for group={group_name} with include_improper={include_improper}"
            )
        R3 = np.stack(R3s, axis=0)
        R3 = np.unique(np.round(R3.reshape(R3.shape[0], -1), 12), axis=0).reshape(
            -1, 3, 3
        )
        return C_from_R3_torch(torch.tensor(R3, device=device, dtype=dtype))

    if group_name != "432":
        raise RuntimeError("pymatgen is required for groups other than 432.")
    q = _hardcoded_fcc_quats_numpy()
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
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
    R3 = np.transpose(R3, (1, 0, 2))  # (24,3,3)
    return C_from_R3_torch(torch.tensor(R3, device=device, dtype=dtype))


def make_group_tensors(
    num_blocks: int,
    group_name: str = "432",
    include_improper: bool = True,
    device=None,
    dtype=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base = rep_mats_from_pointgroup(
        group_name, include_improper, device, dtype
    )  # (G,4,4)
    I = torch.eye(num_blocks, device=base.device, dtype=base.dtype)
    rho = torch.kron(I, base)  # (G, 4B, 4B)
    return rho, rho.transpose(1, 2)


def quat_to_C(q: np.ndarray) -> np.ndarray:
    """
    Quaternion -> 4x4 real representation matrix.
    q: (4,) scalar-first (a,b,c,d)
    """
    a, b, c, d = q
    return np.array(
        [
            [a, -b, -c, -d],
            [b, a, -d, c],
            [c, d, a, -b],
            [d, -c, b, a],
        ],
        dtype=np.float32,
    )


def make_group_tensors_from_orix(sym_class, num_blocks: int, device=None, dtype=None):
    """
    sym_class: e.g. ds.sym_class = orix.quaternion.symmetry.Oh
    num_blocks: number of quaternion blocks (n_feats)
    Returns rho, rho_inv: (G, 4*num_blocks, 4*num_blocks)
    """
    quats = np.array(sym_class.data)  # (G,4), scalar-first
    mats = np.stack([quat_to_C(q) for q in quats], axis=0)  # (G,4,4)

    base = torch.tensor(mats, device=device, dtype=dtype)  # (G,4,4)
    I = torch.eye(num_blocks, device=base.device, dtype=base.dtype)
    rho = torch.kron(I, base)  # (G, 4B, 4B)
    rho_inv = rho.transpose(1, 2)  # inverse = transpose (orthogonal)
    return rho, rho_inv


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
    G = rho_in.shape[0]
    Wp = torch.zeros_like(W)
    for g in range(G):
        Rout, Rin = rho_out[g], rho_in[g]
        left = torch.einsum("ab,bcij->acij", Rout, W)
        right = torch.einsum("acij,dc->adij", left, Rin)
        Wp += right
    return Wp / G


def project_deconv_weight(W, rho_in, rho_out):
    G = rho_in.shape[0]
    Wp = torch.zeros_like(W)
    for g in range(G):
        Rout, Rin = rho_out[g], rho_in[g]
        left = torch.einsum("ab,cbij->caij", Rout, W)
        right = torch.einsum("daij,cd->caij", left, Rin)
        Wp += right
    return Wp / G


# =========================================================
# Quaternion conv modules + upsampler (outputs 4 channels = quaternion)
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
        self.in_q, self.out_q = in_channels, out_channels
        kH, kW = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.r_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kH, kW) * 0.02
        )
        self.i_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kH, kW) * 0.02
        )
        self.j_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kH, kW) * 0.02
        )
        self.k_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kH, kW) * 0.02
        )
        self.stride, self.padding, self.dilation, self.groups = (
            stride,
            padding,
            dilation,
            groups,
        )
        self.bias = None if not bias else nn.Parameter(torch.zeros(4 * out_channels))


class QuaternionTransposeConv(nn.Module):
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
        self.in_q, self.out_q = in_channels, out_channels
        kH, kW = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.r_weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kH, kW) * 0.02
        )
        self.i_weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kH, kW) * 0.02
        )
        self.j_weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kH, kW) * 0.02
        )
        self.k_weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kH, kW) * 0.02
        )
        self.stride, self.padding, self.output_padding = stride, padding, output_padding
        self.dilation, self.groups = dilation, groups
        self.bias = None if not bias else nn.Parameter(torch.zeros(4 * out_channels))


class EquivariantReynoldsWrap(nn.Module):
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
    ):
        super().__init__()
        self.scale, self.n_feat = scale, n_feats
        self.conv_layer = EquivariantReynoldsWrap(
            QuaternionConv(
                n_feats,
                n_feats,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
            ),
            group_tensor,
            group_tensor_inv,
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
            group_tensor,
            group_tensor_inv,
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
            group_tensor,
            group_tensor_inv,
            op_type="conv2d",
        )
        self.head = nn.Conv2d(4 * n_feats, 4, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.dropout(x)
        x = self.transposed_conv(x)
        x = self.post_conv_layer(x)
        y = self.head(x)  # (N,4,H*scale,W*scale)
        y = y.permute(0, 2, 3, 1)
        y = _unit_quat(y).permute(0, 3, 1, 2)
        return y


# =========================================================
# Dataset: paired LR/HR quaternion .npy files + IPF preview (Orix)
# =========================================================

_LAST_INT_RE = re.compile(r"(\d+)(?=\.npy$)")


def _last_int_key(fp: str) -> int:
    m = _LAST_INT_RE.search(os.path.basename(fp))
    return int(m.group(1)) if m else -1


def _as_hw4(arr: np.ndarray) -> np.ndarray:
    """
    Accept (H,W,4) or (4,H,W); return float32 (4,H,W).
    Assumes arrays are already unit quaternions with a>=0 (no renormalize/hemisphere).
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D quaternion array; got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = np.moveaxis(arr, -1, 0)  # -> (4,H,W)
    elif arr.shape[0] != 4:
        raise ValueError(f"Expected (H,W,4) or (4,H,W); got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _ensure_hw4_for_orix(arr: np.ndarray) -> np.ndarray:
    """(H,W,4) or (4,H,W) -> (H,W,4) for Orix."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D quaternion array; got {arr.shape}")
    return arr if arr.shape[-1] == 4 else np.moveaxis(arr, 0, -1)


def _to_torch_4hw(arr: np.ndarray) -> torch.Tensor:
    """Safe tensor from (4,H,W) numpy (handles read-only memmaps)."""
    if (
        (arr.dtype != np.float32)
        or (not arr.flags["C_CONTIGUOUS"])
        or (not arr.flags["WRITEABLE"])
    ):
        arr = np.array(arr, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(arr)


# Aliases → Orix symmetry classes
_SYM_ALIASES = {
    "oh": "Oh",
    "cubic": "Oh",
    "fcc": "Oh",
    "bcc": "Oh",
    "m-3m": "Oh",
    "hcp": "D6h",
    "hex": "D6h",
    "6/mmm": "D6h",
    "d6h": "D6h",
    "d4h": "D4h",
    "d3d": "D3d",
    "d2h": "D2h",
    "td": "Td",
    "o": "O",
}


def _resolve_symmetry(symmetry: Union[str, object]):
    if not isinstance(symmetry, str):
        return symmetry
    key = symmetry.strip().lower()
    canon = _SYM_ALIASES.get(key, symmetry.strip())
    if hasattr(SYM, canon):
        return getattr(SYM, canon)
    tname = canon[:1].upper() + canon[1:]
    if hasattr(SYM, tname):
        return getattr(SYM, tname)
    raise ValueError(f"Unknown symmetry '{symmetry}'")


_DIRS = {
    "X": Vector3d((1, 0, 0)) if _HAS_ORIX else None,
    "Y": Vector3d((0, 1, 0)) if _HAS_ORIX else None,
    "Z": Vector3d((0, 0, 1)) if _HAS_ORIX else None,
}


class QuaternionPairDataset(Dataset):
    """
    Matches LR and HR quaternion .npy files by the last integer in filenames.
    Returns:
      LR: torch.float32 (4,h,w)
      HR: torch.float32 (4,H,W)
    Also provides IPF previews via Orix (single symmetry per dataset).
    """

    def __init__(
        self,
        lr_glob: str,
        hr_glob: str,
        take_first: Optional[int] = None,
        symmetry: Union[str, object] = "Oh",  # one symmetry for the dataset
    ):
        lr_files = sorted(glob.glob(lr_glob, recursive=True), key=_last_int_key)
        hr_files = sorted(glob.glob(hr_glob, recursive=True), key=_last_int_key)
        if not lr_files:
            raise FileNotFoundError(f"No LR files matched glob:\n  {lr_glob}")
        if not hr_files:
            raise FileNotFoundError(f"No HR files matched glob:\n  {hr_glob}")

        lr_map = {k: f for f in lr_files if (k := _last_int_key(f)) >= 0}
        hr_map = {k: f for f in hr_files if (k := _last_int_key(f)) >= 0}
        common = sorted(lr_map.keys() & hr_map.keys())
        if not common:
            raise FileNotFoundError("No matching LR/HR quaternion .npy pairs found.")

        if take_first is not None:
            common = common[:take_first]

        self.pairs: List[Tuple[str, str]] = [(lr_map[k], hr_map[k]) for k in common]

        # Orix setup (if available)
        self._has_orix = _HAS_ORIX
        if self._has_orix:
            self.sym_class = _resolve_symmetry(symmetry)  # e.g., Oh / D6h
            self.ckey = orix_plot.IPFColorKeyTSL(self.sym_class.laue)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lr_fp, hr_fp = self.pairs[idx]
        lr_np = _as_hw4(np.load(lr_fp, mmap_mode="r"))  # (4,h,w)
        hr_np = _as_hw4(np.load(hr_fp, mmap_mode="r"))  # (4,H,W)
        return _to_torch_4hw(lr_np), _to_torch_4hw(hr_np)

    # -------- IPF preview (single index) --------
    def save_ipf_preview(
        self,
        idx: int,
        out_png: str,
        which: str = "HR",
        ref_dir: str = "ALL",
        include_key: bool = True,
    ):
        if not self._has_orix:
            raise RuntimeError("Orix is not available in this environment.")
        lr_fp, hr_fp = self.pairs[idx]
        arr = np.load(hr_fp if which.upper() == "HR" else lr_fp, mmap_mode="r")
        arr_hw4 = _ensure_hw4_for_orix(arr)  # (H,W,4)

        ori = Orientation(arr_hw4)
        ori.symmetry = self.sym_class

        ckey = self.ckey
        show_all = ref_dir.upper() == "ALL"
        ncols = 3 if show_all else 1

        key_cols = 1 if include_key else 0
        fig_cols = ncols + key_cols
        wr = [1] * ncols + ([0.9] if include_key else [])
        fig = plt.figure(
            constrained_layout=False,
            figsize=(5.2 * ncols + (2.6 if include_key else 0), 4.8),
        )
        gs = fig.add_gridspec(1, fig_cols, width_ratios=wr, wspace=0.05)
        axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]

        if show_all:
            for name, ax in zip(("X", "Y", "Z"), axes):
                ckey.direction = _DIRS[name]
                img = ckey.orientation2color(~ori)  # ~ori: lab->crystal
                ax.imshow(img)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"IPF-{name}")
                ax.axis("off")
        else:
            ref = ref_dir.upper()
            if ref not in _DIRS:
                raise ValueError("ref_dir must be 'X','Y','Z', or 'ALL'")
            ckey.direction = _DIRS[ref]
            img = ckey.orientation2color(~ori)
            ax = axes[0]
            ax.imshow(img)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"IPF-{ref}")
            ax.axis("off")

        if include_key:
            ax_ipf = fig.add_subplot(
                gs[0, -1], projection="ipf", symmetry=ori.symmetry.laue
            )
            ax_ipf.plot_ipf_color_key()
            ax_ipf.set_title("")
            for txt in getattr(ax_ipf, "texts", []):
                txt.set_fontsize(12)

        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    # -------- batch helper for many previews --------
    def save_ipf_many(
        self,
        indices: Union[range, List[int]],
        out_dir: str,
        which: str = "HR",
        ref_dir: str = "Z",
        include_key: bool = False,
        prefix: str = "ipf",
    ):
        os.makedirs(out_dir, exist_ok=True)
        for i in indices:
            out_png = os.path.join(
                out_dir, f"{prefix}_{which}_{ref_dir}_idx{i:04d}.png"
            )
            self.save_ipf_preview(
                i, out_png, which=which, ref_dir=ref_dir, include_key=include_key
            )


# =========================================================
# Training
# =========================================================


def _repeat_quat_blocks(x: torch.Tensor, n_feats: int) -> torch.Tensor:
    return x if n_feats == 1 else x.repeat(1, n_feats, 1, 1)


def train_with_config(cfg: Dict[str, Any]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    os.makedirs(cfg["save"], exist_ok=True)

    # Dataset & loaders (single symmetry per dataset; change if needed)
    default_sym = cfg.get("symmetry", "Oh")  # add "symmetry" to config if you want
    train_set = QuaternionPairDataset(
        cfg["lr_glob"],
        cfg["hr_glob"],
        take_first=cfg.get("train_count"),
        symmetry=default_sym,
    )
    val_cap = (cfg.get("train_count", 0) or 0) + (cfg.get("val_count", 0) or 0)
    val_set = QuaternionPairDataset(
        cfg["lr_glob"], cfg["hr_glob"], take_first=val_cap or None, symmetry=default_sym
    )
    if cfg.get("val_count"):
        val_set.pairs = val_set.pairs[-cfg["val_count"] :]
    else:
        val_set.pairs = val_set.pairs[-200:]

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    # Group reps (B = n_feats quaternion blocks)

    rho, rho_inv = make_group_tensors_from_orix(
        train_set.sym_class,
        num_blocks=cfg["n_feats"],
        device=device,
        dtype=dtype,
    )

    # Model (quat->quat)
    model = UpsamplerQuaternionTransposeConv(
        kernel_size=cfg["kernel_size"],
        scale=cfg["scale"],
        n_feats=cfg["n_feats"],
        group_tensor=rho,
        group_tensor_inv=rho_inv,
        dropout_prob=0.0,
    ).to(device=device, dtype=dtype)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    best_psnr = -1.0
    vis_every = int(cfg.get("vis_every", 1))

    for epoch in range(1, cfg["epochs"] + 1):
        # ---------------- Train ----------------
        model.train()
        total_loss = 0.0
        for LR_q, HR_q in train_loader:
            LR_q = LR_q.to(device=device, dtype=dtype, non_blocking=True)
            HR_q = HR_q.to(device=device, dtype=dtype, non_blocking=True)
            x = _repeat_quat_blocks(LR_q, n_feats=cfg["n_feats"])

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                SR_q = model(x)  # (B,4,H,W)
                SR_q_aligned = _hemisphere_align(
                    SR_q.permute(0, 2, 3, 1), HR_q.permute(0, 2, 3, 1)
                ).permute(0, 3, 1, 2)
                loss = _quat_loss_L1(SR_q_aligned, HR_q)

            scaler.scale(loss).backward()
            if cfg["clip"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item() * LR_q.size(0)

        sched.step()
        train_loss = total_loss / len(train_set)

        # ---------------- Validate + PSNR ----------------
        model.eval()
        mse_sum, n_elems = 0.0, 0
        first_pair = None
        with torch.no_grad():
            for LR_q, HR_q in val_loader:
                LR_q = LR_q.to(device=device, dtype=dtype, non_blocking=True)
                HR_q = HR_q.to(device=device, dtype=dtype, non_blocking=True)
                x = _repeat_quat_blocks(LR_q, n_feats=cfg["n_feats"])
                SR_q = model(x).clamp(-1.0, 1.0)
                SR_q_aligned = _hemisphere_align(
                    SR_q.permute(0, 2, 3, 1), HR_q.permute(0, 2, 3, 1)
                ).permute(0, 3, 1, 2)

                if first_pair is None:
                    first_pair = (
                        HR_q[0].detach().cpu(),
                        SR_q_aligned[0].detach().cpu(),
                    )

                SR_01 = (SR_q_aligned + 1.0) * 0.5
                HR_01 = (HR_q + 1.0) * 0.5
                mse_sum += F.mse_loss(SR_01, HR_01, reduction="sum").item()
                n_elems += int(np.prod(HR_01.shape))

        mse = mse_sum / n_elems
        cur_psnr = psnr_from_mse(mse, max_val=1.0)
        print(
            f"Epoch {epoch:03d} | train L1(quat): {train_loss:.4f} | PSNR(quat comps): {cur_psnr:.2f} dB"
        )

        if (epoch % vis_every == 0) and (first_pair is not None):
            save_img = os.path.join(cfg["save"], f"val_epoch_{epoch:03d}.png")
            show_sr_vs_gt_quat(
                first_pair[0], first_pair[1], psnr_val=cur_psnr, save_path=save_img
            )
            print(f"  Saved visualization: {save_img}")

        if cur_psnr > best_psnr:
            best_psnr = cur_psnr
            ckpt_path = os.path.join(
                cfg["save"], f"best_quat_sr_{cfg['group']}_x{cfg['scale']}.pt"
            )
            torch.save(
                {"model": model.state_dict(), "config": cfg, "psnr": best_psnr},
                ckpt_path,
            )
            print(f"  Saved: {ckpt_path}")

    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB")


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg = json.load(f)

    cfg.setdefault(
        "lr_glob",
        "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Train/LR_Images/Open_718_Train_lr_x_block_0.npy",
    )

    cfg.setdefault(
        "hr_glob",
        "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Train/HR_Images/Open_718_Train_hr_x_block_0.npy",
    )
    # "lr_glob": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/LR_Images/**/Open_718_Train_lr_x_normal_*.npy",
    # "hr_glob": "/data/warren/materials/materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/HR_Images/**/Open_718_Train_hr_x_normal_*.npy",
    cfg.setdefault("train_count", None)
    cfg.setdefault("val_count", 200)
    cfg.setdefault("symmetry", "Oh")  # dataset-wide symmetry for IPF

    # Example: quick dataset check + IPF previews (comment out for pure training)
    ds = QuaternionPairDataset(
        lr_glob=cfg["lr_glob"],
        hr_glob=cfg["hr_glob"],
        take_first=10 if cfg.get("train_count") is None else cfg["train_count"],
        symmetry=cfg.get("symmetry", "Oh"),
    )

    print("Pairs:", len(ds))
    lr, hr = ds[0]
    print("LR:", lr.shape, lr.dtype, "HR:", hr.shape, hr.dtype)

    # Save fast IPF examples if Orix is available
    if _HAS_ORIX:
        out_dir = os.path.join(cfg["save"], "ipf_examples")
        os.makedirs(out_dir, exist_ok=True)
        # Single direction Z (fast)
        ds.save_ipf_preview(
            0,
            os.path.join(out_dir, "lr_ipf.png"),
            which="LR",
            ref_dir="ALL",
            include_key=True,
        )
        # ALL directions with color key (nicer)
        ds.save_ipf_preview(
            0,
            os.path.join(out_dir, "hr_ipf_ALL.png"),
            which="HR",
            ref_dir="ALL",
            include_key=True,
        )

    # Kick off training
    # train_with_config(cfg)
