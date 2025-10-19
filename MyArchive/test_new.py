"""
train_eqsr_quat_pairs.py
Quaternion super-resolution on paired LR/HR .npy files (quaternions).

CONFIG (config.json)
{
  "epochs": 5,
  "batch_size": 8,
  "lr": 0.0003,
  "scale": 4,
  "n_feats": 1,
  "kernel_size": 3,
  "group": "432",
  "include_improper": true,
  "save": "checkpoints",
  "amp": false,
  "clip": 1.0,
  "vis_every": 1,

  "lr_glob": "/data/warrenz/materials/fz_reduced/Open_718_Z_Upsampling/Train/LR_Images/**/Open_718_Train_lr_x_normal_*.npy",
  "hr_glob": "/data/warrenz/materials/fz_reduced/Open_718_Z_Upsampling/Train/HR_Images/**/Open_718_Train_hr_x_normal_*.npy",

  "train_count": 2000,
  "val_count": 200
}
"""

import os, re, json, math, glob
from typing import Any, Dict, Tuple, List

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
# Utilities: PSNR + quaternion helpers + visualization
# =========================================================


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val**2) / mse)


def _unit_quat(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x: (..., 4)
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / n


def _hemisphere_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Flip pred -> -pred where dot(pred, gt) < 0 to remove q/~q ambiguity.
    pred, gt: (..., 4)
    """
    d = (pred * gt).sum(dim=-1, keepdim=True)
    sign = torch.where(
        d < 0,
        torch.tensor(-1.0, device=pred.device, dtype=pred.dtype),
        torch.tensor(1.0, device=pred.device, dtype=pred.dtype),
    )
    return pred * sign


def _quat_loss_L1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    pred, gt: (N,4,H,W). Both assumed unit & hemisphere-aligned already.
    """
    return (pred - gt).abs().mean()


def _quat_to_rgb_preview(q: torch.Tensor) -> torch.Tensor:
    """
    Quick visualization: map quaternion (a,b,c,d) to RGB in [0,1].
    Here we just take (b,c,d) -> normalized to [0,1], ignoring 'a' for color.
    q: (4,H,W) or (N,4,H,W)
    """
    single = q.dim() == 3
    if single:
        q = q.unsqueeze(0)  # (1,4,H,W)
    b, c, d = q[:, 1], q[:, 2], q[:, 3]  # (N,H,W)
    # components are in [-1,1]; map to [0,1]
    R = (b + 1.0) * 0.5
    G = (c + 1.0) * 0.5
    B = (d + 1.0) * 0.5
    rgb = torch.stack([R, G, B], dim=1)  # (N,3,H,W)
    if single:
        rgb = rgb[0]
    return rgb.clamp(0, 1)


def show_sr_vs_gt_quat(
    HR_q: torch.Tensor,  # (4,H,W)
    SR_q: torch.Tensor,  # (4,H,W)
    psnr_val: float = None,
    save_path: str = None,
):
    """
    Visualize by coloring quaternions via (b,c,d)->RGB.
    """
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
        plt.savefig(save_path, dpi=150)
    plt.show()
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
        """
        n_feats: number of quaternion blocks (input channels = 4*n_feats). For direct LR quat -> SR quat, set n_feats=1.
        """
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

        # Final head to 4 channels (quaternion)
        self.head = nn.Conv2d(4 * n_feats, 4, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (N, 4*n_feats, h, w)
        x = self.conv_layer(x)
        x = self.dropout(x)
        x = self.transposed_conv(x)
        x = self.post_conv_layer(x)
        y = self.head(x)  # (N, 4, H*scale, W*scale)
        # Normalize to unit quaternions per-pixel
        y = y.permute(0, 2, 3, 1)  # (N,H,W,4)
        y = _unit_quat(y).permute(0, 3, 1, 2)  # back to (N,4,H,W)
        return y


# =========================================================
# Dataset: paired LR/HR quaternion .npy files
# =========================================================


def _last_int_key(fp: str) -> int:
    m = re.findall(r"(\d+)(?=\.npy$)", os.path.basename(fp))
    return int(m[-1]) if m else -1


def _ensure_quat_hw4(arr: np.ndarray) -> np.ndarray:
    # Accept [H,W,4] or [4,H,W] or [H,W]x4 stacked; convert to [4,H,W]
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = np.moveaxis(arr, -1, 0)  # (4,H,W)
    elif arr.ndim == 3 and arr.shape[0] == 4:
        pass  # already (4,H,W)
    else:
        raise ValueError(
            f"Expected quaternion array with 4 channels; got shape {arr.shape}"
        )
    # Normalize & hemisphere (a>=0)
    H, W = arr.shape[1], arr.shape[2]
    q = arr.reshape(4, -1).T
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n == 0] = 1.0
    q = q / n
    # hemisphere on 'a' (scalar) component
    sign = np.where(q[:, 0:1] < 0.0, -1.0, 1.0)
    q = q * sign
    return q.T.reshape(4, H, W).astype(np.float32)


class QuaternionPairDataset(Dataset):
    """
    Matches LR and HR files by the last integer in their filename.
    Returns tensors:
      LR: (4,h,w), HR: (4,H,W)
    """

    def __init__(self, lr_glob: str, hr_glob: str, take_first: int = None):
        lr_files = sorted(glob.glob(lr_glob, recursive=True), key=_last_int_key)
        hr_files = sorted(glob.glob(hr_glob, recursive=True), key=_last_int_key)
        lr_map = {_last_int_key(f): f for f in lr_files}
        hr_map = {_last_int_key(f): f for f in hr_files}
        common = sorted(set(lr_map.keys()) & set(hr_map.keys()))
        if not common:
            raise FileNotFoundError("No matching LR/HR quaternion .npy pairs found.")
        if take_first is not None:
            common = common[:take_first]
        self.pairs: List[Tuple[str, str]] = [(lr_map[k], hr_map[k]) for k in common]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_fp, hr_fp = self.pairs[idx]
        lr = _ensure_quat_hw4(np.load(lr_fp))
        hr = _ensure_quat_hw4(np.load(hr_fp))
        return torch.from_numpy(lr), torch.from_numpy(hr)


# =========================================================
# Training
# =========================================================


def _repeat_quat_blocks(x: torch.Tensor, n_feats: int) -> torch.Tensor:
    # x: (N,4,h,w) -> (N,4*n_feats,h,w)
    if n_feats == 1:
        return x
    return x.repeat(1, n_feats, 1, 1)


def train_with_config(cfg: Dict[str, Any]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    os.makedirs(cfg["save"], exist_ok=True)

    # Dataset & loaders
    train_set = QuaternionPairDataset(
        cfg["lr_glob"], cfg["hr_glob"], take_first=cfg.get("train_count")
    )
    val_set = QuaternionPairDataset(
        cfg["lr_glob"],
        cfg["hr_glob"],
        take_first=cfg.get("train_count", 0) + cfg.get("val_count", 0),
    )
    # keep only the last val_count samples for val
    if cfg.get("val_count"):
        val_set.pairs = val_set.pairs[-cfg["val_count"] :]
    else:
        # default small val
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
    rho, rho_inv = make_group_tensors(
        num_blocks=cfg["n_feats"],
        group_name=cfg["group"],
        include_improper=bool(cfg["include_improper"]),
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
            LR_q = LR_q.to(device=device, dtype=dtype, non_blocking=True)  # (B,4,h,w)
            HR_q = HR_q.to(device=device, dtype=dtype, non_blocking=True)  # (B,4,H,W)

            x = _repeat_quat_blocks(LR_q, n_feats=cfg["n_feats"])  # (B,4*n_feats,h,w)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                SR_q = model(x)  # (B,4,H,W), already unit-norm
                # hemisphere align to GT before loss
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

        # ---------------- Validate + PSNR (on quaternion channels) ----------------
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

                # Save one example for visualization
                if first_pair is None:
                    first_pair = (
                        HR_q[0].detach().cpu(),
                        SR_q_aligned[0].detach().cpu(),
                    )

                # PSNR on components mapped to [0,1]
                SR_01 = (SR_q_aligned + 1.0) * 0.5
                HR_01 = (HR_q + 1.0) * 0.5
                mse_sum += F.mse_loss(SR_01, HR_01, reduction="sum").item()
                n_elems += np.prod(HR_01.shape)

        mse = mse_sum / n_elems
        cur_psnr = psnr_from_mse(mse, max_val=1.0)

        print(
            f"Epoch {epoch:03d} | train L1(quat): {train_loss:.4f} | PSNR(quat comps): {cur_psnr:.2f} dB"
        )

        # ------------- Visualize (first sample) -------------
        if (epoch % vis_every == 0) and (first_pair is not None):
            save_img = os.path.join(cfg["save"], f"val_epoch_{epoch:03d}.png")
            show_sr_vs_gt_quat(
                first_pair[0], first_pair[1], psnr_val=cur_psnr, save_path=save_img
            )
            print(f"  Saved visualization: {save_img}")

        # ---------------- Save best ----------------
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
# Entry point: load config.json
# =========================================================


def _last_int_key(fp: str) -> int:
    m = re.findall(r"(\d+)(?=\.npy$)", os.path.basename(fp))
    return int(m[-1]) if m else -1


if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg = json.load(f)

    # Sanity defaults for new keys
    cfg.setdefault("train_count", None)
    cfg.setdefault("val_count", 200)

    a = np.load(
        "/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/Open_718_Test_hr_x_block_0.npy"
    )

    # quat_dataset.py
    import os
    import re
    import glob
    from typing import List, Tuple, Optional

    import numpy as np
    import torch
    from torch.utils.data import Dataset

    # -----------------------
    # Filename matching utils
    # -----------------------

    def _last_int_key(fp: str) -> int:
        """Return the last integer before .npy in filename; -1 if none."""
        m = re.findall(r"(\d+)(?=\.npy$)", os.path.basename(fp))
        return int(m[-1]) if m else -1

    # -----------------------
    # Quaternion I/O helpers
    # -----------------------

    def _ensure_quat_hw4(arr: np.ndarray) -> np.ndarray:
        """
        Accept [H,W,4] or [4,H,W]; return float32 (4,H,W).
        Also unit-normalize and enforce hemisphere (a >= 0).
        """
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array with 4 channels; got {arr.shape}")

        # Put channels first -> (4,H,W)
        if arr.shape[-1] == 4:
            arr = np.moveaxis(arr, -1, 0)
        elif arr.shape[0] == 4:
            pass
        else:
            raise ValueError(
                f"Expected quaternion array with 4 channels; got {arr.shape}"
            )

        # Normalize to unit quaternions; enforce hemisphere on scalar component 'a'
        q = arr.reshape(4, -1).T  # (N,4)
        n = np.linalg.norm(q, axis=1, keepdims=True)
        n[n == 0.0] = 1.0
        q = q / n
        sign = np.where(q[:, 0:1] < 0.0, -1.0, 1.0)
        q = q * sign

        H, W = arr.shape[1], arr.shape[2]
        # Return a base ndarray (C-contiguous) float32
        return np.array(q.T.reshape(4, H, W), dtype=np.float32, order="C", copy=True)

    # -----------------------
    # Simple IPF-like preview
    # (cubic, ref dir = Z)
    # -----------------------

    def _quat_to_R3(a: float, b: float, c: float, d: float) -> np.ndarray:
        """Quaternion (a,b,c,d) -> 3x3 rotation (scalar-first)."""
        R = np.empty((3, 3), dtype=np.float32)
        R[0, 0] = 1 - 2 * (c * c + d * d)
        R[0, 1] = 2 * (b * c - a * d)
        R[0, 2] = 2 * (b * d + a * c)
        R[1, 0] = 2 * (b * c + a * d)
        R[1, 1] = 1 - 2 * (b * b + d * d)
        R[1, 2] = 2 * (c * d - a * b)
        R[2, 0] = 2 * (b * d - a * c)
        R[2, 1] = 2 * (c * d + a * b)
        R[2, 2] = 1 - 2 * (b * b + c * c)
        return R

    def quat_ipf_rgb(
        q4_hw: np.ndarray, ref_dir: str = "Z", gamma: float = 0.5
    ) -> np.ndarray:
        """
        Quick IPF-like color for cubic: map crystal direction (g^T * ref) to |[h k l]|.
        q4_hw: (4,H,W) unit quats with a>=0. Returns uint8 (H,W,3).
        """
        assert q4_hw.shape[0] == 4
        a, b, c, d = q4_hw[0], q4_hw[1], q4_hw[2], q4_hw[3]  # (H,W)

        # Sample (lab) reference direction
        if ref_dir.upper().startswith("X"):
            n = np.array([1, 0, 0], dtype=np.float32)
        elif ref_dir.upper().startswith("Y"):
            n = np.array([0, 1, 0], dtype=np.float32)
        else:
            n = np.array([0, 0, 1], dtype=np.float32)  # Z default

        H, W = a.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)

        # Loop per pixel (fast enough for previews up to ~1k^2)
        for y in range(H):
            for x in range(W):
                R = _quat_to_R3(
                    float(a[y, x]), float(b[y, x]), float(c[y, x]), float(d[y, x])
                )
                cdir = R.T @ n  # crystal dir corresponding to sample ref
                v = np.abs(cdir)
                s = v.sum()
                if s > 0:
                    v = v / s  # simplex normalize for simple color
                v = np.clip(v, 0.0, 1.0) ** gamma
                rgb[y, x, :] = v

        return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    # -----------------------
    # The dataset (final)
    # -----------------------

    class QuaternionPairDataset(Dataset):
        """
        Matches LR and HR quaternion .npy files by the last integer in their filenames.
        Returns:
        LR: torch.float32 (4,h,w), HR: torch.float32 (4,H,W)
        """

        def __init__(
            self, lr_glob: str, hr_glob: str, take_first: Optional[int] = None
        ):
            lr_files = sorted(glob.glob(lr_glob, recursive=True), key=_last_int_key)
            hr_files = sorted(glob.glob(hr_glob, recursive=True), key=_last_int_key)

            if not lr_files:
                raise FileNotFoundError(f"No LR files matched glob:\n  {lr_glob}")
            if not hr_files:
                raise FileNotFoundError(f"No HR files matched glob:\n  {hr_glob}")

            lr_map = {_last_int_key(f): f for f in lr_files if _last_int_key(f) >= 0}
            hr_map = {_last_int_key(f): f for f in hr_files if _last_int_key(f) >= 0}

            common = sorted(set(lr_map.keys()) & set(hr_map.keys()))
            if not common:
                raise FileNotFoundError(
                    "No matching LR/HR quaternion .npy pairs found.\n"
                    f"LR examples: {lr_files[:3]}\nHR examples: {hr_files[:3]}"
                )

            if take_first is not None:
                common = common[:take_first]

            self.pairs: List[Tuple[str, str]] = [(lr_map[k], hr_map[k]) for k in common]

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, idx: int):
            lr_fp, hr_fp = self.pairs[idx]

            # Ensure plain ndarray (copy=True guarantees base type), then to torch
            lr_np = np.array(
                _ensure_quat_hw4(np.load(lr_fp)), dtype=np.float32, order="C", copy=True
            )
            hr_np = np.array(
                _ensure_quat_hw4(np.load(hr_fp)), dtype=np.float32, order="C", copy=True
            )

            lr_t = torch.tensor(lr_np, dtype=torch.float32)  # (4,h,w)
            hr_t = torch.tensor(hr_np, dtype=torch.float32)  # (4,H,W)
            return lr_t, hr_t

        # Optional: save an IPF-like preview (no Dream3D needed)
        def save_ipf_preview(
            self, idx: int, out_png: str, which: str = "HR", ref_dir: str = "Z"
        ):
            lr_fp, hr_fp = self.pairs[idx]
            arr = _ensure_quat_hw4(np.load(hr_fp if which.upper() == "HR" else lr_fp))
            img = quat_ipf_rgb(arr, ref_dir=ref_dir)
            os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
            from PIL import Image

            Image.fromarray(img, mode="RGB").save(out_png)

    # -----------------------
    # Minimal example
    # -----------------------

    if __name__ == "__main__":
        ds = QuaternionPairDataset(
            lr_glob="/data/warren/materials/materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/LR_Images/**/Open_718_Train_lr_x_normal_*.npy",
            hr_glob="/data/warren/materials/materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/HR_Images/**/Open_718_Train_hr_x_normal_*.npy",
            take_first=10,  # or None
        )

        print("Pairs:", len(ds))

        lr, hr = ds[0]
        print("LR:", lr.shape, lr.dtype, "HR:", hr.shape, hr.dtype)

        # Save quick previews
        ds.save_ipf_preview(0, "lr_ipf.png", which="LR", ref_dir="Z")
        ds.save_ipf_preview(0, "hr_ipf.png", which="HR", ref_dir="Z")

        # train_with_config(cfg)

        # ds = QuaternionPairDataset(
        #     lr_glob="/data/warren/materials/materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/LR_Images/**/Open_718_Train_lr_x_normal_*.npy",
        #     hr_glob="/data/warren/materials/materials_data_mount/fz_reduced/Open_718_Z_Upsampling/Train/HR_Images/**/Open_718_Train_hr_x_normal_*.npy",
        #     take_first=50,  # or None
        # )

        ds = QuaternionPairDataset(
            lr_glob="/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/LR_Images/Open_718_Test_lr_x_block_*.npy",
            hr_glob="/data/warren/materials/materials_data_mount/fz_reduced/Open_718/Test/HR_Images/Open_718_Test_hr_x_block_*.npy",
            take_first=50,  # or None
        )
        # A training sample
        lr, hr = ds[0]  # lr: (4,h,w), hr: (4,H,W), float32, unit quats, a>=0

        # Save a quick IPF-like preview (cubic, ref Z) for HR
        ds.save_ipf_preview(0, "hr_ipf.png", which="HR", ref_dir="Z")

        # Same for LR
        ds.save_ipf_preview(0, "lr_ipf.png", which="LR", ref_dir="Z")
