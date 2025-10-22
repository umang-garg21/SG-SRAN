# -*- coding: utf-8 -*-
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Quaternion lift / project
# =============================================================================


def quat_to_lmat(q: torch.Tensor) -> torch.Tensor:
    """
    Lift quaternion image (B,4,H,W) -> (B,16,H,W) using left-multiplication matrix.
    """
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    L = torch.stack(
        [
            torch.stack([a, -b, -c, -d], dim=1),
            torch.stack([b, a, -d, c], dim=1),
            torch.stack([c, d, a, -b], dim=1),
            torch.stack([d, -c, b, a], dim=1),
        ],
        dim=1,
    )  # (B,4,4,H,W)
    return L.view(q.size(0), 16, q.size(2), q.size(3))


def lmat_to_quat(L: torch.Tensor) -> torch.Tensor:
    """
    Project (B,16,H,W) back to quaternion field (B,4,H,W) by taking the 1st column.
    """
    B, _, H, W = L.shape
    Lm = L.view(B, 4, 4, H, W)
    q = Lm[:, :, 0, :, :]  # (B,4,H,W)
    # unit quaternion normalization
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return q


# =============================================================================
# Quaternion conv helpers
# =============================================================================


def _fan_in_fan_out(weight: torch.Tensor):
    fan_in = weight.size(1)
    fan_out = weight.size(0)
    for s in weight.shape[2:]:
        fan_in *= s
        fan_out *= s
    return fan_in, fan_out


def _he_init_like(wr, wi, wj, wk, criterion: str = "glorot"):
    fan_in, fan_out = _fan_in_fan_out(wr)
    if criterion.lower() == "he":
        s = math.sqrt(2.0 / fan_in)
    else:
        s = math.sqrt(2.0 / (fan_in + fan_out))
    for p in (wr, wi, wj, wk):
        nn.init.normal_(p, mean=0.0, std=s / 2.0)


def quaternion_block_weight(r, i, j, k):
    """
    Build 4x4 real block lifting of quaternion kernel.
    Inputs r,i,j,k have shape (Cq_out, Cq_in/groups, kh, kw).
    Returns weight with shape (4*Cq_out, 4*(Cq_in/groups), kh, kw).
    """
    k_rr = torch.cat([r, -i, -j, -k], dim=1)
    k_ri = torch.cat([i, r, -k, j], dim=1)
    k_rj = torch.cat([j, k, r, -i], dim=1)
    k_rk = torch.cat([k, -j, i, r], dim=1)
    return torch.cat([k_rr, k_ri, k_rj, k_rk], dim=0)


# =============================================================================
# Quaternion Conv / TransposeConv (Torch-convention, grouped)
# =============================================================================


class QuaternionConv(nn.Module):
    """
    Quaternion convolution on lifted channels via real block kernel.
    in_q_channels, out_q_channels are multiples of 4 (real-lifted channels).
    Grouping preserves quaternion algebra: groups must equal in_q (i.e., in_channels/4).
    """

    def __init__(
        self,
        in_q_channels: int,
        out_q_channels: int,
        kernel_size: int | tuple,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        assert in_q_channels % 4 == 0 and out_q_channels % 4 == 0
        self.in_q = in_q_channels // 4
        self.out_q = out_q_channels // 4
        self.groups = self.in_q if groups is None else groups
        assert self.groups == self.in_q, f"groups must equal in_q={self.in_q}"

        kshape = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        # base quat kernels (Cq_out, Cq_in/groups, kh, kw)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4*in_q, H, W)
        w = quaternion_block_weight(
            self.r, self.i, self.j, self.k
        )  # -> (4*out_q, 4*(in_q/groups), kh, kw)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class QuaternionTransposeConv(nn.Module):
    """
    Quaternion transpose convolution (upsample). If overlap=False, picks kernel=stride,
    padding=0, output_padding=0 for exact scale-up. If overlap=True, allows kernel>stride
    which blends overlapping patches (warns + adjusts output_padding when possible).
    """

    def __init__(
        self,
        in_q_channels: int,
        out_q_channels: int,
        stride: int = 2,
        scale_factor: int | None = None,
        overlap: bool = False,
        kernel_size: int | None = None,
        padding: int | None = None,
        output_padding: int | None = None,
        dilation: int = 1,
        groups: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        assert in_q_channels % 4 == 0 and out_q_channels % 4 == 0
        self.in_q = in_q_channels // 4
        self.out_q = out_q_channels // 4
        self.groups = self.in_q if groups is None else groups
        assert self.groups == self.in_q, f"groups must equal in_q={self.in_q}"

        s = scale_factor if scale_factor is not None else stride
        self.stride = s

        if not overlap:
            kernel_size = s
            padding = 0
            output_padding = 0
        else:
            kernel_size = s + 3 if kernel_size is None else kernel_size
            padding = ((kernel_size - s) // 2) if padding is None else padding
            if kernel_size > s:
                warnings.warn(
                    f"[QuaternionTransposeConv] kernel_size ({kernel_size}) > stride ({s}) "
                    "→ overlapping patches, quaternion blending will occur."
                )
            # try to match output size
            req_out_pad = s - kernel_size + 2 * padding
            if req_out_pad >= 0:
                output_padding = (
                    req_out_pad if output_padding is None else output_padding
                )
            else:
                warnings.warn(
                    f"[QuaternionTransposeConv] required_output_padding={req_out_pad} < 0, "
                    "output shape may not match exactly."
                )
                output_padding = 0 if output_padding is None else output_padding

        self.padding = padding
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.dilation = dilation

        kshape = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        # base quat kernels (Cq_in, Cq_out/groups, kh, kw) for transpose conv
        wshape = (self.in_q, self.out_q // self.groups, *kshape)
        self.r = nn.Parameter(torch.empty(wshape))
        self.i = nn.Parameter(torch.empty(wshape))
        self.j = nn.Parameter(torch.empty(wshape))
        self.k = nn.Parameter(torch.empty(wshape))
        _he_init_like(self.r, self.i, self.j, self.k)

        self.bias = nn.Parameter(torch.zeros(out_q_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4*in_q, H, W)
        w = quaternion_block_weight(
            self.r, self.i, self.j, self.k
        )  # (4*in_q, 4*(out_q/groups), kh, kw)
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


# =============================================================================
# Optional: Residual block in lifted space (helps sharpen boundaries)
# =============================================================================


class QuaternionMagnitudeActivation(nn.Module):
    """
    Apply a scalar activation to the magnitude of each quaternion (w,x,y,z) group,
    then rescale:  q_out = act(||q||) * (q / (||q|| + eps))
    Expects input shape (B, C, H, W) with C % 4 == 0.
    """

    def __init__(self, act="relu", eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        if isinstance(act, str):
            act = act.lower()
            if act == "relu":
                self.act = F.relu
            elif act == "gelu":
                self.act = F.gelu
            elif act == "tanh":
                self.act = torch.tanh
            elif act == "sigmoid":
                self.act = torch.sigmoid
            elif act in ("none", "identity", "linear"):
                self.act = lambda x: x
            else:
                raise ValueError(f"Unknown act='{act}'")
        else:
            # a callable, e.g. functools.partial(F.leaky_relu, negative_slope=0.1)
            self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), C multiple of 4
        B, C, H, W = x.shape
        assert (
            C % 4 == 0
        ), f"QuaternionMagnitudeActivation expects channels %4==0, got {C}"
        q = x.view(B, C // 4, 4, H, W)  # (B, Q, 4, H, W)
        mag = torch.linalg.norm(q, dim=2, keepdim=True)  # (B, Q, 1, H, W)
        q_dir = q / (mag + self.eps)
        mag_act = self.act(mag)
        out = q_dir * mag_act
        return out.view(B, C, H, W)


class QuaternionUnitNorm(nn.Module):
    """
    Renormalize each quaternion group (w,x,y,z) to unit norm.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C % 4 == 0, f"QuaternionUnitNorm expects channels %4==0, got {C}"
        q = x.view(B, C // 4, 4, H, W)
        mag = torch.linalg.norm(q, dim=2, keepdim=True)
        q = q / (mag + self.eps)
        return q.view(B, C, H, W)


class QuaternionResBlock(nn.Module):
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        assert channels % 4 == 0, "channels must be multiple of 4"
        g = channels // 4
        self.block = nn.Sequential(
            QuaternionConv(channels, channels, k, padding=k // 2, groups=g),
            QuaternionMagnitudeActivation("relu"),  # <-- CHANGED
            QuaternionConv(channels, channels, k, padding=k // 2, groups=g),
            # Optional: QuaternionUnitNorm(),             # keep lifted quats tight
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# =============================================================================
# The SR network
# =============================================================================


class QuaternionSRNet(nn.Module):
    """
    Deeper, nonlinear quaternion SR net:
        enc (conv+relu)*2 → upsample (transpose conv) → refine (conv+relu → conv)
    All layers operate in the lifted (B,16,H,W) real space with quaternion-structured weights.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_ch = 16
        mid_ch = getattr(cfg, "n_feats", 32)
        out_ch = 16
        scale = getattr(cfg, "scale", 4)
        overlap = getattr(cfg, "overlap", False)
        k = getattr(cfg, "kernel_size", 3)
        use_res = getattr(cfg, "use_resblocks", 0)

        g_in = in_ch // 4
        g_mid = mid_ch // 4

        self.enc = nn.Sequential(
            QuaternionConv(in_ch, mid_ch, k, padding=k // 2, groups=g_in),
            QuaternionMagnitudeActivation("relu"),  # <-- CHANGED
            QuaternionConv(mid_ch, mid_ch, k, padding=k // 2, groups=g_mid),
            QuaternionMagnitudeActivation("relu"),  # <-- CHANGED
            # Optional: QuaternionUnitNorm(),
        )

        self.up = QuaternionTransposeConv(
            in_q_channels=mid_ch,
            out_q_channels=mid_ch,
            scale_factor=scale,
            overlap=overlap,
            groups=g_mid,
        )

        res_blocks = [QuaternionResBlock(mid_ch, k=k) for _ in range(use_res)]
        self.refine = nn.Sequential(
            *res_blocks,
            QuaternionConv(mid_ch, mid_ch, k, padding=k // 2, groups=g_mid),
            QuaternionMagnitudeActivation("relu"),  # <-- CHANGED
            QuaternionConv(mid_ch, out_ch, k, padding=k // 2, groups=g_mid),
            # (no activation before projection)
        )

    def forward(self, q_in: torch.Tensor) -> torch.Tensor:
        x = quat_to_lmat(q_in)
        x = self.enc(x)
        x = self.up(x)
        x = self.refine(x)
        q_out = lmat_to_quat(x)
        # final unit normalization in quaternion space (keep!)
        q_out = q_out / q_out.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return q_out


# =============================================================================
# Loss: rotational distance (in radians), supports (B,4,H,W) or (N,4)
# =============================================================================


# def rotational_distance_loss(
#     q_pred: torch.Tensor, q_target: torch.Tensor, eps: float = 1e-12
# ) -> torch.Tensor:
#     """
#     Quaternion geodesic (rotation angle) loss.
#     Handles (B,4,H,W) by flattening to (N,4); uses hemisphere trick (abs on scalar part).
#     """
#     if q_pred.dim() > 2:
#         qp = q_pred.permute(0, 2, 3, 1).reshape(-1, 4)
#         qt = q_target.permute(0, 2, 3, 1).reshape(-1, 4)
#     else:
#         qp, qt = q_pred, q_target

#     qp = F.normalize(qp, p=2, dim=1, eps=eps)
#     qt = F.normalize(qt, p=2, dim=1, eps=eps)

#     w1, x1, y1, z1 = qp[:, 0], qp[:, 1], qp[:, 2], qp[:, 3]
#     w2, x2, y2, z2 = qt[:, 0], qt[:, 1], qt[:, 2], qt[:, 3]

#     # r = qt ⊗ conj(qp)
#     rw = w2 * w1 + x2 * (-x1) + y2 * (-y1) + z2 * (-z1)
#     rx = w2 * (-x1) + x2 * w1 + y2 * (-z1) + z2 * (y1)
#     ry = w2 * (-y1) + x2 * (z1) + y2 * w1 + z2 * (-x1)
#     rz = w2 * (-z1) + x2 * (-y1) + y2 * (x1) + z2 * w1

#     rw = rw.abs()  # hemisphere
#     v_norm = torch.sqrt(rx * rx + ry * ry + rz * rz + eps)
#     angle = 2.0 * torch.atan2(v_norm, torch.clamp(rw, min=eps))
#     return angle.mean()


# def orientation_gradient_loss(
#     q_pred: torch.Tensor, q_target: torch.Tensor, eps: float = 1e-8
# ) -> torch.Tensor:
#     """
#     Orientation gradient loss: encourages SR output to match the boundary sharpness of HR.
#     Works on (B, 4, H, W) quaternion fields.

#     Computes finite differences in x and y directions, then uses L1 loss
#     between gradient magnitudes (orientation difference emphasis).
#     """
#     # Finite difference filters
#     kernel_x = torch.tensor([[[[-1, 1]]]], dtype=q_pred.dtype, device=q_pred.device)
#     kernel_y = torch.tensor([[[[-1], [1]]]], dtype=q_pred.dtype, device=q_pred.device)

#     def grad_mag(q):
#         gx = F.conv2d(q, kernel_x.expand(q.size(1), 1, 1, 2), groups=q.size(1))
#         gy = F.conv2d(q, kernel_y.expand(q.size(1), 1, 2, 1), groups=q.size(1))
#         # Quaternion gradient magnitude: L2 norm across channel and spatial gradient
#         gmag = torch.sqrt(gx.pow(2) + gy.pow(2) + eps)
#         return gmag

#     grad_pred = grad_mag(q_pred)
#     grad_target = grad_mag(q_target)

#     return F.l1_loss(grad_pred, grad_target)


# def rotational_distance_loss(q_pred, q_target, eps=1e-12, tol=1e-5):
#     if q_pred.dim() > 2:
#         qp = q_pred.permute(0, 2, 3, 1).reshape(-1, 4)
#         qt = q_target.permute(0, 2, 3, 1).reshape(-1, 4)
#     else:
#         qp, qt = q_pred, q_target

#     qp = F.normalize(qp, p=2, dim=1, eps=eps)
#     qt = F.normalize(qt, p=2, dim=1, eps=eps)

#     w1, x1, y1, z1 = qp[:, 0], qp[:, 1], qp[:, 2], qp[:, 3]
#     w2, x2, y2, z2 = qt[:, 0], qt[:, 1], qt[:, 2], qt[:, 3]

#     rw = w2 * w1 + x2 * (-x1) + y2 * (-y1) + z2 * (-z1)
#     rx = w2 * (-x1) + x2 * w1 + y2 * (-z1) + z2 * y1
#     ry = w2 * (-y1) + x2 * z1 + y2 * w1 + z2 * (-x1)
#     rz = w2 * (-z1) + x2 * (-y1) + y2 * x1 + z2 * w1

#     rw = rw.abs()
#     v_norm = torch.sqrt(rx * rx + ry * ry + rz * rz + eps)
#     angle = 2.0 * torch.atan2(v_norm, torch.clamp(rw, min=eps))

#     # Clamp numerical noise
#     angle = torch.where(angle < tol, torch.zeros_like(angle), angle)
#     return angle.mean()


# =============================================================================
# Quick shape test
# =============================================================================
if __name__ == "__main__":

    class Cfg:  # minimal config shim
        n_feats = 64
        scale = 4
        kernel_size = 3
        overlap = False
        use_resblocks = 1

    B, H, W = 2, 32, 32
    q_lr = torch.randn(B, 4, H, W)
    q_lr = q_lr / q_lr.norm(dim=1, keepdim=True).clamp_min(1e-8)

    net = QuaternionSRNet(Cfg())
    with torch.no_grad():
        q_sr = net(q_lr)
    print("LR:", q_lr.shape, "SR:", q_sr.shape)  # (B,4,32,32) -> (B,4,128,128)

    rotational_distance_loss(q_sr, q_sr)
    orientation_gradient_loss(q_pred=q_sr, q_target=q_sr + 4433355555555555)
