# -*-coding:utf-8 -*-
"""
File:        reynolds_qsr.py
Created at:  2025/10/18 14:24:26
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: Reynolds-averaged equivariant quaternion SR model
             with quaternion (transpose) convolutions and a group
             projection wrapper.
"""

from __future__ import annotations
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Quaternion primitives
# =============================================================================


def _infer_op_ndim(x: torch.Tensor) -> int:
    """Return {1,2,3} based on input dims (B,C,*spatial)."""
    if x.dim() == 3:  # (B,C,L)
        return 1
    if x.dim() == 4:  # (B,C,H,W)
        return 2
    if x.dim() == 5:  # (B,C,D,H,W)
        return 3
    raise ValueError(f"Expected input with 3/4/5 dims, got {x.shape}")


def quaternion_conv(
    x: torch.Tensor,
    r: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    groups,
    dilation,
):
    """
    Quaternion convolution via real-to-quat block lifting.
    x: (B, 4*C_in, *spatial)
    r,i,j,k: (C_out/4, C_in/4, k, k) with same shape
    """
    # Build 4x4 block kernel
    k_rr = torch.cat([r, -i, -j, -k], dim=1)
    k_ri = torch.cat([i, r, -k, j], dim=1)
    k_rj = torch.cat([j, k, r, -i], dim=1)
    k_rk = torch.cat([k, -j, i, r], dim=1)
    w = torch.cat([k_rr, k_ri, k_rj, k_rk], dim=0)  # (4*Cout/4, 4*Cin/4, k, k)

    ndim = _infer_op_ndim(x)
    if ndim == 1:
        return F.conv1d(x, w, bias, stride, padding, dilation, groups)
    if ndim == 2:
        return F.conv2d(x, w, bias, stride, padding, dilation, groups)
    return F.conv3d(x, w, bias, stride, padding, dilation, groups)


def quaternion_conv_transpose(
    x: torch.Tensor,
    r: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    """
    Quaternion transpose (deconv) via real-to-quat block lifting.
    """
    k_rr = torch.cat([r, -i, -j, -k], dim=1)
    k_ri = torch.cat([i, r, -k, j], dim=1)
    k_rj = torch.cat([j, k, r, -i], dim=1)
    k_rk = torch.cat([k, -j, i, r], dim=1)
    w = torch.cat([k_rr, k_ri, k_rj, k_rk], dim=0)

    ndim = _infer_op_ndim(x)
    if ndim == 1:
        return F.conv_transpose1d(
            x, w, bias, stride, padding, output_padding, groups, dilation
        )
    if ndim == 2:
        return F.conv_transpose2d(
            x, w, bias, stride, padding, output_padding, groups, dilation
        )
    return F.conv_transpose3d(
        x, w, bias, stride, padding, output_padding, groups, dilation
    )


def _fan_in_fan_out(weight: torch.Tensor):
    # shape (C_out, C_in, *k)
    if weight.dim() < 2:
        raise ValueError("Weight must have at least 2 dims")
    fan_in = weight.size(1)
    fan_out = weight.size(0)
    for s in weight.shape[2:]:
        fan_in *= s
        fan_out *= s
    return fan_in, fan_out


def _he_init_like(wr, wi, wj, wk, criterion="glorot"):
    """
    Simple quaternion-aware init: scale four parts to keep variance.
    """
    fan_in, fan_out = _fan_in_fan_out(wr)
    if criterion.lower() == "he":
        s = math.sqrt(2.0 / (fan_in))
    else:  # glorot
        s = math.sqrt(2.0 / (fan_in + fan_out))
    for p in (wr, wi, wj, wk):
        nn.init.normal_(p, mean=0.0, std=s / 2.0)


class QuaternionConv(nn.Module):
    """
    Quaternion convolution (1D/2D/3D depending on input).
    Expects channels to be multiples of 4 (w,x,y,z).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        init_criterion="glorot",
    ):
        super().__init__()
        assert (
            in_channels % 4 == 0 and out_channels % 4 == 0
        ), "QuaternionConv requires channels multiple of 4."
        self.in_q = in_channels // 4
        self.out_q = out_channels // 4

        if isinstance(kernel_size, int):
            kshape = (
                kernel_size,
            ) * 2  # 2D default; will still work for 1D/3D via F ops
        else:
            kshape = tuple(kernel_size)

        wshape = (self.out_q, self.in_q, *kshape)
        self.r = nn.Parameter(torch.empty(wshape))
        self.i = nn.Parameter(torch.empty(wshape))
        self.j = nn.Parameter(torch.empty(wshape))
        self.k = nn.Parameter(torch.empty(wshape))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        _he_init_like(self.r, self.i, self.j, self.k, init_criterion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quaternion_conv(
            x,
            self.r,
            self.i,
            self.j,
            self.k,
            self.bias,
            self.stride,
            self.padding,
            self.groups,
            self.dilation,
        )


class QuaternionTransposeConv(nn.Module):
    """
    Quaternion transpose convolution (deconv).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride=2,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        init_criterion="glorot",
    ):
        super().__init__()
        assert (
            in_channels % 4 == 0 and out_channels % 4 == 0
        ), "QuaternionTransposeConv requires channels multiple of 4."
        self.in_q = in_channels // 4
        self.out_q = out_channels // 4

        if isinstance(kernel_size, int):
            kshape = (kernel_size,) * 2
        else:
            kshape = tuple(kernel_size)

        wshape = (self.in_q, self.out_q, *kshape)
        self.r = nn.Parameter(torch.empty(wshape))
        self.i = nn.Parameter(torch.empty(wshape))
        self.j = nn.Parameter(torch.empty(wshape))
        self.k = nn.Parameter(torch.empty(wshape))

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        _he_init_like(self.r, self.i, self.j, self.k, init_criterion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quaternion_conv_transpose(
            x,
            self.r,
            self.i,
            self.j,
            self.k,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


# =============================================================================
# Reynolds-averaged equivariance wrapper
# =============================================================================


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

        # Lift: Bunge convention uses s⁻¹ ⊗ x, so apply group_tensor_inv
        x = x.view(B, n_feats, Cg, *spatial)  # (B,n,Cg,*)
        # gamma_x[b,g,n,c,...] = sum_i group_inv[g,c,i] * x[b,n,i,...] = s⁻¹ ⊗ x
        gamma_x = torch.einsum("gci,bni...->bgnc...", self.group_tensor_inv, x).reshape(
            B * G, n_feats * Cg, *spatial
        )

        # Apply wrapped op
        fx = self.fn(gamma_x)  # (B*G, Cout, *s')
        BG, Cout, *spatial_out = fx.shape
        assert BG == B * G and Cout % Cg == 0
        n_out = Cout // Cg

        fx = fx.view(B, G, n_out, Cg, *spatial_out)  # (B,G,n_out,Cg,*)
        # Project back: inverse of s⁻¹ is s, so apply group_tensor = s ⊗ y
        fx = torch.einsum("gci,bgni...->bgnc...", self.group_tensor, fx)

        # Average over group
        return fx.mean(dim=1).reshape(B, Cout, *spatial_out)


# =============================================================================
# Upsampler block (quat deconv + post conv)
# =============================================================================


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
        self.scale = scale

        # Expand features, then transpose-conv stride=scale, then refine
        self.pre = EquivariantReynoldsWrap(
            QuaternionConv(
                in_channels=n_feats,
                out_channels=(scale * scale) * n_feats,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            group_tensor,
            group_tensor_inv,
        )

        self.deconv = EquivariantReynoldsWrap(
            QuaternionTransposeConv(
                in_channels=(scale * scale) * n_feats,
                out_channels=n_feats,
                kernel_size=2 * scale,
                stride=scale,
                padding=scale // 2,
                output_padding=scale % 2,
            ),
            group_tensor,
            group_tensor_inv,
        )

        self.post = EquivariantReynoldsWrap(
            QuaternionConv(
                in_channels=n_feats,
                out_channels=n_feats,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            group_tensor,
            group_tensor_inv,
        )

        self.drop = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.drop(x)
        x = self.deconv(x)
        x = self.post(x)
        return x


# =============================================================================
# Model
# =============================================================================
def print_tensor_info(tensor: torch.Tensor, name: str):
    print(
        f"[DEBUG] {name:<20} | shape={tuple(tensor.shape)} | dtype={tensor.dtype} | device={tensor.device}"
    )


class Reynolds_QSR(nn.Module):
    """
    Simple SR backbone:
      head:   quat conv (4 -> n_feats)
      tail:   quat deconv upsampler + quat conv (n_feats -> 4)
    All modules wrapped with Reynolds equivariance.
    """

    def __init__(self, cfg):
        super().__init__()

        # ------------------------------------------------------------------
        # Load global symmetry group tensors
        # ------------------------------------------------------------------
        gt_path = "/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group.npy"
        gti_path = (
            "/data/home/umang/Materials/Reynolds-QSR/symmetry_groups/O_group_inv.npy"
        )
        gt = torch.tensor(np.load(gt_path), dtype=torch.float32)
        gti = torch.tensor(np.load(gti_path), dtype=torch.float32)
        self.register_buffer("group_tensor", gt)
        self.register_buffer("group_tensor_inv", gti)

        # ------------------------------------------------------------------
        # Read model hyperparameters
        # ------------------------------------------------------------------
        self.n_channels = 4
        self.n_feats = getattr(cfg, "n_feats", 32)
        self.kernel_size = getattr(cfg, "kernel_size", 3)
        self.dropout = getattr(cfg, "dropout", 0.0)
        self.scale = getattr(cfg, "scale", 4)

        self.head = nn.Sequential(
            EquivariantReynoldsWrap(
                QuaternionConv(
                    in_channels=self.n_channels,
                    out_channels=self.n_feats,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                ),
                self.group_tensor,
                self.group_tensor_inv,
            )
        )

        self.tail = nn.Sequential(
            EquivariantReynoldsWrap(
                UpsamplerQuaternionTransposeConv(
                    kernel_size=self.kernel_size,
                    scale=self.scale,
                    n_feats=self.n_feats,
                    group_tensor=self.group_tensor,
                    group_tensor_inv=self.group_tensor_inv,
                    dropout_prob=self.dropout,
                ),
                self.group_tensor,
                self.group_tensor_inv,
            ),
            EquivariantReynoldsWrap(
                QuaternionConv(
                    in_channels=self.n_feats,
                    out_channels=self.n_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                ),
                self.group_tensor,
                self.group_tensor_inv,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.tail(x)
        return x

    # allow partial load (ignore tail size mismatches when strict=False)
    def load_state_dict(self, state_dict, strict: bool = True):
        own = self.state_dict()
        for name, param in state_dict.items():
            if name in own:
                try:
                    own[name].copy_(
                        param if not isinstance(param, nn.Parameter) else param.data
                    )
                except Exception:
                    if strict and "tail" not in name:
                        raise
            elif strict and "tail" not in name:
                raise KeyError(f"Unexpected key in state_dict: {name}")


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":

    class _Args:
        n_resblocks = 0
        n_feats = 8
        scale = 4
        kernel_size = 3
        # Provide your npy paths
        sym_np_path = "/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group.npy"
        sym_inv_np_path = (
            "/home/warren/projects/Reynolds-QSR/symmetry_groups/O_group_inv.npy"
        )
        dropout = 0.0

    args = _Args()
    model = Reynolds_QSR(args).eval()
    x = torch.randn(3, 4, 64, 64)

    # quick forward
    with torch.no_grad():
        y = model(x)
    print("Output shape:", tuple(y.shape))

    def test_model_equivariance(
        model,
        x: torch.Tensor,
        n_check: int | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        """
        Test equivariance of the model under the loaded group action:
            f(g·x) ≈ g·f(x)

        Parameters
        ----------
        model : nn.Module
            Model with .group_tensor (G, Cg, Cg)
        x : torch.Tensor
            Input tensor (B, C, H, W)
        n_check : int or None
            Number of group elements to test; if None, checks all
        atol : float
            Absolute tolerance
        rtol : float
            Relative tolerance

        Returns
        -------
        passed : bool
            True if equivariance holds within tolerance
        max_err : float
            Maximum error across all group elements tested
        errs : list[float]
            Per-group element max error
        """
        model.eval()
        device = next(model.parameters()).device
        x = x.to(device)

        with torch.inference_mode():
            fx = model(x)
            G = model.group_tensor.shape[0]
            Cg = model.group_tensor.shape[1]

            if n_check is None or n_check > G:
                n_check = G
            idx = torch.arange(n_check)

            errs = []
            for g_idx in idx:
                gmat = model.group_tensor[g_idx].to(device)  # (Cg, Cg)

                # g·x
                gx = torch.einsum("ci,bi...->bc...", gmat, x)

                # f(g·x)
                f_gx = model(gx)

                # g·f(x)
                g_fx = torch.einsum("ci,bi...->bc...", gmat, fx)

                # max error for this element
                err = (f_gx - g_fx).abs().max().item()
                errs.append(err)

            max_err = max(errs)
            tol = atol + rtol * fx.abs().max().item()
            passed = max_err <= tol

        print(f"\n[Equivariance Test]")
        print(f"Checked {n_check}/{G} group elements")
        print(f"Tolerance: {tol:.3e}")
        print(f"Max error: {max_err:.3e}")
        print(f"Per-group errors: {[round(e, 6) for e in errs]}")
        print(f"✅ Equivariant: {passed}\n")

        return passed, max_err, errs

        return passed, max_err, errs

    passed, max_err, errs = test_model_equivariance(model, x, n_check=5)
    # # (Optional) quick equivariance smoke test (checks a few group elements)
    # def test_equivariance(m: Reynolds_QSR, x: torch.Tensor, n_check: int = 3):
    #     with torch.no_grad():
    #         fx = m(x)
    #         G = m.group_tensor.shape[0]
    #         idx = torch.linspace(0, G - 1, steps=min(n_check, G)).long()
    #         errs = []
    #         for g in idx:
    #             gmat = m.group_tensor[g]  # (4,4)
    #             gx = torch.einsum("ci,bi...->bc...", gmat, x)  # g·x
    #             f_gx = m(gx)
    #             g_fx = torch.einsum("ci,bi...->bc...", gmat, fx)  # g·f(x)
    #             errs.append((f_gx - g_fx).abs().max().item())
    #         return max(errs), errs

    # try:
    #     me, es = test_equivariance(model, x)
    #     print(f"Equivariance max-err (subset G): {me:.3e}")
    # except Exception as e:
    #     print("Equivariance test skipped:", e)


# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F
# import numpy as np

# # from model.quat_utils.Qops_with_QSN import conv2d, Residual_SA
# from model.quat_utils.quaternion_layers import QuaternionConv, QuaternionTransposeConv

# # from einops import rearrange
# # ─── requirements ───────────────────────────────────────────────────────────────
# # pip install torch e3nn==0.7.4              # e3nn just for rotation utilities
# # ────────────────────────────────────────────────────────────────────────────────
# import torch, torch.nn as nn
# from torchinfo import summary
# import os

# # os.environ["CUDA_VISIBLE_DEVICES"] = ""
# # device = torch.device("cpu")
# # torch.set_default_device(device)


# def make_model(args):
#     return Reynolds_QSR(args)


# class UpsamplerQuaternionTransposeConv(nn.Module):
#     def __init__(
#         self,
#         kernel_size,
#         scale,
#         n_feats,
#         group_tensor,
#         group_tensor_inv,
#         bn=False,
#         act=False,
#         bias=True,
#         dropout_prob=0.2,
#     ):
#         super(UpsamplerQuaternionTransposeConv, self).__init__()

#         self.scale = scale
#         self.n_feat = n_feats

#         self.conv_layer = EquivariantReynoldsWrap(
#             QuaternionConv(
#                 in_channels=n_feats,
#                 out_channels=scale * scale * n_feats,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 padding=kernel_size // 2,
#             ),
#             group_tensor=group_tensor,
#             group_tensor_inv=group_tensor_inv,
#         )

#         # Adding dropout layer after the convolution layer
#         self.dropout = nn.Dropout(p=dropout_prob)  # Dropout with specified probability

#         self.transposed_conv = EquivariantReynoldsWrap(
#             QuaternionTransposeConv(
#                 in_channels=scale * scale * n_feats,
#                 out_channels=n_feats,
#                 kernel_size=scale,
#                 stride=scale,
#                 padding=kernel_size // 2,
#                 output_padding=2,  # Adjust output padding
#             ),
#             group_tensor=group_tensor,
#             group_tensor_inv=group_tensor_inv,
#         )
#         self.post_conv_layer = EquivariantReynoldsWrap(
#             QuaternionConv(
#                 in_channels=n_feats,
#                 out_channels=n_feats,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 padding=kernel_size // 2,
#             ),
#             group_tensor=group_tensor,
#             group_tensor_inv=group_tensor_inv,
#         )

#     def forward(self, x):
#         try:

#             x = self.conv_layer(x)
#             x = self.transposed_conv(x)
#             x = self.post_conv_layer(x)
#             return x
#         except Exception as e:
#             print("Error in Upsampler2DQuaternionTransposeConv:", e)


# class EquivariantReynoldsWrap(nn.Module):
#     """
#     Reynolds operator wrapper: enforces equivariance for any module fn
#     under a group action represented by group_tensor (G, Cg, Cg).
#     Input/output channel dims must be multiples of Cg.
#     Works with inputs (B, C, *spatial) for 1D/2D/3D ops.
#     """

#     def __init__(
#         self, fn: nn.Module, group_tensor: torch.Tensor, group_tensor_inv: torch.Tensor
#     ):
#         super().__init__()
#         self.fn = fn
#         self.register_buffer("group_tensor", group_tensor)  # (G, Cg, Cg)
#         self.register_buffer("group_tensor_inv", group_tensor_inv)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, C, *spatial_in = x.shape
#         G, Cg, _ = self.group_tensor.shape
#         assert C % Cg == 0, f"Channels {C} must be multiple of {Cg}"
#         n_feats = C // Cg

#         # --- Lift: apply group action g·x ---
#         x = x.view(B, n_feats, Cg, *spatial_in)  # (B,n_feats,Cg,*spatial)
#         gamma_x = torch.einsum("gci,bni...->bgnc...", self.group_tensor, x)
#         gamma_x = gamma_x.reshape(B * G, n_feats * Cg, *spatial_in)  # (B*G,C,*spatial)

#         # --- Apply wrapped fn ---
#         fx = self.fn(gamma_x)  # (B*G,Cout,*spatial_out)
#         BGO, Cout, *spatial_out = fx.shape
#         assert BGO == B * G
#         assert Cout % Cg == 0, f"fn must output multiple of {Cg}, got {Cout}"
#         n_feats_out = Cout // Cg

#         # --- Project back with g⁻¹ ---
#         fx = fx.view(
#             B, G, n_feats_out, Cg, *spatial_out
#         )  # (B,G,n_feats_out,Cg,*spatial)
#         fx = torch.einsum("gci,bgni...->bgnc...", self.group_tensor_inv, fx)

#         # --- Average over group and return ---
#         return fx.mean(dim=1).reshape(B, Cout, *spatial_out)


# class Reynolds_QSR(nn.Module):
#     def __init__(self, args):
#         super(Reynolds_QSR, self).__init__()
#         n_resblocks = args.n_resblocks
#         n_channels = 4
#         n_feats = args.n_feats
#         scale = args.scale
#         kernel_size = 3

#         self.register_buffer(
#             "group_tensor", torch.tensor(np.load(args.sym_np_path), dtype=torch.float32)
#         )  # (G, C, C) where C=4

#         self.register_buffer(
#             "group_tensor_inv",
#             torch.tensor(np.load(args.sym_inv_np_path), dtype=torch.float32),
#         )  # (G, C, C) where C=4

#         m_head = [
#             EquivariantReynoldsWrap(
#                 QuaternionConv(
#                     in_channels=n_channels,
#                     out_channels=n_feats,
#                     kernel_size=kernel_size,
#                     stride=1,
#                     padding=kernel_size // 2,
#                 ),
#                 group_tensor=self.group_tensor,
#                 group_tensor_inv=self.group_tensor_inv,
#             ),
#         ]

#         m_tail = [
#             UpsamplerQuaternionTransposeConv(
#                 kernel_size=kernel_size,
#                 scale=scale,
#                 n_feats=n_feats,
#                 group_tensor=self.group_tensor,
#                 group_tensor_inv=self.group_tensor_inv,
#             ),
#             EquivariantReynoldsWrap(
#                 QuaternionConv(
#                     in_channels=n_feats,
#                     out_channels=n_channels,
#                     kernel_size=kernel_size,
#                     stride=1,
#                     padding=kernel_size // 2,
#                 ),
#                 group_tensor=self.group_tensor,
#                 group_tensor_inv=self.group_tensor_inv,
#             ),
#         ]
#         self.head = nn.Sequential(*m_head)
#         # self.body = nn.Sequential(*m_body)
#         self.tail = nn.Sequential(*m_tail)

#     def forward(self, x):
#         # alpha = 1  # learnable or fixed
#         x = self.head(x)
#         # x = self.gen_eqv(x, self.head)
#         # res = self.body(x)
#         # x= res + alpha * x
#         # x = self.gen_eqv(x, self.tail)
#         x = self.tail(x)
#         return x

#     def load_state_dict(self, state_dict, strict=True):
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name in own_state:
#                 if isinstance(param, nn.Parameter):
#                     param = param.data
#                 try:
#                     own_state[name].copy_(param)
#                 except Exception:
#                     if name.find("tail") == -1:
#                         raise RuntimeError(
#                             "While copying the parameter named {}, "
#                             "whose dimensions in the model are {} and "
#                             "whose dimensions in the checkpoint are {}.".format(
#                                 name, own_state[name].size(), param.size()
#                             )
#                         )
#             elif strict:
#                 if name.find("tail") == -1:
#                     raise KeyError('unexpected key "{}" in state_dict'.format(name))


# if __name__ == "__main__":

#     class Custom_Args:
#         def __init__(self):
#             self.n_resblocks = 0
#             self.n_feats = 8
#             self.scale = 4
#             self.n_channels = 4
#             self.sym_np_path = "model/reynolds_utils/fcc_symmetry_group.npy"
#             self.sym_inv_np_path = "model/reynolds_utils/fcc_symmetry_group_inv.npy"

#     args = Custom_Args()
#     model = Reynolds_QSR(args)
#     summary(model, input_size=(7, 4, 64, 64))
#     data = torch.rand((1, 4, 63, 65))
#     model(data)

#     def test_model_equivariance(model, x, atol=1e-6, rtol=1e-5):
#         """
#         Tests equivariance: f(g·x) ≈ g·f(x)
#         for model with group_tensor (G,Cg,Cg).
#         Input shape: (B,C,*spatial) with C % Cg == 0.
#         """
#         model.eval()
#         with torch.no_grad():
#             B, C, *spatial = x.shape
#             G, Cg, _ = model.group_tensor.shape
#             assert C == Cg, f"Channels {C} must be same as group element {Cg}"

#             # f(x)
#             fx = model(x)  # (B,Cout,*spatial_out)
#             _, Cout, *spatial_out = fx.shape

#             errors = []

#             for g in model.group_tensor:  # (Cg,Cg)
#                 # g·x
#                 gx = torch.einsum("ci,bi...->bc...", g, x)  # (B,C,*spatial)
#                 f_gx = model(gx)  # f(g·x)

#                 # g·f(x)
#                 g_fx = torch.einsum("ci,bi...->bc...", g, fx)  # (B,Cout,*spatial_out)

#                 # max error for this g
#                 diff = (f_gx - g_fx).abs().max().item()
#                 errors.append(diff)

#             max_err = max(errors)
#             passed = max_err < atol + rtol * fx.abs().max().item()
#             return passed, max_err, errors

#     passed, max_err, errs = test_model_equivariance(model, data)
#     print("Equivariant:", passed)
#     print("Max error:", max_err)
#     print("Per-group errors:", errs)

#     # a = EquivariantReynoldsWrap(
#     #     QuaternionConv(
#     #         in_channels=4,
#     #         out_channels=8,
#     #         kernel_size=3,
#     #         stride=1,
#     #         padding=3 // 2,
#     #     ),
#     #     group_tensor=self.group_tensor,
#     #     group_tensor_inv=self.group_tensor_inv,
#     # )
#     # m_tail = [
#     #     EquivariantReynoldsWrap(
#     #         QuaternionConv(
#     #             in_channels=n_feats,
#     #             out_channels=scale * scale * n_feats,
#     #             kernel_size=kernel_size,
#     #             stride=1,
#     #             padding=kernel_size // 2,
#     #         ),
#     #         group_tensor=self.group_tensor,
#     #         group_tensor_inv=self.group_tensor_inv,
#     #     ),
#     #     EquivariantReynoldsWrap(
#     #         QuaternionTransposeConv(
#     #             in_channels=scale * scale * n_feats,
#     #             out_channels=n_feats,
#     #             kernel_size=scale,
#     #             stride=scale,
#     #             padding=kernel_size // 2,
#     #             # output_padding=(2, 2),  # Adjust output padding
#     #         ),
#     #         group_tensor=self.group_tensor,
#     #         group_tensor_inv=self.group_tensor_inv,
#     #     ),
#     #     EquivariantReynoldsWrap(
#     #         QuaternionConv(
#     #             in_channels=n_feats,
#     #             out_channels=n_feats,
#     #             kernel_size=kernel_size,
#     #             stride=1,
#     #             padding=kernel_size // 2,
#     #         ),
#     #         group_tensor=self.group_tensor,
#     #         group_tensor_inv=self.group_tensor_inv,
#     #     ),
#     #     EquivariantReynoldsWrap(
#     #         QuaternionConv(
#     #             in_channels=n_feats,
#     #             out_channels=n_channels,
#     #             kernel_size=kernel_size,
#     #             stride=1,
#     #             padding=kernel_size // 2,
#     #         ),
#     #         group_tensor=self.group_tensor,
#     #         group_tensor_inv=self.group_tensor_inv,
#     #     ),
#     # ]

#     # data_out = model(data)
#     # data_out2 = model(data2)[:1, ...]
#     # torch.allclose(data_out, data_out2, rtol=1e-5, atol=1e-9)
#     # # Step 1: Apply group action: gamma_x = g ⋅ x
#     # gamma_x = torch.einsum("gci,bihw->bgchw", model.group_tensor, data)  # (B,G,C,H,W)

#     # c = QuaternionConv(
#     #     in_channels=4,
#     #     out_channels=4,
#     #     kernel_size=5,
#     #     stride=1,
#     #     padding=3 // 2,
#     #     operation="conv3d",
#     # )
#     # r_weight = c.r_weight
#     # i_weight = c.i_weight
#     # j_weight = c.j_weight
#     # k_weight = c.k_weight
#     # cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
#     # cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=1)
#     # cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=1)
#     # cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=1)
#     # cat_kernels_4_quaternion = torch.cat(
#     #     [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0
#     # )

#     # cat_kernels_4_quaternion.dim()

#     # if input.dim() == 3:
#     #     convfunc = F.conv1d
#     # elif input.dim() == 4:
#     #     convfunc = F.conv2d
#     # elif input.dim() == 5:
#     #     convfunc = F.conv3d
#     # else:
#     #     raise Exception(
#     #         "The convolutional input is either 3, 4 or 5 dimensions."
#     #         " input.dim = " + str(input.dim())
#     #     )

#     # return convfunc(
#     #     input, cat_kernels_4_quaternion, bias, stride, padding, dilation, groups
#     # )
