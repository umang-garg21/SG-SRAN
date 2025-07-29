import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

# from model.quat_utils.Qops_with_QSN import conv2d, Residual_SA
from model.quat_utils.quaternion_layers import QuaternionConv

# from einops import rearrange
# ─── requirements ───────────────────────────────────────────────────────────────
# pip install torch e3nn==0.7.4              # e3nn just for rotation utilities
# ────────────────────────────────────────────────────────────────────────────────
import torch, torch.nn as nn
from torchinfo import summary


def make_model(args):
    return Reynolds_QSR(args)


### TRANSPOSE CONV BASED UPSAMPLER ####
class Upsampler2DQuaternionTransposeConv(nn.Module):
    def __init__(
        self,
        kernel_size,
        scale,
        n_feats,
        bn=False,
        act=False,
        bias=True,
        dropout_prob=0.2,
    ):
        super(Upsampler2DQuaternionTransposeConv, self).__init__()

        self.conv_layer = QuaternionConv(
            n_feats,
            scale * scale * n_feats,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        # Adding dropout layer after the convolution layer
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout with specified probability

        self.transposed_conv = nn.ConvTranspose2d(
            in_channels=scale * scale * n_feats,  # e.g., 1024
            out_channels=n_feats,  # e.g., 256
            kernel_size=(scale, scale),  # upsampling only in width
            stride=(scale, scale),
            padding=(1, 1),  # pad width by 1, height unchanged
            output_padding=(2, 2),  # add 2 more pixels in width
            bias=True,
        )

        self.post_conv_layer = QuaternionConv(
            n_feats,
            n_feats,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        # self.up_sample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.scale = scale
        self.n_feat = n_feats

    def forward(self, x):
        try:
            # print("Input:", x.shape)
            # x = x.permute(0, 1, 3, 2)
            # print("Permuted Input:", x.shape)
            x = self.conv_layer(x)
            # print("After conv:", x.shape)
            x = self.transposed_conv(x)
            # print("After transpose:", x.shape)
            x = self.post_conv_layer(x)
            # print("After post_conv:", x.shape)
            # x = x.permute(0, 1, 3, 2)
            # print("Permuted Output:", x.shape)
            return x
        except Exception as e:
            print("Error in Upsampler2DQuaternionTransposeConv:", e)
            # import pdb

            # pdb.set_trace()
            # raise NotImplementedError
        # try:
        #     x = x.permute(0, 1, 3, 2)
        #     x = self.conv_layer(x)
        #     x = self.transposed_conv(x)
        #     x = self.post_conv_layer(x)
        #     # x= self.post_conv_layer(x)
        #     # print(x.shape)
        #     # x = self.dropout(x)
        #     # x = rearrange(x, 'd0 d1 (d2 d3) -> d0 d1 d2 d3', d0=bsize, d1=ch, d2=h, d3=2*w)

        #     x = x.permute(0, 1, 3, 2)
        #     # import pdb; pdb.set_trace()
        #     # print('Upsampler1D: x.shape:', x.shape)
        #     return x

        # except Exception as e:
        #     print("Error in Upsampler1D_transpose_conv_1pass:", e)
        #     import pdb

        #     pdb.set_trace()
        #     raise NotImplementedError


class EquivariantReynoldsWrap(nn.Module):
    """
    Reynolds operator wrapper: enforces equivariance for any module fn
    under a group action represented by group_tensor (G, Cg, Cg).
    Input/output channel dims must be multiples of Cg.
    Works with inputs (B, C, *spatial) for 1D/2D/3D ops.
    """

    def __init__(
        self, fn: nn.Module, group_tensor: torch.Tensor, group_tensor_inv: torch.Tensor
    ):
        super().__init__()
        self.fn = fn
        self.register_buffer("group_tensor", group_tensor)  # (G, Cg, Cg)
        self.register_buffer("group_tensor_inv", group_tensor_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *spatial_in = x.shape
        G, Cg, _ = self.group_tensor.shape
        assert C % Cg == 0, f"Channels {C} must be multiple of {Cg}"
        n_feats = C // Cg

        # --- Lift: apply group action g·x ---
        x = x.view(B, n_feats, Cg, *spatial_in)  # (B,n_feats,Cg,*spatial)
        gamma_x = torch.einsum("gci,bnc...->bgni...", self.group_tensor, x)
        gamma_x = gamma_x.reshape(B * G, n_feats * Cg, *spatial_in)  # (B*G,C,*spatial)

        # --- Apply wrapped fn ---
        fx = self.fn(gamma_x)  # (B*G,Cout,*spatial_out)
        BGO, Cout, *spatial_out = fx.shape
        assert BGO == B * G
        assert Cout % Cg == 0, f"fn must output multiple of {Cg}, got {Cout}"
        n_feats_out = Cout // Cg

        # --- Project back with g⁻¹ ---
        fx = fx.view(
            B, G, n_feats_out, Cg, *spatial_out
        )  # (B,G,n_feats_out,Cg,*spatial)
        fx = torch.einsum("gci,bgni...->bgnc...", self.group_tensor_inv, fx)

        # --- Average over group and return ---
        return fx.mean(dim=1).reshape(B, Cout, *spatial_out)


class Reynolds_QSR(nn.Module):
    def __init__(self, args):
        super(Reynolds_QSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_channels = 4
        n_feats = args.n_feats
        scale = args.scale
        kernel_size = 3
        # act = nn.ReLU(True)

        self.register_buffer(
            "group_tensor", torch.tensor(np.load(args.sym_np_path), dtype=torch.float32)
        )  # (G, C, C) where C=4

        self.register_buffer(
            "group_tensor_inv",
            torch.tensor(np.load(args.sym_inv_np_path), dtype=torch.float32),
        )  # (G, C, C) where C=4

        # m_head = [
        #     # EquivariantReynoldsWrap(
        #     QuaternionConv(
        #         in_channels=n_channels,
        #         out_channels=n_feats,
        #         kernel_size=kernel_size,
        #         stride=1,
        #         padding=kernel_size // 2,
        #     ),
        #     # group_tensor=self.group_tensor,
        #     # group_tensor_inv=self.group_tensor_inv,
        #     # )
        # ]
        # m_tail = [
        #     Upsampler2DQuaternionTransposeConv(
        #         kernel_size=kernel_size,
        #         scale=scale,
        #         n_feats=n_feats,
        #     ),
        #     QuaternionConv(
        #         in_channels=n_feats,
        #         out_channels=n_channels,
        #         kernel_size=kernel_size,
        #         stride=1,
        #         padding=kernel_size // 2,
        #     ),
        # ]
        m_head = [
            EquivariantReynoldsWrap(
                QuaternionConv(
                    in_channels=n_channels,
                    out_channels=n_feats,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                group_tensor=self.group_tensor,
                group_tensor_inv=self.group_tensor_inv,
            )
        ]

        m_tail = [
            EquivariantReynoldsWrap(
                Upsampler2DQuaternionTransposeConv(
                    kernel_size=kernel_size,
                    scale=scale,
                    n_feats=n_feats,
                ),
                group_tensor=self.group_tensor,
                group_tensor_inv=self.group_tensor_inv,
            ),
            EquivariantReynoldsWrap(
                QuaternionConv(
                    in_channels=n_feats,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                group_tensor=self.group_tensor,
                group_tensor_inv=self.group_tensor_inv,
            ),
        ]
        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # alpha = 1  # learnable or fixed
        x = self.head(x)
        # x = self.gen_eqv(x, self.head)
        # res = self.body(x)
        # x= res + alpha * x
        # x = self.gen_eqv(x, self.tail)
        x = self.tail(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") == -1:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}.".format(
                                name, own_state[name].size(), param.size()
                            )
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))


if __name__ == "__main__":

    class Custom_Args:
        def __init__(self):
            self.n_resblocks = 0
            self.n_feats = 4
            self.scale = 4
            self.n_channels = 4
            self.sym_np_path = "model/reynolds_utils/fcc_symmetry_group.npy"
            self.sym_inv_np_path = "model/reynolds_utils/fcc_symmetry_group_inv.npy"

    args = Custom_Args()
    model = Reynolds_QSR(args)

    summary(model, input_size=(7, 4, 64, 64))

    data = torch.rand((1, 4, 63, 65))
    # model(data)


def test_model_equivariance(model, x, atol=1e-6, rtol=1e-5):
    """
    Tests equivariance: f(g·x) ≈ g·f(x)
    for model with group_tensor (G,Cg,Cg).
    Input shape: (B,C,*spatial) with C % Cg == 0.
    """
    model.eval()
    with torch.no_grad():
        B, C, *spatial = x.shape
        G, Cg, _ = model.group_tensor.shape
        assert C == Cg, f"Channels {C} must be same as group element {Cg}"

        # f(x)
        fx = model(x)  # (B,Cout,*spatial_out)
        _, Cout, *spatial_out = fx.shape

        errors = []

        for g in model.group_tensor:  # (Cg,Cg)
            # g·x
            gx = torch.einsum("ci,bi...->bc...", g, x)  # (B,C,*spatial)
            f_gx = model(gx)  # f(g·x)

            # g·f(x)
            g_fx = torch.einsum("ci,bi...->bc...", g, fx)  # (B,Cout,*spatial_out)

            # max error for this g
            diff = (f_gx - g_fx).abs().max().item()
            errors.append(diff)

        max_err = max(errors)
        passed = max_err < atol + rtol * fx.abs().max().item()
        return passed, max_err, errors

    passed, max_err, errs = test_model_equivariance(model, data)
    print("Equivariant:", passed)
    print("Max error:", max_err)
    print("Per-group errors:", errs)


# data = torch.rand((1, 4, 63, 65))
# data2 = torch.cat((data, data), dim=0)
# self = model

# summary(model, input_size=(1, 4, 63, 65))

# self(data)

# data_out = model(data)
# data_out2 = model(data2)[:1, ...]
# torch.allclose(data_out, data_out2, rtol=1e-5, atol=1e-9)
# # Step 1: Apply group action: gamma_x = g ⋅ x
# gamma_x = torch.einsum("gci,bihw->bgchw", model.group_tensor, data)  # (B,G,C,H,W)

# (gamma_x[:, 0, :, :, :] == data).all()

# gamma_x.shape

# gamma_x.view(-1, 4, 63, 65).shape

# out = model(gamma_x.view(-1, 4, 63, 65))

# model.group_tensor_inv

# gamma_T_f_gamma_x = torch.einsum(
#     "gic,bgchw->bgihw",
#     self.group_tensor_inv,
#     fn(gamma_x).view(-1, self.G, self.W, self.N),
# )  # (B, G, W, N)

# gamma_x.view(B, G, C_out, H, W)

# gamma_T_f_gamma_x = torch.einsum(
#     "gij,bgwj->bgwi",
#     self.group_tensor_inv,
#     fn(gamma_x).view(-1, self.G, self.W, self.N),
# )  # (B, G, W, N)
# # return gamma_T_f_gamma_x.mean(dim=1)  # (B, W, N)

# out_reshape = out.view(-1, 24, 4, 252, 260)
# torch.allclose(out_reshape[:, 0], data_out, rtol=1e-5, atol=1e-6)

# c = QuaternionConv(
#     in_channels=4,
#     out_channels=4,
#     kernel_size=5,
#     stride=1,
#     padding=3 // 2,
#     operation="conv3d",
# )
# r_weight = c.r_weight
# i_weight = c.i_weight
# j_weight = c.j_weight
# k_weight = c.k_weight
# cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
# cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=1)
# cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=1)
# cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=1)
# cat_kernels_4_quaternion = torch.cat(
#     [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0
# )

# cat_kernels_4_quaternion.dim()

# if input.dim() == 3:
#     convfunc = F.conv1d
# elif input.dim() == 4:
#     convfunc = F.conv2d
# elif input.dim() == 5:
#     convfunc = F.conv3d
# else:
#     raise Exception(
#         "The convolutional input is either 3, 4 or 5 dimensions."
#         " input.dim = " + str(input.dim())
#     )

# return convfunc(
#     input, cat_kernels_4_quaternion, bias, stride, padding, dilation, groups
# )


# uplayer = Upsampler2DQuaternionTransposeConv(
#     kernel_size=3,
#     scale=args.scale,
#     n_feats=args.n_feats,
# )
