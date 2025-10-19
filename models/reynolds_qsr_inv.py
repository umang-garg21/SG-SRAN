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

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# device = torch.device("cpu")
# torch.set_default_device(device)


# def make_model(args):
#     return Reynolds_QSR(args)


# class InvariantReynoldsWrap(nn.Module):
#     """
#     Reynolds operator wrapper: enforces invariance for any module fn
#     under a group action represented by group_tensor (G, Cg, Cg).
#     Input/output channel dims must be multiples of Cg.
#     Works with inputs (B, C, *spatial) for 1D/2D/3D ops.
#     """

#     def __init__(self, fn: nn.Module, group_tensor: torch.Tensor):
#         super().__init__()
#         self.fn = fn
#         self.register_buffer("group_tensor", group_tensor)  # (G, Cg, Cg)

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
#         BG_out, C_out, *spatial_out = fx.shape
#         assert BG_out == B * G
#         assert C_out % Cg == 0, f"fn must output multiple of {Cg}, got {C_out}"
#         n_feats_out = C_out // Cg

#         # --- Project back with g⁻¹ ---
#         fx = fx.view(
#             B, G, n_feats_out, Cg, *spatial_out
#         )  # (B,G,n_feats_out,Cg,*spatial)
#         # --- Average over group and return ---
#         return fx.mean(dim=1).reshape(B, C_out, *spatial_out)


# class UpsamplerQuaternionTransposeConv(nn.Module):
#     def __init__(
#         self,
#         kernel_size,
#         scale,
#         n_feats,
#         group_tensor,
#         bn=False,
#         act=False,
#         bias=True,
#         dropout_prob=0.2,
#     ):
#         super(UpsamplerQuaternionTransposeConv, self).__init__()

#         self.scale = scale
#         self.n_feat = n_feats

#         self.conv_layer = InvariantReynoldsWrap(
#             QuaternionConv(
#                 in_channels=n_feats,
#                 out_channels=scale * scale * n_feats,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 padding=kernel_size // 2,
#             ),
#             group_tensor=group_tensor,
#         )

#         # Adding dropout layer after the convolution layer
#         self.dropout = nn.Dropout(p=dropout_prob)  # Dropout with specified probability

#         self.transposed_conv = InvariantReynoldsWrap(
#             QuaternionTransposeConv(
#                 in_channels=scale * scale * n_feats,
#                 out_channels=n_feats,
#                 kernel_size=scale,
#                 stride=scale,
#                 padding=kernel_size // 2,
#                 output_padding=2,  # Adjust output padding
#             ),
#             group_tensor=group_tensor,
#         )
#         self.post_conv_layer = InvariantReynoldsWrap(
#             QuaternionConv(
#                 in_channels=n_feats,
#                 out_channels=n_feats,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 padding=kernel_size // 2,
#             ),
#             group_tensor=group_tensor,
#         )

#     def forward(self, x):
#         try:

#             x = self.conv_layer(x)
#             x = self.transposed_conv(x)
#             x = self.post_conv_layer(x)
#             return x
#         except Exception as e:
#             print("Error in Upsampler2DQuaternionTransposeConv:", e)


# class Reynolds_QSR(nn.Module):
#     def __init__(self, args):
#         super(Reynolds_QSR, self).__init__()
#         n_resblocks = args.n_resblocks
#         n_channels = 4
#         n_feats = args.n_feats
#         scale = args.scale
#         kernel_size = 3
#         # act = nn.ReLU(True)

#         self.register_buffer(
#             "group_tensor", torch.tensor(np.load(args.sym_np_path), dtype=torch.float32)
#         )  # (G, C, C) where C=4

#         m_head = [
#             # InvariantReynoldsWrap(
#             QuaternionConv(
#                 in_channels=n_channels,
#                 out_channels=n_feats,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 padding=kernel_size // 2,
#             ),
#             #     group_tensor=self.group_tensor,
#             # ),
#         ]

#         m_tail = [
#             UpsamplerQuaternionTransposeConv(
#                 kernel_size=kernel_size,
#                 scale=scale,
#                 n_feats=n_feats,
#                 group_tensor=self.group_tensor,
#             ),
#             # InvariantReynoldsWrap(
#             QuaternionConv(
#                 in_channels=n_feats,
#                 out_channels=n_channels,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 padding=kernel_size // 2,
#             ),
#             # group_tensor=self.group_tensor,
#             # ),
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
#     self = model

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
#     summary(model, input_size=(7, 4, 64, 64))

#     data = torch.rand((1, 4, 63, 65))
#     x = data
#     a = InvariantReynoldsWrap(
#         QuaternionConv(
#             in_channels=4,
#             out_channels=4,
#             kernel_size=3,
#             stride=1,
#             padding=3 // 2,
#         ),
#         group_tensor=self.group_tensor,
#     )
#     model(data)

#     def test_model_invariance(model, x, atol=1e-6, rtol=1e-5):
#         """
#         Tests invariance: f(g·x) ≈ f(x)
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

#             errors = []

#             for g in model.group_tensor:  # (Cg,Cg)
#                 # g·x
#                 gx = torch.einsum("ci,bi...->bc...", g, x)  # (B,C,*spatial)
#                 f_gx = model(gx)  # f(g·x)
#                 # max error for this g
#                 diff = (f_gx - fx).abs().max().item()
#                 errors.append(diff)

#             max_err = max(errors)
#             passed = max_err < atol + rtol * fx.abs().max().item()
#             return passed, max_err, errors

#     passed, max_err, errs = test_model_invariance(model, data)
#     print("Invariance:", passed)
#     print("Max error:", max_err)
#     print("Per-group errors:", errs)

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
