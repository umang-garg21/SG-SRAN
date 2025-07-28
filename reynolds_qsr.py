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
            print("Input:", x.shape)
            # x = x.permute(0, 1, 3, 2)
            # print("Permuted Input:", x.shape)
            x = self.conv_layer(x)
            print("After conv:", x.shape)
            x = self.transposed_conv(x)
            print("After transpose:", x.shape)
            x = self.post_conv_layer(x)
            print("After post_conv:", x.shape)
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

        m_head = [
            QuaternionConv(
                in_channels=n_channels,
                out_channels=n_feats,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        ]

        # m_body = [Residual_SA(n_feats, n_feats) for _ in range(n_resblocks)]
        # m_body.append(
        #     QuaternionConv(
        #         n_feats,
        #         n_feats,
        #         kernel_size=kernel_size,
        #         stride=1,
        #         padding=kernel_size // 2,
        #     )
        # )

        m_tail = [
            Upsampler2DQuaternionTransposeConv(
                kernel_size=kernel_size,
                scale=scale,
                n_feats=n_feats,
            ),
            QuaternionConv(
                n_feats,
                n_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
        ]

        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def gen_eqv(self, gamma_x, fn):
        # x shape (B,C,H,W)

        # Multiply by group tensor transpose:
        # einsum over group dims: (G, N, N) x (B, G, W, N) -> (B, G, W, N)
        # We'll do einsum with broadcasting:
        # fn(gamma_x) # (B, G, W*N)
        gamma_T_f_gamma_x = torch.einsum(
            "gij,bgwj->bgwi",
            self.group_tensor_inv,
            fn(gamma_x).view(-1, self.G, self.W, self.N),
        )  # (B, G, W, N)
        return gamma_T_f_gamma_x.mean(dim=1)  # (B, W, N)

    def forward(self, x):
        # gamma_x = torch.einsum("gij,bj->bgi", self.group_tensor, x)  # (B, G, N) WZ

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
            self.n_feats = 8
            self.scale = 4
            self.n_channels = 4
            self.sym_np_path = "model/reynolds_utils/fcc_symmetry_group.npy"
            self.sym_inv_np_path = "model/reynolds_utils/fcc_symmetry_group_inv.npy"

    args = Custom_Args()
    model = Reynolds_QSR(args)

    torch.tensor(np.load(args.sym_np_path), dtype=torch.float64)
    data = torch.rand((1, 4, 63, 65))
    data2 = torch.cat((data, data), dim=0)
    # x[:,:,0:h,0:w]
    A = model(data)
    # A.shape
    # x shape (B,C,H,W)
    summary(model, input_size=(1, 4, 63, 65))

    gamma_x = torch.einsum("gci,bihw->bgchw", model.group_tensor, data)  # (bgchw) WZ

    (gamma_x[:, 0, :, :, :] == data).all()

    data_out = model(data)

    data_out2 = model(data2)
    gamma_x.shape

    gamma_x.view(-1, 4, 63, 65).shape

    out = model(gamma_x.view(-1, 4, 63, 65))
    out_reshape = out.view(-1, 24, 4, 252, 260)
    torch.allclose(out_reshape[:, 0], data_out, rtol=1e-5, atol=1e-6)

    c = QuaternionConv(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=3 // 2,
    )
    r_weight = c.r_weight
    i_weight = c.i_weight
    j_weight = c.j_weight
    k_weight = c.k_weight
    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
    cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=1)
    cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=1)
    cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=1)
    cat_kernels_4_quaternion = torch.cat(
        [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0
    )

    cat_kernels_4_quaternion = torch.cat(
        [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0
    )

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception(
            "The convolutional input is either 3, 4 or 5 dimensions."
            " input.dim = " + str(input.dim())
        )

    return convfunc(
        input, cat_kernels_4_quaternion, bias, stride, padding, dilation, groups
    )

# uplayer = Upsampler2DQuaternionTransposeConv(
#     kernel_size=3,
#     scale=args.scale,
#     n_feats=args.n_feats,
# )
