import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from model.quat_utils.Qops_with_QSN import conv2d, Residual_SA
from model.quat_utils.quaternion_layers import QuaternionConv

# from einops import rearrange
# ─── requirements ───────────────────────────────────────────────────────────────
# pip install torch e3nn==0.7.4              # e3nn just for rotation utilities
# ────────────────────────────────────────────────────────────────────────────────
import torch, torch.nn as nn
from torchinfo import summary


def make_model(args):
    return Reynolds_QSR(args)


class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        # import pdb; pdb.set_trace()
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        long_height = x.shape[2]
        short_width = x.shape[3]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view(
            [
                batch_size,
                self.upscale_factor,
                long_channel_len,
                long_height,
                short_width,
            ]
        )
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_height, long_width)

        return x


class PixelUnshuffle1D(torch.nn.Module):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output
    """

    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view(
            [batch_size, long_channel_len, short_width, self.downscale_factor]
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x


class Upsampler1D_pixel_shuffle(nn.Module):
    def __init__(self, kernel_size, scale, n_feat, bn=False, act=False, bias=True):
        super(Upsampler1D_pixel_shuffle, self).__init__()

        self.conv_layer = conv2d(
            n_feat,
            2 * n_feat,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.pixel_shuffle = PixelShuffle1D(2)
        # self.up_sample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.scale = scale
        self.n_feat = n_feat

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        if (self.scale & (self.scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(self.scale, 2))):
                # import pdb; pdb.set_trace()
                bsize, ch, h, w = x.shape
                x = self.conv_layer(x)
                # x = rearrange(x, 'd0 d1 d2 d3 -> d0 d1 (d2 d3)')
                x = self.pixel_shuffle(x)
                # x = rearrange(x, 'd0 d1 (d2 d3) -> d0 d1 d2 d3', d0=bsize, d1=ch, d2=h, d3=2*w)

            x = x.permute(0, 1, 3, 2)
            return x

        else:
            raise NotImplementedError


### TRANSPOSE CONV BASED UPSAMPLER ####
class Upsampler1D_transpose_conv(nn.Module):
    def __init__(
        self,
        kernel_size,
        scale,
        n_feat,
        bn=False,
        act=False,
        bias=True,
        dropout_prob=0.2,
    ):
        super(Upsampler1D_transpose_conv, self).__init__()

        self.conv_layer1 = conv2d(
            n_feat,
            2 * n_feat,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.conv_layer2 = conv2d(
            n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        # Adding dropout layer after the convolution layer
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout with specified probability
        self.pixel_shuffle = PixelShuffle1D(2)
        # self.transposed_conv = TransposedConvUpsampler1D(2*n_feat, n_feat, kernel_size=(3,3), stride=(1,2), padding=(1,1), output_padding=(0,1))
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels=2 * n_feat,  # e.g., 1024
            out_channels=n_feat,  # e.g., 256
            kernel_size=(3, 4),  # upsampling only in width
            stride=(1, 2),
            padding=(1, 1),  # pad width by 1, height unchanged
            output_padding=(0, 0),
            bias=True,
        )
        # self.up_sample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.scale = scale
        self.n_feat = n_feat

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        if (self.scale & (self.scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(self.scale, 2))):
                # import pdb; pdb.set_trace()
                bsize, ch, h, w = x.shape
                # print(x.shape)
                x = self.conv_layer1(x)
                # x = self.dropout(x)
                # x = rearrange(x, 'd0 d1 d2 d3 -> d0 d1 (d2 d3)')
                # x = self.conv_layer2(x)
                # x = self.dropout(x)

                # import pdb; pdb.set_trace()
                # print(x.shape)
                # x = self.pixel_shuffle(x)
                x = self.transposed_conv(x)
                # x= self.conv_layer2(x)
                # print(x.shape)

                # x = self.dropout(x)
                # x = rearrange(x, 'd0 d1 (d2 d3) -> d0 d1 d2 d3', d0=bsize, d1=ch, d2=h, d3=2*w)

            x = x.permute(0, 1, 3, 2)
            # import pdb; pdb.set_trace()
            # print('Upsampler1D: x.shape:', x.shape)
            return x

        else:
            raise NotImplementedError


### TRANSPOSE CONV BASED UPSAMPLER ####
class Upsampler1D_transpose_conv_1pass(nn.Module):
    def __init__(
        self,
        kernel_size,
        scale,
        n_feat,
        bn=False,
        act=False,
        bias=True,
        dropout_prob=0.2,
    ):
        super(Upsampler1D_transpose_conv_1pass, self).__init__()

        self.conv_layer = conv2d(
            n_feat,
            scale * n_feat,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        # Adding dropout layer after the convolution layer
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout with specified probability

        # self.transposed_conv = TransposedConvUpsampler1D(scale*n_feat, n_feat, kernel_size=(1,kernel_size), stride=(1,scale), padding=(0, 2), output_padding=(0,1))
        # self.transposed_conv = TransposedConvUpsampler1D(scale*n_feat, n_feat, kernel_size=(1,kernel_size), stride=(1,scale), padding=(0, scale//2), output_padding=(0, scale%2))

        self.transposed_conv = nn.ConvTranspose2d(
            in_channels=scale * n_feat,  # e.g., 1024
            out_channels=n_feat,  # e.g., 256
            kernel_size=(1, scale),  # upsampling only in width
            stride=(1, scale),
            padding=(0, 1),  # pad width by 1, height unchanged
            output_padding=(0, 2),  # add 2 more pixels in width
            bias=True,
        )

        self.post_conv_layer = conv2d(
            n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        # self.up_sample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.scale = scale
        self.n_feat = n_feat

    def forward(self, x):
        try:
            x = x.permute(0, 1, 3, 2)
            x = self.conv_layer(x)
            x = self.transposed_conv(x)
            x = self.post_conv_layer(x)
            # x= self.post_conv_layer(x)
            # print(x.shape)
            # x = self.dropout(x)
            # x = rearrange(x, 'd0 d1 (d2 d3) -> d0 d1 d2 d3', d0=bsize, d1=ch, d2=h, d3=2*w)

            x = x.permute(0, 1, 3, 2)
            # import pdb; pdb.set_trace()
            # print('Upsampler1D: x.shape:', x.shape)
            return x

        except Exception as e:
            print("Error in Upsampler1D_transpose_conv_1pass:", e)
            import pdb

            pdb.set_trace()
            raise NotImplementedError


### Chatgpt based interpolation
class Upsampler1D_quaternion_interp(nn.Module):
    def __init__(
        self,
        kernel_size,
        scale,
        n_feat,
        bn=False,
        act=False,
        bias=True,
        dropout_prob=0.2,
    ):
        super(Upsampler1D_quaternion_interp, self).__init__()

        self.scale = scale
        self.n_feat = n_feat
        self.kernel_size = kernel_size

        self.smoothing_conv = conv2d(n_feat, n_feat, kernel_size=5, stride=1, padding=2)
        self.refine_conv = conv2d(
            n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        x: (B, 4C, H, W) where 4C = number of quaternion channels
        """
        # import pdb; pdb.set_trace()
        x = x.permute(0, 1, 3, 2)  # (B, 4C, W, H) to match interpolation convention

        for _ in range(int(math.log(self.scale, 2))):
            # Smooth before interpolation
            x = self.smoothing_conv(x)

            # Apply bilinear interpolation per quaternion channel
            x = F.interpolate(
                x, scale_factor=(1, 2), mode="bilinear", align_corners=True
            )

            # Optional: Dropout
            # x = self.dropout(x)

            # Refine interpolated output with quaternion-aware conv
            x = self.refine_conv(x)

        x = x.permute(0, 1, 3, 2)  # Back to (B, 4C, H, W)
        return x


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
        # import pdb; pdb.set_trace()
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

dtype = torch.float64

model.group_tensor

# uplayer = Upsampler2DQuaternionTransposeConv(
#     kernel_size=3,
#     scale=args.scale,
#     n_feats=args.n_feats,
# )
