from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from so3_o_cubic_sr_4e6e_e3nn_consistent import (
    CubicQuotient4eCodec,
    DecodeResult,
    SharedOverMConv2d,
    factorize_scale,
    rand_quaternion_grid,
)


class SharedOverMConvTranspose2d(nn.Module):
    """
    Apply one ConvTranspose2d shared across all m-components of a fixed irrep.
    This preserves the 4e basis structure while mixing multiplicities only.
    """

    def __init__(
        self,
        mul_in: int,
        mul_out: int,
        l: int,
        stride: int = 2,
        kernel_size: Optional[int] = None,
        padding: Optional[int] = None,
        output_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        if kernel_size is None:
            if stride == 2:
                kernel_size = 4
            elif stride == 3:
                kernel_size = 3
            else:
                kernel_size = stride

        if padding is None:
            if stride == 2 and kernel_size == 4:
                padding = 1
            else:
                padding = 0

        self.mul_in = mul_in
        self.mul_out = mul_out
        self.d = 2 * l + 1
        self.deconv = nn.ConvTranspose2d(
            mul_in,
            mul_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        expected = self.mul_in * self.d
        assert c == expected, f"Expected {expected} channels, got {c}"

        x = x.view(b, self.mul_in, self.d, h, w).permute(0, 2, 1, 3, 4).reshape(b * self.d, self.mul_in, h, w)
        y = self.deconv(x)
        h2, w2 = y.shape[-2:]
        y = y.view(b, self.d, self.mul_out, h2, w2).permute(0, 2, 1, 3, 4).reshape(b, self.mul_out * self.d, h2, w2)
        return y


class FiberResidual4eBlock(nn.Module):
    def __init__(self, mul4: int):
        super().__init__()
        self.conv1 = SharedOverMConv2d(mul4, mul4, l=4, kernel_size=3, padding=1)
        self.conv2 = SharedOverMConv2d(mul4, mul4, l=4, kernel_size=3, padding=1)

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x4), inplace=False)
        y = self.conv2(y)
        return x4 + y


class TransposedStage4e(nn.Module):
    def __init__(self, mul4: int, scale_factor: int, num_blocks: int = 1):
        super().__init__()
        self.up_feat = SharedOverMConvTranspose2d(mul4, mul4, l=4, stride=scale_factor)
        self.up_skip = SharedOverMConvTranspose2d(1, 1, l=4, stride=scale_factor)
        self.blocks = nn.ModuleList([FiberResidual4eBlock(mul4) for _ in range(num_blocks)])

    def forward(self, x4: torch.Tensor, x_desc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x4 = self.up_feat(x4)
        x_desc = self.up_skip(x_desc)
        for block in self.blocks:
            x4 = block(x4)
        return x4, x_desc


class SO3OQuotientSRSimpleE3NN(nn.Module):
    """
    Simple e3nn-consistent 4e SR model:
      q_lr -> 4e descriptor -> shared-over-m transpose-conv stages ->
      shared-over-m residual refinement -> descriptor head + skip -> shell normalization.
    """

    def __init__(
        self,
        hidden_mul4: int = 8,
        num_blocks_per_stage: int = 1,
        sr_scale: int = 4,
        passive_input: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if hidden_mul4 < 1:
            raise ValueError(f"hidden_mul4 must be >= 1, got {hidden_mul4}")
        if num_blocks_per_stage < 0:
            raise ValueError(f"num_blocks_per_stage must be >= 0, got {num_blocks_per_stage}")
        if not isinstance(sr_scale, int) or sr_scale < 2:
            raise ValueError(f"sr_scale must be integer >= 2, got {sr_scale}")

        self.sr_scale = sr_scale
        self.codec = CubicQuotient4eCodec(passive_input=passive_input, dtype=dtype)

        self.in4 = SharedOverMConv2d(1, hidden_mul4, l=4, kernel_size=3, padding=1)
        self.stages = nn.ModuleList(
            [
                TransposedStage4e(hidden_mul4, scale_factor=sf, num_blocks=num_blocks_per_stage)
                for sf in factorize_scale(sr_scale)
            ]
        )
        self.out4 = SharedOverMConv2d(hidden_mul4, 1, l=4, kernel_size=3, padding=1)

    def encode(self, q_lr: torch.Tensor) -> torch.Tensor:
        return self.codec.encode_map(q_lr)

    def forward(self, q_lr: torch.Tensor) -> dict[str, torch.Tensor]:
        x_lr = self.encode(q_lr)
        x4 = self.in4(x_lr)
        skip = x_lr

        for stage in self.stages:
            x4, skip = stage(x4, skip)

        x_hr_raw = self.out4(x4) + skip
        radius = self.codec.a4.norm().to(device=x_hr_raw.device, dtype=x_hr_raw.dtype)
        denom = torch.sqrt((x_hr_raw ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12))
        x_hr = radius * x_hr_raw / denom

        return {
            "descriptor_lr": x_lr,
            "descriptor_hr_raw": x_hr_raw,
            "descriptor_hr": x_hr,
        }

    def forward_descriptor(self, q_lr: torch.Tensor) -> torch.Tensor:
        return self.forward(q_lr)["descriptor_hr"]

    @torch.no_grad()
    def forward_quaternion_nn(
        self,
        q_lr: torch.Tensor,
        q_dict: torch.Tensor,
        x_dict: Optional[torch.Tensor] = None,
        chunk: int = 4096,
        topk: int = 1,
    ) -> DecodeResult:
        x_hr = self.forward_descriptor(q_lr)
        x_hr_last = x_hr.permute(0, 2, 3, 1).contiguous()
        return self.codec.decode_by_dictionary(
            x_hr_last,
            q_dict=q_dict,
            x_dict=x_dict,
            chunk=chunk,
            topk=topk,
        )


SO3OQuotientSRSimple = SO3OQuotientSRSimpleE3NN
SO3OQuotientSRNetSimple = SO3OQuotientSRSimpleE3NN


def descriptor_mse_loss(x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_pred, x_gt)


def descriptor_shell_loss(x_pred_raw: torch.Tensor, target_radius: float = 1.0) -> torch.Tensor:
    r = torch.sqrt((x_pred_raw ** 2).sum(dim=1).clamp_min(1e-12))
    return ((r - target_radius) ** 2).mean()


def combined_simple_loss(
    model: SO3OQuotientSRSimpleE3NN,
    pred: dict[str, torch.Tensor],
    hr_q_gt: torch.Tensor,
    lam_shell: float = 1e-2,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        x_gt = model.codec.encode_map(hr_q_gt)

    desc = descriptor_mse_loss(pred["descriptor_hr"], x_gt)
    shell = descriptor_shell_loss(pred["descriptor_hr_raw"], target_radius=model.codec.descriptor_radius)
    total = desc + lam_shell * shell
    return {"descriptor": desc, "shell": shell, "total": total}


def _demo() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    b, h, w = 2, 16, 20

    q_lr = rand_quaternion_grid(b, h, w, device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()
    model = SO3OQuotientSRSimpleE3NN(
        hidden_mul4=8,
        num_blocks_per_stage=1,
        sr_scale=4,
        passive_input=False,
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    out = model(q_lr)
    print("descriptor_lr:", tuple(out["descriptor_lr"].shape))
    print("descriptor_hr:", tuple(out["descriptor_hr"].shape))

    q_dict, x_dict = model.codec.build_dictionary(n=2000, device=device, dtype=dtype)
    dec = model.forward_quaternion_nn(q_lr, q_dict=q_dict, x_dict=x_dict, topk=3)
    print("decoded quaternions:", tuple(dec.quaternions.shape))


if __name__ == "__main__":
    _demo()
