"""
Test EquivariantSpatialConv with and without residual skip.

What this script does:
1) builds paired conv layers with identical TP/spatial weights
   - one with use_residual=True
   - one with use_residual=False
2) runs both on the same input features
3) reports how much output changes due to the residual path
4) verifies the residual decomposition numerically
5) runs a lightweight SO(3)-equivariance sanity check for both versions

No argparse: edit CONFIG directly.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import torch
from e3nn.o3 import Irreps

# Allow direct execution: `python scripts/test_equivariant_spatial_conv_residual.py`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.SR_double_conv_SRattn import (  # noqa: E402
    EquivariantSpatialConv,
    LocalIsoCrystalEncoder,
)


CONFIG = {
    "crystal": "hcp",  # "fcc" or "hcp"
    "d6_convention": "z_axis",
    "device": "cpu",
    "h": 12,
    "w": 10,
    "seed": 0,
    "num_equivariance_trials": 8,
}


def _normalize_quaternions(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0.0, -q, q)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_to_matrix_active(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quaternions(q)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    )
    return R.reshape(*q.shape[:-1], 3, 3)


def _err_metrics(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    d = (x - y).detach()
    rms = float(torch.sqrt(torch.mean(d * d)).item())
    rel = float(torch.linalg.norm(d).item() / (torch.linalg.norm(y.detach()).item() + 1e-12))
    return rel, rms


def _print_tensor_stats(name: str, t: torch.Tensor) -> None:
    x = t.detach()
    print(
        f"{name:<30s}"
        f" shape={tuple(x.shape)}"
        f" min={float(x.min().item()): .3e}"
        f" max={float(x.max().item()): .3e}"
        f" mean={float(x.mean().item()): .3e}"
        f" std={float(x.std(unbiased=False).item()): .3e}"
    )


def _random_unit_quats(n: int, *, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    q = torch.randn(n, 4, generator=g, device=device, dtype=torch.float32)
    return _normalize_quaternions(q)


def _apply_variant(y: torch.Tensor, D: torch.Tensor, Dc: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "D(g)^T":
        return y @ D.T
    if variant == "D(g)":
        return y @ D
    if variant == "D(conj(g))^T":
        return y @ Dc.T
    if variant == "D(conj(g))":
        return y @ Dc
    raise ValueError(f"Unknown variant: {variant}")


def _equivariance_summary(
    conv: EquivariantSpatialConv,
    features: torch.Tensor,
    img_shape: tuple[int, int],
    irreps_in: Irreps,
    irreps_out: Irreps,
    trials: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    variants = ["D(g)^T", "D(g)", "D(conj(g))^T", "D(conj(g))"]
    rels = []
    rmss = []
    best_variant_counts = Counter()

    y_ref = conv(features, img_shape).detach()

    for t in range(int(trials)):
        g = _random_unit_quats(1, seed=seed + 1000 + t, device=device)[0]
        gc = _quat_conjugate(g.view(1, 4))[0]
        R = _quat_to_matrix_active(g.view(1, 4))[0]
        Rc = _quat_to_matrix_active(gc.view(1, 4))[0]

        D_in = irreps_in.D_from_matrix(R).to(features.device, features.dtype)
        Dc_in = irreps_in.D_from_matrix(Rc).to(features.device, features.dtype)
        D_out = irreps_out.D_from_matrix(R).to(features.device, features.dtype)
        Dc_out = irreps_out.D_from_matrix(Rc).to(features.device, features.dtype)

        best = None
        best_name = None
        for v in variants:
            x_t = _apply_variant(features, D_in, Dc_in, v)
            y_t = conv(x_t, img_shape).detach()
            y_exp = _apply_variant(y_ref, D_out, Dc_out, v)
            rel, rms = _err_metrics(y_t, y_exp)
            if best is None or rel < best[0] or (abs(rel - best[0]) < 1e-12 and rms < best[1]):
                best = (rel, rms)
                best_name = v

        assert best is not None
        assert best_name is not None
        rels.append(best[0])
        rmss.append(best[1])
        best_variant_counts[best_name] += 1

    rel_t = torch.tensor(rels)
    rms_t = torch.tensor(rmss)
    top_variant = None
    if len(best_variant_counts) > 0:
        top_variant = best_variant_counts.most_common(1)[0][0]
    return {
        "mean_rel": float(rel_t.mean().item()),
        "max_rel": float(rel_t.max().item()),
        "mean_rms": float(rms_t.mean().item()),
        "max_rms": float(rms_t.max().item()),
        "top_variant": top_variant,
        "variant_counts": dict(best_variant_counts),
    }


def _make_pair(
    irreps_in: Irreps,
    irreps_out: Irreps,
    kernel_size: int,
    seed: int,
    device: torch.device,
) -> tuple[EquivariantSpatialConv, EquivariantSpatialConv]:
    torch.manual_seed(int(seed))
    conv_res = EquivariantSpatialConv(
        kernel_size=kernel_size,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        use_residual=True,
    ).to(device)
    conv_nores = EquivariantSpatialConv(
        kernel_size=kernel_size,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        use_residual=False,
    ).to(device)

    with torch.no_grad():
        conv_nores.tp.load_state_dict(conv_res.tp.state_dict())
        conv_nores.spatial_weights.copy_(conv_res.spatial_weights)
    return conv_res.eval(), conv_nores.eval()


def _run_case(
    case_name: str,
    conv_res: EquivariantSpatialConv,
    conv_nores: EquivariantSpatialConv,
    features: torch.Tensor,
    img_shape: tuple[int, int],
    irreps_in: Irreps,
    irreps_out: Irreps,
    cfg: dict,
) -> None:
    print("\n" + "=" * 96)
    print(case_name)
    _print_tensor_stats("input", features)

    with torch.no_grad():
        y_nores = conv_nores(features, img_shape)
        y_res = conv_res(features, img_shape)
        delta = y_res - y_nores

        n = int(features.shape[0])
        feat_flat = features.reshape(n, -1)
        if conv_res.residual_proj is None:
            residual_term = feat_flat
        else:
            residual_term = conv_res.residual_proj(feat_flat)

        decomp_rel, decomp_rms = _err_metrics(delta, residual_term)
        diff_rel, diff_rms = _err_metrics(y_res, y_nores)

    _print_tensor_stats("output_no_residual", y_nores)
    _print_tensor_stats("output_with_residual", y_res)
    _print_tensor_stats("residual_contribution", delta)
    print(
        "difference(with-vs-without):"
        f" rel={diff_rel:.3e} rms={diff_rms:.3e}"
    )
    print(
        "decomposition check (delta == residual_term):"
        f" rel={decomp_rel:.3e} rms={decomp_rms:.3e}"
    )

    eq_res = _equivariance_summary(
        conv=conv_res,
        features=features,
        img_shape=img_shape,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        trials=int(cfg["num_equivariance_trials"]),
        seed=int(cfg["seed"]),
        device=features.device,
    )
    eq_nores = _equivariance_summary(
        conv=conv_nores,
        features=features,
        img_shape=img_shape,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        trials=int(cfg["num_equivariance_trials"]),
        seed=int(cfg["seed"]),
        device=features.device,
    )

    print("equivariance (with residual):")
    print(
        f"  mean_rel={eq_res['mean_rel']:.3e} max_rel={eq_res['max_rel']:.3e} "
        f"mean_rms={eq_res['mean_rms']:.3e} max_rms={eq_res['max_rms']:.3e} "
        f"top_variant={eq_res['top_variant']}"
    )
    print("equivariance (without residual):")
    print(
        f"  mean_rel={eq_nores['mean_rel']:.3e} max_rel={eq_nores['max_rel']:.3e} "
        f"mean_rms={eq_nores['mean_rms']:.3e} max_rms={eq_nores['max_rms']:.3e} "
        f"top_variant={eq_nores['top_variant']}"
    )


def main() -> None:
    cfg = CONFIG
    device = torch.device(str(cfg["device"]))
    H, W = int(cfg["h"]), int(cfg["w"])
    N = H * W

    encoder = LocalIsoCrystalEncoder(
        crystal=str(cfg["crystal"]),
        d6_convention=str(cfg["d6_convention"]),
        dtype=torch.float32,
        device=device,
    ).eval()

    q = _random_unit_quats(N, seed=int(cfg["seed"]), device=device)
    feat_a1 = encoder.forward_a1(q).detach()
    feat_full = encoder.forward_full(q).detach()

    # Case A: LR conv1-style (a1 -> full), residual uses projected skip.
    conv1_res, conv1_nores = _make_pair(
        irreps_in=encoder.irreps_a1,
        irreps_out=encoder.irreps_full,
        kernel_size=3,
        seed=int(cfg["seed"]) + 10,
        device=device,
    )
    _run_case(
        case_name="Case A: a1 -> full (kernel=3)",
        conv_res=conv1_res,
        conv_nores=conv1_nores,
        features=feat_a1,
        img_shape=(H, W),
        irreps_in=encoder.irreps_a1,
        irreps_out=encoder.irreps_full,
        cfg=cfg,
    )

    # Case B: LR conv2/HR conv1-style (full -> full), residual is identity skip.
    conv2_res, conv2_nores = _make_pair(
        irreps_in=encoder.irreps_full,
        irreps_out=encoder.irreps_full,
        kernel_size=9,
        seed=int(cfg["seed"]) + 20,
        device=device,
    )
    _run_case(
        case_name="Case B: full -> full (kernel=9)",
        conv_res=conv2_res,
        conv_nores=conv2_nores,
        features=feat_full,
        img_shape=(H, W),
        irreps_in=encoder.irreps_full,
        irreps_out=encoder.irreps_full,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
