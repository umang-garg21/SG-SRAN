"""
Stage-wise group-property diagnostics for IsoEmbeddingSRAttn.

Checks, per stage:
1) left SO(3) equivariance
2) right invariance wrt crystal symmetry group
3) left equivariance wrt crystal symmetry group
4) right equivariance wrt crystal symmetry group

No argparse. Edit CONFIG directly.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.SR_double_conv_SRattn import IsoEmbeddingSRAttn


CONFIG = {
    "crystal": "fcc",  # "fcc" or "hcp"
    "d6_convention": "z_axis",
    "lr_h": 2,
    "lr_w": 2,
    "upsample_factor": 2,
    "device": "cpu",
    "seed": 0,
    "num_hr_attn_blocks": 1,
    "hr_attn_num_channels": 8,
    "hr_attn_block_size": 16,
    "decoder_cubochoric_resolution": 1,
    "decoder_num_starts": 1,
    "decoder_steps": 0,
    "decoder_lr": 0.05,
    "decoder_method": "cubochoric",
    "num_so3_trials": 6,
    "num_sym_trials": 12,  # excludes identity when possible
    "tol_rel": 5e-3,
    "tol_rms": 2e-4,
    "out_json": "diagnostics/layer_group_properties.json",
}


def _normalize_quaternions(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(q[..., :1] < 0.0, -q, q)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    out = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    return _normalize_quaternions(out)


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


def _left_action(q: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    gg = g.view(1, 4).expand(q.shape[0], 4)
    return _quat_mul(gg, q)


def _right_action(q: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    gg = g.view(1, 4).expand(q.shape[0], 4)
    return _quat_mul(q, gg)


def _sample_unit_quaternions(n: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    q = torch.randn(n, 4, generator=gen, device=device, dtype=torch.float32)
    return _normalize_quaternions(q)


def _sample_random_so3_quat(seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    axis = torch.randn(3, generator=gen, device=device, dtype=torch.float32)
    axis = axis / axis.norm().clamp_min(1e-12)
    angle = 2.0 * math.pi * torch.rand((), generator=gen, device=device, dtype=torch.float32)
    half = 0.5 * angle
    s = torch.sin(half)
    q = torch.cat([torch.cos(half).view(1), axis * s], dim=0)
    return _normalize_quaternions(q.view(1, 4))[0]


def _err_metrics(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    diff = (x - y).detach()
    rms = float(torch.sqrt(torch.mean(diff * diff)).item())
    rel = float(
        torch.linalg.norm(diff).item()
        / (torch.linalg.norm(y.detach()).item() + 1e-12)
    )
    return rel, rms


def _best_feature_equivariance(
    y_ref: torch.Tensor,
    y_trans: torch.Tensor,
    irreps,
    g: torch.Tensor,
) -> tuple[float, float, str]:
    q = _normalize_quaternions(g.view(1, 4))[0]
    q_conj = _quat_conjugate(q.view(1, 4))[0]
    R = _quat_to_matrix_active(q.view(1, 4))[0]
    Rc = _quat_to_matrix_active(q_conj.view(1, 4))[0]

    D = irreps.D_from_matrix(R).to(y_ref.device, y_ref.dtype)
    Dc = irreps.D_from_matrix(Rc).to(y_ref.device, y_ref.dtype)

    candidates = [
        ("D(g)^T", y_ref @ D.T),
        ("D(g)", y_ref @ D),
        ("D(conj(g))^T", y_ref @ Dc.T),
        ("D(conj(g))", y_ref @ Dc),
    ]

    best = None
    for label, y_expected in candidates:
        rel, rms = _err_metrics(y_trans, y_expected)
        if best is None or (rel < best[0]) or (abs(rel - best[0]) < 1e-12 and rms < best[1]):
            best = (rel, rms, label)
    assert best is not None
    return best


def _apply_attention_stepwise(model, feat: torch.Tensor, hr_shape: tuple[int, int]):
    hr_h, hr_w = hr_shape
    batched = feat.dim() == 3
    if not batched:
        feat = feat.unsqueeze(0)

    bsz, n, c = feat.shape
    if n != hr_h * hr_w:
        raise ValueError(f"Expected N={hr_h*hr_w}, got {n}")

    block_h = min(model.hr_attn_block_size, hr_h)
    block_w = min(model.hr_attn_block_size, hr_w)
    pad_h = (-hr_h) % block_h
    pad_w = (-hr_w) % block_w
    hr_h_pad, hr_w_pad = hr_h + pad_h, hr_w + pad_w

    feat_work = feat
    if pad_h > 0 or pad_w > 0:
        feat_2d = feat_work.reshape(bsz, hr_h, hr_w, c).permute(0, 3, 1, 2)
        feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode="reflect")
        feat_work = feat_2d.permute(0, 2, 3, 1).reshape(bsz, hr_h_pad * hr_w_pad, c)

    def _unpad(x: torch.Tensor) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return x
        return x.reshape(bsz, hr_h_pad, hr_w_pad, c)[:, :hr_h, :hr_w, :].reshape(bsz, hr_h * hr_w, c)

    sh_block = model._get_hr_sh_block(block_h, block_w, feat_work.device, feat_work.dtype)

    stage_outputs = []
    for idx, block in enumerate(model.attention_blocks):
        delta = block(feat_work, sh_block, hr_h_pad, hr_w_pad, block_h, block_w)
        feat_work = feat_work + delta
        cur = _unpad(feat_work)
        if not batched:
            cur = cur.squeeze(0)
        stage_outputs.append((f"attention_block_{idx}_output", cur))

    feat_final = _unpad(feat_work)
    if not batched:
        feat_final = feat_final.squeeze(0)
    return feat_final, stage_outputs


def _run_stages(model, q_lr: torch.Tensor, lr_shape: tuple[int, int]):
    with torch.no_grad():
        feat_a1_lr = model.encode_a1(q_lr)
        feat_lr1 = model.conv_lr1(feat_a1_lr, lr_shape)
        feat_lr2 = model.conv_lr2(feat_lr1, lr_shape)
        feat_up, hr_shape = model.upsample_conv(feat_lr2, lr_shape)
        feat_hr1 = model.conv_hr1(feat_up, hr_shape)
        feat_attn, attn_stage_outputs = _apply_attention_stepwise(model, feat_hr1, hr_shape)
        feat_a1_hr = model.final_proj(feat_attn)
        q_dec_raw = model.decoder(feat_a1_hr)
        q_dec_fz = model.reduce_to_fz(q_dec_raw)
        q_forward = model.forward_sr(q_lr, lr_shape=lr_shape, normalize_input=False)

    stages = [
        {"name": "input_quats_lr", "tensor": q_lr, "kind": "quat", "irreps": None},
        {"name": "encode_a1_lr", "tensor": feat_a1_lr, "kind": "feature", "irreps": model.irreps_a1},
        {"name": "conv_lr1_output", "tensor": feat_lr1, "kind": "feature", "irreps": model.irreps_full},
        {"name": "conv_lr2_output", "tensor": feat_lr2, "kind": "feature", "irreps": model.irreps_full},
        {"name": "upsample_output", "tensor": feat_up, "kind": "feature", "irreps": model.irreps_full},
        {"name": "conv_hr1_output", "tensor": feat_hr1, "kind": "feature", "irreps": model.irreps_full},
    ]
    for nm, t in attn_stage_outputs:
        stages.append({"name": nm, "tensor": t, "kind": "feature", "irreps": model.irreps_full})
    stages.extend(
        [
            {"name": "attention_output", "tensor": feat_attn, "kind": "feature", "irreps": model.irreps_full},
            {"name": "final_proj_output_a1", "tensor": feat_a1_hr, "kind": "feature", "irreps": model.irreps_a1},
            {"name": "decoder_raw_output", "tensor": q_dec_raw, "kind": "quat", "irreps": None},
            {"name": "decoder_fz_output", "tensor": q_dec_fz, "kind": "quat", "irreps": None},
            {"name": "forward_sr_output", "tensor": q_forward, "kind": "quat", "irreps": None},
        ]
    )
    return stages


def _new_stats():
    return {
        "n": 0,
        "sum_rel": 0.0,
        "sum_rms": 0.0,
        "max_rel": 0.0,
        "max_rms": 0.0,
        "pass_count": 0,
        "variant_counts": defaultdict(int),
    }


def _update_stats(stats, rel: float, rms: float, passed: bool, variant: str | None):
    stats["n"] += 1
    stats["sum_rel"] += rel
    stats["sum_rms"] += rms
    stats["max_rel"] = max(stats["max_rel"], rel)
    stats["max_rms"] = max(stats["max_rms"], rms)
    if passed:
        stats["pass_count"] += 1
    if variant is not None:
        stats["variant_counts"][variant] += 1


def _summarize_stats(stats):
    n = max(1, stats["n"])
    variant_counts = dict(stats["variant_counts"])
    best_variant = None
    if len(variant_counts) > 0:
        best_variant = max(variant_counts.items(), key=lambda kv: kv[1])[0]
    return {
        "n": stats["n"],
        "mean_rel": stats["sum_rel"] / n,
        "mean_rms": stats["sum_rms"] / n,
        "max_rel": stats["max_rel"],
        "max_rms": stats["max_rms"],
        "pass_rate": stats["pass_count"] / n,
        "best_variant": best_variant,
        "variant_counts": variant_counts,
    }


def _format_num(x: float) -> str:
    return f"{x: .3e}"


def _print_property_table(property_name: str, result_dict):
    print(f"\n=== {property_name} ===")
    print(
        "stage".ljust(32)
        + " mean_rel".rjust(12)
        + " max_rel".rjust(12)
        + " mean_rms".rjust(12)
        + " max_rms".rjust(12)
        + " pass_rate".rjust(11)
        + "  verdict"
        + "  best_variant"
    )
    for stage_name, row in result_dict.items():
        verdict = "PASS" if row["pass_rate"] >= 0.95 else "FAIL"
        print(
            stage_name.ljust(32)
            + _format_num(row["mean_rel"]).rjust(12)
            + _format_num(row["max_rel"]).rjust(12)
            + _format_num(row["mean_rms"]).rjust(12)
            + _format_num(row["max_rms"]).rjust(12)
            + f"{row['pass_rate']:8.2%}".rjust(11)
            + f"  {verdict:>4s}"
            + f"  {str(row.get('best_variant'))}"
        )


def main() -> None:
    device = torch.device(str(CONFIG["device"]))
    lr_shape = (int(CONFIG["lr_h"]), int(CONFIG["lr_w"]))
    n_lr = lr_shape[0] * lr_shape[1]
    tol_rel = float(CONFIG["tol_rel"])
    tol_rms = float(CONFIG["tol_rms"])

    model = IsoEmbeddingSRAttn(
        crystal=str(CONFIG["crystal"]),
        d6_convention=str(CONFIG["d6_convention"]),
        device=device,
        upsample_factor=int(CONFIG["upsample_factor"]),
        num_hr_attn_blocks=int(CONFIG["num_hr_attn_blocks"]),
        hr_attn_num_channels=int(CONFIG["hr_attn_num_channels"]),
        hr_attn_block_size=int(CONFIG["hr_attn_block_size"]),
        decoder_cubochoric_resolution=int(CONFIG["decoder_cubochoric_resolution"]),
        decoder_num_starts=int(CONFIG["decoder_num_starts"]),
        decoder_steps=int(CONFIG["decoder_steps"]),
        decoder_lr=float(CONFIG["decoder_lr"]),
        decoder_method=str(CONFIG["decoder_method"]),
    ).eval()

    q_base = _sample_unit_quaternions(n_lr, seed=int(CONFIG["seed"]), device=device)
    base_stages = _run_stages(model, q_base, lr_shape)
    stage_names = [s["name"] for s in base_stages]
    base_map = {s["name"]: s for s in base_stages}

    # Stats by property -> stage.
    properties = {
        "left_so3_equivariant": {nm: _new_stats() for nm in stage_names},
        "right_sym_invariant": {nm: _new_stats() for nm in stage_names},
        "left_sym_equivariant": {nm: _new_stats() for nm in stage_names},
        "right_sym_equivariant": {nm: _new_stats() for nm in stage_names},
    }

    # 1) Left SO(3) equivariance
    for t in range(int(CONFIG["num_so3_trials"])):
        g = _sample_random_so3_quat(seed=10_000 + t, device=device)
        q_t = _left_action(q_base, g)
        trans_map = {s["name"]: s for s in _run_stages(model, q_t, lr_shape)}
        for nm in stage_names:
            s_ref = base_map[nm]
            s_tr = trans_map[nm]
            if s_ref["kind"] == "feature":
                rel, rms, var = _best_feature_equivariance(
                    s_ref["tensor"], s_tr["tensor"], s_ref["irreps"], g
                )
            else:
                y_expected = _left_action(s_ref["tensor"], g)
                rel, rms = _err_metrics(s_tr["tensor"], y_expected)
                var = "quat_left_mult"
            passed = (rel <= tol_rel) or (rms <= tol_rms)
            _update_stats(properties["left_so3_equivariant"][nm], rel, rms, passed, var)

    # Symmetry-group elements (exclude identity when possible).
    sym_ops = model.encoder.sym_ops.detach()
    if sym_ops.shape[0] > 1:
        sym_candidates = sym_ops[1:]
    else:
        sym_candidates = sym_ops
    n_sym = min(int(CONFIG["num_sym_trials"]), int(sym_candidates.shape[0]))
    sym_list = [sym_candidates[i] for i in range(n_sym)]

    # 2) Right symmetry invariance: F(q * s) = F(q)
    for s in sym_list:
        q_t = _right_action(q_base, s)
        trans_map = {st["name"]: st for st in _run_stages(model, q_t, lr_shape)}
        for nm in stage_names:
            s_ref = base_map[nm]
            s_tr = trans_map[nm]
            rel, rms = _err_metrics(s_tr["tensor"], s_ref["tensor"])
            passed = (rel <= tol_rel) or (rms <= tol_rms)
            _update_stats(properties["right_sym_invariant"][nm], rel, rms, passed, "identity")

    # 3) Left symmetry equivariance: F(s * q) = rho(s) F(q)
    for s in sym_list:
        q_t = _left_action(q_base, s)
        trans_map = {st["name"]: st for st in _run_stages(model, q_t, lr_shape)}
        for nm in stage_names:
            s_ref = base_map[nm]
            s_tr = trans_map[nm]
            if s_ref["kind"] == "feature":
                rel, rms, var = _best_feature_equivariance(
                    s_ref["tensor"], s_tr["tensor"], s_ref["irreps"], s
                )
            else:
                y_expected = _left_action(s_ref["tensor"], s)
                rel, rms = _err_metrics(s_tr["tensor"], y_expected)
                var = "quat_left_mult"
            passed = (rel <= tol_rel) or (rms <= tol_rms)
            _update_stats(properties["left_sym_equivariant"][nm], rel, rms, passed, var)

    # 4) Right symmetry equivariance: F(q * s) = rho(s) F(q)
    for s in sym_list:
        q_t = _right_action(q_base, s)
        trans_map = {st["name"]: st for st in _run_stages(model, q_t, lr_shape)}
        for nm in stage_names:
            s_ref = base_map[nm]
            s_tr = trans_map[nm]
            if s_ref["kind"] == "feature":
                rel, rms, var = _best_feature_equivariance(
                    s_ref["tensor"], s_tr["tensor"], s_ref["irreps"], s
                )
            else:
                y_expected = _right_action(s_ref["tensor"], s)
                rel, rms = _err_metrics(s_tr["tensor"], y_expected)
                var = "quat_right_mult"
            passed = (rel <= tol_rel) or (rms <= tol_rms)
            _update_stats(properties["right_sym_equivariant"][nm], rel, rms, passed, var)

    summary = {
        "config": CONFIG,
        "stages": stage_names,
        "results": {
            prop: {nm: _summarize_stats(st) for nm, st in by_stage.items()}
            for prop, by_stage in properties.items()
        },
    }

    for prop_name, by_stage in summary["results"].items():
        _print_property_table(prop_name, by_stage)

    out_json = Path(CONFIG["out_json"]).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved report: {out_json}")


if __name__ == "__main__":
    main()

