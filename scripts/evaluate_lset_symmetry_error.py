#!/usr/bin/env python3
"""Evaluate symmetry-error quality gates for multiple cubic harmonic L-sets.

This script answers: "is L={4} good enough, or should we add higher degrees?"

It samples many random unit quaternions, computes e3nn cubic-invariant features,
and reports:
  1) invariance residual under cubic crystal action
  2) nearest-neighbor symmetry-aware misorientation (mean/p50/p95)
  3) collision rate above tolerance tau

It then applies threshold gates and recommends the smallest L-set that passes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / "models" / "e3nn_invariant_autoencoder.py"
_spec = importlib.util.spec_from_file_location("e3nn_invariant_autoencoder", MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
E3nnInvariantAutoencoderBunge = _module.E3nnInvariantAutoencoderBunge


@dataclass
class EvalResult:
    ls_text: str
    ls: tuple[int, ...]
    seed: int
    num_samples: int
    feature_dim: int
    inv_residual_mean: float
    inv_residual_max: float
    se_mean_deg: float
    se_p50_deg: float
    se_p95_deg: float
    collision_rate_tau: float


def parse_int_list(text: str) -> tuple[int, ...]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty integer list, got '{text}'")
    return tuple(vals)


def parse_lsets(text: str) -> list[tuple[int, ...]]:
    groups = [g.strip() for g in text.split(";") if g.strip()]
    if not groups:
        raise ValueError(f"Expected non-empty L-set list, got '{text}'")
    return [parse_int_list(g) for g in groups]


def parse_seeds(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty seed list, got '{text}'")
    return vals


def lset_to_text(ls: Iterable[int]) -> str:
    return ",".join(str(int(x)) for x in ls)


def sample_unit_quaternions(n: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn((n, 4), generator=g, device=device, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q


def wigner_features_batched(
    model: E3nnInvariantAutoencoderBunge,
    q: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, q.shape[0], batch_size):
            chunks.append(model._wigner_invariant_features(q[i : i + batch_size]))
    return torch.cat(chunks, dim=0)


def invariance_residual(
    model: E3nnInvariantAutoencoderBunge,
    q: torch.Tensor,
    z_ref: torch.Tensor,
    batch_size: int,
) -> tuple[float, float]:
    # Metric:
    #   eps_inv = E_q,s ||E(q) - E(s^{-1} ⊗ q)||_2^2
    # under Bunge passive left crystal action.
    total = 0.0
    count = 0
    max_val = 0.0
    with torch.no_grad():
        for s in model.fcc_syms_inv:
            s_batch = s.unsqueeze(0).expand_as(q)
            q_sym = model.quat_mul(s_batch, q)
            z_sym = wigner_features_batched(model, q_sym, batch_size=batch_size)
            d2 = (z_sym - z_ref).pow(2).sum(dim=-1)
            total += float(d2.sum().item())
            count += int(d2.numel())
            max_val = max(max_val, float(d2.max().item()))
    mean_val = total / max(count, 1)
    return mean_val, max_val


def min_orbit_misorientation_deg(
    model: E3nnInvariantAutoencoderBunge,
    q_a: torch.Tensor,
    q_b: torch.Tensor,
) -> torch.Tensor:
    # d_G(q_a, q_b) = min_s 2*acos(|<q_a, s^{-1} ⊗ q_b>|)
    q_a64 = model._normalize_quaternions(q_a.to(torch.float64))
    q_b64 = model._normalize_quaternions(q_b.to(torch.float64))
    n = q_a64.shape[0]
    g = model.fcc_syms_inv.shape[0]
    syms = model.fcc_syms_inv.to(torch.float64).unsqueeze(0).expand(n, -1, -1)
    q_exp = q_b64.unsqueeze(1).expand(-1, g, -1)
    orbit = model.quat_mul(syms, q_exp)
    orbit = model._normalize_quaternions(orbit)
    dots = (q_a64.unsqueeze(1) * orbit).sum(dim=-1).abs().clamp(0.0, 1.0)
    ang = 2.0 * torch.acos(dots).min(dim=1).values
    return ang * (180.0 / math.pi)


def nearest_neighbor_symmetry_error(
    model: E3nnInvariantAutoencoderBunge,
    q: torch.Tensor,
    z: torch.Tensor,
    tau_deg: float,
) -> tuple[float, float, float, float]:
    # Brute-force nearest neighbor in feature space.
    zf = z.to(torch.float32)
    dist = torch.cdist(zf, zf, p=2)
    idx = torch.arange(dist.shape[0], device=dist.device)
    dist[idx, idx] = float("inf")
    nn_idx = dist.argmin(dim=1)
    q_nn = q[nn_idx]

    theta_deg = min_orbit_misorientation_deg(model, q, q_nn)
    se_mean = float(theta_deg.mean().item())
    se_p50 = float(torch.quantile(theta_deg, 0.50).item())
    se_p95 = float(torch.quantile(theta_deg, 0.95).item())
    collision_rate = float((theta_deg > tau_deg).to(torch.float32).mean().item())
    return se_mean, se_p50, se_p95, collision_rate


def evaluate_one(
    ls: tuple[int, ...],
    seed: int,
    num_samples: int,
    batch_size: int,
    normalize_features: bool,
    device: torch.device,
    tau_deg: float,
) -> EvalResult:
    model = E3nnInvariantAutoencoderBunge(
        device=str(device),
        Ls=ls,
        stack_re_im=True,
        normalize_wigner_features=normalize_features,
        latent_dim=8,
        encoder_hidden_dim=16,
        encoder_layers=1,
        decoder_hidden_dim=16,
        decoder_layers=1,
    )
    model.eval()

    q = sample_unit_quaternions(num_samples, seed=seed, device=device)
    z = wigner_features_batched(model, q, batch_size=batch_size)
    inv_mean, inv_max = invariance_residual(model, q, z, batch_size=batch_size)
    se_mean, se_p50, se_p95, collision = nearest_neighbor_symmetry_error(
        model=model,
        q=q,
        z=z,
        tau_deg=tau_deg,
    )

    return EvalResult(
        ls_text=lset_to_text(ls),
        ls=ls,
        seed=int(seed),
        num_samples=int(num_samples),
        feature_dim=int(z.shape[-1]),
        inv_residual_mean=float(inv_mean),
        inv_residual_max=float(inv_max),
        se_mean_deg=float(se_mean),
        se_p50_deg=float(se_p50),
        se_p95_deg=float(se_p95),
        collision_rate_tau=float(collision),
    )


def aggregate(results: list[EvalResult]) -> list[dict[str, float | str | int]]:
    grouped: dict[str, list[EvalResult]] = {}
    for r in results:
        grouped.setdefault(r.ls_text, []).append(r)

    out: list[dict[str, float | str | int]] = []
    for ls_text, rows in grouped.items():
        def mean_std(values: list[float]) -> tuple[float, float]:
            arr = np.asarray(values, dtype=np.float64)
            return float(arr.mean()), float(arr.std(ddof=0))

        inv_mean_m, inv_mean_s = mean_std([r.inv_residual_mean for r in rows])
        inv_max_m, inv_max_s = mean_std([r.inv_residual_max for r in rows])
        se_mean_m, se_mean_s = mean_std([r.se_mean_deg for r in rows])
        se_p50_m, se_p50_s = mean_std([r.se_p50_deg for r in rows])
        se_p95_m, se_p95_s = mean_std([r.se_p95_deg for r in rows])
        col_m, col_s = mean_std([r.collision_rate_tau for r in rows])

        out.append(
            {
                "ls_text": ls_text,
                "num_seeds": len(rows),
                "feature_dim": int(rows[0].feature_dim),
                "inv_residual_mean_mean": inv_mean_m,
                "inv_residual_mean_std": inv_mean_s,
                "inv_residual_max_mean": inv_max_m,
                "inv_residual_max_std": inv_max_s,
                "se_mean_deg_mean": se_mean_m,
                "se_mean_deg_std": se_mean_s,
                "se_p50_deg_mean": se_p50_m,
                "se_p50_deg_std": se_p50_s,
                "se_p95_deg_mean": se_p95_m,
                "se_p95_deg_std": se_p95_s,
                "collision_rate_tau_mean": col_m,
                "collision_rate_tau_std": col_s,
            }
        )
    return out


def write_per_seed_csv(path: Path, rows: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ls_text",
        "seed",
        "num_samples",
        "feature_dim",
        "inv_residual_mean",
        "inv_residual_max",
        "se_mean_deg",
        "se_p50_deg",
        "se_p95_deg",
        "collision_rate_tau",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "ls_text": r.ls_text,
                    "seed": r.seed,
                    "num_samples": r.num_samples,
                    "feature_dim": r.feature_dim,
                    "inv_residual_mean": r.inv_residual_mean,
                    "inv_residual_max": r.inv_residual_max,
                    "se_mean_deg": r.se_mean_deg,
                    "se_p50_deg": r.se_p50_deg,
                    "se_p95_deg": r.se_p95_deg,
                    "collision_rate_tau": r.collision_rate_tau,
                }
            )


def write_agg_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def choose_smallest_passing(
    agg_rows: list[dict[str, float | str | int]],
    lsets_order: list[tuple[int, ...]],
    inv_thresh: float,
    se_p95_thresh: float,
    collision_thresh: float,
) -> str | None:
    order = [lset_to_text(ls) for ls in lsets_order]
    row_map = {str(r["ls_text"]): r for r in agg_rows}
    for ls_text in order:
        r = row_map.get(ls_text)
        if r is None:
            continue
        if (
            float(r["inv_residual_mean_mean"]) <= inv_thresh
            and float(r["se_p95_deg_mean"]) <= se_p95_thresh
            and float(r["collision_rate_tau_mean"]) <= collision_thresh
        ):
            return ls_text
    return None


def plot_metrics(
    agg_rows: list[dict[str, float | str | int]],
    lsets_order: list[tuple[int, ...]],
    tau_deg: float,
    inv_thresh: float,
    se_p95_thresh: float,
    collision_thresh: float,
    out_png: Path,
) -> None:
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [lset_to_text(ls) for ls in lsets_order]
    row_map = {str(r["ls_text"]): r for r in agg_rows}
    labels = [x for x in order if x in row_map]
    x = np.arange(len(labels))

    inv_mean = np.array([float(row_map[k]["inv_residual_mean_mean"]) for k in labels], dtype=float)
    inv_std = np.array([float(row_map[k]["inv_residual_mean_std"]) for k in labels], dtype=float)
    se_p95 = np.array([float(row_map[k]["se_p95_deg_mean"]) for k in labels], dtype=float)
    se_p95_std = np.array([float(row_map[k]["se_p95_deg_std"]) for k in labels], dtype=float)
    col = 100.0 * np.array([float(row_map[k]["collision_rate_tau_mean"]) for k in labels], dtype=float)
    col_std = 100.0 * np.array([float(row_map[k]["collision_rate_tau_std"]) for k in labels], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)

    axes[0].errorbar(x, inv_mean, yerr=inv_std, marker="o", capsize=4)
    axes[0].axhline(inv_thresh, linestyle="--", linewidth=1.2, color="tab:red")
    axes[0].set_yscale("log")
    axes[0].set_title("Invariance Residual")
    axes[0].set_ylabel(r"$\varepsilon_{\mathrm{inv}}$")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(x, se_p95, yerr=se_p95_std, marker="o", capsize=4)
    axes[1].axhline(se_p95_thresh, linestyle="--", linewidth=1.2, color="tab:red")
    axes[1].set_title("NN Symmetry Error (p95)")
    axes[1].set_ylabel("degrees")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(x, col, yerr=col_std, marker="o", capsize=4)
    axes[2].axhline(100.0 * collision_thresh, linestyle="--", linewidth=1.2, color="tab:red")
    axes[2].set_title(f"Collision Rate > {tau_deg:.2f} deg")
    axes[2].set_ylabel("%")
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Cubic-Invariant L-set Adequacy")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lsets",
        type=str,
        default="4;4,6;4,6,8;4,6,8,10;4,6,8,10,12",
        help="Semicolon-separated L-sets; each set is comma-separated.",
    )
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated integer seeds.")
    p.add_argument("--num-samples", type=int, default=4096, help="Quaternions per seed.")
    p.add_argument("--batch-size", type=int, default=1024, help="Feature compute batch size.")
    p.add_argument("--tau-deg", type=float, default=2.0, help="Collision tolerance in degrees.")
    p.add_argument("--inv-thresh", type=float, default=1e-5, help="Gate for invariance residual.")
    p.add_argument("--se-p95-thresh", type=float, default=2.0, help="Gate for p95 symmetry error.")
    p.add_argument("--collision-thresh", type=float, default=0.05, help="Gate for collision rate.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Disable final feature normalization in _wigner_invariant_features.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "lset_symmetry_eval",
        help="Output directory for CSV/JSON/PNG artifacts.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lsets = parse_lsets(args.lsets)
    seeds = parse_seeds(args.seeds)
    device = torch.device(args.device)
    normalize_features = not bool(args.no_normalize_features)

    print("L-set symmetry evaluation")
    print("=" * 72)
    print(f"lsets: {[lset_to_text(ls) for ls in lsets]}")
    print(f"seeds: {seeds}")
    print(f"num_samples per seed: {args.num_samples}")
    print(f"batch_size: {args.batch_size}")
    print(f"device: {device}")
    print(f"normalize_features: {normalize_features}")
    print(f"tau_deg: {args.tau_deg}")
    print(
        f"gates: inv<={args.inv_thresh}, se_p95<={args.se_p95_thresh}, "
        f"collision<={args.collision_thresh}"
    )
    print("=" * 72)

    results: list[EvalResult] = []
    for ls in lsets:
        ls_text = lset_to_text(ls)
        for seed in seeds:
            print(f"[run] L={ls_text}, seed={seed}")
            res = evaluate_one(
                ls=ls,
                seed=seed,
                num_samples=int(args.num_samples),
                batch_size=int(args.batch_size),
                normalize_features=normalize_features,
                device=device,
                tau_deg=float(args.tau_deg),
            )
            results.append(res)
            print(
                "  inv_mean={:.3e} inv_max={:.3e} "
                "SE(p95)={:.3f}deg collision={:.2f}%".format(
                    res.inv_residual_mean,
                    res.inv_residual_max,
                    res.se_p95_deg,
                    100.0 * res.collision_rate_tau,
                )
            )

    agg_rows = aggregate(results)
    picked = choose_smallest_passing(
        agg_rows=agg_rows,
        lsets_order=lsets,
        inv_thresh=float(args.inv_thresh),
        se_p95_thresh=float(args.se_p95_thresh),
        collision_thresh=float(args.collision_thresh),
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_per_seed_csv(out_dir / "per_seed_metrics.csv", results)
    write_agg_csv(out_dir / "aggregated_metrics.csv", agg_rows)
    plot_metrics(
        agg_rows=agg_rows,
        lsets_order=lsets,
        tau_deg=float(args.tau_deg),
        inv_thresh=float(args.inv_thresh),
        se_p95_thresh=float(args.se_p95_thresh),
        collision_thresh=float(args.collision_thresh),
        out_png=out_dir / "metrics_summary.png",
    )

    summary = {
        "lsets": [lset_to_text(ls) for ls in lsets],
        "seeds": seeds,
        "num_samples": int(args.num_samples),
        "tau_deg": float(args.tau_deg),
        "inv_thresh": float(args.inv_thresh),
        "se_p95_thresh": float(args.se_p95_thresh),
        "collision_thresh": float(args.collision_thresh),
        "picked_smallest_passing_lset": picked,
        "normalize_features": normalize_features,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 72)
    if picked is None:
        print("No L-set passed all gates with current thresholds.")
    else:
        print(f"Recommended smallest passing L-set: {picked}")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
