#!/usr/bin/env python3
"""
Train IrrepsToFZPassiveQuaternion as a one-shot decoder:
  irreps -> passive quaternion in FZ with positive scalar.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.irreps_to_fz_passive_quat_decoder import IrrepsToFZPassiveQuaternion
from models.local_iso_codec_model import LocalIsoCodecModel


def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
    q = quats / norm
    return torch.where(q[..., :1] < 0.0, -q, q)


def _quat_dot_loss(q_pred: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
    dots = (q_pred * q_tgt).sum(dim=-1).abs().clamp(max=1.0)
    return 1.0 - dots


def _misorientation_deg(q_pred: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
    dots = (q_pred * q_tgt).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dots) * (180.0 / math.pi)


def _parse_int_list(text: str) -> list[int]:
    vals = [x.strip() for x in str(text).split(",") if x.strip() != ""]
    return [int(v) for v in vals]


def _parse_float_list(text: str) -> list[float]:
    vals = [x.strip() for x in str(text).split(",") if x.strip() != ""]
    return [float(v) for v in vals]


@dataclass(frozen=True)
class TrialConfig:
    hidden_dim: int
    num_layers: int
    dropout: float
    residual_scale_init: float
    lr: float
    weight_decay: float


@torch.no_grad()
def _build_random_fz_dataset(
    codec: LocalIsoCodecModel,
    num_samples: int,
    seed: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = codec.device
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    feats_cpu: list[torch.Tensor] = []
    quats_cpu: list[torch.Tensor] = []
    remaining = int(num_samples)
    chunk_size = max(1, int(chunk_size))
    total = int(num_samples)
    built = 0

    while remaining > 0:
        n = min(remaining, chunk_size)
        q = torch.randn(n, 4, generator=g, dtype=torch.float32, device="cpu").to(device)
        q = _normalize_quaternions(q)
        q_fz = codec.reduce_to_fz(q)
        feat = codec.encode(q_fz, normalize_input=False)

        feats_cpu.append(feat.detach().cpu())
        quats_cpu.append(q_fz.detach().cpu())

        remaining -= n
        built += n
        if built % (5 * chunk_size) == 0 or built == total:
            print(f"[dataset] built {built}/{total} samples", flush=True)

    return torch.cat(feats_cpu, dim=0), torch.cat(quats_cpu, dim=0)


@torch.no_grad()
def _build_lookup_table_dataset(
    codec: LocalIsoCodecModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    dec = codec.decoder
    if not hasattr(dec, "table_feat") or not hasattr(dec, "table_quats"):
        raise RuntimeError(
            "Lookup-table dataset requires decoder_backend='optimizing' with table buffers."
        )
    feat = dec.table_feat.detach().cpu().contiguous()
    quat = dec.table_quats.detach().cpu().contiguous()
    if feat.shape[0] != quat.shape[0]:
        raise RuntimeError(
            f"Lookup table size mismatch: feat={tuple(feat.shape)} quat={tuple(quat.shape)}"
        )
    print(f"[dataset] loaded full lookup table rows={int(feat.shape[0])}", flush=True)
    return feat, quat


def _make_trials(args: argparse.Namespace) -> list[TrialConfig]:
    hidden_dims = _parse_int_list(args.hidden_dims)
    num_layers = _parse_int_list(args.num_layers)
    dropouts = _parse_float_list(args.dropouts)
    residual_scales = _parse_float_list(args.residual_scale_inits)
    lrs = _parse_float_list(args.lrs)

    trials = [
        TrialConfig(
            hidden_dim=h,
            num_layers=nl,
            dropout=d,
            residual_scale_init=rs,
            lr=lr,
            weight_decay=float(args.weight_decay),
        )
        for h, nl, d, rs, lr in itertools.product(
            hidden_dims, num_layers, dropouts, residual_scales, lrs
        )
    ]
    rng = random.Random(int(args.seed))
    rng.shuffle(trials)
    max_trials = min(len(trials), max(1, int(args.max_trials)))
    return trials[:max_trials]


@torch.no_grad()
def _evaluate_constraints(
    model: IrrepsToFZPassiveQuaternion,
    codec: LocalIsoCodecModel,
    feat: torch.Tensor,
    batch_size: int,
    max_eval: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    n = int(min(max_eval, int(feat.shape[0])))
    feat_eval = feat[:n]
    q_chunks: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        qb = model(feat_eval[start:end].to(device, non_blocking=True))
        q_chunks.append(qb.detach())
    q = torch.cat(q_chunks, dim=0)
    q = _normalize_quaternions(q)

    scalar_violation = float((q[:, 0] < -1e-7).float().mean().item())
    q_fz = codec.reduce_to_fz(q)
    fz_violation = float((q - q_fz).abs().max(dim=1).values.gt(1e-6).float().mean().item())
    return {
        "scalar_violation_rate": scalar_violation,
        "fz_violation_rate": fz_violation,
    }


def _train_one_trial(
    trial_idx: int,
    cfg: TrialConfig,
    codec: LocalIsoCodecModel,
    feat_train: torch.Tensor,
    quat_train: torch.Tensor,
    feat_val: torch.Tensor,
    quat_val: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    print(f"[trial {trial_idx:02d}] cfg={cfg}", flush=True)
    torch.manual_seed(int(args.seed) + trial_idx)

    model = IrrepsToFZPassiveQuaternion(
        emb=codec.encoder.embedding,
        active_only=(codec.target_irreps == "a1"),
        hidden_dim=int(cfg.hidden_dim),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
        residual_scale_init=float(cfg.residual_scale_init),
        tangent_reg=float(args.tangent_reg),
        tangent_dtype=torch.float64,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(feat_train, quat_train),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(feat_val, quat_val),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_mis = float("inf")
    best_val_loss = float("inf")
    no_improve = 0
    history: list[dict[str, float]] = []
    t_start = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, qb in train_loader:
            xb = xb.to(device, non_blocking=True)
            qb = qb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            q_pred = model(xb)
            q_loss = _quat_dot_loss(q_pred, qb).mean()
            feat_pred = codec.encode(q_pred, normalize_input=False)
            feat_loss = (feat_pred - xb).pow(2).mean()

            loss = float(args.quat_loss_weight) * q_loss + float(args.feature_loss_weight) * feat_loss
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            train_loss_sum += float(loss.item()) * int(xb.shape[0])
            train_count += int(xb.shape[0])

        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        mis_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, qb in val_loader:
                xb = xb.to(device, non_blocking=True)
                qb = qb.to(device, non_blocking=True)

                q_pred = model(xb)
                q_loss = _quat_dot_loss(q_pred, qb).mean()
                feat_pred = codec.encode(q_pred, normalize_input=False)
                feat_loss = (feat_pred - xb).pow(2).mean()
                vloss = (
                    float(args.quat_loss_weight) * q_loss
                    + float(args.feature_loss_weight) * feat_loss
                )
                val_loss_sum += float(vloss.item()) * int(xb.shape[0])
                val_count += int(xb.shape[0])
                mis_chunks.append(_misorientation_deg(q_pred, qb).detach().cpu())

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        mis = torch.cat(mis_chunks, dim=0) if mis_chunks else torch.empty(0, dtype=torch.float32)
        val_mis_mean = float(mis.mean().item()) if mis.numel() else float("inf")
        val_mis_p95 = (
            float(torch.quantile(mis, torch.tensor(0.95, dtype=mis.dtype)).item())
            if mis.numel()
            else float("inf")
        )
        val_mis_p99 = (
            float(torch.quantile(mis, torch.tensor(0.99, dtype=mis.dtype)).item())
            if mis.numel()
            else float("inf")
        )
        val_mis_max = float(mis.max().item()) if mis.numel() else float("inf")

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mis_mean_deg": val_mis_mean,
                "val_mis_p95_deg": val_mis_p95,
                "val_mis_p99_deg": val_mis_p99,
                "val_mis_max_deg": val_mis_max,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        print(
            f"[trial {trial_idx:02d}] epoch {epoch:03d}/{int(args.epochs)} "
            f"train_loss={train_loss:.6e} val_loss={val_loss:.6e} "
            f"val_mis_mean={val_mis_mean:.4f}deg val_mis_p95={val_mis_p95:.4f}deg "
            f"val_mis_p99={val_mis_p99:.4f}deg",
            flush=True,
        )

        improved = (val_mis_mean < best_val_mis - 1e-6) or (
            abs(val_mis_mean - best_val_mis) <= 1e-6 and val_loss < best_val_loss - 1e-8
        )
        if improved:
            best_val_mis = val_mis_mean
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(args.patience):
            print(
                f"[trial {trial_idx:02d}] early stop at epoch {epoch} (patience={int(args.patience)})",
                flush=True,
            )
            break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_epoch = int(args.epochs)
        best_val_mis = float("inf")
        best_val_loss = float("inf")

    model.load_state_dict(best_state, strict=True)
    constraints = _evaluate_constraints(
        model=model,
        codec=codec,
        feat=feat_val,
        batch_size=int(args.batch_size),
        max_eval=int(args.constraint_eval_samples),
        device=device,
    )

    elapsed = time.time() - t_start
    return {
        "trial_index": int(trial_idx),
        "config": asdict(cfg),
        "best_epoch": int(best_epoch),
        "best_val_mis_mean_deg": float(best_val_mis),
        "best_val_loss": float(best_val_loss),
        "elapsed_sec": float(elapsed),
        "constraint_metrics": constraints,
        "history": history,
        "best_state_dict": best_state,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train IrrepsToFZPassiveQuaternion decoder."
    )
    p.add_argument("--out_dir", type=str, default="out/irreps_to_fz_passive_quat_decoder")
    p.add_argument("--crystal", type=str, default="fcc", choices=["fcc", "hcp"])
    p.add_argument("--d6_convention", type=str, default="z_axis")
    p.add_argument("--target_irreps", type=str, default="a1", choices=["a1", "full"])
    p.add_argument("--device", type=str, default=None)

    p.add_argument(
        "--dataset_source",
        type=str,
        default="random",
        choices=["random", "lookup"],
        help="Use random FZ samples or the full decoder lookup table as supervision.",
    )

    p.add_argument("--train_samples", type=int, default=240000)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--data_chunk_size", type=int, default=65536)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--constraint_eval_samples", type=int, default=20000)

    p.add_argument("--quat_loss_weight", type=float, default=1.0)
    p.add_argument("--feature_loss_weight", type=float, default=0.25)
    p.add_argument("--tangent_reg", type=float, default=1e-8)

    p.add_argument("--hidden_dims", type=str, default="64,128")
    p.add_argument("--num_layers", type=str, default="2,3")
    p.add_argument("--dropouts", type=str, default="0.0")
    p.add_argument("--residual_scale_inits", type=str, default="0.1,0.05")
    p.add_argument("--lrs", type=str, default="0.001,0.0007")
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--max_trials", type=int, default=4)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    if args.device is None:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(str(args.device))
    print(f"[setup] device={dev}", flush=True)

    codec = LocalIsoCodecModel(
        crystal=str(args.crystal),
        d6_convention=str(args.d6_convention),
        device=dev,
        target_irreps=str(args.target_irreps),
        decoder_backend="optimizing",
        decoder_num_starts=1,
        decoder_steps=0,
        decoder_max_table_rows=None,
        decoder_table_cache_dir=None,
    ).eval()
    for p in codec.parameters():
        p.requires_grad_(False)

    if str(args.dataset_source) == "lookup":
        feat_all, quat_all = _build_lookup_table_dataset(codec=codec)
    else:
        feat_all, quat_all = _build_random_fz_dataset(
            codec=codec,
            num_samples=int(args.train_samples),
            seed=int(args.seed),
            chunk_size=int(args.data_chunk_size),
        )
    n_total = int(feat_all.shape[0])
    val_n = max(1, int(n_total * float(args.val_ratio)))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(args.seed) + 12345)
    perm = torch.randperm(n_total, generator=g)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    feat_train = feat_all[train_idx].contiguous()
    quat_train = quat_all[train_idx].contiguous()
    feat_val = feat_all[val_idx].contiguous()
    quat_val = quat_all[val_idx].contiguous()
    print(
        f"[dataset] train={int(feat_train.shape[0])} val={int(feat_val.shape[0])}",
        flush=True,
    )

    trials = _make_trials(args)
    print(f"[sweep] running {len(trials)} trial(s)", flush=True)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_results: list[dict[str, object]] = []
    best_trial: dict[str, object] | None = None

    for trial_idx, cfg in enumerate(trials, start=1):
        result = _train_one_trial(
            trial_idx=trial_idx,
            cfg=cfg,
            codec=codec,
            feat_train=feat_train,
            quat_train=quat_train,
            feat_val=feat_val,
            quat_val=quat_val,
            args=args,
            device=dev,
        )
        trial_results.append(result)

        trial_ckpt_path = out_dir / f"trial_{trial_idx:02d}_decoder.pt"
        torch.save(
            {
                "model_state_dict": result["best_state_dict"],
                "trial_config": result["config"],
                "best_epoch": result["best_epoch"],
                "best_val_mis_mean_deg": result["best_val_mis_mean_deg"],
                "best_val_loss": result["best_val_loss"],
                "constraint_metrics": result["constraint_metrics"],
                "feature_dim": int(codec.feature_dim),
                "crystal": str(args.crystal),
                "d6_convention": str(args.d6_convention),
                "target_irreps": str(args.target_irreps),
                "decoder_type": "irreps_to_fz_passive_quat",
                "seed": int(args.seed),
            },
            trial_ckpt_path,
        )
        print(f"[trial {trial_idx:02d}] saved {trial_ckpt_path}", flush=True)

        if best_trial is None:
            best_trial = result
        else:
            cur_mis = float(result["best_val_mis_mean_deg"])
            best_mis = float(best_trial["best_val_mis_mean_deg"])
            cur_loss = float(result["best_val_loss"])
            best_loss = float(best_trial["best_val_loss"])
            if (cur_mis < best_mis - 1e-6) or (
                abs(cur_mis - best_mis) <= 1e-6 and cur_loss < best_loss - 1e-8
            ):
                best_trial = result

    if best_trial is None:
        raise RuntimeError("No trials were executed.")

    best_idx = int(best_trial["trial_index"])
    best_ckpt_path = out_dir / "best_decoder.pt"
    torch.save(
        {
            "model_state_dict": best_trial["best_state_dict"],
            "trial_config": best_trial["config"],
            "best_epoch": best_trial["best_epoch"],
            "best_val_mis_mean_deg": best_trial["best_val_mis_mean_deg"],
            "best_val_loss": best_trial["best_val_loss"],
            "constraint_metrics": best_trial["constraint_metrics"],
            "feature_dim": int(codec.feature_dim),
            "crystal": str(args.crystal),
            "d6_convention": str(args.d6_convention),
            "target_irreps": str(args.target_irreps),
            "decoder_type": "irreps_to_fz_passive_quat",
            "seed": int(args.seed),
        },
        best_ckpt_path,
    )

    summary_trials: list[dict[str, object]] = []
    for tr in trial_results:
        summary_trials.append(
            {
                "trial_index": tr["trial_index"],
                "config": tr["config"],
                "best_epoch": tr["best_epoch"],
                "best_val_mis_mean_deg": tr["best_val_mis_mean_deg"],
                "best_val_loss": tr["best_val_loss"],
                "elapsed_sec": tr["elapsed_sec"],
                "constraint_metrics": tr["constraint_metrics"],
            }
        )
    summary = {
        "args": vars(args),
        "feature_dim": int(codec.feature_dim),
        "num_trials": len(summary_trials),
        "best_trial_index": best_idx,
        "best_checkpoint": str(best_ckpt_path),
        "best_trial": {
            "trial_index": best_idx,
            "config": best_trial["config"],
            "best_epoch": best_trial["best_epoch"],
            "best_val_mis_mean_deg": best_trial["best_val_mis_mean_deg"],
            "best_val_loss": best_trial["best_val_loss"],
            "constraint_metrics": best_trial["constraint_metrics"],
        },
        "trials": summary_trials,
    }
    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] best trial={best_idx}", flush=True)
    print(f"[done] best checkpoint: {best_ckpt_path}", flush=True)
    print(f"[done] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
