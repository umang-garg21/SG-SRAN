#!/usr/bin/env python3
"""
Train a learnable A1 decoder that always outputs:
  1) unit quaternions with positive scalar part
  2) quaternions reduced to the crystal fundamental zone (FZ)

This script performs an iterative hyperparameter sweep and saves the best model.
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.SR_double_conv_SRattn_a1 import LearnableA1QuaternionDecoder
from models.local_iso_codec_model import LocalIsoCodecModel


def _normalize_quaternions(quats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.norm(quats, dim=-1, keepdim=True).clamp_min(eps)
    q = quats / norm
    return torch.where(q[..., :1] < 0.0, -q, q)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=1,
    )


def _quat_dot_loss(q_pred: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
    dots = (q_pred * q_tgt).sum(dim=-1).abs().clamp(max=1.0)
    return (1.0 - dots).mean()


def _misorientation_deg(q_pred: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
    dots = (q_pred * q_tgt).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dots) * (180.0 / math.pi)


class LearnableIrrepsQuaternionDecoderFZ(nn.Module):
    """
    Wrapper around LearnableA1QuaternionDecoder that additionally reduces to FZ.
    """

    def __init__(
        self,
        input_dim: int,
        sym_ops_inv: torch.Tensor,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        lift_matrix: torch.Tensor | None = None,
        lift_normalize: bool = True,
    ):
        super().__init__()
        self.feature_input_dim = int(input_dim)
        self.use_lift = lift_matrix is not None
        self.lift_normalize = bool(lift_normalize)

        if lift_matrix is not None:
            lift = torch.as_tensor(lift_matrix, dtype=torch.float32)
            if lift.ndim != 2 or int(lift.shape[0]) != self.feature_input_dim:
                raise ValueError(
                    "lift_matrix must have shape (feature_dim, lifted_dim), "
                    f"got {tuple(lift.shape)} for feature_dim={self.feature_input_dim}"
                )
            self.register_buffer("lift_matrix", lift, persistent=True)
            decoder_input_dim = int(self.feature_input_dim + int(lift.shape[1]))
        else:
            self.register_buffer("lift_matrix", torch.empty(0, 0), persistent=False)
            decoder_input_dim = self.feature_input_dim

        self.decoder = LearnableA1QuaternionDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        syms = torch.as_tensor(sym_ops_inv, dtype=torch.float32)
        if syms.ndim != 2 or syms.shape[1] != 4:
            raise ValueError(f"Expected sym_ops_inv shape (S,4), got {tuple(syms.shape)}")
        self.register_buffer("sym_ops_inv", _normalize_quaternions(syms), persistent=True)

    def reduce_to_fz(self, quats: torch.Tensor) -> torch.Tensor:
        q = _normalize_quaternions(quats)
        bsz = int(q.shape[0])
        num_syms = int(self.sym_ops_inv.shape[0])

        q_expanded = q.unsqueeze(1).expand(-1, num_syms, -1)
        syms = self.sym_ops_inv.unsqueeze(0).expand(bsz, -1, -1)
        fam = _quat_mul(syms.reshape(-1, 4), q_expanded.reshape(-1, 4)).view(bsz, num_syms, 4)
        fam = _normalize_quaternions(fam.reshape(-1, 4)).view(bsz, num_syms, 4)

        best_idx = torch.argmax(fam[..., 0].abs(), dim=1)
        batch_idx = torch.arange(bsz, device=q.device)
        q_fz = fam[batch_idx, best_idx]
        return _normalize_quaternions(q_fz)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"Expected features shape (N,C), got {tuple(features.shape)}")
        if int(features.shape[1]) != self.feature_input_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_input_dim}, got {int(features.shape[1])}"
            )

        dec_in = features
        if self.use_lift:
            lifted = dec_in @ self.lift_matrix
            if self.lift_normalize:
                lifted = lifted / lifted.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            dec_in = torch.cat([dec_in, lifted], dim=1)

        q = self.decoder(dec_in)
        return self.reduce_to_fz(q)


# Backward-compatible alias.
LearnableA1QuaternionDecoderFZ = LearnableIrrepsQuaternionDecoderFZ


@dataclass(frozen=True)
class TrialConfig:
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float


def _parse_int_list(text: str) -> list[int]:
    vals = [x.strip() for x in str(text).split(",") if x.strip() != ""]
    return [int(v) for v in vals]


def _parse_float_list(text: str) -> list[float]:
    vals = [x.strip() for x in str(text).split(",") if x.strip() != ""]
    return [float(v) for v in vals]


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


def _make_trials(args: argparse.Namespace) -> list[TrialConfig]:
    hidden_dims = _parse_int_list(args.hidden_dims)
    num_layers = _parse_int_list(args.num_layers)
    dropouts = _parse_float_list(args.dropouts)
    lrs = _parse_float_list(args.lrs)

    all_trials = [
        TrialConfig(
            hidden_dim=h,
            num_layers=nl,
            dropout=d,
            lr=lr,
            weight_decay=float(args.weight_decay),
        )
        for h, nl, d, lr in itertools.product(hidden_dims, num_layers, dropouts, lrs)
    ]
    rng = random.Random(int(args.seed))
    rng.shuffle(all_trials)
    max_trials = min(len(all_trials), max(1, int(args.max_trials)))
    return all_trials[:max_trials]


def _build_a1_lift_matrix(codec: LocalIsoCodecModel) -> torch.Tensor:
    """
    Build a fixed linear lift that approximately inverts the A1 projection:
      x_hat_raw ~= feature_a1 @ lift_matrix

    Each block contributes:
      feature_block = beta * (x_raw @ proj_in)
      x_raw_hat = feature_block @ (pinv(proj_in) / beta)
    """
    if str(codec.target_irreps).lower() != "a1":
        raise ValueError("A1 lift is only defined for target_irreps='a1'.")

    emb = codec.encoder.embedding
    blocks = list(getattr(emb, "blocks", []))
    if len(blocks) < 1:
        raise RuntimeError("Embedding has no blocks; cannot build A1 lift.")

    in_total = int(codec.feature_dim)
    raw_total = int(sum(int(b.proj_in.shape[0]) for b in blocks))
    lift = torch.zeros(in_total, raw_total, dtype=torch.float32)

    in_start = 0
    raw_start = 0
    for block in blocks:
        proj_in = torch.as_tensor(block.proj_in, dtype=torch.float32)  # [raw_dim, in_dim]
        in_dim = int(proj_in.shape[1])
        raw_dim = int(proj_in.shape[0])
        beta = float(getattr(block, "beta", 1.0))
        pinv = torch.linalg.pinv(proj_in)  # [in_dim, raw_dim]
        lift[in_start : in_start + in_dim, raw_start : raw_start + raw_dim] = pinv / max(beta, 1e-12)
        in_start += in_dim
        raw_start += raw_dim

    if in_start != in_total:
        raise RuntimeError(f"Lift assembly mismatch: built in_dim={in_start}, expected {in_total}")
    return lift


def _evaluate_fz_constraints(
    model: LearnableIrrepsQuaternionDecoderFZ,
    feat: torch.Tensor,
    max_eval: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    n = int(min(max_eval, int(feat.shape[0])))
    feat_eval = feat[:n]
    model.eval()

    q_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            qb = model(feat_eval[start:end].to(device, non_blocking=True))
            q_chunks.append(qb.detach().cpu())
    q = torch.cat(q_chunks, dim=0).to(device)
    q = _normalize_quaternions(q)

    scalar_violation = float((q[:, 0] < -1e-7).float().mean().item())
    bsz = int(q.shape[0])
    num_syms = int(model.sym_ops_inv.shape[0])
    q_expanded = q.unsqueeze(1).expand(-1, num_syms, -1)
    syms = model.sym_ops_inv.unsqueeze(0).expand(bsz, -1, -1)
    fam = _quat_mul(syms.reshape(-1, 4), q_expanded.reshape(-1, 4)).view(bsz, num_syms, 4)
    fam = _normalize_quaternions(fam.reshape(-1, 4)).view(bsz, num_syms, 4)
    w_abs = q[:, 0].abs()
    fam_max = fam[..., 0].abs().max(dim=1).values
    fz_violation = float((w_abs + 1e-7 < fam_max).float().mean().item())

    return {
        "scalar_violation_rate": scalar_violation,
        "fz_violation_rate": fz_violation,
    }


def _train_one_trial(
    trial_idx: int,
    cfg: TrialConfig,
    feat_train: torch.Tensor,
    quat_train: torch.Tensor,
    feat_val: torch.Tensor,
    quat_val: torch.Tensor,
    sym_ops_inv: torch.Tensor,
    codec: LocalIsoCodecModel,
    lift_matrix: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    print(
        f"[trial {trial_idx:02d}] cfg={cfg}",
        flush=True,
    )
    torch.manual_seed(int(args.seed) + trial_idx)

    model = LearnableIrrepsQuaternionDecoderFZ(
        input_dim=int(feat_train.shape[1]),
        sym_ops_inv=sym_ops_inv.to(device),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        lift_matrix=None if lift_matrix is None else lift_matrix.to(device),
        lift_normalize=bool(args.lift_normalize),
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
            quat_loss = _quat_dot_loss(q_pred, qb)
            feat_loss = torch.zeros((), device=device, dtype=quat_loss.dtype)
            if float(args.feature_loss_weight) > 0.0:
                feat_pred = codec.encode(q_pred, normalize_input=False)
                feat_loss = (feat_pred - xb).pow(2).mean()
            loss = quat_loss + float(args.feature_loss_weight) * feat_loss
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
        val_feat_loss_sum = 0.0
        mis_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, qb in val_loader:
                xb = xb.to(device, non_blocking=True)
                qb = qb.to(device, non_blocking=True)
                q_pred = model(xb)
                vloss = _quat_dot_loss(q_pred, qb)
                if float(args.feature_loss_weight) > 0.0:
                    vfeat = codec.encode(q_pred, normalize_input=False)
                    val_feat_loss_sum += float((vfeat - xb).pow(2).mean().item()) * int(xb.shape[0])

                val_loss_sum += float(vloss.item()) * int(xb.shape[0])
                val_count += int(xb.shape[0])
                mis_chunks.append(_misorientation_deg(q_pred, qb).detach().cpu())

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        val_feat_loss = val_feat_loss_sum / max(1, val_count)
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
                "val_feature_mse": val_feat_loss,
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
    constraints = _evaluate_fz_constraints(
        model=model,
        feat=feat_val,
        max_eval=int(args.constraint_eval_samples),
        batch_size=int(args.batch_size),
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
        description=(
            "Iteratively train and select a learnable A1 decoder with positive-scalar + "
            "fundamental-zone quaternion outputs."
        )
    )
    p.add_argument("--out_dir", type=str, default="out/learnable_local_iso_decoder")
    p.add_argument("--crystal", type=str, default="fcc", choices=["fcc", "hcp"])
    p.add_argument("--d6_convention", type=str, default="z_axis")
    p.add_argument("--target_irreps", type=str, default="a1", choices=["a1", "full"])
    p.add_argument("--device", type=str, default=None, help="cuda, cuda:0, cpu, etc.")

    p.add_argument("--train_samples", type=int, default=240000)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--data_chunk_size", type=int, default=65536)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--constraint_eval_samples", type=int, default=20000)
    p.add_argument(
        "--feature_loss_weight",
        type=float,
        default=0.0,
        help="Weight for feature-cycle MSE loss: ||encode(q_pred)-input_features||^2",
    )
    p.add_argument(
        "--inverse_mode",
        type=str,
        default="direct",
        choices=["direct", "a1_lifted"],
        help=(
            "direct: MLP on irreps features only. "
            "a1_lifted: augment input with a fixed pseudoinverse lift of A1 projection "
            "(structure-aware inverse for q_FZ -> A1 irreps -> q_FZ)."
        ),
    )
    p.add_argument(
        "--lift_normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --inverse_mode a1_lifted, normalize lifted raw tensor channels.",
    )

    p.add_argument("--hidden_dims", type=str, default="256,384,512")
    p.add_argument("--num_layers", type=str, default="3,4,5")
    p.add_argument("--dropouts", type=str, default="0.0,0.05")
    p.add_argument("--lrs", type=str, default="0.001,0.0007,0.0005")
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--max_trials", type=int, default=8)
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

    # Keep the optimizing decoder tiny because this script only needs
    # encode/reduce_to_fz from LocalIsoCodecModel.
    codec = LocalIsoCodecModel(
        crystal=str(args.crystal),
        d6_convention=str(args.d6_convention),
        device=dev,
        target_irreps=str(args.target_irreps),
        decoder_cubochoric_resolution=1,
        decoder_num_starts=1,
        decoder_steps=0,
        decoder_max_table_rows=32,
        decoder_table_cache_dir=None,
    ).eval()

    for p in codec.parameters():
        p.requires_grad_(False)

    print(
        f"[setup] feature_dim={codec.feature_dim} sym_ops={int(codec.encoder.sym_ops_inv.shape[0])}",
        flush=True,
    )
    feat_all, quat_all = _build_random_fz_dataset(
        codec=codec,
        num_samples=int(args.train_samples),
        seed=int(args.seed),
        chunk_size=int(args.data_chunk_size),
    )
    n_total = int(feat_all.shape[0])
    val_n = max(1, int(n_total * float(args.val_ratio)))
    g_split = torch.Generator(device="cpu")
    g_split.manual_seed(int(args.seed) + 12345)
    perm = torch.randperm(n_total, generator=g_split)
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

    lift_matrix: torch.Tensor | None = None
    if str(args.inverse_mode).lower() == "a1_lifted":
        if str(args.target_irreps).lower() != "a1":
            raise ValueError("--inverse_mode a1_lifted requires --target_irreps a1")
        lift_matrix = _build_a1_lift_matrix(codec).detach().cpu()
        print(
            f"[setup] inverse_mode=a1_lifted lift_shape={tuple(lift_matrix.shape)} "
            f"lift_normalize={bool(args.lift_normalize)}",
            flush=True,
        )
    else:
        print("[setup] inverse_mode=direct", flush=True)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_results: list[dict[str, object]] = []
    best_trial: dict[str, object] | None = None

    for trial_idx, cfg in enumerate(trials, start=1):
        result = _train_one_trial(
            trial_idx=trial_idx,
            cfg=cfg,
            feat_train=feat_train,
            quat_train=quat_train,
            feat_val=feat_val,
            quat_val=quat_val,
            sym_ops_inv=codec.encoder.sym_ops_inv.detach().cpu(),
            codec=codec,
            lift_matrix=lift_matrix,
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
            "seed": int(args.seed),
        },
        best_ckpt_path,
    )

    # Drop large state_dict/history fields from JSON summary for readability.
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
    print(
        "[done] decoder outputs are constrained to positive scalar + fundamental zone in forward().",
        flush=True,
    )


if __name__ == "__main__":
    main()
