#!/usr/bin/env python3
"""
Train a one-shot A1 anchor+residual decoder (no iterative refinement at inference).

Target mapping:
  passive q in FZ -> A1 irreps features -> passive q in FZ
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

from models.a1_oneshot_anchor_decoder import A1OneShotAnchorDecoder
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
    topk: int
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
        feat = codec.encode_a1(q_fz, normalize_input=False)

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
            "Lookup-table dataset requires an optimizing decoder with table buffers."
        )
    feat = dec.table_feat.detach().cpu().contiguous()
    quat = dec.table_quats.detach().cpu().contiguous()
    if feat.shape[0] != quat.shape[0]:
        raise RuntimeError(
            f"Lookup table mismatch: feat={tuple(feat.shape)} quat={tuple(quat.shape)}"
        )
    print(f"[dataset] loaded full lookup table rows={int(feat.shape[0])}", flush=True)
    return feat, quat


def _top2_feature_margin(feat_err: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vals = torch.topk(feat_err, k=min(2, int(feat_err.shape[1])), largest=False, dim=1).values
    best = vals[:, 0]
    if vals.shape[1] > 1:
        second = vals[:, 1]
        margin = second - best
    else:
        second = torch.full_like(best, float("inf"))
        margin = torch.full_like(best, float("inf"))
    return margin, best, second


@torch.no_grad()
def _build_teacher_quats(
    teacher_codec: LocalIsoCodecModel,
    feat: torch.Tensor,
    batch_size: int,
    tag: str,
) -> torch.Tensor:
    n = int(feat.shape[0])
    batch_size = max(1, int(batch_size))
    out_chunks: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = feat[start:end].to(teacher_codec.device, non_blocking=True)
        qb = teacher_codec.decode(xb, reduce_fz=True)
        out_chunks.append(qb.detach().cpu())
        done = end
        if done % (5 * batch_size) == 0 or done == n:
            print(f"[teacher:{tag}] built {done}/{n}", flush=True)
    q = torch.cat(out_chunks, dim=0)
    return _normalize_quaternions(q)


def _make_trials(args: argparse.Namespace) -> list[TrialConfig]:
    topks = _parse_int_list(args.topks)
    hidden_dims = _parse_int_list(args.hidden_dims)
    num_layers = _parse_int_list(args.num_layers)
    dropouts = _parse_float_list(args.dropouts)
    residual_scale_inits = _parse_float_list(args.residual_scale_inits)
    lrs = _parse_float_list(args.lrs)

    trials = [
        TrialConfig(
            topk=tk,
            hidden_dim=h,
            num_layers=nl,
            dropout=d,
            residual_scale_init=rs,
            lr=lr,
            weight_decay=float(args.weight_decay),
        )
        for tk, h, nl, d, rs, lr in itertools.product(
            topks,
            hidden_dims,
            num_layers,
            dropouts,
            residual_scale_inits,
            lrs,
        )
    ]
    rng = random.Random(int(args.seed))
    rng.shuffle(trials)
    max_trials = min(len(trials), max(1, int(args.max_trials)))
    return trials[:max_trials]


def _evaluate_constraints(
    model: A1OneShotAnchorDecoder,
    feat: torch.Tensor,
    batch_size: int,
    max_eval: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    n = int(min(max_eval, int(feat.shape[0])))
    feat_eval = feat[:n]
    q_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            qb = model(feat_eval[start:end].to(device, non_blocking=True))
            q_chunks.append(qb.detach())
    q = torch.cat(q_chunks, dim=0)
    q = _normalize_quaternions(q)

    scalar_violation = float((q[:, 0] < -1e-7).float().mean().item())
    q_fz = model._reduce_to_fz(q)
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
    quat_teacher_train: torch.Tensor | None,
    quat_teacher_val: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    print(f"[trial {trial_idx:02d}] cfg={cfg}", flush=True)
    torch.manual_seed(int(args.seed) + trial_idx)

    model = A1OneShotAnchorDecoder(
        encoder=codec.encoder,
        topk=int(cfg.topk),
        hidden_dim=int(cfg.hidden_dim),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
        residual_scale_init=float(cfg.residual_scale_init),
        cubochoric_resolution=int(args.cubochoric_resolution),
        method=str(args.cubochoric_method),
        max_table_rows=None,
        table_cache_dir=args.table_cache_dir,
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
    has_teacher = quat_teacher_train is not None and quat_teacher_val is not None
    train_n = int(feat_train.shape[0])
    sample_hardness = torch.ones(train_n, dtype=torch.float32)
    sample_floor = 1e-6
    hard_sampling_gen = torch.Generator(device="cpu")
    hard_sampling_gen.manual_seed(int(args.seed) + 9173 + int(trial_idx))

    if has_teacher:
        val_ds = TensorDataset(feat_val, quat_val, quat_teacher_val)
    else:
        val_ds = TensorDataset(feat_val, quat_val)
    val_loader = DataLoader(
        val_ds,
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
        hard_warmup = int(args.hard_mining_warmup_epochs)
        if bool(args.hard_mining_enabled) and epoch > max(0, hard_warmup):
            norm_h = sample_hardness / sample_hardness.mean().clamp_min(sample_floor)
            norm_h = norm_h.clamp_min(sample_floor).clamp_max(20.0)
            sample_w = 1.0 + float(args.hard_mining_alpha) * norm_h
            probs = sample_w / sample_w.sum().clamp_min(sample_floor)
            epoch_indices = torch.multinomial(
                probs,
                num_samples=train_n,
                replacement=True,
                generator=hard_sampling_gen,
            )
        else:
            epoch_indices = torch.randperm(train_n, generator=hard_sampling_gen)

        for start in range(0, train_n, int(args.batch_size)):
            end = min(start + int(args.batch_size), train_n)
            idx = epoch_indices[start:end]
            xb = feat_train[idx].to(device, non_blocking=True)
            qb = quat_train[idx].to(device, non_blocking=True)
            qt = quat_teacher_train[idx].to(device, non_blocking=True) if has_teacher else None
            optimizer.zero_grad(set_to_none=True)

            out = model.forward_candidates(xb)
            q_cand = out["q_candidates"]  # [B,k,4]
            feat_err = out["feat_error"]  # [B,k]
            best_idx = out["best_idx"]  # [B]
            rotvec = out["rotvec"]  # [B,k,3]

            bsz = int(xb.shape[0])
            b = torch.arange(bsz, device=device)
            q_sel = q_cand[b, best_idx]
            feat_sel = feat_err[b, best_idx]
            sep_margin, _, _ = _top2_feature_margin(feat_err)
            sep_loss = torch.relu(float(args.separation_margin) - sep_margin).mean()

            qloss_sel = _quat_dot_loss(q_sel, qb).mean()
            qloss_all = _quat_dot_loss(q_cand, qb.unsqueeze(1))
            qloss_oracle = qloss_all.min(dim=1).values.mean()
            resid_l2 = (rotvec * rotvec).mean()

            loss = (
                float(args.gt_target_weight) * qloss_sel
                + float(args.feature_loss_weight) * feat_sel.mean()
                + float(args.gt_oracle_loss_weight) * qloss_oracle
                + float(args.residual_l2_weight) * resid_l2
                + float(args.separation_loss_weight) * sep_loss
            )
            if qt is not None and float(args.teacher_target_weight) > 0.0:
                qloss_sel_teacher = _quat_dot_loss(q_sel, qt).mean()
                loss = loss + float(args.teacher_target_weight) * qloss_sel_teacher
                if float(args.teacher_oracle_loss_weight) > 0.0:
                    qloss_teacher_all = _quat_dot_loss(q_cand, qt.unsqueeze(1))
                    qloss_teacher_oracle = qloss_teacher_all.min(dim=1).values.mean()
                    loss = loss + float(args.teacher_oracle_loss_weight) * qloss_teacher_oracle
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            train_loss_sum += float(loss.item()) * bsz
            train_count += bsz
            if bool(args.hard_mining_enabled):
                with torch.no_grad():
                    qloss_sample = _quat_dot_loss(q_sel.detach(), qb.detach()).detach()
                    hard_score = qloss_sample + float(args.hard_mining_margin_weight) * torch.relu(
                        float(args.hard_mining_margin_target) - sep_margin.detach()
                    )
                    idx_cpu = idx.detach().cpu()
                    old = sample_hardness[idx_cpu]
                    sample_hardness[idx_cpu] = (
                        float(args.hard_mining_momentum) * old
                        + (1.0 - float(args.hard_mining_momentum)) * hard_score.detach().cpu()
                    ).clamp_min(sample_floor)

        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        mis_chunks: list[torch.Tensor] = []
        teacher_mis_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                if has_teacher:
                    xb, qb, qt = batch
                    qt = qt.to(device, non_blocking=True)
                else:
                    xb, qb = batch
                    qt = None
                xb = xb.to(device, non_blocking=True)
                qb = qb.to(device, non_blocking=True)
                out = model.forward_candidates(xb)
                q_cand = out["q_candidates"]
                feat_err = out["feat_error"]
                best_idx = out["best_idx"]
                rotvec = out["rotvec"]

                bsz = int(xb.shape[0])
                b = torch.arange(bsz, device=device)
                q_sel = q_cand[b, best_idx]
                feat_sel = feat_err[b, best_idx]
                sep_margin, _, _ = _top2_feature_margin(feat_err)
                sep_loss = torch.relu(float(args.separation_margin) - sep_margin).mean()
                qloss_sel = _quat_dot_loss(q_sel, qb).mean()
                qloss_all = _quat_dot_loss(q_cand, qb.unsqueeze(1))
                qloss_oracle = qloss_all.min(dim=1).values.mean()
                resid_l2 = (rotvec * rotvec).mean()

                vloss = (
                    float(args.gt_target_weight) * qloss_sel
                    + float(args.feature_loss_weight) * feat_sel.mean()
                    + float(args.gt_oracle_loss_weight) * qloss_oracle
                    + float(args.residual_l2_weight) * resid_l2
                    + float(args.separation_loss_weight) * sep_loss
                )
                if qt is not None and float(args.teacher_target_weight) > 0.0:
                    qloss_sel_teacher = _quat_dot_loss(q_sel, qt).mean()
                    vloss = vloss + float(args.teacher_target_weight) * qloss_sel_teacher
                    if float(args.teacher_oracle_loss_weight) > 0.0:
                        qloss_teacher_all = _quat_dot_loss(q_cand, qt.unsqueeze(1))
                        qloss_teacher_oracle = qloss_teacher_all.min(dim=1).values.mean()
                        vloss = (
                            vloss
                            + float(args.teacher_oracle_loss_weight) * qloss_teacher_oracle
                        )
                val_loss_sum += float(vloss.item()) * bsz
                val_count += bsz
                mis_chunks.append(_misorientation_deg(q_sel, qb).detach().cpu())
                if qt is not None:
                    teacher_mis_chunks.append(_misorientation_deg(q_sel, qt).detach().cpu())

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
        val_teacher_mis_mean = (
            float(torch.cat(teacher_mis_chunks, dim=0).mean().item())
            if teacher_mis_chunks
            else float("nan")
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mis_mean_deg": val_mis_mean,
                "val_mis_p95_deg": val_mis_p95,
                "val_mis_p99_deg": val_mis_p99,
                "val_mis_max_deg": val_mis_max,
                "val_teacher_mis_mean_deg": val_teacher_mis_mean,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        print(
            f"[trial {trial_idx:02d}] epoch {epoch:03d}/{int(args.epochs)} "
            f"train_loss={train_loss:.6e} val_loss={val_loss:.6e} "
            f"val_mis_mean={val_mis_mean:.4f}deg val_mis_p95={val_mis_p95:.4f}deg "
            f"val_mis_p99={val_mis_p99:.4f}deg "
            f"val_teacher_mis_mean={val_teacher_mis_mean:.4f}deg",
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
        description="Train one-shot A1 anchor+residual decoder (no iterative optimization)."
    )
    p.add_argument("--out_dir", type=str, default="out/a1_oneshot_anchor_decoder")
    p.add_argument("--crystal", type=str, default="fcc", choices=["fcc", "hcp"])
    p.add_argument("--d6_convention", type=str, default="z_axis")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--dataset_source",
        type=str,
        default="random",
        choices=["random", "lookup"],
        help="Train on random FZ samples or full lookup-table coverage.",
    )

    p.add_argument("--train_samples", type=int, default=360000)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--data_chunk_size", type=int, default=65536)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--constraint_eval_samples", type=int, default=20000)

    p.add_argument("--feature_loss_weight", type=float, default=0.25)
    p.add_argument("--gt_target_weight", type=float, default=1.0)
    p.add_argument("--gt_oracle_loss_weight", type=float, default=0.5)
    p.add_argument("--teacher_target_weight", type=float, default=0.0)
    p.add_argument("--teacher_oracle_loss_weight", type=float, default=0.0)
    p.add_argument("--residual_l2_weight", type=float, default=1e-4)
    p.add_argument("--separation_loss_weight", type=float, default=0.25)
    p.add_argument("--separation_margin", type=float, default=2e-6)
    p.add_argument("--hard_mining_enabled", action="store_true")
    p.add_argument("--hard_mining_alpha", type=float, default=4.0)
    p.add_argument("--hard_mining_momentum", type=float, default=0.9)
    p.add_argument("--hard_mining_warmup_epochs", type=int, default=1)
    p.add_argument("--hard_mining_margin_target", type=float, default=2e-6)
    p.add_argument("--hard_mining_margin_weight", type=float, default=1.0)

    p.add_argument("--topks", type=str, default="8,16")
    p.add_argument("--hidden_dims", type=str, default="256,384")
    p.add_argument("--num_layers", type=str, default="3,4")
    p.add_argument("--dropouts", type=str, default="0.0")
    p.add_argument("--residual_scale_inits", type=str, default="0.05")
    p.add_argument("--lrs", type=str, default="0.0007,0.0005")
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--max_trials", type=int, default=6)

    p.add_argument("--cubochoric_resolution", type=int, default=1)
    p.add_argument("--cubochoric_method", type=str, default="cubochoric")
    p.add_argument(
        "--max_table_rows",
        type=int,
        default=None,
        help="Must remain unset/None for one-shot decoder (full lookup table required).",
    )
    p.add_argument("--table_cache_dir", type=str, default="out/decoder_lookup_tables")
    p.add_argument(
        "--teacher_backend",
        type=str,
        default="none",
        choices=["none", "lie_refine", "optimizing"],
        help="Optional teacher decoder used to generate distillation quaternion targets.",
    )
    p.add_argument("--teacher_decode_batch_size", type=int, default=4096)
    p.add_argument("--teacher_steps", type=int, default=25)
    p.add_argument("--teacher_lr", type=float, default=0.05)
    p.add_argument("--teacher_lie_optimizer", type=str, default="lbfgs")
    p.add_argument("--teacher_lie_eps", type=float, default=1e-4)
    p.add_argument("--teacher_lie_l2_reg", type=float, default=1e-6)
    p.add_argument("--teacher_lie_max_init_angle_deg", type=float, default=89.0)
    p.add_argument("--teacher_lie_lbfgs_history_size", type=int, default=10)
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
    if args.max_table_rows is not None:
        raise ValueError(
            "--max_table_rows must be omitted/None for one-shot decoding. "
            "Full lookup table coverage is required."
        )

    # We use an optimizing codec here for feature encoding and (optionally)
    # full lookup-table supervision data.
    codec = LocalIsoCodecModel(
        crystal=str(args.crystal),
        d6_convention=str(args.d6_convention),
        device=dev,
        target_irreps="a1",
        decoder_backend="optimizing",
        decoder_cubochoric_resolution=1,
        decoder_num_starts=1,
        decoder_steps=0,
        decoder_max_table_rows=None,
        decoder_table_cache_dir=str(args.table_cache_dir),
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

    quat_teacher_train: torch.Tensor | None = None
    quat_teacher_val: torch.Tensor | None = None
    teacher_backend = str(args.teacher_backend).lower().strip()
    if teacher_backend != "none":
        teacher_num_starts = _parse_int_list(args.topks)[0]
        print(
            f"[teacher] building distillation targets with backend={teacher_backend}",
            flush=True,
        )
        teacher_codec = LocalIsoCodecModel(
            crystal=str(args.crystal),
            d6_convention=str(args.d6_convention),
            device=dev,
            target_irreps="a1",
            decoder_backend=teacher_backend,
            decoder_cubochoric_resolution=int(args.cubochoric_resolution),
            decoder_method=str(args.cubochoric_method),
            decoder_num_starts=max(1, int(teacher_num_starts)),
            decoder_steps=int(args.teacher_steps),
            decoder_lr=float(args.teacher_lr),
            decoder_max_table_rows=None,
            decoder_table_cache_dir=str(args.table_cache_dir),
            lie_optimizer=str(args.teacher_lie_optimizer),
            lie_eps=float(args.teacher_lie_eps),
            lie_l2_reg=float(args.teacher_lie_l2_reg),
            lie_max_init_angle_deg=float(args.teacher_lie_max_init_angle_deg),
            lie_lbfgs_history_size=int(args.teacher_lie_lbfgs_history_size),
        ).eval()
        for p in teacher_codec.parameters():
            p.requires_grad_(False)

        quat_teacher_train = _build_teacher_quats(
            teacher_codec=teacher_codec,
            feat=feat_train,
            batch_size=int(args.teacher_decode_batch_size),
            tag="train",
        )
        quat_teacher_val = _build_teacher_quats(
            teacher_codec=teacher_codec,
            feat=feat_val,
            batch_size=int(args.teacher_decode_batch_size),
            tag="val",
        )
        teacher_mis = _misorientation_deg(quat_teacher_val, quat_val)
        print(
            f"[teacher] val mean mis={float(teacher_mis.mean().item()):.4f}deg "
            f"p95={float(torch.quantile(teacher_mis, torch.tensor(0.95)).item()):.4f}deg",
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
            quat_teacher_train=quat_teacher_train,
            quat_teacher_val=quat_teacher_val,
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
                "target_irreps": "a1",
                "decoder_type": "a1_oneshot_anchor",
                "cubochoric_resolution": int(args.cubochoric_resolution),
                "cubochoric_method": str(args.cubochoric_method),
                "max_table_rows": args.max_table_rows,
                "teacher_backend": str(args.teacher_backend),
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
            "target_irreps": "a1",
            "decoder_type": "a1_oneshot_anchor",
            "cubochoric_resolution": int(args.cubochoric_resolution),
            "cubochoric_method": str(args.cubochoric_method),
            "max_table_rows": args.max_table_rows,
            "teacher_backend": str(args.teacher_backend),
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
        "teacher_backend": str(args.teacher_backend),
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
