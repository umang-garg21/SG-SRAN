"""
Train QRBSA_1D on a 1D quaternion SR dataset.

Input:  passive dataset quaternions in scalar-first layout (B, 4, H, W) = (w x y z)
Model:  active quaternions internally, scalar-last at the QRBSA boundary
Output: passive scalar-first quaternions for visualization / export.

Uses QuaternionDataset + build_dataloader for data and
visualization/visualize_sr_results for IPF plots.
No irreps — pure quaternion I/O.

Checkpoint / output layout matches train_iso_embedding_sr_attn.py:
  checkpoints/
    best_model.pt
    last_checkpoint.pt
    epoch_XXXX.pt
    history.json
  visualizations/
    loss_curves.png
    epoch_XXXX/lr_sr_hr_ipf.png
    final/lr_sr_hr_ipf.png
  logs/
    run_config.json
    train.log
  runs/          (TensorBoard)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Q-RBSA model + loss ───────────────────────────────────────────────────────
QRBSA_ROOT = ROOT / "Q-RBSA"
sys.path.insert(0, str(QRBSA_ROOT))
sys.path.insert(0, str(QRBSA_ROOT / "model"))

from qrbsa_1d import QRBSA_1D                                    # noqa: E402
from mat_sci_torch_quats.losses import Loss                       # noqa: E402
from mat_sci_torch_quats.symmetries import fcc_syms, hcp_syms    # noqa: E402

from training.data_loading import build_dataloader                # noqa: E402
from utils.symmetry_utils import resolve_symmetry                 # noqa: E402
from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QRBSA_1D — 1D quaternion SR, no irreps")
    p.add_argument("--exp_dir", required=True, help="Experiment dir with config.json")
    p.add_argument("--config", default="config.json")
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--resume", action="store_true", help="Resume from last_checkpoint.pt")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> QRBSA_1D:
    return QRBSA_1D(SimpleNamespace(
        n_colors=4,
        n_resblocks=cfg.get("n_resblocks", 16),
        n_feats=cfg.get("n_feats", 64),
        scale=cfg.get("scale", 4),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Loss  (scalar-first, bypass ActAndLoss convention flip)
# ─────────────────────────────────────────────────────────────────────────────

def build_loss(symmetry_str: str) -> Loss:
    sym_lc = symmetry_str.lower()
    if sym_lc in ("oh", "m-3m", "m3m", "fcc", "cubic"):
        syms = fcc_syms
    elif sym_lc in ("d6h", "6/mmm", "hcp", "hex"):
        syms = hcp_syms
    else:
        syms = None
    return Loss(dist_func="rot_dist_approx", syms=syms)


def compute_loss(loss_fn: Loss, sr_chw: torch.Tensor, hr_chw: torch.Tensor) -> torch.Tensor:
    """Active scalar-first (w x y z) CHW -> mean misorientation loss.

    EuclidToRotApprox.__init__ calls .backward() internally on every forward
    pass, which fails inside torch.no_grad(). torch.enable_grad() overrides
    no_grad() so the init works; during validation the result is only used
    for .item() — no model backward is called.
    """
    sr = sr_chw.permute(0, 2, 3, 1).contiguous()   # (B, H, W, 4)
    hr = hr_chw.permute(0, 2, 3, 1).contiguous()
    with torch.enable_grad():
        return loss_fn(sr, hr).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_chw(q: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 4) or (B, 4, H, W) -> (B, 4, H, W)."""
    if q.dim() == 4 and q.shape[-1] == 4:
        return q.permute(0, 3, 1, 2).contiguous()
    return q.contiguous()


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(q, p=2, dim=1)


def conjugate_scalar_first_chw(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for scalar-first channel-first tensors."""
    q_out = q.clone()
    q_out[:, 1:, ...] *= -1
    return q_out


def to_scalar_last(q: torch.Tensor) -> torch.Tensor:
    """Reorder only: scalar-first (w,x,y,z) -> scalar-last (x,y,z,w) for QRBSA."""
    return torch.cat([q[:, 1:], q[:, :1]], dim=1)


def to_scalar_first(q: torch.Tensor) -> torch.Tensor:
    """Reorder only: scalar-last (x,y,z,w) -> scalar-first (w,x,y,z) for loss/viz."""
    return torch.cat([q[:, -1:], q[:, :-1]], dim=1)


def to_hwc_numpy(q: torch.Tensor) -> np.ndarray:
    q = q.detach().cpu().float()
    if q.dim() == 3 and q.shape[0] == 4:
        return q.permute(1, 2, 0).numpy()
    return q.numpy()


def forward_qrbsa_active_from_passive(model, lr_passive_chw: torch.Tensor) -> torch.Tensor:
    """Run QRBSA on passive scalar-first input and return active scalar-first SR."""
    lr_active_chw = conjugate_scalar_first_chw(lr_passive_chw)
    sr_active_xyzw = model(to_scalar_last(lr_active_chw))
    return normalize_quat(to_scalar_first(sr_active_xyzw))


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler  (cosine with linear warmup)
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, epochs: int, warmup_epochs: int, min_lr: float, base_lr: float):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-8, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (same keys as train_iso_embedding_sr_attn.py)
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(path: Path, *, epoch: int, model, optimizer, scheduler,
                     best_val_loss: float, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": float(best_val_loss),
        "history": history,
    }, path)


def _load_checkpoint(path: Path, *, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch  = int(ckpt.get("epoch", -1)) + 1
    best_val     = float(ckpt.get("best_val_loss", float("inf")))
    history      = ckpt.get("history", {"train": [], "val": [], "lr": []})
    return start_epoch, best_val, history


# ─────────────────────────────────────────────────────────────────────────────
# Loss curve plot  (same as train_iso_embedding_sr_attn.py)
# ─────────────────────────────────────────────────────────────────────────────

def _save_loss_plot(path: Path, history: dict, exp_name: str) -> None:
    try:
        train  = history.get("train", [])
        val    = history.get("val", [])
        lr_log = history.get("lr", [])
        epochs = list(range(1, len(train) + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        fig.suptitle(exp_name, fontsize=11)

        ax1.plot(epochs, train, label="train", linewidth=1.5)
        ax1.plot(epochs, val,   label="val",   linewidth=1.5, linestyle="--")
        ax1.set_ylabel("Misorientation Loss")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, lr_log[:len(epochs)], color="tab:orange", linewidth=1.5)
        ax2.set_ylabel("Learning Rate"); ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[warning] Could not save loss plot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _render_ipf(model, loader, sym_class, out_png: Path) -> None:
    model.eval()
    try:
        batch = next(iter(loader))
        lr_q, hr_q = batch[0][:1], batch[1][:1]
        device = next(model.parameters()).device
        lr_passive_chw = to_chw(lr_q).to(device)
        hr_passive_chw = to_chw(hr_q).to(device)
        with torch.no_grad():
            sr_active_chw = forward_qrbsa_active_from_passive(model, lr_passive_chw)
            sr_passive_chw = conjugate_scalar_first_chw(sr_active_chw)

        sr_np = to_hwc_numpy(sr_passive_chw[0])
        hr_np = to_hwc_numpy(hr_passive_chw[0])
        lr_np = to_hwc_numpy(lr_passive_chw[0])

        out_png.parent.mkdir(parents=True, exist_ok=True)
        render_sr_hr_lr_side_by_side(
            sr_q_arr=sr_np, hr_q_arr=hr_np, lr_q_arr=lr_np,
            sym_class=sym_class,
            out_png=str(out_png),
            ref_dir="ALL", include_key=True,
            overwrite=True, format_input=True, dpi=300,
        )
        print(f"Saved LR/SR/HR IPF: {out_png}")
    except Exception as e:
        print(f"[warning] Visualization failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Train / val loops
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, loss_fn, optimizer, device, train: bool) -> float:
    model.train(train)
    total, count = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            lr_passive_chw = to_chw(batch[0]).to(device)
            hr_passive_chw = to_chw(batch[1]).to(device)
            sr_active_chw = forward_qrbsa_active_from_passive(model, lr_passive_chw)
            hr_active_chw = conjugate_scalar_first_chw(hr_passive_chw)
            loss = compute_loss(loss_fn, sr_active_chw, hr_active_chw)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            count += 1
    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.exp_dir)
    with open(exp_dir / args.config) as f:
        cfg = json.load(f)

    dataset_root  = cfg["dataset_root"]
    epochs        = cfg.get("epochs", 200)
    batch_size    = cfg.get("batch_size", 8)
    base_lr       = cfg.get("lr", 2e-4)
    weight_decay  = cfg.get("weight_decay", 1e-4)
    num_workers   = cfg.get("num_workers", 4)
    seed          = cfg.get("seed", 42)
    save_every    = cfg.get("save_every", 5)
    viz_every     = cfg.get("viz_every", save_every)
    warmup_epochs = cfg.get("warmup_epochs", 3)
    min_lr        = cfg.get("min_lr", 1e-6)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── directories ───────────────────────────────────────────────────────────
    ckpt_dir = exp_dir / "checkpoints"
    viz_dir  = exp_dir / "visualizations"
    log_dir  = exp_dir / "logs"
    for d in (ckpt_dir, viz_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    loaders = {}
    for split, shuf in [("Train", True), ("Val", False), ("Test", False)]:
        try:
            loaders[split] = build_dataloader(
                dataset_root, split=split,
                batch_size=batch_size if shuf else 1,
                shuffle=shuf, num_workers=num_workers, seed=seed,
            )
        except Exception as e:
            print(f"[warning] {split} split unavailable ({e}); skipping.")

    # ── symmetry ──────────────────────────────────────────────────────────────
    with open(os.path.join(dataset_root, "dataset_info.json")) as f:
        dataset_info = json.load(f)
    symmetry_str = dataset_info.get("symmetry", "Oh")
    sym_class    = resolve_symmetry(symmetry_str)
    loss_fn      = build_loss(symmetry_str)

    # ── model / optim / scheduler ─────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs, min_lr, base_lr)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"QRBSA_1D: {total_params:,} params ({trainable_params:,} trainable)")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=exp_dir / "runs")
    except Exception:
        pass

    # ── checkpoint paths ──────────────────────────────────────────────────────
    best_ckpt = ckpt_dir / "best_model.pt"
    last_ckpt = ckpt_dir / "last_checkpoint.pt"
    hist_path = ckpt_dir / "history.json"

    start_epoch   = 0
    best_val_loss = float("inf")
    history       = {"train": [], "val": [], "lr": []}

    if args.resume and last_ckpt.exists():
        start_epoch, best_val_loss, history = _load_checkpoint(
            last_ckpt, model=model, optimizer=optimizer,
            scheduler=scheduler, device=device,
        )
        print(f"Resumed from epoch {start_epoch}  (best_val={best_val_loss:.6e})")

    # ── save resolved config ──────────────────────────────────────────────────
    with open(log_dir / "run_config.json", "w") as f:
        json.dump({**cfg, "symmetry": symmetry_str, "device": str(device)}, f, indent=2)

    # ── logging ───────────────────────────────────────────────────────────────
    log_file = open(log_dir / "train.log", "a")
    def log(msg):
        print(msg); log_file.write(msg + "\n"); log_file.flush()

    log(f"QRBSA_1D training: {epochs} epochs | device={device} | symmetry={symmetry_str}")
    log(f"Dataset: {dataset_root}")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        epoch_1 = epoch + 1   # 1-based for display and paths

        train_loss = _run_epoch(
            model, tqdm(loaders["Train"], desc=f"[{epoch_1}/{epochs}] train", leave=False),
            loss_fn, optimizer, device, train=True,
        )
        val_loss = _run_epoch(
            model, loaders.get("Val", loaders["Train"]),
            loss_fn, None, device, train=False,
        )
        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        history["lr"].append(current_lr)

        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch_1)
            writer.add_scalar("Loss/Val",   val_loss,   epoch_1)
            writer.add_scalar("LR",         current_lr, epoch_1)

        log(f"Epoch {epoch_1:04d}/{epochs} | train={train_loss:.6e} val={val_loss:.6e} lr={current_lr:.2e}")

        # always save last checkpoint
        _save_checkpoint(last_ckpt, epoch=epoch, model=model, optimizer=optimizer,
                         scheduler=scheduler, best_val_loss=best_val_loss, history=history)

        # best model
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            _save_checkpoint(best_ckpt, epoch=epoch, model=model, optimizer=optimizer,
                             scheduler=scheduler, best_val_loss=best_val_loss, history=history)
            log(f"  -> new best model saved (val={best_val_loss:.6e})")

        # periodic epoch checkpoint
        if save_every > 0 and (epoch_1 % save_every == 0):
            _save_checkpoint(ckpt_dir / f"epoch_{epoch_1:04d}.pt",
                             epoch=epoch, model=model, optimizer=optimizer,
                             scheduler=scheduler, best_val_loss=best_val_loss, history=history)

        # history + loss curve (every epoch, same as existing trainer)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        _save_loss_plot(viz_dir / "loss_curves.png", history, exp_dir.name)

        # IPF visualization
        if viz_every > 0 and (epoch_1 % viz_every == 0):
            _render_ipf(model, loaders.get("Val", loaders["Train"]),
                        sym_class, viz_dir / f"epoch_{epoch_1:04d}" / "lr_sr_hr_ipf.png")

    # ── final visualization on test set ───────────────────────────────────────
    _render_ipf(model, loaders.get("Test", loaders.get("Val", loaders["Train"])),
                sym_class, viz_dir / "final" / "lr_sr_hr_ipf.png")

    if writer is not None:
        writer.close()
    log_file.close()

    print(f"Training complete. Best val loss: {best_val_loss:.6e}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
