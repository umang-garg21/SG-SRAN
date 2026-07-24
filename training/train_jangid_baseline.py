"""
Train Jangid/UCSB QEDSR-style baselines on local quaternion SR datasets.

The upstream Q-RBSA repository provides the official 1D QEDSR/QRBSA models used
for sparse-section EBSD super-resolution. This wrapper adapts those models to
the local IN718 dataset layout and checkpoint conventions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_QRBSA = ROOT / "third_party" / "UCSB-Q-RBSA"
LOCAL_QRBSA = ROOT / "Q-RBSA"
QRBSA_ROOT = THIRD_PARTY_QRBSA if THIRD_PARTY_QRBSA.exists() else LOCAL_QRBSA

sys.path.insert(0, str(QRBSA_ROOT))
sys.path.insert(0, str(QRBSA_ROOT / "model"))
sys.path.insert(0, str(ROOT))

from edsr_1d import EDSR as EDSR1D  # noqa: E402
from han_1d import HAN as HAN1D  # noqa: E402
from qedsr_1d import QEDSR as QEDSR1D  # noqa: E402
from qrbsa_1d import QRBSA_1D  # noqa: E402
from mat_sci_torch_quats.losses import Loss  # noqa: E402
from mat_sci_torch_quats.symmetries import fcc_syms, hcp_syms  # noqa: E402
from model import common as qrbsa_common  # noqa: E402
from model.quat_utils.Qops_with_QSN import Residual_D, Residual_SA, conv2d as QConv2d  # noqa: E402
from models.classic_sr_baselines import (  # noqa: E402
    RCAN,
    efficient_covpool,
    make_tail,
    stable_sqrtm,
)
from training.data_loading import build_dataloader  # noqa: E402
from utils.symmetry_utils import resolve_symmetry  # noqa: E402
from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side  # noqa: E402

# Load SAN through the already-imported ``model`` package so its absolute
# imports resolve to the vendored authors' source without shadowing Q-RBSA.
SAN_ROOT = ROOT / "third_party" / "SAN" / "TrainCode" / "model"
if SAN_ROOT.exists():
    import model as _model_package  # noqa: E402

    if str(SAN_ROOT) not in _model_package.__path__:
        _model_package.__path__.append(str(SAN_ROOT))
    from model.MPNCOV.python import MPNCOV as _san_mpncov  # noqa: E402
    from model.san import SAN as SANOfficial  # noqa: E402

    # The original expression x I_hat x^T is exactly centered covariance, but
    # I_hat is an HW-by-HW matrix.  This equivalent implementation is essential
    # for the long 4x1 EBSD patches and leaves SAN's result unchanged.
    _san_mpncov.CovpoolLayer = efficient_covpool
    _san_mpncov.SqrtmLayer = stable_sqrtm
else:  # pragma: no cover - exercised only when the optional vendor tree is absent
    SANOfficial = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jangid/UCSB QEDSR-style baseline")
    parser.add_argument("--exp_dir", required=True, help="Experiment directory containing config.json")
    parser.add_argument("--config", default="config.json", help="Config filename inside exp_dir")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES value, e.g. '0'")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/last_checkpoint.pt")
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="Load model weights from this checkpoint, but start a fresh optimizer/scheduler.",
    )
    parser.add_argument("--max_train_batches", type=int, default=None, help="Optional debugging cap")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Optional debugging cap")
    parser.add_argument("--skip_viz", action="store_true", help="Skip IPF visualization during training")
    return parser.parse_args()


def _scale_to_int(scale) -> int:
    if isinstance(scale, (list, tuple)):
        vals = [int(v) for v in scale]
        non_unit = [v for v in vals if v != 1]
        return int(non_unit[0] if non_unit else vals[0])
    return int(scale)


def _scale_to_pair(scale) -> tuple[int, int]:
    if isinstance(scale, (list, tuple)):
        vals = [int(v) for v in scale]
        if len(vals) != 2:
            raise ValueError(f"Expected scale pair or int, got {scale!r}")
        return vals[0], vals[1]
    s = int(scale)
    return s, s


class QuaternionUpsampler2D(nn.Sequential):
    def __init__(self, scale: int, n_feat: int, kernel_size: int = 3):
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.append(QConv2d(n_feat, 4 * n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(QConv2d(n_feat, 9 * n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
            layers.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError(f"Unsupported 2D scale factor: {scale}")
        super().__init__(*layers)


class QEDSR2D(nn.Module):
    """Quaternion-convolution EDSR with a 2D pixel-shuffle tail."""

    def __init__(self, args):
        super().__init__()
        n_resblocks = int(args.n_resblocks)
        n_feats = int(args.n_feats)
        kernel_size = 3
        scale = int(args.scale)

        self.head = nn.Sequential(
            QConv2d(args.n_colors, n_feats, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )
        self.body = nn.Sequential(
            *[
                Residual_D(n_feats, n_feats, spectral_normed=False, down_sampling=False)
                for _ in range(n_resblocks)
            ],
            QConv2d(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        )
        self.tail = nn.Sequential(
            QuaternionUpsampler2D(scale, n_feats, kernel_size=kernel_size),
            QConv2d(n_feats, args.n_colors, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        return self.tail(res)


class QRBSA2D(nn.Module):
    """Published Q-RBSA head/body with an isotropic 2D pixel-shuffle tail."""

    def __init__(self, args):
        super().__init__()
        n_resblocks = int(args.n_resblocks)
        n_feats = int(args.n_feats)
        kernel_size = 3
        scale = int(args.scale)

        self.head = nn.Sequential(
            QConv2d(args.n_colors, n_feats, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )
        self.body = nn.Sequential(
            *[Residual_SA(n_feats, n_feats) for _ in range(n_resblocks)],
            QConv2d(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        )
        self.tail = nn.Sequential(
            QuaternionUpsampler2D(scale, n_feats, kernel_size=kernel_size),
            QConv2d(n_feats, args.n_colors, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        return self.tail(res)


class EDSR2D(nn.Module):
    """Standard 2D EDSR from the UCSB utility modules, trained on quaternion channels."""

    def __init__(self, args):
        super().__init__()
        n_resblocks = int(args.n_resblocks)
        n_feats = int(args.n_feats)
        kernel_size = 3
        act = nn.ReLU(True)

        self.head = nn.Sequential(qrbsa_common.default_conv(args.n_colors, n_feats, kernel_size))
        self.body = nn.Sequential(
            *[
                qrbsa_common.ResBlock(
                    qrbsa_common.default_conv,
                    n_feats,
                    kernel_size,
                    act=act,
                    res_scale=float(args.res_scale),
                )
                for _ in range(n_resblocks)
            ],
            qrbsa_common.default_conv(n_feats, n_feats, kernel_size),
        )
        self.tail = nn.Sequential(
            qrbsa_common.Upsampler(qrbsa_common.default_conv, int(args.scale), n_feats, act=False),
            qrbsa_common.default_conv(n_feats, args.n_colors, kernel_size),
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        return self.tail(res)


def build_model(cfg: dict) -> nn.Module:
    model_type = str(cfg.get("model_type", "qedsr_1d")).lower()
    scale_pair = _scale_to_pair(cfg.get("scale", 4))
    scale_int = _scale_to_int(cfg.get("scale", 4))
    args = SimpleNamespace(
        n_colors=4,
        n_resblocks=int(cfg.get("n_resblocks", 16)),
        n_resgroups=int(cfg.get("n_resgroups", 10)),
        n_feats=int(cfg.get("n_feats", 64)),
        reduction=int(cfg.get("reduction", 16)),
        res_scale=float(cfg.get("res_scale", 1.0)),
        scale=scale_int,
        rgb_range=float(cfg.get("rgb_range", 1.0)),
    )

    if model_type == "qedsr_1d":
        if 1 not in scale_pair:
            raise ValueError("qedsr_1d expects an anisotropic scale such as [4, 1]")
        return QEDSR1D(args)
    if model_type == "qrbsa_1d":
        if 1 not in scale_pair:
            raise ValueError("qrbsa_1d expects an anisotropic scale such as [4, 1]")
        return QRBSA_1D(args)
    if model_type == "edsr_1d":
        if 1 not in scale_pair:
            raise ValueError("edsr_1d expects an anisotropic scale such as [4, 1]")
        return EDSR1D(args)
    if model_type == "qedsr_2d":
        if scale_pair[0] != scale_pair[1]:
            raise ValueError("qedsr_2d expects an isotropic scale such as [4, 4]")
        return QEDSR2D(args)
    if model_type == "qrbsa_2d":
        if scale_pair[0] != scale_pair[1]:
            raise ValueError("qrbsa_2d expects an isotropic scale such as [4, 4]")
        return QRBSA2D(args)
    if model_type == "edsr_2d":
        if scale_pair[0] != scale_pair[1]:
            raise ValueError("edsr_2d expects an isotropic scale such as [4, 4]")
        return EDSR2D(args)
    if model_type in ("rcan_1d", "rcan_2d"):
        if model_type.endswith("_1d") and 1 not in scale_pair:
            raise ValueError("rcan_1d expects an anisotropic scale such as [4, 1]")
        if model_type.endswith("_2d") and scale_pair[0] != scale_pair[1]:
            raise ValueError("rcan_2d expects an isotropic scale such as [4, 4]")
        return RCAN(
            scale_pair=scale_pair,
            n_resgroups=args.n_resgroups,
            n_resblocks=args.n_resblocks,
            n_feats=args.n_feats,
            reduction=args.reduction,
            n_colors=args.n_colors,
        )
    if model_type in ("han_1d", "han_2d"):
        if args.n_resgroups != 10:
            raise ValueError("The published HAN layer-attention fusion expects n_resgroups=10")
        if model_type.endswith("_1d") and 1 not in scale_pair:
            raise ValueError("han_1d expects an anisotropic scale such as [4, 1]")
        if model_type.endswith("_2d") and scale_pair[0] != scale_pair[1]:
            raise ValueError("han_2d expects an isotropic scale such as [4, 4]")
        model = HAN1D(args)
        model.tail = make_tail(scale_pair, args.n_feats, args.n_colors)
        return model
    if model_type in ("san_1d", "san_2d"):
        if SANOfficial is None:
            raise FileNotFoundError(f"Official SAN source is missing: {SAN_ROOT}")
        if model_type.endswith("_1d") and 1 not in scale_pair:
            raise ValueError("san_1d expects an anisotropic scale such as [4, 1]")
        if model_type.endswith("_2d") and scale_pair[0] != scale_pair[1]:
            raise ValueError("san_2d expects an isotropic scale such as [4, 4]")
        # SAN expects a sequence during construction; the task-specific tail is
        # installed immediately afterward and RGB mean shifts are inapplicable.
        san_args = SimpleNamespace(**vars(args))
        san_args.scale = [scale_int]
        model = SANOfficial(san_args)
        model.sub_mean = nn.Identity()
        model.add_mean = nn.Identity()
        model.tail = make_tail(scale_pair, args.n_feats, args.n_colors)
        return model
    raise ValueError(f"Unknown model_type={model_type!r}")


def build_loss(symmetry_str: str) -> Loss:
    sym_lc = symmetry_str.lower()
    if sym_lc in ("oh", "m-3m", "m3m", "fcc", "cubic"):
        syms = fcc_syms
    elif sym_lc in ("d6", "d6h", "6/mmm", "hcp", "hex"):
        syms = hcp_syms
    else:
        syms = None
    return Loss(dist_func="rot_dist_approx", syms=syms)


def to_chw(q: torch.Tensor) -> torch.Tensor:
    if q.dim() == 4 and q.shape[-1] == 4:
        return q.permute(0, 3, 1, 2).contiguous()
    return q.contiguous()


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return F.normalize(q, p=2, dim=1)


def conjugate_scalar_first_chw(q: torch.Tensor) -> torch.Tensor:
    q_out = q.clone()
    q_out[:, 1:, ...] *= -1
    return q_out


def to_scalar_last(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, 1:], q[:, :1]], dim=1)


def to_scalar_first(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, -1:], q[:, :-1]], dim=1)


def to_hwc_numpy(q: torch.Tensor) -> np.ndarray:
    q = q.detach().cpu().float()
    if q.dim() == 3 and q.shape[0] == 4:
        return q.permute(1, 2, 0).numpy()
    return q.numpy()


def forward_baseline_active_from_passive(model: nn.Module, lr_passive_chw: torch.Tensor) -> torch.Tensor:
    lr_active_chw = conjugate_scalar_first_chw(lr_passive_chw)
    sr_active_xyzw = model(to_scalar_last(lr_active_chw))
    return normalize_quat(to_scalar_first(sr_active_xyzw))


def compute_loss(loss_fn: Loss, sr_active_chw: torch.Tensor, hr_active_chw: torch.Tensor) -> torch.Tensor:
    sr = sr_active_chw.permute(0, 2, 3, 1).contiguous()
    hr = hr_active_chw.permute(0, 2, 3, 1).contiguous()
    with torch.enable_grad():
        return loss_fn(sr, hr).mean()


def build_scheduler(optimizer, epochs: int, warmup_epochs: int, min_lr: float, base_lr: float):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-8, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_checkpoint(path: Path, *, epoch: int, model, optimizer, scheduler, best_val_loss: float, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": float(best_val_loss),
            "history": history,
        },
        path,
    )


def _load_checkpoint(path: Path, *, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val = float(ckpt.get("best_val_loss", float("inf")))
    history = ckpt.get("history", {"train": [], "val": [], "lr": []})
    return start_epoch, best_val, history


def _load_model_init_checkpoint(path: Path, *, model, device) -> None:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)


def _save_loss_plot(path: Path, history: dict, exp_name: str) -> None:
    try:
        train = history.get("train", [])
        val = history.get("val", [])
        lr_log = history.get("lr", [])
        epochs = list(range(1, len(train) + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        fig.suptitle(exp_name, fontsize=11)
        ax1.plot(epochs, train, label="train", linewidth=1.5)
        ax1.plot(epochs, val, label="val", linewidth=1.5, linestyle="--")
        ax1.set_ylabel("Misorientation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, lr_log[: len(epochs)], color="tab:orange", linewidth=1.5)
        ax2.set_ylabel("Learning Rate")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warning] Could not save loss plot: {exc}")


def _render_ipf(model, loader, sym_class, out_png: Path) -> None:
    model.eval()
    try:
        batch = next(iter(loader))
        lr_q, hr_q = batch[0][:1], batch[1][:1]
        device = next(model.parameters()).device
        lr_passive_chw = to_chw(lr_q).to(device)
        hr_passive_chw = to_chw(hr_q).to(device)
        with torch.no_grad():
            sr_active_chw = forward_baseline_active_from_passive(model, lr_passive_chw)
            sr_passive_chw = conjugate_scalar_first_chw(sr_active_chw)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        render_sr_hr_lr_side_by_side(
            sr_q_arr=to_hwc_numpy(sr_passive_chw[0]),
            hr_q_arr=to_hwc_numpy(hr_passive_chw[0]),
            lr_q_arr=to_hwc_numpy(lr_passive_chw[0]),
            sym_class=sym_class,
            out_png=str(out_png),
            ref_dir="ALL",
            include_key=True,
            overwrite=True,
            format_input=True,
            dpi=300,
        )
        print(f"Saved LR/SR/HR IPF: {out_png}")
    except Exception as exc:
        print(f"[warning] Visualization failed: {exc}")


def _run_epoch(model, loader, loss_fn, optimizer, device, train: bool, max_batches: int | None = None) -> float:
    model.train(train)
    total = 0.0
    count = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bidx, batch in enumerate(loader):
            if max_batches is not None and bidx >= int(max_batches):
                break
            lr_passive_chw = to_chw(batch[0]).to(device=device, dtype=torch.float32, non_blocking=True)
            hr_passive_chw = to_chw(batch[1]).to(device=device, dtype=torch.float32, non_blocking=True)
            sr_active_chw = forward_baseline_active_from_passive(model, lr_passive_chw)
            hr_active_chw = conjugate_scalar_first_chw(hr_passive_chw)
            loss = compute_loss(loss_fn, sr_active_chw, hr_active_chw)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at batch {bidx} ({'train' if train else 'eval'})"
                )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


def _make_loader(cfg: dict, split: str, batch_size: int, shuffle: bool):
    take_first = cfg.get(f"{split.lower()}_take_first", cfg.get("take_first"))
    return build_dataloader(
        cfg["dataset_root"],
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(cfg.get("num_workers", 4)),
        seed=int(cfg.get("seed", 42)),
        preload=bool(cfg.get("preload", False)),
        preload_torch=bool(cfg.get("preload_torch", False)),
        persistent_workers=bool(cfg.get("persistent_workers", False)),
        prefetch_factor=int(cfg.get("prefetch_factor", 2)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        take_first=int(take_first) if take_first is not None else None,
        drop_last=bool(shuffle and cfg.get("drop_last", False)),
    )


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    exp_dir = Path(args.exp_dir)
    with open(exp_dir / args.config, "r") as f:
        cfg = json.load(f)

    epochs = int(cfg.get("epochs", 150))
    batch_size = int(cfg.get("batch_size", 2))
    eval_batch_size = int(cfg.get("eval_batch_size", 1))
    base_lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    seed = int(cfg.get("seed", 42))
    save_every = int(cfg.get("save_every", 10))
    viz_every = int(cfg.get("viz_every", save_every))
    warmup_epochs = int(cfg.get("warmup_epochs", cfg.get("scheduler", {}).get("warmup_epochs", 2)))
    min_lr = float(cfg.get("min_lr", cfg.get("scheduler", {}).get("min_lr", 1e-6)))
    val_freq = int(cfg.get("val_freq", 1))
    save_last_checkpoint = bool(cfg.get("save_last_checkpoint", True))
    save_epoch_checkpoints = bool(cfg.get("save_epoch_checkpoints", True))

    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_dir = exp_dir / "checkpoints"
    viz_dir = exp_dir / "visualizations"
    log_dir = exp_dir / "logs"
    for path in (ckpt_dir, viz_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    with open(Path(cfg["dataset_root"]) / "dataset_info.json", "r") as f:
        dataset_info = json.load(f)
    symmetry_str = dataset_info.get("symmetry", cfg.get("symmetry", cfg.get("symmetry_group", "Oh")))
    sym_class = resolve_symmetry(symmetry_str)
    loss_fn = build_loss(symmetry_str)

    loaders = {
        "Train": _make_loader(cfg, "Train", batch_size, True),
        "Val": _make_loader(cfg, "Val", eval_batch_size, False),
        "Test": _make_loader(cfg, "Test", eval_batch_size, False),
    }

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs, min_lr, base_lr)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = str(cfg.get("model_type", "qedsr_1d"))
    print(f"{model_type}: {total_params:,} params ({trainable_params:,} trainable)")

    expected_params = cfg.get("expected_trainable_params")
    if expected_params is not None and trainable_params != int(expected_params):
        raise RuntimeError(
            "Baseline parameter-count mismatch: "
            f"expected {int(expected_params):,}, instantiated {trainable_params:,}"
        )

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=exp_dir / "runs")
    except Exception:
        pass

    best_ckpt = ckpt_dir / "best_model.pt"
    last_ckpt = ckpt_dir / "last_checkpoint.pt"
    hist_path = ckpt_dir / "history.json"
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train": [], "val": [], "lr": []}

    if args.resume and last_ckpt.exists():
        start_epoch, best_val_loss, history = _load_checkpoint(
            last_ckpt, model=model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        print(f"Resumed from epoch {start_epoch} (best_val={best_val_loss:.6e})")
    elif args.init_checkpoint:
        init_checkpoint = Path(args.init_checkpoint)
        if not init_checkpoint.is_absolute():
            init_checkpoint = exp_dir / "checkpoints" / init_checkpoint
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"Initial checkpoint missing: {init_checkpoint}")
        _load_model_init_checkpoint(init_checkpoint, model=model, device=device)
        print(f"Initialized model weights from {init_checkpoint}")

    resolved_config = {
        **cfg,
        "symmetry": symmetry_str,
        "device": str(device),
        "qrbsa_root": str(QRBSA_ROOT),
    }
    with open(log_dir / "run_config.json", "w") as f:
        json.dump(resolved_config, f, indent=2)

    log_file = open(log_dir / "train.log", "a")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"{model_type} training: {epochs} epochs | device={device} | symmetry={symmetry_str}")
    log(f"Dataset: {cfg['dataset_root']}")
    log(f"Codebase: {QRBSA_ROOT}")
    log(
        "Architecture: "
        f"n_feats={cfg.get('n_feats', 64)} | "
        f"n_resgroups={cfg.get('n_resgroups', 'n/a')} | "
        f"n_resblocks={cfg.get('n_resblocks', 16)} | "
        f"trainable_params={trainable_params:,}"
    )

    for epoch in range(start_epoch, epochs):
        epoch_1 = epoch + 1
        train_loss = _run_epoch(
            model,
            tqdm(loaders["Train"], desc=f"[{epoch_1}/{epochs}] train", leave=False),
            loss_fn,
            optimizer,
            device,
            train=True,
            max_batches=args.max_train_batches,
        )
        if val_freq > 0 and epoch_1 % val_freq == 0:
            val_loss = _run_epoch(
                model,
                loaders["Val"],
                loss_fn,
                None,
                device,
                train=False,
                max_batches=args.max_val_batches,
            )
        else:
            val_loss = history["val"][-1] if history["val"] else train_loss

        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        history["lr"].append(current_lr)

        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch_1)
            writer.add_scalar("Loss/Val", val_loss, epoch_1)
            writer.add_scalar("LR", current_lr, epoch_1)

        log(f"Epoch {epoch_1:04d}/{epochs} | train={train_loss:.6e} val={val_loss:.6e} lr={current_lr:.2e}")
        if save_last_checkpoint:
            _save_checkpoint(last_ckpt, epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
                             best_val_loss=best_val_loss, history=history)

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            _save_checkpoint(best_ckpt, epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
                             best_val_loss=best_val_loss, history=history)
            log(f"  -> new best model saved (val={best_val_loss:.6e})")

        if save_epoch_checkpoints and save_every > 0 and epoch_1 % save_every == 0:
            _save_checkpoint(ckpt_dir / f"epoch_{epoch_1:04d}.pt", epoch=epoch, model=model,
                             optimizer=optimizer, scheduler=scheduler,
                             best_val_loss=best_val_loss, history=history)

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        _save_loss_plot(viz_dir / "loss_curves.png", history, exp_dir.name)

        if not args.skip_viz and viz_every > 0 and epoch_1 % viz_every == 0:
            _render_ipf(model, loaders["Val"], sym_class, viz_dir / f"epoch_{epoch_1:04d}" / "lr_sr_hr_ipf.png")

    if not args.skip_viz:
        _render_ipf(model, loaders["Test"], sym_class, viz_dir / "final" / "lr_sr_hr_ipf.png")

    if writer is not None:
        writer.close()
    log_file.close()
    print(f"Training complete. Best val loss: {best_val_loss:.6e}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
