"""
Train the Atindama et al. partial-convolution EBSD inpainting baseline.

The neural network is imported directly from the authors' repository under
third_party/Atindama-EBSD-Restoration.  This file only adapts the local
quaternion LR/HR datasets to the normalized ZXZ Euler-angle representation and
known-pixel masks expected by that implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
AUTHORS_DIR = (
    ROOT
    / "third_party"
    / "Atindama-EBSD-Restoration"
    / "EBSD-hybrid-inpainting-develop"
)
sys.path.insert(0, str(ROOT))

from training.quaternion_dataset import QuaternionDataset  # noqa: E402

EULER_SCALE = np.array([2.0 * np.pi, np.pi, 2.0 * np.pi], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Atindama partial-convolution inpainting baseline"
    )
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="Load model weights from this checkpoint, but start a fresh optimizer.",
    )
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


def _to_hwc_quaternion(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected rank-3 quaternion map, got {array.shape}")
    if array.shape[-1] == 4:
        return array
    if array.shape[0] == 4:
        return np.moveaxis(array, 0, -1)
    raise ValueError(f"Quaternion axis not found in {array.shape}")


def passive_quaternion_to_normalized_zxz(quaternion: np.ndarray) -> np.ndarray:
    """Passive scalar-first quaternion map -> authors' normalized ZXZ Euler map."""
    q = np.asarray(_to_hwc_quaternion(quaternion), dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    invalid = norm[..., 0] < 1e-12
    if np.any(invalid):
        if np.all(invalid):
            q = np.zeros_like(q)
            q[..., 0] = 1.0
            norm = np.ones_like(norm)
        else:
            from scipy import ndimage

            nearest = ndimage.distance_transform_edt(
                invalid,
                return_distances=False,
                return_indices=True,
            )
            q = q[tuple(nearest)]
            norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)

    # scipy Rotation uses active scalar-last quaternions.  The local datasets
    # store passive scalar-first quaternions, so conjugate before reordering.
    active_xyzw = np.empty_like(q)
    active_xyzw[..., :3] = -q[..., 1:]
    active_xyzw[..., 3] = q[..., 0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        euler = Rotation.from_quat(active_xyzw.reshape(-1, 4)).as_euler("ZXZ")
    euler = euler.reshape(*q.shape[:-1], 3)
    euler[..., 0] = np.mod(euler[..., 0], 2.0 * np.pi)
    euler[..., 2] = np.mod(euler[..., 2], 2.0 * np.pi)
    return (euler / EULER_SCALE).astype(np.float32)


def normalized_zxz_to_passive_quaternion(euler_normalized: np.ndarray) -> np.ndarray:
    """Authors' normalized ZXZ Euler map -> passive scalar-first quaternion map."""
    euler = np.asarray(euler_normalized, dtype=np.float64) * EULER_SCALE
    active_xyzw = Rotation.from_euler(
        "ZXZ", euler.reshape(-1, 3)
    ).as_quat().reshape(*euler.shape[:-1], 4)
    passive = np.empty_like(active_xyzw)
    passive[..., 0] = active_xyzw[..., 3]
    passive[..., 1:] = -active_xyzw[..., :3]
    passive /= np.maximum(np.linalg.norm(passive, axis=-1, keepdims=True), 1e-12)
    passive[passive[..., 0] < 0.0] *= -1.0
    return passive.astype(np.float32)


def periodic_known_mask(
    height: int, width: int, scale: int | list[int] | tuple[int, int]
) -> np.ndarray:
    """Create the HR-grid mask whose known samples exactly reproduce LR_Data."""
    if isinstance(scale, (list, tuple)):
        if len(scale) != 2:
            raise ValueError(f"Expected two scale values, got {scale}")
        scale_y, scale_x = int(scale[0]), int(scale[1])
    else:
        scale_y = scale_x = int(scale)
    if scale_y <= 0 or scale_x <= 0:
        raise ValueError(f"Invalid scale {(scale_y, scale_x)}")

    mask = np.zeros((3, int(height), int(width)), dtype=np.float32)
    mask[:, ::scale_y, ::scale_x] = 1.0
    return mask


class PeriodicEulerInpaintingDataset(Dataset):
    """Local quaternion SR pairs exposed as Euler target + periodic known mask."""

    def __init__(
        self,
        dataset_root: str,
        split: str,
        scale,
        cache_dir: str | Path | None = None,
        take_first: int | None = None,
        return_quaternions: bool = False,
    ) -> None:
        base = QuaternionDataset(
            dataset_root=dataset_root,
            split=split,
            take_first=take_first,
            preload=False,
            pin_memory=False,
        )
        self.pairs = list(base.pairs)
        self.scale = scale
        self.return_quaternions = bool(return_quaternions)
        self.cache_dir = Path(cache_dir) / split.lower() if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_euler(self, hr_path: str, hr_quat: np.ndarray) -> np.ndarray:
        cache_path = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{Path(hr_path).stem}_zxz.npy"
            if cache_path.exists():
                return np.load(cache_path)

        euler = passive_quaternion_to_normalized_zxz(hr_quat)
        if cache_path is not None:
            tmp_path = cache_path.with_suffix(".tmp.npy")
            np.save(tmp_path, euler)
            os.replace(tmp_path, cache_path)
        return euler

    def __getitem__(self, index: int):
        lr_path, hr_path = self.pairs[index]
        hr_quat = _to_hwc_quaternion(np.load(hr_path)).astype(np.float32, copy=False)
        euler = self._load_euler(hr_path, hr_quat)
        mask = periodic_known_mask(euler.shape[0], euler.shape[1], self.scale)
        target = torch.from_numpy(np.moveaxis(euler, -1, 0).copy())
        known_mask = torch.from_numpy(mask)
        if not self.return_quaternions:
            return target, known_mask
        lr_quat = _to_hwc_quaternion(np.load(lr_path)).astype(np.float32, copy=False)
        return target, known_mask, torch.from_numpy(lr_quat.copy()), torch.from_numpy(hr_quat.copy())


def load_authors_model(device: torch.device) -> torch.nn.Module:
    """Instantiate the authors' unmodified Model class and move plain mask tensors."""
    model_path = AUTHORS_DIR / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Atindama model source missing: {model_path}")
    spec = importlib.util.spec_from_file_location("atindama_authors_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model().to(device)

    # The upstream implementation stores these fixed convolution kernels as
    # ordinary tensors rather than registered buffers.
    for layer in model.modules():
        if hasattr(layer, "mask_update_kernel"):
            layer.mask_update_kernel = layer.mask_update_kernel.to(device)
    return model


def _axis_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    if axis == "X":
        values = (one, zero, zero, zero, cosine, -sine, zero, sine, cosine)
    elif axis == "Z":
        values = (cosine, -sine, zero, sine, cosine, zero, zero, zero, one)
    else:
        raise ValueError(f"Unsupported Euler axis {axis}")
    return torch.stack(values, dim=-1).reshape(*angle.shape, 3, 3)


def normalized_zxz_to_matrix(euler_grid: torch.Tensor) -> torch.Tensor:
    """Convert (B,3,H,W) normalized intrinsic ZXZ angles to rotation matrices."""
    angles = euler_grid.permute(0, 2, 3, 1)
    scale = angles.new_tensor([2.0 * math.pi, math.pi, 2.0 * math.pi])
    angles = angles * scale
    rz1 = _axis_rotation("Z", angles[..., 0])
    rx = _axis_rotation("X", angles[..., 1])
    rz2 = _axis_rotation("Z", angles[..., 2])
    return rz1 @ rx @ rz2


def geodesic_mse(
    target: torch.Tensor,
    prediction: torch.Tensor,
    known_mask: torch.Tensor,
    unknown_only: bool = True,
) -> torch.Tensor:
    """Authors' symmetry-agnostic squared SO(3) geodesic objective."""
    target_matrix = normalized_zxz_to_matrix(target)
    prediction_matrix = normalized_zxz_to_matrix(prediction)
    relative = target_matrix @ prediction_matrix.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0 + 1e-6, 1.0 - 1e-6
    )
    angle_sq = torch.acos(cosine).square()
    if not unknown_only:
        return angle_sq.mean()
    unknown = known_mask[:, 0] < 0.5
    return angle_sq[unknown].mean()


def _make_loader(
    cfg: dict,
    exp_dir: Path,
    split: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    take_first = cfg.get(f"{split.lower()}_take_first", cfg.get("take_first"))
    dataset = PeriodicEulerInpaintingDataset(
        dataset_root=cfg["dataset_root"],
        split=split,
        scale=cfg["scale"],
        cache_dir=exp_dir / "cache" / "euler_zxz" if cfg.get("cache_euler", True) else None,
        take_first=int(take_first) if take_first is not None else None,
    )
    workers = int(cfg.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=bool(cfg.get("persistent_workers", True)) and workers > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if workers > 0 else None,
        drop_last=bool(shuffle and cfg.get("drop_last", False)),
    )


def _run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    max_batches: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (target, known_mask) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            known_mask = known_mask.to(device=device, dtype=torch.float32, non_blocking=True)
            prediction, _ = model(target * known_mask, known_mask)
            loss = geodesic_mse(target, prediction, known_mask, unknown_only=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


def _save_checkpoint(path: Path, epoch: int, model, optimizer, best_val: float, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": float(best_val),
            "history": history,
            "source_model": str(AUTHORS_DIR / "model.py"),
        },
        path,
    )


def _save_plot(path: Path, history: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Squared SO(3) geodesic loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    exp_dir = Path(args.exp_dir)
    with open(exp_dir / args.config, "r") as handle:
        cfg = json.load(handle)
    with open(Path(cfg["dataset_root"]) / "dataset_info.json", "r") as handle:
        dataset_info = json.load(handle)
    dataset_symmetry = dataset_info.get(
        "symmetry", cfg.get("symmetry_group", "Oh")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    checkpoints_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    visualizations_dir = exp_dir / "visualizations"
    for directory in (checkpoints_dir, logs_dir, visualizations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_loader = _make_loader(
        cfg, exp_dir, "Train", int(cfg.get("batch_size", 8)), True
    )
    val_loader = _make_loader(
        cfg, exp_dir, "Val", int(cfg.get("eval_batch_size", 1)), False
    )

    model = load_authors_model(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    last_path = checkpoints_dir / "last_checkpoint.pt"
    best_path = checkpoints_dir / "best_model.pt"
    start_epoch = 0
    best_val = float("inf")
    history = {"train": [], "val": []}
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val_loss", best_val))
        history = checkpoint.get("history", history)
    elif args.init_checkpoint:
        init_checkpoint = Path(args.init_checkpoint)
        if not init_checkpoint.is_absolute():
            init_checkpoint = exp_dir / "checkpoints" / init_checkpoint
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"Initial checkpoint missing: {init_checkpoint}")
        checkpoint = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        print(f"Initialized model weights from {init_checkpoint}", flush=True)

    resolved = {
        **cfg,
        "device": str(device),
        "parameter_count": parameter_count,
        "source_repository": "https://github.com/Atindama/EBSD-Restoration-Inpainting-and-Denoising",
        "source_commit": "471aa11",
        "source_model": str(AUTHORS_DIR / "model.py"),
        "dataset_symmetry": dataset_symmetry,
        "training_symmetry": "none (authors' SO(3) geodesic objective)",
        "evaluation_symmetry": (
            f"proper rotational subgroup of {dataset_symmetry}"
        ),
    }
    with open(logs_dir / "run_config.json", "w") as handle:
        json.dump(resolved, handle, indent=2)

    log_handle = open(logs_dir / "train.log", "a", buffering=1)

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")

    epochs = int(cfg.get("epochs", 150))
    save_every = int(cfg.get("save_every", 10))
    save_last_checkpoint = bool(cfg.get("save_last_checkpoint", True))
    save_epoch_checkpoints = bool(cfg.get("save_epoch_checkpoints", True))
    log(
        f"Atindama authors Model: {parameter_count:,} parameters | "
        f"device={device} | scale={cfg['scale']}"
    )
    log(f"Dataset: {cfg['dataset_root']}")
    log(
        f"Dataset symmetry: {dataset_symmetry} | "
        "training objective: symmetry-agnostic | "
        "evaluation: proper rotational subgroup"
    )

    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        train_loss = _run_epoch(
            model,
            tqdm(train_loader, desc=f"[{epoch_number}/{epochs}] train", leave=False),
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        log(
            f"Epoch {epoch_number:04d}/{epochs} | "
            f"train={train_loss:.8e} val={val_loss:.8e}"
        )

        if save_last_checkpoint:
            _save_checkpoint(last_path, epoch, model, optimizer, best_val, history)
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(best_path, epoch, model, optimizer, best_val, history)
            log(f"  -> best_model.pt updated (val={best_val:.8e})")
        if save_epoch_checkpoints and save_every > 0 and epoch_number % save_every == 0:
            _save_checkpoint(
                checkpoints_dir / f"epoch_{epoch_number:04d}.pt",
                epoch,
                model,
                optimizer,
                best_val,
                history,
            )
        with open(checkpoints_dir / "history.json", "w") as handle:
            json.dump(history, handle, indent=2)
        _save_plot(visualizations_dir / "loss_curves.png", history)

    log(f"Training complete. Best validation loss: {best_val:.8e}")
    log_handle.close()


if __name__ == "__main__":
    main()
