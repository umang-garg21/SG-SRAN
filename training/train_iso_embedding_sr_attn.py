# -*- coding:utf-8 -*-
"""
Train IsoEmbeddingSRAttn with local-iso feature-space SR loss.

This trainer is specialized for models.SR_double_conv_SRattn.IsoEmbeddingSRAttn
and optimizes model.feature_loss_sr(...) directly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Make project imports robust when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.optimizer_utils import build_optimizer
from training.schedulers import build_scheduler
from training.seed_utils import get_seed_from_config, set_seed
from utils.symmetry_utils import resolve_symmetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IsoEmbeddingSRAttn")
    parser.add_argument(
        "--exp_dir",
        required=True,
        type=str,
        help="Path to experiment directory containing config.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config filename inside exp_dir (default: config.json)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value, e.g. '0' or '0,1'",
    )
    parser.add_argument(
        "--model_module",
        type=str,
        default="models.SR_double_conv_SRattn",
        help="Dotted module path containing the model class (default: models.SR_double_conv_SRattn)",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="IsoEmbeddingSRAttn",
        help="Class name to instantiate from model_module (default: IsoEmbeddingSRAttn)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/last_checkpoint.pt if present",
    )
    return parser.parse_args()


def _flatten_quat_chw_batch(q: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert (B,4,H,W) or (B,H,W,4) quaternion batch to (B,H*W,4)."""
    if q.dim() != 4:
        raise ValueError(f"Expected rank-4 quaternion batch, got {tuple(q.shape)}")
    if q.shape[1] == 4:
        bsz, _, h, w = q.shape
        q_flat = q.permute(0, 2, 3, 1).reshape(bsz, h * w, 4)
        return q_flat, (h, w)
    if q.shape[-1] == 4:
        bsz, h, w, _ = q.shape
        q_flat = q.reshape(bsz, h * w, 4)
        return q_flat, (h, w)
    raise ValueError(f"Expected quaternion axis of size 4 in dim=1 or dim=-1, got {tuple(q.shape)}")


def _to_hwc_quat_single(q: torch.Tensor) -> torch.Tensor:
    """Convert single quaternion image to (H,W,4) from (4,H,W) or (H,W,4)."""
    if q.dim() != 3:
        raise ValueError(f"Expected rank-3 quaternion image, got {tuple(q.shape)}")
    if q.shape[0] == 4:
        return q.permute(1, 2, 0)
    if q.shape[-1] == 4:
        return q
    raise ValueError(f"Expected quaternion axis of size 4 in dim=0 or dim=-1, got {tuple(q.shape)}")


def _get_take_first(cfg, split: str):
    """Resolve optional dataset truncation for split."""
    if bool(getattr(cfg, "smoke_test", False)):
        return int(getattr(cfg, "smoke_take_first", 8))
    key = f"{split.lower()}_take_first"
    val = getattr(cfg, key, None)
    return int(val) if val is not None else None


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _save_history(history_path: Path, history: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def _save_loss_plot(plot_path: Path, history: dict, exp_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train = history.get("train", [])
        val   = history.get("val", [])
        lr    = history.get("lr", [])
        epochs = list(range(1, len(train) + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        fig.suptitle(exp_name, fontsize=11)

        ax1.plot(epochs, train, label="train", linewidth=1.5)
        ax1.plot(epochs, val,   label="val",   linewidth=1.5, linestyle="--")
        ax1.set_ylabel("MSE Loss (feature space)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, lr[:len(epochs)], color="tab:orange", linewidth=1.5)
        ax2.set_ylabel("Learning Rate")
        ax2.set_xlabel("Epoch")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warning] Could not save loss plot: {exc}")


def _save_checkpoint(
    ckpt_path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_loss: float,
    history: dict,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": float(best_val_loss),
            "history": history,
        },
        ckpt_path,
    )


def _load_checkpoint(
    ckpt_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    history = ckpt.get("history", {"train": [], "val": [], "lr": []})
    return start_epoch, best_val_loss, history


def _train_one_epoch(
    model_core: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler,
    memory_debug_every: int = 0,
    cuda_empty_cache_every: int = 0,
) -> float:
    model_core.train()
    total_loss = 0.0
    n_steps = 0

    for lr, hr in tqdm(loader, desc="Train", leave=False):
        lr = lr.to(device=device, dtype=torch.float32, non_blocking=True)
        hr = hr.to(device=device, dtype=torch.float32, non_blocking=True)
        lr_flat, lr_shape = _flatten_quat_chw_batch(lr)
        hr_flat, _ = _flatten_quat_chw_batch(hr)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            loss = model_core.feature_loss_sr(
                lr_flat,
                hr_flat,
                lr_shape=lr_shape,
                normalize_input=True,
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_core.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model_core.parameters(), clip)
            optimizer.step()

        total_loss += float(loss.detach().item())
        n_steps += 1
        if (
            device.type == "cuda"
            and int(cuda_empty_cache_every) > 0
            and (n_steps % int(cuda_empty_cache_every) == 0)
        ):
            torch.cuda.empty_cache()
        if (
            device.type == "cuda"
            and int(memory_debug_every) > 0
            and (n_steps % int(memory_debug_every) == 0)
        ):
            alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
            resv_gb = torch.cuda.memory_reserved(device) / (1024**3)
            max_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            max_resv_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
            print(
                f"[cuda-mem] step={n_steps} alloc={alloc_gb:.2f}G reserved={resv_gb:.2f}G "
                f"max_alloc={max_alloc_gb:.2f}G max_reserved={max_resv_gb:.2f}G"
            )

    return total_loss / max(1, n_steps)


@torch.no_grad()
def _validate_one_epoch(
    model_core: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model_core.eval()
    total_loss = 0.0
    n_steps = 0

    for lr, hr in tqdm(loader, desc="Val", leave=False):
        lr = lr.to(device=device, dtype=torch.float32, non_blocking=True)
        hr = hr.to(device=device, dtype=torch.float32, non_blocking=True)
        lr_flat, lr_shape = _flatten_quat_chw_batch(lr)
        hr_flat, _ = _flatten_quat_chw_batch(hr)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            loss = model_core.feature_loss_sr(
                lr_flat,
                hr_flat,
                lr_shape=lr_shape,
                normalize_input=True,
            )
        total_loss += float(loss.detach().item())
        n_steps += 1

    return total_loss / max(1, n_steps)


def _render_sr_hr_lr_ipf(
    model_core: torch.nn.Module,
    data_loader,
    sym_class,
    out_png: Path,
    ref_dir: str = "ALL",
) -> bool:
    """
    Render LR/SR/HR IPF comparison from the first sample in data_loader.
    """
    try:
        from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side
    except Exception as exc:
        print(f"[warning] Visualization unavailable (import failed): {exc}")
        return False

    batch = next(iter(data_loader), None)
    if batch is None:
        print("[warning] Skipping visualization: loader is empty.")
        return False

    lr, hr = batch
    if lr is None or hr is None or lr.shape[0] == 0 or hr.shape[0] == 0:
        print("[warning] Skipping visualization: empty LR/HR batch.")
        return False

    try:
        model_was_training = model_core.training
        model_core.eval()

        device = next(model_core.parameters()).device
        lr0 = _to_hwc_quat_single(lr[0].to(device=device, dtype=torch.float32))
        hr0 = _to_hwc_quat_single(hr[0].to(device=device, dtype=torch.float32))
        lr_h, lr_w = int(lr0.shape[0]), int(lr0.shape[1])
        hr_h, hr_w = int(hr0.shape[0]), int(hr0.shape[1])

        lr_flat = lr0.reshape(-1, 4)
        q_sr_flat = model_core.forward_sr(
            lr_flat,
            lr_shape=(lr_h, lr_w),
            normalize_input=True,
        )

        if int(q_sr_flat.shape[0]) != hr_h * hr_w:
            raise ValueError(
                f"SR output size mismatch: got N={int(q_sr_flat.shape[0])}, expected {hr_h * hr_w}"
            )

        lr_np = lr0.detach().cpu().numpy()
        hr_np = hr0.detach().cpu().numpy()
        sr_np = q_sr_flat.reshape(hr_h, hr_w, 4).detach().cpu().numpy()

        out_png.parent.mkdir(parents=True, exist_ok=True)
        render_sr_hr_lr_side_by_side(
            sr_q_arr=sr_np,
            hr_q_arr=hr_np,
            lr_q_arr=lr_np,
            sym_class=sym_class,
            out_png=str(out_png),
            ref_dir=str(ref_dir),
            include_key=True,
            overwrite=True,
            format_input=True,
            dpi=300,
        )
        print(f"Saved LR/SR/HR IPF visualization: {out_png}")
        return True
    except Exception as exc:
        print(f"[warning] Failed to render LR/SR/HR IPF visualization: {exc}")
        return False
    finally:
        if "model_was_training" in locals() and model_was_training:
            model_core.train()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / args.config
    run_config_path = exp_dir / "logs" / "run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

    model_cfg = getattr(cfg, "model", {}) or {}
    model_module = model_cfg.get("model_module", args.model_module) if isinstance(model_cfg, dict) else getattr(model_cfg, "model_module", args.model_module)
    model_class = model_cfg.get("model_class", args.model_class) if isinstance(model_cfg, dict) else getattr(model_cfg, "model_class", args.model_class)
    IsoEmbeddingSRAttn = getattr(importlib.import_module(model_module), model_class)
    print(f"Model: {model_module}.{model_class}")

    seed = int(get_seed_from_config(cfg))
    set_seed(seed)

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
        free_gb = float(free_bytes) / (1024**3)
        total_gb = float(total_bytes) / (1024**3)
        print(f"CUDA free memory at startup: {free_gb:.2f} / {total_gb:.2f} GB")
        min_free_cuda_gb = float(getattr(cfg, "min_free_cuda_gb", 0.0))
        if min_free_cuda_gb > 0.0 and free_gb < min_free_cuda_gb:
            raise RuntimeError(
                f"Insufficient free CUDA memory at startup: {free_gb:.2f} GB < "
                f"min_free_cuda_gb={min_free_cuda_gb:.2f} GB. "
                "Another GPU process is likely active. "
                "Run `nvidia-smi` and kill the listed PID(s), then retry."
            )
    amp_dtype_str = str(getattr(cfg, "amp_dtype", "bf16")).lower()
    if amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    elif amp_dtype_str in ("fp16", "float16", "half"):
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported amp_dtype={amp_dtype_str!r}; use 'bf16' or 'fp16'")
    use_amp = bool(getattr(cfg, "use_amp", True)) and device.type == "cuda"
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if (use_amp and amp_dtype == torch.float16)
        else None
    )
    print(f"AMP enabled: {use_amp} (dtype={amp_dtype_str})")

    loaders = {}
    for split in ("train", "val", "test"):
        loaders[split] = build_dataloader(
            dataset_root=cfg.dataset_root,
            split=split.capitalize(),
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
            preload=bool(cfg.preload),
            preload_torch=bool(cfg.preload_torch),
            pin_memory=bool(cfg.pin_memory),
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            take_first=_get_take_first(cfg, split),
            seed=seed,
        )

    import inspect
    _init_params = set(inspect.signature(IsoEmbeddingSRAttn.__init__).parameters)
    _residual_kwargs = {}
    if "use_residual_lr1" in _init_params:
        _residual_kwargs["use_residual_lr1"] = bool(getattr(cfg, "use_residual_lr1", False))
    if "use_residual_lr2" in _init_params:
        _residual_kwargs["use_residual_lr2"] = bool(getattr(cfg, "use_residual_lr2", False))
    if "use_residual_hr1" in _init_params:
        _residual_kwargs["use_residual_hr1"] = bool(getattr(cfg, "use_residual_hr1", False))

    model = IsoEmbeddingSRAttn(
        crystal=str(getattr(cfg, "crystal", "fcc")),
        d6_convention=str(getattr(cfg, "d6_convention", "z_axis")),
        device=device,
        upsample_factor=int(getattr(cfg, "scale", 4)),
        upsample_residual=bool(getattr(cfg, "upsample_residual", True)),
        use_lr_conv1=bool(getattr(cfg, "use_lr_conv1", True)),
        use_lr_conv2=bool(getattr(cfg, "use_lr_conv2", True)),
        **_residual_kwargs,
        use_attention=bool(getattr(cfg, "use_attention", True)),
        num_hr_attn_blocks=int(getattr(cfg, "num_hr_attn_blocks", 1)),
        hr_attn_num_channels=int(getattr(cfg, "hr_attn_num_channels", 8)),
        hr_attn_block_size=int(getattr(cfg, "hr_attn_block_size", 16)),
        hr_attn_tp_out_chunk_size=getattr(cfg, "hr_attn_tp_out_chunk_size", 2048),
        hr_attn_checkpoint=bool(getattr(cfg, "hr_attn_checkpoint", False)),
        decoder_cubochoric_resolution=int(getattr(cfg, "decoder_cubochoric_resolution", 1)),
        decoder_num_starts=int(getattr(cfg, "decoder_num_starts", 2)),
        decoder_steps=int(getattr(cfg, "decoder_steps", 1)),
        decoder_lr=float(getattr(cfg, "decoder_lr", 0.05)),
        decoder_method=str(getattr(cfg, "decoder_method", "cubochoric")),
        decoder_max_table_rows=getattr(cfg, "decoder_max_table_rows", None),
        decoder_table_cache_dir=getattr(
            cfg, "decoder_table_cache_dir", "out/decoder_lookup_tables"
        ),
        decoder_backend=str(getattr(cfg, "decoder_backend", "optimizing")),
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_tb = bool(getattr(getattr(cfg, "logging", {}), "tensorboard", True))
    writer = None
    if use_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            writer = SummaryWriter(log_dir=exp_dir / "runs")
        except Exception as exc:
            print(f"[warning] TensorBoard disabled (could not import SummaryWriter): {exc}")

    checkpoints_dir = Path(cfg.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoints_dir / "history.json"
    best_ckpt = checkpoints_dir / "best_model.pt"
    last_ckpt = checkpoints_dir / "last_checkpoint.pt"

    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train": [], "val": [], "lr": []}

    if args.resume and last_ckpt.exists():
        start_epoch, best_val_loss, history = _load_checkpoint(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"Resumed from {last_ckpt} at epoch {start_epoch}")

    clip = float(getattr(cfg, "clip", 1.0))
    save_every = int(getattr(cfg, "save_every", 1))
    viz_every = int(getattr(cfg, "viz_every", save_every))
    viz_ref_dir = str(getattr(cfg, "viz_ref_dir", "ALL"))
    memory_debug_every = int(getattr(cfg, "memory_debug_every", 0))
    cuda_empty_cache_every = int(getattr(cfg, "cuda_empty_cache_every", 0))
    epochs = int(cfg.epochs)
    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))

    for epoch in range(start_epoch, epochs):
        model_core = _unwrap_model(model)
        train_loss = _train_one_epoch(
            model_core,
            loaders["train"],
            optimizer,
            device,
            clip=clip,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            memory_debug_every=memory_debug_every,
            cuda_empty_cache_every=cuda_empty_cache_every,
        )
        val_loss = _validate_one_epoch(
            model_core,
            loaders["val"],
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        if scheduler is not None:
            scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        history["lr"].append(current_lr)

        if writer is not None:
            writer.add_scalar("Loss/Train", float(train_loss), epoch)
            writer.add_scalar("Loss/Val", float(val_loss), epoch)
            writer.add_scalar("LR", current_lr, epoch)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train={train_loss:.6e} val={val_loss:.6e} lr={current_lr:.2e}"
        )

        _save_checkpoint(
            last_ckpt,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            history=history,
        )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            _save_checkpoint(
                best_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                history=history,
            )

        if save_every > 0 and ((epoch + 1) % save_every == 0):
            epoch_ckpt = checkpoints_dir / f"epoch_{epoch + 1:04d}.pt"
            _save_checkpoint(
                epoch_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                history=history,
            )

        if viz_every > 0 and ((epoch + 1) % viz_every == 0):
            viz_dir = exp_dir / "visualizations" / f"epoch_{epoch + 1:04d}"
            _render_sr_hr_lr_ipf(
                model_core=model_core,
                data_loader=loaders["val"],
                sym_class=sym_class,
                out_png=viz_dir / "lr_sr_hr_ipf.png",
                ref_dir=viz_ref_dir,
            )

        _save_history(history_path, history)
        _save_loss_plot(
            exp_dir / "visualizations" / "loss_curves.png",
            history,
            exp_name=exp_dir.name,
        )

    final_viz_dir = exp_dir / "visualizations" / "final"
    _render_sr_hr_lr_ipf(
        model_core=_unwrap_model(model),
        data_loader=loaders["test"] if "test" in loaders else loaders["val"],
        sym_class=sym_class,
        out_png=final_viz_dir / "lr_sr_hr_ipf.png",
        ref_dir=viz_ref_dir,
    )

    if writer is not None:
        writer.close()

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.6e}")
    print(f"Checkpoints: {checkpoints_dir}")


if __name__ == "__main__":
    main()
