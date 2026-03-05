import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.autoencoder import FCCAutoEncoder
from models.autoencoder_learnable import FCCLearnableDecoderAutoEncoder
from models.SR_conv import FCCAutoEncoder as FCCAutoEncoderWithConv
from models.SR_conv import FCCAutoEncoderSR
from models.SR_global_attn import FCCAutoEncoderSRGlobalAttn
from models.SR_grain_attn import FCCAutoEncoderSRBlockAttn
from models.SR_double_conv import FCCAutoEncoderSR as FCCAutoEncoderSRDoubleConv
from models.SR_double_conv_SRattn import FCCAutoEncoderSRDoubleConvAttn
from models.SR_boundary_guided import FCCAutoEncoderSRBoundaryGuided
from training.autoencoder import Autoencoder
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.loss_functions import build_loss, reduce_to_fz_min_angle_torch_fast
from training.optimizer_utils import build_optimizer
from training.schedulers import build_scheduler
from training.seed_utils import set_seed
from utils.symmetry_utils import resolve_symmetry
from visualization.visualize_sr_results import render_input_output_side_by_side, render_sr_hr_lr_side_by_side


class TrainableFCCAutoEncoder(nn.Module):
    """Train-only wrapper around the simple FCCAutoEncoder core."""

    def __init__(self, core: nn.Module, decode_chunk_size: int = 128):
        super().__init__()
        self.core = core
        self.decode_chunk_size = int(decode_chunk_size)

    def forward(
        self,
        quats: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        # When img_shape is given (conv model), pass the whole image at once so
        # that the spatial conv layer can use the 2-D grid structure.
        if img_shape is not None:
            q_decoded = self.core(quats, img_shape=img_shape, normalize_input=normalize_input)
        else:
            chunks = []
            for start in range(0, quats.shape[0], self.decode_chunk_size):
                end = min(start + self.decode_chunk_size, quats.shape[0])
                chunks.append(self.core(quats[start:end], normalize_input=normalize_input))
            q_decoded = torch.cat(chunks, dim=0)
        norm = torch.norm(q_decoded, dim=-1, keepdim=True).clamp_min(1e-12)
        return q_decoded / norm

    def quat_mul(self, q1: torch.Tensor, q2: torch.Tensor):
        return self.core.quat_mul(q1, q2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FCC autoencoder")
    parser.add_argument("--exp_dir", required=True, type=str, help="Experiment directory")
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
        help="Optional CUDA_VISIBLE_DEVICES override, e.g. '0' or '0,1'",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    return parser.parse_args()


def plot_loss(train_losses, val_losses, save_path=None, start_epoch=1):
    epochs = list(range(start_epoch, start_epoch + len(train_losses)))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", color="blue", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📈 Plot saved to {save_path}")
        plt.close()


@torch.no_grad()
def render_autoencoder_input_output(
    model,
    val_loader,
    sym_class,
    out_png: str,
):
    model_was_training = model.training
    model.eval()

    batch = next(iter(val_loader), None)
    if batch is None:
        return

    lr, _ = batch
    if lr is None or lr.shape[0] == 0:
        return

    def _model_device(m: nn.Module) -> torch.device:
        try:
            return next(m.parameters()).device
        except StopIteration:
            pass
        try:
            return next(m.buffers()).device
        except StopIteration:
            pass
        core = getattr(m, "core", None)
        if core is not None and hasattr(core, "device"):
            return torch.device(core.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First sample from validation batch: (4,H,W) scalar-first -> flatten (N,4)
    lr0 = lr[0]
    h, w = int(lr0.shape[1]), int(lr0.shape[2])
    device = _model_device(model)
    q_flat = lr0.permute(1, 2, 0).reshape(-1, 4).to(device)

    # Match simple_encoder_decoder behavior: normalize inputs before encode/decode
    q_flat = q_flat / torch.norm(q_flat, dim=1, keepdim=True).clamp_min(1e-12)

    # Decode in batches (direct output; no symmetry matching in trainer path).
    step = 1000 if device.type == "cuda" else 500
    q_reconstructed_all = []
    for start in range(0, q_flat.shape[0], step):
        end = min(start + step, q_flat.shape[0])
        q_batch = q_flat[start:end]
        q_dec = model(q_batch, normalize_input=True)
        q_reconstructed_all.append(q_dec)

    q_reconstructed_all = torch.cat(q_reconstructed_all, dim=0)
    q_in = q_flat.reshape(h, w, 4).detach().cpu().numpy()
    q_out = q_reconstructed_all.reshape(h, w, 4).detach().cpu().numpy()

    render_input_output_side_by_side(
        input_q_arr=q_in,
        output_q_arr=q_out,
        sym_class=sym_class,
        out_png=out_png,
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=False,
        dpi=300,
    )

    if model_was_training:
        model.train()


@torch.no_grad()
def render_sr_output(model, val_loader, sym_class, out_png: str):
    """Render LR / SR / HR IPF comparison for the SR model (first val batch)."""
    model_was_training = model.training
    model.eval()

    batch = next(iter(val_loader), None)
    if batch is None:
        return

    lr_batch, hr_batch = batch
    if lr_batch is None or lr_batch.shape[0] == 0:
        return

    def _model_device(m):
        try:
            return next(m.parameters()).device
        except StopIteration:
            pass
        core = getattr(m, "core", None)
        if core is not None and hasattr(core, "device"):
            return torch.device(core.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = _model_device(model)
    _core = getattr(model, "core", model)

    # First sample from the batch
    lr0 = lr_batch[0]   # (4, H_lr, W_lr)
    hr0 = hr_batch[0]   # (4, H_hr, W_hr)

    h_lr, w_lr = int(lr0.shape[1]), int(lr0.shape[2])
    h_hr, w_hr = int(hr0.shape[1]), int(hr0.shape[2])

    lr_flat = lr0.permute(1, 2, 0).reshape(-1, 4).to(device)
    lr_flat = lr_flat / lr_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # Run SR pipeline
    q_sr = _core.forward_sr(lr_flat, lr_shape=(h_lr, w_lr))

    # Reshape to (H, W, 4) numpy arrays for rendering
    q_lr_np = lr_flat.reshape(h_lr, w_lr, 4).detach().cpu().numpy()
    q_sr_np = q_sr.reshape(h_hr, w_hr, 4).detach().cpu().numpy()
    q_hr_np = hr0.permute(1, 2, 0).reshape(h_hr, w_hr, 4).detach().cpu().numpy()

    render_sr_hr_lr_side_by_side(
        sr_q_arr=q_sr_np,
        hr_q_arr=q_hr_np,
        lr_q_arr=q_lr_np,
        sym_class=sym_class,
        out_png=out_png,
        ref_dir="ALL",
        include_key=True,
        overwrite=True,
        format_input=False,
        dpi=300,
    )

    if model_was_training:
        model.train()


@torch.no_grad()
def print_simple_encoder_decoder_stats(model, data_loader, split_name: str = "test"):
    batch = next(iter(data_loader), None)
    if batch is None:
        print(f"⚠️ No data available in {split_name} loader for stats.")
        return

    _, lr  = batch
    if lr is None or lr.shape[0] == 0:
        print(f"⚠️ Empty batch in {split_name} loader for stats.")
        return

    def _model_device(m: nn.Module) -> torch.device:
        try:
            return next(m.parameters()).device
        except StopIteration:
            pass
        try:
            return next(m.buffers()).device
        except StopIteration:
            pass
        core = getattr(m, "core", None)
        if core is not None and hasattr(core, "device"):
            return torch.device(core.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mirror simple_encoder_decoder flow on first sample
    lr0 = lr[0]  # (4,H,W)
    q_all = lr0.permute(1, 2, 0).reshape(-1, 4).to(_model_device(model))
    q_all = q_all / torch.norm(q_all, dim=1, keepdim=True).clamp_min(1e-12)
    num_quats = q_all.shape[0]

    batch_size = 1000 if q_all.device.type == "cuda" else 500
    all_errors = []
    all_misorientation_angles = []

    # Build FZ-reduction helper once (uses model's own 24-element O group via
    # fcc_syms, not orix's Oh which is the 48-element full cubic group).
    _core = getattr(model, "core", model)
    _reduce = getattr(_core, "reduce_to_fz", None)

    def _fz_reduce_flat(q_flat: torch.Tensor) -> torch.Tensor:
        """(N,4) → FZ-reduced (N,4) using the model's fcc_syms (O group)."""
        if _reduce is not None:
            return _reduce(q_flat)
        # Fallback: reshape (N,4) → (1,4,N,1) for the fast function ("O" = 24 ops)
        q_bchw = q_flat.T.unsqueeze(0).unsqueeze(-1)
        q_fz = reduce_to_fz_min_angle_torch_fast(q_bchw, "O")
        return q_fz.squeeze(0).squeeze(-1).T

    for batch_start in range(0, num_quats, batch_size):
        batch_end = min(batch_start + batch_size, num_quats)
        q_batch = q_all[batch_start:batch_end]

        q_dec = model(q_batch, normalize_input=True)

        # FZ-reduce both original and decoded before comparing so that
        # crystal-equivalent outputs (all 24 FCC equivalents map to the same
        # (f4,f6)) are not counted as errors.
        q_dec_fz  = _fz_reduce_flat(q_dec)
        q_orig_fz = _fz_reduce_flat(q_batch)

        # Disorientation = dot product of FZ-reduced quaternions (w of q_orig⁻¹⊗q_dec)
        w_errors = torch.sum(q_orig_fz * q_dec_fz, dim=-1)
        w_errors_clamped = torch.clamp(torch.abs(w_errors), max=1.0)
        errors = 2.0 * torch.acos(w_errors_clamped)
        misorientation_angles = 2.0 * torch.acos(w_errors_clamped) * 180.0 / torch.pi

        all_errors.extend(errors.detach().cpu().tolist())
        all_misorientation_angles.extend(misorientation_angles.detach().cpu().tolist())

    all_errors = np.array(all_errors)
    all_misorientation_angles = np.array(all_misorientation_angles)

    print("\n" + "=" * 70)
    print(f"RECONSTRUCTION ERROR STATISTICS ({split_name.upper()} FIRST SAMPLE)")
    print("=" * 70)
    print(f"Total quaternions processed: {num_quats}")
    print("\nError Distance:")
    print(f"  Maximum: {np.max(all_errors):.6e}")
    print(f"  Mean:    {np.mean(all_errors):.6e}")
    print(f"  Median:  {np.median(all_errors):.6e}")
    print(f"  Std Dev: {np.std(all_errors):.6e}")
    print("\nMisorientation Angle:")
    print(f"  Maximum: {np.max(all_misorientation_angles):.4f}°")
    print(f"  Mean:    {np.mean(all_misorientation_angles):.4f}°")
    print(f"  Median:  {np.median(all_misorientation_angles):.4f}°")
    print(f"  Std Dev: {np.std(all_misorientation_angles):.4f}°")

    worst_idx = int(np.argmax(all_errors))
    print(f"\nWorst Case (index {worst_idx}):")
    print(f"  Original:     {q_all[worst_idx].detach().cpu().numpy()}")
    print(f"  Error:        {all_errors[worst_idx]:.6e}")
    print(f"  Misorientation: {all_misorientation_angles[worst_idx]:.4f}°")

    if np.max(all_errors) < 0.05:
        print("\n>> SUCCESS: All quaternions restored within tolerance!")
    else:
        bad_indices = np.where(all_errors >= 0.05)[0]
        print(f"\n>> WARNING: {len(bad_indices)} quaternion(s) exceeded error threshold of 0.05")
        for bi in bad_indices:
            print(f"   index {bi}: q={q_all[bi].detach().cpu().numpy()}  error={all_errors[bi]:.6e}  mis={all_misorientation_angles[bi]:.4f}°")
    print("   (Note: Error depends on grid size. Increase grid for more precision.)")


_RUN_DT_FMT = "%Y-%m-%d_%H-%M-%S"


def resolve_run_dir(exp_dir: Path, resume: bool) -> Path:
    """Return the run directory for this invocation.

    Fresh run : creates a new ``YYYY-MM-DD_HH-MM-SS`` subfolder.
    Resume    : finds the most-recent timestamped subfolder that contains
                a ``checkpoints/`` directory; falls back to a new folder
                when none exist.
    """
    if resume:
        candidates = [
            d for d in exp_dir.iterdir()
            if d.is_dir() and (d / "checkpoints").is_dir()
            and _is_dt_folder(d.name)
        ]
        if candidates:
            run_dir = max(candidates, key=lambda d: d.name)
            print(f"Resuming run directory: {run_dir.name}")
            return run_dir
        print("No previous run directories with checkpoints found; starting a new run.")

    run_dir = exp_dir / datetime.now().strftime(_RUN_DT_FMT)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.name}")
    return run_dir


def _is_dt_folder(name: str) -> bool:
    try:
        datetime.strptime(name, _RUN_DT_FMT)
        return True
    except ValueError:
        return False


def main():
    args_cli = parse_args()
    exp_dir = Path(args_cli.exp_dir)

    run_dir = resolve_run_dir(exp_dir, args_cli.resume)

    config_path = exp_dir / args_cli.config
    run_config_path = run_dir / "logs" / "run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

    # Redirect all run outputs into the datetime subfolder
    cfg["checkpoints_dir"] = str(run_dir / "checkpoints")
    Path(cfg["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)

    seed = int(getattr(cfg, "seed", 42))
    set_seed(seed)

    if args_cli.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.gpu_ids
        print(f"CUDA_VISIBLE_DEVICES set to: {args_cli.gpu_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"Using device: {device}")

    loaders = {
        split: build_dataloader(
            dataset_root=cfg.dataset_root,
            split=split.capitalize(),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            preload=cfg.preload,
            preload_torch=cfg.preload_torch,
            pin_memory=cfg.pin_memory,
            take_first=10 if cfg.smoke_test else None,
            seed=seed,
        )
        for split in ["train", "val", "test"]
    }

    grid_res = int(getattr(cfg, "grid_res", 2048))
    decode_chunk_size = int(getattr(cfg, "decode_chunk_size", 128))
    print(f"[train_autoencoder] Using grid_res={grid_res}, decode_chunk_size={decode_chunk_size}")

    model_cfg = getattr(cfg, "model", None)
    requested_model_type = str(
        getattr(model_cfg, "type", getattr(cfg, "model_type", "fcc_autoencoder"))
    ).lower()

    if requested_model_type == "fcc_autoencoder_learnable_decoder":
        core_model = FCCLearnableDecoderAutoEncoder(
            device=device,
            hidden_dim=int(getattr(cfg, "decoder_hidden_dim", 128)),
            num_layers=int(getattr(cfg, "decoder_num_layers", 3)),
            dropout=float(getattr(cfg, "decoder_dropout", 0.0)),
        ).to(device)
    elif requested_model_type == "fcc_autoencoder":
        decoder_config = {
            "decoder_cubochoric_resolution": int(getattr(cfg, "decoder_cubochoric_resolution", 3)),
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_learnable_hidden_dim": int(getattr(cfg, "decoder_learnable_hidden_dim", 256)),
            "decoder_learnable_num_layers": int(getattr(cfg, "decoder_learnable_num_layers", 3)),
            "decoder_learnable_dropout": float(getattr(cfg, "decoder_learnable_dropout", 0.0)),
            "decoder_learnable_ckpt_path": getattr(cfg, "decoder_learnable_ckpt_path", None),
            "decoder_learnable_ckpt_strict": bool(getattr(cfg, "decoder_learnable_ckpt_strict", True)),
            "decoder_num_starts": int(getattr(cfg, "decoder_num_starts", 6)),
            "decoder_steps": int(getattr(cfg, "decoder_steps", 25)),
            "decoder_lr": float(getattr(cfg, "decoder_lr", 0.08)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
            "decoder_early_stop_tol": float(getattr(cfg, "decoder_early_stop_tol", 1e-6)),
            "decoder_early_stop_patience": int(getattr(cfg, "decoder_early_stop_patience", 3)),
            "decoder_min_steps": int(getattr(cfg, "decoder_min_steps", 6)),
            "decoder_log_optimization": bool(getattr(cfg, "decoder_log_optimization", False)),
            "decoder_log_every": int(getattr(cfg, "decoder_log_every", 1)),
        }

        core_model = FCCAutoEncoder(
            device=device,
            grid_res=grid_res,
            decoder_backend=str(getattr(cfg, "decoder_backend", "optimizing")),
            decoder_config=decoder_config,
        ).to(device)
    elif requested_model_type == "fcc_autoencoder_with_conv":
        decoder_config = {
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
        }
        core_model = FCCAutoEncoderWithConv(
            device=device,
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=decoder_config,
        ).to(device)
        # Tell the trainer to process each image separately so the conv layer
        # receives the correct 2-D spatial grid.
        cfg["use_img_shape"] = True
    elif requested_model_type == "fcc_autoencoder_sr":
        decoder_config = {
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
        }
        core_model = FCCAutoEncoderSR(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            upsample_residual=bool(getattr(cfg, "upsample_residual", False)),
            upsampler="conv",
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=decoder_config,
        ).to(device)
        # Trainer must receive paired (lr, hr) batches and run the SR pipeline.
        cfg["use_sr"] = True
        cfg["use_img_shape"] = True
    elif requested_model_type == "fcc_autoencoder_sr_attn":
        decoder_config = {
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
        }
        core_model = FCCAutoEncoderSR(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            upsampler="attention",
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=decoder_config,
        ).to(device)
        # Trainer must receive paired (lr, hr) batches and run the SR pipeline.
        cfg["use_sr"] = True
        cfg["use_img_shape"] = True
    elif requested_model_type == "fcc_autoencoder_sr_double_conv":
        decoder_config = {
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
        }
        _lr_shape_raw = getattr(cfg, "lr_shape", None)
        _lr_shape = tuple(int(x) for x in _lr_shape_raw) if _lr_shape_raw is not None else None
        def _opt_int(key):
            v = getattr(cfg, key, None)
            return int(v) if v is not None else None
        core_model = FCCAutoEncoderSRDoubleConv(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            upsample_residual=bool(getattr(cfg, "upsample_residual", False)),
            upsampler="conv",
            lr_shape=_lr_shape,
            lr_conv_kernel_size=_opt_int("lr_conv_kernel_size"),
            lr_conv_kernel_size_1=_opt_int("lr_conv_kernel_size_1"),
            lr_conv_kernel_size_2=_opt_int("lr_conv_kernel_size_2"),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=decoder_config,
        ).to(device)
        cfg["use_sr"] = True
        cfg["use_img_shape"] = True
    elif requested_model_type == "fcc_autoencoder_sr_double_conv_attn":
        decoder_config = {
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
        }
        _lr_shape_raw = getattr(cfg, "lr_shape", None)
        _lr_shape = tuple(int(x) for x in _lr_shape_raw) if _lr_shape_raw is not None else None
        def _opt_int(key):
            v = getattr(cfg, key, None)
            return int(v) if v is not None else None
        core_model = FCCAutoEncoderSRDoubleConvAttn(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            upsample_residual=bool(getattr(cfg, "upsample_residual", False)),
            upsampler="conv",
            lr_shape=_lr_shape,
            lr_conv_kernel_size=_opt_int("lr_conv_kernel_size"),
            lr_conv_kernel_size_1=_opt_int("lr_conv_kernel_size_1"),
            lr_conv_kernel_size_2=_opt_int("lr_conv_kernel_size_2"),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=decoder_config,
            num_hr_attn_blocks=int(getattr(cfg, "hr_attn_num_blocks", 2)),
            hr_attn_num_channels=int(getattr(cfg, "hr_attn_num_channels", 8)),
            hr_attn_block_size=int(getattr(cfg, "hr_attn_block_size", 16)),
        ).to(device)
        cfg["use_sr"] = True
        cfg["use_img_shape"] = True
    elif requested_model_type == "fcc_autoencoder_sr_boundary_guided":
        decoder_config = {
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
        }
        _lr_shape_raw = getattr(cfg, "lr_shape", None)
        _lr_shape = tuple(int(x) for x in _lr_shape_raw) if _lr_shape_raw is not None else None
        def _opt_int(key):
            v = getattr(cfg, key, None)
            return int(v) if v is not None else None
        core_model = FCCAutoEncoderSRBoundaryGuided(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            upsample_residual=bool(getattr(cfg, "upsample_residual", False)),
            lr_shape=_lr_shape,
            lr_conv_kernel_size=_opt_int("lr_conv_kernel_size"),
            lr_conv_kernel_size_1=_opt_int("lr_conv_kernel_size_1"),
            lr_conv_kernel_size_2=_opt_int("lr_conv_kernel_size_2"),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=decoder_config,
            bg_window_size=int(getattr(cfg, "bg_window_size", 3)),
            bg_init_sigma=float(getattr(cfg, "bg_init_sigma", 0.5)),
            bg_init_lambda=float(getattr(cfg, "bg_init_lambda", 2.0)),
            bg_init_gamma=float(getattr(cfg, "bg_init_gamma", 1.0)),
            hr_attn_num_blocks=int(getattr(cfg, "hr_attn_num_blocks", 2)),
            hr_attn_num_channels=int(getattr(cfg, "hr_attn_num_channels", 8)),
            hr_attn_block_size=int(getattr(cfg, "hr_attn_block_size", 16)),
        ).to(device)
        cfg["use_sr"] = True
        cfg["use_img_shape"] = True
    elif requested_model_type == "fcc_autoencoder_sr_global_attn":
        decoder_config = {
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
        }
        core_model = FCCAutoEncoderSRGlobalAttn(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            num_channels=int(getattr(cfg, "attn_num_channels", 8)),
            num_attn_blocks=int(getattr(cfg, "attn_num_blocks", 2)),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=decoder_config,
        ).to(device)
        cfg["use_sr"] = True
        cfg["use_img_shape"] = True
    elif requested_model_type == "fcc_autoencoder_sr_block_attn":
        decoder_config = {
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
        }
        core_model = FCCAutoEncoderSRBlockAttn(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            num_channels=int(getattr(cfg, "attn_num_channels", 8)),
            num_attn_blocks=int(getattr(cfg, "attn_num_blocks", 2)),
            block_size=int(getattr(cfg, "attn_block_size", 16)),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=decoder_config,
        ).to(device)
        cfg["use_sr"] = True
        cfg["use_img_shape"] = True
    else:
        raise ValueError(
            f"Unsupported model.type='{requested_model_type}' for train_autoencoder.py. "
            "Supported: fcc_autoencoder, fcc_autoencoder_learnable_decoder, "
            "fcc_autoencoder_with_conv, fcc_autoencoder_sr, fcc_autoencoder_sr_attn, "
            "fcc_autoencoder_sr_global_attn, fcc_autoencoder_sr_double_conv, "
            "fcc_autoencoder_sr_double_conv_attn, fcc_autoencoder_sr_block_attn, "
            "fcc_autoencoder_sr_boundary_guided. "
            "Use scripts/train.sh for other models like invariant_sr."
        )

    model = TrainableFCCAutoEncoder(core_model, decode_chunk_size=decode_chunk_size).to(device)

    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE & PARAMETER BREAKDOWN")
    print("=" * 80)
    print("\nModel Structure:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_param_count:,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    has_trainable_params = len(trainable_params) > 0
    if not has_trainable_params:
        optimizer = None
        scheduler = None
        print("⚠️ Model has no trainable parameters; skipping training loop (evaluation-only mode).")
    else:
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
    loss_fn = build_loss(cfg)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    writer = SummaryWriter(log_dir=run_dir / "runs")
    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))

    trainer = Autoencoder(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        loss_fn=loss_fn,
        writer=writer,
    )

    train_losses, val_losses = [], []

    last_ckpt = Path(cfg.checkpoints_dir) / "last_checkpoint.pt"
    best_ckpt = Path(cfg.checkpoints_dir) / "best_model.pt"
    start_epoch = 0

    if args_cli.resume:
        ckpt_to_load = last_ckpt if last_ckpt.exists() else best_ckpt if best_ckpt.exists() else None
        if ckpt_to_load is not None:
            trainer.load_checkpoint(ckpt_to_load)
            start_epoch = trainer.epoch + 1
            print(f"Resumed from {ckpt_to_load} at epoch {start_epoch}")
        else:
            print("No checkpoint found to resume from; starting from scratch.")

    save_every = int(getattr(cfg, "save_every", 10))
    if has_trainable_params:
        epoch_bar = tqdm(range(start_epoch, cfg.epochs), desc="Autoencoder Epochs", dynamic_ncols=True)
        for epoch in epoch_bar:
            trainer.epoch = epoch
            train_loss = trainer.train()
            val_loss = trainer.validate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            postfix = {
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "lr": f"{current_lr:.2e}",
            }
            if trainer.last_val_metrics:
                postfix["err"] = f"{trainer.last_val_metrics.get('error_mean', 0.0):.4e}"
                postfix["mis°"] = f"{trainer.last_val_metrics.get('mis_deg_mean', 0.0):.3f}"
            epoch_bar.set_postfix(postfix)

            trainer.maybe_save_best(val_loss)
            trainer.save_last_checkpoint()

            if save_every > 0 and (epoch + 1) % save_every == 0:
                viz_dir = run_dir / "visualizations" / f"epoch_{epoch + 1:04d}"
                viz_dir.mkdir(parents=True, exist_ok=True)
                if bool(cfg.get("use_sr", False)):
                    out_png = str(viz_dir / "sr_hr_lr_ipf.png")
                    try:
                        render_sr_output(
                            model=model,
                            val_loader=loaders["test"],
                            sym_class=sym_class,
                            out_png=out_png,
                        )
                        print(f"🖼️ Saved SR/HR/LR IPF render: {out_png}")
                    except Exception as e:
                        print(f"⚠️ Failed to render SR/HR/LR IPF at epoch {epoch + 1}: {repr(e)}")
                else:
                    out_png = str(viz_dir / "input_output_ipf.png")
                    try:
                        render_autoencoder_input_output(
                            model=model,
                            val_loader=loaders["test"],
                            sym_class=sym_class,
                            out_png=out_png,
                        )
                        print(f"🖼️ Saved input/output IPF render: {out_png}")
                    except Exception as e:
                        print(f"⚠️ Failed to render input/output IPF at epoch {epoch + 1}: {repr(e)}")
    else:
        trainer.epoch = start_epoch
        run_eval_only_validate = bool(getattr(cfg, "eval_only_validate", False))
        if run_eval_only_validate:
            val_loss = trainer.validate()
            val_losses.append(val_loss)
            trainer.maybe_save_best(val_loss)
            trainer.save_last_checkpoint()
            print(f"Evaluation-only validation loss: {val_loss:.6f}")
        else:
            print(
                "Skipping full validation in evaluation-only mode "
                "(set eval_only_validate=true in config to enable)."
            )

    print(f"✅ Autoencoder training complete. Outputs saved in: {run_dir}")

    if has_trainable_params and train_losses and val_losses:
        plot_loss(
            train_losses,
            val_losses,
            save_path=str(run_dir / "visualizations" / "loss_plot_autoencoder.png"),
            start_epoch=start_epoch,
        )

    run_postprocess = bool(getattr(cfg, "eval_only_postprocess", False)) if not has_trainable_params else True
    if run_postprocess:
        try:
            stats_loader = loaders["test"] if "test" in loaders else loaders["val"]
            stats_split = "test" if "test" in loaders else "val"
            print_simple_encoder_decoder_stats(model=model, data_loader=stats_loader, split_name=stats_split)
        except Exception as e:
            print(f"⚠️ Failed to compute reconstruction stats: {repr(e)}")

        final_viz_dir = run_dir / "visualizations" / "final"
        final_viz_dir.mkdir(parents=True, exist_ok=True)
        final_png = str(final_viz_dir / "input_output_ipf.png")
        try:
            render_autoencoder_input_output(
                model=model,
                val_loader=loaders["test"],
                sym_class=sym_class,
                out_png=final_png,
            )
            print(f"🖼️ Saved final input/output IPF render: {final_png}")
        except Exception as e:
            print(f"⚠️ Failed to render final input/output IPF: {repr(e)}")
    else:
        print(
            "Skipping eval-only stats and final visualization "
            "(set eval_only_postprocess=true in config to enable)."
        )


if __name__ == "__main__":
    main()
