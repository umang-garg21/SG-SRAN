import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.autoencoder import FCCAutoEncoder
from models.autoencoder_learnable import FCCLearnableDecoderAutoEncoder
from models.invariant_autoencoder_bunge import InvariantAutoencoderBunge
from models.e3nn_invariant_autoencoder import E3nnInvariantAutoencoderBunge, parse_ls_arg
from training.autoencoder_trainer import AutoencoderTrainer
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.loss_functions import build_loss, reduce_to_fz_min_angle_torch_fast
from training.optimizer_utils import build_optimizer
from training.schedulers import build_scheduler
from training.seed_utils import set_seed
from utils.symmetry_utils import resolve_symmetry
from visualization.visualize_sr_results import render_input_output_side_by_side


class TrainableFCCAutoEncoder(nn.Module):
    """Train-only wrapper around the simple FCCAutoEncoder core."""

    def __init__(self, core: nn.Module, decode_chunk_size: int = 128):
        super().__init__()
        self.core = core
        self.decode_chunk_size = int(decode_chunk_size)

    def forward(self, quats: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
        chunks = []
        for start in range(0, quats.shape[0], self.decode_chunk_size):
            end = min(start + self.decode_chunk_size, quats.shape[0])
            chunks.append(self.core(quats[start:end], normalize_input=normalize_input))
        q_decoded = torch.cat(chunks, dim=0)
        norm = torch.norm(q_decoded, dim=-1, keepdim=True).clamp_min(1e-12)
        return q_decoded / norm

    def quat_mul(self, q1: torch.Tensor, q2: torch.Tensor):
        return self.core.quat_mul(q1, q2)

    def canonicalize_for_metrics(self, quats: torch.Tensor) -> torch.Tensor:
        fn = getattr(self.core, "canonicalize_for_metrics", None)
        if callable(fn):
            return fn(quats)
        return quats


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

    _, hr = batch
    if hr is None or hr.shape[0] == 0:
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
    hr0 = hr[0]
    h, w = int(hr0.shape[1]), int(hr0.shape[2])
    device = _model_device(model)
    q_flat = hr0.permute(1, 2, 0).reshape(-1, 4).to(device)

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
        format_input=True,
        align_output_to_input=True,
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

    _, hr = batch
    if hr is None or hr.shape[0] == 0:
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
    hr0 = hr[0]  # (4,H,W)
    q_all = hr0.permute(1, 2, 0).reshape(-1, 4).to(_model_device(model))
    q_all = q_all / torch.norm(q_all, dim=1, keepdim=True).clamp_min(1e-12)
    num_quats = q_all.shape[0]

    batch_size = 1000 if q_all.device.type == "cuda" else 500
    all_errors = []
    all_misorientation_angles = []

    # Build FZ-reduction helper once (uses model's own 24-element O group via
    # fcc_syms, not orix's Oh which is the 48-element full cubic group).
    _core = getattr(model, "core", model)
    _reduce = getattr(_core, "reduce_to_fz", None)

    for batch_start in range(0, num_quats, batch_size):
        batch_end = min(batch_start + batch_size, num_quats)
        q_batch = q_all[batch_start:batch_end]

        q_dec_fz = model(q_batch, normalize_input=True)
        q_orig_fz = _reduce(q_batch)

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
        print(f"\n>> WARNING: {np.sum(all_errors >= 0.05)} quaternion(s) exceeded error threshold of 0.05")
    print("   (Note: Error depends on grid size. Increase grid for more precision.)")


def main():
    args_cli = parse_args()
    exp_dir = Path(args_cli.exp_dir)

    config_path = exp_dir / args_cli.config
    run_config_path = exp_dir / "logs" / "run_config.json"
    cfg = load_and_prepare_config(config_path, run_config_path)

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
            invariant_adapter_enabled=bool(getattr(cfg, "invariant_adapter_enabled", False)),
            invariant_adapter_method=str(getattr(cfg, "invariant_adapter_method", "hybrid")),
            invariant_adapter_beta=float(getattr(cfg, "invariant_adapter_beta", 64.0)),
            invariant_adapter_apply_to=str(getattr(cfg, "invariant_adapter_apply_to", "lr")),
            invariant_adapter_channel_first=bool(
                getattr(cfg, "invariant_adapter_channel_first", True)
            ),
            invariant_adapter_cache=bool(getattr(cfg, "invariant_adapter_cache", False)),
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
    elif requested_model_type == "invariant_autoencoder_bunge":
        core_model = InvariantAutoencoderBunge(
            device=device,
            latent_dim=int(getattr(cfg, "latent_dim", 32)),
            encoder_hidden_dim=int(getattr(cfg, "encoder_hidden_dim", 128)),
            encoder_layers=int(getattr(cfg, "encoder_layers", 3)),
            decoder_hidden_dim=int(getattr(cfg, "decoder_hidden_dim", 128)),
            decoder_layers=int(getattr(cfg, "decoder_layers", 3)),
            dropout=float(getattr(cfg, "decoder_dropout", 0.0)),
            canonicalize_output=bool(getattr(cfg, "canonicalize_output", True)),
        ).to(device)
    elif requested_model_type == "e3nn_invariant_autoencoder_bunge":
        ls_cfg = getattr(cfg, "e3nn_ls", getattr(cfg, "ls", None))
        core_model = E3nnInvariantAutoencoderBunge(
            device=device,
            Ls=parse_ls_arg(ls_cfg),
            stack_re_im=bool(getattr(cfg, "e3nn_stack_re_im", True)),
            normalize_wigner_features=bool(getattr(cfg, "e3nn_normalize_wigner_features", True)),
            basis_rel_tol=float(getattr(cfg, "e3nn_basis_rel_tol", 1e-8)),
            basis_abs_tol=float(getattr(cfg, "e3nn_basis_abs_tol", 1e-6)),
            basis_eig_tol=float(getattr(cfg, "e3nn_basis_eig_tol", 1e-5)),
            canonicalize_basis=bool(getattr(cfg, "e3nn_canonicalize_basis", True)),
            latent_dim=int(getattr(cfg, "latent_dim", 64)),
            encoder_hidden_dim=int(getattr(cfg, "encoder_hidden_dim", 256)),
            encoder_layers=int(getattr(cfg, "encoder_layers", 2)),
            decoder_hidden_dim=int(getattr(cfg, "decoder_hidden_dim", 256)),
            decoder_layers=int(getattr(cfg, "decoder_layers", 3)),
            dropout=float(getattr(cfg, "decoder_dropout", 0.0)),
            canonicalize_output=bool(getattr(cfg, "canonicalize_output", True)),
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
    else:
        raise ValueError(
            f"Unsupported model.type='{requested_model_type}' for train_autoencoder.py. "
            "Supported: fcc_autoencoder, fcc_autoencoder_learnable_decoder, "
            "invariant_autoencoder_bunge, e3nn_invariant_autoencoder_bunge. "
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

    writer = SummaryWriter(log_dir=exp_dir / "runs")
    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))

    trainer = AutoencoderTrainer(
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
                viz_dir = exp_dir / "visualizations" / f"epoch_{epoch + 1:04d}"
                viz_dir.mkdir(parents=True, exist_ok=True)
                out_png = str(viz_dir / "input_output_ipf.png")
                try:
                    render_autoencoder_input_output(
                        model=model,
                        val_loader=loaders["val"],
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

    print(f"✅ Autoencoder training complete. Outputs saved in: {exp_dir}")

    if has_trainable_params and train_losses and val_losses:
        plot_loss(
            train_losses,
            val_losses,
            save_path=str(exp_dir / "visualizations" / "loss_plot_autoencoder.png"),
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

        final_viz_dir = exp_dir / "visualizations" / "final"
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
