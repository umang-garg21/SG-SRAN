"""
test_inference.py
=================
Run SR inference on the test split using a saved checkpoint and save IPF maps.

Usage
-----
  python test_inference.py --checkpoint <path/to/best_model.pt> [options]

The script automatically resolves:
  * experiment dir  = parent of the run dir  (contains config.json)
  * run dir         = parent of 'checkpoints/'  (receives 'test/' output folder)

Outputs are written to:
  <run_dir>/test/
      sample_<NNNN>_ipf.png   – LR / SR / HR IPF map for each test sample
      metrics_summary.csv     – per-sample misorientation statistics
      metrics_overall.txt     – aggregate stats (mean, median, max, …)

Examples
--------
  # Evaluate specific checkpoint, default GPU
  python test_inference.py --checkpoint experiments/IN718/LAE_sr_double_conv_attn_01/2026-02-27_11-36-07/checkpoints/best_model.pt

  # Limit to first 10 test samples on GPU 1
  python test_inference.py \\
    --checkpoint .../best_model.pt \\
    --gpu_ids 1 \\
    --max_samples 10
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so relative imports work regardless
# of the working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from models.autoencoder import FCCAutoEncoder
from models.autoencoder_learnable import FCCLearnableDecoderAutoEncoder
from models.SR_conv import FCCAutoEncoder as FCCAutoEncoderWithConv
from models.SR_conv import FCCAutoEncoderSR
from models.SR_global_attn import FCCAutoEncoderSRGlobalAttn
from models.SR_grain_attn import FCCAutoEncoderSRBlockAttn
from models.SR_double_conv import FCCAutoEncoderSR as FCCAutoEncoderSRDoubleConv
from models.SR_double_conv_SRattn import FCCAutoEncoderSRDoubleConvAttn
from models.SR_boundary_guided import FCCAutoEncoderSRBoundaryGuided
from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.loss_functions import reduce_to_fz_min_angle_torch_fast
from utils.quat_ops import format_quaternions
from utils.symmetry_utils import resolve_symmetry
from visualization.ipf_render import render_ipf_rgb
from visualization.visualize_sr_results import render_sr_hr_lr_side_by_side


# ---------------------------------------------------------------------------
# TrainableFCCAutoEncoder wrapper  (identical to training/train_autoencoder.py)
# ---------------------------------------------------------------------------

class TrainableFCCAutoEncoder(nn.Module):
    """Thin wrapper that chunks non-conv forward passes for memory efficiency."""

    def __init__(self, core: nn.Module, decode_chunk_size: int = 128):
        super().__init__()
        self.core = core
        self.decode_chunk_size = int(decode_chunk_size)

    def forward(self, quats, img_shape=None, normalize_input=True):
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _fz_reduce(model: nn.Module, q_flat: torch.Tensor) -> torch.Tensor:
    """FZ-reduce (N,4) quaternions using the model's reduce_to_fz or fallback."""
    _core = getattr(model, "core", model)
    _reduce = getattr(_core, "reduce_to_fz", None)
    if _reduce is not None:
        return _reduce(q_flat)
    q_bchw = q_flat.T.unsqueeze(0).unsqueeze(-1)
    q_fz = reduce_to_fz_min_angle_torch_fast(q_bchw, "O")
    return q_fz.squeeze(0).squeeze(-1).T


def sym_misorientation_degrees(
    q_sr: torch.Tensor,
    q_hr: torch.Tensor,
    sym_ops: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Per-pixel misorientation (disorientation) angle in degrees, Bunge-passive convention.

    Algorithm (matches codebase FZ-reduction convention):
      1. Relative rotation (Bunge passive):
             Δg = g_sr⁻¹ ⊗ g_hr  =  conj(q_sr) ⊗ q_hr
      2. Left-multiply by s⁻¹ = conj(s) for each crystal-symmetry operator s:
             s⁻¹ ⊗ Δg
         Scalar part of s⁻¹ ⊗ Δg  =  s·Δg  (4-D dot product),
         because conj(s) negates the vector part and turns the normal
         quaternion-product signs into a straight dot product.
      3. Disorientation angle = 2·acos( max_s |s⁻¹ ⊗ Δg|_w ).

    Parameters
    ----------
    q_sr, q_hr : (N, 4) tensors, scalar-first (w,x,y,z), unit quaternions.
    sym_ops    : (M, 4) tensor of crystal-symmetry group quaternions.
    """
    q_sr = torch.nn.functional.normalize(q_sr, dim=-1, eps=eps)
    q_hr = torch.nn.functional.normalize(q_hr, dim=-1, eps=eps)

    # Step 1 — relative rotation Δg = conj(q_sr) ⊗ q_hr  (Bunge passive)
    w1, x1, y1, z1 = q_sr[:, 0],  q_sr[:, 1],  q_sr[:, 2],  q_sr[:, 3]
    w2, x2, y2, z2 = q_hr[:, 0],  q_hr[:, 1],  q_hr[:, 2],  q_hr[:, 3]
    # conj(q_sr) ⊗ q_hr:
    dw = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    dx = w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2
    dy = w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2
    dz = w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2

    # (N, 1) for broadcasting against (1, M)
    dw = dw.unsqueeze(1); dx = dx.unsqueeze(1)
    dy = dy.unsqueeze(1); dz = dz.unsqueeze(1)

    # Step 2 — scalar part of s⁻¹ ⊗ Δg  =  dot(s, Δg)
    sw = sym_ops[:, 0].unsqueeze(0)   # (1, M)
    sx = sym_ops[:, 1].unsqueeze(0)
    sy = sym_ops[:, 2].unsqueeze(0)
    sz = sym_ops[:, 3].unsqueeze(0)

    w_sym = sw * dw + sx * dx + sy * dy + sz * dz   # (N, M)

    # Step 3 — maximum |w| over all symmetry equivalents → minimum angle
    max_w, _ = torch.max(w_sym.abs(), dim=1)   # (N,)
    max_w = max_w.clamp(max=1.0 - eps)
    return 2.0 * torch.acos(max_w) * (180.0 / torch.pi)


class RunningMetrics:
    """Welford-style running mean for scalar metrics."""

    def __init__(self, *keys: str):
        self._keys  = list(keys)
        self._sum   = {k: 0.0 for k in keys}
        self._count = 0

    def update(self, **kwargs: float) -> None:
        self._count += 1
        for k, v in kwargs.items():
            self._sum[k] += v

    @property
    def count(self) -> int:
        return self._count

    def mean(self, key: str) -> float:
        if self._count == 0:
            return float("nan")
        return self._sum[key] / self._count

    def postfix(self) -> dict:
        """Return a tqdm-friendly dict of running means."""
        return {k: f"{self.mean(k):.4f}" for k in self._keys}


# ---------------------------------------------------------------------------
# Model factory  (mirrors train_autoencoder.py exactly)
# ---------------------------------------------------------------------------

def build_model(cfg, device: torch.device) -> tuple[TrainableFCCAutoEncoder, bool]:
    """
    Build and return (model, use_sr) from a resolved config namespace.
    use_sr=True means the model expects (LR, HR) paired data.
    """
    model_cfg = getattr(cfg, "model", None)
    requested_model_type = str(
        getattr(model_cfg, "type", getattr(cfg, "model_type", "fcc_autoencoder"))
    ).lower()

    grid_res = int(getattr(cfg, "grid_res", 2048))
    decode_chunk_size = int(getattr(cfg, "decode_chunk_size", 128))

    def _decoder_config(extra=None):
        base = {
            "decoder_lookup_resolution": int(getattr(cfg, "decoder_lookup_resolution", 3)),
            "decoder_lookup_chunk_size": int(getattr(cfg, "decoder_lookup_chunk_size", 8192)),
            "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
            "decoder_lookup_rebuild": bool(getattr(cfg, "decoder_lookup_rebuild", False)),
            "decoder_lookup_refine_steps": int(getattr(cfg, "decoder_lookup_refine_steps", 0)),
            "decoder_lookup_refine_lr": float(getattr(cfg, "decoder_lookup_refine_lr", 0.05)),
            "decoder_w6": float(getattr(cfg, "decoder_w6", 0.5)),
        }
        if extra:
            base.update(extra)
        return base

    def _opt_int(key):
        v = getattr(cfg, key, None)
        return int(v) if v is not None else None

    use_sr = False

    if requested_model_type == "fcc_autoencoder_learnable_decoder":
        core = FCCLearnableDecoderAutoEncoder(
            device=device,
            hidden_dim=int(getattr(cfg, "decoder_hidden_dim", 128)),
            num_layers=int(getattr(cfg, "decoder_num_layers", 3)),
            dropout=float(getattr(cfg, "decoder_dropout", 0.0)),
        ).to(device)

    elif requested_model_type == "fcc_autoencoder":
        dc = _decoder_config({
            "decoder_cubochoric_resolution": int(getattr(cfg, "decoder_cubochoric_resolution", 3)),
            "decoder_num_starts": int(getattr(cfg, "decoder_num_starts", 6)),
            "decoder_steps": int(getattr(cfg, "decoder_steps", 25)),
            "decoder_lr": float(getattr(cfg, "decoder_lr", 0.08)),
            "decoder_early_stop_tol": float(getattr(cfg, "decoder_early_stop_tol", 1e-6)),
            "decoder_early_stop_patience": int(getattr(cfg, "decoder_early_stop_patience", 3)),
            "decoder_min_steps": int(getattr(cfg, "decoder_min_steps", 6)),
            "decoder_log_optimization": bool(getattr(cfg, "decoder_log_optimization", False)),
            "decoder_log_every": int(getattr(cfg, "decoder_log_every", 1)),
            "decoder_learnable_hidden_dim": int(getattr(cfg, "decoder_learnable_hidden_dim", 256)),
            "decoder_learnable_num_layers": int(getattr(cfg, "decoder_learnable_num_layers", 3)),
            "decoder_learnable_dropout": float(getattr(cfg, "decoder_learnable_dropout", 0.0)),
            "decoder_learnable_ckpt_path": getattr(cfg, "decoder_learnable_ckpt_path", None),
            "decoder_learnable_ckpt_strict": bool(getattr(cfg, "decoder_learnable_ckpt_strict", True)),
        })
        core = FCCAutoEncoder(
            device=device,
            grid_res=grid_res,
            decoder_backend=str(getattr(cfg, "decoder_backend", "optimizing")),
            decoder_config=dc,
        ).to(device)

    elif requested_model_type == "fcc_autoencoder_with_conv":
        core = FCCAutoEncoderWithConv(
            device=device,
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=_decoder_config(),
        ).to(device)

    elif requested_model_type == "fcc_autoencoder_sr":
        core = FCCAutoEncoderSR(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            upsample_residual=bool(getattr(cfg, "upsample_residual", False)),
            upsampler="conv",
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=_decoder_config(),
        ).to(device)
        use_sr = True

    elif requested_model_type == "fcc_autoencoder_sr_attn":
        core = FCCAutoEncoderSR(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            upsampler="attention",
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=_decoder_config(),
        ).to(device)
        use_sr = True

    elif requested_model_type == "fcc_autoencoder_sr_double_conv":
        _lr_shape_raw = getattr(cfg, "lr_shape", None)
        _lr_shape = tuple(int(x) for x in _lr_shape_raw) if _lr_shape_raw is not None else None
        core = FCCAutoEncoderSRDoubleConv(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            upsample_residual=bool(getattr(cfg, "upsample_residual", False)),
            upsampler="conv",
            lr_shape=_lr_shape,
            lr_conv_kernel_size=_opt_int("lr_conv_kernel_size"),
            lr_conv_kernel_size_1=_opt_int("lr_conv_kernel_size_1"),
            lr_conv_kernel_size_2=_opt_int("lr_conv_kernel_size_2"),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=_decoder_config(),
        ).to(device)
        use_sr = True

    elif requested_model_type == "fcc_autoencoder_sr_double_conv_attn":
        _lr_shape_raw = getattr(cfg, "lr_shape", None)
        _lr_shape = tuple(int(x) for x in _lr_shape_raw) if _lr_shape_raw is not None else None
        core = FCCAutoEncoderSRDoubleConvAttn(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            upsample_residual=bool(getattr(cfg, "upsample_residual", False)),
            upsampler="conv",
            lr_shape=_lr_shape,
            lr_conv_kernel_size=_opt_int("lr_conv_kernel_size"),
            lr_conv_kernel_size_1=_opt_int("lr_conv_kernel_size_1"),
            lr_conv_kernel_size_2=_opt_int("lr_conv_kernel_size_2"),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=_decoder_config(),
            num_hr_attn_blocks=int(getattr(cfg, "hr_attn_num_blocks", 2)),
            hr_attn_num_channels=int(getattr(cfg, "hr_attn_num_channels", 8)),
            hr_attn_block_size=int(getattr(cfg, "hr_attn_block_size", 16)),
        ).to(device)
        use_sr = True

    elif requested_model_type == "fcc_autoencoder_sr_global_attn":
        core = FCCAutoEncoderSRGlobalAttn(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            num_channels=int(getattr(cfg, "attn_num_channels", 8)),
            num_attn_blocks=int(getattr(cfg, "attn_num_blocks", 2)),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=_decoder_config(),
        ).to(device)
        use_sr = True

    elif requested_model_type == "fcc_autoencoder_sr_block_attn":
        core = FCCAutoEncoderSRBlockAttn(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            num_channels=int(getattr(cfg, "attn_num_channels", 8)),
            num_attn_blocks=int(getattr(cfg, "attn_num_blocks", 2)),
            block_size=int(getattr(cfg, "attn_block_size", 16)),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=_decoder_config(),
        ).to(device)
        use_sr = True

    elif requested_model_type == "fcc_autoencoder_sr_boundary_guided":
        core = FCCAutoEncoderSRBoundaryGuided(
            device=device,
            upsample_factor=int(getattr(cfg, "scale", 4)),
            lr_conv_kernel_size_1=getattr(cfg, "lr_conv_kernel_size_1", None),
            lr_conv_kernel_size_2=getattr(cfg, "lr_conv_kernel_size_2", None),
            bg_window_size=int(getattr(cfg, "bg_window_size", 3)),
            bg_init_sigma=float(getattr(cfg, "bg_init_sigma", 0.5)),
            bg_init_lambda=float(getattr(cfg, "bg_init_lambda", 2.0)),
            bg_init_gamma=float(getattr(cfg, "bg_init_gamma", 1.0)),
            hr_attn_num_blocks=int(getattr(cfg, "hr_attn_num_blocks", 2)),
            hr_attn_num_channels=int(getattr(cfg, "hr_attn_num_channels", 8)),
            hr_attn_block_size=int(getattr(cfg, "hr_attn_block_size", 16)),
            decoder_backend=str(getattr(cfg, "decoder_backend", "lookup")),
            decoder_config=_decoder_config(),
        ).to(device)
        use_sr = True

    else:
        raise ValueError(
            f"Unsupported model.type='{requested_model_type}'. "
            "Check config.json in the experiment directory."
        )

    model = TrainableFCCAutoEncoder(core, decode_chunk_size=decode_chunk_size).to(device)
    return model, use_sr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run test-set inference from a saved checkpoint and save IPF maps."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help=(
            "Path to checkpoint file (e.g. .../2026-02-27_11-36-07/checkpoints/best_model.pt). "
            "The experiment config.json is auto-resolved as ../../config.json relative to this file."
        ),
    )
    p.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override, e.g. '0' or '0,1'.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of test samples to process (default: all).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved IPF maps (default: 300).",
    )
    p.add_argument(
        "--ref_dir",
        type=str,
        default="ALL",
        choices=["X", "Y", "Z", "ALL"],
        help="IPF reference direction(s) (default: ALL → X, Y, Z panels).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for test dataloader (default: 1).",
    )
    p.add_argument(
        "--refine_steps",
        type=int,
        default=None,
        help="Override decoder_lookup_refine_steps (gradient-refinement iters per pixel, default: 0).",
    )
    p.add_argument(
        "--refine_lr",
        type=float,
        default=None,
        help="Override decoder_lookup_refine_lr (default: 0.05).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve directories from checkpoint path
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Expected layout: <exp_dir>/<run_dir>/checkpoints/best_model.pt
    checkpoints_dir = ckpt_path.parent       # …/checkpoints/
    run_dir         = checkpoints_dir.parent  # …/2026-02-27_11-36-07/
    exp_dir         = run_dir.parent          # …/LAE_sr_double_conv_attn_01/

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found at expected location: {config_path}\n"
            "Make sure --checkpoint points to a file two levels below the experiment dir."
        )

    out_dir = run_dir / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment dir : {exp_dir}")
    print(f"Run dir        : {run_dir}")
    print(f"Checkpoint     : {ckpt_path}")
    print(f"Output dir     : {out_dir}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"CUDA_VISIBLE_DEVICES set to: {args.gpu_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device   : {device}")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = load_and_prepare_config(config_path)
    cfg["device"] = str(device)
    if args.refine_steps is not None:
        cfg["decoder_lookup_refine_steps"] = args.refine_steps
    if args.refine_lr is not None:
        cfg["decoder_lookup_refine_lr"] = args.refine_lr

    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group", "O"))

    # Pre-build symmetry operator tensor (kept on CPU; moved to device in loop)
    _sym_ops_np = sym_class.data if isinstance(sym_class.data, np.ndarray) else np.array(sym_class.data)
    sym_ops_cpu = torch.tensor(_sym_ops_np, dtype=torch.float32)  # (M, 4)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nBuilding model ...")
    model, use_sr = build_model(cfg, device)

    # Load checkpoint weights (model weights only; no optimizer needed)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    epoch = int(ckpt.get("epoch", -1))
    best_val_loss = float(ckpt.get("best_val_loss", float("nan")))
    print(f"Loaded checkpoint from epoch {epoch}  (best_val_loss={best_val_loss:.6e})")

    model.eval()
    _core = getattr(model, "core", model)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nBuilding test dataloader ...")
    test_loader = build_dataloader(
        dataset_root=cfg.dataset_root,
        split="Test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(getattr(cfg, "num_workers", 0)),
        preload=bool(getattr(cfg, "preload", True)),
        preload_torch=bool(getattr(cfg, "preload_torch", True)),
        pin_memory=bool(getattr(cfg, "pin_memory", True)),
        take_first=args.max_samples,
        seed=int(getattr(cfg, "seed", 42)),
    )
    print(f"Test set size  : {len(test_loader.dataset)} samples")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    per_sample_stats = []   # list of dicts for CSV
    all_mis_deg = []        # flat list of all pixel misorientations
    mis_maps    = []        # per-sample misorientation maps (H_hr, W_hr)

    running = RunningMetrics("mis_loss_rad", "mis_mean_deg", "mse_ipf", "psnr_ipf", "ssim_ipf")

    # Open CSV for streaming writes
    csv_path = out_dir / "metrics.csv"
    fieldnames = [
        "sample_idx", "mis_loss_rad",
        "mis_mean_deg", "mis_median_deg", "mis_max_deg", "mis_std_deg",
        "mse_ipf", "psnr_ipf", "ssim_ipf",
        "h_lr", "w_lr", "h_hr", "w_hr",
    ]
    write_header = not csv_path.exists()
    _csv_file   = open(csv_path, "a", newline="")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=fieldnames)
    if write_header:
        _csv_writer.writeheader()

    sample_idx = 0

    pbar = tqdm(test_loader, desc="Test inference", dynamic_ncols=True)
    for batch in pbar:
        if use_sr:
            lr_batch, hr_batch = batch
        else:
            # Non-SR models receive (hr, lr) but we only need hr=lr for reconstruction
            hr_batch, lr_batch = batch

        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        b, _, h_lr, w_lr = lr_batch.shape
        _, _, h_hr, w_hr = hr_batch.shape

        for img_i in range(b):
            lr0 = lr_batch[img_i]   # (4, h_lr, w_lr)
            hr0 = hr_batch[img_i]   # (4, h_hr, w_hr)

            # Flatten & normalize LR input
            lr_flat = lr0.permute(1, 2, 0).reshape(-1, 4)
            lr_flat = lr_flat / lr_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)

            # Run SR or reconstruction
            if use_sr:
                q_sr = _core.forward_sr(lr_flat, lr_shape=(h_lr, w_lr))
            else:
                q_sr = model(lr_flat, img_shape=(h_lr, w_lr), normalize_input=True)

            # FZ-reduce SR output and HR ground truth for metric computation
            hr_flat  = hr0.permute(1, 2, 0).reshape(-1, 4)

            sym_ops = sym_ops_cpu.to(device)
            mis_deg = sym_misorientation_degrees(q_sr, hr_flat.to(device), sym_ops)
            mis_np  = mis_deg.cpu().numpy()

            # Symmetry-aware misorientation loss in radians
            mis_loss_rad = float((mis_deg * (torch.pi / 180.0)).mean().item())

            # Build numpy arrays for IPF rendering
            q_lr_np = lr_flat.reshape(h_lr, w_lr, 4).cpu().numpy()
            q_sr_np = q_sr.reshape(h_hr, w_hr, 4).detach().cpu().numpy()
            q_hr_np = hr0.permute(1, 2, 0).reshape(h_hr, w_hr, 4).cpu().numpy()

            # ------------------------------------------------------------------
            # Compute SSIM and MSE on IPF RGB maps (Z direction, or first dir)
            # ------------------------------------------------------------------
            def _fmt(arr):
                return format_quaternions(
                    arr, normalize=True, hemisphere=True, reduce_fz=True,
                    sym=sym_class, quat_first=False,
                )

            sr_rgb_all = render_ipf_rgb(_fmt(q_sr_np), sym_class, ref_dir=args.ref_dir)
            hr_rgb_all = render_ipf_rgb(_fmt(q_hr_np), sym_class, ref_dir=args.ref_dir)

            # Pick one channel for scalar SSIM/MSE: use Z (index 2) if ALL, else the single array
            if isinstance(sr_rgb_all, list):
                sr_rgb_ref = sr_rgb_all[2].astype(np.float32)   # Z direction
                hr_rgb_ref = hr_rgb_all[2].astype(np.float32)
            else:
                sr_rgb_ref = sr_rgb_all.astype(np.float32)
                hr_rgb_ref = hr_rgb_all.astype(np.float32)

            mse_ipf  = float(np.mean((sr_rgb_ref - hr_rgb_ref) ** 2))
            psnr_ipf = float(10.0 * np.log10(1.0 / mse_ipf) if mse_ipf > 0 else float("inf"))
            ssim_ipf = float(
                structural_similarity(
                    sr_rgb_ref, hr_rgb_ref,
                    channel_axis=-1,
                    data_range=1.0,
                )
            )

            stats = {
                "sample_idx":     sample_idx,
                "mis_loss_rad":   mis_loss_rad,
                "mis_mean_deg":   float(mis_np.mean()),
                "mis_median_deg": float(np.median(mis_np)),
                "mis_max_deg":    float(mis_np.max()),
                "mis_std_deg":    float(mis_np.std()),
                "mse_ipf":        mse_ipf,
                "psnr_ipf":       psnr_ipf,
                "ssim_ipf":       ssim_ipf,
                "h_lr": h_lr,
                "w_lr": w_lr,
                "h_hr": h_hr,
                "w_hr": w_hr,
            }
            per_sample_stats.append(stats)
            all_mis_deg.extend(mis_np.tolist())
            mis_maps.append(mis_np.reshape(h_hr, w_hr))

            # Update running metrics and flush row to CSV immediately
            running.update(
                mis_loss_rad=mis_loss_rad,
                mis_mean_deg=float(mis_np.mean()),
                mse_ipf=mse_ipf,
                psnr_ipf=psnr_ipf,
                ssim_ipf=ssim_ipf,
            )
            _csv_writer.writerow(stats)
            _csv_file.flush()

            # Save IPF map
            out_png = str(out_dir / f"sample_{sample_idx:04d}_ipf.png")
            render_sr_hr_lr_side_by_side(
                sr_q_arr=q_sr_np,
                hr_q_arr=q_hr_np,
                lr_q_arr=q_lr_np,
                sym_class=sym_class,
                out_png=out_png,
                ref_dir=args.ref_dir,
                include_key=True,
                overwrite=True,
                format_input=True,
                dpi=args.dpi,
                mis_deg_arr=mis_np.reshape(h_hr, w_hr),
            )

            pbar.set_postfix(
                n=running.count,
                mis_loss=f"{running.mean('mis_loss_rad'):.4f}rad",
                mis_deg=f"{running.mean('mis_mean_deg'):.3f}°",
                psnr=f"{running.mean('psnr_ipf'):.2f}dB",
                ssim=f"{running.mean('ssim_ipf'):.4f}",
            )
            sample_idx += 1

            if args.max_samples is not None and sample_idx >= args.max_samples:
                break

        if args.max_samples is not None and sample_idx >= args.max_samples:
            break

    _csv_file.close()
    print(f"\nMetrics CSV saved to: {csv_path}")

    # ------------------------------------------------------------------
    # Save per-pixel misorientation maps
    # ------------------------------------------------------------------
    npy_path = out_dir / "misorientation_test.npy"
    try:
        mis_arr = np.stack(mis_maps, axis=0)   # (N, H_hr, W_hr) if all same shape
    except ValueError:
        mis_arr = np.array(mis_maps, dtype=object)  # ragged: object array of (H,W) maps
    np.save(str(npy_path), mis_arr)
    print(f"Misorientation maps saved to: {npy_path}  shape={mis_arr.shape}")

    # ------------------------------------------------------------------
    # Misorientation histogram over all samples
    # ------------------------------------------------------------------
    all_mis_flat = np.array(all_mis_deg)
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(
        all_mis_flat,
        bins=180,
        range=(0.0, float(np.ceil(all_mis_flat.max()))),
        color="steelblue",
        edgecolor="none",
        density=True,
    )
    ax_hist.axvline(all_mis_flat.mean(),  color="red",    lw=1.5, linestyle="--",
                    label=f"mean={all_mis_flat.mean():.2f}°")
    ax_hist.axvline(np.median(all_mis_flat), color="orange", lw=1.5, linestyle=":",
                    label=f"median={np.median(all_mis_flat):.2f}°")
    ax_hist.set_xlabel("Misorientation angle (°)", fontsize=12)
    ax_hist.set_ylabel("Density", fontsize=12)
    ax_hist.set_title(
        f"SR→HR Misorientation histogram  ({sample_idx} samples, "
        f"{len(all_mis_flat):,} pixels)",
        fontsize=12,
    )
    ax_hist.legend(fontsize=10)
    fig_hist.tight_layout()
    hist_png = out_dir / "misorientation_histogram.png"
    fig_hist.savefig(str(hist_png), dpi=200, bbox_inches="tight")
    plt.close(fig_hist)
    print(f"Misorientation histogram saved to: {hist_png}")

    # ------------------------------------------------------------------
    # Overall stats  (built from running accumulators + full pixel list)
    # ------------------------------------------------------------------
    all_mis       = np.array(all_mis_deg)
    all_mis_loss  = np.array([s["mis_loss_rad"] for s in per_sample_stats])
    all_mse       = np.array([s["mse_ipf"]      for s in per_sample_stats])
    all_psnr      = np.array([s["psnr_ipf"]     for s in per_sample_stats])
    all_ssim      = np.array([s["ssim_ipf"]     for s in per_sample_stats])
    summary_lines = [
        "=" * 70,
        f"TEST SET STATISTICS  ({sample_idx} samples)",
        f"Checkpoint : {ckpt_path}",
        f"Epoch      : {epoch}   best_val_loss={best_val_loss:.6e}",
        "=" * 70,
        "Misorientation loss (radians, training objective):",
        f"  Mean    : {all_mis_loss.mean():.6f} rad",
        f"  Std     : {all_mis_loss.std():.6f} rad",
        "Misorientation (degrees):",
        f"  Mean    : {all_mis.mean():.4f}°",
        f"  Median  : {np.median(all_mis):.4f}°",
        f"  Max     : {all_mis.max():.4f}°",
        f"  Std     : {all_mis.std():.4f}°",
        f"  p95     : {np.percentile(all_mis, 95):.4f}°",
        "IPF RGB MSE (Z-direction):",
        f"  Mean    : {all_mse.mean():.6f}",
        f"  Std     : {all_mse.std():.6f}",
        "IPF RGB PSNR (Z-direction, dB):",
        f"  Mean    : {all_psnr[np.isfinite(all_psnr)].mean():.4f} dB",
        f"  Std     : {all_psnr[np.isfinite(all_psnr)].std():.4f} dB",
        "IPF RGB SSIM (Z-direction):",
        f"  Mean    : {all_ssim.mean():.6f}",
        f"  Std     : {all_ssim.std():.6f}",
        "=" * 70,
    ]
    summary_str = "\n".join(summary_lines)
    print("\n" + summary_str)

    txt_path = out_dir / "metrics_overall.txt"
    txt_path.write_text(summary_str + "\n", encoding="utf-8")
    print(f"Overall stats saved to: {txt_path}")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
