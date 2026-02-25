import argparse
import torch
from pathlib import Path

from visualization.visualize_sr_results import (
    render_input_output_side_by_side,
    render_sr_hr_lr_side_by_side,
)

from utils.quat_ops import torch_to_numpy_quat, to_spatial_quat
from training.data_loading import build_dataloader
from models import build_model
from utils.symmetry_utils import resolve_symmetry
from training.config_utils import (
    load_and_prepare_config,
)


def run_postprocess_from_config(
    exp_dir: str,
    max_samples: int | None = 8,
    ckpt_path: str | None = None,
    output_dir: str | None = None,
):
    """
    Post-process trained model results using the resolved run_config.json in exp_dir.

    Parameters
    ----------
    exp_dir : str
        Path to experiment directory containing logs/run_config.json and checkpoints.
    max_samples : int, default=8
        Number of test samples to visualize.
    output_dir : str, optional
        Custom output directory for visualizations. If None, uses exp_dir/visualizations.
    """
    exp_dir = Path(exp_dir)
    config_path = exp_dir / "logs" / "run_config.json"
    # Determine checkpoint to use: prefer passed-in path, then best_model, then last_checkpoint
    if ckpt_path is None:
        best_ckpt = exp_dir / "checkpoints" / "best_model.pt"
        last_ckpt = exp_dir / "checkpoints" / "last_checkpoint.pt"
        if best_ckpt.exists():
            ckpt_path = best_ckpt
        elif last_ckpt.exists():
            ckpt_path = last_ckpt
        else:
            # Nothing to visualize yet; return gracefully (caller wraps this in try/except)
            raise FileNotFoundError(
                f"❌ No checkpoint found in {exp_dir / 'checkpoints'} (searched for best_model.pt and last_checkpoint.pt)"
            )
    else:
        ckpt_path = Path(ckpt_path)

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Missing config: {config_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"❌ Missing checkpoint: {ckpt_path}")

    # --------------------------
    # ✅ Load full resolved config
    # --------------------------
    cfg = load_and_prepare_config(config_path)
    print(f"✅ Loaded config with symmetry group: {getattr(cfg, 'symmetry_group')}")

    max_samples = float("inf") if max_samples is None else int(max_samples)
    take_first = None if max_samples == float("inf") else max_samples

    # --------------------------
    # Build test dataloader
    # --------------------------
    test_loader = build_dataloader(
        dataset_root=getattr(cfg, "dataset_root"),
        split="Test",
        batch_size=getattr(cfg, "batch_size"),
        num_workers=getattr(cfg, "num_workers"),
        preload=getattr(cfg, "preload"),
        preload_torch=getattr(cfg, "preload_torch"),
        pin_memory=getattr(cfg, "pin_memory"),
        take_first=take_first,
    )

    # --------------------------
    # Load trained model
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    # Support both raw state_dicts and full checkpoint dicts produced by Trainer
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    # If loader returned a full checkpoint dictionary, extract the model_state_dict
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state_dict = ckpt_obj["model_state_dict"]
    else:
        state_dict = ckpt_obj

    # Try loading, with a fallback that strips a potential 'module.' prefix
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Attempt to fix common mismatch where checkpoints were saved from nn.DataParallel/DistributedDataParallel
        try:
            new_state = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if isinstance(k, str) else k
                new_state[new_key] = v
            model.load_state_dict(new_state)
            print(
                "Loaded checkpoint after stripping 'module.' prefixes from state_dict keys"
            )
        except Exception:
            # Re-raise original with more context
            raise RuntimeError(f"Failed to load model state_dict from {ckpt_path}: {e}")
    model.eval()
    print(f"✅ Loaded checkpoint from {ckpt_path}")

    # --------------------------
    # Visualization output dir
    # --------------------------
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = exp_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group"))
    render_fz_ipf = bool(getattr(cfg, "render_fz_ipf", False))
    apply_invariant_sr_symmetry_match = bool(getattr(cfg, "postprocess_match_symmetry", True))
    print(f"Using symmetry group: {getattr(cfg, 'symmetry_group')}")

    # --------------------------
    # Inference + render
    # --------------------------
    core_model = model.module if hasattr(model, "module") else model
    is_invariant_sr = str(getattr(cfg, "model_type", "")).lower() == "invariant_sr"

    sample_counter = 0
    for idx, batch in enumerate(test_loader):
        # Robustly extract (lr, hr) from whatever the dataloader yields
        if isinstance(batch, dict):
            # common keys: 'lr'/'input' and 'hr'/'target'
            lr = batch.get("lr") or batch.get("input") or batch.get("low_res") or None
            hr = batch.get("hr") or batch.get("target") or batch.get("high_res") or None
            if lr is None or hr is None:
                # fallback: take first two values
                vals = list(batch.values())
                if len(vals) >= 2:
                    lr, hr = vals[0], vals[1]
                else:
                    raise ValueError(
                        f"Unexpected batch dict keys: {list(batch.keys())}"
                    )
        else:
            try:
                lr, hr = batch
            except Exception:
                # If batch is a tuple/list with >2 items, assume first two are lr/hr
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    lr, hr = batch[0], batch[1]
                else:
                    raise ValueError(
                        f"Unexpected batch format from dataloader: {type(batch)}"
                    )

        # If lr/hr are nested (e.g., (tensor, meta)), unwrap them
        if isinstance(lr, (list, tuple)):
            lr = lr[0]
        if isinstance(hr, (list, tuple)):
            hr = hr[0]

        with torch.no_grad():
            sr = model(lr.to(device, non_blocking=True))

        # Some models return (output, aux) tuples — grab the first element
        if isinstance(sr, (list, tuple)):
            sr = sr[0]

        batch_size = sr.shape[0]
        for b in range(batch_size):
            if sample_counter >= max_samples:
                break

            sr_b = sr[b]
            hr_b = hr[b]

            if (
                is_invariant_sr
                and apply_invariant_sr_symmetry_match
                and hasattr(core_model, "reduce_to_fz")
                and sr_b.shape == hr_b.shape
                and sr_b.dim() == 3
                and sr_b.shape[0] == 4
            ):
                h_sr, w_sr = int(sr_b.shape[1]), int(sr_b.shape[2])
                sr_flat = sr_b.permute(1, 2, 0).reshape(-1, 4)
                sr_flat = core_model.normalize_quaternions(sr_flat)
                sr_matched = core_model.reduce_to_fz(sr_flat)
                sr_b = sr_matched.reshape(h_sr, w_sr, 4).permute(2, 0, 1)

            sr_np = to_spatial_quat(torch_to_numpy_quat(sr_b))
            lr_np = to_spatial_quat(torch_to_numpy_quat(lr[b]))
            hr_np = to_spatial_quat(torch_to_numpy_quat(hr_b))

            out_path = out_dir / f"sr_hr_lr_comparison_{sample_counter:03d}.png"

            render_sr_hr_lr_side_by_side(
                sr_q_arr=sr_np,
                hr_q_arr=hr_np,
                lr_q_arr=lr_np,
                sym_class=sym_class,
                out_png=str(out_path),
                ref_dir="ALL",
                include_key=True,
                overwrite=True,
            )

            # Also render a simpler input vs output (LR -> SR) comparison
            io_out = out_dir / f"input_output_comparison_{sample_counter:03d}.png"
            render_input_output_side_by_side(
                input_q_arr=lr_np,
                output_q_arr=sr_np,
                sym_class=sym_class,
                out_png=str(io_out),
                ref_dir="ALL",
                include_key=True,
                overwrite=True,
            )

            if render_fz_ipf:
                from visualization.unfolded_ipf import fz_ipf_sr_hr_side_by_side

                fz_ipf_sr_hr_side_by_side(
                    sr_np,
                    hr_np,
                    sym_class=getattr(cfg, "symmetry_group"),
                    ref_dir="Z",
                    max_points=5000,
                    out_png=str(out_dir / f"fz_ipf_sr_hr_{sample_counter:03d}.png"),
                )

            print(f"Rendered sample {sample_counter+1} → {out_path}")
            sample_counter += 1

        if sample_counter >= max_samples:
            break

    print(f"\n✅ Saved {sample_counter} visualization(s) to: {out_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process trained SR model outputs into IPF visualizations"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Experiment directory containing logs/run_config.json and checkpoints",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=8,
        help="Number of test samples to render (default: 8)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional checkpoint path override",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory override",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_postprocess_from_config(
        exp_dir=args.exp_dir,
        max_samples=args.max_samples,
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()