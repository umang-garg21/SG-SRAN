import torch
from pathlib import Path

from visualization.visualize_sr_results import (
    render_sr_hr_side_by_side,
    render_sr_hr_lr_side_by_side,
)
from visualization.unfolded_ipf import fz_ipf_sr_hr_side_by_side  # plot_sr_hr_fz_ipf

from utils.quat_ops import torch_to_numpy_quat, to_spatial_quat
from training.data_loading import build_dataloader
from models import build_model
from utils.symmetry_utils import resolve_symmetry
from training.config_utils import (
    load_and_prepare_config,
)


def run_postprocess_from_config(exp_dir: str, max_samples: int | None = 8):
    """
    Post-process trained model results using the resolved run_config.json in exp_dir.

    Parameters
    ----------
    exp_dir : str
        Path to experiment directory containing logs/run_config.json and checkpoints.
    max_samples : int, default=8
        Number of test samples to visualize.
    """
    exp_dir = Path(exp_dir)
    config_path = exp_dir / "logs" / "run_config.json"
    ckpt_path = exp_dir / "checkpoints" / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Missing config: {config_path}")
    if not ckpt_path.exists():
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
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✅ Loaded checkpoint from {ckpt_path}")

    # --------------------------
    # Visualization output dir
    # --------------------------
    out_dir = exp_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    sym_class = resolve_symmetry(getattr(cfg, "symmetry_group"))
    print(f"Using symmetry group: {getattr(cfg, 'symmetry_group')}")

    # --------------------------
    # Inference + render
    # --------------------------
    sample_counter = 0
    for idx, (lr, hr) in enumerate(test_loader):
        with torch.no_grad():
            sr = model(lr.to(device, non_blocking=True))

        batch_size = sr.shape[0]
        for b in range(batch_size):
            if sample_counter >= max_samples:
                break

            sr_np = to_spatial_quat(torch_to_numpy_quat(sr[b]))
            lr_np = to_spatial_quat(torch_to_numpy_quat(lr[b]))
            hr_np = to_spatial_quat(torch_to_numpy_quat(hr[b]))

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


# # -*- coding:utf-8 -*-
# """
# File:        run_postprocess_from_config.py
# Author:      Warren Zamudio
# Description: Generate SR vs HR IPF visualizations after training.
# """

# import torch
# from pathlib import Path

# from visualization.visualize_sr_results import (
#     render_sr_hr_side_by_side,
#     render_sr_hr_lr_side_by_side,
# )
# from visualization.unfolded_ipf import fz_ipf_render  # plot_sr_hr_fz_ipf

# from utils.quat_ops import torch_to_numpy_quat, to_spatial_quat
# from training.data_loading import build_dataloader
# from models import build_model
# from utils.symmetry_utils import resolve_symmetry
# from training.config_utils import (
#     load_and_prepare_config,
# )


# def run_postprocess_from_config(exp_dir: str, max_samples: int = 8):
#     """
#     Post-process trained model results using the resolved run_config.json in exp_dir.

#     Parameters
#     ----------
#     exp_dir : str
#         Path to experiment directory containing logs/run_config.json and checkpoints.
#     max_samples : int, default=8
#         Number of test samples to visualize.
#     """
#     exp_dir = Path(exp_dir)
#     config_path = exp_dir / "logs" / "run_config.json"
#     ckpt_path = exp_dir / "checkpoints" / "best_model.pt"

#     if not config_path.exists():
#         raise FileNotFoundError(f"❌ Missing config: {config_path}")
#     if not ckpt_path.exists():
#         raise FileNotFoundError(f"❌ Missing checkpoint: {ckpt_path}")

#     # --------------------------
#     # ✅ Load full resolved config (symmetry handled automatically)
#     # --------------------------
#     cfg = load_and_prepare_config(config_path)
#     print(f"✅ Loaded config with symmetry group: {cfg['symmetry_group']}")

#     # --------------------------
#     # Build test dataloader
#     # --------------------------
#     test_loader = build_dataloader(
#         dataset_root=cfg["dataset_root"],
#         split="Test",
#         batch_size=cfg["batch_size"],
#         num_workers=cfg["num_workers"],
#         preload=cfg["preload"],
#         preload_torch=cfg["preload_torch"],
#         pin_memory=cfg["pin_memory"],
#         take_first=max_samples if cfg.get("smoke_test", False) else None,
#     )

#     # --------------------------
#     # Load trained model
#     # --------------------------
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = build_model(cfg).to(device)
#     model.load_state_dict(torch.load(ckpt_path, map_location=device))
#     model.eval()
#     print(f"Loaded checkpoint from {ckpt_path}")

#     # --------------------------
#     # Set up visualization output
#     # --------------------------
#     out_dir = exp_dir / "visualizations"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Use symmetry group from cfg
#     sym_class = resolve_symmetry(cfg["symmetry_group"])
#     print(f"Using symmetry group: {cfg['symmetry_group']}")

#     # --------------------------
#     # Run inference + render
#     # --------------------------
#     sample_counter = 0
#     for idx, (lr, hr) in enumerate(test_loader):
#         with torch.no_grad():
#             sr = model(lr.to(device, non_blocking=True))

#         # sr_np_batch and hr_np_batch are (B, H, W, 4)
#         batch_size = sr.shape[0]

#         for b in range(batch_size):
#             if sample_counter >= max_samples:
#                 break

#             sr_np = to_spatial_quat(torch_to_numpy_quat(sr[b]))
#             lr_np = to_spatial_quat(torch_to_numpy_quat(lr[b]))
#             hr_np = to_spatial_quat(torch_to_numpy_quat(hr[b]))

#             out_path = out_dir / f"sr_hr_lr_comparison_{sample_counter:03d}.png"

#             render_sr_hr_lr_side_by_side(
#                 sr_q_arr=sr_np,
#                 hr_q_arr=hr_np,
#                 lr_q_arr=lr_np,
#                 sym_class=sym_class,
#                 out_png=str(out_path),
#                 ref_dir="ALL",
#                 include_key=True,
#                 overwrite=True,
#             )
#             fz_ipf_render(
#                 sr_np,
#                 hr_np,
#                 sym_class=cfg["symmetry_group"],
#                 ref_dir="Z",
#                 max_points=5000,
#                 out_png=str(
#                     exp_dir
#                     / "visualizations"
#                     / f"fz_ipf_sr_hr_{sample_counter:03d}.png"
#                 ),
#             )
#             # plot_sr_hr_fz_ipf(
#             #     sr_quat=sr_np,
#             #     hr_quat=hr_np,
#             #     sym_class=sym_class,
#             #     ref_dir="Z",
#             #     max_points=5000,
#             #     out_dir=exp_dir / "visualizations" / "fz_ipf",
#             #     prefix=f"sample_{sample_counter:03d}",
#             # )
#             # render_sr_hr_side_by_side(
#             #     sr_np,
#             #     hr_np,
#             #     sym_class=sym_class,
#             #     out_png=str(out_path),
#             #     ref_dir="ALL",
#             #     include_key=True,
#             #     overwrite=True,
#             # )
#             print(f"Rendered sample {sample_counter+1} → {out_path}")
#             sample_counter += 1

#         if sample_counter >= max_samples:
#             break

#     print(f"\nSaved {min(max_samples, idx + 1)} visualization(s) to: {out_dir}")


# import json
# import torch
# from pathlib import Path
# from visualization.ipf_render import render_sr_hr_side_by_side
# from utils.quat_ops import torch_to_numpy_quat, to_spatial_quat
# from training.data_loading import build_dataloader
# from models.reynolds_qsr import Reynolds_QSR
# from orix.quaternion import symmetry


# def run_postprocess_from_config(exp_dir: str, max_samples: int = 8):
#     """
#     Post-process trained model results using run_config.json in exp_dir.

#     Parameters
#     ----------
#     exp_dir : str
#         Path to experiment directory containing logs/run_config.json and checkpoints.
#     max_samples : int, default=8
#         Number of test samples to visualize.
#     """
#     exp_dir = Path(exp_dir)
#     config_path = exp_dir / "logs" / "run_config.json"
#     ckpt_path = exp_dir / "checkpoints" / "best_model.pt"

#     if not config_path.exists():
#         raise FileNotFoundError(f"Missing config: {config_path}")
#     if not ckpt_path.exists():
#         raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

#     # --------------------------
#     # Load config
#     # --------------------------
#     with open(config_path, "r") as f:
#         cfg = json.load(f)

#     print(f"Loaded config from {config_path}")

#     # --------------------------
#     # Rebuild test dataloader
#     # --------------------------
#     loaders = {
#         split: build_dataloader(
#             dataset_root=cfg["dataset_root"],
#             split=split.capitalize(),
#             batch_size=cfg["batch_size"],
#             num_workers=cfg["num_workers"],
#             preload=cfg["preload"],
#             preload_torch=cfg["preload_torch"],
#             pin_memory=cfg["pin_memory"],
#             take_first=5 if cfg.get("smoke_test", False) else None,
#         )
#         for split in ["test"]
#     }

#     # --------------------------
#     # Load trained model
#     # --------------------------
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(cfg)
#     model = Reynolds_QSR(cfg).to(device)
#     model.load_state_dict(torch.load(ckpt_path, map_location=device))
#     model.eval()

#     print(f"Loaded checkpoint from {ckpt_path}")

#     # --------------------------
#     # Set up visualization output
#     # --------------------------
#     out_dir = exp_dir / "visualizations"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     sym_class = symmetry.cubic_oh  # TODO: make configurable

#     # --------------------------
#     # Run inference + render
#     # --------------------------
#     for idx, (lr, hr) in enumerate(loaders["test"]):
#         if idx >= max_samples:
#             break
#         with torch.no_grad():
#             sr = model(lr.to(device, non_blocking=True))

#         sr_np = to_spatial_quat(torch_to_numpy_quat(sr))
#         hr_np = to_spatial_quat(torch_to_numpy_quat(hr))

#         out_path = out_dir / f"sr_hr_comparison_{idx:03d}.png"
#         render_sr_hr_side_by_side(
#             sr_np,
#             hr_np,
#             sym_class=sym_class,
#             out_png=str(out_path),
#             ref_dir="ALL",
#             include_key=True,
#             overwrite=True,
#         )

#     print(f"Saved {min(max_samples, idx+1)} visualization(s) to: {out_dir}")
