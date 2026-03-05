# -*-coding:utf-8 -*-
"""
File:        config_utils.py
Created at:  2025/10/18 19:48:56
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any
from utils.symmetry_utils import canon_symmetry_str, generate_symmetry_files
from utils.config_utils import ConfigNamespace

# from training.symmetry_utils import (
#     prepare_symmetry_files,
#     generate_symmetry_files,
#     canon_symmetry_str,
# )

# ----------------------------------------------------------------------
# Default Training Configuration
# ----------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset_root": "",
    "epochs": 10,
    "batch_size": 5,
    "lr": 1e-3,
    "scale": 4,
    "n_feats": 32,
    "kernel_size": 3,
    "symmetry_group": "O",  # overwritten if dataset_info.json has it
    "loss": "rotational_distance",
    "amp": False,
    "clip": 1.0,
    "vis_every": 2,
    "num_workers": 0,
    "preload": True,
    "preload_torch": True,
    "pin_memory": True,
    "seed": 42,  # Random seed for reproducibility
    "scheduler": {
        "type": "cosine",
        "warmup_epochs": 2,
        "min_lr": 1e-6,
    },
    "optimizer": {
        "type": "AdamW",
        "weight_decay": 0.0001,
    },
    # How often (in epochs) to save visualizations during training. Set to
    # a positive integer (e.g., 100) to generate visualizations every N epochs.
    # Set to 0 or a negative value to disable periodic visualizations.
    "save_every": 100,
    "model": {
        "type": "Reynolds_QSR",
        "dropout": 0.0,
    },
    "logging": {
        "tensorboard": True,
        "save_best_only": True,
        "val_freq": 1,
    },
    "smoke_test": True,
}


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update `base` dict with keys from `override`."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Path) -> Dict[str, Any]:
    """Load user configuration from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def preprocess_config(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user configuration with default values."""
    merged = copy.deepcopy(DEFAULT_CONFIG)
    deep_update(merged, user_cfg)
    return merged


def save_resolved_config(cfg: Dict[str, Any], out_path: Path):
    """Save the merged (resolved) configuration to a file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved resolved config to {out_path}")


def print_config_diff(cfg: Dict[str, Any]):
    """Print only values that differ from the defaults."""
    print("Overridden config values:")
    for key, default_val in DEFAULT_CONFIG.items():
        if isinstance(default_val, dict):
            for subkey, subval in default_val.items():
                if cfg[key][subkey] != subval:
                    print(f"  - {key}.{subkey}: {cfg[key][subkey]} (default: {subval})")
        else:
            if cfg[key] != default_val:
                print(f"  - {key}: {cfg[key]} (default: {default_val})")


# ----------------------------------------------------------------------
# High-level API
# ----------------------------------------------------------------------


def try_load_dataset_symmetry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    If symmetry_group is not provided in cfg, attempt to read it
    from dataset_info.json at dataset_root. If not found, fall back
    to the default value.
    """
    # Only check dataset if symmetry_group wasn't explicitly set
    if not cfg.get("symmetry_group"):
        dataset_root = Path(cfg["dataset_root"])
        info_path = dataset_root / "dataset_info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                dataset_info = json.load(f)
            if "symmetry" in dataset_info:
                cfg["symmetry_group"] = dataset_info["symmetry"]
                print(f"Detected symmetry group from dataset: {cfg['symmetry_group']}")
            else:
                print(
                    f"dataset_info.json found but no 'symmetry' key — using default {DEFAULT_CONFIG['symmetry_group']}"
                )
                cfg["symmetry_group"] = DEFAULT_CONFIG["symmetry_group"]
        else:
            print(
                f"No dataset_info.json found — using default {DEFAULT_CONFIG['symmetry_group']}"
            )
            cfg["symmetry_group"] = DEFAULT_CONFIG["symmetry_group"]
    else:
        print(f"Using symmetry group from config: {cfg['symmetry_group']}")
    return cfg


def prepare_symmetry_files(cfg: Dict, base_dir: Path | None = None) -> Dict:
    """
    Ensures symmetry group files exist and updates cfg with their paths.
    """
    if base_dir is None:
        # normal training path
        base_dir = Path(__file__).resolve().parents[1] / "symmetry_groups"

    base_dir.mkdir(parents=True, exist_ok=True)

    canon_name = canon_symmetry_str(cfg.get("symmetry_group", "Oh"))
    sym_path = base_dir / f"{canon_name}_group.npy"
    sym_inv_path = base_dir / f"{canon_name}_group_inv.npy"

    if not sym_path.exists() or not sym_inv_path.exists():
        print(f"Generating symmetry files for group: {canon_name}")
        generate_symmetry_files(canon_name, base_dir)
    else:
        print(f"Using existing symmetry files for {canon_name}")

    cfg["sym_np_path"] = str(sym_path)
    cfg["sym_inv_np_path"] = str(sym_inv_path)
    return cfg


def flatten_model_config(cfg: dict) -> dict:
    """
    Promote model subkeys to top-level cfg for convenience while
    preserving cfg['model'] structure for readability.
    """
    if "model" in cfg and isinstance(cfg["model"], dict):
        m = cfg["model"]
        # model_type goes to top level
        cfg["model_type"] = m.get("type", None)
        # promote remaining model keys
        for k, v in m.items():
            if k != "type":
                if k in cfg and cfg[k] != v:
                    print(
                        f"[config warning] '{k}' exists in both root and model block; using model value {v}"
                    )
                cfg[k] = v
    return cfg


def load_and_prepare_config(
    config_path: Path, save_path: Path | None = None
) -> Dict[str, Any]:
    """
    Load config from JSON, merge with defaults, print diffs, and save resolved config.
    Also prepares symmetry files and injects their paths into cfg.
    """
    user_cfg = load_config(config_path)
    cfg = preprocess_config(user_cfg)

    # Try to read symmetry from dataset info file
    cfg = try_load_dataset_symmetry(cfg)

    print_config_diff(cfg)

    # Ensure checkpoints_dir exists
    exp_dir = config_path.parent
    if "checkpoints_dir" not in cfg or not cfg["checkpoints_dir"]:
        cfg["checkpoints_dir"] = str(exp_dir / "checkpoints")
    Path(cfg["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)

    # Generate or use existing symmetry files and add to cfg
    prepare_symmetry_files(cfg)

    # Save merged config
    if save_path:
        save_resolved_config(cfg, save_path)

    cfg = flatten_model_config(cfg)

    cfg = ConfigNamespace(cfg)

    return cfg


# # -*-coding:utf-8 -*-
# """
# File:        config_utils.py
# Author:      Warren Zamudio
# Description: Config loading, merging with defaults, and preprocessing.
# """

# import json
# import copy
# from pathlib import Path
# from typing import Dict, Any
# from training.symmetry_utils import prepare_symmetry_files

# # ----------------------------------------------------------------------
# # Default Training Configuration
# # ----------------------------------------------------------------------

# DEFAULT_CONFIG: Dict[str, Any] = {
#     "dataset_root": "",
#     "epochs": 10,
#     "batch_size": 4,
#     "lr": 3e-4,
#     "scale": 4,
#     "n_feats": 32,
#     "kernel_size": 3,
#     "symmetry_group": "O",
#     "amp": False,
#     "clip": 1.0,
#     "vis_every": 1,
#     "num_workers": 0,
#     "preload": True,
#     "preload_torch": True,
#     "pin_memory": True,
#     "scheduler": {
#         "type": "cosine",
#         "warmup_epochs": 1,
#         "min_lr": 1e-6,
#     },
#     "optimizer": {
#         "type": "AdamW",
#         "weight_decay": 0.0001,
#     },
#     "model": {
#         "type": "Reynolds_QSR",
#         "dropout": 0.0,
#     },
#     "logging": {
#         "tensorboard": True,
#         "save_best_only": True,
#         "val_freq": 1,
#     },
#     "smoke_test": True,
# }

# # ----------------------------------------------------------------------
# # 🔸 Helper Functions
# # ----------------------------------------------------------------------


# def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
#     """Recursively update `base` dict with keys from `override`."""
#     for k, v in override.items():
#         if isinstance(v, dict) and isinstance(base.get(k), dict):
#             deep_update(base[k], v)
#         else:
#             base[k] = v
#     return base


# def load_config(path: Path) -> Dict[str, Any]:
#     """Load user configuration from a JSON file."""
#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(f"Config file not found: {path}")
#     with open(path, "r") as f:
#         return json.load(f)


# def preprocess_config(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
#     """Merge user configuration with default values."""
#     merged = copy.deepcopy(DEFAULT_CONFIG)
#     deep_update(merged, user_cfg)
#     return merged


# def save_resolved_config(cfg: Dict[str, Any], out_path: Path):
#     """Save the merged (resolved) configuration to a file."""
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w") as f:
#         json.dump(cfg, f, indent=2)
#     print(f"Saved resolved config to {out_path}")


# def print_config_diff(cfg: Dict[str, Any]):
#     """Print only values that differ from the defaults."""
#     print("Overridden config values:")
#     for key, default_val in DEFAULT_CONFIG.items():
#         if isinstance(default_val, dict):
#             for subkey, subval in default_val.items():
#                 if cfg[key][subkey] != subval:
#                     print(f"  - {key}.{subkey}: {cfg[key][subkey]} (default: {subval})")
#         else:
#             if cfg[key] != default_val:
#                 print(f"  - {key}: {cfg[key]} (default: {default_val})")


# # ----------------------------------------------------------------------
# # High-level API
# # ----------------------------------------------------------------------


# def load_and_prepare_config(
#     config_path: Path, save_path: Path | None = None
# ) -> Dict[str, Any]:
#     """
#     Load config from JSON, merge with defaults, print diffs, and save resolved config.

#     Parameters
#     ----------
#     config_path : Path
#         Path to the input config.json
#     save_path : Path, optional
#         Path to save resolved config (e.g., logs/run_config.json)

#     Returns
#     -------
#     Dict[str, Any]
#         Fully resolved configuration
#     """
#     user_cfg = load_config(config_path)
#     cfg = preprocess_config(user_cfg)
#     print_config_diff(cfg)

#     # Ensure checkpoints_dir exists
#     exp_dir = config_path.parent
#     if "checkpoints_dir" not in cfg or not cfg["checkpoints_dir"]:
#         cfg["checkpoints_dir"] = str(exp_dir / "checkpoints")

#     Path(cfg["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)

#     prepare_symmetry_files(cfg)
#     # Optionally save merged config
#     if save_path:
#         save_resolved_config(cfg, save_path)

#     return cfg
