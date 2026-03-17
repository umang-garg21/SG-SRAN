"""Configuration utilities for IsoEmbeddingSRAttn training/inference."""

import copy
import json
from pathlib import Path
from typing import Dict, Any

from utils.symmetry_utils import canon_symmetry_str, generate_symmetry_files
from utils.config_utils import ConfigNamespace

# ----------------------------------------------------------------------
# Default Training Configuration
# ----------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset_root": "",
    "epochs": 10,
    "batch_size": 4,
    "lr": 3e-4,
    "scale": 4,
    "symmetry_group": "O",
    "crystal": "fcc",
    "d6_convention": "z_axis",
    "upsample_residual": True,
    "num_hr_attn_blocks": 1,
    "hr_attn_num_channels": 8,
    "hr_attn_block_size": 16,
    "decoder_cubochoric_resolution": 1,
    "decoder_num_starts": 2,
    "decoder_steps": 1,
    "decoder_lr": 0.05,
    "decoder_method": "cubochoric",
    "decoder_max_table_rows": None,
    "decoder_table_cache_dir": "out/decoder_lookup_tables",
    "decoder_backend": "optimizing",
    "decoder_learnable_hidden_dim": 256,
    "decoder_learnable_num_layers": 3,
    "decoder_learnable_dropout": 0.0,
    "clip": 1.0,
    "save_every": 5,
    "viz_every": 5,
    "viz_ref_dir": "ALL",
    "num_workers": 0,
    "preload": False,
    "preload_torch": False,
    "pin_memory": True,
    "seed": 42,
    "smoke_test": False,
    "smoke_take_first": 8,
    "train_take_first": None,
    "val_take_first": None,
    "test_take_first": None,
    "checkpoints_dir": "",
    "scheduler": {
        "type": "cosine",
        "warmup_epochs": 2,
        "min_lr": 1e-6,
    },
    "optimizer": {
        "type": "AdamW",
        "weight_decay": 0.0001,
    },
    "model": {
        "type": "iso_embedding_sr_attn",
    },
    "logging": {
        "tensorboard": True,
        "save_best_only": True,
        "val_freq": 1,
    },
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
) -> ConfigNamespace:
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

    return ConfigNamespace(cfg)
