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
    "ocrp_upsample_residual": False,
    "ocrp_upsample_residual_weight": 1.0,
    "ocrp_upsample_residual_weight_final": None,
    "ocrp_upsample_residual_weight_schedule": "constant",
    "use_lr_conv1": True,
    "use_lr_conv2": True,
    "use_attention": True,
    "num_hr_attn_blocks": 1,
    "hr_attn_num_channels": 8,
    "hr_attn_block_size": 16,
    "hr_attn_tp_out_chunk_size": 2048,
    "hr_attn_checkpoint": False,
    "use_amp": False,
    "amp_dtype": "bf16",
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
    "feature_upsampler_type": "shifted_bilinear",
    "use_boundary_gate": False,
    "stats_hidden_dim": 32,
    "lambda_feat": 1.0,
    "lambda_boundary": 0.5,
    "lambda_lr_boundary": 0.10,
    "lambda_side_correct": 0.10,
    "lambda_side_entropy": 0.002,
    "boundary_thr_deg": 3.0,
    "boundary_connectivity": 4,
    "use_focal_boundary": True,
    "focal_gamma": 2.0,
    "side_correct_band_kernel": [3, 3],
    "side_correct_rel_gap": 0.05,
    "clip": 1.0,
    "save_every": 5,
    "viz_every": 5,
    "viz_ref_dir": "ALL",
    "viz_sample_index": 0,
    "viz_sample_key": None,
    "num_workers": 0,
    "persistent_workers": False,
    "prefetch_factor": 2,
    "memory_debug_every": 0,
    "cuda_empty_cache_every": 0,
    "preload": False,
    "preload_torch": False,
    "pin_memory": True,
    "allow_tf32": True,
    "cudnn_benchmark": False,
    "min_free_cuda_gb": 0.0,
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


def _load_dataset_info(dataset_root: str | Path | None) -> Dict[str, Any] | None:
    """Load dataset_info.json when available."""
    if dataset_root is None or not str(dataset_root).strip():
        return None
    info_path = Path(dataset_root) / "dataset_info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r") as f:
        return json.load(f)


def _normalize_crystal_name(crystal: Any) -> str:
    """Normalize config crystal aliases to the model's supported names."""
    key = str(crystal).strip().lower()
    if key in {"fcc", "o", "oh", "cubic"}:
        return "fcc"
    if key in {"hcp", "d6", "d6h", "hex", "hexagonal"}:
        return "hcp"
    raise ValueError(
        f"Unsupported crystal={crystal!r}. Expected one of "
        "'fcc'/'cubic' or 'hcp'/'hexagonal'."
    )


def _crystal_for_symmetry(symmetry_group: Any) -> str | None:
    """Infer the supported crystal family from a symmetry-group name."""
    canon = canon_symmetry_str(symmetry_group)
    if canon in {"O", "Oh", "Td"}:
        return "fcc"
    if str(canon).startswith("D6"):
        return "hcp"
    return None


def try_load_dataset_symmetry(
    cfg: Dict[str, Any],
    user_cfg: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], str | None]:
    """
    Resolve symmetry_group from the explicit config or dataset metadata.

    Returns the updated cfg plus the dataset symmetry (if available).
    """
    explicit_sym = bool(
        isinstance(user_cfg, dict)
        and str(user_cfg.get("symmetry_group", "")).strip()
    )

    dataset_info = _load_dataset_info(cfg.get("dataset_root"))
    dataset_symmetry = None
    if isinstance(dataset_info, dict) and str(dataset_info.get("symmetry", "")).strip():
        dataset_symmetry = canon_symmetry_str(dataset_info["symmetry"])

    if explicit_sym:
        cfg_symmetry = canon_symmetry_str(
            cfg.get("symmetry_group", DEFAULT_CONFIG["symmetry_group"])
        )
        cfg["symmetry_group"] = cfg_symmetry
        print(f"Using symmetry group from config: {cfg_symmetry}")
        if dataset_symmetry is not None and cfg_symmetry != dataset_symmetry:
            raise ValueError(
                "Config symmetry_group does not match dataset metadata: "
                f"config={cfg_symmetry!r}, dataset={dataset_symmetry!r}, "
                f"dataset_root={cfg.get('dataset_root')!r}"
            )
    elif dataset_symmetry is not None:
        cfg["symmetry_group"] = dataset_symmetry
        print(f"Detected symmetry group from dataset: {cfg['symmetry_group']}")
    else:
        cfg["symmetry_group"] = canon_symmetry_str(
            cfg.get("symmetry_group", DEFAULT_CONFIG["symmetry_group"])
        )
        print(f"No dataset_info.json found — using default {cfg['symmetry_group']}")

    return cfg, dataset_symmetry


def harmonize_symmetry_and_crystal(
    cfg: Dict[str, Any],
    user_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Keep symmetry_group and crystal aligned with the dataset and model family."""
    cfg, dataset_symmetry = try_load_dataset_symmetry(cfg, user_cfg=user_cfg)

    explicit_crystal = bool(
        isinstance(user_cfg, dict)
        and str(user_cfg.get("crystal", "")).strip()
    )
    crystal_value = cfg.get("crystal", DEFAULT_CONFIG["crystal"])
    normalized_crystal = _normalize_crystal_name(crystal_value)
    expected_crystal = _crystal_for_symmetry(cfg.get("symmetry_group"))

    if expected_crystal is not None and normalized_crystal != expected_crystal:
        if explicit_crystal:
            raise ValueError(
                "Config crystal/symmetry mismatch: "
                f"symmetry_group={cfg.get('symmetry_group')!r} implies crystal={expected_crystal!r}, "
                f"but config crystal={crystal_value!r}. "
                "This would instantiate the wrong local-iso encoder/decoder family."
            )
        print(
            f"Inferred crystal family {expected_crystal!r} from symmetry_group "
            f"{cfg.get('symmetry_group')!r}."
        )
        normalized_crystal = expected_crystal

    cfg["crystal"] = normalized_crystal

    if dataset_symmetry is not None:
        dataset_crystal = _crystal_for_symmetry(dataset_symmetry)
        if dataset_crystal is not None and cfg["crystal"] != dataset_crystal:
            raise ValueError(
                "Config crystal does not match dataset symmetry metadata: "
                f"crystal={cfg['crystal']!r}, dataset_symmetry={dataset_symmetry!r}, "
                f"dataset_root={cfg.get('dataset_root')!r}"
            )

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

    # Resolve dataset symmetry and validate crystal-family consistency.
    cfg = harmonize_symmetry_and_crystal(cfg, user_cfg=user_cfg)

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
