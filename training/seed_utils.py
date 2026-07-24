"""Utilities for reproducible training."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed_from_config(cfg, default: int = 42) -> int:
    """Extract integer seed from config with fallback default."""
    if hasattr(cfg, "seed"):
        return int(getattr(cfg, "seed"))
    if isinstance(cfg, dict) and "seed" in cfg:
        return int(cfg["seed"])
    return int(default)
