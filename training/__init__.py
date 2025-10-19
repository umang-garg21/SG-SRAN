# training/__init__.py
"""
Training package for Reynolds-QSR.

This package provides:
- Dataloading utilities for quaternion super-resolution
- Training loops and Trainer class
- Loss functions
- Optimizer and scheduler builders
- Config and symmetry handling helpers
"""

from training.trainer import Trainer
from training.data_loading import build_dataloader
from training.loss_functions import build_loss
from training.schedulers import build_scheduler
from training.optimizer_utils import build_optimizer
from training.config_utils import (
    load_config,
    preprocess_config,
    print_config_diff,
)

# from training.symmetry_utils import prepare_symmetry_files

__all__ = [
    "Trainer",
    "build_dataloader",
    "build_loss",
    "build_scheduler",
    "build_optimizer",
    "load_config",
    "preprocess_config",
    "print_config_diff",
    # "prepare_symmetry_files",
]
