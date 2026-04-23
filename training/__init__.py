"""Training package bootstrap and shared exports."""

import os
from pathlib import Path

# `orix` pulls in numba-cached helpers during import. Some environments do not
# provide a writable/default cache locator, so we force a safe cache root early
# before importing the rest of the training stack.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

from training.config_utils import load_and_prepare_config
from training.data_loading import build_dataloader
from training.optimizer_utils import build_optimizer
from training.schedulers import build_scheduler
from training.seed_utils import get_seed_from_config, set_seed

__all__ = [
    "build_dataloader",
    "build_optimizer",
    "build_scheduler",
    "get_seed_from_config",
    "load_and_prepare_config",
    "set_seed",
]
