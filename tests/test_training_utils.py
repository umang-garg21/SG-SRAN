# -*- coding:utf-8 -*-
"""
Unit tests for training utilities: config loading, symmetry file generation,
and optimizer building.
Run with: `pytest -q tests/test_training_utils.py`
"""

import json
from pathlib import Path
import torch
import numpy as np
import training.symmetry_utils as sym_utils
from training.config_utils import load_and_prepare_config
from training.symmetry_utils import prepare_symmetry_files
from training.optimizer_utils import build_optimizer
from models.reynolds_qsr import Reynolds_QSR


# -------------------------------------------------------------------------
# 1. Test Config Loader
# -------------------------------------------------------------------------
def test_load_config_with_defaults(tmp_path):
    user_config = {"epochs": 1, "lr": 1e-4}
    cfg_path = tmp_path / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(user_config, f)

    cfg = load_and_prepare_config(cfg_path)
    assert cfg["epochs"] == 1
    assert cfg["lr"] == 1e-4
    # ✅ merged defaults should be here
    assert "group" in cfg


# -------------------------------------------------------------------------
# 2. Test Symmetry File Preparation
# -------------------------------------------------------------------------
def test_prepare_symmetry_files(tmp_path):
    cfg = {"group": "432"}

    # ✅ Pass tmp_path to avoid writing to project folders
    cfg_out = prepare_symmetry_files(cfg, base_dir=tmp_path)
    sym_path = Path(cfg_out["sym_np_path"])
    sym_inv_path = Path(cfg_out["sym_inv_np_path"])

    # ✅ Assert files are inside tmp_path
    assert sym_path.exists()
    assert sym_inv_path.exists()
    assert sym_path.parent == tmp_path
    assert sym_inv_path.parent == tmp_path

    # ✅ Basic shape check
    group = np.load(sym_path)
    group_inv = np.load(sym_inv_path)
    assert group.shape[1:] == (4, 4)
    assert group_inv.shape[1:] == (4, 4)


# -------------------------------------------------------------------------
# 3. Test Optimizer Builder
# -------------------------------------------------------------------------
def test_build_optimizer(tmp_path):
    cfg = {
        "lr": 0.001,
        "optimizer": {"type": "AdamW", "weight_decay": 0.01},
        "n_feats": 8,
        "scale": 2,
        "kernel_size": 3,
        "group": "432",
        "sym_np_path": str(tmp_path / "sym.npy"),
        "sym_inv_np_path": str(tmp_path / "sym_inv.npy"),
    }

    np.save(cfg["sym_np_path"], np.eye(4, dtype=np.float32))
    np.save(cfg["sym_inv_np_path"], np.eye(4, dtype=np.float32))

    model = Reynolds_QSR(cfg)
    opt = build_optimizer(model, cfg)

    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["lr"] == 0.001
    assert opt.defaults["weight_decay"] == 0.01
