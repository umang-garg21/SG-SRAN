from __future__ import annotations

import types

import torch

from training.config_utils import deep_update, flatten_model_config, preprocess_config
from training.seed_utils import get_seed_from_config, set_seed


def test_deep_update_merges_nested_dicts() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 20, "z": 30}, "b": 2}
    out = deep_update(base, override)
    assert out["a"] == 1
    assert out["b"] == 2
    assert out["nested"] == {"x": 1, "y": 20, "z": 30}


def test_preprocess_and_flatten_model_config() -> None:
    cfg = preprocess_config(
        {
            "epochs": 3,
            "model": {"type": "iso_embedding_sr_attn"},
            "scheduler": {"type": "step", "step_size": 7, "gamma": 0.5},
        }
    )
    cfg = flatten_model_config(cfg)
    assert cfg["epochs"] == 3
    assert cfg["model_type"] == "iso_embedding_sr_attn"
    assert cfg["scheduler"]["type"] == "step"
    assert cfg["scheduler"]["step_size"] == 7
    assert cfg["scheduler"]["gamma"] == 0.5
    assert "lr" in cfg


def test_get_seed_from_config_dict_and_namespace() -> None:
    assert get_seed_from_config({"seed": 7}) == 7
    ns = types.SimpleNamespace(seed=13)
    assert get_seed_from_config(ns) == 13
    assert get_seed_from_config({}, default=19) == 19


def test_set_seed_is_deterministic_for_torch() -> None:
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)

