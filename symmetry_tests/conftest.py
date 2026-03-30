from __future__ import annotations

"""Shared pytest fixtures for symmetry block tests."""

import sys
from pathlib import Path

import pytest
import torch

# Ensure repo root is importable during pytest collection/execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SR_double_conv_SRattn import LocalIsoCrystalEncoder

# These tests depend on e3nn irreps operations; skip cleanly if unavailable.
pytest.importorskip("e3nn")


@pytest.fixture(scope="module", params=["fcc", "hcp"], ids=["G=O_fcc", "G=D6_hcp"])
def crystal(request) -> str:
    """Parameterize tests over both supported crystal families."""
    return str(request.param)


@pytest.fixture(scope="module")
def encoder(crystal: str) -> LocalIsoCrystalEncoder:
    """Provide one frozen encoder per crystal family for block tests."""
    return LocalIsoCrystalEncoder(
        crystal=crystal,
        dtype=torch.float32,
        device="cpu",
    ).eval()
