from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "Paper/EBSD_SR_Nature_v3/evals"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import evaluate_anchorless_test_metrics as evaluator
from utils.symmetry_utils import proper_symmetry_quaternions


def test_misorientation_uses_live_symmetry_binding() -> None:
    identity = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    cubic_quarter_turn_x = np.array(
        [[[np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]]], dtype=np.float32
    )

    evaluator.SYM_QUATS = proper_symmetry_quaternions("D6h")
    d6_error = float(
        evaluator.misorientation_map(identity, cubic_quarter_turn_x).item()
    )

    evaluator.SYM_QUATS = proper_symmetry_quaternions("Oh")
    cubic_error = float(
        evaluator.misorientation_map(identity, cubic_quarter_turn_x).item()
    )

    assert d6_error > 80.0
    assert cubic_error < 0.1
