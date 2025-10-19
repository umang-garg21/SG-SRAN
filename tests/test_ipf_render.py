# -*-coding:utf-8 -*-
"""
Tests for ipf_render.py
"""

import os
import numpy as np
import pytest
import tempfile

from utils.symmetry_utils import resolve_symmetry
from utils.quat_ops import get_dummy_quats, format_quaternions
from visualization.ipf_render import render_ipf_image, render_ipf_rgb

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="session")
def sym_Oh():
    """Cubic symmetry group."""
    return resolve_symmetry("Oh")


@pytest.fixture(scope="session")
def stable_quats():
    """Deterministic quaternion field inside the FZ."""
    q = get_dummy_quats(resolution_deg=3.0)
    N = int(np.sqrt(q.shape[0]))
    q = q[: N * N].reshape(N, N, 4)
    # Canonical formatting once to avoid randomness
    return format_quaternions(
        q, normalize=True, hemisphere=True, reduce_fz=True, sym=resolve_symmetry("Oh")
    )


@pytest.fixture
def temp_png_path():
    """Temporary file path for saving PNGs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "ipf_render_test.png")


# ============================================================
# IPF RGB rendering tests
# ============================================================


def test_render_ipf_rgb_single_direction(stable_quats, sym_Oh):
    rgb = render_ipf_rgb(stable_quats, sym_Oh, ref_dir="X")
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape[:2] == stable_quats.shape[:2]
    assert rgb.shape[2] == 3
    assert (rgb >= 0).all() and (rgb <= 1).all()


def test_render_ipf_rgb_all_directions(stable_quats, sym_Oh):
    rgb_list = render_ipf_rgb(stable_quats, sym_Oh, ref_dir="ALL")
    assert isinstance(rgb_list, list)
    assert len(rgb_list) == 3
    for rgb in rgb_list:
        assert rgb.shape[:2] == stable_quats.shape[:2]
        assert rgb.shape[2] == 3


def test_render_ipf_rgb_invalid_direction(stable_quats, sym_Oh):
    with pytest.raises(ValueError):
        render_ipf_rgb(stable_quats, sym_Oh, ref_dir="bad_dir")


# ============================================================
# IPF image rendering tests
# ============================================================


def test_render_ipf_image_single_direction(stable_quats, sym_Oh, temp_png_path):
    out = render_ipf_image(stable_quats, sym_Oh, out_png=temp_png_path, ref_dir="Z")
    assert os.path.exists(out), f"Expected file at {out}"
    assert out.endswith(".png")


def test_render_ipf_image_all_directions(stable_quats, sym_Oh, temp_png_path):
    out = render_ipf_image(stable_quats, sym_Oh, out_png=temp_png_path, ref_dir="ALL")
    assert os.path.exists(out)


def test_render_ipf_image_include_key(stable_quats, sym_Oh, temp_png_path):
    out = render_ipf_image(
        stable_quats, sym_Oh, out_png=temp_png_path, include_key=True
    )
    assert os.path.exists(out)


def test_render_ipf_image_no_key(stable_quats, sym_Oh, temp_png_path):
    out = render_ipf_image(
        stable_quats, sym_Oh, out_png=temp_png_path, include_key=False
    )
    assert os.path.exists(out)


def test_render_ipf_image_overwrite_behavior(stable_quats, sym_Oh, temp_png_path):
    # First run
    out1 = render_ipf_image(stable_quats, sym_Oh, out_png=temp_png_path, ref_dir="X")
    # Second run should skip because overwrite=False by default
    out2 = render_ipf_image(stable_quats, sym_Oh, out_png=temp_png_path, ref_dir="X")
    assert out1 == out2


def test_render_ipf_image_formatting_off(stable_quats, sym_Oh, temp_png_path):
    # Pass already formatted quats
    out = render_ipf_image(
        stable_quats, sym_Oh, out_png=temp_png_path, format_input=False
    )
    assert os.path.exists(out)
