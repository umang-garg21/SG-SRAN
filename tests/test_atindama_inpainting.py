from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from training.train_atindama_inpainting import (
    geodesic_mse,
    normalized_zxz_to_matrix,
    normalized_zxz_to_passive_quaternion,
    passive_quaternion_to_normalized_zxz,
    periodic_known_mask,
)
from inference.infer_atindama_inpainting import sanitize_prediction


def test_periodic_masks_match_dataset_sampling() -> None:
    mask_4x1 = periodic_known_mask(256, 256, [4, 1])
    mask_4x4 = periodic_known_mask(256, 256, [4, 4])
    assert int(mask_4x1[0].sum()) == 64 * 256
    assert int(mask_4x4[0].sum()) == 64 * 64
    assert np.all(mask_4x1[:, ::4, :] == 1)
    assert np.all(mask_4x4[:, ::4, ::4] == 1)


def test_quaternion_euler_round_trip() -> None:
    active = Rotation.random(128, random_state=7).as_quat()
    passive = np.empty_like(active)
    passive[:, 0] = active[:, 3]
    passive[:, 1:] = -active[:, :3]
    passive = passive.reshape(8, 16, 4).astype(np.float32)
    euler = passive_quaternion_to_normalized_zxz(passive)
    recovered = normalized_zxz_to_passive_quaternion(euler)
    dots = np.abs(np.sum(passive * recovered, axis=-1))
    assert np.min(dots) > 1.0 - 1e-5


def test_torch_zxz_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    normalized = rng.random((2, 3, 5, 7), dtype=np.float32)
    torch_matrix = normalized_zxz_to_matrix(torch.from_numpy(normalized)).numpy()
    angles = normalized.transpose(0, 2, 3, 1) * np.array(
        [2 * np.pi, np.pi, 2 * np.pi]
    )
    scipy_matrix = Rotation.from_euler("ZXZ", angles.reshape(-1, 3)).as_matrix()
    scipy_matrix = scipy_matrix.reshape(2, 5, 7, 3, 3)
    assert np.allclose(torch_matrix, scipy_matrix, atol=1e-5)


def test_geodesic_loss_is_small_for_identical_maps() -> None:
    target = torch.rand(2, 3, 8, 8)
    mask = torch.zeros_like(target)
    loss = geodesic_mse(target, target, mask)
    assert float(loss) < 3e-6


def test_prediction_sanitization_uses_only_observed_samples() -> None:
    target = torch.rand(1, 3, 8, 8)
    mask = torch.from_numpy(periodic_known_mask(8, 8, [4, 1])).unsqueeze(0)
    prediction = torch.full_like(target, float("nan"))
    composite, invalid = sanitize_prediction(prediction, target, mask, [4, 1])
    assert int(invalid.item()) == 64
    assert torch.isfinite(composite).all()
    assert torch.allclose(composite[:, :, ::4, :], target[:, :, ::4, :])
