import numpy as np
import torch
import pytest

from models.autoencoder import FastLookupFCCDecoder, FCCPhysics


def test_requires_lookup_path_raises():
    physics = FCCPhysics("cpu")
    with pytest.raises(ValueError):
        FastLookupFCCDecoder(physics, lookup_npy_path=None)


def test_missing_file_raises(tmp_path):
    physics = FCCPhysics("cpu")
    p = tmp_path / "does_not_exist.npy"
    with pytest.raises(FileNotFoundError):
        FastLookupFCCDecoder(physics, lookup_npy_path=str(p))


def test_loads_dummy_lookup_and_forward(tmp_path):
    # Build a tiny valid lookup table (N=4, 27 cols)
    arr = np.zeros((4, 27), dtype=np.float32)
    # put trivial unit quaternions in the first 4 columns
    arr[:, 0] = 1.0
    lookup_path = tmp_path / "lookup.npy"
    np.save(str(lookup_path), arr)

    physics = FCCPhysics("cpu")
    dec = FastLookupFCCDecoder(physics, lookup_npy_path=str(lookup_path))

    f4 = torch.zeros((1, 9), dtype=torch.float32)
    f6 = torch.zeros((1, 13), dtype=torch.float32)
    out = dec(f4, f6)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 4)
