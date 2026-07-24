# -*-coding:utf-8 -*-
"""
File:        symmetry_utils.py
Created at:  2025/10/17 13:15:39
Author:      Warren Zamudio
Contact:     wzamudio@ucsb.edu
Description: None
"""

from orix.quaternion import symmetry as SYM
import numpy as np
from pathlib import Path

# from utils.quat_ops import quaternion_left_matrix

# ----------------------------------------------------------------------
# Common symmetry aliases and numeric group shorthands
# ----------------------------------------------------------------------
_SYM_ALIASES = {
    # cubic
    "oh": "Oh",
    "cubic": "Oh",
    "fcc": "Oh",
    "bcc": "Oh",
    "m-3m": "Oh",
    "m3m": "Oh",
    # hexagonal / hcp
    "hcp": "D6h",
    "hex": "D6h",
    "6/mmm": "D6h",
    "d6": "D6h",
    "d6h": "D6h",
    # tetragonal / orthorhombic
    "d4h": "D4h",
    "d3d": "D3d",
    "d2h": "D2h",
    # tetrahedral / octahedral
    "td": "Td",
    "o": "O",
    "432": "O",
    # trivial / none
    "none": "C1",
    "na": "C1",
}


def canon_symmetry_str(sym_str):
    """Return canonical symmetry name string."""
    if not isinstance(sym_str, str):
        return getattr(sym_str, "__name__", str(sym_str))
    key = sym_str.strip().lower()
    return _SYM_ALIASES.get(key, sym_str.strip())


def resolve_symmetry(sym):
    """Return orix symmetry object from string or object with alias support."""
    if not isinstance(sym, str):
        return sym
    canon = canon_symmetry_str(sym)
    if hasattr(SYM, canon):
        return getattr(SYM, canon)
    raise ValueError(f"Unknown symmetry: {sym}")


def proper_symmetry_quaternions(sym, decimals: int = 7) -> np.ndarray:
    """Return unique proper rotation quaternions, identifying q and -q.

    Centrosymmetric orix point groups such as Oh and D6h contain inversion-
    related duplicates in ``sym.data``. Crystal-orientation misorientation uses
    only the proper rotational subgroup: O (24 rotations) or D6 (12 rotations).
    """
    symmetry = resolve_symmetry(sym)
    quaternions = np.asarray(symmetry.data, dtype=np.float64).copy()
    improper = np.asarray(
        getattr(symmetry, "improper", np.zeros(len(quaternions), dtype=bool)),
        dtype=bool,
    )
    if improper.shape == (len(quaternions),):
        quaternions = quaternions[~improper]
    quaternions /= np.maximum(
        np.linalg.norm(quaternions, axis=-1, keepdims=True), 1e-12
    )
    flip = quaternions[:, 0] < 0.0
    quaternions[flip] *= -1.0
    rounded = np.round(quaternions, decimals=decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return quaternions[np.sort(unique_indices)].astype(np.float32)


def quaternion_left_matrix(q: np.ndarray) -> np.ndarray:
    """Return the 4x4 left multiplication matrix for quaternion q = [w,x,y,z]."""
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ],
        dtype=np.float32,
    )


def generate_symmetry_files(group_name: str, save_dir: Path) -> tuple[Path, Path]:
    """
    Generate and save group representation and its inverse to .npy files.

    Parameters
    ----------
    group_name : str
        Symmetry group string (e.g., "432", "Oh", "D4").
    save_dir : Path
        Directory where files will be saved.

    Returns
    -------
    sym_path : Path
        Path to saved group tensor file.
    sym_inv_path : Path
        Path to saved inverse group tensor file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # ✅ use canonical name for consistent filenames
    canon_name = canon_symmetry_str(group_name)
    group_obj = resolve_symmetry(canon_name)

    # group_obj.data is (G, 4) quaternions
    quat_array = np.asarray(group_obj.data, dtype=np.float32)
    G = quat_array.shape[0]

    # Convert quaternions to (G,4,4) Reynolds representation
    group_tensor = np.stack([quaternion_left_matrix(q) for q in quat_array], axis=0)

    # For unit quaternions, inverse = transpose
    group_tensor_inv = np.transpose(group_tensor, (0, 2, 1))

    # ✅ canonical naming for saving
    sym_path = save_dir / f"{canon_name}_group.npy"
    sym_inv_path = save_dir / f"{canon_name}_group_inv.npy"

    np.save(sym_path, group_tensor)
    np.save(sym_inv_path, group_tensor_inv)

    print(f"✅ Generated symmetry files for '{canon_name}':")
    print(f"   {sym_path}")
    print(f"   {sym_inv_path}")

    return sym_path, sym_inv_path


if __name__ == "__main__":
    a = resolve_symmetry("cubic")
