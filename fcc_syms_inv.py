import torch
import math 


import argparse
import math
from pathlib import Path

import numpy as np
import torch

def build_fcc_syms() -> torch.Tensor:
	inv_sqrt_2 = 1.0 / math.sqrt(2.0)
	half = 0.5
	return torch.tensor(
		[
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1],
			[inv_sqrt_2, inv_sqrt_2, 0, 0],
			[inv_sqrt_2, 0, inv_sqrt_2, 0],
			[inv_sqrt_2, 0, 0, inv_sqrt_2],
			[inv_sqrt_2, -inv_sqrt_2, 0, 0],
			[inv_sqrt_2, 0, -inv_sqrt_2, 0],
			[inv_sqrt_2, 0, 0, -inv_sqrt_2],
			[0, inv_sqrt_2, inv_sqrt_2, 0],
			[0, inv_sqrt_2, 0, inv_sqrt_2],
			[0, 0, inv_sqrt_2, inv_sqrt_2],
			[0, inv_sqrt_2, -inv_sqrt_2, 0],
			[0, 0, inv_sqrt_2, -inv_sqrt_2],
			[0, inv_sqrt_2, 0, -inv_sqrt_2],
			[half, half, half, half],
			[half, -half, -half, half],
			[half, -half, half, -half],
			[half, half, -half, -half],
			[half, half, half, -half],
			[half, half, -half, half],
			[half, -half, half, half],
			[half, -half, -half, -half],
		],
		dtype=torch.float32,
	)


def build_fcc_syms_inv() -> torch.Tensor:
	inv_sqrt_2 = 1.0 / math.sqrt(2.0)
	half = 0.5
	return torch.tensor(
		[
			[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, -1],
			[inv_sqrt_2, -inv_sqrt_2, 0, 0],
			[inv_sqrt_2, 0, -inv_sqrt_2, 0],
			[inv_sqrt_2, 0, 0, -inv_sqrt_2],
			[inv_sqrt_2, inv_sqrt_2, 0, 0],
			[inv_sqrt_2, 0, inv_sqrt_2, 0],
			[inv_sqrt_2, 0, 0, inv_sqrt_2],
			[0, -inv_sqrt_2, -inv_sqrt_2, 0],
			[0, -inv_sqrt_2, 0, -inv_sqrt_2],
			[0, 0, -inv_sqrt_2, -inv_sqrt_2],
			[0, -inv_sqrt_2, inv_sqrt_2, 0],
			[0, 0, -inv_sqrt_2, inv_sqrt_2],
			[0, -inv_sqrt_2, 0, inv_sqrt_2],
			[half, -half, -half, -half],
			[half, half, half, -half],
			[half, half, -half, half],
			[half, -half, half, half],
			[half, -half, -half, half],
			[half, -half, half, -half],
			[half, half, -half, -half],
			[half, half, half, half],
		],
		dtype=torch.float32,
	)


def normalize_quats(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	q = np.asarray(q, dtype=np.float64)
	norms = np.linalg.norm(q, axis=-1, keepdims=True)
	norms = np.clip(norms, eps, None)
	return (q / norms).astype(np.float64, copy=False)


def quat_inverse(q: np.ndarray) -> np.ndarray:
	q = normalize_quats(q)
	q_inv = q.copy()
	q_inv[..., 1:] *= -1.0
	return q_inv


def as_rotation_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""
	Return sign-invariant quaternion distance proxy matrix: 1 - |dot(a,b)|.
	Lower is better; 0 means same rotation (up to sign).
	"""
	a = normalize_quats(a)
	b = normalize_quats(b)
	abs_dot = np.abs(a @ b.T)
	return 1.0 - abs_dot


def set_match_report(source: np.ndarray, target: np.ndarray, tol: float = 1e-6) -> dict[str, float | int]:
	dist = as_rotation_distance_matrix(source, target)
	best = dist.min(axis=1)
	n_matched = int(np.sum(best <= tol))
	return {
		"n_source": int(source.shape[0]),
		"n_target": int(target.shape[0]),
		"n_matched": n_matched,
		"max_best_dist": float(best.max()),
		"mean_best_dist": float(best.mean()),
	}


def compare_with_orix(fcc_syms_inv: np.ndarray, group_name: str = "O", tol: float = 1e-6) -> dict[str, dict[str, float | int]]:
	from orix.quaternion import symmetry

	if not hasattr(symmetry, group_name):
		raise ValueError(f"Unknown ORIX symmetry group: {group_name}")

	orix_group = getattr(symmetry, group_name)
	orix_ops = np.asarray(orix_group.data, dtype=np.float64)
	orix_ops = normalize_quats(orix_ops)
	orix_inv = quat_inverse(orix_ops)

	return {
		"fcc_inv_vs_orix_ops": set_match_report(fcc_syms_inv, orix_ops, tol=tol),
		"fcc_inv_vs_orix_inv": set_match_report(fcc_syms_inv, orix_inv, tol=tol),
	}



# --- Script mode: hardcoded parameters ---
if __name__ == "__main__":
    out_path = Path("symmetry_groups/fcc_syms_inv.npy")
    group_name = "O"
    tol = 1e-6

    # Compare computed inverse
    fcc_syms = build_fcc_syms().cpu().numpy().astype(np.float64, copy=False)
    fcc_syms_norm = normalize_quats(fcc_syms)
    fcc_syms_inv_computed = quat_inverse(fcc_syms_norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, fcc_syms_inv_computed.astype(np.float32, copy=False))

    print(f"Saved FCC computed inverse symmetries to: {out_path.resolve()}")
    print(f"Shape: {fcc_syms_inv_computed.shape}")

    # Compare hardcoded build_fcc_syms_inv
    fcc_syms_inv_hardcoded = build_fcc_syms_inv().cpu().numpy().astype(np.float64, copy=False)
    fcc_syms_inv_hardcoded = normalize_quats(fcc_syms_inv_hardcoded)

    try:
        print("\n--- Comparison: Computed Inverse ---")
        reports_computed = compare_with_orix(fcc_syms_inv_computed, group_name=group_name, tol=tol)
        for name, rep in reports_computed.items():
            print(f"{name}:")
            print(
                f"  matched {rep['n_matched']}/{rep['n_source']} "
                f"against target size {rep['n_target']}"
            )
            print(f"  mean_best_dist={rep['mean_best_dist']:.3e}")
            print(f"  max_best_dist={rep['max_best_dist']:.3e}")

        print("\n--- Comparison: Hardcoded build_fcc_syms_inv ---")
        reports_hardcoded = compare_with_orix(fcc_syms_inv_hardcoded, group_name=group_name, tol=tol)
        for name, rep in reports_hardcoded.items():
            print(f"{name}:")
            print(
                f"  matched {rep['n_matched']}/{rep['n_source']} "
                f"against target size {rep['n_target']}"
            )
            print(f"  mean_best_dist={rep['mean_best_dist']:.3e}")
            print(f"  max_best_dist={rep['max_best_dist']:.3e}")
    except Exception as exc:
        print(f"ORIX comparison skipped/failed: {exc}")