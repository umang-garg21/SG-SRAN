import math
from pathlib import Path

import numpy as np
import torch

from utils.symmetry_utils import resolve_symmetry


def build_hcp_syms() -> torch.Tensor:
    sqrt3_2 = math.sqrt(3.0) / 2.0
    half = 0.5
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [sqrt3_2, 0.0, 0.0, half],
            [half, 0.0, 0.0, sqrt3_2],
            [0.0, 0.0, 0.0, 1.0],
            [half, 0.0, 0.0, -sqrt3_2],
            [sqrt3_2, 0.0, 0.0, -half],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, half, sqrt3_2, 0.0],
            [0.0, -half, sqrt3_2, 0.0],
            [0.0, sqrt3_2, half, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -sqrt3_2, half, 0.0],
        ],
        dtype=torch.float32,
    )
def build_hcp_syms_mtex() -> torch.Tensor:
    sqrt3_2 = math.sqrt(3.0) / 2.0
    half = 0.5
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, sqrt3_2, -half, 0.0],
            [sqrt3_2, 0.0, 0.0, half],
            [0.0, half, -sqrt3_2, 0.0],
            [half, 0.0, 0.0, sqrt3_2],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -half, -sqrt3_2, 0.0],
            [-half, 0.0, 0.0, sqrt3_2],
            [0.0, -sqrt3_2, -half, 0.0],
            [-sqrt3_2, 0.0, 0.0, half],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

def build_hcp_syms_inv_mtex() -> torch.Tensor:
    sqrt3_2 = math.sqrt(3.0) / 2.0
    half = 0.5
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -sqrt3_2, half, 0.0],
            [sqrt3_2, 0.0, 0.0, -half],
            [0.0, -half, sqrt3_2, 0.0],
            [half, 0.0, 0.0, -sqrt3_2],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, half, sqrt3_2, 0.0],
            [-half, 0.0, 0.0, -sqrt3_2],
            [0.0, sqrt3_2, half, 0.0],
            [-sqrt3_2, 0.0, 0.0, -half],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def normalize_quats(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return (q / norms).astype(np.float64, copy=False)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    q_inv = np.asarray(q, dtype=np.float64).copy()
    q_inv[..., 1:] *= -1.0
    return q_inv


def as_rotation_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return sign-invariant quaternion distance proxy matrix: 1 - |dot(a,b)|."""
    a = normalize_quats(a)
    b = normalize_quats(b)
    abs_dot = np.abs(a @ b.T)
    return 1.0 - abs_dot


def set_match_report(
    source: np.ndarray, target: np.ndarray, tol: float = 1e-6
) -> dict[str, float | int]:
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


def rowwise_report(
    source: np.ndarray, target: np.ndarray, tol: float = 1e-6
) -> dict[str, float | int]:
    source = normalize_quats(source)
    target = normalize_quats(target)
    row_dist = 1.0 - np.abs(np.sum(source * target, axis=-1))
    n_matched = int(np.sum(row_dist <= tol))
    return {
        "n_source": int(source.shape[0]),
        "n_matched": n_matched,
        "max_row_dist": float(row_dist.max()),
        "mean_row_dist": float(row_dist.mean()),
    }


def compare_with_orix(
    hcp_syms_inv: np.ndarray, group_name: str = "hcp", tol: float = 1e-6
) -> dict[str, dict[str, float | int]]:
    orix_group = resolve_symmetry(group_name)
    orix_ops = np.asarray(orix_group.data, dtype=np.float64)[:12]
    orix_inv = quat_inverse(orix_ops)

    return {
        "hcp_inv_vs_orix_ops": set_match_report(hcp_syms_inv, orix_ops, tol=tol),
        "hcp_inv_vs_orix_inv": set_match_report(hcp_syms_inv, orix_inv, tol=tol),
    }


def _print_report(title: str, rep: dict[str, float | int], tol: float) -> bool:
    passed = (
        rep.get("n_matched", 0) == rep.get("n_source", 0)
        and rep.get("max_best_dist", rep.get("max_row_dist", 1.0)) <= tol
    )
    status = "PASS" if passed else "FAIL"
    print(f"{title}: {status}")
    print(
        f"  matched {rep.get('n_matched', 0)}/{rep.get('n_source', 0)}"
        + (
            f" against target size {rep['n_target']}"
            if "n_target" in rep
            else ""
        )
    )
    if "mean_best_dist" in rep:
        print(f"  mean_best_dist={rep['mean_best_dist']:.3e}")
    if "max_best_dist" in rep:
        print(f"  max_best_dist={rep['max_best_dist']:.3e}")
    if "mean_row_dist" in rep:
        print(f"  mean_row_dist={rep['mean_row_dist']:.3e}")
    if "max_row_dist" in rep:
        print(f"  max_row_dist={rep['max_row_dist']:.3e}")
    return passed


def run_tests(group_name: str = "hcp", tol: float = 1e-6) -> bool:
    hcp_syms = build_hcp_syms().cpu().numpy().astype(np.float64, copy=False)
    inv_computed = quat_inverse(hcp_syms)
    inv_hardcoded = build_hcp_syms_inv().cpu().numpy().astype(np.float64, copy=False)
    inv_mtex = build_hcp_syms_inv_mtex().cpu().numpy().astype(np.float64, copy=False)

    all_passed = True

    print("--- Pairwise Set Comparison (order/sign invariant) ---")
    all_passed &= _print_report(
        "computed_inv vs hardcoded_inv",
        set_match_report(inv_computed, inv_hardcoded, tol=tol),
        tol,
    )
    all_passed &= _print_report(
        "computed_inv vs mtex_inv",
        set_match_report(inv_computed, inv_mtex, tol=tol),
        tol,
    )
    all_passed &= _print_report(
        "hardcoded_inv vs mtex_inv",
        set_match_report(inv_hardcoded, inv_mtex, tol=tol),
        tol,
    )

    print("\n--- Rowwise Comparison (same index only) ---")
    _print_report(
        "hardcoded_inv vs mtex_inv (rowwise)",
        rowwise_report(inv_hardcoded, inv_mtex, tol=tol),
        tol,
    )

    c_to_h = as_rotation_distance_matrix(inv_mtex, inv_hardcoded).argmin(axis=1)
    print(f"  best row map mtex -> hardcoded: {c_to_h.tolist()}")

    print("\n--- ORIX Comparison ---")
    for name, table in [
        ("computed_inv", inv_computed),
        ("hardcoded_inv", inv_hardcoded),
        ("mtex_inv", inv_mtex),
    ]:
        print(f"{name}:")
        try:
            reports = compare_with_orix(table, group_name=group_name, tol=tol)
            for report_name, rep in reports.items():
                all_passed &= _print_report(f"  {report_name}", rep, tol)
        except Exception as exc:
            all_passed = False
            print(f"  FAIL: ORIX comparison skipped/failed: {exc}")

    return all_passed


def main() -> None:
    out_path = Path("symmetry_groups/hcp_syms_inv.npy")
    group_name = "hcp"
    tol = 1e-6

    hcp_syms = build_hcp_syms().cpu().numpy().astype(np.float64, copy=False)
    inv_computed = quat_inverse(hcp_syms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, inv_computed.astype(np.float32, copy=False))

    print(f"Saved HCP computed inverse symmetries to: {out_path.resolve()}")
    print(f"Shape: {inv_computed.shape}")
    print("")

    ok = run_tests(group_name=group_name, tol=tol)
    print("\nOverall:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
