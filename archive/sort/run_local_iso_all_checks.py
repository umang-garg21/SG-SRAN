import os
import subprocess
import sys

import torch
from e3nn import o3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.local_iso_embedding_test_slow import (
    build_local_iso_fcc_embedding,
    build_local_iso_hcp_embedding,
)


def _header(title: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)


def _orthogonality_stats(mats: torch.Tensor) -> tuple[float, float, float]:
    eye = torch.eye(3, dtype=mats.dtype, device=mats.device)
    ortho_err = float((mats.transpose(-1, -2) @ mats - eye).abs().max().item())
    dets = torch.linalg.det(mats)
    return ortho_err, float(dets.min().item()), float(dets.max().item())


def _check_group_quats_active(emb, name: str) -> None:
    mats_from_quats = o3.quaternion_to_matrix(emb.group_quats)
    same_index_err = float((mats_from_quats - emb.group_mats).abs().max().item())
    ortho_err, det_min, det_max = _orthogonality_stats(mats_from_quats)

    print(f"{name}: group_quats shape={tuple(emb.group_quats.shape)}", flush=True)
    print(f"{name}: group_mats shape ={tuple(emb.group_mats.shape)}", flush=True)
    print(f"{name}: max |R(q)-stored_R| = {same_index_err:.3e}", flush=True)
    print(f"{name}: max |R^T R - I|    = {ortho_err:.3e}", flush=True)
    print(f"{name}: det(R) min/max     = ({det_min:.12f}, {det_max:.12f})", flush=True)

    if same_index_err > 1e-12:
        raise AssertionError(f"{name}: quaternion->matrix mismatch too large: {same_index_err}")
    if ortho_err > 1e-12:
        raise AssertionError(f"{name}: orthogonality error too large: {ortho_err}")
    if abs(det_min - 1.0) > 1e-10 or abs(det_max - 1.0) > 1e-10:
        raise AssertionError(f"{name}: determinant drift from +1 too large")


def _check_forward_quat_vs_matrix(emb, name: str, n: int = 32) -> None:
    q = torch.randn(n, 4, dtype=emb.dtype, device=emb.group_mats.device)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    y_q_raw = emb.forward_from_quaternions(q, raw=True)
    y_q_ir = emb.forward_from_quaternions(q, raw=False)

    r = o3.quaternion_to_matrix(q)
    y_r_raw = emb.forward_raw(r)
    y_r_ir = emb.forward_irreps(r)

    err_raw = float((y_q_raw - y_r_raw).abs().max().item())
    err_ir = float((y_q_ir - y_r_ir).abs().max().item())

    print(f"{name}: max |raw(q)-raw(R)|        = {err_raw:.3e}", flush=True)
    print(f"{name}: max |irreps(q)-irreps(R)| = {err_ir:.3e}", flush=True)

    if err_raw > 1e-10:
        raise AssertionError(f"{name}: raw quaternion/matrix path mismatch: {err_raw}")
    if err_ir > 1e-10:
        raise AssertionError(f"{name}: irreps quaternion/matrix path mismatch: {err_ir}")


def _check_local_iso_metrics(emb, name: str) -> None:
    eye = torch.eye(3, dtype=emb.dtype, device=emb.group_mats.device)

    g_raw = emb.gram_at_identity(use_raw=True, eps=1e-7)
    g_ir = emb.gram_at_identity(use_raw=False, eps=1e-7)

    err_raw = float((g_raw - eye).abs().max().item())
    err_ir = float((g_ir - eye).abs().max().item())

    print(f"{name}: ||G_raw-I||_max    = {err_raw:.3e}", flush=True)
    print(f"{name}: ||G_irreps-I||_max = {err_ir:.3e}", flush=True)

    raw_tol = 1e-8 if name == "FCC/O" else 1e-6
    if err_raw > raw_tol:
        raise AssertionError(f"{name}: raw Gram error too large: {err_raw} > {raw_tol}")

    # Keep this light so the run-all script finishes quickly.
    inv_raw = emb.right_invariance_error(use_raw=True, n_trials=1, seed=0)
    inv_ir = emb.right_invariance_error(use_raw=False, n_trials=1, seed=0)

    print(f"{name}: right-inv raw      = {inv_raw:.3e}", flush=True)
    print(f"{name}: right-inv irreps   = {inv_ir:.3e}", flush=True)

    if inv_raw > 1e-10:
        raise AssertionError(f"{name}: raw right-invariance too large: {inv_raw}")
    if inv_ir > 1e-10:
        raise AssertionError(f"{name}: irreps right-invariance too large: {inv_ir}")


def _check_feature_shapes(emb_o, emb_d6) -> None:
    r_o = torch.eye(3, dtype=emb_o.dtype, device=emb_o.group_mats.device).unsqueeze(0)
    r_d6 = torch.eye(3, dtype=emb_d6.dtype, device=emb_d6.group_mats.device).unsqueeze(0)

    shape_o_raw = tuple(emb_o.forward_raw(r_o).shape)
    shape_o_ir = tuple(emb_o.forward_irreps(r_o).shape)
    shape_d6_raw = tuple(emb_d6.forward_raw(r_d6).shape)
    shape_d6_ir = tuple(emb_d6.forward_irreps(r_d6).shape)

    print(f"FCC/O feature shapes  : raw={shape_o_raw}, irreps={shape_o_ir}", flush=True)
    print(f"HCP/D6 feature shapes : raw={shape_d6_raw}, irreps={shape_d6_ir}", flush=True)

    if shape_o_raw != (1, 81):
        raise AssertionError(f"Unexpected O raw shape: {shape_o_raw}")
    if shape_o_ir != (1, 14):
        raise AssertionError(f"Unexpected O irreps shape: {shape_o_ir}")
    if shape_d6_raw != (1, 738):
        raise AssertionError(f"Unexpected D6 raw shape: {shape_d6_raw}")
    if shape_d6_ir != (1, 32):
        raise AssertionError(f"Unexpected D6 irreps shape: {shape_d6_ir}")


def _run_pytest() -> None:
    _header("Running pytest suite")
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_local_iso_embedding.py", "-s"]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError("pytest failed for tests/test_local_iso_embedding.py")


def main() -> None:
    torch.set_printoptions(precision=6, sci_mode=False)

    use_cuda = torch.cuda.is_available() and os.environ.get("LOCAL_ISO_USE_CUDA", "0") == "1"
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float64

    _header("Build embeddings (active convention quaternion groups)")
    emb_o = build_local_iso_fcc_embedding(device=device, dtype=dtype)
    emb_d6 = build_local_iso_hcp_embedding(device=device, dtype=dtype, d6_convention="z_axis")

    print(f"device={device}, dtype={dtype}", flush=True)
    print(f"FCC/O irreps_out : {emb_o.irreps_out}", flush=True)
    print(f"HCP/D6 irreps_out: {emb_d6.irreps_out}", flush=True)

    _header("Check group quaternions -> active rotation matrices")
    _check_group_quats_active(emb_o, "FCC/O")
    _check_group_quats_active(emb_d6, "HCP/D6")

    _header("Check quaternion and matrix forward paths")
    _check_forward_quat_vs_matrix(emb_o, "FCC/O")
    _check_forward_quat_vs_matrix(emb_d6, "HCP/D6")

    _header("Check local isometry and right invariance")
    _check_local_iso_metrics(emb_o, "FCC/O")
    _check_local_iso_metrics(emb_d6, "HCP/D6")

    _header("Check feature shapes")
    _check_feature_shapes(emb_o, emb_d6)

    _run_pytest()

    _header("All checks passed")


if __name__ == "__main__":
    main()
