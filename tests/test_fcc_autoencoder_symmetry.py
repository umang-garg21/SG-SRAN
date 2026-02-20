import os
import sys

import torch
from e3nn import o3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.autoencoder import FCCAutoEncoder, FCCEncoder, FCCPhysics, wigner_D_cuda


def _normalized_random_quats(n: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn((n, 4), generator=gen, device=device)
    q = q / torch.norm(q, dim=1, keepdim=True).clamp_min(1e-12)
    return q


def _announce(message: str) -> None:
    print(f"[TEST] {message}")


def test_fcc_encoder_right_action_crystal_invariance():
    _announce("FCCEncoder is invariant under right action by cubic crystal symmetry (q -> qS)")
    device = torch.device("cpu")
    physics = FCCPhysics(device=str(device))
    encoder = FCCEncoder(physics)

    q = _normalized_random_quats(32, device=device, seed=11)
    f4_ref, f6_ref = encoder(q)

    # Right action by cubic crystal symmetry: q -> q * s
    syms = physics.fcc_syms
    max_err_f4 = 0.0
    max_err_f6 = 0.0

    for s in syms:
        s_batch = s.unsqueeze(0).expand_as(q)
        q_rs = FCCAutoEncoder.quat_mul(q, s_batch)
        f4_rs, f6_rs = encoder(q_rs)

        max_err_f4 = max(max_err_f4, (f4_rs - f4_ref).abs().max().item())
        max_err_f6 = max(max_err_f6, (f6_rs - f6_ref).abs().max().item())

    assert max_err_f4 < 1e-4, f"Right-action invariance failed for l=4: {max_err_f4}"
    assert max_err_f6 < 1e-4, f"Right-action invariance failed for l=6: {max_err_f6}"


def test_fcc_encoder_left_action_specimen_equivariance():
    _announce("FCCEncoder is equivariant under left action by specimen rotation (q -> Qq)")
    device = torch.device("cpu")
    physics = FCCPhysics(device=str(device))
    encoder = FCCEncoder(physics)

    q = _normalized_random_quats(24, device=device, seed=23)
    q_left = _normalized_random_quats(1, device=device, seed=99).expand_as(q)

    # Left action by specimen rotation: q -> Q * q
    q_lhs = FCCAutoEncoder.quat_mul(q_left, q)
    f4_lhs, f6_lhs = encoder(q_lhs)

    f4, f6 = encoder(q)

    R_left = o3.quaternion_to_matrix(q_left[:1])
    alpha, beta, gamma = o3.matrix_to_angles(R_left)
    D4_left = wigner_D_cuda(4, alpha, beta, gamma).squeeze(0)
    D6_left = wigner_D_cuda(6, alpha, beta, gamma).squeeze(0)

    f4_rhs = torch.einsum("ij,bj->bi", D4_left, f4)
    f6_rhs = torch.einsum("ij,bj->bi", D6_left, f6)

    err4 = (f4_lhs - f4_rhs).abs().max().item()
    err6 = (f6_lhs - f6_rhs).abs().max().item()

    assert err4 < 2e-4, f"Left-action equivariance failed for l=4: {err4}"
    assert err6 < 2e-4, f"Left-action equivariance failed for l=6: {err6}"


def test_match_closest_symmetry_recovers_orbit_member():
    _announce("FCCAutoEncoder symmetry matcher recovers equivalent orientation under FZ reduction")
    model = FCCAutoEncoder(device="cpu", grid_res=256)

    q_decoded = _normalized_random_quats(40, device=model.device, seed=7)
    syms = model.physics.fcc_syms

    idx = torch.arange(q_decoded.shape[0], device=model.device) % syms.shape[0]
    q_truth = FCCAutoEncoder.quat_mul(q_decoded, syms[idx])

    q_closest, errors, best_idx = model.match_closest_symmetry(q_decoded, q_truth)

    # Matching should be nearly exact in misorientation after symmetry handling.
    assert torch.max(errors).item() < 1e-3, f"Large symmetry match error: {torch.max(errors).item()}"
    assert torch.all((best_idx >= 0) & (best_idx < 24)).item(), "Invalid symmetry indices returned"


def test_match_closest_symmetry_bunge_convention():
    _announce("FCCAutoEncoder symmetry matcher is robust in Bunge convention")
    model = FCCAutoEncoder(device="cpu", grid_res=256)

    q_decoded = _normalized_random_quats(48, device=model.device, seed=123)
    syms = model.physics.fcc_syms
    idx = torch.arange(q_decoded.shape[0], device=model.device) % syms.shape[0]
    q_truth = FCCAutoEncoder.quat_mul(q_decoded, syms[idx])

    q_closest, errors, _ = model.match_closest_symmetry(q_decoded, q_truth)

    assert torch.max(errors).item() < 1e-3, f"Large Bunge/auto symmetry match error: {torch.max(errors).item()}"
    assert torch.all(torch.isfinite(q_closest)).item(), "Non-finite quaternion values returned"


def test_spherical_decoder_uses_f6_signal():
    _announce("Spherical decoder output changes when only f6 changes (f4 fixed)")
    model = FCCAutoEncoder(device="cpu", grid_res=512)

    n = 12
    f4 = torch.zeros((n, 9), device=model.device)
    f6_a = _normalized_random_quats(n, device=model.device, seed=101)
    f6_b = _normalized_random_quats(n, device=model.device, seed=202)

    # Lift 4D random seeds into 13D nontrivial l=6-like vectors.
    f6_a = torch.nn.functional.pad(f6_a, (0, 9), mode="constant", value=0.0)
    f6_b = torch.nn.functional.pad(f6_b, (0, 9), mode="constant", value=0.0)

    q_a = model.decode(f4, f6_a)
    q_b = model.decode(f4, f6_b)

    dot = (q_a * q_b).sum(dim=1).abs()
    assert torch.any(dot < 0.999), "Decoder output appears independent of f6"

