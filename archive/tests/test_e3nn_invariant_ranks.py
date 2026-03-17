import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("e3nn")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "models" / "e3nn_invariant_autoencoder.py"
_spec = importlib.util.spec_from_file_location("e3nn_invariant_autoencoder", MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
E3nnInvariantAutoencoderBunge = _module.E3nnInvariantAutoencoderBunge


def _build_model(normalize_wigner_features: bool = False) -> E3nnInvariantAutoencoderBunge:
    return E3nnInvariantAutoencoderBunge(
        device="cpu",
        Ls=(4, 6, 8, 10, 12),
        latent_dim=8,
        encoder_hidden_dim=16,
        encoder_layers=1,
        decoder_hidden_dim=16,
        decoder_layers=1,
        normalize_wigner_features=normalize_wigner_features,
    )


def _group_wigner_matrices(model: E3nnInvariantAutoencoderBunge, l: int) -> torch.Tensor:
    sym_bunge = model._normalize_quaternions(model.fcc_syms_inv.to(torch.float64))
    sym_active = model._to_active_convention(sym_bunge)
    R = _module.o3.quaternion_to_matrix(sym_active)
    alpha, beta, gamma = _module.o3.matrix_to_angles(R)
    return _module._wigner_D(l, alpha, beta, gamma).to(torch.complex128)


def _normalized_random_quats(n: int, seed: int = 123) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q = torch.randn((n, 4), generator=gen)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _min_orbit_misorientation_bunge(
    model: E3nnInvariantAutoencoderBunge,
    q_pred: torch.Tensor,
    q_target: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetry-aware reconstruction metric for Bunge/passive quaternions:
        min_s 2*acos(|<q_pred, s^{-1} ⊗ q_target>|)
    """
    q_pred64 = model._normalize_quaternions(q_pred.to(torch.float64))
    q_target64 = model._normalize_quaternions(q_target.to(torch.float64))

    n = q_pred64.shape[0]
    g = model.fcc_syms_inv.shape[0]
    syms = model.fcc_syms_inv.to(torch.float64).unsqueeze(0).expand(n, -1, -1)
    q_exp = q_target64.unsqueeze(1).expand(-1, g, -1)

    # Bunge passive crystal orbit: q_equiv = s^{-1} ⊗ q_target
    orbit = model.quat_mul(syms, q_exp)
    orbit = model._normalize_quaternions(orbit)

    dots = (q_pred64.unsqueeze(1) * orbit).sum(dim=-1).abs().clamp(0.0, 1.0)
    angles = 2.0 * torch.acos(dots)
    return angles.min(dim=1).values


def test_cubic_invariant_subspace_ranks_are_stable():
    ls = (4, 6, 8, 10, 12)
    expected_ranks = {4: 1, 6: 1, 8: 1, 10: 1, 12: 2}

    model = _build_model(normalize_wigner_features=False)

    assert model.fcc_syms_inv.shape[0] == 24, "Expected 24 proper cubic symmetry operators"

    observed_ranks = {l: int(getattr(model, f"U_{l}").shape[-1]) for l in ls}
    assert observed_ranks == expected_ranks, (
        f"Unexpected invariant ranks for cubic group. "
        f"expected={expected_ranks}, observed={observed_ranks}"
    )

    expected_out_dim = sum(2 * (2 * l + 1) * expected_ranks[l] for l in ls)
    assert model.wigner_out_dim == expected_out_dim


def test_reynolds_projector_is_hermitian_and_idempotent():
    model = _build_model(normalize_wigner_features=False)

    for l in model.Ls:
        D = _group_wigner_matrices(model, l)
        P = D.mean(dim=0)
        herm_err = (P - P.conj().transpose(-2, -1)).abs().max().item()
        idem_err = (P @ P - P).abs().max().item()

        assert herm_err < 5e-6, f"Hermitian residual too large for l={l}: {herm_err}"
        assert idem_err < 5e-6, f"Idempotence residual too large for l={l}: {idem_err}"


def test_invariant_basis_columns_are_orthonormal_and_group_fixed():
    model = _build_model(normalize_wigner_features=False)

    for l in model.Ls:
        U = getattr(model, f"U_{l}").to(torch.complex128)
        I = torch.eye(U.shape[-1], dtype=torch.complex128, device=U.device)
        ortho_err = (U.conj().transpose(-2, -1) @ U - I).abs().max().item()

        D = _group_wigner_matrices(model, l)
        inv_err = (D @ U.unsqueeze(0) - U.unsqueeze(0)).abs().max().item()

        assert ortho_err < 5e-6, f"Basis orthonormality residual too large for l={l}: {ortho_err}"
        assert inv_err < 1e-5, f"Group invariance residual too large for l={l}: {inv_err}"


@pytest.mark.parametrize("normalize_features", [False, True])
def test_wigner_features_are_invariant_under_cubic_left_action_bunge(normalize_features: bool):
    model = _build_model(normalize_wigner_features=normalize_features)
    q = _normalized_random_quats(16, seed=123)
    f_ref = model._wigner_invariant_features(q)

    max_err = 0.0
    for s in model.fcc_syms_inv:
        s_batch = s.unsqueeze(0).expand_as(q)
        q_sym = model.quat_mul(s_batch, q)
        f_sym = model._wigner_invariant_features(q_sym)
        max_err = max(max_err, (f_sym - f_ref).abs().max().item())

    assert max_err < 5e-5, f"Feature invariance residual too large: {max_err}"


def test_symmetry_aware_reconstruction_metric_is_orbit_invariant():
    model = _build_model(normalize_wigner_features=False)
    q_target = _normalized_random_quats(32, seed=321)

    with torch.no_grad():
        q_pred = model(q_target)

    loss_ref = _min_orbit_misorientation_bunge(model, q_pred, q_target)

    idx = torch.arange(q_target.shape[0]) % model.fcc_syms_inv.shape[0]
    q_target_equiv = model.quat_mul(model.fcc_syms_inv[idx], q_target)
    loss_equiv = _min_orbit_misorientation_bunge(model, q_pred, q_target_equiv)

    diff = (loss_ref - loss_equiv).abs().max().item()
    assert diff < 1e-6, f"Orbit-invariance failed for symmetry-aware metric: {diff}"


def test_symmetry_aware_reconstruction_metric_is_zero_for_exact_orbit_member():
    model = _build_model(normalize_wigner_features=False)
    q_target = _normalized_random_quats(32, seed=777)

    idx = torch.arange(q_target.shape[0]) % model.fcc_syms_inv.shape[0]
    q_pred_orbit = model.quat_mul(model.fcc_syms_inv[idx], q_target)

    loss = _min_orbit_misorientation_bunge(model, q_pred_orbit, q_target)
    assert loss.max().item() < 1e-6, (
        "Expected near-zero symmetry-aware reconstruction loss when prediction "
        "is an exact crystal-symmetry orbit member."
    )
