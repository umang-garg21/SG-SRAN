import itertools
import sympy as sp
from sympy import I, Matrix
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.physics.wigner import wigner_d_small

# -----------------------------------------------------------------------------
# 1) Proper octahedral group O (24 proper cubic rotations): signed permutations
# -----------------------------------------------------------------------------
def proper_octahedral_group_O():
    mats = []
    for perm in itertools.permutations([0, 1, 2], 3):
        P = sp.zeros(3, 3)
        for i, j in enumerate(perm):
            P[i, j] = 1
        for signs in itertools.product([-1, 1], repeat=3):
            S = sp.diag(*signs)
            R = S * P
            if int(R.det()) == 1:
                mats.append(Matrix(R))

    # Deduplicate
    uniq = []
    for R in mats:
        if not any((R - Q).is_zero_matrix for Q in uniq):
            uniq.append(R)

    assert len(uniq) == 24, f"Expected 24, got {len(uniq)}"
    return uniq

# -----------------------------------------------------------------------------
# 2) Rotation matrix -> ZYZ Euler angles (alpha,beta,gamma), R = Rz(a)Ry(b)Rz(g)
#    Robust for cubic rotations with entries in {-1,0,1}
# -----------------------------------------------------------------------------
def rotmat_to_zyz(R: Matrix):
    r13, r23, r33 = R[0, 2], R[1, 2], R[2, 2]
    r31, r32 = R[2, 0], R[2, 1]
    r21, r11 = R[1, 0], R[0, 0]

    beta = sp.acos(sp.simplify(r33))
    sb = sp.simplify(sp.sin(beta))

    if sb == 0:
        # beta = 0 or pi: choose gamma = 0 and absorb into alpha
        alpha = sp.atan2(r21, r11)
        gamma = sp.Integer(0)
        return sp.simplify(alpha), sp.simplify(beta), sp.simplify(gamma)

    alpha = sp.atan2(r23, r13)
    gamma = sp.atan2(r32, -r31)
    return sp.simplify(alpha), sp.simplify(beta), sp.simplify(gamma)

# -----------------------------------------------------------------------------
# 3) Build Wigner D^l(alpha,beta,gamma) from little-d matrix
#    Ordering is m = l, l-1, ..., -l  (same as SymPy)
# -----------------------------------------------------------------------------
def wigner_D_matrix(l: int, alpha, beta, gamma):
    ms = list(range(l, -l - 1, -1))         # l, l-1, ..., -l
    dmat = wigner_d_small(l, beta)          # (2l+1)x(2l+1)

    D = sp.zeros(2 * l + 1, 2 * l + 1)
    for i, m in enumerate(ms):
        for j, mp in enumerate(ms):
            D[i, j] = sp.exp(-I * m * alpha) * dmat[i, j] * sp.exp(-I * mp * gamma)
    return sp.simplify(D)

# -----------------------------------------------------------------------------
# 4) Reynolds projector onto right-action invariant subspace (A1/trivial irrep)
# -----------------------------------------------------------------------------
def reynolds_projector(l: int, group_mats):
    P = sp.zeros(2 * l + 1, 2 * l + 1)
    for R in group_mats:
        a, b, g = rotmat_to_zyz(R)
        P += wigner_D_matrix(l, a, b, g)
    return sp.simplify(P / sp.Integer(len(group_mats)))

# -----------------------------------------------------------------------------
# 5) Invariant basis vectors: nullspace of (P_l - I)
# -----------------------------------------------------------------------------
def invariant_basis_vectors(l: int, group_mats):
    P = reynolds_projector(l, group_mats)
    N = (P - sp.eye(2 * l + 1)).nullspace()

    basis = []
    for v in N:
        v = Matrix(v)
        # Normalize by first nonzero entry to keep it deterministic-ish
        for k in range(v.rows):
            if sp.simplify(v[k]) != 0:
                v = sp.simplify(v / v[k])
                break
        basis.append(v)

    return P, basis

# -----------------------------------------------------------------------------
# 6) Adapted harmonic on the sphere: Y(theta,phi) = Σ_m u_m Y_{l m}
#    Here u is in m = l..-l ordering, matching the code above.
# -----------------------------------------------------------------------------
def adapted_harmonic_Ylm(l: int, u_vec: Matrix, theta, phi):
    ms = list(range(l, -l - 1, -1))  # l..-l
    expr = 0
    for i, m in enumerate(ms):
        expr += sp.simplify(u_vec[i] * Ynm(l, m, theta, phi))
    return sp.simplify(expr)


def invariant_spherical_harmonics(l: int, group_mats, theta=None, phi=None):
    """
    Return only the G-invariant spherical harmonics at degree l.

    Each expression is a basis element of the invariant subspace:
        Y_l^inv(theta,phi) = sum_m u_m Y_{l m}(theta,phi).
    """
    if theta is None:
        theta = sp.symbols("theta", real=True)
    if phi is None:
        phi = sp.symbols("phi", real=True)

    _, basis = invariant_basis_vectors(l, group_mats)
    return [adapted_harmonic_Ylm(l, u, theta, phi) for u in basis]


def complex_invariant_to_real_seed(
    l: int,
    u_vec: Matrix,
    normalize: bool = True,
    center_positive: bool = True,
):
    """
    Convert invariant coefficients from complex SH basis to real SH seed vector.

    Input u_vec ordering is m=l..-l (SymPy/Wigner convention in this file).
    Output s has e3nn-style index layout i = l + m for m=-l..l.

    For m>0:
      cos-like (real) coefficient   at index l+m
      sin-like (imag) coefficient   at index l-m
    """
    ms_desc = list(range(l, -l - 1, -1))  # l..-l
    coeff = {m: sp.simplify(u_vec[i]) for i, m in enumerate(ms_desc)}

    s = sp.zeros(2 * l + 1, 1)
    s[l] = sp.simplify(coeff[0])
    for m in range(1, l + 1):
        up = sp.simplify(coeff[m])
        um = sp.simplify(coeff[-m])
        parity = sp.Integer((-1) ** m)

        # Real SH decomposition from complex Y_l^{±m}
        c_cos = sp.simplify((up + parity * um) / sp.sqrt(2))
        c_sin = sp.simplify((up - parity * um) / (sp.sqrt(2) * I))

        s[l + m] = sp.simplify(c_cos)
        s[l - m] = sp.simplify(c_sin)

    if normalize:
        norm2 = sp.simplify((s.T * s)[0])
        if norm2 != 0:
            s = sp.simplify(s / sp.sqrt(norm2))

    if center_positive and sp.simplify(s[l]) != 0 and sp.N(s[l]) < 0:
        s = -s

    return sp.simplify(s)


def encoder_seed_from_symbolic(
    l: int,
    group_mats=None,
    normalize: bool = True,
    center_positive: bool = True,
):
    """
    Compute the seed vector used by FCCEncoder for degree l from symbolic basis.

    Returns a (2l+1, 1) real SH vector with index layout i=l+m (m=-l..l).
    """
    if group_mats is None:
        group_mats = proper_octahedral_group_O()

    _, basis = invariant_basis_vectors(l, group_mats)
    if not basis:
        raise ValueError(f"No invariant basis exists for l={l}.")
    if len(basis) > 1:
        raise ValueError(
            f"Invariant subspace for l={l} has dimension {len(basis)}; "
            "seed is not unique."
        )

    return complex_invariant_to_real_seed(
        l=l,
        u_vec=basis[0],
        normalize=normalize,
        center_positive=center_positive,
    )

# -----------------------------------------------------------------------------
# 7) Right-invariant SO(3) feature: f_l(R) = D^l(R) u, so f_l(R s) = f_l(R)
# -----------------------------------------------------------------------------
def right_invariant_feature_f(l: int, u_vec: Matrix, alpha, beta, gamma):
    D = wigner_D_matrix(l, alpha, beta, gamma)
    return sp.simplify(D * u_vec)

# -----------------------------------------------------------------------------
# Demo for l=4 and l=6
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sp.init_printing(use_unicode=True)
    O = proper_octahedral_group_O()

    theta, phi = sp.symbols("theta phi", real=True)

    for l in [4, 6]:
        P_l, basis = invariant_basis_vectors(l, O)
        print(f"\n=== l={l} ===")
        print("rank(P_l) =", P_l.rank())
        print("dim invariant subspace =", len(basis))

        if not basis:
            print("No A1 (fully symmetric) invariant at this l.")
            continue

        Y_inv = invariant_spherical_harmonics(l, O, theta, phi)
        for idx, expr in enumerate(Y_inv):
            print(f"\nInvariant spherical harmonic #{idx} (l={l}):")
            sp.pprint(expr, use_unicode=True)

        s = encoder_seed_from_symbolic(l, O)
        print(f"\nSeed s{l} for FCCEncoder (exact, index i=l+m):")
        sp.pprint(s.T, use_unicode=True)
        print(f"Seed s{l} numeric:")
        print([float(sp.N(v, 8)) for v in list(s)])
