# import numpy as np
# from scipy.io import savemat
# import os

# # ---- INPUT ----
# npy_path = r"\data\warren\materials\EBSD\IN718_FZ_2D_SR_x4\Test\HR_Data\IN718_FZ_2D_SR_x4_test_hr_x_block_0.npy".replace("\\", "/") 

# # npy_path = r"\data\warren\materials\EBSD\IN718_FZ_2D_SR_x4\Test\Original_Data\Open_718_Test_hr_x_block_0.npy".replace("\\", "/") 
# # ---- LOAD ----
# arr = np.load(npy_path)

# print("Loaded:", npy_path)
# print("Shape:", arr.shape)
# print("Dtype:", arr.dtype)

# # ---- SAVE ----
# mat_path = os.path.splitext(npy_path)[0] + ".mat"

# savemat(mat_path, {
#     "data": arr
# })

# print("Saved:", mat_path)

import numpy as np

# =========================
# Quaternion utilities (w,x,y,z scalar-first)
# =========================
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return q / n

def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def quat_mul(q1, q2):
    """Hamilton product, scalar-first (w,x,y,z)."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def quat_angle_deg(q):
    """
    Rotation angle (deg) of unit quaternion q (wxyz).
    For relative rotation q_rel, angle = 2*acos(|w|).
    """
    q = quat_normalize(q)
    w = np.clip(np.abs(q[..., 0]), -1.0, 1.0)
    ang = 2.0 * np.arccos(w)
    return ang * (180.0 / np.pi)

def enforce_w_positive(q):
    """Canonical sign (q and -q same rotation)."""
    q = np.asarray(q, dtype=np.float64)
    sgn = np.where(q[..., 0:1] < 0, -1.0, 1.0)
    return q * sgn

# =========================
# Build 24 proper cubic symmetries as quaternions
# via signed permutation matrices -> rotmat -> quat
# =========================
def rotmat_to_quat_wxyz(R):
    """Convert rotation matrix to unit quaternion (w,x,y,z), scalar-first."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    q = quat_normalize(q)
    q = enforce_w_positive(q)
    return q

def cubic_proper_symmetry_quats():
    """Return (24,4) quaternions (wxyz) for proper cubic rotations."""
    from itertools import permutations, product
    mats = []
    for perm in permutations([0, 1, 2]):
        P = np.zeros((3, 3), dtype=int)
        for i, j in enumerate(perm):
            P[i, j] = 1
        for signs in product([-1, 1], repeat=3):
            S = np.diag(signs)
            R = S @ P
            if round(np.linalg.det(R)) == 1:
                mats.append(R.astype(np.float64))
    # dedupe
    uniq = []
    for R in mats:
        if not any(np.allclose(R, U) for U in uniq):
            uniq.append(R)
    assert len(uniq) == 24

    Q = np.stack([rotmat_to_quat_wxyz(R) for R in uniq], axis=0)  # (24,4)
    return Q

# =========================
# Load + layout normalization + ordering heuristic
# =========================
def load_quats_any_layout(npy_path):
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    if arr.shape[2] == 4:
        d = arr  # (H,W,4)
    elif arr.shape[0] == 4:
        d = np.transpose(arr, (1, 2, 0))  # (4,H,W)->(H,W,4)
    else:
        raise ValueError(f"No dimension of size 4 found. shape={arr.shape}")

    return d.astype(np.float64)  # (H,W,4)

def reorder_to_wxyz(d_hw4):
    """
    Guess whether data are stored as [x,y,z,w] or [w,x,y,z] using scalar mean magnitude.
    Return quats in wxyz and normalized.
    """
    flat = d_hw4.reshape(-1, 4)
    means = np.abs(flat.mean(axis=0))
    scalar_idx = int(np.argmax(means))  # 0-based

    if scalar_idx == 3:
        # xyzw -> wxyz
        q = np.stack([d_hw4[..., 3], d_hw4[..., 0], d_hw4[..., 1], d_hw4[..., 2]], axis=-1)
        msg = "xyzw (scalar-last) -> wxyz"
    elif scalar_idx == 0:
        q = d_hw4
        msg = "wxyz (scalar-first)"
    else:
        q = d_hw4
        msg = f"unclear (scalar idx={scalar_idx}); leaving as-is"
    q = quat_normalize(q)
    q = enforce_w_positive(q)
    return q, msg, scalar_idx

# =========================
# Quaternion-only equivalence check
# =========================
def confirm_bunge_symmetry_quat_only(
    npy_path: str,
    num_pixels: int = 25,
    tol_deg: float = 1e-4,
    seed: int = 0,
):
    qB_map = load_quats_any_layout(npy_path)          # (H,W,4) in stored order
    qB_map, msg, scalar_idx = reorder_to_wxyz(qB_map) # now wxyz, normalized

    H, W, _ = qB_map.shape
    print("Loaded:", npy_path)
    print("Shape:", qB_map.shape, "(H,W,4)")
    print("Ordering heuristic:", msg, f"(scalar_idx={scalar_idx} 0-based)")

    # 24 proper cubic sym ops as quats s (wxyz)
    symQ = cubic_proper_symmetry_quats()  # (24,4)
    # their inverses are conjugates (unit quats)
    symQ_inv = quat_conj(symQ)

    rng = np.random.default_rng(seed)
    ii = rng.integers(1, H - 1, size=num_pixels)
    jj = rng.integers(1, W - 1, size=num_pixels)

    errs = []

    for p in range(num_pixels):
        qB = qB_map[ii[p], jj[p], :]  # Bunge (S->C)

        # MTEX equivalent is inverse:
        qM = quat_conj(qB)

        for k in range(24):
            s = symQ[k]
            s_inv = symQ_inv[k]

            # (A) Bunge-side: qB' = s^{-1} ⊗ qB, then convert to MTEX: qM_A = inv(qB')
            qB_prime = quat_mul(s_inv, qB)
            qM_A = quat_conj(qB_prime)

            # (B) MTEX-side: qM' = qM ⊗ s
            qM_B = quat_mul(qM, s)

            # Compare by relative rotation: q_rel = inv(qM_A) ⊗ qM_B
            q_rel = quat_mul(quat_conj(qM_A), qM_B)
            errs.append(quat_angle_deg(q_rel))

    errs = np.array(errs, dtype=np.float64)
    max_err = float(np.max(errs))
    mean_err = float(np.mean(errs))
    p99_err = float(np.quantile(errs, 0.99))

    print(f"\nQuaternion-only symmetry equivalence over {num_pixels} pixels x 24 ops:")
    print(f"  mean mismatch = {mean_err:.3e} deg")
    print(f"  p99  mismatch = {p99_err:.3e} deg")
    print(f"  max  mismatch = {max_err:.3e} deg")

    if max_err < tol_deg:
        print("\nPASS ✅  Bunge symmetry confirmed (quaternion form):")
        print("  qB' = s^{-1} ⊗ qB  and  (qB')^{-1} == qB^{-1} ⊗ s")
    else:
        print("\nFAIL ❌  mismatch too large.")
        print("Likely causes: wrong component ordering, wrong Bunge assumption, or inconsistent quaternion conventions.")

    return {
        "ordering": msg,
        "scalar_idx_0based": scalar_idx,
        "mean_err_deg": mean_err,
        "p99_err_deg": p99_err,
        "max_err_deg": max_err,
    }

if __name__ == "__main__":
    confirm_bunge_symmetry_quat_only(
        "/data/warren/materials/EBSD/IN718_2D_SR_x4/Test/Original_Data/Open_718_Test_hr_x_block_0.npy",
        num_pixels=25,
        tol_deg=1e-4,
        seed=0,
    )


    