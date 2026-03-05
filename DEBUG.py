import numpy as np

# NPY_PATH = r"\data\warren\materials\EBSD\IN718_2D_SR_x4\Test\Original_Data\Open_718_Test_hr_x_block_0.npy".replace("\\", "/")  # <-- change me


# NPY_PATH = r"\data\warren\materials\EBSD\IN718_FZ_2D_SR_x4\Test\HR_Data\IN718_FZ_2D_SR_x4_test_hr_x_block_0.npy".replace("\\", "/") 

NPY_PATH = r"\data\warren\materials\EBSD\IN718_FZ_2D_SR_x4\Test\Original_Data\Open_718_Test_hr_x_block_0.npy".replace("\\", "/") 
# ----------------------------
# Quaternion + rotation utils
# ----------------------------
def quat_norm(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return q / n

def quat_conj(q):
    q = np.asarray(q, float)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def quat_mul(q1, q2):
    """Hamilton product, scalar-first: (w,x,y,z)."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def quat_to_R_wxyz(q):
    """q is (...,4) with scalar-first. Returns (...,3,3)."""
    q = quat_norm(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1 - 2*(yy + zz)
    R[..., 0, 1] = 2*(xy - wz)
    R[..., 0, 2] = 2*(xz + wy)
    R[..., 1, 0] = 2*(xy + wz)
    R[..., 1, 1] = 1 - 2*(xx + zz)
    R[..., 1, 2] = 2*(yz - wx)
    R[..., 2, 0] = 2*(xz - wy)
    R[..., 2, 1] = 2*(yz + wx)
    R[..., 2, 2] = 1 - 2*(xx + yy)
    return R

def rotation_quality(R):
    """Returns orthogonality error and det stats."""
    I = np.eye(3)
    RtR = np.matmul(np.swapaxes(R, -1, -2), R)
    ortho_err = np.linalg.norm(RtR - I, axis=(-2, -1))
    det = np.linalg.det(R)
    return ortho_err, det

# ----------------------------
# Candidate component orderings
# ----------------------------
# We'll assume the file stores 4 channels somehow; try common permutations.
CANDIDATES = {
    "wxyz (scalar-first)": (0, 1, 2, 3),
    "xyzw (scalar-last)":  (3, 0, 1, 2),  # interpret stored [x,y,z,w] as [w,x,y,z]
    "wyxz":                (0, 1, 3, 2),
    "wzyx":                (0, 3, 2, 1),
    "yzwx":                (2, 3, 0, 1),
}

def summarize_candidate(q_raw, perm, name, sample_n=20000, seed=0):
    rng = np.random.default_rng(seed)
    q = q_raw[..., list(perm)]
    q = q.reshape(-1, 4)

    # subsample
    if q.shape[0] > sample_n:
        idx = rng.choice(q.shape[0], size=sample_n, replace=False)
        q = q[idx]

    # norms before normalization
    norms = np.linalg.norm(q, axis=1)
    norm_mean = float(np.mean(norms))
    norm_std  = float(np.std(norms))
    frac_near1 = float(np.mean(np.abs(norms - 1.0) < 1e-3))

    # rotation matrix validity AFTER normalization (ordering should still matter!)
    qn = quat_norm(q)
    R = quat_to_R_wxyz(qn)
    ortho_err, det = rotation_quality(R)
    ortho_mean = float(np.mean(ortho_err))
    ortho_p99  = float(np.quantile(ortho_err, 0.99))
    det_mean   = float(np.mean(det))
    det_p01    = float(np.quantile(det, 0.01))
    det_p99    = float(np.quantile(det, 0.99))
    frac_det_pos = float(np.mean(det > 0.0))
    frac_det_near1 = float(np.mean(np.abs(det - 1.0) < 1e-3))

    return {
        "name": name,
        "perm": perm,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "frac_norm_near1": frac_near1,
        "ortho_err_mean": ortho_mean,
        "ortho_err_p99": ortho_p99,
        "det_mean": det_mean,
        "det_p01": det_p01,
        "det_p99": det_p99,
        "frac_det_pos": frac_det_pos,
        "frac_det_near1": frac_det_near1,
    }

def score(rep):
    """
    Lower is better.
    Emphasize rotation validity (orthogonality + det close to +1),
    and norm closeness as secondary.
    """
    return (
        1e3 * rep["ortho_err_mean"]
        + 5e2 * abs(rep["det_mean"] - 1.0)
        + 1e2 * (1.0 - rep["frac_det_near1"])
        + 1e1 * abs(rep["norm_mean"] - 1.0)
        + 1e1 * rep["norm_std"]
    )

# ----------------------------
# Main
# ----------------------------
arr = np.load(NPY_PATH)
print(f"Loaded: {NPY_PATH}")
print(f"Shape: {arr.shape}  dtype={arr.dtype}")

# Heuristic: find where the 4-component axis is
if arr.ndim >= 3 and 4 in arr.shape:
    axis4 = arr.shape.index(4)
    print(f"Detected a '4' axis at dim={axis4} (0-based).")
else:
    raise ValueError("Couldn't find a dimension of size 4. Is this really quaternion data?")

# Move quat axis to last
q_raw = np.moveaxis(arr, axis4, -1).astype(float)

# Basic sanity
flat = q_raw.reshape(-1, 4)
finite_frac = np.mean(np.isfinite(flat))
print(f"Finite fraction: {finite_frac:.6f}")
print("Component stats (raw, assumed as stored order):")
for i, label in enumerate(["c0","c1","c2","c3"]):
    v = flat[:, i]
    print(f"  {label}: mean={np.mean(v):+.4f}, std={np.std(v):.4f}, min={np.min(v):+.4f}, max={np.max(v):+.4f}")

# Evaluate candidates
reports = []
for name, perm in CANDIDATES.items():
    rep = summarize_candidate(q_raw, perm, name)
    rep["score"] = score(rep)
    reports.append(rep)

reports = sorted(reports, key=lambda r: r["score"])

print("\n=== Candidate rankings (best first) ===")
for r in reports:
    print(
        f"- {r['name']:<18} perm={r['perm']}  score={r['score']:.3e}  "
        f"ortho_mean={r['ortho_err_mean']:.2e} det_mean={r['det_mean']:.6f} "
        f"frac_det~1={r['frac_det_near1']:.3f} norm_mean={r['norm_mean']:.4f}"
    )

best = reports[0]
print("\nBEST GUESS ORDERING:", best["name"], "perm=", best["perm"])

print("\nWhat this confirms:")
print("- It strongly suggests which component ordering yields valid SO(3) rotations (det≈+1, RᵀR≈I).")

print("\nWhat this does NOT confirm (needs external reference):")
print("- Active vs passive interpretation.")
print("- Crystal→sample vs sample→crystal mapping.")
print("- Which SIDE to apply crystal symmetry (right vs left).")

print("\nHow to confirm active vs passive with ONE reference:")
print("1) Pick one pixel and compute its IPF color / a rotated direction in your reference tool (MTEX/orix/Dream3D/OIM).")
print("2) Use the same pixel quaternion here and test v' = q v q^{-1} versus v' = q^{-1} v q.")
print("   The one that matches your reference tool is your convention.")