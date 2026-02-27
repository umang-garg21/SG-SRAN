import sys
import os
import torch
import numpy as np

# ensure project root is on sys.path so `import utils...` works when running scripts
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# define quat_mul

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    wa, xa, ya, za = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    wb, xb, yb, zb = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        wa*wb - xa*xb - ya*yb - za*zb,
        wa*xb + xa*wb + ya*zb - za*yb,
        wa*yb - xa*zb + ya*wb + za*xb,
        wa*zb + xa*yb - ya*xb + za*wb,
    ], dim=-1)

# single reduction

def reduce_to_fz_oh(q_1x4: torch.Tensor, sym_inv: torch.Tensor) -> torch.Tensor:
    G = sym_inv.shape[0]
    cands = quat_mul(sym_inv, q_1x4.expand(G, -1))
    cands = torch.where(cands[:, 0:1] < 0, -cands, cands)
    return cands[cands[:, 0].argmax()]

# original many (potentially buggy) from the notebook (first copy)
def reduce_to_fz_oh_many_orig(q_Nx4: torch.Tensor, sym_inv_Gx4: torch.Tensor) -> torch.Tensor:
    if q_Nx4.dim() == 1:
        q_Nx4 = q_Nx4.unsqueeze(0)
    device = q_Nx4.device
    dtype = q_Nx4.dtype
    sym = sym_inv_Gx4.to(device=device, dtype=dtype)
    s = sym[:, None, :]
    q = q_Nx4[None, :, :]
    wa, xa, ya, za = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    wb, xb, yb, zb = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    c0 = wa*wb - xa*xb - ya*yb - za*zb
    c1 = wa*xb + xa*wb + ya*zb - za*yb
    c2 = wa*yb - xa*zb + ya*wb + za*xb
    c3 = wa*zb + xa*yb - ya*xb + za*wb
    cands = torch.stack([c0, c1, c2, c3], dim=-1)
    cands = torch.where(cands[..., 0:1] < 0, -cands, cands)
    ws = cands[..., 0]
    best_idx = ws.argmax(dim=0)
    ar = torch.arange(q_Nx4.shape[0], device=device)
    best = cands[best_idx, ar, :]
    return best

# corrected many

def reduce_to_fz_oh_many_fixed(q_Nx4: torch.Tensor, sym_inv_Gx4: torch.Tensor) -> torch.Tensor:
    if q_Nx4.dim() == 1:
        q_Nx4 = q_Nx4.unsqueeze(0)
    device = q_Nx4.device
    dtype = q_Nx4.dtype
    sym = sym_inv_Gx4.to(device=device, dtype=dtype)
    G = sym.shape[0]
    N = q_Nx4.shape[0]
    s = sym[:, None, :].expand(G, N, 4)
    q = q_Nx4[None, :, :].expand(G, N, 4)
    cands = quat_mul(s, q)
    cands = torch.where(cands[..., 0:1] < 0, -cands, cands)
    ws = cands[..., 0]
    best_idx = ws.argmax(dim=0)
    ar = torch.arange(N, device=device)
    best = cands[best_idx, ar, :]
    return best


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], axis=-1)



if __name__ == "__main__":


    from orix.quaternion import symmetry as SYM

    # Interactive prompts with sensible defaults
    npy_path = "/data/warren/materials/EBSD/IN718_2D_SR_x4/Test/Original_Data/Open_718_Test_hr_x_block_0.npy"
    default_out = "debug_fz"
    default_sym = "Oh"
    default_max_samples = None


    try:
        print("Loading:", npy_path)
        q_xyzw = np.load(npy_path)
        H, W, C = q_xyzw.shape
        assert C == 4, "Expected (H,W,4) XYZW input"
        q_wxyz = xyzw_to_wxyz(q_xyzw).reshape(-1,4)
        q_t = torch.from_numpy(q_wxyz.astype(np.float32))
    except Exception:
        # fallback: generate random unit quaternions in w,x,y,z ordering
        print("Could not load .npy file, falling back to random quaternions for test")
        N = 256
        q_t = torch.randn(N, 4)
        q_t = q_t / q_t.norm(dim=1, keepdim=True)

    t1 = torch.tensor(q_wxyz, dtype=torch.float32).unsqueeze(0)

    # ---------------------------------------------`------------------------------
    #  --- Oh symmetry operators (48 ops, Laue group) in [w,x,y,z] ---
    # ---------------------------------------------------------------------------
    oh_ops_np = np.asarray(SYM.Oh.data, dtype=np.float32)     # (48, 4)
    oh_ops     = torch.from_numpy(oh_ops_np)                   # (48, 4)
    # Unit-quaternion inverse = conjugate: negate xyz
    oh_ops_inv = oh_ops.clone()
    oh_ops_inv[:, 1:] *= -1.0                                  # (48, 4)
    oh_ops_inv = oh_ops_inv[:24]


    mismatch_count = 0
    total = min(100, q_t.shape[0])
    for i in range(total):
        q = q_t[i].unsqueeze(0)
        a = reduce_to_fz_oh(q, oh_ops_inv)
        b = reduce_to_fz_oh_many_orig(q, oh_ops_inv)
        c = reduce_to_fz_oh_many_fixed(q, oh_ops_inv)
        if not torch.allclose(a, b, atol=1e-6):
            mismatch_count += 1
            print(f"idx {i}: single != many_orig: single={a.numpy()} many_orig={b.numpy()}")
        if not torch.allclose(a, c, atol=1e-6):
            print(f"idx {i}: single != many_fixed: single={a.numpy()} many_fixed={c.numpy()}")

    print('mismatch_count (orig):', mismatch_count)
    print('done')
