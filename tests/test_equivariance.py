import torch
from models.reynolds_qsr import Reynolds_QSR
from models.reynolds_res_qsrnet import Reynolds_res_QSRNet

def test_equivariance_tolerance():
    args = {
        "n_feats": 8,
        "scale": 4,
        "kernel_size": 3,
        "sym_np_path": "symmetry_groups/O_group.npy",
        "sym_inv_np_path": "symmetry_groups/O_group_inv.npy",
    }
    #model = Reynolds_QSR(args).eval()

    model= Reynolds_res_QSRNet(args).eval()
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        fx = model(x)
        G = model.group_tensor.shape[0]
        errs = []
        print("G", G)
        for g in range(G):
            gmat = model.group_tensor[g]
            gx = torch.einsum("ci,bi...->bc...", gmat, x)
            f_gx = model(gx)
            g_fx = torch.einsum("ci,bi...->bc...", gmat, fx)
            errs.append((f_gx - g_fx).abs().max().item())
            print(f"Group element {g}: max error {errs[-1]}")
    assert max(errs) < 1e-5, f"Equivariance violated: max error {max(errs)}"
