import torch
from models.reynolds_qsr import Reynolds_QSR
from models.reynolds_res_qsrnet import Reynolds_res_QSRNet
from models.reynolds_quaternion_res_srnet import Reynolds_quaternion_res_srnet
from models.reynolds_qrbsa_different_upsampler import Reynolds_QRBSA_Different_Upsampler

def test_equivariance_tolerance():
    args = {
        "n_feats": 256,
        "scale": 4,
        "kernel_size": 3,
        "n_resblocks": 0,
    }
    #model = Reynolds_QSR(args).eval()
    #model = Reynolds_res_QSRNet(args).eval()
    #model= Reynolds_quaternion_res_srnet(args).eval()
    model = Reynolds_QRBSA_Different_Upsampler(args).eval()
    # print model summary
    print(model.__dict__)
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        fx = model(x)
        G = model.group_tensor.shape[0]
        errs = []
        print(f"Testing equivariance for {G} group elements...")
        for g in range(G):
            gmat = model.group_tensor[g]
            gx = torch.einsum("ci,bi...->bc...", gmat, x)
            f_gx = model(gx)
            g_fx = torch.einsum("ci,bi...->bc...", gmat, fx)
            errs.append((f_gx - g_fx).abs().max().item())
            print(f"Group element {g}: max error {errs[-1]:.2e}")
    
    max_error = max(errs)
    print(f"\n{'='*60}")
    print(f"Maximum equivariance error: {max_error:.2e}")
    print(f"Threshold: 1e-5")
    
    if max_error < 1e-5:
        print("✅ Equivariance test PASSED!")
    else:
        print("❌ Equivariance test FAILED!")
    print(f"{'='*60}\n")
    
    assert max_error < 1e-5, f"Equivariance violated: max error {max_error}"


if __name__ == "__main__":
    test_equivariance_tolerance()
