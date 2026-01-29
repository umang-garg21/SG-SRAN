import importlib.util
import os
import torch
import numpy as np

root = os.path.dirname(os.path.dirname(__file__))
path_simple = os.path.join(root, 'e3nn_experimentation', 'simple_encoder_decoder.py')
path_equi = os.path.join(root, 'e3nn_experimentation', '4x4_superresolution_transformers', 'equiformer_ebsdSr.py')

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

simple = load_module(path_simple, 'simple_encoder_decoder')
equi = load_module(path_equi, 'equiformer_ebsdSr')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

physics = simple.FCCPhysics(device)
enc_simple = simple.FCCEncoder(physics).to(device)
enc_equi = equi.FCCEncoder(physics).to(device)
model = equi.CrystalTransformerSR(physics, scale_factor=4, depth=2).to(device)

# Small test
H, W = 4, 4
N = H * W
q = torch.randn(N, 4, device=device)
q = q / torch.norm(q, dim=1, keepdim=True)

with torch.no_grad():
    # baseline
    f4_s, f6_s = enc_simple(q)
    decoder = simple.SphericalSamplingDecoder(physics)
    q_baseline = decoder(f4_s, f6_s)

    # equi encoder outputs
    f0, f1, f4_e, f6_e = enc_equi(q, img_shape=(H, W))

    # Replace embedding.linear with a dense identity-block mapping from coeffs->hidden
    emb = model.embedding
    in_dim = emb.in_dim  # 22
    out_dim_emb = emb.out_dim  # hidden dim, e.g., 160
    new_emb_lin = torch.nn.Linear(in_dim, out_dim_emb, bias=False).to(device)
    with torch.no_grad():
        new_emb_lin.weight.zero_()
        # Map f4 (0:9) -> embedding out block [72:72+9]
        f4_len = f4_e.size(1)
        f6_len = f6_e.size(1)
        # Find l=4 and l=6 blocks in embedding output by scanning hidden_irreps
        # From model.hidden_irreps ordering we expect l=4 block starts at 72, l=6 at 108 (verified earlier)
        l4_start = None
        l6_start = None
        idx = 0
        for mul, ir in model.hidden_irreps:
            dim = mul * (2 * ir.l + 1)
            if ir.l == 4 and l4_start is None:
                l4_start = idx
            if ir.l == 6 and l6_start is None:
                l6_start = idx
            idx += dim
        if l4_start is None or l6_start is None:
            raise RuntimeError('Cannot find l4/l6 blocks in hidden_irreps')
        for i in range(f4_len):
            new_emb_lin.weight[l4_start + i, i] = 1.0
        for j in range(f6_len):
            new_emb_lin.weight[l6_start + j, f4_len + j] = 1.0
    emb.linear = new_emb_lin

    # Replace proj with a dense linear that reads back the l4/l6 slots into coeffs
    new_proj = torch.nn.Linear(out_dim_emb, f4_e.size(1) + f6_e.size(1), bias=False).to(device)
    with torch.no_grad():
        new_proj.weight.zero_()
        for i in range(f4_len):
            new_proj.weight[i, l4_start + i] = 1.0
        for j in range(f6_len):
            new_proj.weight[f4_len + j, l6_start + j] = 1.0
    model.proj = new_proj

    # Run model and collect intermediates (first intermediate is after embedding/proj low-res)
    inter = model(f0, f1, f4_e, f6_e, (H, W), return_all_intermediates=True)
    q_pred = inter[0]
    # ensure shape (N,4)
    q_pred = q_pred.reshape(-1, 4)

    # Compute misorientation between q_pred and baseline q_baseline
    w1, x1, y1, z1 = q_pred[:,0], q_pred[:,1], q_pred[:,2], q_pred[:,3]
    w2, x2, y2, z2 = q_baseline[:,0], q_baseline[:,1], q_baseline[:,2], q_baseline[:,3]
    # dot product
    dot = torch.clamp(torch.abs(w1*w2 + x1*x2 + y1*y2 + z1*z2), max=1.0)
    ang = 2*torch.acos(dot)*180/np.pi

    print('Misorientation stats after identity-embedding+proj: mean={:.6f}°, max={:.6f}°'.format(ang.mean().item(), ang.max().item()))

    # Now try with uniform kernel weights and zero boundary_map
    with torch.no_grad():
        K2 = model.embedding.K2
        model.embedding.kernel_weights.data = torch.ones_like(model.embedding.kernel_weights.data) / float(K2)
        f0_zero = torch.zeros_like(f0)
        inter2 = model(f0_zero, f1, f4_e, f6_e, (H, W), return_all_intermediates=True)
        q_pred2 = inter2[0].reshape(-1, 4)
        w1b, x1b, y1b, z1b = q_pred2[:,0], q_pred2[:,1], q_pred2[:,2], q_pred2[:,3]
        dot2 = torch.clamp(torch.abs(w1b*w2 + x1b*x2 + y1b*y2 + z1b*z2), max=1.0)
        ang2 = 2*torch.acos(dot2)*180/np.pi
        print('With uniform kernel & zero boundary_map: mean={:.6f}°, max={:.6f}°'.format(ang2.mean().item(), ang2.max().item()))

print('Done.')
