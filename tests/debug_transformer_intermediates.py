import torch
import numpy as np
import importlib.util
import os

root = os.path.dirname(os.path.dirname(__file__))
path_simple = os.path.join(root, 'e3nn_experimentation', 'simple_encoder_decoder.py')
path_equi = os.path.join(root, 'e3nn_experimentation', '4x4_superresolution_transformers', 'equiformer_ebsdSr.py')

# helper loader
import importlib.util

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

simple = load_module(path_simple, 'simple_encoder_decoder')
equi = load_module(path_equi, 'equiformer_ebsdSr')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

physics = simple.FCCPhysics(device)
enc_simple = simple.FCCEncoder(physics).to(device)
enc_equi = equi.FCCEncoder(physics).to(device)

# small image
H, W = 4, 4
N = H * W
q = torch.randn(N, 4, device=device)
q = q / torch.norm(q, dim=1, keepdim=True)

# simple baseline decoded quaternions
with torch.no_grad():
    f4_s, f6_s = enc_simple(q)
    decoder_simple = simple.SphericalSamplingDecoder(physics)
    q_baseline = decoder_simple(f4_s, f6_s)  # (N,4)

# equiformer path
with torch.no_grad():
    f0, f1, f4_e, f6_e = enc_equi(q, img_shape=(H, W))
    model = equi.CrystalTransformerSR(physics, scale_factor=4, depth=2).to(device)
    # Run with return_all_intermediates to capture decoder outputs at stages
    inter = model(f0, f1, f4_e, f6_e, (H, W), return_all_intermediates=True)

print('\nCollected', len(inter), 'intermediate quaternion outputs from model')

# QUICK CHECK: compute projection immediately after embedding (same as intermediate 0 source)
with torch.no_grad():
    # Prepare inputs as the model does
    f4_img = f4_e
    f6_img = f6_e
    f0_img = f0
    x = torch.cat([f4_img, f6_img], dim=1)
    x_emb = model.embedding(x, boundary_map=f0_img)
    # permute and project
    x_temp = x_emb.permute(0, 2, 3, 1)
    coeffs_temp = model.proj(x_temp)  # (B, H, W, coeff_dim)
    coeffs_ch_first = coeffs_temp.permute(0, 3, 1, 2).contiguous()
    # flatten per-pixel vector
    flat = coeffs_ch_first.reshape(coeffs_ch_first.size(0), coeffs_ch_first.size(1), -1)
    coeffs_flat = flat.permute(0, 2, 1).reshape(-1, model.coeff_dim).clone()

    orig_flat = torch.cat([f4_e.permute(0,2,3,1).reshape(-1, f4_e.size(1)), f6_e.permute(0,2,3,1).reshape(-1, f6_e.size(1))], dim=1)
    diff = coeffs_flat - orig_flat
    print('\nProjection after embedding vs original coefficients:')
    print('  mean_abs_diff=', diff.abs().mean().item())
    print('  max_abs_diff =', diff.abs().max().item())
    # print a small sample
    print('\n Sample original vs projected (first 5 pixels):')
    for i in range(min(5, orig_flat.shape[0])):
        o = orig_flat[i].cpu().numpy()
        p = coeffs_flat[i].cpu().numpy()
        print(f'  pix {i}: orig[:6]={o[:6]}, proj[:6]={p[:6]}, ||diff||={np.linalg.norm(o-p):.6e}')

print('\nProjector & mask diagnostics:')
print(' model.proj irreps_in:', model.proj.irreps_in)
print(' model.proj irreps_out:', model.proj.irreps_out)
try:
    w = next(model.proj.parameters())
    print(' proj weight shape:', tuple(w.shape))
except StopIteration:
    print(' proj has no parameters?')
print(' mask_all shape:', model.mask_all.shape, 'mask_all sum:', model.mask_all.sum().item())

# misorientation helper
def misorientation_deg(q1, q2):
    # q1, q2: (N,4) torch
    # compute relative quaternion q_rel = q1 * conj(q2)
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    # conjugate q2
    w2c, x2c, y2c, z2c = w2, -x2, -y2, -z2
    # multiply q1 * q2c
    wr = w1*w2c - x1*x2c - y1*y2c - z1*z2c
    # clamp abs(wr) into [0,1]
    wr = torch.clamp(torch.abs(wr), max=1.0)
    angles = 2 * torch.acos(wr) * 180.0 / np.pi
    return angles

# compare each intermediate to baseline
for idx, q_int in enumerate(inter):
    # q_int is (N,4)
    if isinstance(q_int, list) or isinstance(q_int, tuple):
        q_int = q_int[0]
    q_int = q_int.reshape(-1,4).to(device)
    # If intermediate is higher-res, downsample to LR grid before comparing
    n_int = q_int.shape[0]
    if n_int == q_baseline.shape[0]:
        q_cmp = q_int
    else:
        # estimate integer scale factor
        scale = int(round((n_int / q_baseline.shape[0]) ** 0.5))
        if scale <= 1:
            q_cmp = q_int[:q_baseline.shape[0]]
        else:
            H_hr = H * scale
            W_hr = W * scale
            try:
                q_hr = q_int.reshape(H_hr, W_hr, 4)
                # pick center pixel within each block
                offset = scale // 2
                hs = [offset + i * scale for i in range(H)]
                ws = [offset + j * scale for j in range(W)]
                sampled = []
                for hi in hs:
                    for wj in ws:
                        sampled.append(q_hr[hi, wj])
                q_cmp = torch.stack(sampled, dim=0)
            except Exception:
                # fallback: take first N entries
                q_cmp = q_int[:q_baseline.shape[0]]

    ang = misorientation_deg(q_cmp, q_baseline)
    print(f"Intermediate {idx}: mean={ang.mean().item():.4f}°, max={ang.max().item():.4f}°")

print('\nDone.')
