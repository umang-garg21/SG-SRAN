import torch
import importlib.util
import os
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
physics = simple.FCCPhysics(device)
enc_simple = simple.FCCEncoder(physics).to(device)
enc_equi = equi.FCCEncoder(physics).to(device)
model = equi.CrystalTransformerSR(physics, scale_factor=4, depth=2).to(device)

H,W = 4,4
N = H*W
q = torch.randn(N,4,device=device)
q = q/torch.norm(q,dim=1,keepdim=True)

with torch.no_grad():
    f4_s,f6_s = enc_simple(q)
    f0,f1,f4_e,f6_e = enc_equi(q, img_shape=(H,W))

# Replace embedding with identity and proj with identity-copy (match incoming channels)
model.embedding = torch.nn.Identity()
in_dim_actual = f4_e.size(1) + f6_e.size(1)  # 22
out_dim = model.coeff_dim
new_proj = torch.nn.Linear(in_dim_actual, out_dim, bias=False).to(device)
with torch.no_grad():
    new_proj.weight.zero_()
    for i in range(min(out_dim, in_dim_actual)):
        new_proj.weight[i, i] = 1.0
model.proj = new_proj

with torch.no_grad():
    x = torch.cat([f4_e, f6_e], dim=1)
    # embedding is identity, so x_emb = x
    x_emb = x
    x_temp = x_emb.permute(0,2,3,1)
    coeffs_temp = model.proj(x_temp)
    coeffs_ch_first = coeffs_temp.permute(0,3,1,2).contiguous()
    flat_proj = coeffs_ch_first.reshape(coeffs_ch_first.size(0), coeffs_ch_first.size(1), -1)
    coeffs_flat = flat_proj.permute(0,2,1).reshape(-1, out_dim).clone()

    decoder = simple.SphericalSamplingDecoder(physics)
    q_proj = decoder(coeffs_flat[:, :f4_e.size(1)], coeffs_flat[:, f4_e.size(1):])
    q_baseline = decoder(f4_s, f6_s)
    w1,x1,y1,z1 = q_proj[:,0], q_proj[:,1], q_proj[:,2], q_proj[:,3]
    w2,x2,y2,z2 = q_baseline[:,0], q_baseline[:,1], q_baseline[:,2], q_baseline[:,3]
    w_rel = torch.clamp(torch.abs(w1*w2 + x1*x2 + y1*y2 + z1*z2), max=1.0)
    ang = 2*torch.acos(w_rel)*180/np.pi
    print('With embedding=Identity and identity-proj: miso mean=', ang.mean().item(), ' max=', ang.max().item())
