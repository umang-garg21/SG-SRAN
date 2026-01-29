    
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

# Replace proj with identity-copy linear that maps first coeff_dim hidden channels to outputs
in_dim = model.hidden_irreps.dim
out_dim = model.coeff_dim
new_proj = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
# zero init
with torch.no_grad():
    new_proj.weight.zero_()
    for i in range(min(out_dim, in_dim)):
        new_proj.weight[i, i] = 1.0
model.proj = new_proj

with torch.no_grad():
    # compute embedding then project
    x = torch.cat([f4_e, f6_e], dim=1)
    x_emb = model.embedding(x, boundary_map=f0)
    x_temp = x_emb.permute(0,2,3,1)
    coeffs_temp = model.proj(x_temp)  # (B,H,W,out_dim)
    coeffs_ch_first = coeffs_temp.permute(0,3,1,2).contiguous()
    flat_proj = coeffs_ch_first.reshape(coeffs_ch_first.size(0), coeffs_ch_first.size(1), -1)
    coeffs_flat = flat_proj.permute(0,2,1).reshape(-1, out_dim).clone()

    orig_flat = torch.cat([f4_e.permute(0,2,3,1).reshape(-1, f4_e.size(1)), f6_e.permute(0,2,3,1).reshape(-1, f6_e.size(1))], dim=1)
    diff = coeffs_flat - orig_flat
    print('After replacing proj with identity-copy: mean_abs_diff=', diff.abs().mean().item(), ' max_abs=', diff.abs().max().item())

    # decode and compare
    decoder = simple.SphericalSamplingDecoder(physics)
    q_proj = decoder(coeffs_flat[:, :f4_e.size(1)], coeffs_flat[:, f4_e.size(1):])
    q_baseline = decoder(f4_s, f6_s)
    # compute misorientation
    w1,x1,y1,z1 = q_proj[:,0], q_proj[:,1], q_proj[:,2], q_proj[:,3]
    w2,x2,y2,z2 = q_baseline[:,0], q_baseline[:,1], q_baseline[:,2], q_baseline[:,3]
    w_rel = torch.clamp(torch.abs(w1*w2 + x1*x2 + y1*y2 + z1*z2), max=1.0)
    ang = 2*torch.acos(w_rel)*180/np.pi
    print(' Miso after identity-proj: mean=', ang.mean().item(), ' max=', ang.max().item())

    # Now test embedding with uniform kernel weights and no boundary_map
    K2 = model.embedding.K2
    # set uniform kernel weights
    model.embedding.kernel_weights.data = torch.ones_like(model.embedding.kernel_weights.data) / float(K2)
    # Run embedding without boundary map
    x_emb_u = model.embedding(x, boundary_map=None)
    x_temp_u = x_emb_u.permute(0,2,3,1)
    coeffs_temp_u = model.proj(x_temp_u)
    coeffs_ch_first_u = coeffs_temp_u.permute(0,3,1,2).contiguous()
    flat_proj_u = coeffs_ch_first_u.reshape(coeffs_ch_first_u.size(0), coeffs_ch_first_u.size(1), -1)
    coeffs_flat_u = flat_proj_u.permute(0,2,1).reshape(-1, out_dim).clone()
    diff_u = coeffs_flat_u - orig_flat
    print('With uniform kernel weights & no boundary_map: mean_abs_diff=', diff_u.abs().mean().item(), ' max_abs=', diff_u.abs().max().item())
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

# Replace proj with identity-copy linear that maps first coeff_dim hidden channels to outputs
in_dim = model.hidden_irreps.dim
out_dim = model.coeff_dim
new_proj = torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
# zero init
with torch.no_grad():
    new_proj.weight.zero_()
    for i in range(min(out_dim, in_dim)):
        new_proj.weight[i, i] = 1.0
model.proj = new_proj

with torch.no_grad():
    # compute embedding then project
    x = torch.cat([f4_e, f6_e], dim=1)
    x_emb = model.embedding(x, boundary_map=f0)
    x_temp = x_emb.permute(0,2,3,1)
    coeffs_temp = model.proj(x_temp)  # (B,H,W,out_dim)
    coeffs_ch_first = coeffs_temp.permute(0,3,1,2).contiguous()
    flat_proj = coeffs_ch_first.reshape(coeffs_ch_first.size(0), coeffs_ch_first.size(1), -1)
    coeffs_flat = flat_proj.permute(0,2,1).reshape(-1, out_dim).clone()

    orig_flat = torch.cat([f4_e.permute(0,2,3,1).reshape(-1, f4_e.size(1)), f6_e.permute(0,2,3,1).reshape(-1, f6_e.size(1))], dim=1)
    diff = coeffs_flat - orig_flat
    print('After replacing proj with identity-copy: mean_abs_diff=', diff.abs().mean().item(), ' max_abs=', diff.abs().max().item())

    # decode and compare
    decoder = simple.SphericalSamplingDecoder(physics)
    q_proj = decoder(coeffs_flat[:, :f4_e.size(1)], coeffs_flat[:, f4_e.size(1):])
    q_baseline = decoder(f4_s, f6_s)
    # compute misorientation
    w1,x1,y1,z1 = q_proj[:,0], q_proj[:,1], q_proj[:,2], q_proj[:,3]
    w2,x2,y2,z2 = q_baseline[:,0], q_baseline[:,1], q_baseline[:,2], q_baseline[:,3]
    w_rel = torch.clamp(torch.abs(w1*w2 + x1*x2 + y1*y2 + z1*z2), max=1.0)
    ang = 2*torch.acos(w_rel)*180/np.pi
    print(' Miso after identity-proj: mean=', ang.mean().item(), ' max=', ang.max().item())

    # Now test embedding with uniform kernel weights and no boundary_map
    K2 = model.embedding.K2
    # set uniform kernel weights
    model.embedding.kernel_weights.data = torch.ones_like(model.embedding.kernel_weights.data) / float(K2)
    # Run embedding without boundary map
    x_emb_u = model.embedding(x, boundary_map=None)
    x_temp_u = x_emb_u.permute(0,2,3,1)
    coeffs_temp_u = model.proj(x_temp_u)
    coeffs_ch_first_u = coeffs_temp_u.permute(0,3,1,2).contiguous()
    flat_proj_u = coeffs_ch_first_u.reshape(coeffs_ch_first_u.size(0), coeffs_ch_first_u.size(1), -1)
    coeffs_flat_u = flat_proj_u.permute(0,2,1).reshape(-1, out_dim).clone()
    diff_u = coeffs_flat_u - orig_flat
    print('With uniform kernel weights & no boundary_map: mean_abs_diff=', diff_u.abs().mean().item(), ' max_abs=', diff_u.abs().max().item())
