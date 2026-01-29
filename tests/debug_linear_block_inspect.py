import importlib.util
import os
import torch
import numpy as np

root = os.path.dirname(os.path.dirname(__file__))
path_equi = os.path.join(root, 'e3nn_experimentation', '4x4_superresolution_transformers', 'equiformer_ebsdSr.py')

spec = importlib.util.spec_from_file_location('equiformer', path_equi)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
physics = mod.FCCPhysics(device)
model = mod.CrystalTransformerSR(physics, scale_factor=4, depth=1).to(device)

print('\n=== Linear block inspector ===')

def irreps_channel_ranges(irreps):
    ranges = []
    idx = 0
    for mul, ir in irreps:
        dim = mul * (2 * ir.l + 1)
        ranges.append((idx, idx + dim, mul, ir.l))
        idx += dim
    return ranges

# Inspect embedding.linear
emb = model.embedding
lin = emb.linear
print('\nEmbedding Linear:')
print(' irreps_in:', lin.irreps_in)
print(' irreps_out:', lin.irreps_out)
try:
    w = lin.weight
    b = lin.bias if hasattr(lin, 'bias') else None
    print(' weight shape:', tuple(w.shape))
except Exception as e:
    print(' Cannot read weight directly:', e)
print('\nEmbedding.linear attributes (dir):')
print([k for k in dir(lin) if not k.startswith('_')])
print('\nEmbedding.linear named_parameters:')
for n, p in lin.named_parameters():
     print(' ', n, tuple(p.shape))

print('\nEmbedding.linear internals:')
if hasattr(lin, 'weight_views'):
    print(' weight_views attribute type:', type(lin.weight_views))
    try:
        wvs = lin.weight_views()
        print(' weight_views() returned list of length', len(wvs))
        for i, v in enumerate(wvs[:5]):
            try:
                print('  view', i, 'shape', tuple(v.shape))
            except Exception:
                print('  view', i, 'repr', repr(v)[:200])
    except Exception as e:
        print(' calling weight_views() failed:', e)
if hasattr(lin, 'weight_numel'):
    try:
        print(' weight_numel:', lin.weight_numel)
    except Exception as e:
        print(' reading weight_numel failed:', e)
if hasattr(lin, 'weight_index_slices'):
    try:
        print(' weight_index_slices repr:', repr(lin.weight_index_slices)[:200])
    except Exception as e:
        print(' reading weight_index_slices failed:', e)
if hasattr(lin, 'weight_view_for_instruction'):
    print(' has weight_view_for_instruction callable')
if hasattr(lin, 'instructions'):
    try:
        instr = lin.instructions
        print(' instructions len/type:', (len(instr) if hasattr(instr,'__len__') else None, type(instr)))
    except Exception as e:
        print(' reading instructions failed:', e)

# Print first few instructions to understand mapping and try weight_view_for_instruction
try:
    for k, inst in enumerate(lin.instructions[:8]):
        print(f' instruction[{k}] repr:')
        print(inst)
        if hasattr(lin, 'weight_view_for_instruction'):
            try:
                vw = lin.weight_view_for_instruction(k)
                print('  weight_view_for_instruction shape:', getattr(vw, 'shape', None))
                arr = vw.detach().cpu().numpy()
                print('  sample weights:', arr.reshape(-1)[:8])
            except Exception as e:
                print('  weight_view_for_instruction failed:', e)
except Exception as e:
    print(' printing instructions failed:', e)

in_ranges = irreps_channel_ranges(lin.irreps_in)
out_ranges = irreps_channel_ranges(lin.irreps_out)
print('\nEmbedding in-range blocks:')
for r in in_ranges:
    print(' ', r)
print('\nEmbedding out-range blocks:')
for r in out_ranges:
    print(' ', r)

# Compute per-block norms (out_block x in_block)
# Try to find weight matrix parameter for embedding.linear
W = None
for n, p in lin.named_parameters():
    if 'weight' in n or 'weights' in n or 'tensor' in n:
        print(' Found param candidate:', n, tuple(p.shape))
        W = p.detach().cpu().numpy()
        break
if W is None:
    print(' No dense weight matrix found for embedding.linear; falling back to printing parameters above.')
else:
    if W.ndim == 2:
        for i, (oi, oj, omul, ol) in enumerate(out_ranges):
            for j, (ii, ij, imul, il) in enumerate(in_ranges):
                sub = W[oi:oj, ii:ij]
                norm = np.linalg.norm(sub)
                mean = sub.mean()
                print(f' Emb block out{i} (l={ol}) x in{j} (l={il}): shape={sub.shape} norm={norm:.6e} mean={mean:.6e}')
    else:
        print(' Found weight parameter but not 2D (shape=', W.shape, '), cannot slice into blocks.')

# Inspect proj linear
proj = model.proj
print('\n\nProj Linear:')
print(' irreps_in:', proj.irreps_in)
print(' irreps_out:', proj.irreps_out)
# Print proj instructions and views
if hasattr(proj, 'instructions'):
    try:
        for k, inst in enumerate(proj.instructions[:12]):
            print(f' proj.instruction[{k}] repr:')
            print(inst)
            if hasattr(proj, 'weight_view_for_instruction'):
                try:
                    vw = proj.weight_view_for_instruction(k)
                    print('  proj weight_view_for_instruction shape:', getattr(vw, 'shape', None))
                    arr = vw.detach().cpu().numpy()
                    print('  sample proj weights:', arr.reshape(-1)[:8])
                except Exception as e:
                    print('  proj.weight_view_for_instruction failed:', e)
    except Exception as e:
        print(' printing proj instructions failed:', e)
print(' weight shape:', tuple(proj.weight.detach().cpu().numpy().shape))

in_ranges_p = irreps_channel_ranges(proj.irreps_in)
out_ranges_p = irreps_channel_ranges(proj.irreps_out)
print('\nProj in-range blocks:')
for r in in_ranges_p:
    print(' ', r)
print('\nProj out-range blocks:')
for r in out_ranges_p:
    print(' ', r)

Wp = proj.weight.detach().cpu().numpy()
for i, (oi, oj, omul, ol) in enumerate(out_ranges_p):
    for j, (ii, ij, imul, il) in enumerate(in_ranges_p):
        sub = Wp[oi:oj, ii:ij]
        norm = np.linalg.norm(sub)
        mean = sub.mean()
        print(f' Proj block out{i} (l={ol}) x in{j} (l={il}): shape={sub.shape} norm={norm:.6e} mean={mean:.6e}')

print('\nDone.')
