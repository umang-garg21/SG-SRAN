import torch
import numpy as np
import importlib.util
import sys
import os

# Load modules by path
root = os.path.dirname(os.path.dirname(__file__))  # ../e3nn_Reynolds
# Paths to modules
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
print('Using device:', device)

# Use a single physics instance to ensure identical seeds
physics = simple.FCCPhysics(device)

enc_simple = simple.FCCEncoder(physics)
enc_equi = equi.FCCEncoder(physics)

enc_simple.to(device)
enc_equi.to(device)

# Create random normalized quaternions
N = 1024
q = torch.randn(N, 4, device=device)
q = q / torch.norm(q, dim=1, keepdim=True)

# Run encoders
with torch.no_grad():
    f4_s, f6_s = enc_simple(q)
    # equi returns (f0,f1,f4,f6) when called with flat input
    out = enc_equi(q)
    if len(out) == 4:
        f0_e, f1_e, f4_e, f6_e = out
    else:
        # fallback: if API changed
        f4_e, f6_e = out

# Move to cpu numpy
f4_s = f4_s.cpu().numpy()
f4_e = f4_e.cpu().numpy()

f6_s = f6_s.cpu().numpy()
f6_e = f6_e.cpu().numpy()

print('\nShapes:')
print(' f4_simple', f4_s.shape)
print(' f4_equi  ', f4_e.shape)
print(' f6_simple', f6_s.shape)
print(' f6_equi  ', f6_e.shape)

# Basic stats
def stats(a, b, name):
    diff = a - b
    print(f"\n{name} diffs:\n mean_abs={np.mean(np.abs(diff)):.6e}, max_abs={np.max(np.abs(diff)):.6e}, mean={np.mean(diff):.6e}, std={np.std(diff):.6e}")

stats(f4_s, f4_e, 'F4')
stats(f6_s, f6_e, 'F6')

# Correlation matrix between components
def comp_correlation(A, B, label):
    # A: (N, d1), B: (N, d2)
    N = A.shape[0]
    Az = A - A.mean(axis=0, keepdims=True)
    Bz = B - B.mean(axis=0, keepdims=True)
    cov = (Az.T @ Bz) / (N - 1)
    stdA = A.std(axis=0, ddof=1)
    stdB = B.std(axis=0, ddof=1)
    denom = np.outer(stdA, stdB)
    corr = cov / (denom + 1e-12)
    print(f"\n{label} correlation matrix shape: {corr.shape}")
    return corr

corr_f4 = comp_correlation(f4_s, f4_e, 'F4')
corr_f6 = comp_correlation(f6_s, f6_e, 'F6')

# For each component in simple, find best matching equi component by abs correlation
def find_mapping(corr, label):
    d1, d2 = corr.shape
    best = np.argmax(np.abs(corr), axis=1)
    best_vals = corr[np.arange(d1), best]
    print(f"\n{label} mapping (simple_idx -> equi_idx) with corr values:")
    for i in range(d1):
        print(f"  {i:2d} -> {best[i]:2d}   corr={best_vals[i]:+.6f}")
    # check bijection
    unique, counts = np.unique(best, return_counts=True)
    if unique.size != d1 or np.any(counts>1):
        print('\n Warning: mapping is not a bijection (duplicates detected)')
    return best, best_vals

map_f4, map_vals_f4 = find_mapping(corr_f4, 'F4')
map_f6, map_vals_f6 = find_mapping(corr_f6, 'F6')

# Compute remapped error for f4 using mapping and sign correction
def remap_and_eval(A, B, mapping, vals, label):
    # mapping: for each i in A, mapping[i] = j index in B
    # Build remapped B' so that A ~ B'
    d = A.shape[1]
    B_remap = np.zeros_like(A)
    for i in range(d):
        j = mapping[i]
        sign = np.sign(vals[i]) if vals[i] != 0 else 1.0
        B_remap[:, i] = sign * B[:, j]
    diff = A - B_remap
    print(f"\n{label} remapped diffs after best-match+sign: mean_abs={np.mean(np.abs(diff)):.6e}, max_abs={np.max(np.abs(diff)):.6e}")
    return diff

remap_and_eval(f4_s, f4_e, map_f4, map_vals_f4, 'F4')
remap_and_eval(f6_s, f6_e, map_f6, map_vals_f6, 'F6')

print('\nDone.')
