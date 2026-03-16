# run from repo root: python show_irreps_random_quat.py
import importlib.util
from pathlib import Path
import torch

# Load local_iso_embedding directly (avoids models/__init__.py side effects)
p = Path("models/local_iso_embedding.py").resolve()
spec = importlib.util.spec_from_file_location("local_iso_embedding", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

torch.manual_seed(0)

# Random quaternion (passive, wxyz) and normalize
q = torch.randn(1, 4, dtype=torch.float32)
q = q / q.norm(dim=-1, keepdim=True)

fcc = mod.build_local_iso_fcc_embedding(dtype=torch.float32).eval()
hcp = mod.build_local_iso_hcp_embedding(dtype=torch.float32, d6_convention="z_axis").eval()

# Passive input path
fcc_feat = fcc.forward_irreps_passive(q)[0]
hcp_feat = hcp.forward_irreps_passive(q)[0]

def print_irreps(name, emb, feat):
    print(f"\n{name} irreps_out: {emb.irreps_out} (dim={emb.irreps_out.dim})")
    for (mul, ir), sl in zip(emb.irreps_out, emb.irreps_out.slices()):
        vals = ", ".join(f"{float(v):+.6f}" for v in feat[sl])
        print(f"  {mul}x{ir}: [{vals}]")

print("random quat [w x y z]:", [float(x) for x in q[0]])
print_irreps("FCC", fcc, fcc_feat)
print_irreps("HCP", hcp, hcp_feat)
