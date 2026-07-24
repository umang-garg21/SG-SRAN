import json, numpy as np, matplotlib.pyplot as plt

npz = np.load("out/fcc_invariant_irreps.npz", allow_pickle=True)
meta = json.loads(str(npz["meta_json"]))
ls = [int(x) for x in meta["basis_ranks"].keys()]
ranks = [meta["basis_ranks"][str(l)] for l in ls]

plt.figure(); plt.bar(ls, ranks); plt.xlabel("l"); plt.ylabel("rank(U_l)"); plt.title("FCC invariant multiplicity")
plt.show()

for l in ls:
    U = npz[f"U_{l}_real"] + 1j*npz[f"U_{l}_imag"]   # shape: (2l+1, rank)
    m = np.arange(-l, l+1)
    for j in range(U.shape[1]):
        plt.figure()
        plt.stem(m, np.abs(U[:, j])**2, use_line_collection=True)
        plt.xlabel("m"); plt.ylabel(r"$|U_{lm}|^2$")
        plt.title(f"l={l}, invariant basis #{j}")
        plt.show()
