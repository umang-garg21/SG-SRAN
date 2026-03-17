import numpy as np
from orix.quaternion import symmetry


def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


ops = symmetry.O.data
n_ops = len(ops)

self_inverse = []

for k in range(n_ops-1):
    s = ops[k]
    s_inv = quat_conj(s)

    if np.allclose(s, s_inv):
        self_inverse.append(k)

print("Self-inverse operators (s = s^-1):")
print(self_inverse)
print(f"Total: {len(self_inverse)} out of {n_ops}")