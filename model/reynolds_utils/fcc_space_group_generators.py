import numpy as np
import sympy as sp

# Quaternion Matrix Expression form
a, b, c, d = sp.symbols("a b c d")
quat_matrix_expr = sp.Matrix(
    [[a, -b, -c, -d], [b, a, -d, c], [c, d, a, -b], [d, -c, b, a]]
)

# FCC Symms Matrix
(h, i) = sp.symbols("half inv_srqt_2")
h = 1 / 2
i = 1 / np.sqrt(2)

fcc_symms = sp.Matrix(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [i, i, 0, 0],
        [i, 0, i, 0],
        [i, 0, 0, i],
        [i, -i, 0, 0],
        [i, 0, -i, 0],
        [i, 0, 0, -i],
        [0, i, i, 0],
        [0, i, 0, i],
        [0, 0, i, i],
        [0, i, -i, 0],
        [0, 0, i, -i],
        [0, i, 0, -i],
        [h, h, h, h],
        [h, -h, -h, h],
        [h, -h, h, -h],
        [h, h, -h, -h],
        [h, h, h, -h],
        [h, h, -h, h],
        [h, -h, h, h],
        [h, -h, -h, -h],
    ]
)

operator_list = []
for mat_row in np.arange(sp.shape(fcc_symms)[0]):
    current_quat = fcc_symms.row(mat_row)
    current_operator = quat_matrix_expr.subs(
        {a: current_quat[0], b: current_quat[1], c: current_quat[2], d: current_quat[3]}
    )
    operator_list.append(current_operator)

s, x, y, z = sp.symbols("s x y z")
test_quat = sp.Matrix([[s, x, y, z]])

test_list = []
for operator in operator_list:
    test_result = test_quat @ operator
    test_list.append(test_result)

fcc_symmetry_group = np.array([m.tolist() for m in operator_list], dtype=np.float64)

np.save(
    "fcc_symmetry_group.npy",
    fcc_symmetry_group,
)

fcc_symmetry_group_inv = np.array([m.inv() for m in operator_list], dtype=np.float64)
np.save("fcc_symmetry_group_inv.npy", fcc_symmetry_group_inv)

s, x, y, z = sp.symbols("s x y z")
test_quat = sp.Matrix([[s, x, y, z]])

test_list_inv = []
for i in range(len(operator_list)):
    test_result = test_quat @ operator_list[i]
    # test_result = test_result @ operator_list[i].T
    test_result = test_result @ fcc_symmetry_group_inv[i]
    test_list_inv.append(test_result)


# self.register_buffer("group_tensor_T", group_tensor.transpose(-2, -1))  # (G, N, N)
