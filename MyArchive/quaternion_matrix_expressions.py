import numpy as np
import sympy as sp
from itertools import permutations, product

import quaternion_matrix_expression_farebrother

# There are a total 48 total quaternion matrix expression forms
# The purpose of this script is to assess their uniqueness and determine how many duplicate quat forms exist
# We will also test the impact of FCC symmetry on them as well

# Quaternion Matrix Expression form

# Symbols for prefixes and quaternion components
a, b, c, d, s = sp.symbols("a b c d s")
i = sp.symbols("i", commutative=False)
j = sp.symbols("j", commutative=False)
k = sp.symbols("k", commutative=False)

# Build Raw Quat Operation Matrices
# Total number of Operation Matrices = 48
# 24 from number of unique 4 term permutations
# Each permutation has 2 forms because q @ q_conj != q_cong @ q
# Note: these 2 forms are skew symmetric
component_list = [i, j, k]
conj_list = [-i, -j, -k]
sign_list = [1, -1]
sign_combos = product(sign_list, repeat=3)
sign_combos = list(sign_combos)


q_combos = permutations(component_list, 3)
q_list = list(q_combos)


q_list_full = []
q_conj_list_full = []
for sign in sign_combos:
    for q in q_list:
        q_list_full.append([s, sign[0] * q[0], sign[1] * q[1], sign[2] * q[2]])
        q_conj_list_full.append([s, sign[0] * -q[0], sign[1] * -q[1], sign[2] * -q[2]])

# Build raw symbolic list
matrix_list = []

for q_ind, q in enumerate(q_list_full):
    q_conj = q_conj_list_full[q_ind]
    q1 = sp.Matrix([list(q)])
    q1_conj = sp.Matrix(list(q_conj))
    mat1 = q1_conj @ q1  # @ q1_conj
    matrix_list.append(mat1)

# Simplify Operation Matrices with i,j,k equivalencies
simplified_list = []

for current_mat in matrix_list:
    # Specific instance of substitution: we only want instances of scalar*scalar to be substituted for a
    # Applying s**2 here allows for that
    current_mat = current_mat.subs(i**2, -(s**2))
    current_mat = current_mat.subs(j**2, -(s**2))
    current_mat = current_mat.subs(k**2, -(s**2))
    current_mat = current_mat.subs(i * j, k)
    current_mat = current_mat.subs(j * i, -k)
    current_mat = current_mat.subs(i * k, -j)
    current_mat = current_mat.subs(k * i, j)
    current_mat = current_mat.subs(j * k, i)
    current_mat = current_mat.subs(k * j, -i)
    simplified_list.append(current_mat)

# Perform Piecewise Substitutions of a, b, c, and d
matrix_quat_list = []

for current_mat in simplified_list:
    # Perform a subsitution
    current_mat = current_mat.subs(s**2, a)
    # Remove all non s**2 s instances
    current_mat = current_mat.subs(s, 1)
    current_mat = current_mat.subs(i, b)
    current_mat = current_mat.subs(j, c)
    current_mat = current_mat.subs(k, d)
    matrix_quat_list.append(current_mat)

# Handedness calculation
quat_handedness = []
for current_quat in q_list_full:
    current_val = current_quat[0] * current_quat[1] * current_quat[2] * current_quat[3]
    current_val = current_val.subs(i * j, k)
    current_val = current_val.subs(j * i, -k)
    current_val = current_val.subs(i * k, -j)
    current_val = current_val.subs(k * i, j)
    current_val = current_val.subs(j * k, i)
    current_val = current_val.subs(k * j, -i)
    current_val = current_val.subs(s, 1)
    current_val = current_val.subs(i**2, -1)
    current_val = current_val.subs(j**2, -1)
    current_val = current_val.subs(k**2, -1)
    quat_handedness.append(current_val)
print(f"handedness = {quat_handedness}")

# determinant_list = []
# current_matrix_form = simplified_list[0]
# current_determinant = sp.det(current_matrix_form)
# current_determinant = current_determinant.subs(s, 1)
# current_val= current_determinant
# current_val = current_val.subs(i * j, k)
# current_val = current_val.subs(j * i, -k)
# current_val = current_val.subs(i * k, -j)
# current_val = current_val.subs(k * i, j)
# current_val = current_val.subs(j * k, i)
# current_val = current_val.subs(k * j, -i)
# current_val = current_val.subs(s, 1)
# current_val = current_val.subs(i ** 2, -1)
# current_val = current_val.subs(j ** 2, -1)
# current_val = current_val.subs(k ** 2, -1)
# determinant_list.append(current_val)

# With all 48 matrix quat forms now defined, we will go through the FCC generation process with each of them
# Once generated, we will compare them for uniqueness in matrix form as well as associated quaternion form

# Define FCC Symms Matrix
(h, i) = sp.symbols("h i")
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
        # Negatives
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [-i, -i, 0, 0],
        [-i, 0, -i, 0],
        [-i, 0, 0, -i],
        [-i, i, 0, 0],
        [-i, 0, i, 0],
        [-i, 0, 0, i],
        [0, -i, -i, 0],
        [0, -i, 0, -i],
        [0, 0, -i, -i],
        [0, -i, i, 0],
        [0, 0, -i, i],
        [0, -i, 0, i],
        [-h, -h, -h, -h],
        [-h, h, h, -h],
        [-h, h, -h, h],
        [-h, -h, h, h],
        [-h, -h, -h, h],
        [-h, -h, h, -h],
        [-h, h, -h, -h],
        [-h, h, h, h],
    ]
)
# triclinic_syms = sp.Matrix([[1, 0, 0, 0], [-1, 0, 0, 0]])
# fcc_symms = triclinic_syms

# Operators will be generated here and stored as a Python list of SymPy Matrix elements
# SymPy subs() along with numpy numpy.array().astype() can be used to convert these into numpy arrays
symbolic_form_list = []

for quat_matrix_form in matrix_quat_list:
    operator_list = []
    for mat_row in np.arange(sp.shape(fcc_symms)[0]):
        current_quat = fcc_symms.row(mat_row)
        current_operator = quat_matrix_form.subs(
            {
                a: current_quat[0],
                b: current_quat[1],
                c: current_quat[2],
                d: current_quat[3],
            }
        )
        operator_list.append(current_operator)
    symbolic_form_list.append(operator_list)

# Test to Verify Equivalence of Matrix forms
# Comparison as hashable sets appears to have debug issues, so we will do the comparison as brute force
# despite it being slow and suboptimal

comparison_list = []
comparison_target = symbolic_form_list[0]

for comparator in symbolic_form_list:
    true_count = 0
    for current_comparison in comparator:
        if current_comparison in comparison_target:
            true_count += 1

    comparison_list.append(true_count)

print(f"Comparison counts = {comparison_list}")


# Comparison of vector forms after group operations
# First we apply operations to an arbitrary quaternion s
w, x, y, z = sp.symbols("w x y z")
test_quat = sp.Matrix([[w, x, y, z]])

quat_list = []

for symbolic_form in symbolic_form_list:
    applied_list = []
    for current_operator in symbolic_form:
        # Apply operation
        current_quat = test_quat @ current_operator
        applied_list.append(current_quat)
    quat_list.append(applied_list)


quat_comparison_target = quat_list[0]
quat_comparison_list = []

for comparator in quat_list:
    true_count = 0
    for current_comparison in comparator:
        if current_comparison in quat_comparison_target:
            true_count += 1

    quat_comparison_list.append(true_count)
    # Do something with the matching operator

    # test = []
    # for q in quat_list[1]:
    #     if q in quat_list[0]:
    #         test.append(q)

print(f"Quat comparison counts = {quat_comparison_list}")

# Farebrother Comparisons
[type1, type2] = quaternion_matrix_expression_farebrother.build_farebrother_forms()
fare_full = type1 + type2
# Operators will be generated here and stored as a Python list of SymPy Matrix elements
# SymPy subs() along with numpy numpy.array().astype() can be used to convert these into numpy arrays
fare_matrix_quat_list = []
for fare_form in fare_full:
    current_matrix_quat = (
        a * fare_form[0] + b * fare_form[1] + c * fare_form[2] + d * fare_form[3]
    )
    fare_matrix_quat_list.append(current_matrix_quat)

fare_symb_form_list = []

for quat_matrix_form in fare_matrix_quat_list:
    operator_list = []
    for mat_row in np.arange(sp.shape(fcc_symms)[0]):
        current_quat = fcc_symms.row(mat_row)
        current_operator = quat_matrix_form.subs(
            {
                a: current_quat[0],
                b: current_quat[1],
                c: current_quat[2],
                d: current_quat[3],
            }
        )
        operator_list.append(current_operator)
    fare_symb_form_list.append(operator_list)

# Test to Verify Equivalence of Matrix forms
# Comparison as hashable sets appears to have debug issues, so we will do the comparison as brute force
# despite it being slow and suboptimal

comparison_list = []
comparison_target = symbolic_form_list[0]
# comparison_target = fare_symb_form_list[24]

for comparator in fare_symb_form_list:
    true_count = 0
    for current_comparison in comparator:
        if current_comparison in comparison_target:
            true_count += 1

    comparison_list.append(true_count)

print(f"Fare Comparison counts = {comparison_list}")


# Comparison of vector forms after group operations
# First we apply operations to an arbitrary quaternion s
w, x, y, z = sp.symbols("w x y z")
test_quat = sp.Matrix([[w, x, y, z]])

fare_quat_list = []

for symbolic_form in fare_symb_form_list:
    applied_list = []
    for current_operator in symbolic_form:
        # Apply operation
        current_quat = test_quat @ current_operator
        applied_list.append(current_quat)
    fare_quat_list.append(applied_list)


quat_comparison_target = quat_list[0]
# quat_comparison_target = fare_quat_list[24]
quat_comparison_list = []

for comparator in fare_quat_list:
    true_count = 0
    for current_comparison in comparator:
        if current_comparison in quat_comparison_target:
            true_count += 1

    quat_comparison_list.append(true_count)
    # Do something with the matching operator

    # test = []
    # for q in quat_list[1]:
    #     if q in quat_list[0]:
    #         test.append(q)

print(f"Fare Quat comparison counts = {quat_comparison_list}")
