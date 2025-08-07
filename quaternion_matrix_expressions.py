import numpy as np
import sympy as sp
from itertools import permutations


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
component_list = [s, i, j, k]
conj_list = [s, -i, -j, -k]
q_combos = permutations(component_list, 4)
q_list = list(q_combos)

q_conj_combos = permutations(conj_list, 4)
q_conj_list = list(q_conj_combos)

# Build raw symbolic list
matrix_list = []

for q_ind, q in enumerate(q_list):
    q_conj = q_conj_list[q_ind]
    q1 = sp.Matrix(list(q))
    q1_conj = sp.Matrix([list(q_conj)])
    mat1 = q1 @ q1_conj
    matrix_list.append(mat1)
    q2 = sp.Matrix(list(q_conj))
    q2_conj = sp.Matrix([list(q)])
    mat2 = q2 @ q2_conj
    matrix_list.append(mat2)

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
triclinic_syms = sp.Matrix([[1, 0, 0, 0], [-1, 0, 0, 0]])
fcc_symms = triclinic_syms

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
# Since we know these operators are unique but not ordered, we will verify with element-wise comparison
# This process is a bit awful as sympy mutable dense matrices, so we will convert to strings to have something hashable
string_form_list = []

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
        current_operator = str(current_operator)
        operator_list.append(current_operator)
    string_form_list.append(operator_list)

# As strings, we can compare using set() without an iterable
comparison_list = []
comparison_target = set(string_form_list[0])
for comparator in string_form_list:
    current_comparison = set(comparator)
    check_val = comparison_target == current_comparison
    comparison_list.append(check_val)
print(f"Matrix forms equal: {all(comparison_list)}")

# Comparison of vector forms after group operations
# We will do a similar operation to above but with the added vector multiplication and simplification
# We will also convert all group operators to numeric form first to prevent any symbol swap issues in string comparison
# For readability, we will evaluate sqrt(2)/2 as 0.707
numeric_form_list = []

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
        current_operator = current_operator.subs({h: 0.5, i: 0.707})
        current_operator = np.array(current_operator)
        operator_list.append(current_operator)
    numeric_form_list.append(operator_list)

# With numeric form, we apply operations to an arbitrary quaternion and convert the results to strings
w, x, y, z = sp.symbols("w x y z")
test_quat = sp.Matrix([[w, x, y, z]])

quat_list = []

for numeric_form in numeric_form_list:
    applied_list = []
    for current_operator in numeric_form:
        # Apply operation
        current_quat = test_quat @ current_operator
        # Convert to String
        current_quat = str(current_quat)
        applied_list.append(current_quat)
    quat_list.append(applied_list)

# As strings, we can (again) compare using set() without an iterable
quat_comparison_list = []
comparison_target = set(quat_list[0])
for comparator in quat_list:
    current_comparison = set(comparator)
    check_val = comparison_target == current_comparison
    quat_comparison_list.append(check_val)
print(f"Quaternion forms equal: {all(quat_comparison_list)}")

# It seems they are not equal... we can also check this manually
print(quat_list[0])
print(quat_list[1])
