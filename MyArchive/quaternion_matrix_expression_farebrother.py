import numpy as np
import sympy as sp
from itertools import permutations, product

# There are a total 48 total quaternion matrix expression forms
# The purpose of this script is to assess their uniqueness and determine how many duplicate quat forms exist
# We will also test the impact of FCC symmetry on them as well

# Quaternion Matrix Expression form

def kron_prod_2x2(mat1, mat2):
    kron_prod = np.block([[mat1[0,0]*mat2, mat1[0,1]*mat2], [mat1[1,0]*mat2, mat1[1,1]*mat2]])
    return sp.Matrix(kron_prod)

def build_farebrother_forms():
    # Symbols for prefixes and quaternion components
    a, b, c, d, s = sp.symbols("a b c d s")
    i = sp.symbols("i", commutative=False)
    j = sp.symbols("j", commutative=False)
    k = sp.symbols("k", commutative=False)

    B = sp.Matrix([[0, 1], [-1, 0]])
    C = sp.Matrix([[0, 1], [1, 0]])
    D = sp.Matrix([[1, 0], [0, -1]])
    E = sp.Matrix([[1, 0], [0, 1]])

    H1 = kron_prod_2x2(B,D)
    J1 = kron_prod_2x2(E,B)
    K1 = kron_prod_2x2(B,C)

    H2 = kron_prod_2x2(D,B)
    J2 = kron_prod_2x2(B,E)
    K2 = kron_prod_2x2(C,B)

    component_list_1 = [H1, J1, K1]
    type1_combos = list(permutations(component_list_1, 2))

    component_list_2 = [H2, J2, K2]
    type2_combos = list(permutations(component_list_2, 2))

    sign_list = [1, -1]
    sign_combos = list(product(sign_list, repeat=2))

    type1_list = []
    type2_list = []
    for sign_combo in sign_combos:
        for type_index, type1_combo in enumerate(type1_combos):
            type1_combo = list(type1_combo)
            type2_combo = list(type2_combos[type_index])
            current_type1 = [sp.Matrix(np.eye(4, dtype=int)), sign_combo[0]*type1_combo[0], sign_combo[1]*type1_combo[1]]
            current_type2 = [sp.Matrix(np.eye(4, dtype=int)), sign_combo[0]*type2_combo[0], sign_combo[1]*type2_combo[1]]
            type1_dependent = current_type1[1] @ current_type1[2]
            type2_dependent = current_type2[1] @ current_type2[2]
            current_type1.append(type1_dependent)
            current_type2.append(type2_dependent)
            type1_list.append(current_type1)
            type2_list.append(current_type2)

    return [type1_list, type2_list]

if __name__ == "__main__":
    [t1, t2] = build_farebrother_forms()



