import cirq
import numpy as np
from utils import *
import sympy as sp

#I = [[1, 0], [0, -1]]
#I = np.kron(I, I)
#A = pauli_decomposition(I)
#print(A)
#A = matrix_from_pauli_string('010')
#print(A)

#rho = [[1, 0], [0, 0]]
operation_dictionary = {'00' : 0.4, '11' : 0.2, '22' : 0.2, '33' : 0.2}
rho_decomposition = operational_to_channel(rho = None, operation_dictionary = operation_dictionary, n_qubits = 2, symbolic = True)
rho, _ = symbolic_rho(2)
print(rho)
new_rho = 0
for pauli_string in rho_decomposition:
    pauli = sp.Symbol('P_'+pauli_string)
    new_rho += pauli*rho_decomposition[pauli_string]

print(new_rho)
print("here")
print(new_rho//rho)