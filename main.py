import jet
import cirq
import numpy as np
from tools import *

n_qubits = 4
layers = 10*n_qubits
n_circuits_array = [n_qubits*i for i in [10, 50, 100, 150, 200, 500]]

fid = []
for n_circuits in n_circuits_array:
    angles = np.random.random((n_circuits, layers, n_qubits))
    paulis = np.random.randint(0,3,(n_circuits, layers, n_qubits))

    final_rho = 0
    for i in range(n_circuits):
        rho = cirq.final_density_matrix(generate_random_qnn(cirq.GridQubit.rect(1, n_qubits), angles[i], paulis[i], layers))
        #print(np.trace(rho))
        final_rho += rho

    final_rho = final_rho/n_circuits

    fidelity = cirq.qis.fidelity(I_d(n_qubits), final_rho, qid_shape = 2**n_qubits)
    fid.append(fidelity)

print(fid)
