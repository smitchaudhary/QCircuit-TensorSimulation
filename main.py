import jet
import cirq
import numpy as np
from tools import *

fids = []
n_qubits_array = [2, 3, 4, 5, 6]
for n_qubits in n_qubits_array:
    layers_array = [10*n_qubits]
    n_circuits_array = [i*n_qubits*n_qubits for i in [5, 10, 25, 50, 100]]#[n_qubits*i for i in [1, 10, 50, 100, 200, 500]]

    for layers in layers_array:
        fid = []
        for n_circuits in n_circuits_array:
            angles = np.random.random((n_circuits, layers, n_qubits))
            paulis = np.random.randint(0,3,(n_circuits, layers, n_qubits))

            final_rho = 0
            for i in range(n_circuits):
                rho = cirq.final_density_matrix(generate_random_qnn(cirq.GridQubit.rect(1, n_qubits), angles[i], paulis[i], layers, initial_state = "+"))
                #print(np.trace(rho))
                final_rho += rho

            final_rho = final_rho/n_circuits

            fidelity = cirq.qis.fidelity(I_d(n_qubits), final_rho, qid_shape = 2**n_qubits)
            fid.append(fidelity)
        fids.append(fid)
        print(f'For {layers} layers, fidelity is {fid} for {n_circuits_array} circuits')
    print("-----------------------------")
