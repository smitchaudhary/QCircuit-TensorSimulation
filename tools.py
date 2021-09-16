import jet
import numpy as np
import cirq

def fid_with_max_mixed(rho):
    w, v = np.linalg.eig(rho)
    w = np.sqrt(w)
    d = len(w)
    #print(d)
    F = np.sum(w)/d
    return np.real(F)

def make_density_matrix(state):
    bra = np.matrix(state)
    ket = bra.getH()
    rho = np.matmul(ket, bra)
    #print(rho)

    return rho

def generate_random_qnn(qubits, angles, paulis, depth):
    circuit = cirq.Circuit()
    for qubit in qubits:
        circuit += cirq.ry(np.pi/4)(qubit)

    for d in range(depth):
        for i, qubit in enumerate(qubits):
            random_n = np.random.random()
            random_rot = 2*np.pi*np.random.random()

            if paulis[d, i] == 0:
                circuit += cirq.rz(angles[d, i])(qubit)
            elif paulis[d, i] == 1:
                circuit += cirq.ry(angles[d, i])(qubit)
            else:
                circuit += cirq.rx(angles[d, i])(qubit)

        for src, dest in zip(qubits, qubits[1:]):
            circuit += cirq.CZ(src, dest)

    return circuit

def I_d(n_qubits):
    return np.eye(2**n_qubits)/(2**n_qubits)
