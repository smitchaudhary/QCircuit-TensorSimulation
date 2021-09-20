#import jet
import numpy as np
import cirq

def generate_random_qnn(qubits, angles, paulis, depth, initial_state = "0", noisy = False, noise = None):
    circuit = cirq.Circuit()
    if initial_state == "+":
        for qubit in qubits:
            circuit += cirq.ry(np.pi/4)(qubit)
    elif initial_state == "i":
        for qubit in qubits:
            circuit += cirq.rx(-np.pi/4)(qubit)

    if noisy:
        assert noise != None, "Need to provide noise model for noisy circuit."

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

class Correlated_Dephasing(cirq.TwoQubitGate):
    def __init__(self, p: float) -> None:
        self._p = p

    def _num_qubits_(self):
        return 2

    def _mixture_(self):
        ps = [self._p, self._p, self._p, 1.0 - self._p]
        ops = [cirq.unitary(cirq.X)*cirq.unitary(cirq.X), cirq.unitary(cirq.Y)*cirq.unitary(cirq.Y), cirq.unitary(cirq.Z)*cirq.unitary(cirq.Z), cirq.unitary(cirq.I)*cirq.unitary(cirq.I)]
        return tuple(zip(ps, ops))

    def _has_mixture_(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args='cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return cirq.CircuitDiagramInfo(wire_symbols=(f"DP({self._p})", f"DP({self._p})"), exponent=1.0, connected=True)

class MyGateDepolarizingNoiseModel(cirq.NoiseModel):
    def noisy_operation(self, op):
        if isinstance(op.gate, MyGate):
            return [op, cirq.depolarize(p).on(op.qubits[0])]
        return op
