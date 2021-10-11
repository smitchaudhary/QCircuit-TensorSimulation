#import jet
import numpy as np
import cirq
import matplotlib.pyplot as plt

def generate_random_qnn(qubits, angles, paulis, depth, initial_state = "0", noisy = False, noise_model = None):
    '''
    Generates random Circuits with the following structure:
        Adds single qubit rotations by a random angle about a random axis.
        Followed by CZ gates on each neighbouring pair.
        This is repeated depth times.
    
    Parameters:
        qubits : list
            List of qubit objects
        angles : np.ndarray
            List of angles of rotation for the circuit.
        paulis : np.ndarray
            List of what Pauli rotations to apply in the circuit.
        depth : int
            The depth of the circuit.
        initial_state : str
            What is the initial state of the qubits. "0" is default. Starts with ground state.
            "+" starts with equal superpositionof all staets.
            "i" starts with all qubits individually being in +i state.
            "random" starts the circuit in some random state.
        noisy : bool
            Defaults to False. Gives whether the circuit is noisy or not.
        noise_model : cirq Noise Model
            Defaults to None. Noise model of the circuit if any.
    
    Returns:
        circuit : cirq Circuit
            The random circuit created.

    '''
    circuit = cirq.Circuit()
    if initial_state == "+":
        for qubit in qubits:
            circuit += cirq.ry(np.pi/4)(qubit)
    elif initial_state == "i":
        for qubit in qubits:
            circuit += cirq.rx(-np.pi/4)(qubit)
    elif initial_state == "random":
        pass

    if noisy:
        assert noise_model != None, "Need to provide noise model for noisy circuit."

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
    #print(type(circuit))
    return circuit

def I_d(n_qubits):
    '''
    Returns the maximally mixed state with n_qubits.

    Parameters:
        n_qubits : int
            Number of qubits.
    
    Returns:
        np.ndarray
            The maximally mixed state of n_qubits qubits.
    '''
    return np.eye(2**n_qubits)/(2**n_qubits)

class Correlated_Dephasing(cirq.TwoQubitGate):
    '''
    Derived from cirq's TwoQubitGate class.
    Implements a channel with 
    '''
    def __init__(self, p: float) -> None:
        '''
        Initialise the correlated dephasing type channel.
        Parameters:
            p : float
                The probability of noisy channel.
        Returns:
            None
        '''
        self._p = p

    def _num_qubits_(self):
        '''
        returns and defines the number of qubits in the circuit.
        Right now it si for 2 qubits. Needs to be expanded.

        Parameters:
            None
        Returns:
            num_qubits: int
                Number of qubits in the circuit.
        '''
        return 2

    def _mixture_(self):
        '''
        Returns the operations that need to be done with their corresponding probabilities.
        Parameters:
            None
        Returns:
            tuple
                Tuple of the probability and the operation.
        '''
        ps = [self._p, self._p, self._p, 1.0 - 3*self._p]
        ops = [cirq.unitary(cirq.X)*cirq.unitary(cirq.X), cirq.unitary(cirq.Y)*cirq.unitary(cirq.Y), cirq.unitary(cirq.Z)*cirq.unitary(cirq.Z), cirq.unitary(cirq.I)*cirq.unitary(cirq.I)]
        return tuple(zip(ps, ops))

    def _has_mixture_(self) -> bool:
        '''
        Boolean to report if it is a mixture.
        '''
        return True

    def _circuit_diagram_info_(self, args='cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        '''
        Returns the information about the circuit diagram info when printed.
        '''
        return cirq.CircuitDiagramInfo(wire_symbols=(f"DP({self._p})", f"DP({self._p})"), exponent=1.0, connected=True)

class MyGateDepolarizingNoiseModel(cirq.NoiseModel):
    def noisy_operation(self, op):
        if isinstance(op.gate, MyGate):
            return [op, cirq.depolarize(p).on(op.qubits[0])]
        return op

def make_histogram(data, threshold = 0.9):
    '''
    Plots a histogram of given data with a user defined set of bin sizes.

    Parameters:
        data : np.ndarray
            Data that is to be plotted.
        threshold : float
            The threshold for the vertical line. Defaults to 0.9.
    
    Returns:
        None
    '''
    edges = [0.1*i for i in range(10)]
    temp_edges = [0.9 + 0.01*(i+1) for i in range(10) ]
    edges = edges + temp_edges
    plt.hist(data, bins = edges, range = (0,1), density = True)
    plt.axvline(x=threshold, color = 'r')
    #hist, edges = np.histogram(data, bins = edges, range = (0,1), density = True)
    plt.show()
    #return histogram
