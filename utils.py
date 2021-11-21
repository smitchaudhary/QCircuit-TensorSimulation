#import jet
import numpy as np
import cirq
import matplotlib.pyplot as plt
import sympy as sp

# Identity Gate
I = np.array([[1, 0],#
              [0, 1]])

# X Gate
X = np.array([[0, 1],#
              [1, 0]])

# Y Gate
Y = np.array([[0, -1j],#
              [1j, 0]])

#Z Gate
Z = np.array([[1, 0],#
              [0, -1]])

# Dictionary containing the Pauli Gates
paulis = {'0' : I, '1' : X, '2' : Y, '3' : Z}


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
            #random_n = np.random.random()
            #random_rot = 2*np.pi*np.random.random()

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
    temp_edges = [0.9 + 0.01*(i+1) for i in range(9) ]
    edges = edges + temp_edges
    temp_edges = [0.99 + 0.001*(i+1) for i in range(100)]
    edges = edges + temp_edges
    print(edges)
    plt.hist(data, bins = edges, range = (0,1), density = True)
    plt.axvline(x=threshold, color = 'r')
    #hist, edges = np.histogram(data, bins = edges, range = (0,1), density = True)
    plt.show()
    #return histogram

def random_initial_state(n_qubits):
    '''
    Produces an n-qubit random superposition state.
    Parameters :
        n_qubits : int
            Number of qubits in the system
    Returns :
        rand_state : np.ndarray
            A random n_qubit quantum state
    '''
    RC = np.random.randn(2**n_qubits, 2**n_qubits) + 1j*np.random.randn(2**n_qubits, 2**n_qubits)
    q, r = np.linalg.qr(RC)
    d = np.diag(r)
    ph = d/np.abs(d)
    q = q*ph
    init_state = np.array([0 for i in range(2**n_qubits)])
    init_state[0] = 1
    rand_state = np.matmul(q, init_state)
    return rand_state

def operational_to_channel(rho = None, operation_dictionary = None, n_qubits = None, symbolic = False):
    '''
    From the operational definitio of a channel, such as apply X, Y, Z with probability p and do nothing with probability 1-p,
    it gives the channel form that gives how the density matrix evolves.
    Parameters :
        rho : np.ndarray or None
            The density matrix of the system. Either given in form of a 2D array or treated as symbolic rho by default.
        operation_dictionary : dict
            The Dictionary containing what noise to apply with what probability.
        n_qubit : int
            Number of qubits in the system. If not given, and rho is not symbolic, deduced from the size of rho.
        symbolic : bool
            A boolean indicating if rho is symbolic or given as an array.
    Returns :
        new_rho_decomposition : dict
            The new density matrix
    '''
    if not symbolic:
        n_qubits = int(np.log2(len(rho)))
        assert rho != None, "Not a symbolic calculation. Need to provide the current density matrix."
        if not operation_dictionary:
            print(f'No noise applied since no operations are given.')
            #operation_dictionary = {}
            #operation_dictionary['0'] = 1
            return rho
        total_prob = 0
        for pauli_string in operation_dictionary:
            total_prob += operation_dictionary[pauli_string]
        assert total_prob == 1, "Check the probability of each operation. The total probability needs to be 1"
        if isinstance(rho, dict):
            rho_decomposition = dict(rho)
        else:
            rho_decomposition = pauli_decomposition(rho, verbose = False)
        new_rho_decomposition = {}
        for component in rho_decomposition:
            new_rho_decomposition[component] = 0
            for pauli_string in operation_dictionary:
                new_rho_decomposition[component] += rho_decomposition[component]*commute_or_anti_commute(component, pauli_string)*operation_dictionary[pauli_string]
        return new_rho_decomposition
    else:
        assert n_qubits != None, "If rho is not given explicitly, you have to provide the number of qubits."
        rho, rho_decomposition = symbolic_rho(n_qubits)
        new_rho_decomposition = {}
        for component in rho_decomposition:
            new_rho_decomposition[component] = 0
            for pauli_string in operation_dictionary:
                new_rho_decomposition[component] += rho_decomposition[component]*commute_or_anti_commute(component, pauli_string)*operation_dictionary[pauli_string]
        return new_rho_decomposition


def generate_all_pauli_strings(n_qubits = 1):
    '''
    Generates all 4**n_qubits paulis trings for any n_qubit.
    Parameters:
        n_qubits : int
            Number of qubits in the system. 1 by default.
    Returns:
        indices : list
            A list with all indices with appropriate padding.
    '''
    indices = [np.base_repr(i, base = 4, padding = (n_qubits - len(np.base_repr(i, base = 4)))) for i in range(4**n_qubits)]
    indices[0] = '0'*n_qubits
    return indices

def pauli_decomposition(M, verbose = False):
    '''
    Deconstructs any 2D square matrix of dimensions 2**n_qubits into sum of Pauli strings.
    Parameters:
        M : np.ndarray
            The matrix that is to be decomposed.
        verbose : bool
            False by default. If True, it outputs 0 components as well.
    Returns :
        decomposition : dict
            Dictionary with paulis string as key and the corresponding component as the value.
    '''
    size = np.shape(M)
    assert size[0] == size[1] and len(size) == 2, "Please provide a 2D square matrix."
    n_qubits = int(np.log2(size[0]))
    assert np.log2(size[0]) - n_qubits == 0, "The matrix must have the dimensions 2**n_qubits"
    pauli_strings = generate_all_pauli_strings(n_qubits = n_qubits)
    decomposition = {}
    for string in pauli_strings:
        A = matrix_from_index_string(string)
        component = np.trace(np.matmul(A, M))/2**n_qubits
        if verbose or component:
            decomposition[string] = component
    return decomposition

def matrix_recomposition(pauli_components):
    '''
    Reconstructs full 2D matrix based on the Pauli components given.
    Parameters:
        pauli_components : dict
            Dictionarty containing the pauli string in index form as key and the component as the value.
    Returns:
        M : np.ndarray
            The finaly matrix corresponding to the components given.
    '''
    M = 0
    for string in pauli_components:
        A = matrix_from_index_string(string)
        M += pauli_components[string]*A
    return M

def matrix_from_index_string(string):
    '''
    Gives you an explicit 2D matrix based on the Pauli string.
    Position of the index determines on which qubit.
    0 : I
    1 : X
    2 : Y
    3 : Z
    Parameters:
        string : str
            String of indices for the Pauli string.
    Returns:
        mat : np.ndarray
            2D array that is the explicit matrix for the pauli string given.
    '''
    mat = [1]
    for pauli in string:
        mat = np.kron(mat, paulis[pauli])
    return mat

def commute_or_anti_commute(string1, string2):
    '''
    Tells you if two pauli strings commute or anti commute.
    Outputs 1 if they commute and -1 if they do not.
    Parameters :
        string1 : str
            First Pauli string
        string2 : str
            Second Pauli string
    Returns :
        ans : int
            ans = 1 if the two strings commute and ans = -1 if they anti commute.
    '''
    assert len(string1) == len(string2), "The two strings have to be same length to compare"
    ans = 1
    for i in range(len(string1)):
        if string1[i] == '0' or string2[i] == '0' or string1[i] == string2[i]:
            continue
        ans *= -1
    return ans

def symbolic_rho(n_qubits):
    '''
    Gives you the symbolic rho as well as the decomposition into pauli string components.
    Parameters :
        n_qubits : int
            The number of qubits in the system.
    Returns :
        rho : sympy object
            The symbolic form of rho
        rho_decomposition : dict
            Dictionary containing the components for each pauli string
    '''
    paulis_strings = generate_all_pauli_strings(n_qubits)
    rho = 0
    rho_decomposition = {}
    for string in paulis_strings:
        if string == '0'*n_qubits:
            pauli = sp.Symbol('P_'+string)
            component = 1/2**n_qubits
            rho += pauli*component
            rho_decomposition[string] = 1/2**n_qubits
        else:
            pauli = sp.Symbol('P_'+string)
            component = sp.Symbol('rho_'+string)
            rho += pauli*component
            rho_decomposition[string] = component
    return rho, rho_decomposition
