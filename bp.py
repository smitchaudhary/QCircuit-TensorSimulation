import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

def generate_random_qnn(qubits, symbol, depth, noisy = False):
    circuit = cirq.Circuit()
    if noisy:
        circuit = circuit.with_noise(cirq.depolarize(0.05))
    for qubit in qubits:
        circuit += cirq.ry(np.pi/4)(qubit)

    for d in range(depth):
        for i, qubit in enumerate(qubits):
            random_n = np.random.random()
            random_rot = 2*np.pi*np.random.random() if i != 0 or d != 0 else symbol

            if random_n > 2/3:
                circuit += cirq.rz(random_rot)(qubit)
            elif random_n > 1/3:
                circuit += cirq.ry(random_rot)(qubit)
            else:
                circuit += cirq.rx(random_rot)(qubit)

        for src, dest in zip(qubits, qubits[1:]):
            circuit += cirq.CZ(src, dest)

    return circuit

#generate_random_qnn(cirq.GridQubit.rect(1, 3), sympy.Symbol('theta'), 2)

def process_batch(circuits, symbol, op):
    expectation = tfq.layers.Expectation()

    circuit_tensor = tfq.convert_to_tensor(circuits)
    values_tensor = tf.convert_to_tensor(np.random.uniform(0, 2 * np.pi, (n_circuits, 1)).astype(np.float32))

    with tf.GradientTape() as g:
        g.watch(values_tensor)
        forward = expectation(circuit_tensor, operators=op, symbol_names=[symbol], symbol_values=values_tensor)

    grads = g.gradient(forward, values_tensor)
    grad_var = tf.math.reduce_std(grads, axis=0)
    return grad_var.numpy()[0]

n_qubits = [2*i for i in range(2, 7)]
depth = 50
n_circuits = 200
theta_var = []
theta_var_noisy = []

for n in n_qubits:
    qubits = cirq.GridQubit.rect(1, n)
    symbol = sympy.Symbol('theta')
    circuits = [generate_random_qnn(qubits, symbol, depth) for _ in range(n_circuits)]
    op = cirq.Z(qubits[0]) * cirq.Z(qubits[1])
    theta_var.append(process_batch(circuits, symbol, op))

for n in n_qubits:
    qubits = cirq.GridQubit.rect(1, n)
    symbol = sympy.Symbol('theta')
    circuits = [generate_random_qnn(qubits, symbol, depth, noisy = True) for _ in range(n_circuits)]
    op = cirq.Z(qubits[0]) * cirq.Z(qubits[1])
    theta_var_noisy.append(process_batch(circuits, symbol, op))


#print(cirq.final_density_matrix(circuits[0]))
plt.semilogy(n_qubits, theta_var)
plt.semilogy(n_qubits, theta_var_noisy)
plt.title('Gradient Variance in QNNs')
plt.xlabel('n_qubits')
plt.ylabel('$\\partial \\theta$ variance')
plt.show()
