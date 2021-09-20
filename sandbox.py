import cirq
import numpy as np
from tools import *

cor_DP = Correlated_Dephasing(p=0.05)

#for prob, kraus in cirq.mixture(cor_DP):
#    print(f"With probability {prob}, apply\n", kraus, end="\n\n")

qubits = cirq.LineQubit.range(2)
circ = cirq.Circuit(cor_DP.on_each(qubits))
circ = circ.with_noise(cirq.depolarize(0.05))
#print(circ)
for qubit in qubits:
    circ += cirq.ry(np.pi/4)(qubit)
for qubit in qubits:
    circ += cirq.rx(np.pi/4)(qubit)
print(circ)
