import jet

size = [2, 2]
indices = ["i", "j"]

pauli_x_data = [0, 1, 1, 0]
pauli_y_data = [0, -1j, 1j, 0]
pauli_z_data = [1, 0, 0, -1]

X = jet.Tensor(indices, size, pauli_x_data);
Y = jet.Tensor(indices, size, pauli_y_data);
Z = jet.Tensor(indices, size, pauli_z_data);

bra = jet.Tensor(indices = ["i"], shape=[2], data = [1,0] )
ket = jet.Tensor(bra)

op_ket = Z.contract_with_tensor(ket)
bra_op_ket = bra.contract_with_tensor(op_ket)

print("<0|Z|0> =", bra_op_ket.scalar)
