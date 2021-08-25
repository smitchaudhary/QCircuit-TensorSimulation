import jet

q0 = jet.Tensor(["i"], [2], [1,0])
q1 = jet.Tensor(["j"], [2], [1,0])

hadamard = [2**-0.5, 2**-0.5, 2**-0.5, -1*2**-0.5]

H = jet.Tensor(["i", "k"], [2, 2], hadamard)

CNOT = jet.Tensor(["k", "j", "m", "n"], [2, 2, 2, 2])
CNOT.set_value((0,0,0,0),1)
CNOT.set_value((0,1,0,1),1)
CNOT.set_value((1,0,1,1),1)
CNOT.set_value((1,1,1,0),1)

tn = jet.TensorNetwork()

tn.add_tensor(q0)
tn.add_tensor(q1)
tn.add_tensor(H)
tn.add_tensor(CNOT)

result = tn.contract()

print("Amplitude of |00> = ", result.get_value([0,0]))
print("Amplitude of |01> = ", result.get_value([0,1]))
print("Amplitude of |10> = ", result.get_value([1,0]))
print("Amplitude of |11> = ", result.get_value([1,1]))
