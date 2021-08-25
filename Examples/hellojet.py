import jet

lhs = jet.Tensor(["i", "j", "k"], [2, 2, 2])
rhs = jet.Tensor(["j", "k", "l"], [2, 2, 2])

lhs.fill_random()
rhs.fill_random()
res = jet.contract_tensors(lhs, rhs)

for datum in res.data:
    print(f"{datum:.5f}")

print("You have successfully used Jet version", jet.version())
