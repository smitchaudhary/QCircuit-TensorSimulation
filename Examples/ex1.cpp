#include <array>
#include <complex>
#include <iostream>

#include <Jet.hpp>

int main(){
    using Tensor = Jet::Tensor<std::complex<float>>;

    std::array<Tensor, 3> tensors;
    tensors[0] = Tensor({"i", "j", "k"}, {2, 2, 2});
    tensors[1] = Tensor({"j", "k", "l"}, {2, 2, 2});

    tensors[0].FillRandom();
    tensors[1].FillRandom();
    tensors[2] = Tensor::ContractTensors(tensors[0], tensors[1]);

    for (const auto &datum : tensors[2].GetData()) {
        std::cout << datum << std::endl;
    }

    std::cout << "You have successfully used Jet version " << Jet::Version() << std::endl;

    return 0;
}
