#include <array>
#include <complex>
#include <iostream>

#include <Jet.hpp>

int main(){
    using Tensor = Jet::Tensor<std::complex<float>>;

    Tensor A;
    Tensor B({"i"}, {2});
    Tensor C({"i", "j"}, {4, 3});
    Tensor D({"i", "j", "k"}, {3, 2, 4})

    A.FillRandom();
    B.FillRandom();
    C.FillRandom(7);
    D.FillRandom(7);

    for (const auto &datum : A.GetData()) {
        std::cout << datum << std::endl;
    }

    std::cout << "You have successfully used Jet version " << Jet::Version() << std::endl;


    return 0;
}
