#include "Activations.h"

double Aurilius::MachineLearning::Activations::step(double t) {
    return (t < 0) ? 0.0 : 1.0;
}

Matrix<double> Aurilius::MachineLearning::Activations::softmax(const Aurilius::Matrix<double> &L) {
    auto denom = Matrix<double>::sum(Matrix<double>::map(L, [] (double el) {
        return std::exp(el);
    }));
    auto res = Matrix<double> {1, L.cols()};
    for(int i = 0; i < L.cols(); i++) {
        res[0][i] = std::exp(L[0][i]) / denom;
    }
    return res;
}

double Aurilius::MachineLearning::Activations::sigmoid(double inp) {
    return 1.0 / (1 + std::exp(-inp));
}


double Aurilius::MachineLearning::Activations::sigmoid_prime(double inp) {
    return sigmoid(inp) * (1 - sigmoid(inp));
}