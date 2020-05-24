#pragma once
#include "../Aurilius.h"
#include <map>

namespace Aurilius::MachineLearning::Activations {
    double step(double);
    Matrix<double> softmax(const Matrix<double> &);
    double sigmoid(double);

    double sigmoid_prime(double);
}
