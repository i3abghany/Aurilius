#pragma once
#include "Aurilius.h"

namespace Aurilius::Activations {
    double step(double);
    Matrix<double> softmax(const Matrix<double> &);
    double sigmoid(double);
}
