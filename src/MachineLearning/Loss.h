#pragma once
#include "../Aurilius.h"

namespace Aurilius::MachineLearning::Loss {
    double cross_entropy(const Matrix<double> &y, const Matrix<double> &p);
}