#include "Loss.h"

double Aurilius::MachineLearning::Loss::cross_entropy(const Matrix<double> &y, const Matrix<double> &p) {
    return -Matrix<double>::sum(Matrix<double>::elementwise_mul(y, Matrix<double>::map(p, [](double elem) {
        return log(elem);
    })));
}
