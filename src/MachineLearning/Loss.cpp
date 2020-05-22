#include "Loss.h"

double Aurilius::MachineLearning::Loss::cross_entropy(const Matrix<double> &y, const Matrix<double> &p) {
    auto one = Matrix<double>::ones(y.rows(), y.cols());
    auto t1 = Matrix<double>::elementwise_mul(y, Matrix<double>::map(p, [](double x) {
        return log(x);
    }));

    auto t2 = Matrix<double>::elementwise_mul((one - y), Matrix<double>::map(one - p, [](double x) {
        return log(x);
    }));
    return -Matrix<double>::sum(t1 + t2);
}