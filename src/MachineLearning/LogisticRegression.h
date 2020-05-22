#pragma once
#include "MachineLearning.h"

class Aurilius::MachineLearning::LogisticRegression {
    Matrix<double> X, y;
    Matrix<double> W;    double bias;
    std::string file_name;
    double learn_rate;
    double error(double, double);
    double output(size_t);
    void update_weights(size_t);
public:

    explicit LogisticRegression(const std::string &, double learn_rate = 0.01);
    void read_data();
    void shuffle_data();
    void train(size_t = 100);
    double predict(const Matrix<double>&);
};
