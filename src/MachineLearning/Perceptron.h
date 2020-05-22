#pragma once
#include "../Matrix/Matrix.h"
#include "Activations.h"

namespace Aurilius::MachineLearning {
    class Perceptron;
}

class Aurilius::MachineLearning::Perceptron {
    Matrix<double> X, y;
    Matrix<double> W;
    std::string file_name;
    double bias;
    double learn_rate;

    void training_step();
    void increase_weights(size_t);
    void decrease_weights(size_t);
    double predict(size_t);
public:
    explicit Perceptron(const std::string &file_name, double learn_rate = 0.01);
    void read_data();
    void shuffle_data();
    void train(size_t epochs = 25);
    double predict(const Matrix<double>&);
};
