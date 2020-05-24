#pragma once
#include "../MachineLearning.h"
#include "Neuron.h"
#include "Layer.h"

class Aurilius::MachineLearning::ANN::NeuralNetwork {
public:
    NeuralNetwork(std::vector<size_t>, std::function<double(double)>, std::function<double(double)>, double = 0.01, size_t = 500);
    Matrix<double> predict(const std::vector<double> &);
    ~NeuralNetwork();
private:
    void initialize_layers();
    void initialize_weights();
    void load_data(const std::string &);

    Matrix<double> X, y;
    size_t num_features, num_targets;
    size_t epochs;
    double learn_rate;
    void train();
    Matrix<double> feed_forward(size_t);
    Matrix<double> feed_forward(const Matrix<double> &);
    std::vector<size_t> topology;
    std::vector<Layer *> layers;
    std::vector<Matrix<double> *> weight_matrices;
    std::function<double(double)> activation, activation_prime;
};
