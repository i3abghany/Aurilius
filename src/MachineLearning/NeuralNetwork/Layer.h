#pragma once
#include "../MachineLearning.h"

class Aurilius::MachineLearning::ANN::Layer {
public:
    Layer(size_t, std::function<double(double)>, std::function<double(double)>);
    ~Layer();
    void set_vals(const Matrix<double> &);
    Matrix<double> get_vals();
    Matrix<double> get_activated_vals();
    Matrix<double> get_derived_vals();
    Neuron *operator[](size_t idx) {
        return neurons.at(idx);
    }
private:
    size_t size;
    std::vector<Neuron *> neurons;
    std::function<double(double)> activation, activation_prime;
    void activate();
    void differentiate();
};
