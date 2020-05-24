#include "Neuron.h"
using namespace Aurilius::MachineLearning::ANN;

Neuron::Neuron(double val, std::function<double(double)> activation, std::function<double(double)> derivative) {
    this->value = val;
    this->activation = std::move(activation);
    this->activation_prime = std::move(derivative);
}

void Neuron::activate() {
    activated_value = activation(value);
}

void Neuron::differentiate() {
    derived_value = activation_prime(activated_value);
}

void Neuron::set_value(double val) {
    this->value = val;
    this->activate();
    this->differentiate();
}
