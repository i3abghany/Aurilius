#include "Layer.h"

using namespace Aurilius::MachineLearning::ANN;

Layer::Layer(size_t size, std::function<double(double)> activation, std::function<double(double)> activation_prime) {
    this->size = size;
    this->activation = std::move(activation);
    this->activation_prime = std::move(activation_prime);
    this->neurons.resize(size);
    for(auto &el : this->neurons) {
        el = new Neuron(0.0, this->activation, this->activation_prime);
    }
}

void Layer::activate() {
    for(auto &n : neurons) {
        n->activate();
    }
}

void Layer::differentiate() {
    for(auto &n : neurons) {
        n->differentiate();
    }
}

Layer::~Layer() {
    for (auto &n : neurons) {
        delete n;
    }
    this->neurons.clear();
}

Matrix<double> Layer::get_vals() {
    auto res = Matrix<double>{this->size, 1};
    for(size_t i = 0; i < this->size; i++) {
        res[i][0] = (*this)[i]->get_value();
    }
    return res;
}

Matrix<double> Layer::get_activated_vals() {
    auto res = Matrix<double>{this->size, 1};
    for(size_t i = 0; i < this->size; i++) {
        res[i][0] = (*this)[i]->get_activated_value();
    }
    return res;
}

Matrix<double> Layer::get_derived_vals() {
    auto res = Matrix<double>{this->size, 1};
    for(size_t i = 0; i < this->size; i++) {
        res[i][0] = (*this)[i]->get_derived_value();
    }
    return res;
}

void Layer::set_vals(const Matrix<double> &inp) {
    for(int i = 0; i < this->size; i++) {
        this->neurons[i]->set_value(inp[0][i]);
    }
}
