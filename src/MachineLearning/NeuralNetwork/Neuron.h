#pragma once
#include "../MachineLearning.h"

class Aurilius::MachineLearning::ANN::Neuron {
public:
    Neuron(double, std::function<double(double)>, std::function<double(double)>);
    void activate();
    void differentiate();

    double get_value() const { return value; };
    double get_activated_value() const { return activated_value; };
    double get_derived_value() const { return derived_value; };

    void set_value(double val);
private:
    double value, activated_value, derived_value;
    std::function<double(double)> activation, activation_prime;
};
