#include "Perceptron.h"

Aurilius::Perceptron::Perceptron(const std::string &file_name, double learn_rate) {
    this->file_name = file_name;
    this->learn_rate = learn_rate;
    this->bias = 0;
}

void Aurilius::Perceptron::read_data() {
    std::ifstream f(file_name, std::ifstream::in);
    if(!f.good()) {
        throw std::runtime_error("Could not open the data file for Perceptron.\n");
    }
    f >> X;
    shuffle_data();
    y = X.get_col(X.cols() - 1);
    X.remove_col(X.cols() - 1);
    const size_t dim = X.cols();
    W = Matrix<double>::rand(dim, 1);
    W = Matrix<double>{{0.294665}, {0.53058676}};
//    bias = W[0][0] + X.max();
    bias = 1.1915207869474949;
}


void Aurilius::Perceptron::shuffle_data() {
    X.shuffle_rows();
}

double Aurilius::Perceptron::predict(size_t idx) {
    auto ro = Matrix<double>::row_vector(X[idx]);
    return predict(ro);
}

double Perceptron::predict(const Matrix<double>& inp) {
    auto dot_prod = Matrix<double>::dot_prod(inp, W) + bias;
    return Aurilius::Activations::step(dot_prod);
}

void Aurilius::Perceptron::training_step() {
    size_t len = X.rows();
    for(int i = 0; i < len; i++) {
        double y_hat = predict(i);
        if(fabs(y[i][0] - y_hat - 1.0) <= EPS) {
            increase_weights(i);
        } else if(fabs(y[i][0] - y_hat + 1.0) <= EPS) {
            decrease_weights(i);
        }
    }
}

void Aurilius::Perceptron::increase_weights(size_t idx) {
    for(int i = 0; i < W.rows(); i++) {
        W[i][0] += X[idx][i] * learn_rate;
    }
    bias += learn_rate;
}

void Aurilius::Perceptron::decrease_weights(size_t idx) {
    for(int i = 0; i < W.rows(); i++) {
        W[i][0] -= X[idx][i] * learn_rate;
    }
    bias -= learn_rate;
}

void Aurilius::Perceptron::train(size_t epochs) {
    for(size_t i = 0; i < epochs; i++) {
        training_step();
    }
}