#include "Perceptron.h"

Aurilius::MachineLearning::Perceptron::Perceptron(const std::string &file_name, double learn_rate) {
    this->file_name = file_name;
    this->learn_rate = learn_rate;
    this->bias = 0;
}

void Aurilius::MachineLearning::Perceptron::read_data() {
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
    bias = W[0][0] + X.max();
}


void Aurilius::MachineLearning::Perceptron::shuffle_data() {
    X.shuffle_rows();
}

double Aurilius::MachineLearning::Perceptron::predict(size_t idx) {
    auto ro = Matrix<double>::row_vector(X[idx]);
    return predict(ro);
}

double Aurilius::MachineLearning::Perceptron::predict(const Matrix<double>& inp) {
    auto dot_prod = Matrix<double>::dot_prod(inp, W) + bias;
    return Aurilius::MachineLearning::Activations::step(dot_prod);
}

void Aurilius::MachineLearning::Perceptron::training_step() {
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

void Aurilius::MachineLearning::Perceptron::increase_weights(size_t idx) {
    for(int i = 0; i < W.rows(); i++) {
        W[i][0] += X[idx][i] * learn_rate;
    }
    bias += learn_rate;
}

void Aurilius::MachineLearning::Perceptron::decrease_weights(size_t idx) {
    for(int i = 0; i < W.rows(); i++) {
        W[i][0] -= X[idx][i] * learn_rate;
    }
    bias -= learn_rate;
}

void Aurilius::MachineLearning::Perceptron::train(size_t epochs) {
    for(size_t i = 0; i < epochs; i++) {
        training_step();
    }
}
