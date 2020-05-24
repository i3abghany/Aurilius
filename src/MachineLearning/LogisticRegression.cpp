#include "LogisticRegression.h"

Aurilius::MachineLearning::LogisticRegression::LogisticRegression(const std::string &file_name, double learn_rate) {
    this->file_name = file_name;
    this->learn_rate = learn_rate;
    this->bias = 0;
}

void Aurilius::MachineLearning::LogisticRegression::read_data() {
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
}

void Aurilius::MachineLearning::LogisticRegression::shuffle_data() {
    X.shuffle_rows();
}

void Aurilius::MachineLearning::LogisticRegression::train(size_t epochs) {
    for(int e = 0; e < epochs; e++) {
        for(int i = 0; i < X.rows(); i++) {
            update_weights(i);
        }
    }
}

double Aurilius::MachineLearning::LogisticRegression::error(double yi, double op) {
    return -yi * log(op) - (1 - yi) * log(1 - op);
}

double Aurilius::MachineLearning::LogisticRegression::output(size_t i) {
    return Activations::sigmoid(Matrix<double>::dot_prod(Matrix<double>::row_vector(X[i]), W) + bias);
}

void Aurilius::MachineLearning::LogisticRegression::update_weights(size_t i) {
    auto op = output(i);
    auto d_error = (op - y[i][0]) * learn_rate;
    W -= (Matrix<double>::col_vector(X[i]) * d_error);
    bias -= d_error;
}

double Aurilius::MachineLearning::LogisticRegression::predict(const Matrix<double>& inp) {
    auto dot_prod = Matrix<double>::dot_prod(inp, W) + bias;
    return Aurilius::MachineLearning::Activations::step(dot_prod);
}
