#include "NeuralNetwork.h"

using namespace Aurilius::MachineLearning::ANN;

NeuralNetwork::NeuralNetwork(std::vector<size_t> topology,
        std::function<double (double)> activation,
        std::function<double (double)> activation_prime,
        double learn_rate, size_t epochs) {
    this->topology = std::move(topology);
    this->activation = std::move(activation);
    this->activation_prime = std::move(activation_prime);

    initialize_layers();
    initialize_weights();
    this->learn_rate = learn_rate, this->epochs = epochs;
}

void NeuralNetwork::initialize_layers() {
    this->layers.resize(this->topology.size());

    for(int i = 0; i < this->topology.size(); i++) {
        this->layers[i] = new Layer(this->topology[i], this->activation, this->activation_prime);
    }
}

void NeuralNetwork::initialize_weights() {
    this->weight_matrices.resize(this->topology.size() - 1);
    for(int i = 0; i < this->topology.size() - 1; i++) {
        this->weight_matrices[i] = new Matrix<double> {this->topology[i], this->topology[i + 1]};
        this->weight_matrices[i]->randomize();
    }
}

void NeuralNetwork::load_data(const std::string &file_name) {
    std::ifstream f(file_name, std::ifstream::in);
    if (!f.good()) {
        throw std::runtime_error("Could not open the data file for Neural Network..\n");
    }
    size_t result_cols = this->topology.at(this->topology.size() - 1);
    f >> X;
    for (size_t i = X.cols() - result_cols - 1; i < X.cols(); i++) {
        y.tuck_cols(X.get_col(i));
        X.remove_col(i);
    }
    this->num_features = X.cols();
    this->num_targets = y.cols();
}

void NeuralNetwork::train() {
    const size_t data_rows = X.rows();
    for (size_t i = 0; i < data_rows; i++) {
        auto y_hat = feed_forward(i);
    }
}

Matrix<double> NeuralNetwork::feed_forward(size_t inp_idx) {
    Matrix<double> a = Matrix<double>::row_vector(X[inp_idx]);
    return feed_forward(a);
}


Matrix<double> NeuralNetwork::feed_forward(const Matrix<double> &inp) {
    Matrix<double>  a = inp; // Inputs to the layer.
    Matrix<double> *b; // Weights Matrix.
    Matrix<double>  c; // Output values of the layer, to be fed to the next.

    Matrix<double> outp = Matrix<double>();
    for (size_t i = 0; i < this->layers.size() - 1; i++) {
        if (i != 0) {
            a = Matrix<double>::transpose(this->layers[i]->get_activated_vals());
        }
        b = this->weight_matrices[i];
        c = Matrix<double>::matmul(a, *b);
        this->layers[i + 1]->set_vals(c);
    }
    return this->layers[this->layers.size() - 1]->get_activated_vals();
}


NeuralNetwork::~NeuralNetwork() {
    for (auto &l : layers) {
        delete l;
    }
    for (auto &m : weight_matrices) {
        delete m;
    }
}

Matrix<double> NeuralNetwork::predict(const std::vector<double> &a) {
    auto inp = Matrix<double>::row_vector(a);
    return this->feed_forward(inp);
}

