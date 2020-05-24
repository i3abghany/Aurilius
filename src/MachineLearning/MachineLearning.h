#pragma once

namespace Aurilius::MachineLearning {
    class Perceptron;
    class LogisticRegression;
    namespace ANN {
        class Neuron;
        class Layer;
        class NeuralNetwork;
    }
}

#include "../Matrix/Matrix.h"
#include "Perceptron.h"
#include "LogisticRegression.h"
#include "Activations.h"
#include "Loss.h"
#include "NeuralNetwork/NeuralNetwork.h"