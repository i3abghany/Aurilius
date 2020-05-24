project(Aurilius)

set(CMAKE_CXX_STANDARD 17)
SET(GCC_COVERAGE_COMPILE_FLAGS "-fopenmp")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
add_executable(pin src/main.cpp src/Aurilius.h src/Matrix/Matrix.h src/Matrix/Matrix_impl.hpp src/NumericalIntegration/Numerical_integration.hpp src/MachineLearning/Activations.h src/MachineLearning/Perceptron.h src/MachineLearning/Perceptron.cpp src/MachineLearning/Activations.cpp src/MachineLearning/MachineLearning.h src/MachineLearning/Loss.h src/MachineLearning/Loss.cpp src/MachineLearning/LogisticRegression.cpp src/MachineLearning/LogisticRegression.h src/MachineLearning/NeuralNetwork/Neuron.cpp src/MachineLearning/NeuralNetwork/Neuron.h src/MachineLearning/NeuralNetwork/Layer.cpp src/MachineLearning/NeuralNetwork/Layer.h src/MachineLearning/NeuralNetwork/NeuralNetwork.cpp src/MachineLearning/NeuralNetwork/NeuralNetwork.h)
