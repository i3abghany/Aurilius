#include <iostream>
#include <iomanip>
#include "Aurilius.h"

using namespace Aurilius;
using namespace Aurilius::MachineLearning;

int main() {

    Matrix<double> A {
            {1, 1},
            {1, 2},
            {1, 3},
            {1, 4},
            {1, 5},
    };

    auto b = Matrix<double>::col_vector({1, 2, 2, 2, 2});
    //project the b vector into the column space of A C(A)
    //and then tuck it to make an augmented matrix.
    A.tuck_cols(Matrix<double>::project_into_col_space(A, b));
    A.gaussian_elimination(true); // will print out c and d in (y = c + d*x), the best-fit line.

    const auto f = [](double x) {
        return 1.0 / std::sqrt(1 + x);
    };

//     will approximate f(x) = 1/sqrt(1 + x) from 0 to 2.
    std::cout << std::setprecision(6) << trapezoidal(f, 0, 2);

    return 0;

}