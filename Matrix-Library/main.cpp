#include <iostream>
#include "Matrix.h"

int main() {

	Matrix<double> A{
			{1, 1},
			{1, 2},
			{1, 3},
			{1, 4},
			{1, 5},
	};

	auto b = Matrix<double>::col_vector({ 1, 2, 2, 2, 2 });
	
	// //project the b vector into the column space of A C(A)
	// //and then tuck it to make an augmented matrix.
	A.tuck_cols(Matrix<double>::project_into_col_space(A, b));
	A.gaussian_elimination(); // will print out c and d in (y = c + d*x), the best-fit line.
	
	std::cout << 5 * Matrix<double>::eye(4) - Matrix<double>::ones(4, 4);

	return 0;
}
