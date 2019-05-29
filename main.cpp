#include <iostream>
#include "Matrix.h"

using std::cout;
using std::endl;

int main() {
	Matrix<int> x {{1,2,3}, {2,2,2}};
	Matrix<int> y {{1,2,3}, {2,2,2}};
    	cout << -y;
	return 0;
}
