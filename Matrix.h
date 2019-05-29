#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H

#include <iostream>
#include <vector>
#include <iomanip>
#include <math.h>
#include <limits>
const double EPS = std::numeric_limits<double>::epsilon();

template<typename T> class Matrix {
    T **data;
    int cols, rows;
public:
    Matrix();
    Matrix(int, int); // initialized diminsions.
    Matrix(T**, int, int);
    Matrix(std::initializer_list<std::initializer_list<T>>);

    int get_cols() const {return this->cols;}
    int get_rows() const {return this->rows;}

    void exchange_rows(int, int);
    void exchange_cols(int, int);

    static Matrix<T> matmul(const Matrix<T>&, const Matrix<T>&);
    static Matrix<T> transpose(const Matrix<T>&);
	static Matrix<T> pow(const Matrix<T>&, const size_t &);
    friend std::ostream& operator<<(std::ostream &out, const Matrix<T> & mat) {
        for(int i = 0; i < mat.get_rows(); i++) {
            out << "{";
            for(int j = 0; j < mat.get_cols(); j++) {
                if(fabs(mat[i][j] - 0) < EPS)
                    out << std::fixed << std::setprecision(3) << std::setw(6) << (mat[i][j] = 0);
                else out << std::fixed << std::setprecision(3) << std::setw(6) << mat[i][j];
                if(j != mat.get_cols() - 1)
                    out << ' ';
            }
            out << "}";
            if(i != mat.get_rows() - 1)
                out << ',' << std::endl;
        }
        return out;
    }

    friend Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b) {
    	Matrix<T> res = matmul(a, b);
	return res;
    }

    friend Matrix<T> operator + (const Matrix<T>& a, const Matrix<T>& b) {
    	if(a.get_rows() != b. get_rows() || a.get_cols() != b.get_cols())
			throw std::runtime_error{"Matrices of different dimensions cannot be added"};
		Matrix<T> res {a.get_rows(), a.get_cols()};
		for(int i = 0; i < res.get_rows(); i++) {
			for(int j = 0; j < res.get_cols(); j++)
				res[i][j] = (a[i][j] + b[i][j]);
		}
		return res;
    }

	Matrix<T> operator *= (const Matrix<T>&);
	Matrix<T> operator += (const Matrix<T>&);
	
	friend Matrix<T> operator -  (const Matrix<T>& a) {
	    auto tmp = Matrix<T> {a.data, a.get_rows(), a.get_cols()};
	    for(int i = 0; i < a.get_rows(); i++) {
	        for(int j = 0; j < a.get_cols(); j++) {
		    tmp[i][j] = -1 * tmp[i][j];
  	        }
	    }
	    return tmp;
	}


    static Matrix<T> eye(int N);

    T* operator[](int i) const {
        return data[i];
    }

    void gaussian_elemination();
    int zero_rows();
    bool is_inconsistent();

    void upper();

    ~Matrix();
};

#endif // MATRIX_MATRIX_H
