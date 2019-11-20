#pragma once

#include<iostream>
#include <vector>
#include <iomanip>
#include <math.h>
#include <limits>
#include <algorithm>
#include <sstream>
#include <type_traits>

const double EPS = std::numeric_limits<double>::epsilon() * 1e6;

template<typename T>
class Matrix {
	static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
		"Matrix template type must be a floating point or integral type.");
	std::vector<std::vector<T>> data;
	std::pair<Matrix<T>, std::size_t> upper();
	void print_solutions(const std::vector<std::size_t>&);
	static std::size_t num_of_perms(const std::vector<std::size_t>&);

	static T elem_det(Matrix<T>);
	static T big_det(Matrix<T>);

public:
	Matrix() = default;
	Matrix(std::size_t, std::size_t, T = T{}); // initialized dimensions.
	Matrix(T**, std::size_t, std::size_t);
	Matrix(std::initializer_list<std::initializer_list<T>>);
	explicit Matrix(std::vector<std::vector<T>>);

	Matrix(const Matrix<T>&) = default;
	Matrix(Matrix<T>&&) noexcept = default;

	Matrix<T>& operator=(const Matrix<T>&) = default;
	Matrix<T>& operator=(Matrix<T>&&) noexcept = default;

	std::size_t cols() const { return this->data[0].size(); }
	std::size_t rows() const { return this->data.size(); }
	std::pair<std::size_t, std::size_t> size() const { return { this->rows(), this->cols() }; }

	void add_row(const std::vector<T>&);
	void add_col(const std::vector<T>&);

	void tuck_rows(const Matrix<T>&);
	void tuck_cols(const Matrix<T>&);

	void exchange_rows(const std::size_t, const std::size_t);
	void exchange_cols(const std::size_t, const std::size_t);

	static T dot_prod(const Matrix<T>&, const Matrix<T>&);
	void normalize_col(const std::size_t);

	static Matrix<T> project(const Matrix<T>&, const Matrix<T>&);
	static Matrix<T> project_into_col_space(const Matrix<T>&, const Matrix<T>&);

	static Matrix<T> col_vector(const std::vector<T>&);
	static Matrix<T> row_vector(const std::vector<T>&);

	bool is_row() const;
	bool is_col() const;

	Matrix<T> get_col(const std::size_t);
	void insert_col(const Matrix<T>&, const std::size_t);
	void remove_col(const std::size_t);

	void fill(const T);

	static Matrix<T> matmul(const Matrix<T>&, const Matrix<T>&);
	static Matrix<T> transpose(const Matrix<T>&);
	static Matrix<T> upper(Matrix<T>);
	static std::pair<Matrix<T>, Matrix<T>> LU(const Matrix<T>&);
	static bool is_symmetric(const Matrix&);
	static T det(Matrix<T>&);
	static T trace(const Matrix<T>&);


	friend std::ostream& operator<<(std::ostream& out, const Matrix<T>& mat) {
		for (std::size_t i = 0; i < mat.rows(); i++) {
			out << "[";
			for (std::size_t j = 0; j < mat.cols(); j++) {
				if (fabs(mat[i][j] - 0) < EPS) {
					out << std::fixed << std::setprecision(3) << std::setw(6) << (mat[i][j] = 0);
				}
				else out << std::fixed << std::setprecision(3) << std::setw(6) << mat[i][j];
				if (j != mat.cols() - 1) {
					out << ' ';
				}
			}
			out << "]";
			if (i != mat.rows() - 1) {
				out << ',' << std::endl;
			}
		}
		out << std::endl;
		return out;
	}

	// rows separated by semicolon.
	friend std::istream& operator>>(std::istream & in, Matrix<T> & mat) {
		// [1 2 3; 4 5 6; 7 8 9;]
		std::string tmp;
		std::string raw_form;
		while (getline(in, tmp) && tmp.find(']') == std::string::npos) {
			raw_form += tmp;
		}
		raw_form += tmp;
		raw_form.erase(std::begin(raw_form));
		raw_form.erase(std::end(raw_form) - 1);
		if(raw_form.back() != ';') {
            raw_form.push_back(';');
        }
		std::size_t num_rows = std::count_if(std::begin(raw_form), std::end(raw_form), [](const int c) { return c == ';'; });
		std::replace_if(std::begin(raw_form), std::end(raw_form), [](const int c) { return c == ';'; }, '\n');
		std::stringstream ss(raw_form);
		std::vector<std::vector<T>> tmp_data(num_rows, std::vector<T>());
		int i = 0;
        ss.exceptions(std::istream::badbit);
        try {
            while (getline(ss, tmp)) {
                std::stringstream row_ss(tmp);
                T data_member;
                while (row_ss >> data_member) {
                    tmp_data[i].push_back(data_member);
                }
                i++;
            }
        } catch(const std::istream::failure& e) {
            throw e;
        }
		std::size_t r = tmp_data.size();
		std::size_t c = tmp_data[0].size();
		for (const auto& ro : tmp_data) {
			if (ro.size() != c) {
				throw std::runtime_error{ "Rows can't have different numbers of elements." };
			}
		}
		mat = Matrix<T>(tmp_data);
		return in;
	}

	friend Matrix<T> operator+(const Matrix<T> & a, const Matrix<T> & b) {
		if (a.rows() != b.rows() || a.cols() != b.cols())
			throw std::runtime_error{ "Matrices of different dimensions cannot be added together" };
		Matrix<T> res{ a.rows(), a.cols() };
		for (std::size_t i = 0; i < res.rows(); i++) {
			for (std::size_t j = 0; j < res.cols(); j++)
				res[i][j] = (a[i][j] + b[i][j]);
		}
		return res;
	}

	friend Matrix<T> operator*(const Matrix<T> & a, const Matrix<T> & b) {
		Matrix<T> res = matmul(a, b);
		return res;
	}

	Matrix<T>& operator+=(const Matrix<T>&);
	Matrix<T>& operator*=(const Matrix<T>&);
	Matrix<T>& operator-=(const Matrix<T>&);

	friend bool operator==(const Matrix<T> & a, const Matrix<T> & b) {
		if (a.data.size() != b.data.size() || a.data[0].size() != b.data[0].size()) {
			return false;
		}
		for (std::size_t i = 0; i < a.rows(); i++) {
			for (std::size_t j = 0; j < a.cols(); j++) {
				if (a[i][j] != b[i][j]) {
					return false;
				}
			}
		}
		return true;
	}

	friend bool operator!=(const Matrix<T> & a, const Matrix<T> & b) {
		return !(a == b);
	}

	friend Matrix<T> operator-(const Matrix<T> & a) {
		auto tmp = Matrix<T>{ a.data, a.rows(), a.cols() };
		for (std::size_t i = 0; i < a.rows(); i++) {
			for (std::size_t j = 0; j < a.cols(); j++) {
				tmp[i][j] = -1 * tmp[i][j];
			}
		}
		return tmp;
	}

	friend Matrix<T> operator-(const Matrix<T> & a, const Matrix<T> & b) {
		if (a.size() != b.size()) {
			throw std::runtime_error("Matrices are not of the same size.");
		}
		auto tmp = Matrix<T>{ a };
		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < a.cols(); j++) {
				tmp[i][j] -= b[i][j];
			}
		}
		return tmp;
	}

	friend Matrix<T> operator*(Matrix<T> mat, const T & b) {
		for (std::size_t i = 0; i < mat.rows(); i++) {
			for (std::size_t j = 0; j < mat.cols(); j++) {
				mat[i][j] *= b;
			}
		}
		return mat;
	}

	friend Matrix<T> operator*(const T & b, Matrix<T> mat) {
		return mat * b;
	}

	std::vector<T>& operator[](const std::size_t i) {
		return data.at(i);
	}

	std::vector<T> operator[](const std::size_t i) const {
		return data.at(i);
	}

	static Matrix<T> eye(const std::size_t);
	static Matrix<T> pascal(const std::size_t);
	static Matrix<T> zeros(const std::size_t, const std::size_t);
	static Matrix<T> ones(const std::size_t, const std::size_t);
	static Matrix<T> permutation_matrix(const std::size_t, const std::size_t, const std::size_t);

	static Matrix<T> randn(const std::size_t, const std::size_t);
	static Matrix<T> rand (const std::size_t, const std::size_t);
	static Matrix<T> randi(const std::size_t, const std::size_t, const int, const int);
	static Matrix<T> randi(const std::size_t, const std::size_t, const int);

	void gaussian_elimination(bool mode = false);
	static Matrix<T> inverse(const Matrix<T>&);

	void orthogonalize();
	void orthonormalize();
	static std::pair<Matrix<T>, Matrix<T>> QR(const Matrix<T>&);

	std::size_t zero_rows();
	bool zero_row(const std::size_t);
	bool is_inconsistent();

	~Matrix();
};

#include "Matrix_impl.hpp"
