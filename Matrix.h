#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H

#include<iostream>
#include <vector>
#include <iomanip>
#include <math.h>
#include <limits>
#include <algorithm>
#include <sstream>
#include <type_traits>

const double EPS = std::numeric_limits<double>::epsilon() * 100.0;

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &p) {
    out << "{";
    for (size_t j = 0; j < p.size(); j++) {
        out << std::fixed << std::setprecision(3) << std::setw(6) << p[j];
        if (j != p.size() - 1) {
            out << ' ';
        }
    }
    return out << "}" << std::endl;
}

template<typename T> class Matrix {
    static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
                  "result_type must be a floating point or integral type");
    std::vector<std::vector<T>> data;
    void print_solutions(const std::vector<size_t> &);
public:
    Matrix() = default;
    Matrix(size_t, size_t, T = T{}); // initialized dimensions.
    Matrix(T**, size_t, size_t);
    Matrix(std::initializer_list<std::initializer_list<T>>);
    explicit Matrix(std::vector<std::vector<T>>);

    Matrix(const Matrix<T> &) = default;
    Matrix(Matrix<T> &&) noexcept = default;

    Matrix<T>& operator=(const Matrix<T> &) = default;
    Matrix<T>& operator=(Matrix<T> &&) noexcept = default;

    size_t cols() const {return this->data[0].size();}
    size_t rows() const {return this->data.size();}
    std::pair<size_t, size_t> size() {return {cols(), rows()};}

    void add_row(const std::vector<T> &);
    void add_col(const std::vector<T> &);

    void tuck_rows(const Matrix<T> &);
    void tuck_cols(const Matrix<T> &);

    void exchange_rows(const size_t &, const size_t &);
    void exchange_cols(const size_t &, const size_t &);

    static T dot_prod(const Matrix<T> &, const Matrix<T> &);
    static Matrix<T> project(const Matrix<T> &, const Matrix<T> &);
    static Matrix<T> project_into_col_space(const Matrix<T> &, const Matrix<T> &);

    static Matrix<T> col_matrix(const std::vector<T> &);
    static Matrix<T> row_matrix(const std::vector<T> &);

    bool is_row() const;
    bool is_col() const;

    void fill(const T&);

    static Matrix<T> matmul(const Matrix<T> &, const Matrix<T> &);
    static Matrix<T> transpose(const Matrix<T> &);
    static Matrix<T> upper(Matrix<T>);
    static std::pair<Matrix<T>, Matrix<T>> LU(const Matrix &);
    static bool is_symmetric(const Matrix &);

    friend std::ostream& operator<<(std::ostream &out, const Matrix<T> &mat) {
        for (size_t i = 0; i < mat.rows(); i++) {
//            out << "{";
//            for (size_t j = 0; j < mat.cols(); j++) {
//                if (fabs(mat[i][j] - 0) < EPS)
//                    out << std::fixed << std::setprecision(3) << std::setw(6) << (mat[i][j] = 0);
//                else out << std::fixed << std::setprecision(3) << std::setw(6) << mat[i][j];
//                if (j != mat.cols() - 1)
//                    out << ' ';
//            }
//            out << "}";
//            if (i != mat.rows() - 1)
//                out << ',' << std::endl;
            out << mat[i];
        }
        out << std::endl << std::endl;
        return out;
    }

    friend Matrix<T> operator + (const Matrix<T>& a, const Matrix<T>& b) {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error{"Matrices of different dimensions cannot be added together"};
        Matrix<T> res{a.rows(), a.cols()};
        for (size_t i = 0; i < res.rows(); i++) {
            for (size_t j = 0; j < res.cols(); j++)
                res[i][j] = (a[i][j] + b[i][j]);
        }
        return res;
    }

    friend Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b) {
        Matrix<T> res = matmul(a, b);
        return res;
    }

    Matrix<T>& operator+=(const Matrix<T>&);
    Matrix<T>& operator*=(const Matrix<T>&);

    friend bool operator==(const Matrix<T> &a, const Matrix<T> &b) {
        if (a.data.size() != b.data.size() || a.data[0].size() != b.data[0].size()) {
            return false;
        }
        for (size_t i = 0; i < a.rows(); i++) {
            for (size_t j = 0; j < a.cols(); j++) {
                if (a[i][j] != b[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    friend bool operator != (const Matrix<T> &a, const Matrix<T> &b) {
        return !(a == b);
    }
    
        
    friend Matrix<T> operator - (const Matrix<T>& a) {
        auto tmp = Matrix<T>{a.data, a.rows(), a.cols()};
        for (size_t i = 0; i < a.rows(); i++) {
            for (size_t j = 0; j < a.cols(); j++) {
                tmp[i][j] = -1 * tmp[i][j];
            }
        }
        return tmp;
    }

    friend Matrix<T> operator * (Matrix<T> mat, const T &b) {
        for (size_t i = 0; i < mat.rows(); i++) {
            for (size_t j = 0; j < mat.cols(); j++) {
                mat[i][j] *= b;
            }
        }
        return mat;
    }

    friend Matrix<T> operator * (const T &b, Matrix<T> mat) {
        return mat * b;
    }

    // rows separated by semicolon.
    friend std::istream& operator>>(std::istream &in, Matrix<T> &mat) {
        // [1 2 3; 4 5 6; 7 8 9;]
        std::string tmp;
        std::string raw_form;
        while (getline(in, tmp) && tmp.find(']') == std::string::npos)
            raw_form += tmp;
        raw_form += tmp;
        raw_form.erase(std::begin(raw_form));
        raw_form.erase(std::end(raw_form) - 1);

        size_t num_rows = std::count_if(std::begin(raw_form), std::end(raw_form), [](int c) { return c == ';'; });
        std::replace_if(std::begin(raw_form), std::end(raw_form), [](int c) { return c == ';'; }, '\n');
        std::stringstream ss(raw_form);
        std::vector<std::vector<T>> tmp_data(num_rows, std::vector<T>());
        int i = 0;
        while (getline(ss, tmp)) {
            std::stringstream row_ss(tmp);
            T data_member;
            while (row_ss >> data_member) {
                tmp_data[i].push_back(data_member);
            }
            i++;
        }
        size_t r = tmp_data.size();
        size_t c = tmp_data[0].size();
        for (auto ro : tmp_data) {
            if (ro.size() != c) {
                throw std::runtime_error{"Rows can't have different numbers of elements."};
            }
        }
        mat = Matrix<T>(tmp_data);
        return in;
    }
    
    std::vector<T> &operator[](const size_t &i) {
        return data.at(i);
    }

    std::vector<T> operator[](const size_t &i) const {
        return data.at(i);
    }

    static Matrix<T> eye(const size_t &N);
    static Matrix<T> pascal(const size_t &N);
    static Matrix<T> zeros(const size_t &rows, const size_t &cols);
    static Matrix<T> permutation_matrix(const size_t &size, const size_t &, const size_t &);
    static Matrix<T> inverse(const Matrix<T> &);
    static Matrix<T> randn(const size_t &r, const size_t &c);
    static Matrix<T> rand(const size_t &, const size_t &);
    static Matrix<T> randi(const size_t &, const size_t &, const int&, const int&);
    static Matrix<T> randi(const size_t &, const size_t &, const int&);

    void gaussian_elimination(bool mode = false);
    size_t zero_rows();
    bool zero_row(const size_t &);
    bool is_inconsistent();
    
    Matrix<T> upper();

    ~Matrix();
};

#endif // MATRIX_MATRIX_H
#include "Matrix_impl.hpp"
