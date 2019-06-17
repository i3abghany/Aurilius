#ifndef MATRIX_MATRIX_IMPL_HPP
#define MATRIX_MATRIX_IMPL_HPP

#include <assert.h>
#include <algorithm>
#include <random>
#include "Matrix.h"


template <typename T> Matrix<T>::Matrix(size_t ROWS, size_t COLS, T initial) {
    this->data.resize(ROWS);
    for (size_t i = 0; i < this->data.size(); i++) {
        this->data[i].resize(COLS, initial);
    }
}

template <typename T> Matrix<T>::Matrix(T ** data_to_copy, size_t ROWS, size_t COLS) {
    this->data.resize(ROWS);
    for (size_t i = 0; i < this->data.size(); i++) {
        this->data[i].resize(COLS);
    }
    for (size_t i = 0; i < ROWS; i++) {
        for (size_t j = 0; j < COLS; j++) {
            this->data[i][j] = data_to_copy[i][j];
        }
    }
}

template <typename T> Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> data_to_copy) {
    int ROWS = data_to_copy.size();
    int COLS = std::begin(data_to_copy)->size();
    for (auto row : data_to_copy) {
        if (row.size() != COLS) {
            throw std::runtime_error{"Rows can't have different numbers of elements."};
        }
    }
    this->data.resize(ROWS);
    for (size_t i = 0; i < this->data.size(); i++) {
        this->data[i].resize(COLS);
    }
    int i = 0, j = 0;
    for (auto row : data_to_copy) {
        for (auto el : row) {
            this->data[i][j++] = el;
        }
        i++;
        j = 0;
    }
}

template<typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> data_to_copy) {
    int COLS = std::begin(data_to_copy)->size();
    for (auto &row : data_to_copy) {
        if (row.size() != COLS) {
            throw std::runtime_error{"Rows can't have different numbers of elements."};
        }
    }
    this->data = data_to_copy;
}

template<typename T> void Matrix<T>::exchange_cols(const size_t &c1, const size_t &c2) {
    for (size_t i = 0; i < this->rows(); i++) {
        std::swap(this->data[i][c1], this->data[i][c2]);
    }
}

// eexchanges rows r1 and r2.
template <typename T> void Matrix<T>::exchange_rows(const size_t &r1, const size_t &r2) {
    std::swap(this->data[r1], this->data[r2]);
}

// Matrix multiplication.
template <typename T> Matrix<T> Matrix<T>::matmul(const Matrix<T> & first, const Matrix<T> & second) {

    if (first.cols() != second.rows())
        throw std::runtime_error{"Size doesn't match for matrix multiplication."};
    Matrix<T> result{first.rows(), second.cols()};
    for (size_t i = 0; i < first.rows(); i++) {
        for (size_t j = 0; j < second.cols(); j++) {
            for (size_t k = 0; k < first.cols(); k++) {
                result[i][j] += first[i][k] * second[k][j];
            }
        }
    }
    return result;
}

// Transposes a matrix and returns the transposed copy.
template <typename T> Matrix<T> Matrix<T>::transpose(const Matrix<T>& mat) {
    Matrix<T> result{mat.cols(), mat.rows()};
    for (size_t i = 0; i < result.rows(); i++) {
        for (size_t j = 0; j < result.cols(); j++) {
            result[i][j] = mat[j][i];
        }
    }
    return result;
}

// returns an identity matrix of size NxN
template <typename T> Matrix<T> Matrix<T>::eye(const size_t &N) {
    Matrix<T> result = {N, N};
    for (size_t i = 0; i < N; i++) {
        result[i][i] = 1;
    }
    return result;
}

template <typename T> Matrix<T> Matrix<T>::pascal(const size_t &N) {
    Matrix<T> result{N, N, T{0}};
    std::vector<T> &first_row = *std::begin(result.data);
    for (size_t i = 0; i < N; i++) {
        result[i][0] = 1;
    }
    for (size_t i = 1; i < N; i++) {
        for (size_t j = 1; j < N; j++) {
            result.data[i][j] = result.data[i - 1][j] + result.data[i - 1][j - 1];
        }
    }
    return result;
}

// returns the number of zero rows in the end of the matrix.
template <typename T> size_t Matrix<T>::zero_rows() {
    size_t rank = this->rows();
    for (size_t i = 0; i < this->rows(); i++) {
        bool is_free_row = zero_row(i);
        if (is_free_row) {
            rank--;
        }
    }
    return this->rows() - rank;
}

// only applied after elimination, checks if the last non-zero row has all zeros but b (in Ax=b) is not zero.
template <typename T> bool Matrix<T>::is_inconsistent() {
    const size_t free_rows = this->zero_rows();
    const size_t last_row = this->rows() - free_rows - 1;
    bool is_all_zero_but_last = true;
    for (size_t elem = 0; elem < this->cols() - 1; elem++) {
        is_all_zero_but_last &= ((*this)[last_row][elem] == 0);
    }
    return is_all_zero_but_last && ((*this)[last_row][this->cols() - 1] != 0);
}

template <typename T> void Matrix<T>::gaussian_elimination(bool mode) {
    std::vector<size_t> piv_idxs;
    bool free_col = false;
    for (size_t row = 0; row < this->rows(); row++) {
        T pivot = (*this)[row][row];
        int piv_idx = row;
        if (free_col) {
            for (size_t c = row; c < this->rows(); c++) {
                if ((*this)[row][c] != 0) {
                    pivot = (*this)[row][c];
                    piv_idx = c;
                    break;
                }
            }
        }
        if (pivot != 0) {
            if (pivot != 1) {
                for (size_t elem = 0; elem < this->cols(); elem++)
                    (*this)[row][elem] /= pivot;
            }
            piv_idxs.push_back(piv_idx);
            for (size_t r = 0; r < this->rows(); r++) {
                if (r == row)
                    continue;
                T multiplier = (*this)[r][piv_idx/* row */];
                for (size_t elem = 0; elem < this->cols(); elem++) {
                    (*this)[r][elem] -= multiplier * (*this)[row][elem];
                    if(fabs((*this)[r][elem]) < EPS) {
                        (*this)[r][elem] = T{0};
                    }
                }
                if (!mode) {
                    std::cout << *(this) << "\n\n\n";
                }
            }
        }

            // if a pivot is equal to 0, free column.
        else /* if(pivot == 0) */ {
            bool found_piv = false;
            for (size_t successor_row = row + 1; successor_row < this->rows(); successor_row++) {
                if ((*this)[successor_row][row] != 0) {
                    found_piv = true;
                    exchange_rows(successor_row, row);
                    break;
                }
            }

            bool finish = true;
            for (size_t fr = row; fr < this->rows(); fr++) {
                finish &= zero_row(fr);
            }
            if (!finish) row--;
            free_col = !found_piv;
        }
    } // END OF ELIMINATION.
    for(size_t i = 0; i < this->rows(); i++) {
        for (int j = 0; j < this->cols(); j++) {
            if (fabs(this->data[i][j]) <= EPS) {
                this->data[i][j] = T{0};
            }
        }
    }
    if (!mode) {
        print_solutions(piv_idxs);
    }
}


template<typename T>
void Matrix<T>::print_solutions(const std::vector<size_t> &piv_idxs) {
    const size_t num_of_zero_rows = this->zero_rows();
    if (is_inconsistent()) {
        throw std::runtime_error("The system is inconsistent.");
    }
    std::vector<std::string> solutions;
    std::cout << std::endl;
    for (size_t row = 0; row < this->rows() && !zero_row(row); row++) {
        std::string sol = ("x" + std::to_string(piv_idxs[row] + 1) + " = ");
        for (size_t i = piv_idxs[row] + 1; i < this->cols() - 1; i++) {
            if (fabs((*this)[row][i]) <= EPS) {
                (*this)[row][i] = 0;
                continue;
            }

            sol += ((*this)[row][i] > 0 ? "-" : "") + std::to_string((*this)[row][i]) + "x" +
                   std::to_string(i + 1) + " ";
        }
        sol += ((*this)[row][this->cols() - 1] > 0 ? "+ " : "");
        if((*this)[row][this->cols() - 1] != 0) {
            sol += std::to_string((*this)[row][this->cols() - 1]);
        }
        solutions.push_back(sol);
    }

    for (const auto &s : solutions)
        std::cout << s << std::endl;
}

template <typename T> std::pair<Matrix<T>, size_t> Matrix<T>::upper() {
    size_t no_of_exchanges { 0 };
    Matrix<T> L{eye(this->rows())};
    Matrix<T> tmp_L{eye(this->rows())};
    bool free_col = false;
    for (size_t row = 0; row < this->rows(); row++) {
        T pivot = (*this)[row][row];
        int piv_idx = row;
        if (free_col) {
            for (size_t c = row; c < this->rows(); c++) {
                if ((*this)[row][c] != 0) {
                    pivot = (*this)[row][c];
                    piv_idx = c;
                    break;
                }
            }
        }
        if (pivot != 0) {
            for (size_t successor_row = row + 1; successor_row < this->rows(); successor_row++) {
                T multiplier = (*this)[successor_row][piv_idx] / pivot;

                for (size_t elem = 0; elem < this->cols(); elem++) {
                    (*this)[successor_row][elem] -= multiplier * (*this)[row][elem];
                    if (elem == row) {
                        tmp_L = eye(this->rows());
                        tmp_L[successor_row][elem] = -1 * multiplier;
                        L = tmp_L * L;
                    }
                }
            }
        } else {
            bool found_piv = false;
            for (size_t successor_row = row + 1; successor_row < this->rows(); successor_row++) {
                if ((*this)[successor_row][row] != 0) {
                    found_piv = true;
                    exchange_rows(successor_row, row);
                    L = inverse(permutation_matrix(this->rows(), successor_row, row)) * L;
                    no_of_exchanges++;
                    break;
                }
            }
            bool finish = true;
            for (size_t fr = row; fr < this->rows(); fr++) {
                finish &= zero_row(fr);
            }
            if (!finish) row--;
            free_col = !found_piv;
        }
    }
    return { L, no_of_exchanges };
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T> &m) {
    *this = (*this) + m;
    return (*this);
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T> &m) {
    *this = (*this) * m;
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &m) {
    *this = (*this) - m;
    return *this;
}

/* LU decomposition.
 * Returns the [L, U] decomposition for a matrix m
 * Multiplies by P^-1 in both sides of PA = LU if
 * the system needs a row exchange operation.
 * A = P^(-1)*LU
 */
template<typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::LU(const Matrix &m) {
    Matrix<T> mat = m;
    Matrix<T> L = inverse(mat.upper().first), U = mat;
    return {L, U};
}

// Returns the upper-triangular copy matrix.
template<typename T>
Matrix<T> Matrix<T>::upper(Matrix<T> m) {
    m.upper();
    return m;
}

// Returns whether the matrix is symmetric or not.
template<typename T>
bool Matrix<T>::is_symmetric(const Matrix &m) {
    if (m.rows() != m.cols()) {
        return false;
    }
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (m[i][j] != m[j][i]) {
                return false;
            }
        }
    }
    return true;
}

// Returns a permutation matrix that exchanges first_row and second_row.
template<typename T>
Matrix<T> Matrix<T>::permutation_matrix(const size_t &size, const size_t &first_row, const size_t &second_row) {
    auto P = Matrix<T>::eye(size);
    P.exchange_rows(first_row, second_row);
    return P;
}

// Inverses a matrix or throws if non square or singular matrix.
template<typename T>
Matrix<T> Matrix<T>::inverse(const Matrix<T> &A) {
    if (A.rows() != A.cols()) {
        throw std::runtime_error{"Non-square Matrix."};
    }
    Matrix<T> AI = A;
    const size_t dim = A.rows();
    Matrix<T> I = Matrix<T>::eye(dim);
    size_t i = 0;
    for (auto &R : AI.data) {
        R.insert(std::end(R),
                 std::make_move_iterator(std::begin(I[i])),
                 std::make_move_iterator(std::end(I[i])));
        ++i;
    }
    I = Matrix<T>{dim, dim}, i = 0;
    AI.gaussian_elimination(true);
    for (auto &R : AI.data) {
        I[i++] = {std::make_move_iterator(std::begin(R)), std::make_move_iterator(std::begin(R) + dim)};
        R.erase(std::begin(R), std::begin(R) + dim);
    }
    if (I != eye(dim)) {
        throw std::runtime_error{"Singular."};
    }
    return AI;
}

// Returns whether the row is all zeroes or not.
template<typename T>
bool Matrix<T>::zero_row(const size_t &i) {
    bool is_free_row = true;
    for (size_t j = 0; j < this->cols(); j++) {
        is_free_row &= ((*this)[i][j] == 0);
    }
    return is_free_row;
}

// Fills the matrix with the specified value.
template<typename T>
void Matrix<T>::fill(const T& val) {
    for (auto &R : this->data) {
        std::fill(std::begin(R), std::end(R), val);
    }
}

// Concatenates a vector as a row.
template<typename T>
void Matrix<T>::add_row(const std::vector<T> &r) {
    if(r.size() != this->cols()) {
        throw std::runtime_error("Row size not consistent with the matrix size.");
    }
    this->data.push_back(r);
}

// Concatenates a vector as a column.
template<typename T>
void Matrix<T>::add_col(const std::vector<T> &c) {
    size_t i = 0;
    if(c.size() != this->rows()) {
        throw std::runtime_error("Column size not consistent with the matrix size.");
    }
    for(auto &R : this->data) {
        R.push_back(c[i++]);
    }
}

// Returns a matrix of size {r, c} with zero elements.
template<typename T>
Matrix<T> Matrix<T>::zeros(const size_t &r, const size_t &c) {
    return Matrix<T> {r, c, T{0}};
}

// Concatenate the param. matrix' rows.
template<typename T>
void Matrix<T>::tuck_rows(const Matrix<T> &m) {
    if(m.cols() != this->cols()) {
        throw std::runtime_error("Invalid size of tucked matrix.");
    }
    for(auto &R : m.data) {
        this->add_row(R);
    }
}

// Concatenate the param. matrix' columns.
template<typename T>
void Matrix<T>::tuck_cols(const Matrix<T> &m) {
    for(size_t i = 0; i < m.cols(); i++) {
        std::vector<T> col(m.rows());
        for(size_t j = 0; j < m.rows(); j++) {
            col[j] = (m[j][i]);
        }
        this->add_col(col);
    }
}


template<typename T>
Matrix<T> Matrix<T>::get_col(const size_t &col) {
    Matrix<T> res{this->rows(), 1};
    for(size_t row = 0; row < this->rows(); row++) {
        res[row][0] = this->data[row][col];
    }
    return res;
}

template<typename T>
void Matrix<T>::insert_col(const Matrix<T> &c, const size_t &idx) {
    for(size_t i = 0; i < this->rows(); i++) {
        this->data[i].insert(std::begin(this->data[i]) + idx, c[i][0]);
    }
}

template<typename T>
void Matrix<T>::remove_col(const size_t &r) {
    for(size_t i = 0; i < this->rows(); i++) {
        this->data[i].erase(std::begin(this->data[i]) + r);
    }
}


/* returns a matrix of size {r, c} with elements
 * uniformly distributed random numbers
 * whose values lies between 0 and 1.
 * Since it is uniformly distributed,
 * therefore the mean value is 0.5.
 */
template<typename T>
Matrix<T> Matrix<T>::rand(const size_t &r, const size_t &c) {
    Matrix<T> result {r, c};
    static_assert(std::is_floating_point<T>::value,
                  "template must be a floating point type");
    const T range_from  = T{0.0};
    const T range_to    = T{1.0};
    std::random_device                  rand_dev;
    std::mt19937                        generator(rand_dev());
    std::uniform_real_distribution<T>   distr(range_from, range_to);
    for(size_t i = 0; i < r; i++) {
        for(size_t j = 0; j < c; j++) {
            result[i][j] = distr(generator);
        }
    }
    return result;
}

/* returns a matrix with size {r, c} with elements that are
 * normally (Gaussian) distributed random numbers whose
 * values theoretically lies between -Infinity to Infinity
 * having 0 mean and 1 variance.
 */
template<typename T>
Matrix<T> Matrix<T>::randn(const size_t &r, const size_t &c) {
    static_assert(std::is_floating_point<T>::value,
                  "template must be a floating point type");
    Matrix<T> result {r, c};
    std::random_device rand_dev;
    std::mt19937_64 generator(rand_dev());
    const T mean      = T{0.0};
    const T variance  = T{1.0};
    std::normal_distribution<T> distr(mean, variance);
    for(size_t i = 0; i < r; i++) {
        for(size_t j = 0; j < c; j++) {
            result[i][j] = distr(generator);
        }
    }
    return result;
}

/* Returns a matrix of size {r, c}, its elements are
 * uniformly distributed pseudorandom.
 */
template<typename T>
Matrix<T> Matrix<T>::randi(const size_t &r, const size_t &c, const int &imin, const int &imax) {
    Matrix<T> result {r, c};
    std::random_device rand_dev;
    std::mt19937_64 generator(rand_dev());
    std::uniform_int_distribution<int> distr(imin, imax);
    for(size_t i = 0; i < r; i++) {
        for(size_t j = 0; j < c; j++) {
            result[i][j] = static_cast<T>(distr(generator));
        }
    }
    return result;
}

// Returns the dot product of two vectors a and b.
template<typename T>
T Matrix<T>::dot_prod(const Matrix<T> &a, const Matrix<T> &b) {
    if((!a.is_col() && !a.is_row()) || (!b.is_col() && !b.is_row())) {
        std::cout << a.rows() << ' ' << a.cols() << ' ' << b.rows() << ' ' << b.cols();
        throw std::runtime_error("Dot product can only be done on vectors.");
    }
    T res = T{0};
    Matrix<T> vecA = a, vecB = b;
    if(!a.is_row()) {
        vecA = transpose(a);
    }
    if(!b.is_row()) {
        vecB = transpose(b);
    }
    for(size_t i = 0; i < vecA.cols(); i++) {
        res += (vecA[0][i] * vecB[0][i]);
    }
    return res;
}

// Projects a vector a in the direction of another vector b.
template<typename T>
Matrix<T> Matrix<T>::project(const Matrix<T> &a, const Matrix<T> &b) {
    if((a.cols() != 1 && a.rows() != 1) || (b.cols() != 1 && b.rows() != 1)) {
        throw std::runtime_error("Dot product can only be done on vectors.");
    }

    Matrix<T> vecA = a, vecB = b;
    if(b.rows() != 1) {
        vecB = transpose(b);
    }
    if(a.rows() != 1) {
        vecA = transpose(a);
    }
    auto prod = Matrix<T>::dot_prod(vecA, vecB);
    auto bTb  = Matrix<T>::dot_prod(vecB, vecB);

    Matrix<T> result(1, vecB.cols());
    for(size_t i = 0; i < vecB.cols(); i++) {
        result[0][i] = vecB[0][i] * (prod / bTb);
    }
    return transpose(result);
}

// Projects a vector b to the column space of a matrix A.
template<typename T>
Matrix<T> Matrix<T>::project_into_col_space(const Matrix<T> &A, const Matrix<T> &b) {
    auto P = inverse(transpose(A) * A);
    P = A * P;
    P = P * transpose(A);
    if(!b.is_col()) {
        throw std::runtime_error("Projection of non column vector.");
    }
    auto res = P * b;
//    std::cout << res;
    return res;
}

template <typename T> bool Matrix<T>::is_col() const {
    return (this->cols() == 1);
}

template <typename T> bool Matrix<T>::is_row() const {
    return (this->rows() == 1);
}

// Returns a matrix that consists of one column
// made out of the elements of vector v.
template<typename T>
Matrix<T> Matrix<T>::col_matrix(const std::vector<T> &v) {
    auto res = Matrix<T> {v.size(), 1};
    for(size_t i = 0; i < v.size(); i++) {
        res[i][0] = v[i];
    }
    return res;
}

// Returns a matrix that consists of one row
// made out of the elements of vector v.
template<typename T>
Matrix<T> Matrix<T>::row_matrix(const std::vector<T> &v) {
    return Matrix<T> {std::vector<std::vector<T>>(1, v)};
}

template<typename T>
void Matrix<T>::normalize_col(const size_t &c) {
    T mag = std::sqrt(dot_prod(get_col(c), get_col(c)));
    for(int i = 0; i < this->rows(); i++) {
        this->data[i][c] /= mag;
    }
}

template<typename T>
void Matrix<T>::orthogonalize() {
    for(size_t j = 1; j < this->cols(); j++) {
        auto col = this->get_col(j);
        this->remove_col(j);
        for(size_t prev_col = 0; prev_col < j; prev_col++) {
            col -= project(col, get_col(prev_col));
        }
        this->insert_col(col, j);
    }
}


template<typename T>
void Matrix<T>::orthonormalize() {
    this->orthogonalize();
    for(int j = 0; j < this->cols(); j++) {
        this->normalize_col(j);
    }
}

template<typename T>
Matrix<T>::~Matrix() {
    this->data.clear();
}

template<typename T>
T Matrix<T>::det(Matrix<T> m) {
    if(m.cols() != m.rows()) {
        throw std::runtime_error("Determinants is only for square matrices.");
    }
    auto p = m.upper();
    size_t no_of_exchanges = p.second;
    T res = 1;
    for(int i = 0; i < m.rows(); i++) {
        res *= m[i][i];
    }
    return res * ((no_of_exchanges % 2 != 0) ? -1 : 1);
}

#endif //MATRIX_MATRIX_IMPL_HPP
