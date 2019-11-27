# The Matrix Library:
### Construction:
A Matrix<T> object can be made out of `std::initializer_list<std::initializer_list<T>>`
```c++
Aurilius::Matrix<double> mat {
  {1, 2, 3},
  {4, 5, 6},
  {7, 8, 9}
};

// Random initialization.
auto rand_mat = Aurilius::Matrix<double>::randi(3, 3, 0, 10);
auto rand_2   = Aurilius::Matrix<double>::rand(3, 3);
auto rand_3   = Aurilius::Matrix<double>::randn(3, 3);
```
  ### Special Matrices:
  ```c++
    auto identity = Aurilius::Matrix<double>::eye(4); // generates identity of size (4, 4)
    auto pascal   = Aurilius::Matrix<double>::pascal(5); // generates pascal matrix of size(5, 5)
    auto perm_mat = Aurilius::Matrix<double>::permutation_matrix(5, 2, 3); // generates a permutation matrix of sisze(5, 5)
                                                                 // and exchanges rows 2 and 3.
    auto zeroes_mat = Aurilius::Matrix<double>::zeros(3, 4);

  ```
  ### Operations:
  ```c++
  Aurilius::Matrix<double> I = Matrix<T>::eye(3);
  auto add_result  = mat + I;
  auto mult_result = mat * I; // mult_result == mat.
  auto negation    = -mat;
  std::cout << mat << add_result << mult_result;
  Aurilius::Matrix<double> inp;
  std::cin >> inp; // in form [1 2 3; 4 5 6; 7 8 9;]
  ```

  ### Factorizations:
  ```c++
  Matrix<double> mat {
    {1, 6, 2},
    {5, 2, 1},
    {4, 2, 5}
  }

  // LU decomposition
  auto [L, U] = Aurilius::Matrix<double>::LU(mat);
  std::cout << L << U;

  // QR decomposition.
  auto [Q, R] = Aurilius::Matrix<double>::QR(mat);
  std::cout << Q << R;
  ```

  ### Elimination:
  ```c++
  // Will print out the solutions if the system is solvable, will throw if it's inconsistent.
  mat.gaussian_elimination();
 ```

 ### Solving Systems using inverse.
 ``` c++
 auto inv = Matrix<double>::inverse(mat);
 auto b   = Matrix<double>::col_vector({2, 4, 1});
 auto sol = inv * b;
 std::cout << sol;
 ```
 
 
 # Numerical Integration:
 the library currently only implement `simpson` and the `trapezoial` rules, which are under functions given the same names
 in the `Aurilius` namespace.
 Both of them take a `callable` object, which takes in a `double` and returns a `double`, which will represent the function yo integrate.
 
 ```c++
 auto func = [](const double x) {
    return 1.0 / std::sqrt(1 + x);
 };
 // it takes also the lower and the upper bounds of the integration.
 auto area = Aurilius::simpson(func, 0, 2);
 ```
 
 Both of them also take an optionally-supplied parameter which is the `order`, which is set defaultly as `100`.
