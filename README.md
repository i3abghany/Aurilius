### Construction:
A Matrix<T> object can be made out of `std::initializer_list<std::initializer_list<T>>`
```c++
Matrix<double> mat {
  {1, 2, 3},
  {4, 5, 6},
  {7, 8, 9}
};
```
  ### Special Matrices:
  ```c++
    auto identity = Matrix<double>::eye(4); // generates identity of size (4, 4)
    auto pascal   = Matrix<double>::pascal(5); // generates pascal matrix of size(5, 5)
    auto perm_mat = Matrix<double>::permutation_matrix(5, 2, 3); // generates a permutation matrix of sisze(5, 5)
                                                                 // and exchanges rows 2 and 3.
  ```
  ### Operations:
  ```c++
  Matrix<double> I = Matrix<T>::eye(3);
  auto add_result  = mat + I;
  auto mult_result = mat * I; // mult_result == mat.
  auto negation    = -mat;
  std::cout << mat << add_result << mult_result;
  Matrix<double> inp;
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
  auto [L, U] = Matrix<double>::LU(mat);
  std::cout << L << U;

  // QR decomposition.
  auto [Q, R] = Matrix<double>::QR(mat);
  std::cout << Q << R;
  ```

  ### Elimination:
  ```c++
  // Will print out the solutions if the system is solvable, will throw if it's inconsistent.
  mat.gaussian_elimination();
 ```
