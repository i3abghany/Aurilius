project(Aurilius)

set(CMAKE_CXX_STANDARD 17)

add_executable(pin src/main.cpp src/Aurilius.h src/Matrix.h src/Matrix_impl.hpp src/Numerical_integration.hpp)
