project(Aurilius)

set(CMAKE_CXX_STANDARD 17)
SET(GCC_COVERAGE_COMPILE_FLAGS "-fopenmp")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
add_executable(pin src/main.cpp src/Aurilius.h src/Matrix.h src/Matrix_impl.hpp src/Numerical_integration.hpp src/Vector/Vector2d.cpp src/Vector/Vector2d.h src/Vector/Vector3d.cpp src/Vector/Vector3d.h)
