cmake_minimum_required(VERSION 3.17)
project(sokudo_opencl)

add_library(sokudo_opencl STATIC
        cl_helper.h
        cl_helper.cpp
        blas/level1/asum.h
        blas/level1/asum.cpp
        blas/level1/amax.h
        blas/level1/amax.cpp
        blas/level1/amin.h
        blas/level1/amin.cpp
        blas/level1/axpy.h blas/level1/axpy.cpp)

target_link_libraries(sokudo_opencl OpenCL)
target_compile_definitions(sokudo_opencl PUBLIC SOKUDO_OPENCL CL_HPP_ENABLE_EXCEPTIONS)
target_compile_features(sokudo_opencl PUBLIC cxx_std_20)

