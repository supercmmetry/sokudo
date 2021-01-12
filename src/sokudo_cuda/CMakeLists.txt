cmake_minimum_required(VERSION 3.17)
project(sokudo_cuda CUDA)

add_library(sokudo_cuda STATIC
        cuda_test.cu
        cuda_helper.h
        cuda_helper.cu
        cuda_test.h blas/level1/add.h)

set(CMAKE_CUDA_STANDARD 14)

target_link_libraries(sokudo_cuda cuda cudart)
target_compile_definitions(sokudo_cuda PUBLIC SOKUDO_CUDA)
