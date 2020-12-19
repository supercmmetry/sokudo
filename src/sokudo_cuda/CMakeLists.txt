cmake_minimum_required(VERSION 3.17)
project(sokudo_cuda CUDA)

add_library(sokudo_cuda STATIC
        cuda_test.h
        cuda_test.cu)

target_link_libraries(sokudo_cuda cuda cudart)

