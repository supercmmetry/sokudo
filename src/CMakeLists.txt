cmake_minimum_required(VERSION 3.17)
project(sokudo_lib CUDA)

add_library(sokudo STATIC
        sokudo.h
        sokudo.cpp sokudo_opencl/cl_test.cpp kernel.h errors.h)

target_compile_features(sokudo PUBLIC cxx_std_20)

if (SOKUDO_OPENCL)
    add_subdirectory(sokudo_opencl)
    target_link_libraries(sokudo sokudo_opencl)
endif ()

if (SOKUDO_CUDA)
    add_subdirectory(sokudo_cuda)
    target_link_libraries(sokudo sokudo_cuda)
endif ()

