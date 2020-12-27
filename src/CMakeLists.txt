cmake_minimum_required(VERSION 3.17)
project(sokudo_lib CUDA)

add_library(sokudo STATIC
        sokudo.h
        kernel.h
        errors.h
        task.h
        kernel_enums.h
        kernels/add_test.h
        kernels/add_test.cpp)

target_compile_features(sokudo PUBLIC cxx_std_20)

if (SOKUDO_OPENCL)
    add_subdirectory(sokudo_opencl)
    target_link_libraries(sokudo sokudo_opencl)
endif ()

if (SOKUDO_CUDA)
    add_subdirectory(sokudo_cuda)
    target_link_libraries(sokudo sokudo_cuda)
endif ()

