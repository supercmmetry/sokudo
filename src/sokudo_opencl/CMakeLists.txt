cmake_minimum_required(VERSION 3.17)
project(sokudo_opencl)

add_library(sokudo_opencl STATIC
        cl_test.h
        cl_test.cpp
        cl_helper.h
        cl_helper.cpp)

target_link_libraries(sokudo_opencl OpenCL)
target_compile_definitions(sokudo_opencl PUBLIC SOKUDO_OPENCL CL_HPP_ENABLE_EXCEPTIONS)

