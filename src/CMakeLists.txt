cmake_minimum_required(VERSION 3.17)
project(sokudo_lib)

add_library(sokudo STATIC
        sokudo.h sokudo.cpp)

target_compile_features(sokudo PUBLIC cxx_std_20)

if (SOKUDO_OPENCL)
    target_link_libraries(sokudo OpenCl)
    add_compile_definitions(sokudo ENABLE_OPENCL)
endif ()

