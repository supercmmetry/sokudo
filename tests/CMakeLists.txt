cmake_minimum_required(VERSION 3.17)
project(sokudo_test)

add_subdirectory(googletest)
include_directories(googletest/googletest/include)
add_executable(sokudo_test debug.cpp blas_level1.cpp)

target_link_libraries(sokudo_test PUBLIC gtest gtest_main sokudo)