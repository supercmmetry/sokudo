cmake_minimum_required(VERSION 3.17)
project(sokudo)

set(CMAKE_CXX_STANDARD 20)

include_directories(src)
add_subdirectory(src)

enable_testing()
add_subdirectory(tests)