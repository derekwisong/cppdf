cmake_minimum_required(VERSION 3.10)
project(DataFrame LANGUAGES CXX)

include(CTest)
enable_testing()

add_subdirectory(vendor)
add_subdirectory(src)