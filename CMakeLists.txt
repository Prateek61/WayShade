cmake_minimum_required(VERSION 3.22)
project(WayShade LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Halide REQUIRED)

add_subdirectory(halide)
add_subdirectory(examples/fx_cli)