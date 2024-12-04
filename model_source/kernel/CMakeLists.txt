cmake_minimum_required(VERSION 3.6)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(BasicUnitNMS)
add_subdirectory(pybind11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-O3)
pybind11_add_module(BasicUnitNMS BasicUnitNMS.cpp)
