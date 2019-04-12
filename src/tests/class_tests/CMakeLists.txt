

cmake_minimum_required(VERSION 3.0.2 FATAL_ERROR)
project("TensorBase_class_tests")

#------------------------------------------------------------------------------
# set new CMAKE_RUNTIME_OUTPUT_DIRECTORY for tests and remember old setting
set(_TMP_CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

#------------------------------------------------------------------------------
# add TensorBase specific tests
add_subdirectory(tensorbase)

#------------------------------------------------------------------------------
# restore old CMAKE_RUNTIME_OUTPUT_DIRECTORY
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${_TMP_CMAKE_RUNTIME_OUTPUT_DIRECTORY})
