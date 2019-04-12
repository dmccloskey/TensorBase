### example CMakeLists.txt to develop programs using TensorBase
project("TensorBase_ExternalCodeTest")
cmake_minimum_required(VERSION 3.0.2)

## list all your executables here (a corresponding .cpp file should exist, e.g. TestExternalCode.cpp)
set(my_executables
	TestExternalCode
)

## list all classes here, which are required by your executables
## (all these classes will be linked into a library)
set(my_sources
	ExampleLibraryFile.cpp
)

## find TensorBase package and register target "TensorBase" (our library)
## Note: This is customized to fit the nightly test scenario. In a
##       regular build find_package(TensorBase) should be sufficient.
find_package(TensorBase PATHS "$ENV{TENSORBASE_BUILD_TREE}" NO_CMAKE_PACKAGE_REGISTRY)

# check whether the TensorBase package was found
if (TensorBase_FOUND)

  ## include directories for TensorBase headers (and contrib)
  include_directories(${TensorBase_INCLUDE_DIRECTORIES})

  ## append precompiler macros specific to TensorBase
  ## Warning: this could be harmful to your project. Check this
  ## if problems occur
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TENSORBASE_ADDCXX_FLAGS}")
  add_definitions(${TENSORBASE_DEFINITIONS})

  ## library with additional classes from above
  add_library(my_custom_lib STATIC ${my_sources})
  target_link_libraries(my_custom_lib TensorBase)

  ## add targets for the executables
  foreach(i ${my_executables})
    # create the executable
    add_executable(${i} ${i}.cpp)
    ## link executables against TensorBase
	target_link_libraries(${i} TensorBase my_custom_lib)
  endforeach(i)

else(TensorBase_FOUND)
  message(FATAL_ERROR "TensorBaseConfig.cmake file not found!")
endif(TensorBase_FOUND)

## Enable testing - for Nightly Build log
include(Dart)
add_test(TestExternalCode TestExternalCode)
