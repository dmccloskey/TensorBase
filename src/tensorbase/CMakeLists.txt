cmake_minimum_required(VERSION 3.0.2 FATAL_ERROR)
project("TensorBase")

#------------------------------------------------------------------------------
# naming conventions:
#
# prefix a variable with 'CF_' if it is used to configure a file!
# e.g., CF_LibTensorBaseExport
set(CF_TENSORBASE_PACKAGE_VERSION "${TENSORBASE_PACKAGE_VERSION_MAJOR}.${TENSORBASE_PACKAGE_VERSION_MINOR}.${TENSORBASE_PACKAGE_VERSION_PATCH}" CACHE INTERNAL "TensorBase VERSION" FORCE)

#------------------------------------------------------------------------------
# En/disable assertions
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	set(CF_TENSORBASE_ASSERTIONS 1)
else()
	set(CF_TENSORBASE_ASSERTIONS 0)
endif()
set(CF_TENSORBASE_ASSERTIONS ${CF_TENSORBASE_ASSERTIONS} CACHE INTERNAL "Enables debug messages (precondition and postconditions are enabled, a bit slower) - this is NOT changing any compiler flags!" FORCE)

#------------------------------------------------------------------------------
# external libs (contrib or system)
#------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/cmake_findExternalLibs.cmake)

#------------------------------------------------------------------------------
# At this point make a summary of where data and doc will be located:
message(STATUS "Info: CF_TENSORBASE_DATA_PATH: ${CF_TENSORBASE_DATA_PATH}")
message(STATUS "Info: CF_TENSORBASE_DOC_PATH: ${CF_TENSORBASE_DOC_PATH}")

#------------------------------------------------------------------------------
# configure config.h
#------------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/configh.cmake)

#------------------------------------------------------------------------------
# big include file for headers and cpp files, that fills the TensorBase_sources variable
include (${PROJECT_SOURCE_DIR}/includes.cmake)

#------------------------------------------------------------------------------
# all the dependency libraries are linked into libTensorBase.so
set(TENSORBASE_DEP_LIBRARIES  ${Boost_LIBRARIES})

# xerces requires linking against CoreFoundation&CoreServices
if(APPLE)
  find_library(CoreFoundation_LIBRARY CoreFoundation )
  find_library(CoreServices_LIBRARY CoreServices )
  set(TENSORBASE_DEP_LIBRARIES ${TENSORBASE_DEP_LIBRARIES}
                           ${CoreFoundation_LIBRARY}
                           ${CoreServices_LIBRARY})
endif()

if (TBB_FOUND)
 list(APPEND TENSORBASE_DEP_LIBRARIES ${TBB_LIBRARIES})
endif()
if (MSVC)
	list(APPEND TENSORBASE_DEP_LIBRARIES opengl32.lib)
endif()

smartpeak_add_library(TARGET_NAME  TensorBase
                   SOURCE_FILES  ${TensorBase_sources}
                   HEADER_FILES  ${TensorBase_sources_h}
                                 ${TensorBase_configured_headers}
                   INTERNAL_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
                                     ${CMAKE_CURRENT_BINARY_DIR}/include
                   EXTERNAL_INCLUDES ${EIGEN3_INCLUDE_DIR}
                                     ${CEREAL_INCLUDE_DIRS}
									 #${CUB_INCLUDE_DIR}
                   LINK_LIBRARIES ${APPLE_EXTRA_LIBS}
                                  ${TENSORBASE_DEP_LIBRARIES}
                   DLL_EXPORT_PATH "TensorBase/")

#------------------------------------------------------------------------------
# since the share basically belongs to TensorBase core we control its installation
# here
# Note: that slash(/) is important here, otherwise the whole directory
#       (not its content) will be copied!
install_directory(${TENSORBASE_HOST_DIRECTORY}/share/TensorBase/ ${INSTALL_SHARE_DIR} share)

#------------------------------------------------------------------------------
# register relevant paths for the doxygen doc generation
smartpeak_doc_path("${PROJECT_SOURCE_DIR}/include")
