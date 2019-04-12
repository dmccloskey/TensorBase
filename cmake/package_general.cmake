
# --------------------------------------------------------------------------
# general definitions used for building TensorBase packages
set(CPACK_PACKAGE_NAME "TensorBase")
set(CPACK_PACKAGE_VENDOR "TensorBase.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "TensorBase - A framework for mass spectrometry")
set(CPACK_PACKAGE_VERSION "${TENSORBASE_PACKAGE_VERSION_MAJOR}.${TENSORBASE_PACKAGE_VERSION_MINOR}.${TENSORBASE_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION_MAJOR "${TENSORBASE_PACKAGE_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${TENSORBASE_PACKAGE_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${TENSORBASE_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "TensorBase-${CPACK_PACKAGE_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_FILE ${PROJECT_SOURCE_DIR}/cmake/TensorBasePackageDescriptionFile.cmake)
set(CPACK_RESOURCE_FILE_LICENSE ${PROJECT_SOURCE_DIR}/License.txt)
set(CPACK_RESOURCE_FILE_WELCOME ${PROJECT_SOURCE_DIR}/cmake/TensorBasePackageResourceWelcomeFile.txt)
set(CPACK_RESOURCE_FILE_README ${PROJECT_SOURCE_DIR}/cmake/TensorBasePackageResourceReadme.txt)

########################################################### Fixing dynamic dependencies
# Done on Windows via copying external and internal dlls to the install/bin/ folder
# Done on Mac via fixup_bundle for the GUI apps (TOPPView, TOPPAS) and via fix_mac_dependencies for the TOPP tools
# which recursively gathers dylds, copies them to install/lib/ and sets the install_name of the binaries to @executable_path/../lib
# Not done on Linux. Either install systemwide (omit CMAKE_INSTALL_PREFIX or set it to /usr/) or install and add the
# install/lib/ folder to the LD_LIBRARY_PATH

#install(CODE "
#  include(BundleUtilities)
#  GET_BUNDLE_ALL_EXECUTABLES(\${CMAKE_INSTALL_PREFIX}/${INSTALL_BIN_DIR} EXECS)
#  fixup_bundle(\"${EXECS}\" \"\" \"\${CMAKE_INSTALL_PREFIX}/${INSTALL_LIB_DIR}\")
#  " COMPONENT applications)