if (CMAKE_HOST_WIN32)
    # Requires version 3.20 for baseline support of icx, icx-cl
    cmake_minimum_required(VERSION 3.20)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)
    find_package(oneDPLWindowsIntelLLVM)
else()
    # Requires version 3.11 for use of icpc with c++17 requirement
    cmake_minimum_required(VERSION 3.11)
endif()

project (maxloc_reductions LANGUAGES CXX)

set(ONEDPL_BACKEND dpcpp)
# Add oneDPL to the build.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ ${CMAKE_CURRENT_BINARY_DIR}/oneDPL)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
	message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
	set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
	     STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
	     FORCE)
endif ()

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add an executable target from source files
add_executable(maxloc_buffered maxloc_buffered.cpp)
add_executable(maxloc_implicit maxloc_implicit.cpp)
add_executable(maxloc_operator maxloc_operator.cpp)
add_executable(maxloc_usm maxloc_usm.cpp)
# Specify libraries to link with
target_link_libraries(maxloc_buffered tbb)
target_link_libraries(maxloc_buffered oneDPL)
target_link_libraries(maxloc_implicit tbb)
target_link_libraries(maxloc_implicit oneDPL)
target_link_libraries(maxloc_operator tbb)
target_link_libraries(maxloc_operator oneDPL)
target_link_libraries(maxloc_usm tbb)
target_link_libraries(maxloc_usm oneDPL)

add_test(NAME maxloc_buffered COMMAND $<TARGET_FILE:maxloc_buffered>)
add_test(NAME maxloc_implicit COMMAND $<TARGET_FILE:maxloc_implicit>)
add_test(NAME maxloc_operator COMMAND $<TARGET_FILE:maxloc_operator>)
add_test(NAME maxloc_usm COMMAND $<TARGET_FILE:maxloc_usm>)
