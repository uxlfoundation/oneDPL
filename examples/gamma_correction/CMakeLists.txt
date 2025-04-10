if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "dpcpp")
endif()

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

cmake_minimum_required (VERSION 3.0)
if (WIN32)
    list(APPEND CMAKE_PREFIX_PATH /path/to/oneDPL/cmake)
    find_package(oneDPLWindowsIntelLLVM)
endif()
project(gamma_correction LANGUAGES CXX)
enable_testing()
# Specify oneDPL backend
set(ONEDPL_BACKEND dpcpp)
# Add oneDPL to the build.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ ${CMAKE_CURRENT_BINARY_DIR}/oneDPL)
add_subdirectory (src)
