if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "dpcpp")
endif()

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

cmake_minimum_required (VERSION 3.0)
project(histogram LANGUAGES CXX)
find_package(oneDPL REQUIRED)
add_subdirectory (src)
