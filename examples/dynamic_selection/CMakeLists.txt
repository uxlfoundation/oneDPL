cmake_minimum_required (VERSION 3.4.0)

set(CMAKE_CXX_COMPILER "icpx")

project (dynamic_selection)
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
	message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
	set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
	     STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
	     FORCE)
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option (BUILD_NSTREAM "Build targets from subproject nstream" OFF)
option (BUILD_SEPIA-FILTER-DS "Build targets from subproject sepia-filter-ds" OFF)

if (BUILD_NSTREAM)
	add_subdirectory (nstream)
endif ()

if (BUILD_SEPIA-FILTER-DS)
	add_subdirectory (sepia-filter-ds)
endif()
