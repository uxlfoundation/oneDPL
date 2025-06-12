if (CMAKE_HOST_WIN32)
    # Requires version 3.20 for baseline support of icx, icx-cl
    cmake_minimum_required(VERSION 3.20)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)
    find_package(oneDPLWindowsIntelLLVM)
else()
    # Requires version 3.11 for use of icpc with c++17 requirement
    cmake_minimum_required(VERSION 3.11)
endif()

project (dynamic_selection)

set(ONEDPL_BACKEND dpcpp_only)
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

option (BUILD_NSTREAM "Build targets from subproject nstream" ON)
option (BUILD_SEPIA-FILTER-DS "Build targets from subproject sepia-filter-ds" ON)

if (BUILD_NSTREAM)
	add_subdirectory (nstream)
endif ()

if (BUILD_SEPIA-FILTER-DS)
	add_subdirectory (sepia-filter-ds)
endif()
