cmake_minimum_required (VERSION 3.4.0)

if (UNIX)
	# Direct CMake to use icpx rather than the default C++ compiler/linker
	set(CMAKE_CXX_COMPILER icpx)
else() # Windows
	# Force CMake to use icx-cl rather than the default C++ compiler/linker
	# (needed on Windows only)
	include (CMakeForceCompiler)
	CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
	include (Platform/Windows-Clang)
endif()

project (pSTL_offload)
# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
	message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
	set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
	     STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
	     FORCE)
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option (BUILD_SYCL_MIGRATED_SUBPROJECT "Build targets from subproject FileWordCount" ON)
option (BUILD_SYCL_MIGRATED_SUBPROJECT "Build targets from subproject WordCount" ON)

if (BUILD_SYCL_MIGRATED_SUBPROJECT)
	add_subdirectory (FileWordCount)
endif ()

if (BUILD_SYCL_MIGRATED_SUBPROJECT)
	add_subdirectory (WordCount)
endif ()