if (CMAKE_HOST_WIN32)
    # Requires version 3.20 for baseline support of icx, icx-cl
    cmake_minimum_required(VERSION 3.20)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)
    find_package(oneDPLWindowsIntelLLVM)
else()
    # Requires version 3.11 for use of icpc with c++17 requirement
    cmake_minimum_required(VERSION 3.11)
endif()

project (pSTL_offload)

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

option (TARGET_GPU "Build targets for GPU target" OFF)
option (BUILD_FILEWORDCOUNT "Build targets from subproject FileWordCount" ON)
option (BUILD_WORDCOUNT "Build targets from subproject WordCount" ON)
option (BUILD_PARSTLTESTS "Build targets from subproject ParSTLTests" ON)

if(TARGET_GPU)
  #To build for Intel® Data Center GPU Max 1550 or 1100
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-pstl-offload=gpu -D GPU")
else()
  #To build for CPU 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CPU ")
endif()


if (BUILD_FILEWORDCOUNT)
	add_subdirectory (FileWordCount)
endif ()

if (BUILD_WORDCOUNT)
	add_subdirectory (WordCount)
endif ()

if (BUILD_PARSTLTESTS)
	add_subdirectory (ParSTLTests)
endif ()
