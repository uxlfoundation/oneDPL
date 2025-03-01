##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

if (CMAKE_HOST_WIN32)
    # Requires version 3.20 for baseline support of icx, icx-cl
    cmake_minimum_required(VERSION 3.20)
else()
    # Requires version 3.11 for use of icpc with c++17 requirement
    cmake_minimum_required(VERSION 3.11)
endif()

# oneDPL is a subproject only when PARENT_DIRECTORY is defined
get_directory_property(_onedpl_is_subproject PARENT_DIRECTORY)

include(CMakeDependentOption)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(ONEDPL_FPGA_STATIC_REPORT "Enable the static report generation for the FPGA device" OFF)
option(ONEDPL_USE_AOT_COMPILATION "Enable the ahead of time compilation" OFF)
option(ONEDPL_ENABLE_SIMD "Enable SIMD vectorization by passing an OpenMP SIMD flag to the compiler if supported" ON)
cmake_dependent_option(ONEDPL_TEST_WIN_ICX_FIXES "Enable icx workarounds for Windows" ON "CMAKE_HOST_WIN32;NOT _onedpl_is_subproject" OFF)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/oneapi/dpl/internal/version_impl.h
    _onedpl_version_info
    LIMIT 1024)
string(REGEX REPLACE ".*#define ONEDPL_VERSION_MAJOR ([0-9]+).*" "\\1" _onedpl_ver_major "${_onedpl_version_info}")
string(REGEX REPLACE ".*#define ONEDPL_VERSION_MINOR ([0-9]+).*" "\\1" _onedpl_ver_minor "${_onedpl_version_info}")
string(REGEX REPLACE ".*#define ONEDPL_VERSION_PATCH ([0-9]+).*" "\\1" _onedpl_ver_patch "${_onedpl_version_info}")

# CMAKE_CXX_COMPILER_ID and CMAKE_CXX_COMPILER_VERSION cannot be used because
# CMake 3.19 and older will detect IntelLLVM compiler as CLang with CLang-specific version, see https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
if (CMAKE_CXX_COMPILER MATCHES ".*(dpcpp-cl|dpcpp|icx-cl|icpx|icx)(.exe)?$")
    set(INTEL_LLVM_COMPILER TRUE)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE INTEL_LLVM_COMPILER_VERSION_RAW)
    string(REGEX MATCH "[0-9][0-9][0-9][0-9]\\.[0-9]\\.[0-9]" INTEL_LLVM_COMPILER_VERSION ${INTEL_LLVM_COMPILER_VERSION_RAW})
else()
    set(INTEL_LLVM_COMPILER FALSE)
endif()

if (CMAKE_HOST_WIN32 AND INTEL_LLVM_COMPILER)
    if (_onedpl_is_subproject)
        # If oneDPL is a subproject, warn if workarounds are not included
        if (NOT oneDPLWindowsIntelLLVM_FOUND)
            message(WARNING "On Windows, ${CMAKE_CXX_COMPILER} requires some workarounds to function properly with CMake. Please see \"Using oneDPL package on Windows\" in oneDPL/cmake/README.md for more details.")
        endif()
    elseif(ONEDPL_TEST_WIN_ICX_FIXES)
        #if oneDPL is not a subproject, automatically include workarounds
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
        find_package(oneDPLWindowsIntelLLVM)
    endif()
endif()

project(oneDPL VERSION ${_onedpl_ver_major}.${_onedpl_ver_minor}.${_onedpl_ver_patch} LANGUAGES CXX)
message(STATUS "Configuring ${PROJECT_NAME} ${PROJECT_VERSION}")

find_program(FIND_GXX_EXE g++)
if (FIND_GXX_EXE)
    execute_process(COMMAND ${FIND_GXX_EXE} -dumpfullversion OUTPUT_VARIABLE _onedpl_gxx_version)
endif()

# RelWithAsserts has the same flags as Release, but removes NDEBUG.
# This build type can be enabled via CMAKE_BUILD_TYPE=RelWithAsserts.
# It is useful to fully cover tests relying on asserts, such as those exported from libc++, or Tested Standard C++ API tests.
foreach(FLAGS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
        CMAKE_STATIC_LINKER_FLAGS)
    if (FLAGS STREQUAL "CMAKE_CXX_FLAGS")
        string(REGEX REPLACE "(^| )[/-]DNDEBUG( |$)" " " BORROWED_FLAGS "${${FLAGS}_RELEASE}")
    else()
        set(BORROWED_FLAGS "${${FLAGS}_RELEASE}")
    endif()
    set(${FLAGS}_RELWITHASSERTS "${BORROWED_FLAGS}" CACHE STRING "Flags used during RELWITHASSERTS builds." FORCE)
endforeach()

include(CMakePackageConfigHelpers)
include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)
include(GNUInstallDirs)

# Detect SYCL support
if (NOT DEFINED ONEDPL_BACKEND OR ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
    message(STATUS "Checking for SYCL support")

    set(FSYCL_OPTION "-fsycl")
    check_cxx_compiler_flag(${FSYCL_OPTION} _fsycl_option)
    if (_fsycl_option)
        set(FSYCL_OPTION_IF_SUPPORTED ${FSYCL_OPTION})
    endif()

    CHECK_INCLUDE_FILE_CXX("sycl/sycl.hpp" SYCL_HEADER ${FSYCL_OPTION_IF_SUPPORTED})
    if (NOT SYCL_HEADER)
        CHECK_INCLUDE_FILE_CXX("CL/sycl.hpp" SYCL_HEADER_OLD ${FSYCL_OPTION_IF_SUPPORTED})
    endif()
    if (SYCL_HEADER OR SYCL_HEADER_OLD)
        set(SYCL_SUPPORT TRUE)
    endif()

    if (SYCL_SUPPORT)
        message(STATUS "SYCL is supported")
    else()
        if (ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
            message(FATAL_ERROR "SYCL is not supported. It is required for ONEDPL_BACKEND=${ONEDPL_BACKEND}")
        else()
            message(STATUS "SYCL is not supported")
        endif()
    endif()
endif()

# Set the default backend if one has not been explicitly provided
if (NOT DEFINED ONEDPL_BACKEND)
    if (SYCL_SUPPORT)
        set(ONEDPL_BACKEND "dpcpp" CACHE STRING "Threading backend")
    else()
        set(ONEDPL_BACKEND "tbb" CACHE STRING "Threading backend")
    endif()
    string(TOUPPER ${ONEDPL_BACKEND} ONEDPL_BACKEND)
    message(STATUS "Using ${ONEDPL_BACKEND} as default backend")
endif()

###############################################################################
# Setup the oneDPL library target
###############################################################################
add_library(oneDPL INTERFACE)
target_compile_features(oneDPL INTERFACE cxx_std_17)
target_compile_definitions(oneDPL INTERFACE $<$<CONFIG:Debug>:PSTL_USE_DEBUG=1>)

if (CMAKE_BUILD_TYPE)
    message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type is not set")
endif()

if (MSVC)
    target_compile_options(oneDPL INTERFACE /Zc:__cplusplus /EHsc)
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

string(TOUPPER ${ONEDPL_BACKEND} ONEDPL_BACKEND)
message(STATUS "Using parallel policies with ${ONEDPL_BACKEND} backend")
string(TOLOWER ${ONEDPL_BACKEND} ONEDPL_BACKEND)

if (ONEDPL_ENABLE_SIMD)
    if (NOT INTEL_LLVM_COMPILER OR NOT INTEL_LLVM_COMPILER_VERSION VERSION_LESS 2021.4)
        foreach(_simd_flag -fopenmp-simd /Qopenmp-simd -qopenmp-simd -openmp-simd)
            string(MAKE_C_IDENTIFIER ${_simd_flag} FLAG_DISPLAY_NAME)
            check_cxx_compiler_flag(${_simd_flag} ${FLAG_DISPLAY_NAME}_option)
            if (${FLAG_DISPLAY_NAME}_option)
                target_compile_options(oneDPL INTERFACE ${_simd_flag})
                set(_simd_enabled_flag ${_simd_flag})
                break()
            endif()
        endforeach()
    endif()

    if (_simd_enabled_flag)
        message(STATUS "oneDPL: OpenMP SIMD is enabled by passing '${_simd_enabled_flag}' to compiler")
    else()
        message(STATUS "oneDPL: no effect from enabled ONEDPL_ENABLE_SIMD; unsupported for current compiler")
    endif()
else()
    message(STATUS "oneDPL: ONEDPL_ENABLE_SIMD is OFF, corresponding compiler flag is not used")
endif()

if (NOT _ONEDPL_PSTL_OFFLOAD)
    set(_ONEDPL_PSTL_OFFLOAD "off")
else()
    if (NOT ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
        message(FATAL_ERROR "${ONEDPL_BACKEND} doesn't support pstl offload, dpcpp or dpcpp_only backend required")
    endif()
endif()

if (ONEDPL_BACKEND MATCHES "^(tbb|dpcpp|dpcpp_only)$")
    string(TOUPPER "${ONEDPL_BACKEND}" ONEDPL_BACKEND_NAME)
    set(SET_BACKEND_${ONEDPL_BACKEND_NAME} TRUE)

    if (ONEDPL_BACKEND MATCHES "^(tbb|dpcpp)$")
        find_package(TBB 2021 REQUIRED tbb OPTIONAL_COMPONENTS tbbmalloc)
        message(STATUS "oneDPL uses oneTBB ${TBB_VERSION}")
        target_link_libraries(oneDPL INTERFACE TBB::tbb)
    endif()

    # It is for Clang and Intel® oneAPI DPC++ Compiler (while the last one is detected as Clang; for Linux only), which are used with libstdc++ standard library
    if (UNIX)
        if (CMAKE_CXX_COMPILER_ID STREQUAL Clang)
            if (FIND_GXX_EXE)
                string(REPLACE "\n" "" _onedpl_tbb_use_glibcxx_version ${_onedpl_gxx_version})
                string(REPLACE "\." "0" _onedpl_tbb_use_glibcxx_version ${_onedpl_tbb_use_glibcxx_version})
                target_compile_definitions(oneDPL INTERFACE TBB_USE_GLIBCXX_VERSION=${_onedpl_tbb_use_glibcxx_version})
            else()
                target_compile_definitions(oneDPL INTERFACE TBB_USE_GLIBCXX_VERSION=70300)
            endif()
        endif()
    endif()

    target_compile_definitions(oneDPL INTERFACE
        $<$<CONFIG:Debug>:TBB_USE_DEBUG=1>
        $<$<BOOL:${SET_BACKEND_DPCPP_ONLY}>:ONEDPL_USE_TBB_BACKEND=0>
        $<$<BOOL:${SET_BACKEND_TBB}>:ONEDPL_USE_DPCPP_BACKEND=0>
        )

    if (ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
        # check device type for oneDPL test targets
        if (ONEDPL_DEVICE_TYPE MATCHES "^(CPU|GPU|FPGA_EMU|FPGA_HW)$")
            if (NOT _ONEDPL_PSTL_OFFLOAD STREQUAL off)
                message(FATAL_ERROR "PSTL offload does device selection on its own, ONEDPL_DEVICE_TYPE ${ONEDPL_DEVICE_TYPE} is useless.")
            endif()
            message(STATUS "Using ${ONEDPL_DEVICE_TYPE} device type")
            set(ONEDPL_USE_DEVICE_${ONEDPL_DEVICE_TYPE} TRUE)
        elseif (DEFINED ONEDPL_DEVICE_BACKEND)
            message(FATAL_ERROR "Unsupported device type: ${ONEDPL_DEVICE_TYPE}.\n"
                "Select one of the following devices: CPU, GPU, FPGA_EMU or FPGA_HW")
        elseif (NOT _ONEDPL_PSTL_OFFLOAD STREQUAL off)
            # PSTL offload does device selection on its own
        else()
            set(ONEDPL_DEVICE_TYPE "GPU")
            message(STATUS "Using a default device type (GPU)")
        endif()

        # check device backend
        if (ONEDPL_DEVICE_BACKEND MATCHES "^(opencl|level_zero|cuda|hip|\\*)$")
            message(STATUS "Using ${ONEDPL_DEVICE_BACKEND} device backend")
        elseif (DEFINED ONEDPL_DEVICE_BACKEND)
            message(FATAL_ERROR "Unsupported device backend: ${ONEDPL_DEVICE_BACKEND}.\n"
                "Select one of the following device backends: opencl, level_zero, cuda, hip or *")
        else()
            set(ONEDPL_DEVICE_BACKEND "*")
            message(STATUS "Using a default device backend (*)")
        endif()

        if (INTEL_LLVM_COMPILER AND INTEL_LLVM_COMPILER_VERSION VERSION_LESS 2023.1)
            set(DEVICE_SELECTION_VARIABLE "SYCL_DEVICE_FILTER")
            set(FPGA_DEVICE_TYPE "acc")
        else()
            set(DEVICE_SELECTION_VARIABLE "ONEAPI_DEVICE_SELECTOR")
            set(FPGA_DEVICE_TYPE "fpga")
        endif()

        # set up device selection line basing on ONEDPL_DEVICE_BACKEND and ONEDPL_DEVICE_TYPE values
        if (DEFINED ONEDPL_DEVICE_BACKEND AND DEFINED ONEDPL_DEVICE_TYPE)
            if (ONEDPL_DEVICE_TYPE MATCHES "FPGA")
                set(DEVICE_SELECTION_LINE "${DEVICE_SELECTION_VARIABLE}=${ONEDPL_DEVICE_BACKEND}:${FPGA_DEVICE_TYPE}")
            else()
                string(TOLOWER ${ONEDPL_DEVICE_TYPE} ONEDPL_DEVICE_TYPE)
                set(DEVICE_SELECTION_LINE "${DEVICE_SELECTION_VARIABLE}=${ONEDPL_DEVICE_BACKEND}:${ONEDPL_DEVICE_TYPE}")
            endif()
        endif()

        # Check correctness of STATIC_REPORT
        if (ONEDPL_FPGA_STATIC_REPORT)
            if (NOT ONEDPL_USE_DEVICE_FPGA_HW)
                message(FATAL_ERROR "Static report can only be generated for FPGA hardware")
            else()
                message(STATUS "Static report will be generated")
            endif()
        endif()

        # settings for the specific compilation type
        if (NOT ONEDPL_USE_AOT_COMPILATION)
            message(STATUS "Using just-in-time compilation")
        else()
            if (NOT ONEDPL_AOT_ARCH)
                if (ONEDPL_USE_DEVICE_GPU)
                    set(ONEDPL_AOT_ARCH "*")
                endif()
                if (ONEDPL_USE_DEVICE_CPU)
                    set(ONEDPL_AOT_ARCH "avx")
                endif()
            endif()
            message(STATUS "Using ahead-of-time compilation for \"${ONEDPL_AOT_ARCH}\" architecture(s)")

            if (ONEDPL_USE_DEVICE_GPU)
                # -cl-kernel-debug-enable is not needed since https://github.com/intel/compute-runtime/commit/9c237d99ec49ceb3678c815314bc7c745226e4a2
                # TODO: remove that option when older driver versions are not expected to be used
                set(ONEDPL_AOT_EXTRA_BACKEND_OPTIONS "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:-internal_options -cl-kernel-debug-enable>")
                set(ONEDPL_AOT_OPTIONS "-fsycl-targets=spir64_gen -Xs \"-device ${ONEDPL_AOT_ARCH} ${ONEDPL_AOT_EXTRA_BACKEND_OPTIONS}\"")
            endif()
            if (ONEDPL_USE_DEVICE_CPU)
                set(ONEDPL_AOT_OPTIONS "-fsycl-targets=spir64_x86_64 -Xs \"-march=${ONEDPL_AOT_ARCH}\"")
            endif()
        endif()

        # DPC++ specific compiler options
        target_compile_options(oneDPL INTERFACE
            ${FSYCL_OPTION_IF_SUPPORTED}
            $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>>:-fintelfpga>
            )
        if (DEFINED ONEDPL_USE_UNNAMED_LAMBDA)
            if(ONEDPL_USE_UNNAMED_LAMBDA)
                message(STATUS "Adding -fsycl-unnamed-lambda option")
                target_compile_options(oneDPL INTERFACE -fsycl-unnamed-lambda)
            else()
                message(STATUS "Adding -fno-sycl-unnamed-lambda option")
                target_compile_options(oneDPL INTERFACE -fno-sycl-unnamed-lambda)
            endif()
        endif()

        if (DEFINED ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION)
            if(ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION)
                message(STATUS "Adding -DONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION=1 option")
                target_compile_options(oneDPL INTERFACE "-DONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION=1")
            else()
                message(STATUS "Adding -DONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION=0 option")
                target_compile_options(oneDPL INTERFACE "-DONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION=0")
            endif()
        endif()

        # DPC++ specific macro
        target_compile_definitions(oneDPL INTERFACE
            $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>>:ONEDPL_FPGA_DEVICE>
            $<$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>:ONEDPL_FPGA_EMULATOR>
            )

        # DPC++ specific link options
        target_link_libraries(oneDPL INTERFACE
            ${FSYCL_OPTION_IF_SUPPORTED}
            $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>>:-fintelfpga>
            $<$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>:-Xshardware>
            $<$<AND:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${ONEDPL_FPGA_STATIC_REPORT}>>:-fsycl-link>
            ${ONEDPL_AOT_OPTIONS}
            )
    endif()

elseif(ONEDPL_BACKEND MATCHES "^(serial)$")
    target_compile_definitions(oneDPL INTERFACE
        ONEDPL_USE_TBB_BACKEND=0
        ONEDPL_USE_DPCPP_BACKEND=0
        ONEDPL_USE_OPENMP_BACKEND=0
        )
    message(STATUS "Compilation for the host due to serial backend")

elseif(ONEDPL_BACKEND MATCHES "^(omp)$")
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        message(STATUS "Compilation for the host due to OpenMP backend.")
        # Due to minor correctness issues with -fiopenmp / -Qiopenmp, we are using -fopenmp / -Qopenmp until they are corrected.
        # Once correctness issues are resolved, we will limit this workaround to affected versions of specific compilers.
        if (OpenMP_CXX_FLAGS MATCHES ".*-fiopenmp.*")
            set(_openmp_flag -fopenmp)
        elseif (OpenMP_CXX_FLAGS MATCHES ".*[-/]Qiopenmp.*")
            set(_openmp_flag -Qopenmp)
        endif()
        if (_openmp_flag)
            message(STATUS "Using ${_openmp_flag} for openMP")
            target_compile_options(oneDPL INTERFACE ${_openmp_flag})
            target_link_libraries(oneDPL INTERFACE ${_openmp_flag})
        else()
            target_link_libraries(oneDPL INTERFACE OpenMP::OpenMP_CXX)
        endif()
        target_compile_definitions(oneDPL INTERFACE
            ONEDPL_USE_TBB_BACKEND=0
            ONEDPL_USE_DPCPP_BACKEND=0
            ONEDPL_USE_OPENMP_BACKEND=1
            )
    else()
        message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support OpenMP.\n"
            "It is required if ONEDPL_BACKEND=${ONEDPL_BACKEND}")
    endif()

else()
    message(STATUS "Using Parallel Policies, but not oneTBB/DPC++")
    if (TARGET ${ONEDPL_BACKEND})
        target_link_libraries(oneDPL INTERFACE ${ONEDPL_BACKEND})
    else()
        find_package(${ONEDPL_BACKEND} REQUIRED)
        target_link_libraries(oneDPL INTERFACE ${${ONEDPL_BACKEND}_IMPORTED_TARGETS})
    endif()
endif()

if (NOT _ONEDPL_PSTL_OFFLOAD STREQUAL off)
    if (_ONEDPL_PSTL_OFFLOAD STREQUAL default)
        set(FSYCL_PSTL_OFFLOAD_OPTION "-fsycl-pstl-offload")
    elseif(_ONEDPL_PSTL_OFFLOAD STREQUAL cpu OR _ONEDPL_PSTL_OFFLOAD STREQUAL gpu)
        set(FSYCL_PSTL_OFFLOAD_OPTION "-fsycl-pstl-offload=${_ONEDPL_PSTL_OFFLOAD}")
    else()
        message(FATAL_ERROR "Unknown value of _ONEDPL_PSTL_OFFLOAD = ${_ONEDPL_PSTL_OFFLOAD}")
    endif()

    check_cxx_compiler_flag("${FSYCL_PSTL_OFFLOAD_OPTION}" _fsycl_pstl_offload_option)
    if (_fsycl_pstl_offload_option)
        message(STATUS "Adding ${FSYCL_PSTL_OFFLOAD_OPTION} compiler option")
        target_compile_options(oneDPL INTERFACE ${FSYCL_PSTL_OFFLOAD_OPTION})
        target_link_libraries(oneDPL INTERFACE ${FSYCL_PSTL_OFFLOAD_OPTION})
    endif()
endif()

target_include_directories(oneDPL
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

###############################################################################
# Setup tests
###############################################################################
if (NOT _onedpl_is_subproject)
    enable_testing()
    add_subdirectory(test)
endif()

###############################################################################
# Installation instructions
###############################################################################
install(CODE "set(OUTPUT_DIR \"${CMAKE_INSTALL_FULL_LIBDIR}/cmake/oneDPL\")")
install(CODE "set(SKIP_HEADERS_SUBDIR TRUE)")
install(SCRIPT cmake/scripts/generate_config.cmake)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY licensing DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(FILES cmake/oneDPLWindowsIntelLLVMApply.cmake cmake/oneDPLWindowsIntelLLVMConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oneDPL/)
