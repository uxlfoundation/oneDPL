##===----------------------------------------------------------------------===##
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

if (NOT OUTPUT_DIR)
    set(OUTPUT_DIR "output")
endif()

if (SKIP_HEADERS_SUBDIR)
    set(HANDLE_HEADERS_PATH "
get_filename_component(_onedpl_headers \"\${_onedpl_root}/include\" ABSOLUTE)
")
else()
    set(HANDLE_HEADERS_PATH "
if (WIN32)
    set(_onedpl_headers_subdir windows)
else()
    set(_onedpl_headers_subdir linux)
endif()

get_filename_component(_onedpl_headers \"\${_onedpl_root}/\${_onedpl_headers_subdir}/include\" ABSOLUTE)
")
endif()

set(ONEDPL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

file(READ ${ONEDPL_ROOT}/include/oneapi/dpl/internal/version_impl.h _onedpl_version_info LIMIT 1024)
string(REGEX REPLACE ".*#define ONEDPL_VERSION_MAJOR ([0-9]+).*" "\\1" _onedpl_ver_major "${_onedpl_version_info}")
string(REGEX REPLACE ".*#define ONEDPL_VERSION_MINOR ([0-9]+).*" "\\1" _onedpl_ver_minor "${_onedpl_version_info}")
string(REGEX REPLACE ".*#define ONEDPL_VERSION_PATCH ([0-9]+).*" "\\1" _onedpl_ver_patch "${_onedpl_version_info}")

set(PROJECT_VERSION "${_onedpl_ver_major}.${_onedpl_ver_minor}.${_onedpl_ver_patch}")

configure_file("${ONEDPL_ROOT}/cmake/templates/oneDPLConfig.cmake.in"
               "${OUTPUT_DIR}/oneDPLConfig.cmake"
               @ONLY)
configure_file("${ONEDPL_ROOT}/cmake/templates/oneDPLConfigVersion.cmake.in"
               "${OUTPUT_DIR}/oneDPLConfigVersion.cmake"
               @ONLY)

if (SKIP_HEADERS_SUBDIR)
    set(_onedpl_pkgconfig_header_suffix include)
    configure_file("${ONEDPL_ROOT}/integration/pkgconfig/dpl.pc.in" "${OUTPUT_DIR}/dpl.pc" @ONLY)
else()
    set(_onedpl_pkgconfig_header_suffix windows/include)
    configure_file("${ONEDPL_ROOT}/integration/pkgconfig/dpl.pc.in" "${OUTPUT_DIR}/pkgconfig-win/dpl.pc" @ONLY)
    set(_onedpl_pkgconfig_header_suffix linux/include)
    configure_file("${ONEDPL_ROOT}/integration/pkgconfig/dpl.pc.in" "${OUTPUT_DIR}/pkgconfig-lin/dpl.pc" @ONLY)
endif()

message(STATUS "oneDPL ${PROJECT_VERSION} configuration files were created in '${OUTPUT_DIR}'")
