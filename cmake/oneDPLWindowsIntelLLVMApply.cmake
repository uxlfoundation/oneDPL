##===-- oneDPLWindowsIntelLLVMApply.cmake ---------------------------------------===##
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

# The following fixes are requested by oneDPLWindowsIntelLLVMConfig.cmake (and must be applied after the project() call)
if (INTELLLVM_MSVC_WIN_STDOPTION_FIX)
    # Fix std compiler options for icx, icx-cl (Adapted from https://github.com/Kitware/CMake/commit/42ca6416afeabd445bc6c19749e68604c9c2d733)
    set(CMAKE_CXX14_STANDARD_COMPILE_OPTION  "-Qstd:c++14")
    set(CMAKE_CXX14_EXTENSION_COMPILE_OPTION "-Qstd:c++14")
    set(CMAKE_CXX17_STANDARD_COMPILE_OPTION  "-Qstd:c++17")
    set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION "-Qstd:c++17")
    set(CMAKE_CXX20_STANDARD_COMPILE_OPTION  "-Qstd:c++20")
    set(CMAKE_CXX20_EXTENSION_COMPILE_OPTION "-Qstd:c++20")
endif()

if (INTELLLVM_MSVC_WIN_LINKORDER_FIX)
    # Fixing linker rule to use the compiler for linking and moving link options before /link
    # Adapted from fix in CMake 3.23: https://github.com/Kitware/CMake/commit/5d5a7123034361b6cacff96d3ed20d2bb78c33cc
    set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_COMMAND> -E vs_link_exe --intdir=<OBJECT_DIR> --rc=<CMAKE_RC_COMPILER> --mt=<CMAKE_MT> --manifests <MANIFESTS> -- <CMAKE_CXX_COMPILER> /nologo <CMAKE_CXX_LINK_FLAGS> <OBJECTS> <LINK_FLAGS> <LINK_LIBRARIES> /link /out:<TARGET> /implib:<TARGET_IMPLIB> /pdb:<TARGET_PDB> /version:<TARGET_VERSION_MAJOR>.<TARGET_VERSION_MINOR>")
endif()

if (INTELLLVM_WIN_OFFICIAL_SUPPORT_FIX)
    # Intel provided workaround for CMake version 3.23+
    find_package(IntelDPCPP REQUIRED)
endif()

if (INTELLLVM_WIN_STD_IGNORE_FIX)
    include(Compiler/CMakeCommonCompilerMacros)
    # For IntelLLVM versions greater or equal to 2020, setting c++ standard default to 14, and enable usage of CMAKE_CXX_STANDARD
    __compiler_check_default_language_standard(CXX 2020 14)
endif()
