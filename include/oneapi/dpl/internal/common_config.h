// -*- C++ -*-
//===-- common_config.h ---------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_COMMON_CONFIG_H
#define _ONEDPL_COMMON_CONFIG_H

#if __cplusplus >= 201703L
#    define _ONEDPL___cplusplus __cplusplus
#elif defined(_MSVC_LANG) && _MSVC_LANG >= 201703L
#    define _ONEDPL___cplusplus _MSVC_LANG
#else
#    error "oneDPL requires the C++ language version not less than C++17"
#endif

// Disable use of TBB in Parallel STL from libstdc++.
// This workaround is for GCC only, so we should use __cplusplus macro
// instead of _ONEDPL___cplusplus one.
#if __cplusplus >= 201703L
// - New TBB version with incompatible APIs is found (libstdc++ v9/v10)
#    if __has_include(<tbb/version.h>)
#        if defined(_GLIBCXX_RELEASE) && (_GLIBCXX_RELEASE == 9 || _GLIBCXX_RELEASE == 10)
//           If STL headers are included before oneDPL, __PSTL_USE_PAR_POLICIES,
//           __PSTL_PAR_BACKEND_TBB and _PSTL_PAR_BACKEND_TBB macros can be defined
//           before this config file
#            ifdef __PSTL_USE_PAR_POLICIES
#                undef __PSTL_USE_PAR_POLICIES
#                define __PSTL_USE_PAR_POLICIES 0
#            endif
#            ifdef __PSTL_PAR_BACKEND_TBB
#                undef __PSTL_PAR_BACKEND_TBB
#                define __PSTL_PAR_BACKEND_TBB 0
#            endif
#            ifdef _PSTL_PAR_BACKEND_TBB // For GCC10
#                undef _PSTL_PAR_BACKEND_TBB
#                define _PSTL_PAR_BACKEND_SERIAL
#            endif
#        endif
#        ifndef PSTL_USE_PARALLEL_POLICIES
#            define PSTL_USE_PARALLEL_POLICIES (_GLIBCXX_RELEASE != 9)
#        endif
#        ifndef _GLIBCXX_USE_TBB_PAR_BACKEND
#            define _GLIBCXX_USE_TBB_PAR_BACKEND (_GLIBCXX_RELEASE > 10)
#        endif
#    endif // __has_include(<tbb/version.h>)
// - TBB is not found (libstdc++ v9)
#    if !__has_include(<tbb/tbb.h>) && !defined(PSTL_USE_PARALLEL_POLICIES)
#        ifdef __PSTL_USE_PAR_POLICIES
#            undef __PSTL_USE_PAR_POLICIES
#            define __PSTL_USE_PAR_POLICIES 0
#        endif
#        ifdef __PSTL_PAR_BACKEND_TBB
#            undef __PSTL_PAR_BACKEND_TBB
#            define __PSTL_PAR_BACKEND_TBB 0
#        endif
#        define PSTL_USE_PARALLEL_POLICIES (_GLIBCXX_RELEASE != 9)
#    endif
#endif // __cplusplus >= 201703L

#endif // _ONEDPL_COMMON_CONFIG_H
