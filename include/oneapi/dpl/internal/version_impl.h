// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_VERSION_IMPL_H
#define _ONEDPL_VERSION_IMPL_H

// The library version
#define ONEDPL_VERSION_MAJOR 2022
#define ONEDPL_VERSION_MINOR 11
#define ONEDPL_VERSION_PATCH 0

// The oneAPI Specification version this implementation is compliant with
#define ONEDPL_SPEC_VERSION 104

// -- Check for C++ standard library feature macros --
#if __has_include(<version>)
#    include <version>
#    define _ONEDPL_STD_FEATURE_MACROS_PRESENT 1
#    define _ONEDPL_CPP20_CONCEPTS_PRESENT (__cpp_concepts >= 201907L && __cpp_lib_concepts >= 202002L)
#else
#    define _ONEDPL_STD_FEATURE_MACROS_PRESENT 0
#    define _ONEDPL_CPP20_CONCEPTS_PRESENT 0
#endif

// -- Check for C++20 Ranges support --
#if _ONEDPL_CPP20_CONCEPTS_PRESENT
//   Ranges library is available if the standard library provides it and concepts are supported
#    if defined(_LIBCPP_VERSION)
#        define _ONEDPL_LIBCPP_VERSION_CHECK (_LIBCPP_VERSION >= 16000)
#    else
#        define _ONEDPL_LIBCPP_VERSION_CHECK 1
#    endif

//   Clang 15 and older do not support range adaptors, see https://bugs.llvm.org/show_bug.cgi?id=44833
#    if defined(__clang__)
#        define _ONEDPL_CLANG_VERSION_CHECK (__clang_major__ >= 16)
#    else
#        define _ONEDPL_CLANG_VERSION_CHECK 1
#    endif

#    define _ONEDPL_CPP20_RANGES_PRESENT                                                                               \
        ((__cpp_lib_ranges >= 201911L) && _ONEDPL_LIBCPP_VERSION_CHECK && _ONEDPL_CLANG_VERSION_CHECK)
#else
#    define _ONEDPL_CPP20_RANGES_PRESENT 0
#endif

#ifndef _PSTL_VERSION
#    define _PSTL_VERSION 14000
#    define _PSTL_VERSION_MAJOR (_PSTL_VERSION / 1000)
#    define _PSTL_VERSION_MINOR ((_PSTL_VERSION % 1000) / 10)
#    define _PSTL_VERSION_PATCH (_PSTL_VERSION % 10)
#endif

// -- Define oneDPL feature macros --
#define ONEDPL_HAS_RANDOM_NUMBERS         202409L
#if _ONEDPL_CPP20_RANGES_PRESENT
#    define ONEDPL_HAS_RANGE_ALGORITHMS   202509L
#endif

#endif // _ONEDPL_VERSION_IMPL_H
