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
#define ONEDPL_VERSION_MINOR 7
#define ONEDPL_VERSION_PATCH 0

#if _ONEDPL___cplusplus >= 202002L && __has_include(<version>)
#    include <version> // The standard C++20 header
#    define _ONEDPL_STD_FEATURE_MACROS_PRESENT 1
// Clang 15 and older do not support range adaptors, see https://bugs.llvm.org/show_bug.cgi?id=44833
#    define _ONEDPL_CPP20_RANGES_PRESENT ((__cpp_lib_ranges >= 201911L) && !(__clang__ && __clang_major__ < 16))
#else
#    define _ONEDPL_STD_FEATURE_MACROS_PRESENT 0
#    define _ONEDPL_CPP20_RANGES_PRESENT 0
#endif

#ifndef _PSTL_VERSION
#    define _PSTL_VERSION 14000
#    define _PSTL_VERSION_MAJOR (_PSTL_VERSION / 1000)
#    define _PSTL_VERSION_MINOR ((_PSTL_VERSION % 1000) / 10)
#    define _PSTL_VERSION_PATCH (_PSTL_VERSION % 10)
#endif

// Feature macros
#define ONEDPL_HAS_RANDOM_NUMBERS         202409L
#if _ONEDPL_CPP20_RANGES_PRESENT
#    define ONEDPL_HAS_RANGE_ALGORITHMS   202409L
#endif

#endif // _ONEDPL_VERSION_IMPL_H
