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
#define ONEDPL_VERSION_MINOR 14
#define ONEDPL_VERSION_PATCH 0

// The oneAPI Specification version this implementation is compliant with
#define ONEDPL_SPEC_VERSION 105

// -- Check availability of heterogeneous backends --

// If DPCPP backend is explicitly requested, optimistically assume SYCL availability;
// otherwise, make sure that it is definitely available additionally checking SYCL_LANGUAGE_VERSION
#if __has_include(<sycl/sycl.hpp>) || __has_include(<CL/sycl.hpp>)
#    if SYCL_LANGUAGE_VERSION || CL_SYCL_LANGUAGE_VERSION || ONEDPL_USE_DPCPP_BACKEND
#        define _ONEDPL_SYCL_AVAILABLE 1
#    endif
#else
#    if ONEDPL_USE_DPCPP_BACKEND
#        error "Device execution policies are requested, but SYCL* headers are not found"
#    endif
#endif

// If DPCPP backend is not explicitly turned off and SYCL is available, enable it
#if (ONEDPL_USE_DPCPP_BACKEND || !defined(ONEDPL_USE_DPCPP_BACKEND)) && _ONEDPL_SYCL_AVAILABLE
#    define _ONEDPL_BACKEND_SYCL 1
#endif

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
// Ranges library is available if the standard library provides it and concepts are supported
// Clang 15 and older do not support range adaptors, see https://bugs.llvm.org/show_bug.cgi?id=44833
#    define _ONEDPL_CPP20_RANGES_PRESENT ((__cpp_lib_ranges >= 201911L) && !(__clang__ && __clang_major__ < 16))
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
#define ONEDPL_HAS_RANDOM_NUMBERS         202603L
#if _ONEDPL_CPP20_RANGES_PRESENT
#    define ONEDPL_HAS_RANGE_ALGORITHMS   202608L
#endif

#if _ONEDPL_BACKEND_SYCL
// Device containers will only be defined with the dpcpp backend.
#    define ONEDPL_HAS_DEVICE_CONTAINERS 202608L
#endif

#endif // _ONEDPL_VERSION_IMPL_H
