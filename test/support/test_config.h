// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _TEST_CONFIG_H
#define _TEST_CONFIG_H

// Any include from standard library required to have correct state of _GLIBCXX_RELEASE
#if __has_include(<version>)
#   include <version>
#else
#   include <ciso646>
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// *** When updating we must audit each usage to ensure that the issue still exists in the latest version ***

//
// This section contains macros representing the "Latest" version of compilers, STL implementations, etc. for use in
// broken macros to represent the latest version of something which still has an ongoing issue. The intention is to
// update this section regularly to reflect the latest version.
//
// When such an issue is fixed, we must replace the usage of these "Latest" macros with the appropriate version number
// before updating to the newest version in this section.

#define _PSTL_TEST_LATEST_INTEL_LLVM_COMPILER 20250200

#define _PSTL_TEST_LATEST_MSVC_STL_VERSION 143

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#define _PSTL_TEST_STRING(X) _PSTL_TEST_STRING_AUX(oneapi/dpl/X)
#define _PSTL_TEST_STRING_AUX(X) #X
//to support the optional including: <algorithm>, <memory>, <numeric> or <pstl/algorithm>, <pstl/memory>, <pstl/numeric>
#define _PSTL_TEST_HEADER(HEADER_ID) _PSTL_TEST_STRING(HEADER_ID)

#if defined(_MSC_VER) && defined(_DEBUG)
#define _SCL_SECURE_NO_WARNINGS //to prevent the compilation warning. Microsoft STL implementation has specific checking of an iterator range in DEBUG mode for the containers from the standard library.
#endif

// ICC 18 (Windows) has encountered an unexpected problem on some tests
#define _PSTL_ICC_18_VC141_TEST_SIMD_LAMBDA_RELEASE_BROKEN                                                            \
    (!_DEBUG && __INTEL_COMPILER >= 1800 && __INTEL_COMPILER < 1900 && _MSC_VER == 1910)
// ICC 18 doesn't vectorize the loop
#define _PSTL_ICC_18_TEST_EARLY_EXIT_MONOTONIC_RELEASE_BROKEN (!_DEBUG && __INTEL_COMPILER && __INTEL_COMPILER == 1800)
// ICC 18 generates wrong result with omp simd early_exit
#define _PSTL_ICC_18_TEST_EARLY_EXIT_AVX_RELEASE_BROKEN                                                               \
    (!_DEBUG && __INTEL_COMPILER == 1800 && __AVX__ && !__AVX2__ && !__AVX512__)
// ICC 19 has encountered an unexpected problem: Segmentation violation signal raised
#define _PSTL_ICC_19_TEST_IS_PARTITIONED_RELEASE_BROKEN                                                               \
    (!PSTL_USE_DEBUG && (__linux__ || __APPLE__) && __INTEL_COMPILER == 1900)
// ICC 19 generates wrong result with UDS on Windows
#define _PSTL_ICC_19_TEST_SIMD_UDS_WINDOWS_RELEASE_BROKEN (__INTEL_COMPILER == 1900 && _MSC_VER && !_DEBUG)
// ICPC compiler generates wrong "openMP simd" code for a user defined scan operation(UDS)
#define _PSTL_ICC_TEST_SIMD_UDS_BROKEN                                                                                \
    (__INTEL_COMPILER && __INTEL_COMPILER_BUILD_DATE < 20211123)
// ICC 18,19 generate wrong result
#define _PSTL_ICC_18_19_TEST_SIMD_MONOTONIC_WINDOWS_RELEASE_BROKEN                                                    \
    ((__INTEL_COMPILER == 1800 || __INTEL_COMPILER == 1900) && _MSC_VER && !_DEBUG)
// ICC 18,19 generate wrong result with for_loop_strided and reverse iterators
#define _PSTL_ICC_18_19_TEST_REVERSE_ITERATOR_WITH_STRIDE_BROKEN                                                      \
    (__i386__ && (__INTEL_COMPILER == 1800 || __INTEL_COMPILER == 1900))
// VC14 uninitialized_fill with no policy has broken implementation
#define _PSTL_STD_UNINITIALIZED_FILL_BROKEN (_MSC_VER == 1900)
// GCC10 produces wrong answer calling exclusive_scan using vectorized polices
#define TEST_GCC10_EXCLUSIVE_SCAN_BROKEN (_GLIBCXX_RELEASE == 10)
// GCC7 std::get doesn't return const rvalue reference from const rvalue reference of tuple
#define _PSTL_TEST_GCC7_RVALUE_TUPLE_GET_BROKEN (_GLIBCXX_RELEASE > 0 && _GLIBCXX_RELEASE < 8)

#define _PSTL_SYCL_TEST_USM 1

#define TEST_SYCL_HEADER_PRESENT (__has_include(<sycl/sycl.hpp>) || __has_include(<CL/sycl.hpp>))
#define TEST_SYCL_LANGUAGE_VERSION_PRESENT (SYCL_LANGUAGE_VERSION || CL_SYCL_LANGUAGE_VERSION)
#define TEST_SYCL_AVAILABLE (TEST_SYCL_HEADER_PRESENT && TEST_SYCL_LANGUAGE_VERSION_PRESENT)

// If SYCL is available, and DPCPP backend is not explicitly turned off, enable its testing
#if TEST_SYCL_AVAILABLE && !defined(ONEDPL_USE_DPCPP_BACKEND)
#    define TEST_DPCPP_BACKEND_PRESENT 1
// If DPCPP backend was explicitly requested, enable its testing, even if SYCL availability has not been proven
// this can be used to force DPCPP backend testing for environments where SYCL_LANGUAGE_VERSION is not predefined
#elif ONEDPL_USE_DPCPP_BACKEND
#    define TEST_DPCPP_BACKEND_PRESENT 1
// Define to 0 in other cases since some tests may rely at the macro value at runtime
#else
#    define TEST_DPCPP_BACKEND_PRESENT 0
#endif

#ifdef __SYCL_UNNAMED_LAMBDA__
#define TEST_UNNAMED_LAMBDAS 1
#else
#define TEST_UNNAMED_LAMBDAS 0
#endif

// The TEST_EXPLICIT_KERNEL_NAMES macro may be defined on CMake level in CMakeLists.txt
// so we should check here if it is defined or not
#ifndef TEST_EXPLICIT_KERNEL_NAMES
#    if __SYCL_UNNAMED_LAMBDA__
#        define TEST_EXPLICIT_KERNEL_NAMES 0
#    else
#        define TEST_EXPLICIT_KERNEL_NAMES 1
#    endif // __SYCL_UNNAMED_LAMBDA__
#endif // !TEST_EXPLICIT_KERNEL_NAMES

// Enables full scope of testing
#ifndef TEST_LONG_RUN
#define TEST_LONG_RUN 0
#endif

// Enable test when the TBB backend is available
#if !defined(ONEDPL_USE_TBB_BACKEND) || ONEDPL_USE_TBB_BACKEND
#define TEST_TBB_BACKEND_PRESENT 1
#endif

// Check for C++ standard and standard library for the use of ranges API
#if !defined(_ENABLE_RANGES_TESTING)
#define _TEST_RANGES_FOR_CPP_17_DPCPP_BE_ONLY TEST_DPCPP_BACKEND_PRESENT
#if defined(_GLIBCXX_RELEASE)
#    define _ENABLE_RANGES_TESTING (_TEST_RANGES_FOR_CPP_17_DPCPP_BE_ONLY && _GLIBCXX_RELEASE >= 8 && __GLIBCXX__ >= 20180502)
#elif defined(_LIBCPP_VERSION)
#    define _ENABLE_RANGES_TESTING (_TEST_RANGES_FOR_CPP_17_DPCPP_BE_ONLY && _LIBCPP_VERSION >= 7000)
#else
#    define _ENABLE_RANGES_TESTING (_TEST_RANGES_FOR_CPP_17_DPCPP_BE_ONLY)
#endif
#endif //!defined(_ENABLE_RANGES_TESTING)

#if (__cplusplus >= 202002L || _MSVC_LANG >= 202002L) && __has_include(<version>)
#    include <version>
#    define TEST_STD_FEATURE_MACROS_PRESENT 1
#endif

#if TEST_STD_FEATURE_MACROS_PRESENT
// Make sure _ENABLE_STD_RANGES_TESTING is always defined for the use at runtime, e.g. by TestUtils::done
// Clang 15 and older do not support range adaptors, see https://bugs.llvm.org/show_bug.cgi?id=44833
#   if __cpp_lib_ranges >= 201911L && !(__clang__ && __clang_major__ < 16)
#       define _ENABLE_STD_RANGES_TESTING 1
#   else
#       define _ENABLE_STD_RANGES_TESTING 0
#   endif
#   define TEST_CPP20_SPAN_PRESENT (__cpp_lib_span >= 202002L)
#else
#   define _ENABLE_STD_RANGES_TESTING 0
#   define TEST_CPP20_SPAN_PRESENT 0
#endif // TEST_STD_FEATURE_MACROS_PRESENT

#define TEST_HAS_NO_INT128
#define _PSTL_TEST_COMPLEX_NON_FLOAT_AVAILABLE (_MSVC_STL_VERSION < 143)

#define _PSTL_GLIBCXX_TEST_COMPLEX_BROKEN (__GLIBCXX__ >= 7)

#define _PSTL_GLIBCXX_TEST_COMPLEX_POW_BROKEN _PSTL_GLIBCXX_TEST_COMPLEX_BROKEN
#define _PSTL_GLIBCXX_TEST_COMPLEX_DIV_EQ_BROKEN _PSTL_GLIBCXX_TEST_COMPLEX_BROKEN
#define _PSTL_GLIBCXX_TEST_COMPLEX_MINUS_EQ_BROKEN _PSTL_GLIBCXX_TEST_COMPLEX_BROKEN
#define _PSTL_GLIBCXX_TEST_COMPLEX_PLUS_EQ_BROKEN _PSTL_GLIBCXX_TEST_COMPLEX_BROKEN
#define _PSTL_GLIBCXX_TEST_COMPLEX_TIMES_EQ_BROKEN _PSTL_GLIBCXX_TEST_COMPLEX_BROKEN

#define _PSTL_MSVC_LESS_THAN_CPP20_COMPLEX_CONSTEXPR_BROKEN (_MSC_VER && __cplusplus < 202002L && _MSVC_LANG < 202002L)

#define _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER (__INTEL_LLVM_COMPILER <= _PSTL_TEST_LATEST_INTEL_LLVM_COMPILER)

#define _PSTL_ICC_TEST_COMPLEX_ASIN_MINUS_INF_NAN_BROKEN_SIGNBIT          _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER
#define _PSTL_ICC_TEST_COMPLEX_COSH_MINUS_INF_MINUS_ZERO_BROKEN_SIGNBIT   _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER
#define _PSTL_ICC_TEST_COMPLEX_COSH_MINUS_ZERO_MINUS_ZERO_BROKEN_SIGNBIT  _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER
#define _PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_COMPLEX_PASS_BROKEN_TEST_EDGES _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER
#define _PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_SCALAR_PASS_BROKEN_TEST_EDGES  _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER
#define _PSTL_ICC_TEST_COMPLEX_POW_SCALAR_COMPLEX_PASS_BROKEN_TEST_EDGES  _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER
#define _PSTL_ICC_TEST_COMPLEX_NORM_MINUS_INF_NAN_BROKEN_TEST_EDGES       _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER
#define _PSTL_ICC_TEST_COMPLEX_POLAR_BROKEN_TEST_EDGES                    _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER
#define _PSTL_ICC_TEST_COMPLEX_EXP_BROKEN_TEST_EDGES                     (20240201 < __INTEL_LLVM_COMPILER && __INTEL_LLVM_COMPILER < 20250100)
#define _PSTL_ICC_TEST_COMPLEX_EXP_BROKEN_TEST_EDGES_LATEST              (20240201 < __INTEL_LLVM_COMPILER && __INTEL_LLVM_COMPILER <= _PSTL_TEST_LATEST_INTEL_LLVM_COMPILER)
#define _PSTL_TEST_COMPLEX_ACOS_BROKEN_IN_KERNEL                         (__SYCL_DEVICE_ONLY__ && __INTEL_LLVM_COMPILER < 20250100)
#define _PSTL_TEST_COMPLEX_EXP_BROKEN                                    (__SYCL_DEVICE_ONLY__ && __INTEL_LLVM_COMPILER < 20250100)
#define _PSTL_TEST_COMPLEX_TANH_BROKEN_IN_KERNEL                         (__SYCL_DEVICE_ONLY__ && __INTEL_LLVM_COMPILER < 20250100)


#define _PSTL_ICC_TEST_COMPLEX_ISINF_BROKEN (_MSVC_STL_VERSION && __INTEL_LLVM_COMPILER)
#define _PSTL_ICC_TEST_COMPLEX_ISNAN_BROKEN (_MSVC_STL_VERSION && __INTEL_LLVM_COMPILER)

#define _PSTL_TEST_COMPLEX_OP_BROKEN (_MSVC_STL_VERSION && _MSVC_STL_VERSION <= _PSTL_TEST_LATEST_MSVC_STL_VERSION)

#define _PSTL_TEST_COMPLEX_ACOS_BROKEN  _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_ACOSH_BROKEN _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_ASINH_BROKEN _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_ATANH_BROKEN _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_COS_BROKEN   _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_COSH_BROKEN  _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_LOG10_BROKEN _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_SIN_BROKEN   _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_SINH_BROKEN  _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_TANH_BROKEN  _PSTL_TEST_COMPLEX_OP_BROKEN

#define _PSTL_TEST_COMPLEX_OP_USING_DOUBLE (_MSVC_STL_VERSION && _MSVC_STL_VERSION <= _PSTL_TEST_LATEST_MSVC_STL_VERSION)
#define _PSTL_TEST_COMPLEX_OP_ACOS_USING_DOUBLE               _PSTL_TEST_COMPLEX_OP_USING_DOUBLE
#define _PSTL_TEST_COMPLEX_OP_ACOSH_USING_DOUBLE              _PSTL_TEST_COMPLEX_OP_USING_DOUBLE
#define _PSTL_TEST_COMPLEX_OP_ASIN_USING_DOUBLE               _PSTL_TEST_COMPLEX_OP_USING_DOUBLE
#define _PSTL_TEST_COMPLEX_OP_ASINH_USING_DOUBLE              _PSTL_TEST_COMPLEX_OP_USING_DOUBLE
#define _PSTL_TEST_COMPLEX_OP_LOG_USING_DOUBLE                _PSTL_TEST_COMPLEX_OP_USING_DOUBLE
#define _PSTL_TEST_COMPLEX_OP_LOG10_USING_DOUBLE              _PSTL_TEST_COMPLEX_OP_USING_DOUBLE
#define _PSTL_TEST_COMPLEX_OP_POW_SCALAR_COMPLEX_USING_DOUBLE _PSTL_TEST_COMPLEX_OP_USING_DOUBLE

// oneAPI DPC++ compiler 2025.0.0 and earlier is unable to eliminate a "dead" function call to an undefined function
// within a sycl kernel which MSVC uses to allow comparisons with literal zero without warning
#define _PSTL_TEST_COMPARISON_BROKEN                                                                                   \
    ((__cplusplus >= 202002L || _MSVC_LANG >= 202002L) && _MSVC_STL_VERSION >= 143 && _MSVC_STL_UPDATE >= 202303L &&   \
    __INTEL_LLVM_COMPILER > 0 && __INTEL_LLVM_COMPILER < 20250100)

#define _PSTL_TEST_COMPLEX_TIMES_COMPLEX_BROKEN (_PSTL_TEST_COMPLEX_OP_BROKEN || _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER)
#define _PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN _PSTL_TEST_COMPLEX_OP_BROKEN
#define _PSTL_TEST_COMPLEX_DIV_COMPLEX_BROKEN_IN_INTEL_LLVM_COMPILER _PSTL_TEST_COMPLEX_OP_BROKEN_IN_INTEL_LLVM_COMPILER

#define _PSTL_ICC_TEST_UNDERLYING_TYPE_BROKEN (_GLIBCXX_RELEASE && _GLIBCXX_RELEASE < 9)

// Known limitation:
// Due to specifics of Microsoft* Visual C++, some standard floating-point math functions require device support for double precision.
#define _PSTL_ICC_TEST_COMPLEX_MSVC_MATH_DOUBLE_REQ _MSC_VER

#define _PSTL_CLANG_TEST_COMPLEX_ACOS_IS_NAN_CASE_BROKEN __clang__
#define _PSTL_CLANG_TEST_COMPLEX_ATAN_IS_CASE_BROKEN __clang__
#define _PSTL_CLANG_TEST_COMPLEX_SIN_IS_CASE_BROKEN __clang__

#define TEST_DYNAMIC_SELECTION_AVAILABLE (TEST_DPCPP_BACKEND_PRESENT && __INTEL_LLVM_COMPILER >= 20230000)

// oneAPI DPC++ compiler in 2023.2 release build crashes during optimization of reduce_by_segment.pass.cpp
// with TBB backend.
#if !PSTL_USE_DEBUG && TEST_TBB_BACKEND_PRESENT && defined(__INTEL_LLVM_COMPILER)
#   define _PSTL_ICPX_TEST_RED_BY_SEG_OPTIMIZER_CRASH ((__INTEL_LLVM_COMPILER >= 20230200) && (__INTEL_LLVM_COMPILER <= 20240100))
#else
#   define _PSTL_ICPX_TEST_RED_BY_SEG_OPTIMIZER_CRASH 0
#endif

// If the workaround macro for the 64-bit type bug is not defined by the user, then exclude 64-bit type testing
// in reduce_by_segment.pass.cpp.
// TODO: When a driver fix is provided to resolve this issue, consider altering this macro or checking the driver version at runtime
// of the underlying sycl::device to determine whether to include or exclude 64-bit type tests.
#if !PSTL_USE_DEBUG && defined(__INTEL_LLVM_COMPILER)
#    define _PSTL_ICPX_TEST_RED_BY_SEG_BROKEN_64BIT_TYPES 1
#endif

// Group reduction produces wrong results with multiplication of 64-bit for certain driver versions
// TODO: When a driver fix is provided to resolve this issue, consider altering this macro or checking the driver version at runtime
// of the underlying sycl::device to determine whether to include or exclude 64-bit type tests.
#define _PSTL_GROUP_REDUCTION_MULT_INT64_BROKEN 1

// oneAPI DPC++ compiler 2022.2 an below show an internal compiler error during the backend code generation of
// minmax_element.pass.cpp affecting min_element, max_element, and minmax_element calls.

#define _PSTL_ICPX_TEST_MINMAX_ELEMENT_PASS_BROKEN                                                                     \
    (TEST_DPCPP_BACKEND_PRESENT && __INTEL_LLVM_COMPILER > 0 && __INTEL_LLVM_COMPILER < 20220300)

// oneAPI DPC++ compiler fails to compile the sum of an integer and an iterator to a usm-allocated std vector when
// building for an FPGA device.  This prevents fpga compilation of usm-allocated std vector wrapped in zip, transform,
// and permutation iterators (as a map).
#if (TEST_DPCPP_BACKEND_PRESENT && defined(ONEDPL_FPGA_DEVICE) && defined(__INTEL_LLVM_COMPILER) &&                   \
        __INTEL_LLVM_COMPILER < 20250100)
#    define _PSTL_ICPX_FPGA_TEST_USM_VECTOR_ITERATOR_BROKEN 1
#else
#    define _PSTL_ICPX_FPGA_TEST_USM_VECTOR_ITERATOR_BROKEN 0
#endif

// A specific kernel compilation order causes incorrect results on Windows with the DPCPP backend. For now, we reorder
// the test while the issue is being reported to the compiler team. Once it is resolved, this macro can be removed
// or limited to older compiler versions.
#define _PSTL_RED_BY_SEG_WINDOWS_COMPILE_ORDER_BROKEN                                                                  \
    (_MSC_VER && TEST_DPCPP_BACKEND_PRESENT && __INTEL_LLVM_COMPILER < 20250100)

// Intel(R) oneAPI DPC++/C++ compiler produces 'Unexpected kernel lambda size issue' error
#define _PSTL_LAMBDA_PTR_TO_MEMBER_WINDOWS_BROKEN (_MSC_VER && TEST_DPCPP_BACKEND_PRESENT && __INTEL_LLVM_COMPILER < 20250300)

#if TEST_ONLY_HETERO_POLICIES && !TEST_DPCPP_BACKEND_PRESENT
#    error "TEST_ONLY_HETERO_POLICIES is passed but device backend is not available"
#endif

#endif // _TEST_CONFIG_H
