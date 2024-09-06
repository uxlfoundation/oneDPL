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

#ifndef _ONEDPL_CONFIG_H
#define _ONEDPL_CONFIG_H

#include "../internal/version_impl.h"
// The version header also defines a few configuration macros used in this file

#if defined(ONEDPL_FPGA_DEVICE)
#    undef _ONEDPL_FPGA_DEVICE
#    define _ONEDPL_FPGA_DEVICE ONEDPL_FPGA_DEVICE
#endif

#if defined(ONEDPL_FPGA_EMULATOR)
#    undef _ONEDPL_FPGA_EMU
#    define _ONEDPL_FPGA_EMU ONEDPL_FPGA_EMULATOR
#endif

#if defined(ONEDPL_USE_PREDEFINED_POLICIES)
#    undef _ONEDPL_PREDEFINED_POLICIES
#    define _ONEDPL_PREDEFINED_POLICIES ONEDPL_USE_PREDEFINED_POLICIES
#elif !defined(_ONEDPL_PREDEFINED_POLICIES)
#    define _ONEDPL_PREDEFINED_POLICIES 1
#endif

// Check availability of parallel backends
#if __has_include(<tbb/tbb.h>)
#    define _ONEDPL_TBB_AVAILABLE 1
#endif
#if ONEDPL_USE_TBB_BACKEND && !_ONEDPL_TBB_AVAILABLE
#    error "Parallel execution policies with oneTBB or Intel(R) TBB support are enabled, but the library is not found"
#endif

#if defined(_OPENMP) && __has_include(<omp.h>)
#    define _ONEDPL_OPENMP_AVAILABLE 1
#endif
// During compilation for a device _OPENMP may not be set, so avoid throwing an error if
// __SYCL_DEVICE_ONLY__ found
#if ONEDPL_USE_OPENMP_BACKEND && !_ONEDPL_OPENMP_AVAILABLE && !defined(__SYCL_DEVICE_ONLY__)
#    error "Parallel execution policies with OpenMP* support are enabled, \
        but OpenMP headers are not found or the compiler does not support OpenMP"
#endif

#if (defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)) &&                                           \
    (__has_include(<sycl/sycl.hpp>) || __has_include(<CL/sycl.hpp>))
#    define _ONEDPL_SYCL_AVAILABLE 1
#endif
#if ONEDPL_USE_DPCPP_BACKEND && !_ONEDPL_SYCL_AVAILABLE
#    error "Device execution policies are enabled, \
        but SYCL* headers are not found or the compiler does not support SYCL"
#endif

// Check the user-defined macro for warnings
#if !defined(_PSTL_USAGE_WARNINGS) && defined(PSTL_USAGE_WARNINGS)
#    define _PSTL_USAGE_WARNINGS PSTL_USAGE_WARNINGS
// Check the internal macro for warnings
#elif !defined(_PSTL_USAGE_WARNINGS)
#    define _PSTL_USAGE_WARNINGS 0
#endif

// Portability "#pragma" definition
#ifdef _MSC_VER
#    define _ONEDPL_PRAGMA(x) __pragma(x)
#else
#    define _ONEDPL_PRAGMA(x) _Pragma(#    x)
#endif

#define _ONEDPL_STRING_AUX(x) #x
#define _ONEDPL_STRING(x) _ONEDPL_STRING_AUX(x)
#define _ONEDPL_STRING_CONCAT(x, y) x #y

// note that when ICC or Clang is in use, _ONEDPL_GCC_VERSION might not fully match
// the actual GCC version on the system.
#define _ONEDPL_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if __clang__
// according to clang documentation, version can be vendor specific
#    define _ONEDPL_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#endif

// Enable SIMD for compilers that support OpenMP 4.0
#if (_OPENMP >= 201307) || __INTEL_LLVM_COMPILER || (__INTEL_COMPILER >= 1600) ||                                      \
    (!defined(__INTEL_LLVM_COMPILER) && !defined(__INTEL_COMPILER) && _ONEDPL_GCC_VERSION >= 40900)
#    define _ONEDPL_PRAGMA_SIMD _ONEDPL_PRAGMA(omp simd)
#    define _ONEDPL_PRAGMA_DECLARE_SIMD _ONEDPL_PRAGMA(omp declare simd)
#    define _ONEDPL_PRAGMA_SIMD_REDUCTION(PRM) _ONEDPL_PRAGMA(omp simd reduction(PRM))
#elif defined(_PSTL_PRAGMA_SIMD)
#    define _ONEDPL_PRAGMA_SIMD _PSTL_PRAGMA_SIMD
#    define _ONEDPL_PRAGMA_DECLARE_SIMD _PSTL_PRAGMA_DECLARE_SIMD
#    define _ONEDPL_PRAGMA_SIMD_REDUCTION(PRM) _PSTL_PRAGMA_SIMD_REDUCTION(PRM)
#else //no simd
#    define _ONEDPL_PRAGMA_SIMD
#    define _ONEDPL_PRAGMA_DECLARE_SIMD
#    define _ONEDPL_PRAGMA_SIMD_REDUCTION(PRM)
#endif //Enable SIMD

// Enable loop unrolling pragmas where supported
#if (__INTEL_LLVM_COMPILER || __INTEL_COMPILER ||                                                                      \
     (!defined(__INTEL_LLVM_COMPILER) && !defined(__INTEL_COMPILER) &&                                                 \
      ((_ONEDPL_GCC_VERSION >= 80000) || (_ONEDPL_CLANG_VERSION >= 30700))))
#    define _ONEDPL_PRAGMA_UNROLL _ONEDPL_PRAGMA(unroll)
#else //no pragma unroll
#    define _ONEDPL_PRAGMA_UNROLL
#endif

#if (__INTEL_LLVM_COMPILER || __INTEL_COMPILER)
#    define _ONEDPL_PRAGMA_FORCEINLINE _ONEDPL_PRAGMA(forceinline)
#elif defined(_PSTL_PRAGMA_FORCEINLINE)
#    define _ONEDPL_PRAGMA_FORCEINLINE _PSTL_PRAGMA_FORCEINLINE
#else
#    define _ONEDPL_PRAGMA_FORCEINLINE
#endif

#if (__INTEL_LLVM_COMPILER >= 20230100 || __INTEL_COMPILER >= 1900)
#    define _ONEDPL_PRAGMA_SIMD_SCAN(PRM) _ONEDPL_PRAGMA(omp simd reduction(inscan, PRM))
#    define _ONEDPL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM) _ONEDPL_PRAGMA(omp scan inclusive(PRM))
#    define _ONEDPL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM) _ONEDPL_PRAGMA(omp scan exclusive(PRM))
#elif defined(_PSTL_PRAGMA_SIMD_SCAN)
#    define _ONEDPL_PRAGMA_SIMD_SCAN(PRM) _PSTL_PRAGMA_SIMD_SCAN(PRM)
#    define _ONEDPL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM) _PSTL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM)
#    define _ONEDPL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM) _PSTL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM)
#else
#    define _ONEDPL_PRAGMA_SIMD_SCAN(PRM)
#    define _ONEDPL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM)
#    define _ONEDPL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM)
#endif

// Required to check if libstdc++ is 5.1.0 or greater
#if defined(__clang__)
#    if __GLIBCXX__ && __has_include(<experimental/any>)
#        define _ONEDPL_LIBSTDCXX_5_OR_GREATER 1
#    else
#        define _ONEDPL_LIBSTDCXX_5_OR_GREATER 0
#    endif // __GLIBCXX__ && __has_include(<experimental/any>)
#else
#    define _ONEDPL_LIBSTDCXX_5_OR_GREATER (__GLIBCXX__ && _ONEDPL_GCC_VERSION >= 50100)
#endif // defined(__clang__)

// Should be defined to 1 for environments with a vendor implementation of C++17 execution policies
#define _ONEDPL_CPP17_EXECUTION_POLICIES_PRESENT                                                                       \
    (_ONEDPL___cplusplus >= 201703L && (_MSC_VER >= 1912 || (_GLIBCXX_RELEASE >= 9 && __GLIBCXX__ >= 20190503)))

#define _ONEDPL_EARLYEXIT_PRESENT (__INTEL_COMPILER >= 1800)
#if (defined(_PSTL_PRAGMA_SIMD_EARLYEXIT) && _PSTL_EARLYEXIT_PRESENT)
#    define _ONEDPL_PRAGMA_SIMD_EARLYEXIT _PSTL_PRAGMA_SIMD_EARLYEXIT
#elif _ONEDPL_EARLYEXIT_PRESENT
#    define _ONEDPL_PRAGMA_SIMD_EARLYEXIT _ONEDPL_PRAGMA(omp simd early_exit)
#else
#    define _ONEDPL_PRAGMA_SIMD_EARLYEXIT
#endif

#define _ONEDPL_MONOTONIC_PRESENT (__INTEL_COMPILER >= 1800)
#if (defined(_PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC) && _PSTL_MONOTONIC_PRESENT)
#    define _ONEDPL_PRAGMA_SIMD_ORDERED_MONOTONIC(PRM) _PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(PRM)
#    define _ONEDPL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(PRM1, PRM2)                                                    \
        _PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(PRM1, PRM2)
#elif _ONEDPL_MONOTONIC_PRESENT
#    define _ONEDPL_PRAGMA_SIMD_ORDERED_MONOTONIC(PRM) _ONEDPL_PRAGMA(omp ordered simd monotonic(PRM))
#    define _ONEDPL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(PRM1, PRM2)                                                    \
        _ONEDPL_PRAGMA(omp ordered simd monotonic(PRM1, PRM2))
#else
#    define _ONEDPL_PRAGMA_SIMD_ORDERED_MONOTONIC(PRM)
#    define _ONEDPL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(PRM1, PRM2)
#endif

#if (_OPENMP >= 201307 || __INTEL_LLVM_COMPILER || __INTEL_COMPILER >= 1900 ||                                         \
     !defined(__INTEL_LLVM_COMPILER) && !defined(__INTEL_COMPILER) && _ONEDPL_GCC_VERSION >= 40900)
#    define _ONEDPL_UDR_PRESENT 1
#else
#    define _ONEDPL_UDR_PRESENT 0
#endif

// TODO: enable UDS on Windows with Intel LLVM-based compiler when it is fixed
#if (__INTEL_LLVM_COMPILER >= 20230100 && !defined(_MSC_VER)) ||                                                       \
    (__INTEL_COMPILER >= 1900 && __INTEL_COMPILER_BUILD_DATE >= 20180626)
#    define _ONEDPL_UDS_PRESENT 1
#else
#    define _ONEDPL_UDS_PRESENT 0
#endif

// Declaration of reduction functor, where
// NAME - the name of the functor
// OP - type of the callable object with the reduction operation
// omp_in - refers to the local partial result
// omp_out - refers to the final value of the combiner operator
// omp_priv - refers to the private copy of the initial value
// omp_orig - refers to the original variable to be reduced
#ifdef _PSTL_PRAGMA_DECLARE_REDUCTION
#    define _ONEDPL_PRAGMA_DECLARE_REDUCTION(NAME, OP) _PSTL_PRAGMA_DECLARE_REDUCTION(NAME, OP)
#else
#    define _ONEDPL_PRAGMA_DECLARE_REDUCTION(NAME, OP)                                                                 \
        _ONEDPL_PRAGMA(omp declare reduction(NAME : OP : omp_out(omp_in)) initializer(omp_priv = omp_orig))
#endif

#if defined(_PSTL_PRAGMA_VECTOR_UNALIGNED) && (__INTEL_COMPILER < 1600)
#    define _ONEDPL_PRAGMA_VECTOR_UNALIGNED _PSTL_PRAGMA_VECTOR_UNALIGNED
#elif (__INTEL_COMPILER >= 1600)
#    define _ONEDPL_PRAGMA_VECTOR_UNALIGNED _ONEDPL_PRAGMA(vector unaligned)
#else
#    define _ONEDPL_PRAGMA_VECTOR_UNALIGNED
#endif

// Check the user-defined macro to use non-temporal stores
#ifndef _PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED
#    if defined(PSTL_USE_NONTEMPORAL_STORES) && (__INTEL_LLVM_COMPILER || __INTEL_COMPILER >= 1600)
#        define _PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED _PSTL_PRAGMA(vector nontemporal)
#    else
#        define _PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED
#    endif
#endif

#if _MSC_VER || __INTEL_COMPILER //the preprocessors don't type a message location
#    define _ONEDPL_PRAGMA_LOCATION __FILE__ ":" _ONEDPL_STRING(__LINE__) ": [Parallel STL message]: "
#else
#    define _ONEDPL_PRAGMA_LOCATION " [Parallel STL message]: "
#endif

#ifndef _PSTL_PRAGMA_MESSAGE_IMPL
#    define _PSTL_PRAGMA_MESSAGE_IMPL(x) _ONEDPL_PRAGMA(message(_ONEDPL_STRING_CONCAT(_ONEDPL_PRAGMA_LOCATION, x)))
#endif

#ifndef _PSTL_PRAGMA_MESSAGE
#    if _PSTL_USAGE_WARNINGS
#        define _PSTL_PRAGMA_MESSAGE(x) _PSTL_PRAGMA_MESSAGE_IMPL(x)
#        define _PSTL_PRAGMA_MESSAGE_POLICIES(x) _PSTL_PRAGMA_MESSAGE_IMPL(x)
#    else
#        define _PSTL_PRAGMA_MESSAGE(x)
#        define _PSTL_PRAGMA_MESSAGE_POLICIES(x)
#    endif
#endif

// Some  C++ standard libraries contain 'exclusive_scan' declaration (version with binary_op)
// w/o "enable_if". So, a call 'exclusive_scan' may be ambiguous in case of a custom policy using.
#define _ONEDPL_EXCLUSIVE_SCAN_WITH_BINARY_OP_AMBIGUITY                                                                \
    (_ONEDPL___cplusplus >= 201703L && __GLIBCXX__ && __GLIBCXX__ > 20190503)

// some algorithms in <numeric> such as 'reduce' were added since libstdc++-9.3, we
// have to provide our own implementation if legacy libstdc++ is in use.
#define _ONEDPL_HAS_NUMERIC_SERIAL_IMPL                                                                                \
    (__GLIBCXX__ && (_GLIBCXX_RELEASE < 9 || (_GLIBCXX_RELEASE == 9 && __GLIBCXX__ < 20200312)))

#if ONEDPL_USE_DPCPP_BACKEND || (!defined(ONEDPL_USE_DPCPP_BACKEND) && _ONEDPL_SYCL_AVAILABLE)
#    define _ONEDPL_BACKEND_SYCL 1
#endif

// if SYCL policy switch on then let's switch hetero policy macro on
#if _ONEDPL_BACKEND_SYCL
#    if _ONEDPL_HETERO_BACKEND
#        undef _ONEDPL_HETERO_BACKEND
#    endif
#    define _ONEDPL_HETERO_BACKEND 1
// Include sycl specific options
// FPGA doesn't support sub-groups
#    if !(_ONEDPL_FPGA_DEVICE)
#        define _USE_SUB_GROUPS 1
#        define _USE_GROUP_ALGOS 1
#    endif

#    define _USE_RADIX_SORT (_USE_SUB_GROUPS && _USE_GROUP_ALGOS)

// Compilation of a kernel is requiried to obtain valid work_group_size
// when target devices are CPU or FPGA emulator. Since CPU and GPU devices
// cannot be distinguished during compilation, the macro is enabled by default.
#    if !defined(_ONEDPL_COMPILE_KERNEL)
#        define _ONEDPL_COMPILE_KERNEL 1
#    endif
#endif

#if !defined(ONEDPL_ALLOW_DEFERRED_WAITING)
#    define ONEDPL_ALLOW_DEFERRED_WAITING 0
#endif

//'present' macros
// shift_left, shift_right; GCC 10; VS 2019 16.1
#define _ONEDPL_CPP20_SHIFT_LEFT_RIGHT_PRESENT                                                                         \
    (_ONEDPL___cplusplus >= 202002L && (_MSC_VER >= 1921 || _GLIBCXX_RELEASE >= 10))

#if _ONEDPL_STD_FEATURE_MACROS_PRESENT
#    define _ONEDPL_CPP20_CONCEPTS_PRESENT (__cpp_concepts >= 201907L && __cpp_lib_concepts >= 202002L)
#    define _ONEDPL_CPP23_TUPLE_LIKE_COMMON_REFERENCE_PRESENT                                                          \
        (_ONEDPL___cplusplus >= 202302L && __cpp_lib_tuple_like >= 202207L)
#    define _ONEDPL_CPP23_RANGES_ZIP_PRESENT (_ONEDPL___cplusplus >= 202302L && __cpp_lib_ranges_zip >= 202110L)
#else
#    define _ONEDPL_CPP20_CONCEPTS_PRESENT 0
#    define _ONEDPL_CPP23_TUPLE_LIKE_COMMON_REFERENCE_PRESENT 0
#    define _ONEDPL_CPP23_RANGES_ZIP_PRESENT 0
#endif

// When C++20 concepts are available, we must use std::tuple as a proxy reference to satisfy iterator concepts, which
// requires the changes to std::tuple in P2321R2 and the tuple-like basic_common_reference specialization in P2165R4.
#define _ONEDPL_CAN_USE_STD_TUPLE_PROXY_ITERATOR                                                                       \
    (!_ONEDPL_CPP20_CONCEPTS_PRESENT ||                                                                                \
     (_ONEDPL_CPP23_RANGES_ZIP_PRESENT && _ONEDPL_CPP23_TUPLE_LIKE_COMMON_REFERENCE_PRESENT))

#if _ONEDPL_CPP20_CONCEPTS_PRESENT
#    define _ONEDPL_CPP20_REQUIRES(req) requires(req)
#else
#    define _ONEDPL_CPP20_REQUIRES(req)
#endif

#define _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT __has_builtin(__builtin_sycl_unique_stable_name)

#if defined(_MSC_VER) && __INTEL_LLVM_COMPILER < 20240100
#    define _ONEDPL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN 1
#else
#    define _ONEDPL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN 0
#endif

// The implementation of std::bit_floor in MS STL does not meet requirements for SYCL device functions
#if defined(_MSC_VER) && (__SYCL_DEVICE_ONLY__ || __SYCL_SINGLE_SOURCE__)
#    define _ONEDPL_STD_BIT_FLOOR_BROKEN 1
#else
#    define _ONEDPL_STD_BIT_FLOOR_BROKEN 0
#endif

#endif // _ONEDPL_CONFIG_H
