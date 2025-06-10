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

// This file contains SYCL-specific macros and abstractions to support different versions of SYCL library

#ifndef _UTILS_SYCL_DEFS_H
#define _UTILS_SYCL_DEFS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#if ONEDPL_FPGA_DEVICE
#    include <sycl/ext/intel/fpga_extensions.hpp>
#endif // ONEDPL_FPGA_DEVICE

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define TEST_LIBSYCL_VERSION                                                                                    \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#else
#    define TEST_LIBSYCL_VERSION 0
#endif

namespace TestUtils
{

template <sycl::usm::alloc alloc_type>
constexpr std::size_t
uniq_kernel_index()
{
    return static_cast<std::underlying_type_t<sycl::usm::alloc>>(alloc_type);
}

template <typename Op, std::size_t CallNumber>
struct unique_kernel_name;

template <typename Policy, int idx>
using new_kernel_name = unique_kernel_name<typename std::decay_t<Policy>::kernel_name, idx>;

} /* namespace TestUtils */

#endif //  _UTILS_SYCL_DEFS_H
