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

#ifndef _UTILS_CONST_H
#define _UTILS_CONST_H

#include <cstddef>

#include "test_config.h"

namespace TestUtils
{
#if TEST_DPCPP_BACKEND_PRESENT
// Defined in utils_invoke.h; declared here so the get_scan_test_*_max_n
// helpers below can be defined without including SYCL.
bool
test_queue_is_cpu();
#endif

#define _SKIP_RETURN_CODE 77

// Test data ranges other than those that start at the beginning of an input.
constexpr ::std::size_t max_n = 100000;

// Default caps used by scan-based tests (scan, copy_if, unique, partition,
// set_*, etc.). The runtime helpers below (defined in utils_sycl.h when the
// dpcpp backend is present) shrink these when the test queue targets a CPU
// device, where the default sizes are too slow.
constexpr std::size_t scan_test_max_n_default = 100000;
constexpr std::size_t scan_test_set_max_n_default = 100000;
constexpr std::size_t scan_test_unique_max_n_default = 1000000;

// Shrunken caps applied for CPU+dpcpp runs. Sized to still exercise
// multi-work-group code paths in the geometric n walk used by the tests.
constexpr std::size_t scan_test_max_n_cpu = 32000;
constexpr std::size_t scan_test_set_max_n_cpu = 24000;
constexpr std::size_t scan_test_unique_max_n_cpu = 32000;

inline std::size_t
get_scan_test_max_n()
{
#if TEST_DPCPP_BACKEND_PRESENT
    if (test_queue_is_cpu())
        return scan_test_max_n_cpu;
#endif
    return scan_test_max_n_default;
}

inline std::size_t
get_scan_test_set_max_n()
{
#if TEST_DPCPP_BACKEND_PRESENT
    if (test_queue_is_cpu())
        return scan_test_set_max_n_cpu;
#endif
    return scan_test_set_max_n_default;
}

inline std::size_t
get_scan_test_unique_max_n()
{
#if TEST_DPCPP_BACKEND_PRESENT
    if (test_queue_is_cpu())
        return scan_test_unique_max_n_cpu;
#endif
    return scan_test_unique_max_n_default;
}

// All these offset consts used for indirect testing of calculation an offset parameter
// (as a result dpl::begin(buf) + offset) for further passing within sycl::accessor constructor.
constexpr ::std::size_t inout1_offset = 3;
constexpr ::std::size_t inout2_offset = 5;
constexpr ::std::size_t inout3_offset = 7;
constexpr ::std::size_t inout4_offset = 9;

} /* namespace TestUtils */

#endif // _UTILS_CONST_H
