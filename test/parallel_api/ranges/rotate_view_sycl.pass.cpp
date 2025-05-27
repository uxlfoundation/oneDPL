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

#include "support/test_config.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    int data[max_n]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int expected[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int rotate_val = 6;

    using namespace oneapi::dpl::experimental;
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));
        sycl::buffer<int> B(expected, sycl::range<1>(max_n));
        ranges::copy(TestUtils::dpcpp_policy(), 
                     ranges::views::all_read(A) | ranges::views::rotate(rotate_val), ranges::views::all_write(B));
    }

    //check result
    ::std::rotate(data, data + rotate_val, data + max_n);

    EXPECT_EQ_N(expected, data, max_n, "wrong result from rotate view on a device");

#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
