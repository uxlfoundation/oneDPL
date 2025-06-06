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

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#if _ENABLE_RANGES_TESTING
#include _PSTL_TEST_HEADER(ranges)
#endif

#include "support/utils.h"

#include <iostream>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data2[max_n];

    auto lambda1 = [](auto i) { return i * i; };
    auto lambda2 = [](auto i, auto j) { return i + j; };

    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));

        auto sv = oneapi::dpl::experimental::ranges::all_view(A);
        auto view = oneapi::dpl::experimental::ranges::views::reverse(sv) | oneapi::dpl::experimental::ranges::views::transform(lambda1);

        auto range_res = oneapi::dpl::experimental::ranges::all_view<int, sycl::access::mode::write>(B);
        oneapi::dpl::experimental::ranges::transform(TestUtils::get_dpcpp_test_policy(), view, view, range_res, lambda2);
    }

    //check result
    int expected[max_n];
    ::std::reverse(data, data + max_n);
    ::std::transform(data, data + max_n, expected, lambda1);
    ::std::transform(expected, expected + max_n, expected, expected, lambda2);

    EXPECT_EQ_N(expected, data2, max_n, "wrong effect from transform2 with sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
