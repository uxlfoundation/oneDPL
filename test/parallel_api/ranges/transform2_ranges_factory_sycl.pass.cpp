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
    int data3[max_n];

    auto lambda1 = [](auto i) { return i * i; };
    auto lambda2 = [](auto i, auto j) { return i + j; };

    {
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));
        sycl::buffer<int> C(data3, sycl::range<1>(max_n));

        auto view = oneapi::dpl::experimental::ranges::iota_view(0, max_n) | oneapi::dpl::experimental::ranges::views::transform(lambda1);
        auto range_res = oneapi::dpl::experimental::ranges::all_view<int, sycl::access::mode::write>(B);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(exec);
        auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

        oneapi::dpl::experimental::ranges::transform(exec1, view, view, range_res, lambda2);
        oneapi::dpl::experimental::ranges::transform(exec2, view, view, C, lambda2); //check passing sycl buffer
    }

    //check result
    int expected[max_n];
    ::std::transform(data, data + max_n, expected, lambda1);
    ::std::transform(expected, expected + max_n, expected, expected, lambda2);

    EXPECT_EQ_N(expected, data2, max_n, "wrong effect from transform2 with sycl ranges");
    EXPECT_EQ_N(expected, data3, max_n, "wrong effect from transform2 with sycl buffer");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
