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
#include <oneapi/dpl/numeric>

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
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data1[max_n];
    int data2[max_n];

    auto lambda = [](auto i) { return i * i; };
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));
        sycl::buffer<int> B(data1, sycl::range<1>(max_n));
        sycl::buffer<int> C(data2, sycl::range<1>(max_n));

        using namespace oneapi::dpl::experimental;

        auto view = ranges::all_view<int, sycl::access::mode::read>(A);
        auto view_res = ranges::all_view<int, sycl::access::mode::write>(B);

        auto exec = TestUtils::get_dpcpp_test_policy();
        using Policy = decltype(exec);
        auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

        ranges::transform_exclusive_scan(exec1, view, view_res, 100, ::std::plus<int>(), lambda);
        ranges::transform_exclusive_scan(exec2, A, C, 100, ::std::plus<int>(), lambda);
    }

    //check result
    int expected[max_n];
    ::std::transform_exclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected, 100, ::std::plus<int>(), lambda);

    EXPECT_EQ_N(expected, data1, max_n, "wrong effect from transform_exclusive_scan with init, sycl ranges");
    EXPECT_EQ_N(expected, data2, max_n, "wrong effect from transform_exclusive_scan with init, sycl buffers");

#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
