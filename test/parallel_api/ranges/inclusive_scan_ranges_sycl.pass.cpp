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
    int data1[max_n], data2[max_n], data3[max_n];

    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));
        sycl::buffer<int> B1(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B2(data2, sycl::range<1>(max_n));
        sycl::buffer<int> B3(data3, sycl::range<1>(max_n));

        using namespace oneapi::dpl::experimental;

        auto view = ranges::all_view<int, sycl::access::mode::read>(A);
        auto view_res1 = ranges::all_view<int, sycl::access::mode::write>(B1);
        auto view_res3 = ranges::all_view<int, sycl::access::mode::write>(B3);

        auto exec = TestUtils::get_dpcpp_test_policy();
        using Policy = decltype(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 2>>(exec);
        auto exec3 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 3>>(exec);

        ranges::inclusive_scan(exec, A, view_res1);
        ranges::inclusive_scan(exec2, view, B2, ::std::plus<int>());
        ranges::inclusive_scan(exec3, view, view_res3, ::std::plus<int>(), 100);
    }

    //check result
    int expected1[max_n], expected2[max_n], expected3[max_n];
    ::std::inclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected1);
    ::std::inclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected2, ::std::plus<int>());
    ::std::inclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected3, ::std::plus<int>(), 100);

    EXPECT_EQ_N(expected1, data1, max_n, "wrong effect from inclusive_scan with sycl ranges");
    EXPECT_EQ_N(expected2, data2, max_n, "wrong effect from inclusive_scan with binary operation, sycl ranges");
    EXPECT_EQ_N(expected3, data3, max_n, "wrong effect from inclusive_scan with binary operation and init, sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
