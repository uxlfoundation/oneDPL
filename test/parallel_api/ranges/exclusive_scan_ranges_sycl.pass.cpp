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
    int data1[max_n], data2[max_n];

    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));
        sycl::buffer<int> B1(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B2(data2, sycl::range<1>(max_n));

        using namespace TestUtils;
        using namespace oneapi::dpl::experimental;

        auto view = ranges::all_view<int, sycl::access::mode::read>(A);
        auto view_res1 = ranges::all_view<int, sycl::access::mode::write>(B1);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(TestUtils::default_dpcpp_policy);

        ranges::exclusive_scan(exec, A, view_res1, 100);
        ranges::exclusive_scan(make_new_policy<new_kernel_name<Policy, 0>>(exec), view, B2, 100, ::std::plus<int>());
    }

    //check result
    int expected1[max_n], expected2[max_n];
    ::std::exclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected1, 100);
    ::std::exclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected2, 100, ::std::plus<int>());

    EXPECT_EQ_N(expected1, data1, max_n, "wrong effect from exclusive_scan with init, sycl ranges");
    EXPECT_EQ_N(expected2, data2, max_n, "wrong effect from exclusive_scan with init andbinary operation, sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
