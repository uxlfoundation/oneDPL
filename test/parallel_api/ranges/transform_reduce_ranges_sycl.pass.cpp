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

    auto lambda1 = [](auto val) { return val = val * val; };

    auto res1 = -1, res2 = -1, res3 = -1;
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));

        auto view = oneapi::dpl::experimental::ranges::all_view<int, sycl::access::mode::read>(A);

        auto exec = TestUtils::get_dpcpp_test_policy();
        using Policy = decltype(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 2>>(exec);
        auto exec3 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 3>>(exec);

        res1 = oneapi::dpl::experimental::ranges::transform_reduce(exec, A, view, 0);
        res2 = oneapi::dpl::experimental::ranges::transform_reduce(exec2, view, A, 0, ::std::plus<int>(), ::std::multiplies<int>());
        res3 = oneapi::dpl::experimental::ranges::transform_reduce(exec3, view, 0, ::std::plus<int>(), lambda1);
    }

    //check result
    auto expected1 = ::std::inner_product(data, data + max_n, data, 0);
    auto expected2 = ::std::inner_product(data, data + max_n, data, 0, ::std::plus<int>(), ::std::multiplies<int>());

    //the name nano::ranges::views::all is not injected into oneapi::dpl::experimental::ranges namespace
    auto data_view = __nanorange::nano::views::all(data) | oneapi::dpl::experimental::ranges::views::transform(lambda1);
    auto expected3 = ::std::accumulate(data_view.begin(), data_view.end(), 0);

    EXPECT_EQ(expected1, res1, "wrong effect from transform_reduce1 with sycl ranges");
    EXPECT_EQ(expected2, res2, "wrong effect from transform_reduce2 with sycl ranges");
    EXPECT_EQ(expected3, res3, "wrong effect from transform_reduce3 with sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
