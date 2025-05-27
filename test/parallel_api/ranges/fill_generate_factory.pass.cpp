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
    int expected1[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int expected2[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto lambda = [](auto i) { return i * i; };

    using namespace oneapi::dpl::experimental;

    auto view1 = ranges::views::fill(-1, max_n) | ranges::views::transform(lambda);
    auto res1 = std::all_of(view1.begin(), view1.end(), [](auto i) { return i == 1;});

    auto view2 = ranges::views::generate([]() { return -1;}, max_n) | ranges::views::transform(lambda);
    auto res2 = std::all_of(view2.begin(), view2.end(), [](auto i) { return i == 1;});

    //check result
    EXPECT_TRUE(res1, "wrong result from fill factory");
    EXPECT_TRUE(res2, "wrong result from generate factory");

    //checks on a device
    {
        sycl::buffer<int> A(expected1, sycl::range<1>(max_n));
        sycl::buffer<int> B(expected2, sycl::range<1>(max_n));

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(exec);
        auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

        ranges::copy(exec1, view1, A);
        ranges::copy(exec2, view2, B);
    }

    auto res3 = std::all_of(expected1, expected1 + max_n, [](auto i) { return i == 1;});
    auto res4 = std::all_of(expected2, expected2 + max_n, [](auto i) { return i == 1;});

    //check result
    EXPECT_TRUE(res3, "wrong result from fill factory on a device");
    EXPECT_TRUE(res4, "wrong result from generate factory on a device");

#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
