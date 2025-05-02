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
    const int count1 = 10;
    int data1[count1] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    const int count2 = 3;
    int data2[count2] = {5, 6, 7};

    const int idx = 5;
    int res1 = -1, res2 = -1;

    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data1, sycl::range<1>(count1));
        sycl::buffer<int> B(data2, sycl::range<1>(count2));

        auto view_a = all_view(A);
        auto view_b = all_view(B);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(TestUtils::default_dpcpp_policy);

        res1 = search(exec, A, view_b);
        res2 = search(make_new_policy<new_kernel_name<Policy, 0>>(exec), view_a, B, [](auto a, auto b) { return a == b; });
    }

    //check result
    EXPECT_TRUE(res1 == idx, "wrong effect from 'search' with sycl ranges");
    EXPECT_TRUE(res2 == idx, "wrong effect from 'search' with predicate, sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
