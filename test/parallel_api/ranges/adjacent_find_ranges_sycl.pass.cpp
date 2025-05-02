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
    constexpr int n = 10;
    int data[n] = {5, 6, 7, 3, 4, 5, 6, 7, 8, 9};

    constexpr int idx = 5;
    data[idx] = -1, data[idx + 1] = -1;

    int res1 = -1, res2 = -1;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(n));

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(exec);
        auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

        res1 = adjacent_find(exec1, views::all_read(A));
        res2 = adjacent_find(exec2, A, [](auto a, auto b) {return a == b;});
    }

    //check result
    EXPECT_TRUE(res1 == idx, "wrong effect from 'adjacent_find', sycl ranges");
    EXPECT_TRUE(res2 == idx, "wrong effect from 'adjacent_find' with predicate, sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
