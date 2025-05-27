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
    int data1[max_n]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data2[max_n]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data3[max_n]     = {0, 1, 2, -1, 4, 5, 6, 7, 8, 9};

    bool res1 = false, res2 = false;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));
        sycl::buffer<int> C(data3, sycl::range<1>(max_n));

        auto view = views::all(A);
                          
        auto exec = TestUtils::dpcpp_policy();
        using Policy = decltype(exec);
        auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);
             
        res1 = equal(exec1, view, B);
        res2 = equal(exec2, C, view, ::std::equal_to<>{});
    }

    //check result
    EXPECT_TRUE(res1, "wrong result from equal with sycl ranges");
    EXPECT_FALSE(res2, "wrong result from equal with sycl ranges");

#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
