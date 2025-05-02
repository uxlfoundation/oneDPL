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
    constexpr int n = 10, n_exp = 6;
    int data[n] = {1, 1, 2, 2, 4, 5, 6, 6, 6, 9};
    int expected[n_exp] = {1, 2, 4, 5, 6, 9};

    auto is_equal = [](auto i, auto j) { return i == j; };

    auto exec = TestUtils::default_dpcpp_policy;
    using Policy = decltype(exec);
    auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
    auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

    using namespace oneapi::dpl::experimental::ranges;

    sycl::buffer<int> A(n);
    sycl::buffer<int> B(n);
    sycl::buffer<int> C(data, sycl::range<1>(n));
    
    auto res1 = unique_copy(exec1, views::all_read(C), A);
    auto res2 = unique_copy(exec2, views::all_read(C), views::all_write(B), is_equal);

    //check result
    EXPECT_TRUE(res1 == n_exp, "wrong return result from unique_copy, sycl ranges");
    EXPECT_TRUE(res2 == n_exp, "wrong return result from unique_copy with predicate, sycl ranges");

    EXPECT_EQ_N(expected, views::host_all(A).begin(), n_exp, "wrong effect from unique_copy, sycl ranges");
    EXPECT_EQ_N(expected, views::host_all(B).begin(), n_exp, "wrong effect from unique_copy with predicate, sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
