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

#include <vector>
#include <iostream>
#include <iterator>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    using T = int;

    auto lambda1 = [](T val) -> bool { return val % 2 == 0; };
    auto lambda2 = [](T val) -> bool { return val % 3 == 0; };
    ::std::vector<T> data = {2, 5, 2, 4, 2, 0, 6, -7, 7, 3};

    ::std::vector<T> in(data);
    ::std::vector<T>::difference_type in_end_n;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<T> A(in.data(), sycl::range<1>(in.size()));

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(exec);
        auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

        in_end_n = remove_if(exec1, A, lambda1); //check passing a buffer
        in_end_n = remove_if(exec2, views::all(A) | views::take(in_end_n), lambda2); //check passing a view
    }

    //check result
    ::std::vector<T> exp(data);
    auto exp_end = ::std::remove_if(exp.begin(), exp.end(), lambda1);
    exp_end = ::std::remove_if(exp.begin(), exp_end, lambda2);

    EXPECT_TRUE(::std::distance(exp.begin(), exp_end) == in_end_n, "wrong effect from remove with sycl ranges");
    EXPECT_EQ_N(exp.begin(), in.begin(), in_end_n, "wrong effect from remove with sycl ranges");
#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}

