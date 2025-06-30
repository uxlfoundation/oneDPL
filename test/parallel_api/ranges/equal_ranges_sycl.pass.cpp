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
#include "support/utils_invoke.h" // for CLONE_TEST_POLICY macro

#include <iostream>

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void
test_impl(Policy&& exec)
{
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
                          
        res1 = equal(CLONE_TEST_POLICY_IDX(exec, 0), view, B);
        res2 = equal(CLONE_TEST_POLICY_IDX(exec, 1), C, view, std::equal_to<>{});
    }

    //check result
    EXPECT_TRUE(res1, "wrong result from equal with sycl ranges");
    EXPECT_FALSE(res2, "wrong result from equal with sycl ranges");
}
#endif // _ENABLE_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });

#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
