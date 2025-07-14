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
    int data1[max_n] = {-1, 1, -1, 3, 4, 5, 6, -1, 8, 9};
    int data2[max_n] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};

    auto lambda = [](auto i) { return i % 2 == 0; };

    bool res1 = false, res2 = false, res3 = false;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));

        res1 = any_of(CLONE_TEST_POLICY_IDX(exec, 0), views::all(A), lambda);
        res2 = all_of(CLONE_TEST_POLICY_IDX(exec, 1), B, lambda);
        res3 = none_of(CLONE_TEST_POLICY_IDX(exec, 2), B, [](auto i) { return i == -1; });
    }

    EXPECT_TRUE(res1, "wrong result from any_of with sycl ranges");
    EXPECT_TRUE(res2, "wrong result from all_of with sycl ranges");
    EXPECT_TRUE(res3, "wrong result from none_of with sycl ranges");
}
#endif // _ENABLE_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });
#endif
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
