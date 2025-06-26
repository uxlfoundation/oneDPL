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
#include "support/utils_invoke.h" // CREATE_NEW_POLICY

#include <iostream>

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void
test_impl(Policy&& exec)
{
    const int max_n = 10;
    int data1[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data2[max_n] = {0, 1, 2, -1, 4, 5, 6, 7, 8, 9};

    bool res1 = false;
    bool res2 = false;
    bool res3 = false;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));

        res1 = is_sorted(CLONE_TEST_POLICY_IDX(exec, 0), all_view(A));
        res2 = is_sorted(CLONE_TEST_POLICY_IDX(exec, 1), B);
        res3 = is_sorted(CLONE_TEST_POLICY_IDX(exec, 2), A, [](auto a, auto b) { return a > b;});
    }

    //check result
    EXPECT_TRUE(res1, "wrong effect from 'is_sorted' with sycl ranges (sorted)");
    EXPECT_TRUE(!res2, "wrong effect from 'is_sorted' with sycl ranges (unsorted)");
    EXPECT_TRUE(!res3, "wrong effect from 'is_sorted', sycl ranges, with predicate (unsorted)");
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
