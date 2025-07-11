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
    const int count1 = 10;
    int data1[count1] = {5, 6, 7, 3, 4, 5, 6, 7, 8, 9};

    const int count2 = 3;
    int data2[count2] = {5, 6, 7};

    const int idx1 = 5;
    const int idx2 = 7;

    int res1 = -1;
    int res2 = -1;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data1, sycl::range<1>(count1));
        sycl::buffer<int> B(data2, sycl::range<1>(count2));

        auto view_a = all_view(A);
        auto view_b = all_view(B);

        res1 = find_end(CLONE_TEST_POLICY_IDX(exec, 0), view_a, view_b);
        res2 = find_end(CLONE_TEST_POLICY_IDX(exec, 1), A, B, TestUtils::IsNotEqual<int>()); //check passing sycl buffer directly
    }

    //check result
    EXPECT_TRUE(res1 == idx1, "wrong effect from 'find_end' with sycl ranges");
    EXPECT_TRUE(res2 == idx2, "wrong effect from 'find_end', sycl ranges, with predicate");
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
