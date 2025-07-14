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
#include "support/utils_invoke.h" // CLONE_TEST_POLICY

#include <iostream>

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void
test_impl(Policy&& exec)
{
    const int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    const int idx_val = 5;
    const int val = -1;
    data[idx_val] = val;
    const int idx_max = max_n - 1;

    int res1 = -1, res2 = - 1, res3 = -1, res4 = -1, res5 = -1;
    std::pair<int, int> res_minmax1(-1, -1);
    std::pair<int, int> res_minmax2(-1, -1);

    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));

        auto view = all_view(A);

        //min element
        res1 = min_element(CLONE_TEST_POLICY_IDX(exec, 0), A);
        res2 = min_element(CLONE_TEST_POLICY_IDX(exec, 1), view, std::less<int>());
        res3 = min_element(CLONE_TEST_POLICY_IDX(exec, 2), view | views::take(1));

        //max_element
        res4 = max_element(CLONE_TEST_POLICY_IDX(exec, 3), A);
        res5 = max_element(CLONE_TEST_POLICY_IDX(exec, 4), view, std::less<int>());

        res_minmax1 = minmax_element(CLONE_TEST_POLICY_IDX(exec, 5), A);
        res_minmax2 = minmax_element(CLONE_TEST_POLICY_IDX(exec, 6), view, std::less<int>());
    }

    //check result
    EXPECT_TRUE(res1 == idx_val, "wrong effect from 'min_element', sycl ranges");
    EXPECT_TRUE(res2 == idx_val, "wrong effect from 'min_element' with predicate,  sycl ranges");
    EXPECT_TRUE(res3 == 0, "wrong effect from 'min_element' with trivial sycl ranges");

    EXPECT_TRUE(res4 == idx_max, "wrong effect from 'max_element', sycl ranges");
    EXPECT_TRUE(res5 == idx_max, "wrong effect from 'max_element' with predicate,  sycl ranges");

    EXPECT_TRUE(res_minmax1.first == idx_val && res_minmax1.second == idx_max, "wrong effect from 'minmax_element', sycl ranges");
    EXPECT_TRUE(res_minmax2.first == idx_val && res_minmax2.second == idx_max, "wrong effect from 'minmax_element' with predicate, sycl ranges");
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
