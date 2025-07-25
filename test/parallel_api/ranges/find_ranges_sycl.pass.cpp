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
#include "support/utils_invoke.h" // CLONE_TEST_POLICY_IDX

#include <iostream>

#if _ENABLE_RANGES_TESTING

template <typename T>
struct IsEqOp
{
    T val;

    template <typename T1>
    bool
    operator()(T1 a) const
    {
        return a == val;
    }
};

struct IsGreatEqThanZeroOp
{
    template <typename T>
    bool
    operator()(T a) const
    {
        return a >= 0;
    }
};

template <typename Policy>
void
test_impl(Policy&& exec)
{
    const int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    const int idx_val = 5;
    const int val = -1;
    data[idx_val] = val;

    int res1 = -1, res2 = - 1, res3 = -1;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));

        auto view = all_view(A);

        res1 = find(CLONE_TEST_POLICY_IDX(exec, 0), view, val); //check passing all_view
        res1 = find(CLONE_TEST_POLICY_IDX(exec, 1), A, val);    //check passing sycl::buffer directly
        res2 = find_if(CLONE_TEST_POLICY_IDX(exec, 2), view, IsEqOp<int>{val});
        res2 = find_if(CLONE_TEST_POLICY_IDX(exec, 3), A, IsEqOp<int>{val});
        res3 = find_if_not(CLONE_TEST_POLICY_IDX(exec, 4), view, IsGreatEqThanZeroOp());
        res3 = find_if_not(CLONE_TEST_POLICY_IDX(exec, 5), A, IsGreatEqThanZeroOp());
    }

    //check result
    EXPECT_TRUE(res1 == idx_val, "wrong effect from 'find' with sycl ranges");
    EXPECT_TRUE(res2 == idx_val, "wrong effect from 'find_if' with sycl ranges");
    EXPECT_TRUE(res3 == idx_val, "wrong effect from 'find_if_not' with sycl ranges");
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
