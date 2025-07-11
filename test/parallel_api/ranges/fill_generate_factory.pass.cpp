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
template <typename T>
struct get_const_fo
{
    T val;

    T operator()() const
    {
        return val;
    }
};

template <typename Policy>
void
test_impl(Policy&& exec)
{
    constexpr int max_n = 10;
    int expected1[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int expected2[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto lambda_pow_2 = TestUtils::Pow2<int>();
    auto lambda_eq_1 = TestUtils::IsEqualTo<int>{1};

    using namespace oneapi::dpl::experimental;

    auto view1 = ranges::views::fill(-1, max_n) | ranges::views::transform(lambda_pow_2);
    auto res1 = std::all_of(view1.begin(), view1.end(), lambda_eq_1);

    auto view2 = ranges::views::generate(get_const_fo<int>{-1}, max_n) | ranges::views::transform(lambda_pow_2);
    auto res2 = std::all_of(view2.begin(), view2.end(), lambda_eq_1);

    //check result
    EXPECT_TRUE(res1, "wrong result from fill factory");
    EXPECT_TRUE(res2, "wrong result from generate factory");

    //checks on a device
    {
        sycl::buffer<int> A(expected1, sycl::range<1>(max_n));
        sycl::buffer<int> B(expected2, sycl::range<1>(max_n));

        ranges::copy(CLONE_TEST_POLICY_IDX(exec, 0), view1, A);
        ranges::copy(CLONE_TEST_POLICY_IDX(exec, 1), view2, B);
    }

    auto res3 = std::all_of(expected1, expected1 + max_n, lambda_eq_1);
    auto res4 = std::all_of(expected2, expected2 + max_n, lambda_eq_1);

    //check result
    EXPECT_TRUE(res3, "wrong result from fill factory on a device");
    EXPECT_TRUE(res4, "wrong result from generate factory on a device");
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
