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
#include <oneapi/dpl/algorithm>

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void
test_impl(Policy&& exec)
{
    constexpr int max_n = 10;
    constexpr int new_val = -1;
    auto pred = TestUtils::IsEven<int>();

    using namespace oneapi::dpl::experimental::ranges;

    sycl::buffer<int> A(max_n);

    auto src = views::iota(0, max_n);
    auto res = replace_copy_if(std::forward<Policy>(exec), src, A, pred, new_val);

    //check result
    int expected[max_n];
    auto res_exp = ::std::replace_copy_if(src.begin(), src.end(), expected, pred, new_val) - expected;

    EXPECT_EQ(res_exp, res, "wrong result from replace_copy_if");
    EXPECT_EQ_N(expected, views::host_all(A).begin(), max_n, "wrong effect from replace_copy_if");
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
