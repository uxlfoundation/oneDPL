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

    auto pred = [](auto i) { return i % 2 == 0; };

    using namespace oneapi::dpl::experimental::ranges;

    sycl::buffer<int> A(max_n);
    sycl::buffer<int> B(max_n);
    sycl::buffer<int> C(max_n);

    auto src = views::iota(0, max_n);

    auto res1 = copy_if(CLONE_TEST_POLICY_IDX(exec, 0), src, A, pred);
    auto res2 = remove_copy_if(CLONE_TEST_POLICY_IDX(exec, 1), src, views::all_write(B), pred);
    auto res3 = remove_copy(CLONE_TEST_POLICY_IDX(exec, 2), src, views::all_write(C), 0);

    EXPECT_TRUE(res1 == 5, "wrong return result from copy_if with sycl buffer");
    EXPECT_TRUE(res2 == 5, "wrong return result from remove_copy_if with sycl ranges");
    EXPECT_TRUE(res3 == 9, "wrong return result from remove_copy with sycl ranges");

    //check result
    int expected[max_n];

    std::copy_if(src.begin(), src.end(), expected, pred);
    EXPECT_EQ_N(expected, views::host_all(A).begin(), res1, "wrong effect from copy_if with sycl ranges");

    std::remove_copy_if(src.begin(), src.end(), expected, pred);
    EXPECT_EQ_N(expected, views::host_all(B).begin(), res2, "wrong effect from remove_copy_if with sycl ranges");

    std::remove_copy(src.begin(), src.end(), expected, 0);
    EXPECT_EQ_N(expected, views::host_all(C).begin(), res3, "wrong effect from remove_copy with sycl ranges");
}
#endif //_ENABLE_RANGES_TESTING

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
