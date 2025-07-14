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
#include "support/utils_invoke.h" // for CLONE_TEST_POLICY macro

#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm> // std::remove_if

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void
test_impl(Policy&& exec)
{
    using T = int;

    auto lambda1 = TestUtils::IsEven<T>();
    auto lambda2 = TestUtils::IsMultipleOf<T>{3};
    std::vector<T> data = {2, 5, 2, 4, 2, 0, 6, -7, 7, 3};

    std::vector<T> in(data);
    std::vector<T>::difference_type in_end_n;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<T> A(in.data(), sycl::range<1>(in.size()));

        in_end_n = remove_if(CLONE_TEST_POLICY_IDX(exec, 0), A, lambda1); //check passing a buffer
        in_end_n = remove_if(CLONE_TEST_POLICY_IDX(exec, 1), views::all(A) | views::take(in_end_n), lambda2); //check passing a view
    }

    //check result
    std::vector<T> exp(data);
    auto exp_end = std::remove_if(exp.begin(), exp.end(), lambda1);
    exp_end = std::remove_if(exp.begin(), exp_end, lambda2);

    EXPECT_EQ(std::distance(exp.begin(), exp_end), in_end_n, "wrong effect from remove with sycl ranges");
    EXPECT_EQ_N(exp.begin(), in.begin(), in_end_n, "wrong effect from remove with sycl ranges");
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

