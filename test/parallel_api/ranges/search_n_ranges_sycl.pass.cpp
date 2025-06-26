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
    const int count = 10;
    int data[count] = {0, 1, 2, 3, 4, 4, 4, 7, 8, 9};

    const int n_val = 3;
    const int idx = 4;
    const int val = data[idx];
    int res1 = -1, res2 = -1;

    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(count));

        auto view_a = all_view(A);

        res1 = search_n(CLONE_TEST_POLICY_IDX(exec, 0), view_a, n_val, val, [](auto a, auto b) { return a == b; });
        res2 = search_n(CLONE_TEST_POLICY_IDX(exec, 1), A, n_val, val);
    }

    //check result
    EXPECT_TRUE(res1 == idx, "wrong effect from 'search_n' sycl ranges, with predicate");
    EXPECT_TRUE(res2 == idx, "wrong effect from 'search_n' with sycl buffer");
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
