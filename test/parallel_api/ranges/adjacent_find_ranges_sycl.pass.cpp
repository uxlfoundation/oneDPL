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
    constexpr int n = 10;
    int data[n] = {5, 6, 7, 3, 4, 5, 6, 7, 8, 9};

    constexpr int idx = 5;
    data[idx] = -1, data[idx + 1] = -1;

    int res1 = -1, res2 = -1;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(n));

        res1 = adjacent_find(CLONE_TEST_POLICY_IDX(exec, 0), views::all_read(A));
        res2 = adjacent_find(CLONE_TEST_POLICY_IDX(exec, 1), A, [](auto a, auto b) {return a == b;});
    }

    //check result
    EXPECT_TRUE(res1 == idx, "wrong effect from 'adjacent_find', sycl ranges");
    EXPECT_TRUE(res2 == idx, "wrong effect from 'adjacent_find' with predicate, sycl ranges");
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
