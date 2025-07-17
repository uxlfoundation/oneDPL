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

#include <numeric> // std::accumulate, std::inner_product
#include <iostream>

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void
test_impl(Policy&& exec)
{
    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    using namespace oneapi::dpl::experimental::ranges;
    auto res1 = -1, res2 = -1, res3 = -1;
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));

        auto view = all_view<int, sycl::access::mode::read>(A);

        res1 = reduce(CLONE_TEST_POLICY_IDX(exec, 0), A);
        res2 = reduce(CLONE_TEST_POLICY_IDX(exec, 1), view, 100);
        res3 = reduce(CLONE_TEST_POLICY_IDX(exec, 2), view, 100, std::plus<int>());
    }

    //check result
    auto expected1 = std::accumulate(data, data + max_n, 0);
    auto expected2 = std::accumulate(data, data + max_n, 100);
    auto expected3 = expected2;

    EXPECT_EQ(expected1, res1, "wrong effect from reduce with sycl ranges");
    EXPECT_EQ(expected2, res2, "wrong effect from reduce with init, sycl ranges");
    EXPECT_EQ(expected3, res3, "wrong effect from reduce with init and binary operation, sycl ranges");
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
