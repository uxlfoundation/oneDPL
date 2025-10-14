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

#include <algorithm> // std::transform
#include <iostream>

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void
test_impl(Policy&& exec)
{
    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data2[max_n];
    int data3[max_n];

    auto pred1 = TestUtils::Pow2<int>();
    auto pred2 = TestUtils::SumWithOp<int>{200};

    using namespace oneapi::dpl::experimental::ranges;

    {
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));
        sycl::buffer<int> C(data3, sycl::range<1>(max_n));

        auto view = iota_view(0, max_n) | views::transform(pred1);
        auto range_res = all_view<int, sycl::access::mode::write>(B);

        transform(CLONE_TEST_POLICY_IDX(exec, 0), view, range_res, pred2);
        transform(CLONE_TEST_POLICY_IDX(exec, 1), view, C, pred2); //check passing sycl buffer
    }

    //check result
    int expected[max_n];
    std::transform(data, data + max_n, expected, pred1);
    std::transform(expected, expected + max_n, expected, pred2);

    EXPECT_EQ_N(expected, data2, max_n, "wrong effect from transform with sycl ranges");
    EXPECT_EQ_N(expected, data3, max_n, "wrong effect from transform with sycl buffer");
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
