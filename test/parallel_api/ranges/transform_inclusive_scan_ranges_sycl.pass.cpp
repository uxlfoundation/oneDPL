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
#include <oneapi/dpl/numeric>

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"
#include "support/utils_invoke.h" // CLONE_TEST_POLICY_IDX

#include <iostream>

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void
test_impl(Policy&& exec)
{
    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data1[max_n], data2[max_n];

    int init = 100;
    auto pred = TestUtils::Pow2<int>();
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));
        sycl::buffer<int> B1(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B2(data2, sycl::range<1>(max_n));

        using namespace oneapi::dpl::experimental;

        auto view = ranges::all_view<int, sycl::access::mode::read>(A);
        auto view_res1 = ranges::all_view<int, sycl::access::mode::write>(B1);

        ranges::transform_inclusive_scan(CLONE_TEST_POLICY_IDX(exec, 0), A, view_res1, std::plus<int>(), pred);
        ranges::transform_inclusive_scan(CLONE_TEST_POLICY_IDX(exec, 1), view, B2, std::plus<int>(), pred, init);
    }

    //check result
    int expected1[max_n], expected2[max_n];
    std::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected1, std::plus<int>(), pred);
    std::transform_inclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected2, std::plus<int>(), pred, init);

    EXPECT_EQ_N(expected1, data1, max_n, "wrong effect from transform_inclusive_scan, sycl ranges");
    EXPECT_EQ_N(expected2, data2, max_n, "wrong effect from transform_inclusive_scan with init, sycl ranges");
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
