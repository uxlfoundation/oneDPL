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

#include <algorithm>  // std::is_sorted
#include <functional> // std::greater

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void
test_impl(Policy&& exec)
{
    const int max_n = 10;
    int data1[max_n] = {0, 1, 2, -1, 4, 5, 6, 7, 8, 9};
    int data2[max_n] = {0, 1, 2, -1, 4, 5, -6, 7, 8, 9};

    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));

        stable_sort(CLONE_TEST_POLICY_IDX(exec, 0), A); //check passing sycl buffer directly
        stable_sort(CLONE_TEST_POLICY_IDX(exec, 1), all_view<int, sycl::access::mode::read_write>(B), std::greater<int>());
    }

    //check result
    bool res1 = std::is_sorted(data1, data1 + max_n);
    EXPECT_TRUE(res1, "wrong effect from 'stable_sort' with sycl ranges");

    bool res2 = std::is_sorted(data2, data2 + max_n, std::greater<int>());
    EXPECT_TRUE(res2, "wrong effect from 'stable_sort with comparator' with sycl ranges");
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
