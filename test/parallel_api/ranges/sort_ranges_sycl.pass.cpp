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

#include <cstddef>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

#if _ENABLE_RANGES_TESTING
struct get_from_pair_fo
{
    template <typename T>
    auto
    operator()(T&& t) const
    {
        return std::get<1>(std::forward<T>(t));
    }
};

template <typename Policy>
void
test_impl(Policy&& exec)
{
    const int max_n = 10;
    int data1[max_n] = {0, 1, 2, -1, 4, 5, 6, 7, 8, 9};
    int data2[max_n] = {0, 1, 2, -1, 4, 5, 6, 7, 8, 9};

    using namespace oneapi::dpl::experimental::ranges;

    {
        sycl::buffer<int> A(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));

        sort(CLONE_TEST_POLICY_IDX(exec, 0), A); //check passing sycl buffer directly
        sort(CLONE_TEST_POLICY_IDX(exec, 1), all_view<int, sycl::access::mode::read_write>(B), std::greater<int>());
    }

    //check result
    bool res1 = std::is_sorted(data1, data1 + max_n);
    EXPECT_TRUE(res1, "wrong effect from 'sort' with sycl ranges");

    bool res2 = std::is_sorted(data2, data2 + max_n, std::greater<int>());
    EXPECT_TRUE(res2, "wrong effect from 'sort with comparator' with sycl ranges");

    //test with random number and projection usage
    std::default_random_engine gen{std::random_device{}()};
    std::uniform_real_distribution<float> dist(0.0, 100.0);

    constexpr std::size_t N = 1 << 20;
    std::vector<int> keys(N);
    std::generate(keys.begin(), keys.end(), [&] { return dist(gen); });
    std::vector<int> values(keys);

    {
        sycl::buffer<int> A(values.begin(), values.end());
        A.set_final_data(values.begin());
        A.set_write_back(true);
        sycl::buffer<int> B(keys.begin(), keys.end());
        B.set_final_data(keys.begin());
        B.set_write_back(true);

        sort(CLONE_TEST_POLICY_IDX(exec, 2), zip_view(views::all(A), views::all(B)), std::less{}, get_from_pair_fo{});
    }
    bool res3 = std::is_sorted(values.begin(), values.end(), std::less{});
    EXPECT_TRUE(res3, "wrong effect from 'sort by key'");
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
