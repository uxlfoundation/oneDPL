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

#if _ENABLE_STD_RANGES_TESTING

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include <vector>
#include <ranges>

#endif // _ENABLE_STD_RANGES_TESTING

#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

template <typename Policy>
void test_count(Policy policy)
{
    std::vector<int> v = {0, 1, 2, 3, 4, 5};
    TestUtils::MinimalisticRange r{v};
    auto count = oneapi::dpl::ranges::count(policy, r, 3);
    std::string msg = "wrong return value from count, " + std::string(typeid(Policy).name());
    EXPECT_EQ(count, 1, msg.c_str());
}

template <typename Policy>
void test_merge(Policy policy)
{
    std::vector<int> v1 = {0, 2, 4, 6, 8, 10};
    std::vector<int> v2 = {1, 3, 5, 7, 9, 11};
    std::vector<int> v3_expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<int> v3(12, 42);
    TestUtils::MinimalisticRange r1{v1};
    TestUtils::MinimalisticRange r2{v2};
    TestUtils::MinimalisticRange r3{v3};

    oneapi::dpl::ranges::merge(policy, r1, r2, r3);
    std::string msg = "wrong effect from merge, " + std::string(typeid(Policy).name());
    EXPECT_EQ_N(v3_expected.data(), r3.begin(), v3_expected.size(), msg.c_str());
}

#endif // _ENABLE_STD_RANGES_TESTING

int main()
{

#if _ENABLE_STD_RANGES_TESTING
    test_count(oneapi::dpl::execution::seq);
    test_count(oneapi::dpl::execution::unseq);
    test_count(oneapi::dpl::execution::par);
    test_count(oneapi::dpl::execution::par_unseq);
#if TEST_DPCPP_BACKEND_PRESENT
    test_count(TestUtils::get_dpcpp_test_policy());
#endif

    test_merge(oneapi::dpl::execution::seq);
    test_merge(oneapi::dpl::execution::unseq);
    test_merge(oneapi::dpl::execution::par);
    test_merge(oneapi::dpl::execution::par_unseq);
#if TEST_DPCPP_BACKEND_PRESENT
    test_merge(TestUtils::get_dpcpp_test_policy());
#endif

#endif // _ENABLE_STD_RANGES_TESTING
    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
