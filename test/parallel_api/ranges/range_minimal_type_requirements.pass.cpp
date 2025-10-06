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
#include <functional> // for std::invoke

#endif // _ENABLE_STD_RANGES_TESTING

#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

template <typename Policy>
void test_count(Policy policy)
{
    std::vector<int> v = {0, 1, 2, 3, 4, 5};

    TestUtils::MinimalisticRange r{v.begin(), v.end()};
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

    TestUtils::MinimalisticRange r1{v1.begin(), v1.end()};
    TestUtils::MinimalisticRange r2{v2.begin(), v2.end()};
    TestUtils::MinimalisticRange r3{v3.begin(), v3.end()};

    oneapi::dpl::ranges::merge(policy, r1, r2, r3);
    std::string msg = "wrong effect from merge, " + std::string(typeid(Policy).name());
    EXPECT_EQ_N(v3_expected.data(), r3.begin(), v3_expected.size(), msg.c_str());
}

template <typename Policy>
void test_copy_if(Policy policy)
{
    std::vector<int> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> v3(6);
    std::vector<int> v3_expected = {0, 2, 4, 6, 8, 10};

    TestUtils::MinimalisticRange r1{v1.begin(), v1.end()};
    TestUtils::MinimalisticRange r3{v3.begin(), v3.end()};

    oneapi::dpl::ranges::copy_if(policy, r1, r3, [](int x) { return x % 2 == 0; });

    std::string msg = "wrong effect from copy_if, " + std::string(typeid(Policy).name());
    EXPECT_EQ_N(v3_expected.data(), r3.begin(), v3_expected.size(), msg.c_str());
}

template <typename Policy>
void test_transform(Policy policy)
{
    std::vector<int> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> v2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> v3(11);
    std::vector<int> v3_expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

    TestUtils::MinimalisticRange r1{v1.begin(), v1.end()};
    TestUtils::MinimalisticRange r2{v2.begin(), v2.end()};
    TestUtils::MinimalisticRange r3{v3.begin(), v3.end()};

    oneapi::dpl::ranges::transform(policy, r1, r2, r3, [](int x1, int x2) { return x1 + x2; });

    std::string msg = "wrong effect from transform, " + std::string(typeid(Policy).name());
    EXPECT_EQ_N(v3_expected.data(), r3.begin(), v3_expected.size(), msg.c_str());
}

template <typename Algorithm>
void call_test_algo(Algorithm&& algo)
{
    std::invoke(algo, oneapi::dpl::execution::seq);
    std::invoke(algo, oneapi::dpl::execution::unseq);
    std::invoke(algo, oneapi::dpl::execution::par);
    std::invoke(algo, oneapi::dpl::execution::par_unseq);
#if TEST_DPCPP_BACKEND_PRESENT
    std::invoke(algo, TestUtils::get_dpcpp_test_policy());
#endif
}

#endif // _ENABLE_STD_RANGES_TESTING

int main()
{
#if _ENABLE_STD_RANGES_TESTING

    call_test_algo([](auto policy) { test_count(policy); });
    call_test_algo([](auto policy) { test_merge(policy); });
    call_test_algo([](auto policy) { test_copy_if(policy); });
    call_test_algo([](auto policy) { test_transform(policy); });

#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
