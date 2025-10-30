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
#include <memory>     // for std::uninitialized_copy

#endif // _ENABLE_STD_RANGES_TESTING

#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

using TestingType = int;
using TestingVector = std::vector<TestingType>;

struct test_count
{
    template <typename Policy>
    std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        TestingVector v = {0, 1, 2, 3, 4, 5};

        TestUtils::MinimalisticView r1(v.begin(), v.end());

        auto count = oneapi::dpl::ranges::count(policy, r1, 3);

        std::string msg = "wrong return value from count, " + std::string(typeid(Policy).name());
        EXPECT_EQ(count, 1, msg.c_str());
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    std::enable_if_t<!oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        auto queue = policy.queue();

        TestingVector v = {0, 1, 2, 3, 4, 5};

        auto v1_begin = sycl::malloc_shared<TestingType>(v.size(), queue);
        auto v1_end = v1_begin + v.size();

        std::uninitialized_copy(v1_begin, v.data(), v.size() * sizeof(TestingType));

        TestUtils::MinimalisticView r1(v1_begin, v1_end);

        auto count = oneapi::dpl::ranges::count(policy, r1, 3);

        std::string msg = "wrong return value from count, " + std::string(typeid(Policy).name());
        EXPECT_EQ(count, 1, msg.c_str());

        sycl::free(v1_begin, queue);
    }
#endif
};

struct test_merge
{
    template <typename Policy>
    std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        TestingVector v1 = {0, 2, 4, 6, 8, 10};
        TestingVector v2 = {1, 3, 5, 7, 9, 11};
        TestingVector v3_expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        TestingVector v3(v3_expected.size(), 42);

        TestUtils::MinimalisticView r1(v1.begin(), v1.end());
        TestUtils::MinimalisticView r2(v2.begin(), v2.end());
        TestUtils::MinimalisticView r3(v3.begin(), v3.end());

        oneapi::dpl::ranges::merge(policy, r1, r2, r3);

        std::string msg = "wrong effect from merge, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3.begin(), v3_expected.size(), msg.c_str());
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    std::enable_if_t<!oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        TestingVector v1 = {0, 2, 4, 6, 8, 10};
        TestingVector v2 = {1, 3, 5, 7, 9, 11};
        TestingVector v3_expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        TestingVector v3(v3_expected.size(), 42);

        auto queue = policy.queue();

        auto v1_begin = sycl::malloc_shared<TestingType>(v1.size(), queue);
        auto v1_end = v1_begin + v1.size();

        auto v2_begin = sycl::malloc_shared<TestingType>(v2.size(), queue);
        auto v2_end = v2_begin + v2.size();

        auto v3_begin = sycl::malloc_shared<TestingType>(v1.size() + v2.size(), queue);
        auto v3_end = v3_begin + v1.size() + v2.size();

        std::uninitialized_copy(v1_begin, v1.data(), v1.size() * sizeof(TestingType));
        std::uninitialized_copy(v2_begin, v2.data(), v2.size() * sizeof(TestingType));
        std::uninitialized_copy(v3_begin, v3.data(), (v1.size() + v2.size()) * sizeof(TestingType));

        TestUtils::MinimalisticView r1(v1_begin, v1_end);
        TestUtils::MinimalisticView r2(v2_begin, v2_end);
        TestUtils::MinimalisticView r3(v3_begin, v3_end);

        oneapi::dpl::ranges::merge(policy, r1, r2, r3);

        std::string msg = "wrong effect from merge, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3_begin, v3_expected.size(), msg.c_str());

        sycl::free(v1_begin, queue);
        sycl::free(v2_begin, queue);
        sycl::free(v3_begin, queue);
    }
#endif
};

struct test_copy_if
{
    template <typename Policy>
    std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        TestingVector v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        TestingVector v3(6);
        TestingVector v3_expected = {0, 2, 4, 6, 8, 10};

        TestUtils::MinimalisticView r1(v1.begin(), v1.end());
        TestUtils::MinimalisticView r2(v3.begin(), v3.end());

        oneapi::dpl::ranges::copy_if(policy, r1, r2, [](TestingType x) { return x % 2 == 0; });

        std::string msg = "wrong effect from copy_if, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3.begin(), v3_expected.size(), msg.c_str());
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    std::enable_if_t<!oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        auto queue = policy.queue();

        TestingVector v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        TestingVector v3_expected = {0, 2, 4, 6, 8, 10};
        TestingVector v3(v3_expected.size());

        auto v1_begin = sycl::malloc_shared<TestingType>(v1.size(), queue);
        auto v1_end = v1_begin + v1.size();

        auto v3_begin = sycl::malloc_shared<TestingType>(v3.size(), queue);
        auto v3_end = v3_begin + v3.size();

        std::uninitialized_copy(v1_begin, v1.data(), v1.size() * sizeof(TestingType));
        std::uninitialized_copy(v3_begin, v3.data(), v3.size() * sizeof(TestingType));

        TestUtils::MinimalisticView r1(v1_begin, v1_end);
        TestUtils::MinimalisticView r3(v3_begin, v3_end);

        oneapi::dpl::ranges::copy_if(policy, r1, r3, [](TestingType x) { return x % 2 == 0; });

        std::string msg = "wrong effect from copy_if, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3_begin, v3_expected.size(), msg.c_str());

        sycl::free(v1_begin, queue);
        sycl::free(v3_begin, queue);
    }
#endif
};

struct test_transform
{
    template <typename Policy>
    std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        TestingVector v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        TestingVector v2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        TestingVector v3(v1.size());
        TestingVector v3_expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

        TestUtils::MinimalisticView r1(v1.begin(), v1.end());
        TestUtils::MinimalisticView r2(v2.begin(), v2.end());
        TestUtils::MinimalisticView r3(v3.begin(), v3.end());

        oneapi::dpl::ranges::transform(policy, r1, r2, r3, [](TestingType x1, TestingType x2) { return x1 + x2; });

        std::string msg = "wrong effect from transform, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3.begin(), v3_expected.size(), msg.c_str());
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    std::enable_if_t<!oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        auto queue = policy.queue();

        TestingVector v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        TestingVector v2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        TestingVector v3_expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

        auto v1_begin = sycl::malloc_shared<TestingType>(v1.size(), queue);
        auto v1_end = v1_begin + v1.size();

        auto v2_begin = sycl::malloc_shared<TestingType>(v2.size(), queue);
        auto v2_end = v2_begin + v2.size();

        auto v3_begin = sycl::malloc_shared<TestingType>(v3_expected.size(), queue);
        auto v3_end = v3_begin + v3_expected.size();

        TestingVector v3(v1.size());
        std::uninitialized_copy(v1_begin, v1.data(), sizeof(TestingType) * v1.size());
        std::uninitialized_copy(v2_begin, v2.data(), sizeof(TestingType) * v2.size());
        std::uninitialized_copy(v3_begin, v3.data(), sizeof(TestingType) * v3_expected.size());

        TestUtils::MinimalisticView r1(v1_begin, v1_end);
        TestUtils::MinimalisticView r2(v2_begin, v2_end);
        TestUtils::MinimalisticView r3(v3_begin, v3_end);

        oneapi::dpl::ranges::transform(policy, r1, r2, r3, [](TestingType x1, TestingType x2) { return x1 + x2; });

        std::string msg = "wrong effect from transform, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3_begin, v3_expected.size(), msg.c_str());

        sycl::free(v1_begin, queue);
        sycl::free(v2_begin, queue);
        sycl::free(v3_begin, queue);
    }
#endif
};

template <typename Algorithm>
void call_test_algo()
{
    Algorithm{}(oneapi::dpl::execution::seq);
    Algorithm{}(oneapi::dpl::execution::unseq);
    Algorithm{}(oneapi::dpl::execution::par);
    Algorithm{}(oneapi::dpl::execution::par_unseq);
#if TEST_DPCPP_BACKEND_PRESENT
    Algorithm{}(TestUtils::get_dpcpp_test_policy());
#endif
}

#endif // _ENABLE_STD_RANGES_TESTING

int main()
{
#if _ENABLE_STD_RANGES_TESTING

    call_test_algo<test_count>    ();
    call_test_algo<test_merge>    ();
    call_test_algo<test_copy_if>  ();
    call_test_algo<test_transform>();

#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
