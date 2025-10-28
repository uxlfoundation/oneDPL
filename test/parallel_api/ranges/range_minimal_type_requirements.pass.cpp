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

constexpr std::size_t kPaddingFront = 16 * 1024;
constexpr std::size_t kPaddingRear = 16 * 1024;

template <typename T>
std::pair<T*, T*>
get_malloc_shared(std::size_t size, sycl::queue queue)
{
    auto ptr = sycl::malloc_shared<T>(size + kPaddingFront + kPaddingRear, queue);

    for (std::size_t i = 0; i < kPaddingFront; ++i)
    {
        ptr[i] = T(i);
    }

    for (std::size_t i = kPaddingFront + size; i < kPaddingFront + size + kPaddingRear; ++i)
    {
        ptr[i] = T(i);
    }

    return {ptr + kPaddingFront, ptr};
}

template <typename T>
void
sycl_free(T* ptr, std::size_t size, sycl::queue queue)
{
    for (std::size_t i = 0; i < kPaddingFront; ++i)
    {
        //printf("check before padding at index %zu: %d\n", i, ptr[i]);
        if (ptr[i] != T(i))
        {
            std::cout << "\t\tInvalid data detected in front padding at index " << i << " : " << ptr[i] << " != " << T(i) << std::endl;
            assert(false);
        }
    }

    for (std::size_t i = kPaddingFront + size; i < kPaddingFront + size + kPaddingRear; ++i)
    {
        //printf("check after padding at index %zu: %d\n", i, ptr[i]);
        if (ptr[i] != T(i))
        {
            std::cout << "\t\tInvalid data detected in rear padding at index " << i << " : " << ptr[i] << " != " << T(i) << std::endl;
            assert(false);
        }
    }

    sycl::free(ptr, queue);
}

struct test_count
{
    template <typename Policy>
    std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        std::vector<int> v = {0, 1, 2, 3, 4, 5};

        TestUtils::MinimalisticRange r{v.begin(), v.end()};
        auto count = oneapi::dpl::ranges::count(policy, r, 3);
        std::string msg = "wrong return value from count, " + std::string(typeid(Policy).name());
        EXPECT_EQ(count, 1, msg.c_str());
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    std::enable_if_t<!oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        auto queue = policy.queue();

        constexpr std::size_t v_size = 6;
        
        auto [v1_begin, v1_buf]  = get_malloc_shared<int>(v_size, queue);
        auto v1_end = v1_begin + v_size;

        std::vector<int> v = {0, 1, 2, 3, 4, 5};

        std::memcpy(v1_begin, v.data(), v_size * sizeof(int));

        TestUtils::MinimalisticRange r{v1_begin, v1_end};

        auto count = oneapi::dpl::ranges::count(policy, std::ranges::subrange(r), 3);

        std::string msg = "wrong return value from count, " + std::string(typeid(Policy).name());
        EXPECT_EQ(count, 1, msg.c_str());

        sycl_free(v1_buf, v_size, queue);
    }
#endif
};

struct test_merge
{
    template <typename Policy>
    std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
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
        EXPECT_EQ_N(v3_expected.begin(), v3.begin(), v3_expected.size(), msg.c_str());
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    std::enable_if_t<!oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        auto queue = policy.queue();

        constexpr std::size_t v_size = 6;

        auto [v1_begin, v1_buf] = get_malloc_shared<int>(v_size, queue);
        auto v1_end = v1_begin + v_size;

        auto [v2_begin, v2_buf] = get_malloc_shared<int>(v_size, queue);
        auto v2_end = v2_begin + v_size;

        auto [v3_begin, v3_buf] = get_malloc_shared<int>(2 * v_size, queue);
        auto v3_end = v3_begin + 2 * v_size;

        std::vector<int> v1 = {0, 2, 4, 6, 8, 10};
        std::vector<int> v2 = {1, 3, 5, 7, 9, 11};
        std::vector<int> v3_expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        std::vector<int> v3(12, 42);

        std::memcpy(v1_begin, v1.data(), v_size * sizeof(int));
        std::memcpy(v2_begin, v2.data(), v_size * sizeof(int));
        std::memcpy(v3_begin, v3.data(), 2 * v_size * sizeof(int));

        TestUtils::MinimalisticRange r1{v1_begin, v1_end};
        TestUtils::MinimalisticRange r2{v2_begin, v2_end};
        TestUtils::MinimalisticRange r3{v3_begin, v3_end};

        oneapi::dpl::ranges::merge(policy, std::ranges::subrange(r1), std::ranges::subrange(r2), std::ranges::subrange(r3));
        std::string msg = "wrong effect from merge, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3_begin, v3_expected.size(), msg.c_str());

        sycl_free(v1_buf, v_size, queue);
        sycl_free(v2_buf, v_size, queue);
        sycl_free(v3_buf, 2 * v_size, queue);
    }
#endif
};

struct test_copy_if
{
    template <typename Policy>
    std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        std::vector<int> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> v3(6);
        std::vector<int> v3_expected = {0, 2, 4, 6, 8, 10};

        TestUtils::MinimalisticRange r1{v1.begin(), v1.end()};
        TestUtils::MinimalisticRange r3{v3.begin(), v3.end()};

        oneapi::dpl::ranges::copy_if(policy, r1, r3, [](int x) { return x % 2 == 0; });

        std::string msg = "wrong effect from copy_if, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3.begin(), v3_expected.size(), msg.c_str());
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    std::enable_if_t<!oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        auto queue = policy.queue();

        constexpr std::size_t v_size1 = 11;
        constexpr std::size_t v_size3 = 6;

        auto [v1_begin, v1_buf] = get_malloc_shared<int>(v_size1, queue);
        auto v1_end = v1_begin + v_size1;

        auto [v3_begin, v3_buf] = get_malloc_shared<int>(v_size3, queue);
        auto v3_end = v1_begin + v_size3;

        std::vector<int> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> v3(6);
        std::vector<int> v3_expected = {0, 2, 4, 6, 8, 10};

        std::memcpy(v1_begin, v1.data(), v_size1 * sizeof(int));
        std::memcpy(v3_begin, v3.data(), v_size3 * sizeof(int));

        TestUtils::MinimalisticRange r1{v1_begin, v1_end};
        TestUtils::MinimalisticRange r3{v3_begin, v3_end};

        oneapi::dpl::ranges::copy_if(policy, std::ranges::subrange(r1), std::ranges::subrange(r3), [](int x) { return x % 2 == 0; });

        std::string msg = "wrong effect from copy_if, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3_begin, v3_expected.size(), msg.c_str());

        sycl_free(v1_buf, v_size1, queue);
        sycl_free(v3_buf, v_size3, queue);
    }
#endif
};

struct test_transform
{
    template <typename Policy>
    std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        std::vector<int> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> v2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> v3(v1.size());
        std::vector<int> v3_expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

        TestUtils::MinimalisticRange r1{v1.begin(), v1.end()};
        TestUtils::MinimalisticRange r2{v2.begin(), v2.end()};
        TestUtils::MinimalisticRange r3{v3.begin(), v3.end()};

        oneapi::dpl::ranges::transform(policy, r1, r2, r3, [](int x1, int x2) { return x1 + x2; });

        std::string msg = "wrong effect from transform, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3.begin(), v3_expected.size(), msg.c_str());
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    std::enable_if_t<!oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value>
    operator()(Policy&& policy)
    {
        auto queue = policy.queue();

        std::vector<int> v1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> v2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        //std::vector<int> v3_expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
        std::vector<int> v3_expected = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

        auto [v1_begin, v1_buf] = get_malloc_shared<int>(v1.size(), queue);
        auto v1_end = v1_begin + v1.size();

        auto [v2_begin, v2_buf] = get_malloc_shared<int>(v2.size(), queue);
        auto v2_end = v2_begin + v2.size();

        auto [v3_begin, v3_buf] = get_malloc_shared<int>(v3_expected.size(), queue);
        auto v3_end = v3_begin + v3_expected.size();

        std::vector<int> v3(v1.size());
        std::memcpy(v1_begin, v1.data(), sizeof(int) * v1.size());
        std::memcpy(v2_begin, v2.data(), sizeof(int) * v2.size());
        std::memcpy(v3_begin, v3.data(), sizeof(int) * v3_expected.size());

        TestUtils::MinimalisticRange r1{v1_begin, v1_end};
        TestUtils::MinimalisticRange r2{v2_begin, v2_end};
        TestUtils::MinimalisticRange r3{v3_begin, v3_end};

        static_assert(!oneapi::dpl::__ranges::__has_subscription_op<decltype(r1)>::value);
        static_assert(!oneapi::dpl::__ranges::__has_subscription_op<decltype(r2)>::value);
        static_assert(!oneapi::dpl::__ranges::__has_subscription_op<decltype(r3)>::value);

        auto r1_ss = oneapi::dpl::__ranges::__get_subscription_view(r1);
        auto r2_ss = oneapi::dpl::__ranges::__get_subscription_view(r2);
        auto r3_ss = oneapi::dpl::__ranges::__get_subscription_view(r3);

        static_assert(oneapi::dpl::__ranges::__has_subscription_op<decltype(r1_ss)>::value);
        static_assert(oneapi::dpl::__ranges::__has_subscription_op<decltype(r2_ss)>::value);
        static_assert(oneapi::dpl::__ranges::__has_subscription_op<decltype(r3_ss)>::value);

        oneapi::dpl::ranges::transform(policy, std::ranges::subrange(r1), std::ranges::subrange(r2), std::ranges::subrange(r3), [](int x1, int x2) { return x1 + x2; });
        //oneapi::dpl::ranges::transform(oneapi::dpl::execution::seq, r1, r2, r3, [](int x1, int x2) { return x1 + x2; });

        std::string msg = "wrong effect from transform, " + std::string(typeid(Policy).name());
        EXPECT_EQ_N(v3_expected.begin(), v3_begin, v3_expected.size(), msg.c_str());

        sycl_free(v1_buf, v1.size(), queue);
        sycl_free(v2_buf, v2.size(), queue);
        sycl_free(v3_buf, v3_expected.size(), queue);
    }
#endif
};

template <typename Algorithm>
void call_test_algo()
{
    std::cout << "\toneapi::dpl::execution::seq" << std::endl;
    Algorithm{}(oneapi::dpl::execution::seq);
    std::cout << "\toneapi::dpl::execution::unseq" << std::endl;
    Algorithm{}(oneapi::dpl::execution::unseq);
    std::cout << "\toneapi::dpl::execution::par" << std::endl;  
    Algorithm{}(oneapi::dpl::execution::par);
    std::cout << "\toneapi::dpl::execution::par_unseq" << std::endl;
    Algorithm{}(oneapi::dpl::execution::par_unseq);
#if TEST_DPCPP_BACKEND_PRESENT
    std::cout << "\toneapi::dpl::execution::dpcpp" << std::endl;
    Algorithm{}(TestUtils::get_dpcpp_test_policy());
#endif
}

#endif // _ENABLE_STD_RANGES_TESTING

int main()
{
#if _ENABLE_STD_RANGES_TESTING
    try
    {
        std::cout << "test_count" << std::endl;
        call_test_algo<test_count>    ();
        std::cout << "test_merge" << std::endl;
        call_test_algo<test_merge>    ();
        std::cout << "test_copy_if" << std::endl;
        call_test_algo<test_copy_if>  ();
        std::cout << "test_transform" << std::endl;
        call_test_algo<test_transform>();
    }
    catch (const std::exception& exc)
    {
        std::stringstream str;

        str << "Exception occurred";
        if (exc.what())
            str << " : " << exc.what();

        TestUtils::issue_error_message(str);
    }
#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
