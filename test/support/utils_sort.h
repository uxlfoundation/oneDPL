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

#ifndef _UTILS_SORT_H
#define _UTILS_SORT_H

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include <random>
#include <atomic>
#include <string>
#include <vector>
#include <utility>
#include <cstdint>
#include <type_traits>

// TODO: check if still needed
#define _CRT_SECURE_NO_WARNINGS

#include "utils.h"
#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/sycl_alloc_utils.h"
#endif

// The struct allows filtering configurations to avoid exponential growth of test cases
// and to pass an error message prefix for better diagnostics
struct SortTestConfig
{
    bool is_stable = false;
    // Ignored with host policies
    bool test_usm_device = false;
    bool test_usm_shared = false;

    std::string err_msg_prefix = "";

    SortTestConfig() = default;

    SortTestConfig(const SortTestConfig& cfg, const std::string& err_msg_prefix)
        : is_stable(cfg.is_stable), test_usm_device(cfg.test_usm_device),
          test_usm_shared(cfg.test_usm_shared), err_msg_prefix(err_msg_prefix) {}
};

inline std::vector<std::size_t>
test_sizes(std::size_t max_size)
{
    std::vector<std::size_t> sizes;
    sizes.reserve(100);
    auto step = [](std::size_t size){ return size <= 16 ? size + 1 : size_t(3.1415 * size);};
    for (std::size_t size = 0; size <= max_size; size = step(size))
    {
        sizes.push_back(size);
    }
    return sizes;
}

template <typename T>
struct Converter
{
    T operator()(size_t k, size_t val) const
    {
        return T(val) * (k % 2 ? 1 : -1);
    }
};

using Host = TestUtils::invoke_on_all_host_policies;
#if TEST_DPCPP_BACKEND_PRESENT
template <std::size_t CallNumber>
using Device = TestUtils::invoke_on_all_hetero_policies<CallNumber>;
#endif

// Checks that an operator() can be without const qualifier
struct NonConstLess
{
    template<typename T, typename U>
    bool operator()(T x, U y)
    {
        return x < y;
    }
};

// ConstLess/ConstGreater can be used instead of std::less/std::greater to test a
// comparison-based sort (e.g. merge-sort) instead of a non-comparison-based sort (e.g. radix-sort)
struct ConstLess
{
    template<typename T, typename U>
    bool operator()(const T& x, const U& y) const
    {
        return x < y;
    }
};
struct ConstGreater
{
    template<typename T, typename U>
    bool operator()(const T& x, const U& y) const
    {
        return x > y;
    }
};

//! Number of extant keys
static std::atomic<std::int32_t> KeyCount;

//! One more than highest index in array to be sorted.
static std::uint32_t LastIndex;

class ParanoidKey;

inline bool
Equal(const ParanoidKey& x, const ParanoidKey& y, bool);

//! A key to be sorted, with lots of checking.
class ParanoidKey
{
    //! Value used by comparator
    std::int32_t value;
    //! Original position or special value (Empty or Dead)
    std::int32_t index;
    //! Special value used to mark object without a comparable value, e.g. after being moved from.
    static const std::int32_t Empty = -1;
    //! Special value used to mark destroyed objects.
    static const std::int32_t Dead = -2;
    // True if key object has comparable value
    bool
    isLive() const
    {
        return (std::uint32_t)(index) < LastIndex;
    }
    // True if key object has been constructed.
    bool
    isConstructed() const
    {
        return isLive() || index == Empty;
    }

  public:
    ParanoidKey()
    {
        ++KeyCount;
        index = Empty;
        value = Empty;
    }
    ParanoidKey(const ParanoidKey& k) : value(k.value), index(k.index)
    {
        EXPECT_TRUE(k.isLive(), "source for copy-constructor is dead");
        ++KeyCount;
    }
    ~ParanoidKey()
    {
        EXPECT_TRUE(isConstructed(), "double destruction");
        index = Dead;
        --KeyCount;
    }
    ParanoidKey&
    operator=(const ParanoidKey& k)
    {
        EXPECT_TRUE(k.isLive(), "source for copy-assignment is dead");
        EXPECT_TRUE(isConstructed(), "destination for copy-assignment is dead");
        value = k.value;
        index = k.index;
        return *this;
    }
    ParanoidKey(std::int32_t index, std::int32_t value, TestUtils::OddTag) : value(value), index(index) {}
    ParanoidKey(ParanoidKey&& k) : value(k.value), index(k.index)
    {
        EXPECT_TRUE(k.isConstructed(), "source for move-construction is dead");
// std::stable_sort() fails in move semantics on paranoid test before VS2015
#if !defined(_MSC_VER) || _MSC_VER >= 1900
        k.index = Empty;
#endif // !defined(_MSC_VER) || _MSC_VER >= 1900
        ++KeyCount;
    }
    ParanoidKey&
    operator=(ParanoidKey&& k)
    {
        EXPECT_TRUE(k.isConstructed(), "source for move-assignment is dead");
        EXPECT_TRUE(isConstructed(), "destination for move-assignment is dead");
        value = k.value;
        index = k.index;
// std::stable_sort() fails in move semantics on paranoid test before VS2015
#if !defined(_MSC_VER) || _MSC_VER >= 1900
        k.index = Empty;
#endif // !defined(_MSC_VER) || _MSC_VER >= 1900
        return *this;
    }
    friend class KeyCompare;
    friend bool
    Equal(const ParanoidKey& x, const ParanoidKey& y, bool);
};

class KeyCompare
{
    enum statusType
    {
        //! Special value used to mark defined object.
        Live = 0xabcd,
        //! Special value used to mark destroyed objects.
        Dead = -1
    } status;

  public:
    KeyCompare(TestUtils::OddTag) : status(Live) {}
    ~KeyCompare() { status = Dead; }
    bool
    operator()(const ParanoidKey& j, const ParanoidKey& k) const
    {
        EXPECT_TRUE(status == Live, "key comparison object not defined");
        EXPECT_TRUE(j.isLive(), "first key to operator() is not live");
        EXPECT_TRUE(k.isLive(), "second key to operator() is not live");
        return j.value < k.value;
    }
};

inline bool
Equal(const ParanoidKey& x, const ParanoidKey& y, bool is_stable)
{
    return (x.value == y.value && !is_stable) || (x.index == y.index);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
bool
Equal(const T& x, const T& y, bool /*is_stable*/)
{
    return x == y;
}

#if TEST_DPCPP_BACKEND_PRESENT
inline bool
Equal(const sycl::half& x, const sycl::half& y, bool /*is_stable*/)
{
    return x == y;
}
#endif

template <typename T, typename Compare>
bool check_by_predicate(T t1, T t2, Compare c)
{
    return !c(t2, t1);
}

template <typename T>
bool check_by_predicate(T t1, T t2)
{
    return !(t2 < t1);
}

template <typename InputIterator, typename OutputIterator1, typename OutputIterator2, typename Size>
void
copy_data(InputIterator first, OutputIterator1 expected_first, OutputIterator1 expected_last, OutputIterator2 tmp_first,
          Size n)
{
    std::copy_n(first, n, expected_first);
    std::copy_n(first, n, tmp_first);
}

template <typename... Args>
void
call_sort(bool is_stable, Args&&... args)
{
    if (is_stable)
        oneapi::dpl::stable_sort(std::forward<Args>(args)...);
    else
        oneapi::dpl::sort(std::forward<Args>(args)...);
}

template <typename... Args>
void
call_reference_sort(bool is_stable, Args&&... args)
{
    if (is_stable)
        std::stable_sort(std::forward<Args>(args)...);
    else
        std::sort(std::forward<Args>(args)...);
}

template <typename OutputIterator1, typename OutputIterator2, typename Size, typename... Compare>
void
check_results(SortTestConfig config,
              OutputIterator1 expected_first, OutputIterator2 tmp_first, Size n, Compare... compare)
{
    auto pred = *tmp_first;
    const std::string msg_size = config.err_msg_prefix + ", total size = " + std::to_string(n);
    for (size_t i = 0; i < n; ++i, ++expected_first, ++tmp_first)
    {
        // Check that expected[i] is equal to tmp[i]
        bool reference_check = Equal(*expected_first, *tmp_first, config.is_stable);
        if (!reference_check)
        {
            const std::string msg = msg_size + ", mismatch with reference at index " + std::to_string(i);
            EXPECT_TRUE(reference_check, msg.c_str());
        }
        // Compare with the previous element using the predicate
        if (i > 1 && i < n-1) // first and last were not sorted
        {
            bool is_sorted_check = check_by_predicate(pred, *tmp_first, compare...);
            if (!is_sorted_check)
            {
                const std::string msg = msg_size + ", wrong order at index " + std::to_string(i);
                EXPECT_TRUE(is_sorted_check, msg.c_str());
            }
        }
        pred = *tmp_first;
    }
}

// Additional check for std::execution::par_unseq is required because standard execution policy is
// not a host execution policy in terms of oneDPL and the eligible overload of run_test would not be found
// while testing PSTL offload
template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
          typename... Compare>
std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value
#if __SYCL_PSTL_OFFLOAD__
                 || std::is_same_v<std::decay_t<Policy>, std::execution::parallel_unsequenced_policy>
#endif
                 >
run_test(SortTestConfig config,
         Policy&& exec, OutputIterator tmp_first, OutputIterator tmp_last,OutputIterator2 expected_first,
         OutputIterator2 expected_last, InputIterator first, InputIterator /*last*/, Size n, Compare ...compare)
{
    // Prepare data for sort algorithm
    copy_data(first, expected_first, expected_last, tmp_first, n);
    call_reference_sort(config.is_stable, expected_first + 1, expected_last - 1, compare...);

    // Call sort algorithm on prepared data
    const std::int32_t count0 = KeyCount;
    call_sort(config.is_stable, std::forward<Policy>(exec), tmp_first + 1, tmp_last - 1, compare...);

    check_results(config, expected_first, tmp_first, n, compare...);
    const std::int32_t count1 = KeyCount;
    EXPECT_EQ(count0, count1, "key cleanup error");
}

// TODO: fall back to sycl::buffer (USM is an optional feature)
#if TEST_DPCPP_BACKEND_PRESENT && _PSTL_SYCL_TEST_USM
template <sycl::usm::alloc alloc_type, typename Policy, typename InputIterator, typename OutputIterator,
          typename OutputIterator2, typename Size, typename ...Compare>
void
test_usm(SortTestConfig config,
         Policy&& exec, OutputIterator tmp_first, OutputIterator tmp_last, OutputIterator2 expected_first,
         OutputIterator2 expected_last, InputIterator first, InputIterator /* last */, Size n, Compare... compare)
{
    // Prepare data for sort algorithm
    copy_data(first, expected_first, expected_last, tmp_first, n);
    call_reference_sort(config.is_stable, expected_first + 1, expected_last - 1, compare...);

    using ValueType = typename std::iterator_traits<OutputIterator>::value_type;

    auto queue = exec.queue();

    // Call sort algorithm on prepared data
    const auto it_from = tmp_first + 1;
    const auto it_to = tmp_last - 1;
    TestUtils::usm_data_transfer<alloc_type, ValueType> dt_helper(queue, it_from, it_to);
    auto sortingData = dt_helper.get_data();

    const std::int32_t count0 = KeyCount;

    // Call the tested algorithm
    const auto size = it_to - it_from;
    call_sort(config.is_stable, std::forward<Policy>(exec), sortingData, sortingData + size, compare...);

    dt_helper.retrieve_data(it_from);
    check_results(config, expected_first, tmp_first, n, compare...);
    const std::int32_t count1 = KeyCount;
    EXPECT_EQ(count0, count1, "key cleanup error");
}

template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
          typename... Compare>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<Policy>
run_test(SortTestConfig config,
         Policy&& exec, OutputIterator tmp_first, OutputIterator tmp_last, OutputIterator2 expected_first,
         OutputIterator2 expected_last, InputIterator first, InputIterator last, Size n, Compare ...compare)
{
    // Run tests for USM shared memory (external testing for USM shared memory, once already covered in sycl_iterator.pass.cpp)
    if (config.test_usm_shared)
    {
        test_usm<sycl::usm::alloc::shared>(config, exec, tmp_first, tmp_last,
                                           expected_first, expected_last,first, last, n, compare...);
    }
    if (config.test_usm_device)
    {
        // Run tests for USM device memory
        test_usm<sycl::usm::alloc::device>(config, std::forward<Policy>(exec), tmp_first, tmp_last,
                                           expected_first, expected_last, first, last, n, compare...);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT && _PSTL_SYCL_TEST_USM

template <typename T>
struct test_sort_op
{
    SortTestConfig config;

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename... Compare>
    std::enable_if_t<
        TestUtils::is_base_of_iterator_category_v<std::random_access_iterator_tag, InputIterator> &&
            (TestUtils::can_use_default_less_operator_v<T> || sizeof...(Compare) > 0)>
    operator()(Policy&& exec, OutputIterator tmp_first, OutputIterator tmp_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, InputIterator first, InputIterator last, Size n, Compare ...compare)
    {
        run_test(config, std::forward<Policy>(exec), tmp_first, tmp_last, expected_first, expected_last,
                first, last, n, compare...);
    }

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename... Compare>
    std::enable_if_t<
        !TestUtils::is_base_of_iterator_category_v<std::random_access_iterator_tag, InputIterator> ||
            !(TestUtils::can_use_default_less_operator_v<T> || sizeof...(Compare) > 0)>
    operator()(Policy&& /* exec */, OutputIterator /* tmp_first */, OutputIterator /* tmp_last */,
               OutputIterator2 /* expected_first */, OutputIterator2 /* expected_last */, InputIterator /* first */,
               InputIterator /* last */, Size /* n */, Compare .../*compare*/)
    {
    }
};

template <typename T, typename Invoker, typename Converter, typename... Compare>
void
test_sort(SortTestConfig config, const std::vector<std::size_t>& sizes, Invoker invoker, Converter converter,
          Compare... compare)
{
    std::srand(42);
    for (auto n: sizes)
    {
        LastIndex = n + 2;
        // The rand()%(2*n+1) encourages generation of some duplicates.
        // Sequence is padded with an extra element at front and back, to detect overwrite bugs.
        TestUtils::Sequence<T> in(n + 2, [=](size_t k) { return converter(k, rand() % (2 * n + 1)); });
        TestUtils::Sequence<T> expected(in);
        TestUtils::Sequence<T> tmp(in);
        invoker(test_sort_op<T>{config}, tmp.begin(), tmp.end(), expected.begin(), expected.end(),
                in.begin(), in.end(), in.size(), compare...);
    }
};

#if TEST_DPCPP_BACKEND_PRESENT && __SYCL_UNNAMED_LAMBDA__
inline void
test_default_name_gen(SortTestConfig config)
{
    TestUtils::Sequence<int> in({1, 0, 3, 2, 5, 4, 7, 6, 9, 8});
    TestUtils::Sequence<int> expected(in);
    TestUtils::Sequence<int> tmp(in);
    auto my_policy = TestUtils::make_device_policy(TestUtils::get_test_queue());

    TestUtils::iterator_invoker<std::random_access_iterator_tag, /*IsReverse*/ std::false_type>()(
        my_policy, test_sort_op<int>{config}, tmp.begin(), tmp.end(), expected.begin(), expected.end(),
        in.begin(), in.end(), in.size(), std::greater<>());
    TestUtils::iterator_invoker<std::random_access_iterator_tag, /*IsReverse*/ std::false_type>()(
        my_policy, test_sort_op<int>{config}, tmp.begin(), tmp.end(), expected.begin(), expected.end(),
        in.begin(), in.end(), in.size(), std::less<>());
}
#endif //TEST_DPCPP_BACKEND_PRESENT && __SYCL_UNNAMED_LAMBDA__

#endif /* _UTILS_SORT_H */
