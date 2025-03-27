// -*- C++ -*-
//===-- copy_if.pass.cpp --------------------------------------------------===//
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

// Tests for copy_if and remove_copy_if
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if  !defined(_PSTL_TEST_COPY_IF) && !defined(_PSTL_TEST_REMOVE_COPY_IF)
#define _PSTL_TEST_COPY_IF
#define _PSTL_TEST_REMOVE_COPY_IF
#endif

using namespace TestUtils;

template <typename Type>
struct run_copy_if
{
#if _PSTL_ICC_18_19_TEST_SIMD_MONOTONIC_WINDOWS_RELEASE_BROKEN
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(oneapi::dpl::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 expected_last, Size n,
               Predicate pred, T trash)
    {
    }
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size n, Predicate pred, T trash)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator
#if !TEST_DPCPP_BACKEND_PRESENT
               out_last
#endif
               , OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size n,
               Predicate pred, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, n, trash);
        ::std::fill_n(out_first, n, trash);

        // Run copy_if
        [[maybe_unused]] auto i = copy_if(first, last, expected_first, pred);
        auto k = copy_if(exec, first, last, out_first, pred);
#if !TEST_DPCPP_BACKEND_PRESENT
        EXPECT_EQ_N(expected_first, out_first, n, "wrong copy_if effect");
        for (size_t j = 0; j < GuardSize; ++j)
        {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from copy_if");
#else
        auto expected_count = ::std::distance(expected_first, i);
        auto out_count = ::std::distance(out_first, k);
        EXPECT_TRUE(expected_count == out_count, "wrong return value from copy_if");
        EXPECT_EQ_N(expected_first, out_first, expected_count, "wrong copy_if effect");
#endif
    }
};

template <typename Type>
struct run_remove_copy_if
{
#if _PSTL_ICC_18_19_TEST_SIMD_MONOTONIC_WINDOWS_RELEASE_BROKEN
template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(oneapi::dpl::execution::unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size n, Predicate pred, T trash)
    {
    }
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size n, Predicate pred, T trash)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator
#if !TEST_DPCPP_BACKEND_PRESENT
               out_last
#endif
               , OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size n,
               Predicate pred, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, n, trash);
        ::std::fill_n(out_first, n, trash);

        // Run remove_copy_if
        [[maybe_unused]] auto i = remove_copy_if(first, last, expected_first, [=](const T& x) { return !pred(x); });
        auto k = remove_copy_if(exec, first, last, out_first, [=](const T& x) { return !pred(x); });
#if !TEST_DPCPP_BACKEND_PRESENT
        EXPECT_EQ_N(expected_first, out_first, n, "wrong remove_copy_if effect");
        for (size_t j = 0; j < GuardSize; ++j)
        {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from remove_copy_if");
#else
        auto expected_count = ::std::distance(expected_first, i);
        auto out_count = ::std::distance(out_first, k);
        EXPECT_TRUE(expected_count == out_count, "wrong return value from remove_copy_if");
        EXPECT_EQ_N(expected_first, out_first, expected_count, "wrong remove_copy_if effect");
#endif
    }
};

template <typename T, typename Predicate, typename Convert>
void
test(T trash, Predicate pred, Convert convert, bool check_weakness = true)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
#if !TEST_DPCPP_BACKEND_PRESENT
        // count is number of output elements, plus a handful
        // more for sake of detecting buffer overruns.
        size_t count = GuardSize;
#else
        size_t count = n;
#endif
        Sequence<T> in(n, [&](size_t k) -> T {
            T val = convert(n ^ k);
#if !TEST_DPCPP_BACKEND_PRESENT
            count += pred(val) ? 1 : 0;
#endif
            return val;
        });

        Sequence<T> out(count, [=](size_t) { return trash; });
        Sequence<T> expected(count, [=](size_t) { return trash; });
        if (check_weakness)
        {
            auto expected_result = copy_if(in.cfbegin(), in.cfend(), expected.begin(), pred);
            size_t m = expected_result - expected.begin();
            EXPECT_TRUE(n / 4 <= m && m <= 3 * (n + 1) / 4, "weak test for copy_if");
        }
#if defined(_PSTL_TEST_COPY_IF)
        invoke_on_all_policies<0>()(run_copy_if<T>(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                                    expected.end(), count, pred, trash);
        invoke_on_all_policies<1>()(run_copy_if<T>(), in.cbegin(), in.cend(), out.begin(), out.end(), expected.begin(),
                                    expected.end(), count, pred, trash);
#endif
#if defined(_PSTL_TEST_REMOVE_COPY_IF)
        invoke_on_all_policies<2>()(run_remove_copy_if<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), count, pred, trash);
        invoke_on_all_policies<3>()(run_remove_copy_if<T>(), in.cbegin(), in.cend(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), count, pred, trash);
#endif
    }
}

struct test_non_const_copy_if
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        auto is_even = [&](float64_t v) {
            std::uint32_t i = (std::uint32_t)v;
            return i % 2 == 0;
        };
        copy_if(exec, input_iter, input_iter, out_iter, non_const(is_even));
    }
};

struct test_non_const_remove_copy_if
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
        operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        auto is_even = [&](float64_t v) {
            std::uint32_t i = (std::uint32_t)v;
            return i % 2 == 0;
        };
        remove_copy_if(exec, input_iter, input_iter, out_iter, non_const(is_even));
    }
};

int
main()
{
    test<float64_t>(-666.0, [](const float64_t& x) { return x * x <= 1024; },
                    [](size_t j) { return ((j + 1) % 7 & 2) != 0 ? float64_t(j % 32) : float64_t(j % 33 + 34); });

#if !ONEDPL_FPGA_DEVICE
    test<std::int16_t>(-666, [](const std::int16_t& x) { return x != 42; },
                  [](size_t j) { return ((j + 1) % 5 & 2) != 0 ? std::int16_t(j + 1) : 42; });
#endif // ONEDPL_FPGA_DEVICE

#if !TEST_DPCPP_BACKEND_PRESENT
    test<Number>(Number(42, OddTag()), IsMultiple(3, OddTag()), [](std::int32_t j) { return Number(j, OddTag()); });
#endif
    test<std::int32_t>(-666, [](const std::int32_t&) { return true; }, [](size_t j) { return j; }, false);

#if defined(_PSTL_TEST_REMOVE_COPY_IF)
    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const_remove_copy_if>());
#endif

#if defined(_PSTL_TEST_COPY_IF)
    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const_copy_if>());
#endif

    return done();
}
