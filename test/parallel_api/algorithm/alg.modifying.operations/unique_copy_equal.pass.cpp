// -*- C++ -*-
//===-- unique_copy_equal.pass.cpp ----------------------------------------===//
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

// Tests for unique_copy
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace TestUtils;

template <typename T>
struct run_unique_copy
{
// dummy specializations to skip testing in case of broken configuration
#if _PSTL_ICC_18_19_TEST_SIMD_MONOTONIC_WINDOWS_RELEASE_BROKEN
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(oneapi::dpl::execution::unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size n, T trash)
    {
    }

    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size n, T trash)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator
#if !TEST_DPCPP_BACKEND_PRESENT
               out_last
#endif
               , OutputIterator2 expected_first, OutputIterator2 /* expected_last */,
               Size n, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, n, trash);
        ::std::fill_n(out_first, n, trash);

        // Run unique_copy
        [[maybe_unused]] auto i = unique_copy(first, last, expected_first);
        auto k = unique_copy(exec, first, last, out_first);
#if !TEST_DPCPP_BACKEND_PRESENT
        EXPECT_EQ_N(expected_first, out_first, n, "wrong unique_copy effect");
        for (size_t j = 0; j < GuardSize; ++j)
        {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from unique_copy");
#else
        auto expected_count = ::std::distance(expected_first, i);
        auto out_count = ::std::distance(out_first, k);

        EXPECT_TRUE(expected_count == out_count, "wrong return value from unique_copy");
        EXPECT_EQ_N(expected_first, out_first, expected_count, "wrong unique_copy effect");
#endif
    }
};

template <typename T>
struct run_unique_copy_predicate
{
    // dummy specializations to skip testing in case of broken configuration
#if _PSTL_ICC_18_19_TEST_SIMD_MONOTONIC_WINDOWS_RELEASE_BROKEN
    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate>
    void
    operator()(oneapi::dpl::execution::unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size n, Predicate pred, T trash)
    {
    }

    template <typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate>
    void
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, OutputIterator out_last, OutputIterator2 expected_first,
               OutputIterator2 expected_last, Size n, Predicate pred, T trash)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate>
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

        // Run unique_copy with predicate
        [[maybe_unused]] auto i = unique_copy(first, last, expected_first, pred);
        auto k = unique_copy(exec, first, last, out_first, pred);
#if !TEST_DPCPP_BACKEND_PRESENT
        EXPECT_EQ_N(expected_first, out_first, n, "wrong unique_copy with predicate effect");
        for (size_t j = 0; j < GuardSize; ++j)
        {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from unique_copy with predicate");
#else
        auto expected_count = ::std::distance(expected_first, i);
        auto out_count = ::std::distance(out_first, k);

        EXPECT_TRUE(expected_count == out_count, "wrong return value from unique_copy with predicate");
        EXPECT_EQ_N(expected_first, out_first, expected_count, "wrong unique_copy with predicate effect");
#endif
    }
};

template <typename T, typename BinaryPredicate, typename Convert>
void
test(T trash, BinaryPredicate pred, Convert convert, bool check_weakness = true)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        // count is number of output elements, plus a handful
        // more for sake of detecting buffer overruns.
        Sequence<T> in(n, [&](size_t k) -> T { return convert(k ^ n); });
        using namespace std;
#if !TEST_DPCPP_BACKEND_PRESENT
        size_t count = GuardSize;
        for (size_t k = 0; k < in.size(); ++k)
            count += k == 0 || !pred(in[k], in[k - 1]) ? 1 : 0;
#else
        size_t count = n;
#endif
        Sequence<T> out(count, [=](size_t) { return trash; });
        Sequence<T> expected(count, [=](size_t) { return trash; });
        if (check_weakness)
        {
            auto expected_result = unique_copy(in.begin(), in.end(), expected.begin(), pred);
            size_t m = expected_result - expected.begin();
            EXPECT_TRUE(n / (n < 10000 ? 4 : 6) <= m && m <= (3 * n + 1) / 4, "weak test for unique_copy");
        }
        invoke_on_all_policies<>()(run_unique_copy<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), count, trash);
        invoke_on_all_policies<>()(run_unique_copy_predicate<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), count, pred, trash);
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        unique_copy(exec, input_iter, input_iter, out_iter, non_const(::std::equal_to<T>()));
    }
};

int
main()
{
#if !TEST_DPCPP_BACKEND_PRESENT
    test<Number>(Number(42, OddTag()), ::std::equal_to<Number>(),
                 [](std::int32_t j) { return Number(3 * j / 13 ^ (j & 8), OddTag()); });
#endif

    test<float64_t>(float64_t(42), ::std::equal_to<float64_t>(),
                    [](std::int32_t j) { return float64_t(5 * j / 23 ^ (j / 7)); });
#if !ONEDPL_FPGA_DEVICE
    test<float32_t>(float32_t(42), [](float32_t, float32_t) { return false; },
                    [](std::int32_t j) { return float32_t(j); }, false);
#endif

    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    return done();
}
