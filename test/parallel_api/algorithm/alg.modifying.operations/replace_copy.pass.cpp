// -*- C++ -*-
//===-- replace_copy.pass.cpp ---------------------------------------------===//
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

// Tests for replace_copy and replace_copy_if

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if  !defined(_PSTL_TEST_REPLACE_COPY) && !defined(_PSTL_TEST_REPLACE_COPY_IF)
#define _PSTL_TEST_REPLACE_COPY
#define _PSTL_TEST_REPLACE_COPY_IF
#endif

using namespace TestUtils;

template <typename t>
struct test_replace_copy
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size n,
               Predicate /* pred */, const T& old_value, const T& new_value, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, n, trash);
        ::std::fill_n(out_first, n, trash);
        // Run replace_copy
        ::std::replace_copy(first, last, expected_first, old_value, new_value);
        auto k = ::std::replace_copy(exec, first, last, out_first, old_value, new_value);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong replace_copy effect");
        EXPECT_TRUE(out_last == k, "wrong return value from replace_copy");
    }
};

template <typename t>
struct test_replace_copy_if
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size n,
               Predicate pred, const T& /* pld_value */, const T& new_value, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, n, trash);
        ::std::fill_n(out_first, n, trash);
        // Run replace_copy_if
        replace_copy_if(first, last, expected_first, pred, new_value);
        auto k = replace_copy_if(exec, first, last, out_first, pred, new_value);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong replace_copy_if effect");
        EXPECT_TRUE(out_last == k, "wrong return value from replace_copy_if");
    }
};

template <typename T, typename Convert, typename Predicate>
void
test(T trash, const T& old_value, const T& new_value, Predicate pred, Convert convert)
{
    // Try sequences of various lengths.
    for (size_t n : TestUtils::get_pattern_for_test_sizes())
    {
        Sequence<T> in(n, [&](size_t k) -> T { return convert(n ^ k); });
        Sequence<T> out(n, [=](size_t) { return trash; });
        Sequence<T> expected(n, [=](size_t) { return trash; });

#ifdef _PSTL_TEST_REPLACE_COPY
        invoke_on_all_policies<0>()(test_replace_copy<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), out.size(), pred, old_value, new_value, trash);
        invoke_on_all_policies<1>()(test_replace_copy<T>(), in.cbegin(), in.cend(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), out.size(), pred, old_value, new_value, trash);
#endif
#ifdef _PSTL_TEST_REPLACE_COPY_IF
        invoke_on_all_policies<2>()(test_replace_copy_if<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), out.size(), pred, old_value, new_value, trash);
        invoke_on_all_policies<3>()(test_replace_copy_if<T>(), in.cbegin(), in.cend(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), out.size(), pred, old_value, new_value, trash);
#endif
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        auto is_even = [&](float64_t v) {
            std::uint32_t i = (std::uint32_t)v;
            return i % 2 == 0;
        };

        replace_copy_if(exec, input_iter, input_iter, out_iter, non_const(is_even), T(0));
    }
};

int
main()
{

    test<float64_t>(-666.0, 8.5, 0.33, [](const float64_t& x) { return x * x <= 1024; },
                    [](size_t j) { return ((j + 1) % 7 & 2) != 0 ? 8.5 : float64_t(j % 32 + j); });

    test<std::int32_t>(-666, 42, 99, [](const std::int32_t& x) { return x != 42; },
                  [](size_t j) { return ((j + 1) % 5 & 2) != 0 ? 42 : -1 - std::int32_t(j); });

    test<std::uint8_t>(123, 42, 99, [](const std::uint8_t& x) { return x != 42; },
                  [](size_t j) { return ((j + 1) % 5 & 2) != 0 ? 42 : 255; });


#if !TEST_DPCPP_BACKEND_PRESENT
    test<Number>(Number(42, OddTag()), Number(2001, OddTag()), Number(2017, OddTag()), IsMultiple(3, OddTag()),
                 [](std::int32_t j) { return ((j + 1) % 3 & 2) != 0 ? Number(2001, OddTag()) : Number(j, OddTag()); });
#endif

#ifdef _PSTL_TEST_REPLACE_COPY_IF
    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());
#endif

    return done();
}
