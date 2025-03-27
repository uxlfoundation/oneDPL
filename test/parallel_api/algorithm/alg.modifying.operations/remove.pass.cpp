// -*- C++ -*-
//===-- remove.pass.cpp ---------------------------------------------------===//
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

// Test for remove, remove_if
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if  !defined(_PSTL_TEST_REMOVE) && !defined(_PSTL_TEST_REMOVE_IF)
#define _PSTL_TEST_REMOVE
#define _PSTL_TEST_REMOVE_IF
#endif

using namespace TestUtils;

template <typename T>
struct run_remove
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator expected_first, OutputIterator expected_last, Size,
               const T& value)
    {
        // Cleaning
        ::std::copy(first, last, expected_first);
        ::std::copy(first, last, out_first);

        // Run remove
        OutputIterator i = remove(expected_first, expected_last, value);
        OutputIterator k = remove(exec, out_first, out_last, value);
        EXPECT_TRUE(::std::distance(expected_first, i) == ::std::distance(out_first, k), "wrong return value from remove");
        EXPECT_EQ_N(expected_first, out_first, ::std::distance(expected_first, i), "wrong remove effect");
    }
};

template <typename T>
struct run_remove_if
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename Predicate>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator expected_first, OutputIterator expected_last, Size,
               Predicate pred)
    {
        // Cleaning
        ::std::copy(first, last, expected_first);
        ::std::copy(first, last, out_first);

        // Run remove_if
        OutputIterator i = remove_if(expected_first, expected_last, pred);
        OutputIterator k = remove_if(exec, out_first, out_last, pred);
        EXPECT_TRUE(::std::distance(expected_first, i) == ::std::distance(out_first, k),
                    "wrong return value from remove_if");
        EXPECT_EQ_N(expected_first, out_first, ::std::distance(expected_first, i), "wrong remove_if effect");
    }
};

template <typename T, typename Predicate, typename Convert>
void
test(T trash, const T& value, Predicate pred, Convert convert)
{
    const ::std::size_t max_size = 100000;
    Sequence<T> out(max_size, [trash](size_t) { return trash; });
    Sequence<T> expected(max_size, [trash](size_t) { return trash; });

    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> data(n, [&](size_t k) -> T { return convert(k); });

#ifdef _PSTL_TEST_REMOVE
        invoke_on_all_policies<>()(run_remove<T>(), data.begin(), data.end(), out.begin(), out.begin() + n,
                                   expected.begin(), expected.begin() + n, n, value);
#endif
#ifdef _PSTL_TEST_REMOVE_IF
        invoke_on_all_policies<>()(run_remove_if<T>(), data.begin(), data.end(), out.begin(), out.begin() + n,
                                   expected.begin(), expected.begin() + n, n, pred);
#endif
    }
}

struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto is_even = [&](float64_t v) {
            std::uint32_t i = (std::uint32_t)v;
            return i % 2 == 0;
        };

        remove_if(exec, iter, iter, non_const(is_even));
    }
};

int
main()
{
#if !_PSTL_ICC_18_TEST_EARLY_EXIT_MONOTONIC_RELEASE_BROKEN
    test<std::uint32_t>(666, 42, [](std::uint32_t) { return true; }, [](size_t j) { return j; });
#endif

    test<std::int32_t>(666, 2001, [](const std::int32_t& val) { return val != 2001; },
                  [](size_t j) { return ((j + 1) % 5 & 2) != 0 ? 2001 : -1 - std::int32_t(j); });
    test<float64_t>(-666.0, 8.5, [](const float64_t& val) { return val != 8.5; },
                    [](size_t j) { return ((j + 1) % 7 & 2) != 0 ? 8.5 : float64_t(j % 32 + j); });

#if !TEST_DPCPP_BACKEND_PRESENT
    test<Number>(Number(-666, OddTag()), Number(42, OddTag()), IsMultiple(3, OddTag()),
                 [](std::int32_t j) { return Number(j, OddTag()); });
#endif

#ifdef _PSTL_TEST_REMOVE_IF
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const>());
#endif
#if !TEST_DPCPP_BACKEND_PRESENT
    test<MemoryChecker>(MemoryChecker{0}, MemoryChecker{1},
        [](const MemoryChecker& val){ return val.value() == 1; },
        [](::std::size_t idx){ return MemoryChecker{::std::int32_t(idx % 3 == 0)}; }
    );
    EXPECT_TRUE(MemoryChecker::alive_objects() == 0, "wrong effect from remove,remove_if: number of ctor and dtor calls is not equal");
#endif

    return done();
}
