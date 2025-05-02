// -*- C++ -*-
//===-- copy_move.pass.cpp ------------------------------------------------===//
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

// Tests for copy, move and copy_n

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if  !defined(_PSTL_TEST_COPY) && !defined(_PSTL_TEST_COPY_N) && !defined(_PSTL_TEST_MOVE)
#define _PSTL_TEST_COPY
#define _PSTL_TEST_COPY_N
#define _PSTL_TEST_MOVE
#endif

using namespace TestUtils;

template <typename T>
struct run_copy
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size size,
               Size, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, size, trash);
        ::std::fill_n(out_first, size, trash);

        // Run copy
        copy(first, last, expected_first);
        auto k = copy(exec, first, last, out_first);
        for (size_t j = 0; j < GuardSize; ++j)
            ++k;
        EXPECT_EQ_N(expected_first, out_first, size, "wrong effect from copy");
        EXPECT_TRUE(out_last == k, "wrong return value from copy");
    }
};

template <typename T>
struct run_copy_n
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size size,
               Size n, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, size, trash);
        ::std::fill_n(out_first, size, trash);

        // Run copy_n
        copy(first, last, expected_first);
        auto k = copy_n(exec, first, n, out_first);
        for (size_t j = 0; j < GuardSize; ++j)
            ++k;
        EXPECT_EQ_N(expected_first, out_first, size, "wrong effect from copy_n");
        EXPECT_TRUE(out_last == k, "wrong return value from copy_n");
    }
};

template <typename T>
struct run_move
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size size,
               Size, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, size, trash);
        ::std::fill_n(out_first, size, trash);

        // Run move
        move(first, last, expected_first);
        auto k = move(exec, first, last, out_first);
        for (size_t j = 0; j < GuardSize; ++j)
            ++k;
        EXPECT_EQ_N(expected_first, out_first, size, "wrong effect from move");
        EXPECT_TRUE(out_last == k, "wrong return value from move");
    }
};

template <typename T>
struct run_move<Wrapper<T>>
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 /* expected_first */, OutputIterator2 /* expected_last */, Size size,
               Size /* n */, Wrapper<T> trash)
    {
        // Cleaning
        ::std::fill_n(out_first, size, trash);
        Wrapper<T>::SetMoveCount(0);

        // Run move
        auto k = move(exec, first, last, out_first);
        for (size_t j = 0; j < GuardSize; ++j)
            ++k;
        EXPECT_TRUE(Wrapper<T>::MoveCount() == size, "wrong effect from move");
        EXPECT_TRUE(out_last == k, "wrong return value from move");
    }
};

template <typename T, typename Convert>
void
test(T trash, Convert convert)
{
    // Try sequences of various lengths.
    for (size_t n : TestUtils::get_pattern_for_test_sizes())
    {
        // count is number of output elements, plus a handful
        // more for sake of detecting buffer overruns.
        Sequence<T> in(n, [&](size_t k) -> T {
            T val = convert(n ^ k);
            return val;
        });

        const size_t outN = n + GuardSize;
        Sequence<T> out(outN, [=](size_t) { return trash; });
        Sequence<T> expected(outN, [=](size_t) { return trash; });
#ifdef _PSTL_TEST_COPY
        invoke_on_all_policies<0>()(run_copy<T>(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                                    expected.end(), outN, n, trash);
        invoke_on_all_policies<1>()(run_copy<T>(), in.cbegin(), in.cend(), out.begin(), out.end(), expected.begin(),
                                    expected.end(), outN, n, trash);
#endif
#ifdef _PSTL_TEST_COPY_N
        invoke_on_all_policies<2>()(run_copy_n<T>(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                                    expected.end(), outN, n, trash);
        invoke_on_all_policies<3>()(run_copy_n<T>(), in.cbegin(), in.cend(), out.begin(), out.end(), expected.begin(),
                                    expected.end(), outN, n, trash);
#endif
#ifdef _PSTL_TEST_MOVE
        invoke_on_all_policies<4>()(run_move<T>(), in.begin(), in.end(), out.begin(), out.end(), expected.begin(),
                                    expected.end(), n, n, trash);
#endif

        // For this test const iterator isn't suitable
        // because const rvalue-reference call copy assignment operator
    }
}

int
main()
{
    test<std::int32_t>(-666, [](size_t j) { return std::int32_t(j); });
    test<float64_t>(-666.0, [](size_t j) { return float64_t(j); });

    test<std::uint16_t>(42, [](size_t j) { return std::uint16_t(j); });
    test<std::uint8_t>(42, [](size_t j) { return std::uint8_t(j); });

#if !TEST_DPCPP_BACKEND_PRESENT
    /*TODO: copy support of a class with no default constructor*/
    test<Wrapper<float64_t>>(Wrapper<float64_t>(-666.0), [](std::int32_t j) { return Wrapper<float64_t>(j); });
    test<Number>(Number(42, OddTag()), [](std::int32_t j) { return Number(j, OddTag()); });
#endif

    return done();
}
