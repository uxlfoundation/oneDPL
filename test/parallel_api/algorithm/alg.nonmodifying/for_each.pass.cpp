// -*- C++ -*-
//===-- for_each.pass.cpp -------------------------------------------------===//
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

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if  !defined(FOR_EACH) && !defined(FOR_EACH_N)
#define FOR_EACH
#define FOR_EACH_N
#endif

using namespace TestUtils;

template <typename Type>
struct Gen
{
    Type
    operator()(::std::size_t k)
    {
        return Type(k % 5 != 1 ? 3 * k + 7 : 0);
    };
};

template <typename T>
struct Flip
{
    std::int32_t val;
    Flip(std::int32_t y) : val(y) {}
    T
    operator()(T& x) const
    {
        return x = val - x;
    }
};

template <typename Type>
struct test_for_each
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Iterator expected_first, Iterator expected_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator>::value_type T;

        // Try for_each
        ::std::for_each(expected_first, expected_last, Flip<T>(1));
        for_each(exec, first, last, Flip<T>(1));
        EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_each");
    }
};

template <typename Type>
struct test_for_each_n
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator, Iterator expected_first, Iterator /* expected_last */, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator>::value_type T;

        // Try for_each_n
        ::std::for_each_n(oneapi::dpl::execution::seq, expected_first, n, Flip<T>(1));
        for_each_n(exec, first, n, Flip<T>(1));
        EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_each_n");
    }
};

template <typename T>
void
test()
{
    for (size_t n : TestUtils::get_pattern_for_test_sizes())
    {
        Sequence<T> in_out(n, Gen<T>());
        Sequence<T> expected(n, Gen<T>());
#ifdef FOR_EACH
        invoke_on_all_policies<>()(test_for_each<T>(), in_out.begin(), in_out.end(), expected.begin(), expected.end(),
                                   in_out.size());
#endif
#ifdef FOR_EACH_N
        invoke_on_all_policies<>()(test_for_each_n<T>(), in_out.begin(), in_out.end(), expected.begin(), expected.end(),
                                   in_out.size());
#endif
    }
}

struct test_non_const_for_each
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto f = [](typename ::std::iterator_traits<Iterator>::reference x) { x = x + 1; };

        for_each(exec, iter, iter, non_const(f));
    }
};

struct test_non_const_for_each_n
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto f = [](typename ::std::iterator_traits<Iterator>::reference x) { x = x + 1; };

        for_each_n(exec, iter, 0, non_const(f));
    }
};

int
main()
{
    test<std::int32_t>();
    test<std::uint16_t>();
    test<float64_t>();


#ifdef FOR_EACH
    test_algo_basic_single<std::int64_t>(run_for_rnd_fw<test_non_const_for_each>());
#endif
#ifdef FOR_EACH_N
    test_algo_basic_single<std::int64_t>(run_for_rnd_fw<test_non_const_for_each_n>());
#endif

    return done();
}
