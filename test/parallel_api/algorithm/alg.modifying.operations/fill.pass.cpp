// -*- C++ -*-
//===-- fill.pass.cpp -----------------------------------------------------===//
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

#if  !defined(_PSTL_TEST_FILL) && !defined(_PSTL_TEST_FILL_N)
#define _PSTL_TEST_FILL
#define _PSTL_TEST_FILL_N
#endif

using namespace TestUtils;

template <typename T>
struct test_fill
{
    template <typename It>
    bool
    check(It first, It last, const T& value)
    {
        for (; first != last; ++first)
            if (*first != value)
                return false;
        return true;
    }

    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& value)
    {
        fill(first, last, T(value + 1)); // initialize memory with different value

        fill(exec, first, last, value);
        EXPECT_TRUE(check(first, last, value), "fill wrong result");
    }
};

template <typename T>
struct test_fill_n
{
    template <typename It, typename Size>
    bool
    check(It first, Size n, const T& value)
    {
        for (Size i = 0; i < n; ++i, ++first)
            if (*first != value)
                return false;
        return true;
    }

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Size n, const T& value)
    {
        fill_n(first, n, T(value + 1)); // initialize memory with different value

        const Iterator one_past_last = fill_n(exec, first, n, value);
        const Iterator expected_return = ::std::next(first, n);

        EXPECT_TRUE(expected_return == one_past_last, "fill_n should return Iterator to one past the element assigned");
        EXPECT_TRUE(check(first, n, value), "fill_n wrong result");

        //n == -1
        const Iterator res = fill_n(exec, first, -1, value);
        EXPECT_TRUE(res == first, "fill_n wrong result for n == -1");
    }
};

template <typename T>
void
test_fill_by_type(::std::size_t n)
{
    Sequence<T> in(n, [](::std::size_t) -> T { return T(0); }); //fill with zeros
    T value = -1;

#ifdef _PSTL_TEST_FILL
    invoke_on_all_policies<>()(test_fill<T>(), in.begin(), in.end(), value);
#endif
#ifdef _PSTL_TEST_FILL_N
    invoke_on_all_policies<>()(test_fill_n<T>(), in.begin(), n, value);
#endif
}

int
main()
{
    for (std::size_t n : TestUtils::get_pattern_for_test_sizes())
    {
        test_fill_by_type<std::int8_t>(n);
        test_fill_by_type<std::int16_t>(n);
        test_fill_by_type<std::int32_t>(n);
        test_fill_by_type<float64_t>(n);
    }

    return done();
}
