// -*- C++ -*-
//===-- find_if.pass.cpp --------------------------------------------------===//
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

// Tests for find_if and find_if_not
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if  !defined(_PSTL_TEST_FIND_IF) && !defined(_PSTL_TEST_FIND_IF_NOT)
#define _PSTL_TEST_FIND_IF
#define _PSTL_TEST_FIND_IF_NOT
#endif

using namespace TestUtils;

template <typename T>
struct test_find_if
{
    template <typename Policy, typename Iterator, typename Predicate, typename NotPredicate>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Predicate pred, NotPredicate /* not_pred */)
    {
        auto i = ::std::find_if(first, last, pred);
        auto j = find_if(exec, first, last, pred);
        EXPECT_TRUE(i == j, "wrong return value from find_if");
    }
};

template <typename T>
struct test_find_if_not
{
    template <typename Policy, typename Iterator, typename Predicate, typename NotPredicate>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Predicate pred, NotPredicate not_pred)
    {
        auto i = ::std::find_if(first, last, pred);
        auto i_not = find_if_not(exec, first, last, not_pred);
        EXPECT_TRUE(i_not == i, "wrong return value from find_if_not");
    }
};

template <typename T, typename Predicate, typename Hit, typename Miss>
void
test(Predicate pred, Hit hit, Miss miss)
{
    auto not_pred = [pred](T x) { return !pred(x); };
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> in(n, [&](size_t k) -> T { return miss(n ^ k); });
        // Try different find positions, including not found.
        // By going backwards, we can add extra matches that are *not* supposed to be found.
        // The decreasing exponential gives us O(n) total work for the loop since each find takes O(m) time.
        for (size_t m = n; m > 0; m *= 0.6)
        {
            if (m < n)
                in[m] = hit(n ^ m);
#ifdef _PSTL_TEST_FIND_IF
            invoke_on_all_policies<0>()(test_find_if<T>(), in.begin(), in.end(), pred, not_pred);
            invoke_on_all_policies<1>()(test_find_if<T>(), in.cbegin(), in.cend(), pred, not_pred);
#endif
#ifdef _PSTL_TEST_FIND_IF_NOT
            invoke_on_all_policies<2>()(test_find_if_not<T>(), in.begin(), in.end(), pred, not_pred);
            invoke_on_all_policies<3>()(test_find_if_not<T>(), in.cbegin(), in.cend(), pred, not_pred);
#endif
        }
    }
}

struct test_non_const_find_if
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto is_even = [&](float64_t v) {
            std::uint32_t i = (std::uint32_t)v;
            return i % 2 == 0;
        };

        find_if(exec, iter, iter, non_const(is_even));
    }
};

struct test_non_const_find_if_not
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto is_even = [&](float64_t v) {
            std::uint32_t i = (std::uint32_t)v;
            return i % 2 == 0;
        };

        find_if_not(exec, iter, iter, non_const(is_even));
    }
};

int
main()
{
#if !TEST_DPCPP_BACKEND_PRESENT
    // Note that the "hit" and "miss" functions here avoid overflow issues.
    test<Number>(IsMultiple(5, OddTag()), [](std::int32_t j) { return Number(j - j % 5, OddTag()); }, // hit
                 [](std::int32_t j) { return Number(j % 5 == 0 ? j ^ 1 : j, OddTag()); });            // miss
#endif

    // Try type for which algorithm can really be vectorized.
    test<float32_t>([](float32_t x) { return x >= 0; }, [](float32_t j) { return j * j; },
                    [](float32_t j) { return -1 - j * j; });

#ifdef _PSTL_TEST_FIND_IF
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const_find_if>());
#endif
#ifdef _PSTL_TEST_FIND_IF_NOT
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const_find_if_not>());
#endif

    return done();
}
