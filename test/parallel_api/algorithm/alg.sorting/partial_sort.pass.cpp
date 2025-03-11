// -*- C++ -*-
//===-- partial_sort.pass.cpp ---------------------------------------------===//
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

#include <cmath>

using namespace TestUtils;

#if !TEST_DPCPP_BACKEND_PRESENT
static ::std::atomic<std::int32_t> count_val;
static ::std::atomic<std::int32_t> count_comp;

template <typename T>
struct Num
{
    T val;

    Num() { ++count_val; }
    Num(T v) : val(v) { ++count_val; }
    Num(const Num<T>& v) : val(v.val) { ++count_val; }
    Num(Num<T>&& v) : val(v.val) { ++count_val; }
    ~Num() { --count_val; }
    Num<T>&
    operator=(const Num<T>& v)
    {
        val = v.val;
        return *this;
    }
    operator T() const { return val; }
    bool
    operator<(const Num<T>& v) const
    {
        ++count_comp;
        return val < v.val;
    }
};
#endif

template <typename Type>
struct test_brick_partial_sort
{
    template <typename Policy, typename InputIterator, typename Compare>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::random_access_iterator_tag, InputIterator>>
    operator()(Policy&& exec, InputIterator first, InputIterator last, InputIterator exp_first, InputIterator exp_last,
               InputIterator tmp_first, InputIterator tmp_last, Compare compare)
    {
        const ::std::size_t n = last - first;
        ::std::copy_n(first, n, exp_first);
        ::std::copy_n(first, n, tmp_first);

        for (::std::size_t m = 0; m < n; m = m <= 16 ? m + 1 : ::std::size_t(31.415 * m))
        {
            auto m1 = tmp_first + m;
            auto m2 = exp_first + m;

            ::std::partial_sort(exp_first, m2, exp_last, compare);
#if !TEST_DPCPP_BACKEND_PRESENT && PSTL_USE_DEBUG
            count_comp = 0;
#endif
            ::std::partial_sort(exec, tmp_first, m1, tmp_last, compare);
            EXPECT_EQ_N(exp_first, tmp_first, m, "wrong effect from partial_sort with predicate");

#if !TEST_DPCPP_BACKEND_PRESENT && PSTL_USE_DEBUG
            //checking upper bound number of comparisons; O(p*(last-first)log(middle-first)); where p - number of threads;
            if (m1 - tmp_first > 1)
            {
                auto complex = ::std::ceil(n * ::std::log(float32_t(m1 - tmp_first)));
#if TEST_TBB_BACKEND_PRESENT
                auto p = tbb::this_task_arena::max_concurrency();
#else
                auto p = 1;
#endif
                if (count_comp > complex * p)
                {
                    ::std::cout << "complexity exceeded" << ::std::endl;
                }
            }
#endif // !TEST_DPCPP_BACKEND_PRESENT && PSTL_USE_DEBUG
        }
    }

    template <typename Policy, typename InputIterator, typename Compare>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, InputIterator>>
    operator()(Policy&& /* exec */, InputIterator /* first */, InputIterator /* last */, InputIterator /* exp_first */,
               InputIterator /* exp_last */, InputIterator /* tmp_first */, InputIterator /* tmp_last */, Compare /* compare */)
    {
    }

    template <typename Policy, typename InputIterator>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::random_access_iterator_tag, InputIterator> &&
                           can_use_default_less_operator_v<Type>>
    operator()(Policy&& exec, InputIterator first, InputIterator last, InputIterator exp_first, InputIterator exp_last,
               InputIterator tmp_first, InputIterator tmp_last)
    {
        const ::std::size_t n = last - first;
        ::std::copy_n(first, n, exp_first);
        ::std::copy_n(first, n, tmp_first);

        for (::std::size_t p = 0; p < n; p = p <= 16 ? p + 1 : ::std::size_t(31.415 * p))
        {
            auto m1 = tmp_first + p;
            auto m2 = exp_first + p;

            ::std::partial_sort(exp_first, m2, exp_last);
            ::std::partial_sort(exec, tmp_first, m1, tmp_last);
            EXPECT_EQ_N(exp_first, tmp_first, p, "wrong effect from partial_sort without predicate");
        }
    }

    template <typename Policy, typename InputIterator>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, InputIterator> ||
                           !can_use_default_less_operator_v<Type>>
    operator()(Policy&& /* exec */, InputIterator /* first */, InputIterator /* last */, InputIterator /* exp_first */,
               InputIterator /* exp_last */, InputIterator /* tmp_first */, InputIterator /* tmp_last */)
    {
    }
};

template <typename T, typename Compare>
void
test_partial_sort(Compare compare)
{
    const ::std::size_t n_max = 100000;

    ::std::srand(42);
    // The rand()%(2*k+1) encourages generation of some duplicates.
    Sequence<T> in(n_max, [](::std::size_t k){ return T(rand() % (2 * k + 1)); });
    Sequence<T> exp(n_max);
    Sequence<T> tmp(n_max);

    for (::std::size_t n = 0; n < n_max; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        invoke_on_all_policies<0>()(test_brick_partial_sort<T>(), in.begin(), in.begin() + n,
                                    exp.begin(), exp.begin() + n, tmp.begin(), tmp.begin() + n, compare);
        invoke_on_all_policies<1>()(test_brick_partial_sort<T>(), in.begin(), in.begin() + n,
                                    exp.begin(), exp.begin() + n, tmp.begin(), tmp.begin() + n);
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        partial_sort(exec, iter, iter, iter, non_const(::std::less<T>()));
    }
};

int
main()
{
// Disable the test for SYCL as it relies on global atomic for counting number of comparisons
#if !TEST_DPCPP_BACKEND_PRESENT
    count_val = 0;

    test_partial_sort<Num<float32_t>>([](Num<float32_t> x, Num<float32_t> y) { return x < y; });

    EXPECT_TRUE(count_val == 0, "cleanup error");
#endif

    test_partial_sort<std::int32_t>(
        [](std::int32_t x, std::int32_t y) { return x > y; }); // Reversed so accidental use of < will be detected.

    test_algo_basic_single<std::int32_t>(run_for_rnd<test_non_const<std::int32_t>>());

    return done();
}
