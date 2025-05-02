// -*- C++ -*-
//===-- includes.pass.cpp -------------------------------------------------===//
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

template <typename T>
struct Num
{
    T val;
    explicit Num(const T& v) : val(v) {}

    //for "includes" checks
    template <typename T1>
    bool
    operator<(const Num<T1>& v1) const
    {
        return val < v1.val;
    }

    //The types Type1 and Type2 must be such that an object of type InputIt can be dereferenced and then implicitly converted to both of them
    template <typename T1>
    operator Num<T1>() const
    {
        return Num<T1>((T1)val);
    }
};

template <typename T>
struct test_without_compare
{
    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1> && can_use_default_less_operator_v<T>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2)
    {
        auto expect_res = ::std::includes(first1, last1, first2, last2);
        auto res = ::std::includes(exec, first1, last1, first2, last2);

        EXPECT_TRUE(expect_res == res, "wrong result for includes without predicate");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1> || !can_use_default_less_operator_v<T>>
    operator()(Policy&& /* exec */, InputIterator1 /* first1 */, InputIterator1 /* last1 */, InputIterator2 /* first2 */, InputIterator2 /* last2 */)
    {
    }
};

template <typename T>
struct test_with_compare
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {

        auto expect_res = ::std::includes(first1, last1, first2, last2, comp);
        auto res = ::std::includes(exec, first1, last1, first2, last2, comp);

        EXPECT_TRUE(expect_res == res, "wrong result for includes with predicate");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& /* exec */, InputIterator1 /* first1 */, InputIterator1 /* last1 */, InputIterator2 /* first2 */, InputIterator2 /* last2 */,
               Compare /* comp */)
    {
    }
};

template <typename T1, typename T2, typename Compare>
void
test_includes(Compare compare)
{
    ::std::srand(42);
    const ::std::size_t n_max = 1000000;

    Sequence<T1> in1(n_max, [](::std::size_t k) { return rand() % (2 * k + 1); });
    Sequence<T2> in2(n_max, [](::std::size_t k) { return rand() % (k + 1); });

    Sequence<T1> in3(n_max, [](::std::size_t k) { return rand() % (k + 1); });
    Sequence<T2> in4(n_max, [](::std::size_t k) { return rand() % (k/2 + 1); });

    //prepare the input ranges: 1-2 for testing with compare, 3-4 for testing without compare
    ::std::sort(in1.begin(), in1.end(), compare);
    ::std::sort(in2.begin(), in2.end(), compare);
    ::std::sort(in3.begin(), in3.end());
    ::std::sort(in4.begin(), in4.end());

    for (::std::size_t n = 0; n < n_max; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        for (::std::size_t m = 0; m < n_max; m = m <= 16 ? m + 1 : size_t(2.71828 * m))
        {
            invoke_on_all_policies<0>()(test_with_compare<T1>(), in1.begin(), in1.begin() + n, in2.cbegin(),
                                        in2.cbegin() + m, compare);
            invoke_on_all_policies<1>()(test_without_compare<T1>(), in3.begin(), in3.begin() + n, in4.begin(),
                                        in4.begin() + m);
            //test w/ non constant predicate
            if (n < 5 && m < 5)
                invoke_on_all_host_policies()(test_with_compare<T1>(), in1.begin(), in1.end(), in2.cbegin(),
                                              in2.cend(), non_const(compare));
        }
    }
}

int
main()
{
    test_includes<float64_t, float64_t>([](const float64_t x, const float64_t y){ return x > y; });
#if !TEST_DPCPP_BACKEND_PRESENT
    test_includes<Num<std::int64_t>, Num<std::int32_t>>([](const Num<std::int64_t>& x, const Num<std::int32_t>& y) { return x < y; });
#endif

    return done();
}
