// -*- C++ -*-
//===-- adjacent_difference.pass.cpp --------------------------------------===//
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
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"

#include <iterator>

using namespace TestUtils;

template <typename T>
struct wrapper
{
    T t;
    explicit wrapper(T t_) : t(t_) {}
    template <typename T2>
    wrapper(const wrapper<T2>& a)
    {
        t = a.t;
    }
    template <typename T2>
    void
    operator=(const wrapper<T2>& a)
    {
        t = a.t;
    }
    wrapper<T>
    operator-(const wrapper<T>& a) const
    {
        return wrapper<T>(t - a.t);
    }
};

template <typename T>
bool
compare(const T& a, const T& b)
{
    return a == b;
}

template <typename T>
bool
compare(const wrapper<T>& a, const wrapper<T>& b)
{
    return a.t == b.t;
}

template <typename Iterator1, typename Iterator2, typename T, typename Function>
::std::enable_if_t<!::std::is_floating_point_v<T>, bool>
compute_and_check(Iterator1 first, Iterator1 last, Iterator2 d_first, T, Function f)
{
    using T2 = typename ::std::iterator_traits<Iterator2>::value_type;

    if (first == last)
        return true;

    T2 temp(*first);
    if (!compare(temp, *d_first))
        return false;
    Iterator1 second = ::std::next(first);

    ++d_first;
    for (; second != last; ++first, ++second, ++d_first)
    {
        T2 temp(f(*second, *first));
        if (!compare(temp, *d_first))
            return false;
    }

    return true;
}

// we don't want to check equality here
// because we can't be sure it will be strictly equal for floating point types
template <typename Iterator1, typename Iterator2, typename T, typename Function>
::std::enable_if_t<::std::is_floating_point_v<T>, bool>
compute_and_check(Iterator1 /* first */, Iterator1 /* last */, Iterator2 /* d_first */, T, Function)
{
    return true;
}

template <typename Type>
struct test_adjacent_difference
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e,
               T trash)
    {
        using namespace std;
        using T2 = typename ::std::iterator_traits<Iterator1>::value_type;

        fill(actual_b, actual_e, trash);

        Iterator2 actual_return = adjacent_difference(exec, data_b, data_e, actual_b);
        EXPECT_TRUE(compute_and_check(data_b, data_e, actual_b, T2(0), ::std::minus<T2>()),
                    "wrong effect of adjacent_difference");
        EXPECT_TRUE(actual_return == actual_e, "wrong result of adjacent_difference");
    }
};

template <typename Type>
struct test_adjacent_difference_functor
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T, typename Function>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e,
               T trash, Function f)
    {
        using namespace std;
        using T2 = typename ::std::iterator_traits<Iterator1>::value_type;

        fill(actual_b, actual_e, trash);

        Iterator2 actual_return = adjacent_difference(exec, data_b, data_e, actual_b, f);
        EXPECT_TRUE(compute_and_check(data_b, data_e, actual_b, T2(0), f),
                    "wrong effect of adjacent_difference with functor");
        EXPECT_TRUE(actual_return == actual_e, "wrong result of adjacent_difference with functor");
    }
};

template <typename T1, typename T2, typename Pred>
void
test(Pred pred)
{
    const auto test_sizes = TestUtils::get_pattern_for_test_sizes();
    const std::size_t max_len = test_sizes.back();

    const T2 value = T2(77);
    const T1 trash = T1(31);

    Sequence<T1> actual(max_len, [](::std::size_t i) { return T1(i); });

    Sequence<T2> data(max_len, [=](::std::size_t i) { return i % 3 == 2 ? T2(i * i) : value; });

    for (std::size_t len : test_sizes)
    {
        invoke_on_all_policies<0>()(test_adjacent_difference<T1>(), data.begin(), data.begin() + len, actual.begin(),
                                    actual.begin() + len, trash);
        invoke_on_all_policies<1>()(test_adjacent_difference_functor<T1>(), data.begin(), data.begin() + len,
                                    actual.begin(), actual.begin() + len, trash, pred);
        invoke_on_all_policies<2>()(test_adjacent_difference<T1>(), data.cbegin(), data.cbegin() + len,
                                    actual.begin(), actual.begin() + len, trash);
        invoke_on_all_policies<3>()(test_adjacent_difference_functor<T1>(), data.cbegin(), data.cbegin() + len,
                                    actual.begin(), actual.begin() + len, trash, pred);
    }
}

int
main()
{
    test<std::uint8_t, std::uint32_t>([](std::uint32_t a, std::uint32_t b) { return a - b; });
    test<std::uint16_t, std::uint16_t>([](std::uint16_t a, std::uint16_t b) { return a > b ? a - b : b - a; });
    test<std::int32_t, std::int64_t>([](std::int64_t a, std::int64_t b) { return a / (b + 1); });
    test<std::int64_t, float32_t>([](float32_t a, float32_t b) { return (a + b) / 2; });
#if !TEST_DPCPP_BACKEND_PRESENT
    test<wrapper<std::int32_t>, wrapper<std::int64_t>>(
        [](const wrapper<std::int64_t>& a, const wrapper<std::int64_t>& b) { return a - b; });
#endif

    return done();
}
