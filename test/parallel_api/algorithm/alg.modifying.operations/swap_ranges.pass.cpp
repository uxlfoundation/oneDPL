// -*- C++ -*-
//===-- swap_ranges.pass.cpp ----------------------------------------------===//
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

#include <iterator>
#include <numeric>

using namespace TestUtils;

template <typename T>
struct wrapper
{
    T t;
    ::std::size_t number_of_swaps = 0;
    wrapper(): t() {}
    explicit wrapper(T t_) : t(t_) {}
    template <typename U>
    void
    operator=(const U& b)
    {
        t = b;
    }
    bool
    operator==(const wrapper<T>& a) const
    {
        return t == a.t;
    }
};

template <typename T>
void
swap(wrapper<T>& a, wrapper<T>& b)
{
    ::std::swap(a.t, b.t);
    a.number_of_swaps++;
    b.number_of_swaps++;
}

template <typename T>
struct check_swap
{
    bool
    operator()(T&)
    {
        return true;
    }
};

template <typename T>
struct check_swap<wrapper<T>>
{
    bool
    operator()(wrapper<T>& a)
    {
        bool temp = (a.number_of_swaps == 1);
        a.number_of_swaps = 0;
        return temp;
    }
};

template <typename Type>
struct test_one_policy
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e)
    {
        using namespace std;
        using T_ref = typename iterator_traits<Iterator1>::reference;
        using T = typename iterator_traits<Iterator1>::value_type;

        iota(data_b, data_e, 0);
        iota(actual_b, actual_e, ::std::distance(data_b, data_e));

        Iterator2 actual_return = swap_ranges(exec, data_b, data_e, actual_b);
        bool check_return = (actual_return == actual_e);
        EXPECT_TRUE(check_return, "wrong result of swap_ranges");
        if (check_return)
        {
            ::std::size_t i = 0;
            bool check = all_of(actual_b, actual_e, [&i](T_ref a) { return a == T(i++); }) &&
                         all_of(data_b, data_e, [&i](T_ref a) { return a == T(i++); });

            EXPECT_TRUE(check, "wrong effect of swap_ranges");

            if (check)
            {
                bool swap_check =
                    all_of(data_b, data_e, check_swap<T>()) && all_of(actual_b, actual_e, check_swap<T>());
                EXPECT_TRUE(swap_check, "wrong effect of swap_ranges swap check");
            }
        }
    }
};

template <typename T>
void
test()
{
    const auto test_sizes = TestUtils::get_pattern_for_test_sizes();
    const std::size_t max_len = test_sizes.back();

    Sequence<T> data(max_len);
    Sequence<T> actual(max_len);

    for (std::size_t len : test_sizes)
    {
        invoke_on_all_policies<>()(test_one_policy<T>(), data.begin(), data.begin() + len, actual.begin(),
                                   actual.begin() + len);
    }
}

int
main()
{
    test<wrapper<std::uint16_t>>();
    test<wrapper<float32_t>>();
    test<std::uint8_t>();
    test<std::int32_t>();
    test<float64_t>();

    return done();
}
