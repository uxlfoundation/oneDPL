// -*- C++ -*-
//===-- reverse_copy.pass.cpp ---------------------------------------------===//
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

using namespace TestUtils;

template <typename T>
struct wrapper
{
    T t;
    wrapper() = default;
    explicit wrapper(T t_) : t(t_) {}
    bool
    operator==(const wrapper& t_) const
    {
        return t == t_.t;
    }
};

template <typename T1, typename T2>
bool
eq(const wrapper<T1>& a, const wrapper<T2>& b)
{
    return a.t == b.t;
}

template <typename T1, typename T2>
bool
eq(const T1& a, const T2& b)
{
    return a == b;
}

template<typename T1, typename T2>
struct test_one_policy
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, Iterator1>>
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e)
    {
        using namespace std;
        fill(actual_b, actual_e, T2(-123));
        Iterator2 actual_return = reverse_copy(exec, data_b, data_e, actual_b);

        EXPECT_TRUE(actual_return == actual_e, "wrong result of reverse_copy");

        const auto n = ::std::distance(data_b, data_e);
        Sequence<T2> res(n);
        ::std::copy(::std::reverse_iterator<Iterator1>(data_e), ::std::reverse_iterator<Iterator1>(data_b), res.begin());

        EXPECT_EQ_N(res.begin(), actual_b, n, "wrong effect of reverse_copy");
    }

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, Iterator1>>
    operator()(ExecutionPolicy&& /* exec */, Iterator1 /* data_b */, Iterator1 /* data_e */, Iterator2 /* actual_b */, Iterator2 /* actual_e*/)
    {
    }
};

template <typename T1, typename T2>
void
test()
{
    const auto test_sizes = TestUtils::get_pattern_for_test_sizes();
    const std::size_t max_len = test_sizes.back();
    Sequence<T2> actual(max_len);
    Sequence<T1> data(max_len, [](::std::size_t i) { return T1(i); });

    for (std::size_t len : test_sizes)
    {
        invoke_on_all_policies<0>()(test_one_policy<T1, T2>(),
                                    data.begin(), data.begin() + len, actual.begin(), actual.begin() + len);
        invoke_on_all_policies<1>()(test_one_policy<T1, T2>(),
                                    data.cbegin(), data.cbegin() + len, actual.begin(), actual.begin() + len);
    }
}

int
main()
{
    // clang-3.8 fails to correctly auto vectorize the loop in some cases of different types of container's elements,
    // for example: std::int32_t and std::int8_t. This issue isn't detected for clang-3.9 and newer versions.
    test<std::uint8_t, std::uint8_t>();
    test<std::int16_t, std::int8_t>();
    test<std::uint16_t, float32_t>();
    test<float64_t, std::int64_t>();
    test<wrapper<float32_t>, wrapper<float32_t>>();

    return done();
}
